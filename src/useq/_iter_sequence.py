from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from heapq import merge
from typing import TYPE_CHECKING, Any

from useq._channel import Channel  # noqa: TC001  # noqa: TCH001
from useq._enums import AXES, Axis
from useq._mda_event import Channel as EventChannel
from useq._mda_event import MDAEvent, ReadOnlyDict
from useq._position import Position
from useq._z import AnyZPlan  # noqa: TC001  # noqa: TCH001

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from useq._mda_sequence import MDASequence
    from useq._position import PositionBase


@dataclass(frozen=True)
class _AxisPlan:
    """An axis iterator together with the sequence that owns its plan."""

    key: str
    values: tuple[Channel | float | PositionBase, ...]
    sequence: MDASequence


_AxisSelection = dict[str, tuple[int, Any, _AxisPlan]]


@cache
def _iter_axis(seq: MDASequence, ax: str) -> tuple[Channel | float | PositionBase, ...]:
    return tuple(seq.iter_axis(ax))


@cache
def _sizes(seq: MDASequence) -> dict[str, int]:
    return {k: len(list(_iter_axis(seq, k))) for k in seq.axis_order}


@cache
def _used_axes(seq: MDASequence) -> str:
    return "".join(k for k in seq.axis_order if _sizes(seq)[k])


def _axis_plans(seq: MDASequence) -> tuple[_AxisPlan, ...]:
    return tuple(_AxisPlan(ax, _iter_axis(seq, ax), seq) for ax in _used_axes(seq))


def _sort_axis_plans(
    plans: Iterable[_AxisPlan], order: tuple[str, ...]
) -> tuple[_AxisPlan, ...]:
    """Sort plans by the root order, preserving insertion order for unlisted axes."""
    rank = {str(axis): idx for idx, axis in enumerate(order)}
    return tuple(sorted(plans, key=lambda plan: rank.get(plan.key, len(rank))))


def _iter_plan_product(
    axes: tuple[_AxisPlan, ...],
    *,
    prefix: _AxisSelection | None = None,
    active_sequence: MDASequence,
) -> Iterator[tuple[_AxisSelection, MDASequence]]:
    """Iterate one position branch's effective axis plans."""
    prefix = {} if prefix is None else prefix
    if not axes:
        if prefix:
            yield prefix, active_sequence
        return

    current, *remaining = axes
    for idx, value in enumerate(current.values):
        branch_prefix = {**prefix, current.key: (idx, value, current)}
        yield from _iter_plan_product(
            tuple(remaining),
            prefix=branch_prefix,
            active_sequence=active_sequence,
        )


def _iter_axis_combinations(
    sequence: MDASequence,
) -> Iterator[tuple[_AxisSelection, MDASequence]]:
    """Iterate root and position axes in one global acquisition order.

    Each position is a sparse branch: its sub-sequence overrides matching root
    plans and may introduce new axes.  Branches are individually iterated in the
    root order and lazily merged by their axis indices.  Consequently an axis that
    exists only in a sub-sequence can still be ordered before ``p`` without
    duplicating events for positions that do not define that axis.
    """
    root_axes = _axis_plans(sequence)
    position_plan = next(
        (plan for plan in root_axes if plan.key == Axis.POSITION), None
    )
    if position_plan is None:
        yield from _iter_plan_product(
            _sort_axis_plans(root_axes, sequence.axis_order),
            active_sequence=sequence,
        )
        return

    inherited_axes = tuple(plan for plan in root_axes if plan is not position_plan)
    branch_iterators: list[Iterator[tuple[_AxisSelection, MDASequence]]] = []
    extra_order: list[str] = []

    for p_idx, position in enumerate(position_plan.values):
        child = position.sequence if isinstance(position, Position) else None
        child_axes = _axis_plans(child) if child is not None else ()
        if any(plan.key == Axis.POSITION for plan in child_axes):
            raise ValueError(
                "A Position sequence cannot have multiple stage positions."
            )

        override_keys = {plan.key for plan in child_axes}
        effective_axes = (
            tuple(plan for plan in inherited_axes if plan.key not in override_keys)
            + child_axes
        )
        effective_axes = _sort_axis_plans(effective_axes, sequence.axis_order)
        for plan in effective_axes:
            if plan.key not in sequence.axis_order and plan.key not in extra_order:
                extra_order.append(plan.key)

        prefix = {str(Axis.POSITION): (p_idx, position, position_plan)}
        branch_iterators.append(
            _iter_plan_product(
                effective_axes,
                prefix=prefix,
                active_sequence=child if child is not None else sequence,
            )
        )

    global_order = (*sequence.axis_order, *extra_order)

    def _sort_key(item: tuple[_AxisSelection, MDASequence]) -> tuple[int, ...]:
        selection, _ = item
        # A branch without an axis behaves like a singleton at index zero.
        return tuple(
            selection[axis][0] if axis in selection else 0 for axis in global_order
        )

    yield from merge(*branch_iterators, key=_sort_key)


def iter_sequence(sequence: MDASequence) -> Iterator[MDAEvent]:
    """Iterate over all events in the MDA sequence.'.

    !!! note
        This method will usually be used via [`useq.MDASequence.iter_events`][], or by
        simply iterating over the sequence.

    This does the job of iterating over all the frames in the MDA sequence,
    handling the logic of merging all z plans in channels and stage positions
    defined in the plans for each axis.

    The is the most "logic heavy" part of `useq-schema` (the rest of which is
    almost entirely declarative).  This iterator is useful for consuming `MDASequence`
    objects in a python runtime, but it isn't considered a "core" part of the schema.

    The `sequence.setup` event is not yielded during iteration. It is intended
    to be handled separately by an engine.

    Parameters
    ----------
    sequence : MDASequence
        The sequence to iterate over.

    Yields
    ------
    MDAEvent
        Each event in the MDA sequence.
    """
    if not (keep_shutter_open_axes := sequence.keep_shutter_open_across):
        yield from _iter_sequence(sequence)
        return

    it = _iter_sequence(sequence)
    if (this_e := next(it, None)) is None:  # pragma: no cover
        return

    for next_e in it:
        # set `keep_shutter_open` to `True` if and only if ALL axes whose index
        # changes betwee this_event and next_event are in `keep_shutter_open_axes`
        if all(
            axis in keep_shutter_open_axes
            for axis, idx in this_e.index.items()
            if idx != next_e.index[axis]
        ):
            this_e = this_e.model_copy(update={"keep_shutter_open": True})
        yield this_e
        this_e = next_e
    yield this_e


def _iter_sequence(sequence: MDASequence) -> Iterator[MDAEvent]:
    """Helper function for `iter_sequence`.

    This function expands the branch-aware axis combinations, constructs events, and
    inserts autofocus events.  The outer `iter_sequence` function may then inspect
    adjacent events to determine shutter behavior.

    Parameters
    ----------
    sequence : MDASequence
        The sequence to iterate over.

    Yields
    ------
    MDAEvent
        Each event in the MDA sequence.
    """
    last_t_idx = -1
    last_p_idx = -1
    for selection, active_sequence in _iter_axis_combinations(sequence):
        index, time, position, grid, channel, z_pos = _parse_axes(
            (key, (idx, value)) for key, (idx, value, _plan) in selection.items()
        )
        z_selection = selection.get(str(Axis.Z))
        z_plan = z_selection[2].sequence.z_plan if z_selection is not None else None

        if _should_skip(channel, index, z_plan):
            continue

        event_kwargs: dict[str, Any] = {
            "sequence": sequence,
            "index": ReadOnlyDict(index),
            **_xyzpos(position, channel, z_plan, grid, z_pos),
        }
        if position and position.name:
            event_kwargs["pos_name"] = position.name

        p_idx = index.get(Axis.POSITION, -1)
        if position and position.properties and p_idx != last_p_idx:
            event_kwargs["properties"] = list(position.properties)
        last_p_idx = p_idx

        if channel:
            event_kwargs["channel"] = EventChannel.model_construct(
                config=channel.config, group=channel.group
            )
            if channel.exposure is not None:
                event_kwargs["exposure"] = channel.exposure
        if time is not None:
            event_kwargs["min_start_time"] = time

        if index.get(Axis.TIME) == 0 and last_t_idx != 0:
            event_kwargs["reset_event_timer"] = True

        event = MDAEvent.model_construct(**event_kwargs)
        autofocus_plan = active_sequence.autofocus_plan or sequence.autofocus_plan
        if autofocus_plan:
            af_input = event
            if z_plan is not sequence.z_plan:
                effective_sequence = sequence.model_copy(update={"z_plan": z_plan})
                af_input = event.model_copy(update={"sequence": effective_sequence})
            af_event = autofocus_plan.event(af_input)
            if af_event:
                if af_event.sequence is not sequence:
                    af_event = af_event.model_copy(update={"sequence": sequence})
                yield af_event
        yield event
        last_t_idx = event.index.get(Axis.TIME, last_t_idx)


# ###################### Helper functions ######################


def _parse_axes(
    event: Iterable[tuple[str, tuple[int, Any]]],
) -> tuple[
    dict[str, int],
    float | None,  # time
    Position | None,
    PositionBase | None,
    Channel | None,
    float | None,  # z
]:
    """Parse an individual event from the product of axis iterators.

    Returns typed objects for each axis, and the index of the event.
    """
    # NOTE: this is currently the biggest time sink in iter_sequence.
    # It is called for every event and takes ~40% of the cumulative time.
    _ev = dict(event)
    index = {ax: _ev[ax][0] for ax in AXES if ax in _ev}
    # this needs to be tuple(...) to work for mypyc
    axes = tuple(_ev[ax][1] if ax in _ev else None for ax in AXES)
    return (index, *axes)  # type: ignore [return-value]


def _should_skip(
    channel: Channel | None,
    index: dict[str, int],
    z_plan: AnyZPlan | None,
) -> bool:
    """Return True if this event should be skipped."""
    if channel:
        # skip channels
        if Axis.TIME in index and index[Axis.TIME] % channel.acquire_every:
            return True

        # only acquire on the middle plane:
        if (
            not channel.do_stack
            and z_plan is not None
            and index[Axis.Z] != z_plan.num_positions() // 2
        ):
            return True

    return False


def _xyzpos(
    position: Position | None,
    channel: Channel | None,
    z_plan: AnyZPlan | None,
    grid: PositionBase | None = None,
    z_pos: float | None = None,
) -> dict[str, float | None]:
    if z_pos is not None:
        # combine z_pos with z_offset
        if channel and channel.z_offset is not None:
            z_pos += channel.z_offset
        if z_plan and z_plan.is_relative:
            # TODO: either disallow without position z, or add concept of "current"
            z_pos += getattr(position, Axis.Z, None) or 0
    elif position:
        z_pos = position.z

    if grid:
        x_pos: float | None = grid.x
        y_pos: float | None = grid.y
        if grid.is_relative:
            px = getattr(position, "x", 0) or 0
            py = getattr(position, "y", 0) or 0
            x_pos = x_pos + px if x_pos is not None else None
            y_pos = y_pos + py if y_pos is not None else None
    else:
        x_pos = getattr(position, "x", None)
        y_pos = getattr(position, "y", None)

    return {"x_pos": x_pos, "y_pos": y_pos, "z_pos": z_pos}
