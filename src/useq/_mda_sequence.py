from __future__ import annotations

import contextlib
from itertools import product
from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)
from uuid import UUID, uuid4
from warnings import warn

import numpy as np
from pydantic import Field, PrivateAttr, root_validator, validator

from . import MDAEvent, _mda_event
from ._autofocus import AnyAF, AxesBasedAF, NoAF
from ._base_model import UseqModel
from ._channel import Channel
from ._grid import AnyGridPlan, GridPosition, NoGrid
from ._position import Position
from ._time import AnyTimePlan, NoT
from ._z import AnyZPlan, NoZ

TIME = "t"
CHANNEL = "c"
POSITION = "p"
Z = "z"
GRID = "g"
INDICES = (TIME, POSITION, GRID, CHANNEL, Z)

Undefined = object()


class MDASequence(UseqModel):
    """A sequence of MDA (Multi-Dimensional Acquisition) events.

    This is the core object in the `useq` library, and is used define a sequence of
    events to be run on a microscope. It object may be constructed manually, or from
    file (e.g. json or yaml).

    The object itself acts as an iterator for [`useq.MDAEvent`][] objects:

    Attributes
    ----------
    metadata : dict
        A dictionary of user metadata to be stored with the sequence.
    axis_order : str
        The order of the axes in the sequence. Must be a permutation of `"tpgcz"`. The
        default is `"tpgcz"`.
    stage_positions : tuple[Position, ...]
        The stage positions to visit. (each with `x`, `y`, `z`, `name`, and `sequence`,
        all of which are optional).
    grid_plan : GridFromEdges, GridRelative, NoGrid
        The grid plan to follow. One of `GridFromEdges`, `GridRelative` or `NoGrid`.
    channels : tuple[Channel, ...]
        The channels to acquire. see `Channel`.
    time_plan : MultiPhaseTimePlan | TIntervalDuration | TIntervalLoops \
        | TDurationLoops | NoT
        The time plan to follow. One of `TIntervalDuration`, `TIntervalLoops`,
        `TDurationLoops`, `MultiPhaseTimePlan`, or `NoT`
    z_plan : ZTopBottom | ZRangeAround | ZAboveBelow | ZRelativePositions | \
        ZAbsolutePositions | NoZ
        The z plan to follow. One of `ZTopBottom`, `ZRangeAround`, `ZAboveBelow`,
        `ZRelativePositions`, `ZAbsolutePositions`, or `NoZ`.
    autofocus_plan : AxesBasedAF | NoAF
        The hardware autofocus plan to follow. One of `AxesBasedAF` or `NoAF`.
        TODO: write info...
    uid : UUID
        A read-only unique identifier (uuid version 4) for the sequence. This will be
        generated, do not set.

    Examples
    --------
    >>> from useq import MDASequence, Position, Channel, TIntervalDuration
    >>> seq = MDASequence(
    ...     time_plan={"interval": 0.1, "loops": 2},
    ...     stage_positions=[(1, 1, 1)],
    ...     grid_plan={"rows": 2, "cols": 2},
    ...     z_plan={"range": 3, "step": 1},
    ...     channels=[{"config": "DAPI", "exposure": 1}]
    ... )
    >>> print(seq)
    Multi-Dimensional Acquisition ▶ nt: 2, np: 1, nc: 1, nz: 4, ng: 4

    >>> for event in seq:
    ...     print(event)

    >>> print(seq.yaml())
    channels:
    - config: DAPI
      exposure: 1.0
    grid_plan:
      columns: 2
      rows: 2
    stage_positions:
    - x: 1.0
      y: 1.0
      z: 1.0
    time_plan:
      interval: '0:00:00.100000'
      loops: 2
    z_plan:
      range: 3.0
      step: 1.0
    """

    metadata: Dict[str, Any] = Field(default_factory=dict)
    axis_order: str = "".join(INDICES)
    stage_positions: Tuple[Position, ...] = Field(default_factory=tuple)
    grid_plan: AnyGridPlan = Field(default_factory=NoGrid)
    channels: Tuple[Channel, ...] = Field(default_factory=tuple)
    time_plan: AnyTimePlan = Field(default_factory=NoT)
    z_plan: AnyZPlan = Field(default_factory=NoZ)
    autofocus_plan: AnyAF = Field(default_factory=NoAF)

    _uid: UUID = PrivateAttr(default_factory=uuid4)
    _length: Optional[int] = PrivateAttr(default=None)
    _fov_size: Tuple[float, float] = PrivateAttr(default=(1, 1))

    @property
    def uid(self) -> UUID:
        """A unique identifier for this sequence."""
        return self._uid

    def set_fov_size(self, fov_size: Tuple[float, float]) -> None:
        """Set the field of view size.

        This is used to calculate the number of positions in a grid plan.
        """
        self._fov_size = fov_size

    @no_type_check
    def replace(
        self,
        metadata: Dict[str, Any] = Undefined,
        axis_order: str = Undefined,
        stage_positions: Tuple[Position, ...] = Undefined,
        grid_plan: AnyGridPlan = Undefined,
        channels: Tuple[Channel, ...] = Undefined,
        time_plan: AnyTimePlan = Undefined,
        z_plan: AnyZPlan = Undefined,
        autofocus_plan: AnyAF = Undefined,
    ) -> MDASequence:
        """Return a new `MDAsequence` replacing specified kwargs with new values.

        MDASequences are immutable, so this method is useful for creating a new
        sequence with only a few fields changed.  The uid of the new sequence will
        be different from the original.
        """
        kwargs = {
            k: v for k, v in locals().items() if v is not Undefined and k != "self"
        }
        state = self.dict(exclude={"uid"})
        return type(self)(**{**state, **kwargs})

    def __hash__(self) -> int:
        return hash(self.uid)

    @validator("z_plan", pre=True)
    def validate_zplan(cls, v: Any) -> Union[dict, NoZ]:
        return v or NoZ()

    @validator("time_plan", pre=True)
    def validate_time_plan(cls, v: Any) -> Union[dict, NoT]:
        return {"phases": v} if isinstance(v, (tuple, list)) else v or NoT()

    @validator("stage_positions", pre=True)
    def validate_positions(cls, v: Any) -> Any:
        if isinstance(v, np.ndarray):
            if v.ndim == 1:
                return [v]
            elif v.ndim == 2:
                return list(v)
        return v

    @validator("axis_order", pre=True)
    def validate_axis_order(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise TypeError(f"acquisition order must be a string, got {type(v)}")
        order = v.lower()
        extra = {x for x in order if x not in INDICES}
        if extra:
            raise ValueError(
                f"Can only iterate over axes: {INDICES!r}. Got extra: {extra}"
            )
        if len(set(order)) < len(order):
            raise ValueError(f"Duplicate entries found in acquisition order: {order}")

        return order

    @root_validator
    def validate_mda(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "axis_order" in values:
            values["axis_order"] = cls._check_order(
                values["axis_order"],
                z_plan=values.get("z_plan"),
                stage_positions=values.get("stage_positions", ()),
                channels=values.get("channels", ()),
                grid_plan=values.get("grid_plan"),
                autofocus_plan=values.get("autofocus_plan"),
            )
        return values

    def __eq__(self, other: Any) -> bool:
        """Return `True` if two `MDASequences` are equal (uid is excluded)."""
        if isinstance(other, MDASequence):
            return bool(self.dict(exclude={"uid"}) == other.dict(exclude={"uid"}))
        else:
            return False

    @staticmethod
    def _check_order(
        order: str,
        z_plan: Optional[AnyZPlan] = None,
        stage_positions: Sequence[Position] = (),
        channels: Sequence[Channel] = (),
        grid_plan: Optional[AnyGridPlan] = None,
        autofocus_plan: Optional[AnyAF] = None,
    ) -> str:
        if (
            Z in order
            and POSITION in order
            and order.index(Z) < order.index(POSITION)
            and z_plan
            and any(p.sequence.z_plan for p in stage_positions if p.sequence)
        ):
            raise ValueError(
                f"{Z!r} cannot precede {POSITION!r} in acquisition order if "
                "any position specifies a z_plan"
            )

        if (
            CHANNEL in order
            and TIME in order
            and any(c.acquire_every > 1 for c in channels)
            and order.index(CHANNEL) < order.index(TIME)
        ):
            warn(
                f"Channels with skipped frames detected, but {CHANNEL!r} precedes "
                "{TIME!r} in the acquisition order: may not yield intended results.",
                stacklevel=2,
            )

        if (
            GRID in order
            and POSITION in order
            and grid_plan is not None
            and not isinstance(grid_plan, NoGrid)
            and not grid_plan.is_relative
            and len(stage_positions) > 1
        ):
            sub_position_grid_plans = [
                p for p in stage_positions if p.sequence and p.sequence.grid_plan
            ]
            if len(stage_positions) - len(sub_position_grid_plans) > 1:
                warn(
                    "Global grid plan will override sub-position grid plans.",
                    stacklevel=2,
                )

        if (
            POSITION in order
            and stage_positions
            and any(p.sequence.stage_positions for p in stage_positions if p.sequence)
        ):
            raise ValueError(
                "Currently, a Position sequence cannot have multiple stage positions!"
            )

        if (
            Z in order
            and z_plan is not None
            and not z_plan.is_relative
            and isinstance(autofocus_plan, AxesBasedAF)
        ):
            raise ValueError("Autofocus plan cannot be used with absolute Z positions!")

        return order

    def __str__(self) -> str:
        shape = [
            f"n{k.lower()}: {len(list(self.iter_axis(k)))}" for k in self.axis_order
        ]
        return "Multi-Dimensional Acquisition ▶ " + ", ".join(shape)

    def __len__(self) -> int:
        """Return the number of events in this sequence."""
        if self._length is None:
            self._length = len(list(self.iter_events()))
        return self._length

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of this sequence.

        !!! note
            This doesn't account for jagged arrays, like skipped Z or channel frames.
        """
        return tuple(s for s in self.sizes.values() if s)

    @property
    def sizes(self) -> Dict[str, int]:
        """Mapping of axis to size of that axis."""
        return {k: len(list(self.iter_axis(k))) for k in self.axis_order}

    @property
    def used_axes(self) -> str:
        """Single letter string of axes used in this sequence, e.g. `ztc`."""
        return "".join(k for k in self.axis_order if self.sizes[k])

    def iter_axis(
        self, axis: str
    ) -> Iterator[Position | Channel | float | GridPosition]:
        """Iterate over the events of a given axis."""
        yield from {
            TIME: self.time_plan,
            POSITION: self.stage_positions,
            Z: self.z_plan,
            CHANNEL: self.channels,
            GRID: self.grid_plan.iter_grid_positions(
                self._fov_size[0], self._fov_size[1]
            ),
        }[axis]

    def __iter__(self) -> Iterator[MDAEvent]:  # type: ignore [override]
        """Same as `iter_events`. Supports `for event in sequence: ...` syntax."""
        yield from self.iter_events()

    class _SkipFrame(Exception):
        pass

    def iter_events(self) -> Iterator[MDAEvent]:
        """Iterate over all events in the MDA sequence.

        See source of [useq._mda_sequence.iter_sequence][] for details on how
        events are constructed and yielded.

        Yields
        ------
        MDAEvent
            Each event in the MDA sequence.
        """
        return iter_sequence(self)

    def _combine_z(
        self,
        z_pos: float,
        z_ind: int,
        channel: Optional[Channel],
        position: Optional[Position],
    ) -> float:
        if channel:
            # only acquire on the middle plane:
            if not channel.do_stack and z_ind != len(self.z_plan) // 2:
                raise self._SkipFrame()
            if channel.z_offset is not None:
                z_pos += channel.z_offset
        if self.z_plan.is_relative:
            # TODO: either disallow without position z, or add concept of "current"
            z_pos += getattr(position, Z, None) or 0
        return z_pos

    def to_pycromanager(self) -> list[dict]:
        """Convenience to convert this sequence to a list of pycro-manager events.

        See: <https://pycro-manager.readthedocs.io/en/latest/apis.html>
        """
        return [event.to_pycromanager() for event in self]


MDAEvent.update_forward_refs(MDASequence=MDASequence)
Position.update_forward_refs(MDASequence=MDASequence)


def _get_autofocus_axes(sequence: MDASequence) -> dict[str, Union[str, None]]:
    """Return the axes that have autofocus plans and set them to 'None'."""
    return (
        # {ax: None for ax in sequence.autofocus_plan.axes if ax in sequence.used_axes}
        {ax: None for ax in sequence.autofocus_plan.axes}
        if isinstance(sequence.autofocus_plan, AxesBasedAF)
        and sequence.autofocus_plan.axes is not None
        and sequence.autofocus_plan.autofocus_z_device_name
        else {}
    )


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

    Yields
    ------
    MDAEvent
        Each event in the MDA sequence.
    """
    global_index = 0
    order = sequence.used_axes

    # get the axes that have autofocus plans and set them to 'None' to make sure that
    # autofocus will be executed at the first event
    # example: autofocus_axes {'t': None, 'p': None}
    autofocus_axes = _get_autofocus_axes(sequence)

    event_iterator = (enumerate(sequence.iter_axis(ax)) for ax in order)
    for item in product(*event_iterator):
        if not item:  # the case with no events
            continue  # pragma: no cover

        _ev = dict(zip(order, item))

        index = {k: _ev[k][0] for k in INDICES if k in _ev}
        position = cast(
            "Position | None", _ev[POSITION][1] if POSITION in _ev else None
        )
        channel = cast("Channel | None", _ev[CHANNEL][1] if CHANNEL in _ev else None)

        # check if we should skip this event
        if _skip(channel, position, index):
            continue

        # skip if also in position.sequence
        if position and position.sequence:
            # NOTE: if we ever add more plans, they will need to be explicitly added
            # https://github.com/pymmcore-plus/useq-schema/pull/85

            # get if sub-sequence has any plan
            plans = any(
                (
                    position.sequence.grid_plan,
                    position.sequence.z_plan,
                    position.sequence.time_plan,
                )
            )
            # overwriting the *global* channel index since it is no longer relevant.
            # if channel IS SPECIFIED in the position.sequence WITH any plan,
            # we skip otherwise the channel will be acquired twice. Same happens if
            # the channel IS NOT SPECIFIED but ANY plan is.
            if (
                CHANNEL in index
                and index[CHANNEL] != 0
                and ((position.sequence.channels and plans) or not plans)
            ):
                continue
            if Z in index and index[Z] != 0 and position.sequence.z_plan:
                continue
            if GRID in index and index[GRID] != 0 and position.sequence.grid_plan:
                continue

        pos_name = getattr(position, "name", None)
        time = cast("int | None", _ev[TIME][1] if TIME in _ev else None)
        grid = cast("GridPosition | None", _ev[GRID][1] if GRID in _ev else None)
        _exposure = getattr(channel, "exposure", None)
        _channel = (
            _mda_event.Channel(config=channel.config, group=channel.group)
            if channel
            else None
        )

        # get the z position
        try:
            z_pos = _get_z(sequence, _ev, index, position, channel)
        except sequence._SkipFrame:
            continue

        # get the x and y position
        if grid is not None:
            x_pos, y_pos = _get_grid_xy(position, grid)
        else:
            x_pos = getattr(position, "x", None)
            y_pos = getattr(position, "y", None)

        # if autofocus plan is defined for the sequence, check if it should be executed
        if isinstance(sequence.autofocus_plan, AxesBasedAF):
            autofocus, autofocus_axes = _setup_autofocus(
                sequence, autofocus_axes, index, z_pos
            )
        else:
            autofocus = NoAF()

        # if position has a sequence containing ONLY an autofocus plan, using
        # 'position.sequence' will return None. So we need to check directly
        # 'position.sequence.autofocus_plan'.
        try:
            pos_seq_af = position.sequence.autofocus_plan  # type: ignore
        except AttributeError:
            pos_seq_af = None

        # if position has a sequence or has a sequence containing ONLY an autofocus plan
        if position and (position.sequence or pos_seq_af):
            pos_seq = cast(MDASequence, position.sequence)

            if isinstance(pos_seq.autofocus_plan, NoAF):
                pos_seq = pos_seq.replace(autofocus_plan=sequence.autofocus_plan)

            if isinstance(autofocus, AxesBasedAF) and autofocus.axes:
                # reset autofocus_axes to None
                for ax in autofocus.axes:
                    autofocus_axes[ax] = None

            # if the sub-sequence has ONLY autofocus plan, 'iter_sequence' will not work
            # because it will yield an empty list. So we need to create a new MDAEvent
            # with the autofocus plan.
            events_list = list(iter_sequence(pos_seq)) or [
                MDAEvent(autofocus=pos_seq_af)
            ]

            for sub_event in events_list:
                # we're going to create a modified sub-event, inheriting some of the
                # values from the parent event, and shifting the position of the
                # event to account for global position offsets (or override if the
                # sub-event has an absolute XYZ position plan.)
                update_kwargs = dict(
                    global_index=global_index,
                    index={**index, **sub_event.index},
                    sequence=sequence,
                    pos_name=position.name or pos_name,
                    **_maybe_shifted_positions(
                        sub_event=sub_event,
                        position=position,
                        x_pos=x_pos,
                        y_pos=y_pos,
                        z_pos=z_pos,
                    ),
                )

                # time, exposure, and channel are inherited from the parent event
                # if they are not present on the sub-event.
                for val, name in [
                    (time, "min_start_time"),
                    (_exposure, "exposure"),
                    (_channel, "channel"),
                ]:
                    subval = getattr(sub_event, name)
                    update_kwargs[name] = subval if subval is not None else val

                if not autofocus_axes:
                    autofocus_axes = _get_autofocus_axes(pos_seq)

                # in case each position sequence has a different autofocus plan we
                # need to reset the autofocus_axes
                elif any(
                    ax
                    for ax in autofocus_axes
                    if isinstance(pos_seq.autofocus_plan, AxesBasedAF)
                    and ax not in pos_seq.autofocus_plan.axes
                ):
                    autofocus_axes = _get_autofocus_axes(pos_seq)

                # update sub_event autofocus plan
                autofocus, autofocus_axes = _setup_autofocus(
                    pos_seq,
                    autofocus_axes,
                    update_kwargs["index"],
                    update_kwargs.get("z_pos"),
                    sequence,
                )
                update_kwargs["autofocus"] = autofocus

                yield sub_event.copy(update=update_kwargs)
                global_index += 1

            continue

        yield MDAEvent(
            index=index,
            min_start_time=time,
            pos_name=pos_name,
            x_pos=x_pos,
            y_pos=y_pos,
            z_pos=z_pos,
            autofocus=autofocus,
            exposure=_exposure,
            channel=_channel,
            sequence=sequence,
            global_index=global_index,
        )
        global_index += 1


def _skip(
    channel: Channel | None, position: Position | None, index: dict[str, Any]
) -> bool:
    # skip channels
    if channel and TIME in index and index[TIME] % channel.acquire_every:
        return True
    # skip if also in position.sequence
    if position and position.sequence:
        # NOTE: if we ever add more plans, they will need to be explicitly added
        # https://github.com/pymmcore-plus/useq-schema/pull/85

        # get if sub-sequence has any plan
        plans = any(
            (
                position.sequence.grid_plan,
                position.sequence.z_plan,
                position.sequence.time_plan,
            )
        )
        # overwriting the *global* channel index since it is no longer relevant.
        # if channel IS SPECIFIED in the position.sequence WITH any plan,
        # we skip otherwise the channel will be acquired twice (see PR
        # https://github.com/pymmcore-plus/useq-schema/pull/93).
        # Same happens if the channel IS NOT SPECIFIED but ANY plan is.
        if (
            CHANNEL in index
            and index[CHANNEL] != 0
            and ((position.sequence.channels and plans) or not plans)
        ):
            return True
        if Z in index and index[Z] != 0 and position.sequence.z_plan:
            return True
        if GRID in index and index[GRID] != 0 and position.sequence.grid_plan:
            return True
    return False


def _get_z(
    sequence: MDASequence,
    _ev: dict[str, Any],
    index: dict[str, Any],
    position: Position | None,
    channel: Channel | None,
) -> float | None:
    """Return the z position for the current event."""
    return (
        sequence._combine_z(_ev[Z][1], index[Z], channel, position)
        if Z in _ev
        else position.z + channel.z_offset
        if (
            position
            and position.z is not None
            and channel
            and channel.z_offset is not None
        )
        else position.z
        if position
        else None
    )


def _get_grid_xy(
    position: Position | None, grid: GridPosition
) -> tuple[float | None, float | None]:
    """Return the x and y position for the current event based on a grid_plan."""
    x_pos: Optional[float] = grid.x
    y_pos: Optional[float] = grid.y
    if grid.is_relative:
        px = getattr(position, "x", 0) or 0
        py = getattr(position, "y", 0) or 0
        x_pos = x_pos + px if x_pos is not None else None
        y_pos = y_pos + py if y_pos is not None else None
    return x_pos, y_pos


def _setup_autofocus(
    sequence: MDASequence,
    autofocus_axes: dict[str, Union[str, None]],
    index: dict[str, Any],
    z_pos: float | None,
    main_sequence: Optional[MDASequence] = None,
) -> tuple[AnyAF, dict[str, Union[str, None]]]:
    """Return the autofocus plan for the current event and update autofocus_axes.

    Example:
    -------
    INPUTS:
        autofocus_axes {'t': None, 'p': None}
        index = {'t': 0, 'p': 0, 'z': 0}
    OUTPUTS:
        autofocus {
            'autofocus_z_device_name': 'z',
            'z_stage_position': 0.0,
            'af_motor_offset': 0.0
            'axes': ('t', 'p')}
        autofocus_axes {'t': 0, 'p': 0}.
    """
    autofocus = sequence.autofocus_plan

    # remove autofocus axes that are not in the current index
    for key in list(autofocus_axes):
        if key not in index:
            autofocus_axes.pop(key)

    if not autofocus_axes:
        return NoAF(), autofocus_axes

    for ax in autofocus_axes:
        with contextlib.suppress(KeyError):
            if autofocus_axes[ax] is None or autofocus_axes[ax] != index[ax]:
                # update the autofocus "z_stage_position" with z_pos
                update_af_kwargs = {}
                if z_pos is not None:
                    if "z" not in index:
                        z = z_pos
                    else:
                        # if z_plan, get the z from the center of the stack
                        try:
                            z = z_pos - list(sequence.z_plan)[index["z"]]
                        # if the z_plan is not defined in the sub_sequence, then use
                        # the z_plan from the main_sequence
                        except IndexError:
                            z = (
                                (z_pos - list(main_sequence.z_plan)[index["z"]])
                                if main_sequence is not None
                                else None
                            )
                    update_af_kwargs["z_stage_position"] = z
                autofocus = sequence.autofocus_plan.copy(update=update_af_kwargs)
                break

    # if doesn't have z_stage_position or af_motor_offset, then set to NoAF
    if autofocus.z_stage_position is None or autofocus.af_motor_offset is None:
        autofocus = NoAF()

    # update the autofocus_axes dict with the current index if AxesBasedAF
    if isinstance(autofocus, AxesBasedAF):
        with contextlib.suppress(KeyError):
            for ax in autofocus.axes:
                autofocus_axes[ax] = index[ax]

    return autofocus, autofocus_axes


def _maybe_shifted_positions(
    sub_event: MDAEvent,  # the event we just created in the position sequence
    position: Position,  # the position we are iterating
    z_pos: float | None,  # global z position
    x_pos: float | None,  # global x position
    y_pos: float | None,  # global y position
) -> dict:
    """Return a dict of kwargs for the sub_event, accounting for position shifts."""
    kwargs = {}
    # this function should only be called inside the sub-iteration loop
    # so we can assume that position.sequence is not None
    pos_seq = cast("MDASequence", position.sequence)

    # if the position sequence has no z_plan, then we can use the global z_pos
    # elif the position has a relative z_plan, then we need to shift the sub_event
    if not pos_seq.z_plan:
        kwargs["z_pos"] = z_pos
    elif pos_seq.z_plan.is_relative:
        kwargs["z_pos"] = _shift_axis("z", position.z, sub_event)

    # if the position sequence has no grid_plane, then we can use the global z_pos
    # elif the position has a relative grid_plan, then we need to shift the sub_event
    if not pos_seq.grid_plan:
        kwargs["x_pos"] = x_pos
        kwargs["y_pos"] = y_pos
    elif pos_seq.grid_plan.is_relative:
        kwargs["x_pos"] = _shift_axis("x", position.x, sub_event)
        kwargs["y_pos"] = _shift_axis("y", position.y, sub_event)
    return kwargs


def _shift_axis(axis: str, new_val: float | None, event: MDAEvent) -> float | None:
    """Return a new value for the axis, accounting for the sub_event's position.

    If both new_val and the corresponding axis position for the sub_event are None,
    return None, otherwise return the sum of the two.
    """
    sub_pos = getattr(event, f"{axis}_pos")
    return (
        None
        if new_val is None and sub_pos is None
        else (new_val or 0) + sub_pos
        if sub_pos is not None
        else new_val
    )
