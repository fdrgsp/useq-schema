from __future__ import annotations

from heapq import merge
from typing import TYPE_CHECKING, TypeVar

from useq.v2._axes_iterator import AxisIterable, MultiAxisSequence

if TYPE_CHECKING:
    from collections.abc import Iterator

    from useq.v2._axes_iterator import AxesIndex, AxesIndexWithContext


V = TypeVar("V", covariant=True)


def order_axes(
    seq: MultiAxisSequence,
    axis_order: tuple[str, ...] | None = None,
) -> list[AxisIterable]:
    """Returns the axes of a MultiDimSequence in the order specified by seq.axis_order.

    If axis_order is provided, it overrides the sequence's axis_order.
    """
    if axis_order is None:
        axis_order = seq.axis_order
    if axis_order:
        axes_map = {axis.axis_key: axis for axis in seq.axes}
        return [axes_map[key] for key in axis_order if key in axes_map]
    return list(seq.axes)


def iterate_axes_recursive(
    axes: list[AxisIterable],
    prefix: AxesIndex | None = None,
    parent_order: tuple[str, ...] | None = None,
    context: tuple[MultiAxisSequence, ...] = (),
) -> Iterator[AxesIndexWithContext]:
    """Recursively iterate over a list of axes one at a time.

    If an axis yields a nested MultiDimSequence with a non-None value,
    that nested sequence acts as an override for its axis key.
    The parent's remaining axes having matching keys are removed, and the nested
    sequence's axes (ordered by its own axis_order if provided, or else the parent's)
    are appended.

    Before yielding a final combination (when no axes remain), we call should_skip
    on each axis (using the full prefix).
    """
    if prefix is None:
        prefix = {}

    if not axes:
        # Ask each axis in the prefix if the combination should be skipped
        if not any(axis.should_skip(prefix) for *_, axis in prefix.values()):
            yield prefix, context
        return

    current_axis, *remaining_axes = axes

    for idx, item in enumerate(current_axis):
        branch_prefix = dict(prefix)
        branch_context = context
        if isinstance(item, MultiAxisSequence):
            if item.value is None:
                raise NotImplementedError("Nested sequences must have a value.")

            value = item.value
            override_keys = {ax.axis_key for ax in item.axes}
            order = item.axis_order if item.axis_order is not None else parent_order

            # A nested axis that has already been consumed cannot be moved outside
            # the branch that introduced it.  Match the v1 compatibility behavior:
            # evaluate the override under parent index 0, then let the child replace
            # that selection within the branch.
            consumed_overrides = override_keys.intersection(branch_prefix)
            if any(branch_prefix[key][0] != 0 for key in consumed_overrides):
                continue
            for key in consumed_overrides:
                branch_prefix.pop(key)

            # Remove parent axes overridden by the nested sequence, merge in the
            # nested axes, and rank the complete continuation by the inherited order.
            parent_axes_not_overridden = [
                ax for ax in remaining_axes if ax.axis_key not in override_keys
            ]
            nested_axes_in_order = order_axes(item, order)
            updated_axes = parent_axes_not_overridden + nested_axes_in_order
            if order:
                rank = {key: idx for idx, key in enumerate(order)}
                updated_axes.sort(key=lambda axis: rank.get(axis.axis_key, len(rank)))

            # Use the nested sequence as the new context
            branch_context = (*context, item)
        else:
            value = item
            updated_axes = remaining_axes

        yield from iterate_axes_recursive(
            updated_axes,
            {**branch_prefix, current_axis.axis_key: (idx, value, current_axis)},
            parent_order=parent_order,
            context=branch_context,
        )


def iterate_multi_dim_sequence(
    seq: MultiAxisSequence, axis_order: tuple[str, ...] | None = None
) -> Iterator[AxesIndexWithContext]:
    """Iterate over a MultiDimSequence.

    Orders the base axes (if an axis_order is provided) and then iterates
    over all index combinations using iterate_axes_recursive.
    The parent's axis_order is passed down to nested sequences.

    Yields
    ------
    AxesIndexWithContext
        A tuple of (AxesIndex, MultiAxisSequence) where AxesIndex is a dictionary
        mapping axis keys to tuples of (index, value, axis), and MultiAxisSequence
        is the context that generated this axes combination.
    """
    if axis_order is None:
        axis_order = seq.axis_order
    ordered_axes = order_axes(seq, axis_order)
    position_axis = next((axis for axis in ordered_axes if axis.axis_key == "p"), None)
    if position_axis is None:
        yield from iterate_axes_recursive(
            ordered_axes, parent_order=axis_order, context=(seq,)
        )
        return

    inherited_axes = [axis for axis in ordered_axes if axis is not position_axis]
    root_order = list(axis_order or (axis.axis_key for axis in ordered_axes))
    extra_order: list[str] = []
    branch_iterators: list[Iterator[AxesIndexWithContext]] = []

    for p_idx, item in enumerate(position_axis):
        if isinstance(item, MultiAxisSequence):
            if item.value is None:
                raise NotImplementedError("Nested sequences must have a value.")
            value = item.value
            child_axes = order_axes(item, axis_order or item.axis_order)
            if any(axis.axis_key == "p" for axis in child_axes):
                raise ValueError(
                    "A Position sequence cannot have multiple stage positions."
                )
            context = (seq, item)
        else:
            value = item
            child_axes = []
            context = (seq,)

        override_keys = {axis.axis_key for axis in child_axes}
        effective_axes = [
            axis for axis in inherited_axes if axis.axis_key not in override_keys
        ] + child_axes
        rank = {key: idx for idx, key in enumerate(root_order)}
        effective_axes.sort(key=lambda axis: rank.get(axis.axis_key, len(rank)))
        for axis in effective_axes:
            if axis.axis_key not in root_order and axis.axis_key not in extra_order:
                extra_order.append(axis.axis_key)

        prefix = {position_axis.axis_key: (p_idx, value, position_axis)}
        branch_iterators.append(
            iterate_axes_recursive(
                effective_axes,
                prefix,
                parent_order=axis_order,
                context=context,
            )
        )

    global_order = (*root_order, *extra_order)

    def _sort_key(item: AxesIndexWithContext) -> tuple[int, ...]:
        selection, _ = item
        return tuple(
            selection[axis][0] if axis in selection else 0 for axis in global_order
        )

    yield from merge(*branch_iterators, key=_sort_key)
