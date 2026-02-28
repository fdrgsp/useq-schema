from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import useq

p_inputs = [
    ({"x": 0, "y": 1, "z": 2}, (0, 1, 2)),
    ({"y": 200}, (None, 200, None)),
    ((100, 200, 300), (100, 200, 300)),
    (
        {
            "z": 100,
            "sequence": {"z_plan": {"above": 8, "below": 4, "step": 2}},
        },
        (None, None, 100),
    ),
    (np.ones(3), (1, 1, 1)),
    ((None, 200, None), (None, 200, None)),
    (np.ones(2), (1, 1, None)),
    (np.array([0, 0, 0]), (0, 0, 0)),
    (np.array([0, 0]), (0, 0, None)),
    (useq.Position(x=100, y=200, z=300), (100, 200, 300)),
]


@pytest.mark.parametrize("position, pexpectation", p_inputs)
def test_stage_positions(position: Any, pexpectation: Sequence[float]) -> None:
    position = useq.Position.model_validate(position)
    assert (position.x, position.y, position.z) == pexpectation


_ABSOLUTE_GRID_PLANS = [
    pytest.param(
        {"top": 1, "bottom": -1, "left": 0, "right": 0},
        id="GridFromEdges",
    ),
    pytest.param(
        {"vertices": [(0, 0), (4, 0), (2, 4)], "fov_width": 2, "fov_height": 2},
        id="GridFromPolygon",
    ),
]

_RELATIVE_SUB_GRID_PLANS = [
    pytest.param({"rows": 2, "columns": 2}, id="GridRowsColumns"),
    pytest.param(
        useq.RandomPoints(num_points=3, max_width=100, max_height=100),
        id="RandomPoints",
    ),
]


@pytest.mark.parametrize("grid_plan", _ABSOLUTE_GRID_PLANS)
def test_position_warns_on_absolute_sub_sequence_grid(
    grid_plan: dict,
) -> None:
    """Position clears x/y and warns at construction when sub-sequence uses an absolute grid."""
    with pytest.warns(UserWarning, match="is ignored when a position sequence uses"):
        pos = useq.Position(x=1, y=2, sequence={"grid_plan": grid_plan})
    assert pos.x is None
    assert pos.y is None


@pytest.mark.parametrize("sub_plan", _RELATIVE_SUB_GRID_PLANS)
def test_position_no_warn_on_relative_sub_sequence_grid(sub_plan: Any) -> None:
    """Position keeps x/y when sub-sequence uses a relative grid."""
    pos = useq.Position(x=1, y=2, sequence={"grid_plan": sub_plan})
    assert pos.x == 1
    assert pos.y == 2


@pytest.mark.parametrize("sub_plan", _RELATIVE_SUB_GRID_PLANS)
def test_position_none_xy_allowed_with_relative_sub_sequence_grid(
    sub_plan: Any,
) -> None:
    """Position with x=None or y=None is allowed when sub-sequence uses a relative grid."""
    pos = useq.Position(x=None, y=None, z=3, sequence={"grid_plan": sub_plan})
    assert pos.x is None
    assert pos.y is None
    assert pos.z == 3


def test_add_with_none_coordinates() -> None:
    """__add__ preserves None when self coordinate is None (no fallback to other)."""
    pos = useq.Position(x=None, y=None, z=3)
    rel = useq.RelativePosition(x=5, y=10, z=0)
    result = pos + rel
    assert result.x is None
    assert result.y is None
    assert result.z == 3


def test_add_both_have_values() -> None:
    """__add__ sums values when both sides have them."""
    pos = useq.Position(x=1, y=2, z=3)
    rel = useq.RelativePosition(x=5, y=10, z=1)
    result = pos + rel
    assert result.x == 6
    assert result.y == 12
    assert result.z == 4


def test_radd() -> None:
    """__radd__ supports reversed addition (RelativePosition + AbsolutePosition)."""
    pos = useq.Position(x=1, y=2, z=3)
    rel = useq.RelativePosition(x=5, y=10, z=0)
    result = rel + pos
    assert result.x == 6
    assert result.y == 12
    assert result.z == 3


@pytest.mark.parametrize("grid_plan", _ABSOLUTE_GRID_PLANS)
def test_z_only_position_with_absolute_grid(grid_plan: dict) -> None:
    """Position(x=None, y=None, z=3) stays AbsolutePosition with absolute grid."""
    seq = useq.MDASequence(
        stage_positions=[(None, None, 3)],
        grid_plan=grid_plan,
    )
    assert len(seq.stage_positions) == 1
    pos = seq.stage_positions[0]
    assert isinstance(pos, useq.Position)
    assert pos.x is None
    assert pos.y is None
    assert pos.z == 3


@pytest.mark.parametrize("grid_plan", _ABSOLUTE_GRID_PLANS)
def test_warns_and_clears_xy_with_absolute_grid(grid_plan: dict) -> None:
    """Warns when positions have x/y with absolute grid, clears x/y."""
    with pytest.warns(UserWarning, match="is ignored when using"):
        seq = useq.MDASequence(
            stage_positions=[(1, 2, 3)],
            grid_plan=grid_plan,
        )
    pos = seq.stage_positions[0]
    assert isinstance(pos, useq.Position)
    assert pos.x is None
    assert pos.y is None
    assert pos.z == 3


@pytest.mark.parametrize("grid_plan", _ABSOLUTE_GRID_PLANS)
def test_sub_sequence_absolute_grid_stays_position(grid_plan: dict) -> None:
    """Position with sub-sequence absolute grid stays AbsolutePosition."""
    seq = useq.MDASequence(
        stage_positions=[
            useq.Position(x=None, y=None, z=3.0, sequence={"grid_plan": grid_plan})
        ]
    )
    pos = seq.stage_positions[0]
    assert isinstance(pos, useq.Position)
    assert pos.x is None
    assert pos.y is None
    assert pos.z == 3.0
    assert pos.sequence is not None


@pytest.mark.parametrize("grid_plan", _ABSOLUTE_GRID_PLANS)
def test_sub_sequence_absolute_grid_warns_and_clears_xy(grid_plan: dict) -> None:
    """Position with x/y + sub-sequence absolute grid warns then clears x/y."""
    with pytest.warns(UserWarning, match="is ignored when a position sequence uses"):
        seq = useq.MDASequence(
            stage_positions=[
                useq.Position(x=1, y=2, z=3.0, sequence={"grid_plan": grid_plan})
            ]
        )
    pos = seq.stage_positions[0]
    assert isinstance(pos, useq.Position)
    assert pos.x is None
    assert pos.y is None
    assert pos.z == 3.0


def test_warns_global_abs_grid_does_not_mutate_original_position() -> None:
    """Clearing x/y for a global absolute grid must not mutate the original Position."""
    pos = useq.Position(x=1, y=2, z=3)
    with pytest.warns(UserWarning, match="is ignored when using"):
        seq = useq.MDASequence(
            stage_positions=[pos],
            grid_plan={"top": 1, "bottom": -1, "left": 0, "right": 0},
        )
    assert pos.x == 1  # original must be untouched
    assert pos.y == 2
    assert seq.stage_positions[0].x is None  # sequence view is updated
    assert seq.stage_positions[0].y is None


def test_relative_position_rejected_in_stage_positions() -> None:
    """RelativePosition is always rejected in stage_positions."""
    with pytest.raises(Exception, match="RelativePosition cannot be used"):
        useq.MDASequence(stage_positions=[useq.RelativePosition(x=1, y=2, z=3)])

    # Also rejected even with an absolute grid
    with pytest.raises(Exception, match="RelativePosition cannot be used"):
        useq.MDASequence(
            stage_positions=[useq.RelativePosition(z=3)],
            grid_plan={"top": 1, "bottom": -1, "left": 0, "right": 0},
        )


_RELATIVE_GRID_PLANS = [
    pytest.param({"rows": 2, "columns": 2}, id="GridRowsColumns"),
    pytest.param(
        useq.RandomPoints(num_points=3, max_width=100, max_height=100),
        id="RandomPoints",
    ),
]


@pytest.mark.parametrize("grid_plan", _RELATIVE_GRID_PLANS)
def test_position_none_xy_allowed_with_relative_grid(grid_plan: dict) -> None:
    """Position with x=None or y=None is allowed when global grid is relative."""
    seq = useq.MDASequence(
        stage_positions=[(None, None, 3)],
        grid_plan=grid_plan,
    )
    pos = seq.stage_positions[0]
    assert pos.x is None
    assert pos.y is None
    assert pos.z == 3


@pytest.mark.parametrize("grid_plan", _RELATIVE_GRID_PLANS)
def test_position_none_xy_with_relative_grid_exempt_if_own_absolute_sub_grid(
    grid_plan: dict,
) -> None:
    """Position with x=None/y=None is exempt if it has its own absolute sub-seq grid."""
    seq = useq.MDASequence(
        stage_positions=[
            useq.Position(x=1, y=2, z=3),
            useq.Position(
                name="sub",
                sequence={"grid_plan": {"top": 1, "bottom": -1, "left": 0, "right": 0}},
            ),
        ],
        grid_plan=grid_plan,
    )
    assert len(seq.stage_positions) == 2


@pytest.mark.parametrize("global_plan", _ABSOLUTE_GRID_PLANS)
@pytest.mark.parametrize("sub_plan", _RELATIVE_SUB_GRID_PLANS)
def test_position_with_xy_and_relative_sub_grid_exempt_from_absolute_global_grid(
    global_plan: dict, sub_plan: Any
) -> None:
    """Position with x/y + relative sub-grid is exempt from global abs grid x/y clearing."""
    seq = useq.MDASequence(
        stage_positions=[
            useq.Position(x=1, y=2, z=3, sequence={"grid_plan": sub_plan})
        ],
        grid_plan=global_plan,
    )
    pos = seq.stage_positions[0]
    assert pos.x == 1
    assert pos.y == 2


@pytest.mark.parametrize("grid_plan", _ABSOLUTE_GRID_PLANS)
def test_absolute_grid_warns_only_positions_with_xy(grid_plan: dict) -> None:
    """With a global absolute grid, only positions that have x/y produce a warning."""
    with pytest.warns(UserWarning) as record:
        seq = useq.MDASequence(
            stage_positions=[(1, 2, 3), (None, None, 5)],
            grid_plan=grid_plan,
        )
    # exactly one "is ignored" warning: only the first position had x/y
    xy_warns = [w for w in record if "is ignored when using" in str(w.message)]
    assert len(xy_warns) == 1
    assert seq.stage_positions[0].x is None
    assert seq.stage_positions[0].y is None
    assert seq.stage_positions[0].z == 3
    assert seq.stage_positions[1].x is None  # was already None, no warning
    assert seq.stage_positions[1].y is None
    assert seq.stage_positions[1].z == 5


@pytest.mark.parametrize("grid_plan", _RELATIVE_GRID_PLANS)
def test_relative_grid_allows_missing_xy(grid_plan: dict) -> None:
    """With a relative global grid, missing x/y are allowed (stay None)."""
    seq = useq.MDASequence(
        stage_positions=[
            useq.Position(x=1, y=2, z=0),
            useq.Position(x=None, y=None, z=3),
        ],
        grid_plan=grid_plan,
    )
    assert seq.stage_positions[0].x == 1
    assert seq.stage_positions[0].y == 2
    assert seq.stage_positions[1].x is None
    assert seq.stage_positions[1].y is None


@pytest.mark.parametrize("grid_plan", _ABSOLUTE_GRID_PLANS)
def test_z_only_position_iterates_with_absolute_grid(grid_plan: dict) -> None:
    """Position(x=None, y=None, z=3) iterates correctly: grid provides x/y, pos z."""
    seq = useq.MDASequence(
        stage_positions=[useq.Position(x=None, y=None, z=3)],
        grid_plan=grid_plan,
    )
    events = list(seq)
    assert len(events) > 0
    for event in events:
        assert event.x_pos is not None
        assert event.y_pos is not None
        assert event.z_pos == 3.0
