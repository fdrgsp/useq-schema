from __future__ import annotations

import re
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

import useq
from useq import FocusSurface

# z = 0.01 * x
TILTED = ((0, 0, 0), (1000, 0, 10), (0, 1000, 0), (1000, 1000, 10))
GRID = useq.GridRowsColumns(rows=1, columns=2, fov_width=10, fov_height=10)


def _surface(points: Any = TILTED, method: str = "linear") -> FocusSurface:
    return FocusSurface(points=points, method=method)


def _area(polygon: np.ndarray) -> float:
    """Signed area of `polygon`: positive if counterclockwise."""
    x, y = polygon.T
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


@pytest.mark.parametrize("method", ["linear", "plane"])
def test_reproduces_a_tilted_plane(method: str) -> None:
    # inside the points, on them, and outside them (extrapolated)
    xy = [(490, 100), (0, 0), (1000, 1000), (250, 750), (-200, 500), (1500, -300)]
    z = _surface(method=method).evaluate(xy)
    np.testing.assert_allclose(z, [4.9, 0, 10, 2.5, -2, 15], atol=1e-9)


def test_linear_follows_the_local_points() -> None:
    bump = [(0, 0, 0), (100, 0, 0), (0, 100, 0), (100, 100, 0), (50, 50, 8)]
    np.testing.assert_allclose(_surface(bump).evaluate([(50, 50), (25, 25)]), [8, 4])
    # the plane through the same points averages the bump out
    assert _surface(bump, "plane").evaluate([(50, 50)])[0] == pytest.approx(1.6)


def test_extrapolation_follows_the_edge_and_the_plane() -> None:
    surface = _surface(
        [(0, 0, 0), (100, 0, 0), (0, 100, 0), (100, 100, 0), (50, 50, 8)]
    )
    # continuous across the edge of the points' hull...
    z_in, z_on, z_out = surface.evaluate([(50, 1e-6), (50, 0), (50, -1e-6)])
    assert z_in == pytest.approx(z_on, abs=1e-5) == z_out
    # ...and away from it, with the slope of the fitted plane (flat here)
    assert surface.evaluate([(-50, -50)])[0] == pytest.approx(0)
    hull = surface.hull()
    assert {tuple(p) for p in hull} == {(0, 0), (100, 0), (100, 100), (0, 100)}
    assert _area(hull) > 0  # counterclockwise


@pytest.mark.parametrize(
    "points",
    [
        # random
        np.random.default_rng(0).uniform(-5000, 5000, (40, 2)),
        # collinear points first in the sweep, with the next point on either side
        [(0, 0), (0, 1000), (0, 2000), (0, 3000), (1000, 500), (2000, 2500)],
        [(0, 0), (0, 1000), (0, 2000), (-1000, 500)],
        # three points along a plate edge within 0.3 µm
        [(0, 0), (5000, 0.3), (10000, 0), (0, 8000), (10000, 8000), (4000, 4000)],
        # a regular grid: many points on the same circles
        [(x, y) for x in range(0, 4000, 1000) for y in range(0, 3000, 1000)],
    ],
)
def test_triangulation_covers_the_hull(points: list) -> None:
    xy = np.asarray(points, dtype=float)
    z = np.random.default_rng(1).normal(size=len(xy))
    surface = FocusSurface(points=np.column_stack([xy, z]).tolist())
    compiled = surface._compiled
    triangles = xy[compiled._triangles]
    assert sum(_area(t) for t in triangles) == pytest.approx(_area(surface.hull()))
    # Delaunay: no point inside the circumcircle of any triangle
    for a, b, c in compiled._unit[compiled._triangles]:
        for d in compiled._unit:
            assert useq._focus_surface._in_circle(a, b, c, d) < 1e-9
    # it goes through every point
    np.testing.assert_allclose(surface.evaluate(xy), z, atol=1e-9)


def test_ties_do_not_depend_on_the_order_of_the_points() -> None:
    rng = np.random.default_rng(2)
    grid = [
        (x, y, rng.normal()) for x in range(0, 4000, 1000) for y in range(0, 3000, 1000)
    ]
    xy = rng.uniform(0, 3000, (200, 2))
    z = _surface(grid).evaluate(xy)
    for _ in range(10):
        shuffled = [grid[i] for i in rng.permutation(len(grid))]
        np.testing.assert_allclose(_surface(shuffled).evaluate(xy), z)


@pytest.mark.parametrize(
    "points, error",
    [
        ([], "At least 3 reference points are needed (0 set)."),
        (TILTED[:2], "At least 3 reference points are needed (2 set)."),
        ([*TILTED, (0.01, 0, 3)], "Reference points 1 and 5 are at the same XY."),
        ([(0, 0, 0), (10, 10, 1), (20, 20, 2)], "lie on a line"),
        ([(0, 0, 0), (1000, 0, 1), (500, 0.1, 2)], "lie on a line"),
        ([(0, 0, 0), (1, 0, float("nan")), (0, 1, 0)], "finite number"),
    ],
)
def test_invalid_points(points: list, error: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(error)):
        FocusSurface(points=points)


def test_serialization() -> None:
    seq = useq.MDASequence(stage_positions=[(100, 100)], focus_surface=_surface())
    assert useq.MDASequence.model_validate_json(seq.model_dump_json()) == seq
    assert seq.model_dump()["focus_surface"] == {"points": TILTED, "method": "linear"}


# ------------------------------ MDASequence ---------------------------------


def _zs(**kwargs: Any) -> list[float | None]:
    return [e.z_pos for e in useq.MDASequence(focus_surface=_surface(), **kwargs)]


def test_positions_without_z_follow_the_surface() -> None:
    channels = [{"config": "DAPI", "z_offset": 1}, "FITC"]
    # no z plan: surface Z plus the channel offset
    positions = [(100, 100), (500, 500)]
    zs = _zs(stage_positions=positions, channels=channels)
    assert zs == pytest.approx([2, 1, 6, 5])
    # relative z plan: centered on the surface Z
    z_plan = useq.ZRangeAround(range=2, step=2)
    assert _zs(stage_positions=[(500, 500)], z_plan=z_plan) == pytest.approx([4, 6])
    # a position's own z wins
    assert _zs(stage_positions=[(100, 100, 7)]) == [7]
    # without XY, nothing to follow
    assert _zs(channels=["DAPI"]) == [None]


@pytest.mark.parametrize("axis_order", ["tpgcz", "tpcgz"])
def test_grid_tiles_follow_the_surface(axis_order: str) -> None:
    seq = useq.MDASequence(
        stage_positions=[
            useq.Position(x=100, y=100, sequence=useq.MDASequence(grid_plan=GRID)),
            (500, 500),
        ],
        channels=["DAPI", "FITC"],
        axis_order=axis_order,
        focus_surface=_surface(),
    )
    plain = seq.replace(focus_surface=None)
    # the same events, in the same order, each at the surface Z of its own XY
    events = list(seq)
    assert [(e.index, e.x_pos, e.y_pos) for e in events] == [
        (e.index, e.x_pos, e.y_pos) for e in plain
    ]
    assert {e.x_pos: e.z_pos for e in events} == pytest.approx(
        {95: 0.95, 105: 1.05, 500: 5}
    )


def test_absolute_grid_and_plate() -> None:
    grid = useq.GridFromEdges(
        top=5, bottom=5, left=100, right=120, fov_width=10, fov_height=10
    )
    assert _zs(grid_plan=grid) == pytest.approx([1.05, 1.15])
    plate = useq.WellPlatePlan(
        plate=96, a1_center_xy=(100, 100), selected_wells=([0], [0, 1])
    )
    assert _zs(stage_positions=plate) == pytest.approx([1, 91])


def test_not_with_an_absolute_z_plan() -> None:
    with pytest.raises(ValidationError, match="absolute z_plan"):
        useq.MDASequence(
            z_plan={"top": 1, "bottom": 0, "step": 1}, focus_surface=_surface()
        )


def test_surface_evaluated_once_per_xy() -> None:
    seq = useq.MDASequence(
        stage_positions=[(100, 100)],
        time_plan={"interval": 0, "loops": 3},
        z_plan=useq.ZRangeAround(range=4, step=1),
        focus_surface=_surface(),
    )
    evaluate = FocusSurface.evaluate
    with patch.object(
        FocusSurface, "evaluate", autospec=True, side_effect=evaluate
    ) as mock:
        assert len(list(seq)) == 15
    mock.assert_called_once()
