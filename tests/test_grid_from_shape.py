from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from pydantic import TypeAdapter, ValidationError

import useq
from useq import GridFromShape, OrderMode, Shape
from useq._grid import MAX_SHAPE_GRID_CANDIDATES

# Geometry tolerance (µm) used by the independent coverage checks below.
TOL = 1e-6


def _footprints(plan: GridFromShape) -> list[shapely.Polygon]:
    fw, fh = plan.fov_width, plan.fov_height
    assert fw and fh
    return [
        shapely.box(p.x - fw / 2, p.y - fh / 2, p.x + fw / 2, p.y + fh / 2)
        for p in plan
    ]


def _boundary_samples(plan: GridFromShape, n: int = 2000) -> np.ndarray:
    """Points on the *true* (smooth) boundary of the area, independent of the plan."""
    w2, h2 = plan.width / 2, plan.height / 2
    if plan.shape == Shape.ELLIPSE:
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack((w2 * np.cos(ang), h2 * np.sin(ang)))
    else:
        s = np.linspace(-1, 1, n // 4)
        pts = np.concatenate(
            [
                np.column_stack((s * w2, np.full_like(s, h2))),
                np.column_stack((s * w2, np.full_like(s, -h2))),
                np.column_stack((np.full_like(s, w2), s * h2)),
                np.column_stack((np.full_like(s, -w2), s * h2)),
            ]
        )
    t = math.radians(plan.rotation)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return pts @ rot.T  # type: ignore[no-any-return]


def _interior_samples(plan: GridFromShape, n: int = 3000) -> np.ndarray:
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, size=(n, 2))
    if plan.shape == Shape.ELLIPSE:
        u = u[(u**2).sum(axis=1) <= 1]
    pts = u * [plan.width / 2, plan.height / 2]
    t = math.radians(plan.rotation)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return pts @ rot.T  # type: ignore[no-any-return]


def _assert_covers(plan: GridFromShape) -> None:
    union = shapely.union_all(_footprints(plan)).buffer(TOL)
    for pts in (_boundary_samples(plan), _interior_samples(plan)):
        covered = shapely.covers(union, shapely.points(pts))
        assert covered.all(), f"{(~covered).sum()} sample points not covered"


COVER_CASES = [
    # rectangles: exact multiple, non-multiple, overlap per axis, non-square FOV
    {"shape": "rectangle", "width": 1000, "height": 1000, "fov": (100, 100)},
    {"shape": "rectangle", "width": 1050, "height": 730, "fov": (100, 100)},
    {
        "shape": "rectangle",
        "width": 1000,
        "height": 600,
        "fov": (120, 80),
        "ov": (10, 25),
    },
    {"shape": "rectangle", "width": 50, "height": 40, "fov": (100, 100)},
    # ellipses/circles
    {"shape": "ellipse", "width": 1000, "height": 1000, "fov": (100, 100)},
    {"shape": "ellipse", "width": 6400, "height": 6400, "fov": (512, 512), "ov": 10},
    {"shape": "ellipse", "width": 1500, "height": 700, "fov": (130, 90), "ov": (5, 30)},
    {"shape": "ellipse", "width": 60, "height": 60, "fov": (100, 100)},
    # rotated areas (tiles stay axis-aligned)
    {"shape": "rectangle", "width": 1000, "height": 400, "fov": (100, 80), "rot": 17},
    {"shape": "rectangle", "width": 1000, "height": 400, "fov": (100, 80), "rot": 90},
    {"shape": "ellipse", "width": 1200, "height": 500, "fov": (100, 80), "rot": -33},
    {"shape": "ellipse", "width": 1200, "height": 500, "fov": (90, 90), "rot": 180},
]


def _make(case: dict) -> GridFromShape:
    fw, fh = case["fov"]
    return GridFromShape(
        shape=case["shape"],
        width=case["width"],
        height=case["height"],
        overlap=case.get("ov", 0),
        rotation=case.get("rot", 0),
        fov_width=fw,
        fov_height=fh,
    )


@pytest.mark.parametrize("case", COVER_CASES)
def test_grid_from_shape_covers_area(case: dict) -> None:
    plan = _make(case)
    _assert_covers(plan)
    assert plan.num_positions() == len(list(plan))


@pytest.mark.parametrize("case", COVER_CASES)
def test_grid_from_shape_keeps_only_intersecting_tiles(case: dict) -> None:
    """Every retained tile touches the area; compare with polygonal geometry."""
    plan = _make(case)
    # A circumscribing polygon approximation of an ellipse can only *add* contact,
    # so every retained tile must intersect it (buffered by a small tolerance).
    n = 4096
    verts = plan.boundary_vertices(n)
    area = shapely.Polygon(verts)
    if plan.shape == Shape.ELLIPSE:
        area = shapely.affinity.scale(
            area, 1 / math.cos(math.pi / n), 1 / math.cos(math.pi / n), origin=(0, 0)
        )
    area = area.buffer(TOL)
    assert all(area.intersects(f) for f in _footprints(plan))


def test_rectangle_counts() -> None:
    # exact multiples: no extra row/column from floating point noise
    plan = GridFromShape(
        shape="rectangle", width=1000, height=500, fov_width=100, fov_height=100
    )
    assert plan.num_positions() == 10 * 5
    plan = GridFromShape(
        shape="rectangle", width=0.3 * 10, height=0.1 * 3, fov_width=0.3, fov_height=0.1
    )
    assert plan.num_positions() == 10 * 3
    # non-multiple
    plan = GridFromShape(
        shape="rectangle", width=1001, height=500, fov_width=100, fov_height=100
    )
    assert plan.num_positions() == 11 * 5
    # overlap per axis: step x = 50 -> 1 + ceil(900 / 50) = 19; step y = 100
    plan = GridFromShape(
        shape="rectangle",
        width=1000,
        height=500,
        overlap=(50, 0),
        fov_width=100,
        fov_height=100,
    )
    assert plan.num_positions() == 19 * 5
    # smaller than one FOV -> one centered tile
    plan = GridFromShape(
        shape="ellipse", width=10, height=20, fov_width=100, fov_height=100
    )
    assert [(p.x, p.y) for p in plan] == [(0, 0)]


def test_lattice_is_centered_and_spaced_by_overlap() -> None:
    plan = GridFromShape(
        shape="rectangle",
        width=950,
        height=420,
        overlap=(10, 20),
        fov_width=100,
        fov_height=50,
    )
    pos = list(plan)
    xs = sorted({round(p.x, 9) for p in pos})
    ys = sorted({round(p.y, 9) for p in pos})
    assert np.allclose(np.diff(xs), 90)
    assert np.allclose(np.diff(ys), 40)
    assert math.isclose(xs[0], -xs[-1]) and math.isclose(ys[0], -ys[-1])
    # top-left first in row-wise orders; y decreases with increasing row
    first = pos[0]
    assert first.grid_row == 0 and first.grid_col == 0
    assert first.x == xs[0] and first.y == ys[-1]


def test_ellipse_keeps_tiles_with_center_outside() -> None:
    plan = GridFromShape(
        shape="ellipse", width=1000, height=1000, fov_width=100, fov_height=100
    )
    outside = [p for p in plan if (p.x**2 + p.y**2) > 500**2]
    assert outside, "expected tiles whose center is outside but footprint intersects"
    # corner tiles of the bounding lattice do not touch the circle and are removed
    assert plan.num_positions() < 10 * 10


def test_snake_order_and_metadata_with_omitted_cells() -> None:
    plan = GridFromShape(
        shape="ellipse", width=1000, height=1000, fov_width=150, fov_height=150
    )
    pos = list(plan)
    rc = [(p.grid_row or 0, p.grid_col or 0) for p in pos]
    assert len(set(rc)) == len(rc)
    # rows visited in order; columns alternate direction (row-wise snake)
    rows = [r for r, _ in rc]
    assert rows == sorted(rows)
    for r in set(rows):
        cols = [c for rr, c in rc if rr == r]
        assert cols == (sorted(cols) if r % 2 == 0 else sorted(cols, reverse=True))
    assert [p.name for p in pos] == [f"{i:04d}" for i in range(len(pos))]
    # same tiles for all orders, only the order changes
    for mode in OrderMode:
        other = list(plan.iter_grid_positions(order=mode))
        assert {(p.grid_row, p.grid_col) for p in other} == set(rc)


def test_equality_is_not_affected_by_evaluation() -> None:
    kwargs = {
        "shape": "ellipse",
        "width": 700,
        "height": 500,
        "fov_width": 60,
        "fov_height": 60,
    }
    a, b = GridFromShape(**kwargs), GridFromShape(**kwargs)
    a.num_positions()
    assert a == b
    assert a.model_copy(update={"fov_width": 30}) != b


def test_repeated_iteration_is_stable() -> None:
    plan = GridFromShape(
        shape="ellipse", width=2000, height=1300, fov_width=110, fov_height=90
    )
    a = [(p.x, p.y, p.grid_row, p.grid_col) for p in plan]
    b = [(p.x, p.y, p.grid_row, p.grid_col) for p in plan]
    assert a == b
    assert plan.num_positions() == len(a)
    # changing the (mutable) FOV recomputes the layout
    plan.fov_width = 220
    assert plan.num_positions() != len(a)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"width": -1},
        {"height": float("inf")},
        {"width": float("nan")},
        {"overlap": 100},
        {"overlap": (10, -1)},
        {"shape": "triangle"},
    ],
)
def test_invalid_fields(kwargs: dict) -> None:
    base = {"shape": "rectangle", "width": 100, "height": 100}
    with pytest.raises(ValidationError):
        GridFromShape(**{**base, **kwargs})


def test_shape_is_required() -> None:
    with pytest.raises(ValidationError):
        GridFromShape(width=100, height=100)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "fov", [(None, 10), (10, None), (0, 10), (10, -1), (float("nan"), 10)]
)
def test_invalid_fov(fov: tuple) -> None:
    plan = GridFromShape(
        shape="rectangle", width=100, height=100, fov_width=fov[0], fov_height=fov[1]
    )
    with pytest.raises(ValueError, match="ield of view"):
        plan.num_positions()
    with pytest.raises(ValueError, match="ield of view"):
        list(plan)


def test_excessive_candidate_count_raises() -> None:
    side = math.sqrt(MAX_SHAPE_GRID_CANDIDATES) * 2
    plan = GridFromShape(
        shape="rectangle", width=side, height=side, fov_width=1, fov_height=1
    )
    with pytest.raises(ValueError, match="maximum"):
        plan.num_positions()


# ------------------------ serialization / unions ------------------------


def test_round_trip_and_union_identification() -> None:
    plan = GridFromShape(
        shape="ellipse",
        width=1000,
        height=800,
        overlap=(5, 10),
        mode="column_wise_snake",
        fov_width=100,
        fov_height=80,
    )
    for dump in (
        plan.model_dump(mode="json"),
        plan.model_dump(mode="json", exclude_defaults=True),
        plan.model_dump(mode="json", exclude_unset=True),
    ):
        for union in (useq.RelativeMultiPointPlan, useq.MultiPointPlan):
            ta = TypeAdapter(union)
            loaded = ta.validate_python(dump)
            # the default smart union must also pick the right class
            assert isinstance(loaded, GridFromShape), union
        loaded = GridFromShape.model_validate_json(plan.model_dump_json())
        assert loaded == plan
        assert list(loaded) == list(plan)

    # GridWidthHeight dicts still load as GridWidthHeight
    gwh = useq.WellPlatePlan(
        plate=96, a1_center_xy=(0, 0), well_points_plan={"width": 100, "height": 100}
    )
    assert type(gwh.well_points_plan) is useq.GridWidthHeight


def test_json_schema() -> None:
    schema = useq.MDASequence.model_json_schema()
    assert "GridFromShape" in schema["$defs"]
    assert "GridFromShape" in useq.WellPlatePlan.model_json_schema()["$defs"]


def test_in_mda_sequence() -> None:
    plan = GridFromShape(
        shape="ellipse", width=500, height=500, fov_width=100, fov_height=100
    )
    seq = useq.MDASequence(stage_positions=[(1000, 2000, 0)], grid_plan=plan)
    assert isinstance(seq.grid_plan, GridFromShape)
    events = list(seq)
    assert len(events) == plan.num_positions()
    offsets = [(p.x, p.y) for p in plan]
    assert [(e.x_pos, e.y_pos) for e in events] == [
        (1000 + x, 2000 + y) for x, y in offsets
    ]

    reloaded = useq.MDASequence.model_validate_json(seq.model_dump_json())
    assert isinstance(reloaded.grid_plan, GridFromShape)
    assert len(list(reloaded)) == len(events)

    import yaml

    yaml_seq = useq.MDASequence.model_validate(yaml.safe_load(seq.yaml()))
    assert isinstance(yaml_seq.grid_plan, GridFromShape)


def test_well_plate_plan() -> None:
    plan = GridFromShape(
        shape="ellipse", width=2000, height=2000, fov_width=500, fov_height=500
    )
    pp = useq.WellPlatePlan(
        plate=96,
        a1_center_xy=(100, 200),
        selected_wells=([0, 1], [0, 2]),
        well_points_plan=plan,
    )
    n = plan.num_positions()
    assert pp.num_points_per_well == n
    assert len(pp) == 2 * n
    positions = list(pp)
    wells = pp.selected_well_positions
    for i, well in enumerate(wells):
        chunk = positions[i * n : (i + 1) * n]
        for p, off in zip(chunk, plan, strict=True):
            assert p.plate_row == well.plate_row and p.plate_col == well.plate_col
            assert p.x is not None and p.y is not None
            assert well.x is not None and well.y is not None
            assert math.isclose(p.x, well.x + off.x)
            assert math.isclose(p.y, well.y + off.y)
            assert p.name == f"{well.name}_{off.name}"

    reloaded = useq.WellPlatePlan.model_validate_json(pp.model_dump_json())
    assert isinstance(reloaded.well_points_plan, GridFromShape)
    assert reloaded.well_points_plan == plan
    assert [(p.x, p.y) for p in reloaded] == [(p.x, p.y) for p in pp]


def test_well_plate_plan_rotation_convention() -> None:
    """The plate rotation is applied to the area, matching the plate's well layout."""
    plan = GridFromShape(
        shape="rectangle", width=2000, height=500, fov_width=100, fov_height=100
    )
    pp = useq.WellPlatePlan(
        plate=96, a1_center_xy=(0, 0), rotation=12, well_points_plan=plan
    )
    wpp = pp.well_points_plan
    assert isinstance(wpp, GridFromShape)
    assert wpp.rotation == 12
    # the direction from A1 to A2 (the well's +x axis) must match the area's rotation
    a1, a2 = pp.all_well_positions[0], pp.all_well_positions[1]
    assert a1.x is not None and a1.y is not None
    assert a2.x is not None and a2.y is not None
    angle = math.degrees(math.atan2(a2.y - a1.y, a2.x - a1.x))
    assert math.isclose(angle, wpp.rotation, abs_tol=1e-9)
    _assert_covers(wpp)

    # an explicitly set rotation is kept
    explicit = plan.replace(rotation=0)
    pp = useq.WellPlatePlan(
        plate=96, a1_center_xy=(0, 0), rotation=12, well_points_plan=explicit
    )
    assert pp.well_points_plan.rotation == 0  # type: ignore[union-attr]


# ------------------------ placement="inside" ------------------------


def _area_polygon(plan: GridFromShape) -> shapely.Polygon:
    """A polygon containing the true area (circumscribed for ellipses)."""
    n = 4096
    area = shapely.Polygon(plan.boundary_vertices(n))
    if plan.shape == Shape.ELLIPSE:
        k = 1 / math.cos(math.pi / n)
        area = shapely.affinity.scale(area, k, k, origin=(0, 0))
    return area.buffer(TOL)


@pytest.mark.parametrize("case", COVER_CASES)
def test_inside_placement_keeps_tiles_inside(case: dict) -> None:
    plan = _make(case).replace(placement="inside")
    area = _area_polygon(plan)
    assert all(area.covers(f) for f in _footprints(plan))
    assert plan.num_positions() == len(list(plan))
    # never more tiles than when covering the area
    assert plan.num_positions() <= _make(case).num_positions()


def test_inside_placement_counts() -> None:
    kw = {"shape": "rectangle", "fov_width": 300, "fov_height": 300}
    # 1 FOV + 2 steps fit in 1000 µm (a 4th would stick out)
    plan = GridFromShape(width=1000, height=600, placement="inside", **kw)
    assert plan.num_positions() == 3 * 2
    # exact fit, no floating point loss of a row/column
    plan = GridFromShape(width=900, height=300, placement="inside", **kw)
    assert plan.num_positions() == 3 * 1
    # centered: symmetric margins
    xs = sorted({p.x for p in plan})
    assert xs == [-300, 0, 300]
    # smaller than one FOV: no positions at all
    plan = GridFromShape(width=200, height=1000, placement="inside", **kw)
    assert plan.num_positions() == 0
    assert list(plan) == []

    # a circle: every corner of every tile is inside the circle
    circle = GridFromShape(
        shape="ellipse",
        width=6400,
        height=6400,
        placement="inside",
        fov_width=500,
        fov_height=400,
        overlap=10,
    )
    assert circle.num_positions() > 0
    for p in circle:
        for dx in (-250, 250):
            for dy in (-200, 200):
                assert math.hypot(p.x + dx, p.y + dy) <= 3200 * (1 + 1e-9)


def test_inside_placement_round_trip() -> None:
    plan = GridFromShape(
        shape="ellipse",
        width=3000,
        height=2000,
        placement="inside",
        fov_width=200,
        fov_height=200,
    )
    loaded = TypeAdapter(useq.RelativeMultiPointPlan).validate_json(
        plan.model_dump_json(exclude_unset=True)
    )
    assert loaded == plan
    with pytest.raises(ValidationError):
        GridFromShape(shape="ellipse", width=10, height=10, placement="center")  # type: ignore[arg-type]
