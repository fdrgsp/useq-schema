"""A focus surface `z = f(x, y)` through manually focused reference points."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import FiniteFloat, model_validator
from shapely import MultiPoint, delaunay_triangles, get_coordinates

from useq._base_model import UseqModel

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import ArrayLike, NDArray

# Two points closer than this (in XY, µm) would need two heights at one place.
MIN_POINT_DISTANCE = 0.1
# Below this ratio of the XY spread's minor to major axis, the points are a line.
MIN_SPREAD_RATIO = 1e-3
# Barycentric weights above minus this are inside a triangle.
INSIDE_TOLERANCE = 1e-9
# Bound the temporary position-by-triangle arrays when evaluating a dense map.
MAX_BATCH_PAIRS = 262_144


class FocusSurface(UseqModel):
    """A surface `z = f(x, y)` through manually focused reference points.

    When a sequence has a `focus_surface`, the Z of each event at a position without
    its own `z` is the surface Z at the event's XY (e.g. each grid tile's or plate
    image's), plus the event's relative Z plan and channel offsets.

    Attributes
    ----------
    points : tuple[tuple[float, float, float], ...]
        The `(x, y, z)` of at least 3 reference points, in µm, not all on a line.
    method : Literal["linear", "plane"]
        `"linear"` (default) interpolates linearly inside the Delaunay triangulation
        of the points, following local changes of the height. `"plane"` is their
        least-squares plane. Outside the points' convex hull, `"linear"` continues
        from the nearest point of the hull with the slope of that plane.
    """

    points: tuple[tuple[FiniteFloat, FiniteFloat, FiniteFloat], ...]
    method: Literal["linear", "plane"] = "linear"

    @model_validator(mode="after")
    def _validate_geometry(self) -> Self:
        self._compiled  # noqa: B018
        return self

    @cached_property
    def _compiled(self) -> _CompiledSurface:
        return _CompiledSurface(self)

    def evaluate(self, xy: ArrayLike) -> NDArray[np.float64]:
        """Return the surface Z at each `(x, y)`."""
        return self._compiled(np.asarray(xy, dtype=float).reshape(-1, 2))

    def hull(self) -> NDArray[np.float64]:
        """Return the XY of the points' convex hull, counterclockwise."""
        return self._compiled.hull


class _CompiledSurface:
    def __init__(self, surface: FocusSurface) -> None:
        if (n := len(surface.points)) < 3:
            raise ValueError(f"At least 3 reference points are needed ({n} set).")
        xyz = np.array(surface.points, dtype=float)
        xy, self._z = xyz[:, :2], xyz[:, 2]

        dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        if (close := np.argwhere(np.triu(dist < MIN_POINT_DISTANCE, 1))).size:
            i, j = close[0] + 1
            raise ValueError(f"Reference points {i} and {j} are at the same XY.")

        # One common scale for X and Y, so the triangulation is the physical one.
        self._center = xy.mean(axis=0)
        self._scale = np.abs(xy - self._center).max()
        self._unit = unit = (xy - self._center) / self._scale
        spread = np.linalg.svd(unit - unit.mean(axis=0), compute_uv=False)
        if spread[1] < MIN_SPREAD_RATIO * spread[0]:
            raise ValueError(
                "The reference points lie on a line: add one away from it."
            )

        self._method = surface.method
        design = np.column_stack([unit, np.ones(n)])
        self._plane = np.linalg.lstsq(design, self._z, rcond=None)[0]
        self._triangles, hull = _delaunay(unit)
        self.hull: NDArray[np.float64] = xy[hull]
        self._hull_edges = np.column_stack([hull, np.roll(hull, -1)])
        if self._method == "plane":
            return
        # barycentric transform of each triangle: (T, 2, 2) inverse and (T, 2) origin
        a, b, c = (unit[self._triangles[:, i]] for i in range(3))
        self._inverse = np.linalg.inv(np.stack([b - a, c - a], axis=-1))
        self._origin = a

    def __call__(self, xy: NDArray[np.float64]) -> NDArray[np.float64]:
        unit = (xy - self._center) / self._scale
        if self._method == "plane":
            return self._plane_z(unit)
        z = np.empty(len(unit))
        batch = max(1, MAX_BATCH_PAIRS // len(self._triangles))
        for start in range(0, len(unit), batch):
            z[start : start + batch] = self._linear_z(unit[start : start + batch])
        return z

    def _linear_z(self, unit: NDArray[np.float64]) -> NDArray[np.float64]:
        # barycentric weights of each position (rows) in each triangle (columns)
        uv = np.einsum("tij,mtj->mti", self._inverse, unit[:, None] - self._origin)
        weights = np.concatenate([1 - uv.sum(axis=-1, keepdims=True), uv], axis=-1)
        margins = weights.min(axis=-1)
        # Prefer the most interior triangle when rounding makes several eligible.
        tri = margins.argmax(axis=1)
        inside = margins[np.arange(len(unit)), tri] >= -INSIDE_TOLERANCE
        z = np.empty(len(unit))
        rows = np.flatnonzero(inside)
        vertex_z = self._z[self._triangles[tri[rows]]]
        z[rows] = (weights[rows, tri[rows]] * vertex_z).sum(axis=1)
        z[~inside] = self._extrapolate(unit[~inside])
        return z

    def _plane_z(self, unit: NDArray[np.float64]) -> NDArray[np.float64]:
        z: NDArray[np.float64] = unit @ self._plane[:2] + self._plane[2]
        return z

    def _extrapolate(self, unit: NDArray[np.float64]) -> NDArray[np.float64]:
        """Continue from the nearest hull point with the slope of the plane fit."""
        a, b = (self._unit[i] for i in self._hull_edges.T)
        za, zb = (self._z[i] for i in self._hull_edges.T)
        # projection of each position (rows) on each hull edge (columns)
        ab = b - a
        t = np.einsum("mek,ek->me", unit[:, None] - a, ab) / (ab**2).sum(axis=1)
        t = t.clip(0, 1)
        nearest = a + t[..., None] * ab
        edge = np.linalg.norm(unit[:, None] - nearest, axis=-1).argmin(axis=1)
        rows = np.arange(len(unit))
        q, tq = nearest[rows, edge], t[rows, edge]
        z_q = za[edge] + tq * (zb[edge] - za[edge])
        z: NDArray[np.float64] = z_q + (unit - q) @ self._plane[:2]
        return z


def _delaunay(pts: NDArray[np.float64]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Delaunay triangles and CCW boundary, including collinear boundary points."""
    # Sort before GEOS triangulation so cocircular ties are independent of input order.
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    coords = get_coordinates(delaunay_triangles(MultiPoint(pts[order])))
    indices = {tuple(point): i for i, point in enumerate(pts)}
    triangles = np.array([indices[tuple(p)] for p in coords], dtype=np.intp)
    triangles = triangles.reshape(-1, 4)[:, :3].copy()
    a, b, c = (pts[triangles[:, i]] for i in range(3))
    clockwise = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) < (b[:, 1] - a[:, 1]) * (
        c[:, 0] - a[:, 0]
    )
    triangles[clockwise] = triangles[clockwise, ::-1]
    edges = {
        (int(a), int(b))
        for tri in triangles
        for a, b in zip(tri, np.roll(tri, -1), strict=True)
    }
    boundary = {a: b for a, b in edges if (b, a) not in edges}
    hull = [min(boundary, key=lambda i: tuple(pts[i]))]
    while (vertex := boundary[hull[-1]]) != hull[0]:
        hull.append(vertex)
    return triangles, np.array(hull, dtype=np.intp)
