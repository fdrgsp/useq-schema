"""A focus surface `z = f(x, y)` through manually focused reference points."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import FiniteFloat, model_validator

from useq._base_model import UseqModel

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import ArrayLike, NDArray

# Two points closer than this (in XY, µm) would need two heights at one place.
MIN_POINT_DISTANCE = 0.1
# Below this ratio of the XY spread's minor to major axis, the points are a line.
MIN_SPREAD_RATIO = 1e-3
# Tolerance (in normalized units) of the geometric tests.
EPS = 1e-12
# Barycentric weights above minus this are inside a triangle.
INSIDE_TOLERANCE = 1e-9


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
        # barycentric transform of each triangle: (T, 2, 2) inverse and (T, 2) origin
        a, b, c = (unit[self._triangles[:, i]] for i in range(3))
        self._inverse = np.linalg.inv(np.stack([b - a, c - a], axis=-1))
        self._origin = a

    def __call__(self, xy: NDArray[np.float64]) -> NDArray[np.float64]:
        unit = (xy - self._center) / self._scale
        if self._method == "plane":
            return self._plane_z(unit)
        # barycentric weights of each position (rows) in each triangle (columns)
        uv = np.einsum("tij,mtj->mti", self._inverse, unit[:, None] - self._origin)
        weights = np.concatenate([1 - uv.sum(axis=-1, keepdims=True), uv], axis=-1)
        contained = (weights >= -INSIDE_TOLERANCE).all(axis=-1)
        inside = contained.any(axis=1)
        z = np.empty(len(unit))
        rows = np.flatnonzero(inside)
        tri = contained[rows].argmax(axis=1)
        vertex_z = self._z[self._triangles[tri]]
        z[rows] = (weights[rows, tri] * vertex_z).sum(axis=1)
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


def _orient(a: NDArray, b: NDArray, c: NDArray) -> float:
    """Twice the signed area of `abc`: positive if counterclockwise."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _in_circle(a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> float:
    """Positive if `d` is inside the circumcircle of the counterclockwise `abc`."""
    rows = np.array([a - d, b - d, c - d])
    return float(np.linalg.det(np.column_stack([rows, (rows**2).sum(axis=1)])))


def _delaunay(pts: NDArray[np.float64]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return the Delaunay triangles (T, 3) and the convex hull of 2D `pts`.

    A sweep over the points sorted by (x, y) triangulates their convex hull, then
    edges are flipped until every triangle has no point inside its circumcircle
    (Lawson). Points on a circumcircle (e.g. the corners of a rectangle) keep the
    sweep's split, so the result does not depend on the order of the points. The
    points must not all lie on a line. The hull is counterclockwise.
    """
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]
    # the first point off the line through the first two starts the sweep
    k = next(i for i in range(2, len(p)) if abs(_orient(p[0], p[1], p[i])) > EPS)
    triangles: list[list[int]] = []
    for i in range(k - 1):
        tri = [i, i + 1, k]
        triangles.append(tri if _orient(*p[tri]) > 0 else [i + 1, i, k])
    # counterclockwise hull: the line of the first k points, then the point k
    hull = [*range(k), k]
    if _orient(p[0], p[k - 1], p[k]) < 0:
        hull = [*range(k - 1, -1, -1), k]
    for i in range(k + 1, len(p)):
        # the new point is right of the hull: connect it to every edge it sees
        n = len(hull)
        visible = [
            j
            for j in range(n)
            if _orient(p[hull[j]], p[hull[(j + 1) % n]], p[i]) < -EPS
        ]
        triangles += [[hull[(j + 1) % n], hull[j], i] for j in visible]
        # the visible edges are consecutive: replace their inner vertices with i
        first = next(j for j in visible if (j - 1) % n not in visible)
        last = next(j for j in visible if (j + 1) % n not in visible)
        start, stop = (first + 1) % n, (last + 1) % n
        if start <= stop:
            hull = [*hull[:start], i, *hull[stop:]]
        else:  # the visible chain wraps around the end of the list
            hull = [*hull[stop:start], i]
    triangles = _flip_to_delaunay(p, triangles)
    return order[np.array(triangles)], order[np.array(hull)]


def _flip_to_delaunay(p: NDArray, triangles: list[list[int]]) -> list[list[int]]:
    """Flip edges of the counterclockwise `triangles` until they are Delaunay."""
    # each directed edge (a, b) belongs to the triangle on its left
    owner: dict[tuple[int, int], int] = {}
    for t, (a, b, c) in enumerate(triangles):
        owner[a, b], owner[b, c], owner[c, a] = t, t, t
    stack = list(owner)
    for _ in range(100 * len(triangles) ** 2):
        if not stack:
            return triangles
        a, b = stack.pop()
        if (a, b) not in owner or (b, a) not in owner:
            continue  # flipped away, or on the hull
        t, u = owner[a, b], owner[b, a]
        c = next(v for v in triangles[t] if v not in (a, b))
        d = next(v for v in triangles[u] if v not in (a, b))
        if _in_circle(p[a], p[b], p[c], p[d]) <= EPS:
            continue
        # replace (a, b, c) and (b, a, d) with (c, a, d) and (d, b, c)
        triangles[t], triangles[u] = [c, a, d], [d, b, c]
        del owner[a, b], owner[b, a]
        owner[c, a] = owner[a, d] = owner[d, c] = t
        owner[d, b] = owner[b, c] = owner[c, d] = u
        stack += [(a, d), (d, b), (b, c), (c, a)]
    raise RuntimeError("The triangulation did not converge.")  # pragma: no cover
