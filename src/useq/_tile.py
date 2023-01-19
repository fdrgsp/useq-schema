from __future__ import annotations

import itertools
import math
from typing import Iterator, Literal, Union

from useq._base_model import FrozenModel
from useq._position import Position


class Coordinate(FrozenModel):
    """Defines a position in 2D space.

    Attributes
    ----------
    x : float
        X position in microns.
    y : float
        Y position in microns.
    """

    x: float
    y: float


def _calculate_grid(
    top_left: Coordinate,
    increment_x: float,
    increment_y: float,
    rows: int,
    cols: int,
    snake_order: bool,
) -> Iterator[tuple[float, float]]:
    # starting position must be the top left
    for r, c in itertools.product(range(rows), range(cols)):
        y_pos = top_left.y - (r * increment_y)
        if snake_order and r % 2 == 1:
            x_pos = top_left.x + ((cols - c - 1) * increment_x)
        else:
            x_pos = top_left.x + (c * increment_x)
        yield (x_pos, y_pos)



class _TilePlan(FrozenModel):
    """Base class for all tile plans.

    Attributes
    ----------
    overlap : float | tuple[float, float]
        Overlap between tiles in percent. If a single value is provided, it is
        used for both x and y. If a tuple is provided, the first value is used
        for x and the second for y.
    snake_order : bool
        If `True`, tiles are arranged in a snake order (i.e. back and forth).
        If `False`, tiles are arranged in a row-wise order.
    """

    overlap: float | tuple[float, float] = 0.0
    snake_order: bool = True

    def iter_tiles(self, fov_width: float, fov_height: float) -> Iterator[Position]:
        """Iterate over all tiles, given a field of view size."""
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(list(self.iter_tiles(1, 1)))

    @property
    def is_relative(self) -> bool:
        """Return True if the tile plan is relative to some (yet unknown) position."""
        return True


class TileFromCorners(_TilePlan):
    """Define tile positions from two corners.

    Attributes
    ----------
    corner1 : Coordinate
        First bounding coordinate (e.g. "top left").
    corner2 : Coordinate
        Second bounding coordinate (e.g. "bottom right").
    """

    corner1: Coordinate
    corner2: Coordinate

    # TODO: add also top_right and bottom_left
    def iter_tiles(self, fov_width: float, fov_height: float) -> Iterator[Position]:
        """Yield absolute tile positions to visit.

        fov_width and fov_height should be in physical units (not pixels).
        """
        over = (self.overlap,) * 2 if isinstance(self.overlap, float) else self.overlap
        overlap_x, overlap_y = over

        cam_width_minus_overlap = fov_width - (fov_width * overlap_x) / 100
        cam_height_minus_overlap = fov_height - (fov_height * overlap_y) / 100

        total_width = abs(self.corner1.x + self.corner2.x)
        total_height = abs(self.corner1.y + self.corner2.y)

        rows = math.ceil(total_width / cam_width_minus_overlap)
        cols = math.ceil(total_height / cam_height_minus_overlap)

        increment_x = cam_width_minus_overlap if overlap_x > 0 else fov_width
        increment_y = cam_height_minus_overlap if overlap_y > 0 else fov_height

        # TODO
        top_left = self.corner1

        yield from _calculate_grid(
            top_left, increment_x, increment_y, rows, cols, self.snake_order
        )

    @property
    def is_relative(self) -> bool:
        """Return True if the tile plan is relative to some (yet unknown) position."""
        return False


class TileRelative(_TilePlan):
    """Define a list of positions for a tile acquisition.

    ...relative to a specified position (`start_coords`).

    Attributes
    ----------
    rows: int
        Number of rows.
    cols: int
        Number of columns.
    relative_to: Literal["center", "top_left"]:
        Define if the position list will be generated using the
        `start_coords` as a central grid position (`center`) or
        as the first top_left grid position (`top_left`).
    """

    rows: int
    cols: int
    relative_to: Literal["center", "top_left"] = "center"

    def iter_tiles(self, fov_width: float, fov_height: float) -> Iterator[Position]:
        """Yield deltas relative to some position.

        fov_width and fov_height should be in physical units (not pixels).
        """
        over = (self.overlap,) * 2 if isinstance(self.overlap, float) else self.overlap
        overlap_x, overlap_y = over

        x_pos, y_pos = (0.0, 0.0)
        cam_width_minus_overlap = fov_width - (fov_width * overlap_x) / 100
        cam_height_minus_overlap = fov_height - (fov_height * overlap_y) / 100

        if self.relative_to == "center":
            # move to top left corner
            move_x = (fov_width / 2) * (self.cols - 1) - cam_width_minus_overlap
            move_y = (fov_height / 2) * (self.rows - 1) - cam_height_minus_overlap
            x_pos -= move_x + fov_width
            y_pos += move_y + fov_height

        increment_x = cam_width_minus_overlap if overlap_x > 0 else fov_width
        increment_y = cam_height_minus_overlap if overlap_y > 0 else fov_height

        yield from _calculate_grid(
            Coordinate(x=x_pos, y=y_pos),
            increment_x,
            increment_y,
            self.rows,
            self.cols,
            self.snake_order,
        )


class NoTile(_TilePlan):
    def iter_tiles(self, fov_width: float, fov_height: float) -> Iterator[Position]:
        return iter([])


AnyTilePlan = Union[TileFromCorners, TileRelative, NoTile]
