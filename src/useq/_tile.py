from __future__ import annotations

from typing import Iterator, Literal, Sequence, Union

from ._base_model import FrozenModel
from ._position import Position


class TilePlan(FrozenModel):
    overlap_x: float
    overlap_y: float
    pixel_size: float
    camera_roi: tuple[int, int]

    def __iter__(self) -> Iterator[Position]:
        yield from self.tiles()

    def tiles(self) -> Sequence[Position]:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.tiles())


class TileFromCorners(TilePlan):
    """Define a list of positions for a tile acquisition.

    ...depending on a grid determined by a top_left and a
    borrom_right positions.

    Attributes
    ----------
    top_left : tuple[float, float]
        Grid top_left position.
    bottom_right : tuple[float, float]
        Grid bottom_right position.
    overlap_x: float
        Percentage of overlap between tiles along the x axis.
    overlap_y: float
        Percentage of overlap between tiles along the y axis.
    pixel_size : float
        Pixel size of the imaging system.
    camera_roi : tuple[int, int]
        Camera ROI dimension.
    """

    top_left: tuple[float, float]
    bottom_right: tuple[float, float]
    overlap_x: float
    overlap_y: float
    pixel_size: float
    camera_roi: tuple[int, int]


class TileRelative(TilePlan):
    """Define a list of positions for a tile acquisition.

   ...relative to a specified position (`start_coords`).

    Attributes
    ----------
    rows: int
        Number of rows.
    cols: int
        Number of columns.
    overlap_x: float
        Percentage of overlap between tiles along the x axis.
    overlap_y: float
        Percentage of overlap between tiles along the y axis.
    pixel_size : float
        Pixel size of the imaging system.
    camera_roi : tuple[int, int]
        Camera ROI dimension.
    start_coords: tuple[float, float, float | None]
        Starting position coordinates used to generate the grid.
    relative_to_coords: Literal["center", "top_left"]:
        Define if the position list will be generated using the
        `start_coords` as a central grid position (`center`) or
        as the first top_left grid position (`top_left`).
    """

    rows: int
    cols: int
    overlap_x: float
    overlap_y: float
    pixel_size: float
    camera_roi: tuple[int, int]
    start_coords: tuple[float, float, float | None]
    relative_to_coords: Literal["center", "top_left"]

    def tiles(self, name_prefix: str = "") -> Sequence[Position]:

        x_pos, y_pos, z_pos = self.start_coords
        cam_width, cam_height = self.camera_roi
        overlap_x = cam_width - (cam_width * self.overlap_x) / 100
        overlap_y = cam_height - (cam_height * self.overlap_y) / 100

        if self.relative_to_coords == "center":
            # move to top left corner
            move_x = (cam_width / 2) * (self.cols - 1) - overlap_x
            move_y = (cam_height / 2) * (self.rows - 1) - overlap_y
            x_pos -= self.pixel_size * (move_x + cam_width)
            y_pos += self.pixel_size * (move_y + cam_height)

        # calculate position increments depending on pixle size
        increment_x = (
            overlap_x * self.pixel_size 
            if self.overlap_x > 0
            else cam_width * self.pixel_size
        )
        increment_y = (
            overlap_y * self.pixel_size 
            if self.overlap_y > 0
            else cam_height * self.pixel_size
        )

        tile_pos_list: list[Position] = []
        pos_count = 0
        for r in range(self.rows):
            if r % 2:  # for odd rows
                col = self.cols - 1
                for c in range(self.cols):
                    if c == 0:
                        y_pos -= increment_y
                    tile_pos_list.append(Position(
                        name=f"{name_prefix}Pos{pos_count:03d}", x=x_pos, y=y_pos, z=z_pos
                    ))
                    pos_count += 1
                    if col > 0:
                        col -= 1
                        x_pos -= increment_x
            else:  # for even rows
                for c in range(self.cols):
                    if r > 0 and c == 0:
                        y_pos -= increment_y
                    tile_pos_list.append(Position(
                        name=f"{name_prefix}Pos{pos_count:03d}", x=x_pos, y=y_pos, z=z_pos
                    ))
                    pos_count += 1
                    if c < self.cols - 1:
                        x_pos += increment_x

        return tile_pos_list


AnyTilePlan = Union[TileFromCorners, TileRelative]












