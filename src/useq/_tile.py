from __future__ import annotations

from typing import Iterator, Literal, Sequence, Union

from useq._base_model import FrozenModel
from useq._position import Position


class TilePlan(FrozenModel):
    tile_name: str  # maybe try a kind of setter method??? at the moment is immutable
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
    tile_name : str
        Name of the tile.
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

    tile_name: str
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
    tile_name : str
        Name of the tile. It will be added as a prefix name for each
        tile positions.
        (e.g. tile_name = 'Grid' -> `Grid_Pos000`, `Grid_Pos001`)
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
        Starting x, y, z position coordinates used to generate the grid.
    snake_order : bool
        Organize the positions for a `snake-type` acquisition. By default, `True`.
    relative_to: Literal["center", "top_left"]:
        Define if the position list will be generated using the
        `start_coords` as a central grid position (`center`) or
        as the first top_left grid position (`top_left`).
    """

    tile_name: str = ""
    rows: int
    cols: int
    overlap_x: float
    overlap_y: float
    pixel_size: float
    camera_roi: tuple[int, int]
    start_coords: tuple[float, float, float | None]
    snake_order: bool = True
    relative_to: Literal["center", "top_left"]

    def tiles(self) -> Sequence[Position]:
        """Generate the position list."""
        prefix = f"{self.tile_name}_" if self.tile_name else ""

        x_pos, y_pos, z_pos = self.start_coords
        cam_width, cam_height = self.camera_roi
        overlap_x = cam_width - (cam_width * self.overlap_x) / 100
        overlap_y = cam_height - (cam_height * self.overlap_y) / 100

        if self.relative_to == "center":
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
        starting_x = x_pos
        for _ in range(self.rows):
            x_pos = starting_x
            for c in range(self.cols):
                if c != 0:
                    x_pos += increment_x
                tile_pos_list.append(Position(
                    name=f"{prefix}Pos{pos_count:03d}", x=x_pos, y=y_pos, z=z_pos
                ))
                if c == self.cols - 1:
                    y_pos -= increment_y

        if self.snake_order:
            snake_ordered_list = []
            insert = False
            for idx, pos in enumerate(tile_pos_list):
                if idx != 0 and idx % self.cols == 0:
                    insert = not insert
                    index = idx
                if insert:
                    snake_ordered_list.insert(index, pos)
                else:
                    snake_ordered_list.append(pos)
            
            return snake_ordered_list

        return tile_pos_list


AnyTilePlan = Union[TileFromCorners, TileRelative]
