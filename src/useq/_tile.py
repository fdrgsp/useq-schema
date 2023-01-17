from __future__ import annotations

from typing import Iterator, Literal, Sequence, Union

from useq._base_model import FrozenModel
from useq._position import Position


def _calculate_grid(
    starting_position: tuple[float, float, float |  None],
    increment_x: float,
    increment_y: float,
    rows: int,
    cols: int,
    snake_order: bool,
    prefix: str
) -> list[Position]:

    x_pos, y_pos, z_pos = starting_position
    tile_pos_list: list[Position] = []
    pos_count = 0
    starting_x = x_pos
    for _ in range(rows):
        x_pos = starting_x
        for c in range(cols):
            if c != 0:
                x_pos += increment_x
            tile_pos_list.append(Position(
                name=f"{prefix}Pos{pos_count:03d}", x=x_pos, y=y_pos, z=z_pos
            ))
            pos_count += 1
            if c == cols - 1:
                y_pos -= increment_y

    if snake_order:
        snake_ordered_list = []
        insert = False
        for idx, pos in enumerate(tile_pos_list):
            if idx != 0 and idx % cols == 0:
                insert = not insert
                index = idx
            if insert:
                snake_ordered_list.insert(index, pos)
            else:
                snake_ordered_list.append(pos)

        # rename positions
        for idx, pos in enumerate(snake_ordered_list):
            snake_ordered_list[idx] = Position(
                    name=f"{prefix}Pos{idx:03d}", x=pos.x, y=pos.y, z=pos.z
                )

        return snake_ordered_list

    return tile_pos_list


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
        Name of the tile. It will be added as a prefix in front of each
        tile position name. By default "".
        (e.g. tile_name = 'Grid' -> `Grid_Pos000`, `Grid_Pos001`)
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
        Camera ROI along the x and y dimensions.
    snake_order : bool
        Organize the positions for a `snake-type` acquisition. By default, `True`.
    """

    tile_name: str = ""
    top_left: tuple[float, float, float | None]
    bottom_right: tuple[float, float, float | None]
    overlap_x: float
    overlap_y: float
    pixel_size: float
    camera_roi: tuple[int, int]
    snake_order: bool = True

    def tiles(self) -> Sequence[Position]:
        prefix = f"{self.tile_name}_" if self.tile_name else ""

        top_left_x, top_left_y, _ = self.top_left
        bottom_right_x, bottom_right_y, _ = self.bottom_right
        
        cam_width, cam_height = self.camera_roi
        cam_width_minus_overlap = cam_width - (cam_width * self.overlap_x) / 100
        cam_height_minus_overlap = cam_height - (cam_height * self.overlap_y) / 100

        total_width = abs(top_left_x + bottom_right_x)
        total_height = abs(top_left_y + bottom_right_y)

        rows = int(total_width / (cam_width_minus_overlap * self.pixel_size))
        cols = int(total_height / (cam_height_minus_overlap * self.pixel_size))

        increment_x = (
            cam_width_minus_overlap * self.pixel_size 
            if self.overlap_x > 0
            else cam_width * self.pixel_size
        )
        increment_y = (
            cam_height_minus_overlap * self.pixel_size 
            if self.overlap_y > 0
            else cam_height * self.pixel_size
        )

        return _calculate_grid(
            self.top_left,
            increment_x,
            increment_y,
            rows,
            cols,
            self.snake_order,
            prefix
        )


class TileRelative(TilePlan):
    """Define a list of positions for a tile acquisition.

   ...relative to a specified position (`start_coords`).

    Attributes
    ----------
    tile_name : str
        Name of the tile. It will be added as a prefix in front of each
        tile position name. By default "".
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
        Camera ROI along the x and y dimensions.
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

        x_pos, y_pos, _ = self.start_coords
        cam_width, cam_height = self.camera_roi
        cam_width_minus_overlap = cam_width - (cam_width * self.overlap_x) / 100
        cam_height_minus_overlap = cam_height - (cam_height * self.overlap_y) / 100

        if self.relative_to == "center":
            # move to top left corner
            move_x = (cam_width / 2) * (self.cols - 1) - cam_width_minus_overlap
            move_y = (cam_height / 2) * (self.rows - 1) - cam_height_minus_overlap
            x_pos -= self.pixel_size * (move_x + cam_width)
            y_pos += self.pixel_size * (move_y + cam_height)

        increment_x = (
            cam_width_minus_overlap * self.pixel_size 
            if self.overlap_x > 0
            else cam_width * self.pixel_size
        )
        increment_y = (
            cam_height_minus_overlap * self.pixel_size 
            if self.overlap_y > 0
            else cam_height * self.pixel_size
        )
        
        return _calculate_grid(
            self.start_coords,
            increment_x,
            increment_y,
            self.rows,
            self.cols,
            self.snake_order,
            prefix
        )


AnyTilePlan = Union[TileFromCorners, TileRelative]
