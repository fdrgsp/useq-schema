from __future__ import annotations

from typing import Iterator, Literal, Sequence, Union

from ._base_model import FrozenModel
from ._position import Position

# from useq._base_model import FrozenModel
# from useq._position import Position


class TilePlan(FrozenModel):
    overlap: float  # percentage
    pixel_size: float
    camera_roi: tuple[int, int]

    def __iter__(self) -> Iterator[Position]:
        yield from self.tiles()

    def tiles(self) -> Sequence[Position]:
        raise NotImplementedError()

    def __len__(self) -> int:
        return len(self.tiles())


class TileFromCorners(TilePlan):
    top_left: tuple[float, float]
    top_right: tuple[float, float]
    overlap: float
    pixel_size: float
    camera_roi: tuple[int, int]


class TileRelative(TilePlan):
    rows: int
    cols: int
    overlap: float
    pixel_size: float
    camera_roi: tuple[int, int]
    start_coords: tuple[float, float, float | None]
    relative_to_coords: Literal["center", "top_left"]

    def tiles(self) -> Sequence[Position]:

        x_pos, y_pos, z_pos = self.start_coords
        cam_width, cam_height = self.camera_roi
        overlap_x = cam_width - (cam_width * self.overlap) / 100
        overlap_y = cam_height - (cam_height * self.overlap) / 100

        if self.relative_to_coords == "center":
            # move to top left corner
            move_x = (cam_width / 2) * (self.cols - 1) - overlap_x
            move_y = (cam_height / 2) * (self.rows - 1) - overlap_y
            x_pos -= self.pixel_size * (move_x + cam_width)
            y_pos += self.pixel_size * (move_y + cam_height)

        # calculate position increments depending on pixle size
        if self.overlap > 0:
            increment_x = overlap_x * self.pixel_size
            increment_y = overlap_y * self.pixel_size
        else:
            increment_x = cam_width * self.pixel_size
            increment_y = cam_height * self.pixel_size

        tile_pos_list: list[Position] = []
        pos_count = 0
        for r in range(self.rows):
            if r % 2:  # for odd rows
                col = self.cols - 1
                for c in range(self.cols):
                    if c == 0:
                        y_pos -= increment_y
                    tile_pos_list.append(Position(
                        name=f"Pos{pos_count:03d}", x=x_pos, y=y_pos, z=z_pos
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
                        name=f"Pos{pos_count:03d}", x=x_pos, y=y_pos, z=z_pos
                    ))
                    pos_count += 1
                    if c < self.cols - 1:
                        x_pos += increment_x

        return tile_pos_list


AnyTilePlan = Union[TileFromCorners, TileRelative]












