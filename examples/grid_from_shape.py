"""Cover an elliptical area of each well with tiles, without any GUI.

`GridFromShape` yields *relative* offsets that fully cover the area; edge tiles may
extend outside of it. The same plan can be used as the `well_points_plan` of a
`WellPlatePlan` or as the `grid_plan` of an `MDASequence`.
"""

import useq

# a 4 mm circle (e.g. a restricted area at the center of a 6.4 mm well),
# imaged with a 650 x 550 µm field of view and 10% overlap
coverage = useq.GridFromShape(
    shape="ellipse",
    width=4000,
    height=4000,
    overlap=10,
    mode="row_wise_snake",
    fov_width=650,
    fov_height=550,
)
print("tiles per well:", coverage.num_positions())

plate_plan = useq.WellPlatePlan(
    plate="96-well",
    a1_center_xy=(0, 0),
    selected_wells=([0, 0, 1], [0, 1, 1]),
    well_points_plan=coverage,
)
print("total positions:", len(plate_plan))
for pos in plate_plan[:3]:
    print(pos.name, round(pos.x or 0, 1), round(pos.y or 0, 1))

# the plan round-trips through JSON and keeps its type
reloaded = useq.WellPlatePlan.model_validate_json(plate_plan.model_dump_json())
assert isinstance(reloaded.well_points_plan, useq.GridFromShape)

coverage.plot(show=True)
