import numpy as np

from homestri.tasks.base import (
    CUBE_HALF_SIZE,
    TABLE_HEIGHT,
    Task,
    body_speed,
    body_yaw,
    cube_xml,
    is_grasped,
    sample_positions,
    set_free_body_pose,
    set_geom_color,
)

BIN_HALF_INNER = 0.07
BIN_WALL = 0.006
BIN_HEIGHT = 0.05


def bin_xml(name: str = "bin") -> str:
    h, w, t = BIN_HALF_INNER, BIN_HEIGHT / 2, BIN_WALL / 2
    o = h + t
    rgba = "0.35 0.35 0.38 1"
    return f"""
    <body name="{name}" pos="0 0 {TABLE_HEIGHT}">
      <geom name="{name}_floor" type="box" size="{o + t} {o + t} {t}" pos="0 0 {t}" rgba="{rgba}" />
      <geom type="box" size="{t} {o + t} {w}" pos="{o} 0 {w}" rgba="{rgba}" />
      <geom type="box" size="{t} {o + t} {w}" pos="{-o} 0 {w}" rgba="{rgba}" />
      <geom type="box" size="{o + t} {t} {w}" pos="0 {o} {w}" rgba="{rgba}" />
      <geom type="box" size="{o + t} {t} {w}" pos="0 {-o} {w}" rgba="{rgba}" />
    </body>"""


class PlaceInBin(Task):
    """Put the cube of the instructed color into the bin."""

    name = "place_in_bin"

    def __init__(self, num_distractors: int = 1, colors=("red", "green", "blue", "yellow")):
        super().__init__()
        if num_distractors + 1 > len(colors):
            raise ValueError("not enough colors for the requested number of distractors")
        self.num_cubes = num_distractors + 1
        self.colors = tuple(colors)
        self.target = "cube0"
        self.target_color = colors[0]

    def objects_xml(self) -> str:
        bodies = "".join(cube_xml(f"cube{i}") for i in range(self.num_cubes))
        return f"<worldbody>{bin_xml()}{bodies}</worldbody>"

    def reset(self, model, data, rng) -> None:
        colors = rng.permutation(self.colors)[: self.num_cubes]
        # The bin takes the first slot; it needs more clearance than a cube.
        points = sample_positions(rng, self.num_cubes + 1, min_distance=0.15)
        bin_xy = points[0]
        model.body("bin").pos[:] = [bin_xy[0], bin_xy[1], TABLE_HEIGHT]
        for i, (color, xy) in enumerate(zip(colors, points[1:], strict=True)):
            set_geom_color(model, f"cube{i}_geom", color)
            z = TABLE_HEIGHT + CUBE_HALF_SIZE
            set_free_body_pose(
                model, data, f"cube{i}", [xy[0], xy[1], z], rng.uniform(-np.pi, np.pi)
            )
        self.target = "cube0"
        self.target_color = str(colors[0])
        self.instruction = f"put the {self.target_color} cube in the bin"

    def success(self, model, data) -> bool:
        cube = data.body(self.target).xpos
        bin_pos = data.body("bin").xpos
        inside = np.all(np.abs(cube[:2] - bin_pos[:2]) < BIN_HALF_INNER)
        low = cube[2] < TABLE_HEIGHT + BIN_HEIGHT
        resting = body_speed(data, self.target) < 0.05 and not is_grasped(model, data, self.target)
        return bool(inside and low and resting)

    def scripted_targets(self, model, data):
        pick = data.body(self.target).xpos.copy()
        place = data.body("bin").xpos.copy() + np.array([0.0, 0.0, BIN_HEIGHT + 0.04])
        return pick, body_yaw(data, self.target), place
