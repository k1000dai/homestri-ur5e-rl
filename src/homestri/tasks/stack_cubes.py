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


class StackCubes(Task):
    """Stack the instructed cube on top of another instructed cube."""

    name = "stack_cubes"

    def __init__(self, num_distractors: int = 1, colors=("red", "green", "blue", "yellow")):
        super().__init__()
        if num_distractors + 2 > len(colors):
            raise ValueError("not enough colors for the requested number of distractors")
        self.num_cubes = num_distractors + 2
        self.colors = tuple(colors)
        self.top, self.bottom = "cube0", "cube1"

    def objects_xml(self) -> str:
        bodies = "".join(cube_xml(f"cube{i}") for i in range(self.num_cubes))
        return f"<worldbody>{bodies}</worldbody>"

    def reset(self, model, data, rng) -> None:
        colors = rng.permutation(self.colors)[: self.num_cubes]
        points = sample_positions(rng, self.num_cubes, min_distance=0.09)
        for i, (color, xy) in enumerate(zip(colors, points, strict=True)):
            set_geom_color(model, f"cube{i}_geom", color)
            z = TABLE_HEIGHT + CUBE_HALF_SIZE
            set_free_body_pose(
                model, data, f"cube{i}", [xy[0], xy[1], z], rng.uniform(-np.pi, np.pi)
            )
        self.top, self.bottom = "cube0", "cube1"
        self.instruction = f"stack the {colors[0]} cube on the {colors[1]} cube"

    def success(self, model, data) -> bool:
        top = data.body(self.top).xpos
        bottom = data.body(self.bottom).xpos
        stacked = (
            np.linalg.norm(top[:2] - bottom[:2]) < 0.02
            and abs(top[2] - bottom[2] - 2 * CUBE_HALF_SIZE) < 0.01
        )
        resting = body_speed(data, self.top) < 0.05 and body_speed(data, self.bottom) < 0.05
        return bool(stacked and resting and not is_grasped(model, data, self.top))

    def scripted_targets(self, model, data):
        pick = data.body(self.top).xpos.copy()
        place = data.body(self.bottom).xpos.copy() + np.array(
            [0.0, 0.0, 2 * CUBE_HALF_SIZE + 0.005]
        )
        return pick, body_yaw(data, self.top), place
