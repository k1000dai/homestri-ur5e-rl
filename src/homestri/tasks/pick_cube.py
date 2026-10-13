import numpy as np

from homestri.tasks.base import (
    CUBE_HALF_SIZE,
    TABLE_HEIGHT,
    Task,
    body_yaw,
    cube_xml,
    is_grasped,
    sample_positions,
    set_free_body_pose,
    set_geom_color,
)


class PickCube(Task):
    """Lift the cube of the instructed color."""

    name = "pick_cube"
    lift_height = 0.10

    def __init__(self, num_distractors: int = 2, colors=("red", "green", "blue", "yellow")):
        super().__init__()
        if num_distractors + 1 > len(colors):
            raise ValueError("not enough colors for the requested number of distractors")
        self.num_cubes = num_distractors + 1
        self.colors = tuple(colors)
        self.target = "cube0"
        self.target_color = colors[0]

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
        self.target = "cube0"
        self.target_color = str(colors[0])
        self.instruction = f"pick up the {self.target_color} cube"

    def success(self, model, data) -> bool:
        lifted = data.body(self.target).xpos[2] > TABLE_HEIGHT + CUBE_HALF_SIZE + self.lift_height
        return bool(lifted and is_grasped(model, data, self.target))

    def scripted_targets(self, model, data):
        pick = data.body(self.target).xpos.copy()
        place = pick + np.array([0.0, 0.0, self.lift_height + 0.05])
        return pick, body_yaw(data, self.target), place
