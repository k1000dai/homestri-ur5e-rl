"""Task interface and shared helpers for tabletop pick-and-place tasks."""

from abc import ABC, abstractmethod

import mujoco
import numpy as np

TABLE_HEIGHT = 0.80
CUBE_HALF_SIZE = 0.02
# Reachable region on the table in world coordinates (robot mount at the origin).
WORKSPACE_LOW = np.array([0.40, -0.22])
WORKSPACE_HIGH = np.array([0.68, 0.22])

COLORS = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.20, 0.70, 0.25, 1.0),
    "blue": (0.15, 0.35, 0.85, 1.0),
    "yellow": (0.90, 0.80, 0.15, 1.0),
    "purple": (0.55, 0.25, 0.75, 1.0),
}

PAD_GEOMS = (
    ("2f85:left_pad1", "2f85:left_pad2"),
    ("2f85:right_pad1", "2f85:right_pad2"),
)


class Task(ABC):
    """A task adds objects to the scene, randomizes them and judges success.

    Positions used by tasks are in world coordinates.
    """

    name: str = "task"

    def __init__(self):
        self.instruction = ""

    @abstractmethod
    def objects_xml(self) -> str:
        """Extra MJCF sections (assets, bodies) added to the scene."""

    @abstractmethod
    def reset(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator) -> None:
        """Randomize object poses/colors and set `self.instruction`."""

    @abstractmethod
    def success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """Whether the task is solved in the current state."""

    @abstractmethod
    def scripted_targets(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """World pick position, pick yaw and world place position for a scripted policy."""


# ---- helpers -----------------------------------------------------------------


def cube_xml(name: str, half_size: float = CUBE_HALF_SIZE) -> str:
    return f"""
    <body name="{name}" pos="0 0 -1">
      <freejoint name="{name}_joint" />
      <geom name="{name}_geom" type="box" size="{half_size} {half_size} {half_size}"
        mass="0.05" friction="1.2 0.01 0.001" solref="0.004 1" condim="4" rgba="1 1 1 1" />
    </body>"""


def set_free_body_pose(model, data, body: str, pos, yaw: float = 0.0) -> None:
    joint_id = model.body(body).jntadr[0]
    adr = model.jnt_qposadr[joint_id]
    data.qpos[adr : adr + 3] = pos
    data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    dof = model.jnt_dofadr[joint_id]
    data.qvel[dof : dof + 6] = 0.0


def body_yaw(data, body: str) -> float:
    w, x, y, z = data.body(body).xquat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def body_speed(data, body: str) -> float:
    return float(np.linalg.norm(data.body(body).cvel[3:]))


def set_geom_color(model, geom: str, color: str) -> None:
    model.geom(geom).rgba[:] = COLORS[color]


def sample_positions(
    rng: np.random.Generator, n: int, min_distance: float, low=WORKSPACE_LOW, high=WORKSPACE_HIGH
) -> list[np.ndarray]:
    """Rejection-sample `n` xy positions at least `min_distance` apart."""
    for _ in range(1000):
        points: list[np.ndarray] = []
        for _ in range(n):
            for _ in range(100):
                p = rng.uniform(low, high)
                if all(np.linalg.norm(p - q) >= min_distance for q in points):
                    points.append(p)
                    break
        if len(points) == n:
            return points
    raise RuntimeError(f"could not place {n} objects {min_distance} m apart")


def is_grasped(model, data, body: str) -> bool:
    """True when the body touches a pad on both gripper fingers."""
    body_id = model.body(body).id
    touching = [False, False]
    for contact in data.contact[: data.ncon]:
        for other, this in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
            if model.geom_bodyid[this] != body_id:
                continue
            name = model.geom(other).name
            for side, pads in enumerate(PAD_GEOMS):
                if name in pads:
                    touching[side] = True
    return all(touching)
