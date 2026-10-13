"""A scripted pick-and-place policy that emits `eef_pose` actions.

Useful for smoke tests and demos; not meant as a strong baseline.
"""

import numpy as np

from homestri.envs.manip_env import ManipEnv

APPROACH_HEIGHT = 0.10
MAX_STEP = 0.025  # metres per control step


def top_down_quat(yaw: float) -> np.ndarray:
    """Gripper pointing down (eef z axis = -world z), rotated by `yaw` about world z."""
    # Snap to the cube symmetry so the wrist rotates at most 45 degrees.
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    down = np.array([0.0, 1.0, 0.0, 0.0])
    rot = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    w1, x1, y1, z1 = rot
    w2, x2, y2, z2 = down
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


class ScriptedPickPlace:
    """Waypoint policy: approach, descend, grasp, lift, move, descend, release, retreat."""

    def __init__(self, env: ManipEnv):
        if env.control_mode != "eef_pose":
            raise ValueError("ScriptedPickPlace needs control_mode='eef_pose'")
        self.env = env
        self._plan: list[tuple[np.ndarray, np.ndarray, float, int]] = []

    def reset(self) -> None:
        env = self.env
        robot = env.robot
        pick_world, yaw, place_world = env.task.scripted_targets(env.model, env.data)
        pick, _ = robot.world_to_base(pick_world)
        place, _ = robot.world_to_base(place_world)
        quat = top_down_quat(yaw)
        up = np.array([0.0, 0.0, APPROACH_HEIGHT])
        grasp = pick + np.array([0.0, 0.0, 0.005])
        # (position, quaternion, gripper, extra hold steps)
        waypoints = [
            (pick + up, quat, 0.0, 2),
            (grasp, quat, 0.0, 2),
            (grasp, quat, 1.0, 6),
            (pick + up, quat, 1.0, 0),
            (place + up, quat, 1.0, 0),
            (place, quat, 1.0, 2),
            (place, quat, 0.0, 6),
            (place + up, quat, 0.0, 2),
        ]
        self._plan = []
        current = robot.eef_pose[0]
        for pos, q, grip, hold in waypoints:
            n = max(1, int(np.ceil(np.linalg.norm(pos - current) / MAX_STEP)))
            for i in range(1, n + 1):
                self._plan.append((current + (pos - current) * i / n, q, grip))
            self._plan.extend([(pos, q, grip)] * hold)
            current = pos

    @property
    def done(self) -> bool:
        return not self._plan

    def __call__(self, obs=None) -> np.ndarray:
        if not self._plan:
            pos, quat = self.env.robot.eef_pose
            return np.r_[pos, quat, 0.0].astype(np.float32)
        pos, quat, grip = self._plan.pop(0)
        return np.r_[pos, quat, grip].astype(np.float32)
