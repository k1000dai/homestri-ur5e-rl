"""Gymnasium environment for tabletop manipulation with the UR5e + 2F-85."""

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from homestri.sim.camera import CameraRenderer
from homestri.sim.robot import HOME_QPOS, Robot
from homestri.sim.scene import build_model
from homestri.tasks import TASKS, Task

CONTROL_MODES = ("joint_pos", "joint_delta", "eef_pose", "eef_delta")
OBS_MODES = ("rgb", "state")


class ManipEnv(gym.Env):
    """Pick-and-place environment.

    Actions (all poses in the robot base frame, quaternions wxyz, gripper in [0, 1]):
      joint_pos   (7,) absolute joint positions + gripper
      joint_delta (7,) joint deltas + gripper
      eef_pose    (8,) position (3) + quaternion (4) + gripper
      eef_delta   (7,) delta position (3) + delta rotation vector (3, world-aligned) + gripper

    Delta actions are applied to the previous commanded target rather than the measured
    state, so a sequence of deltas is not shortened by tracking lag.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        task: str | Task = "pick_cube",
        control_mode: str = "eef_delta",
        obs_mode: str = "rgb",
        camera_names: tuple[str, ...] = ("front", "wrist"),
        image_size: tuple[int, int] = (224, 224),
        control_hz: float = 10.0,
        render_mode: str | None = None,
        render_size: tuple[int, int] = (480, 640),
        max_episode_steps: int = 200,
        max_delta_pos: float = 0.05,
        max_delta_rot: float = 0.25,
        task_kwargs: dict[str, Any] | None = None,
    ):
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}, got {control_mode!r}")
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}, got {obs_mode!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self.task = TASKS[task](**(task_kwargs or {})) if isinstance(task, str) else task
        self.control_mode = control_mode
        self.obs_mode = obs_mode
        self.camera_names = tuple(camera_names)
        self.image_size = tuple(image_size)
        self.render_mode = render_mode
        self.render_size = tuple(render_size)
        self.max_episode_steps = max_episode_steps
        self.max_delta_pos = max_delta_pos
        self.max_delta_rot = max_delta_rot

        self.model = build_model(self.task.objects_xml())
        self.data = mujoco.MjData(self.model)
        self.robot = Robot(self.model, self.data)
        self.n_substeps = max(1, round(1.0 / (control_hz * self.model.opt.timestep)))
        self.metadata = {**self.metadata, "render_fps": control_hz}
        for name in self.camera_names:
            self.model.camera(name)  # raises KeyError for unknown cameras

        self.action_space = self._make_action_space()
        self.observation_space = self._make_observation_space()

        self._camera_renderer: CameraRenderer | None = None
        self._render_renderer: CameraRenderer | None = None
        self._viewer = None
        self._steps = 0
        self._last_ik_error = 0.0
        self._eef_target: tuple[np.ndarray, np.ndarray] | None = None

    # ---- spaces --------------------------------------------------------------

    def _make_action_space(self) -> spaces.Box:
        low, high = self.robot.joint_low, self.robot.joint_high
        if self.control_mode == "joint_pos":
            lo, hi = np.r_[low, 0.0], np.r_[high, 1.0]
        elif self.control_mode == "joint_delta":
            lim = np.full(6, self.max_delta_rot)
            lo, hi = np.r_[-lim, 0.0], np.r_[lim, 1.0]
        elif self.control_mode == "eef_pose":
            lo = np.r_[np.full(3, -1.5), np.full(4, -1.0), 0.0]
            hi = np.r_[np.full(3, 1.5), np.full(4, 1.0), 1.0]
        else:
            lim = np.r_[np.full(3, self.max_delta_pos), np.full(3, self.max_delta_rot)]
            lo, hi = np.r_[-lim, 0.0], np.r_[lim, 1.0]
        return spaces.Box(lo.astype(np.float32), hi.astype(np.float32), dtype=np.float32)

    def _make_observation_space(self) -> spaces.Dict:
        state = spaces.Dict(
            {
                "qpos": spaces.Box(-np.inf, np.inf, (6,), np.float32),
                "gripper": spaces.Box(0.0, 1.0, (1,), np.float32),
                "eef_pos": spaces.Box(-np.inf, np.inf, (3,), np.float32),
                "eef_quat": spaces.Box(-1.0, 1.0, (4,), np.float32),
            }
        )
        obs = {"state": state}
        if self.obs_mode == "rgb":
            h, w = self.image_size
            obs["images"] = spaces.Dict(
                {name: spaces.Box(0, 255, (h, w, 3), np.uint8) for name in self.camera_names}
            )
        return spaces.Dict(obs)

    # ---- gym API -------------------------------------------------------------

    @property
    def instruction(self) -> str:
        return self.task.instruction

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.robot.reset(HOME_QPOS, gripper=0.0)
        self.task.reset(self.model, self.data, self.np_random)
        mujoco.mj_forward(self.model, self.data)
        # Let objects settle on the table before the episode starts.
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self._steps = 0
        self._last_ik_error = 0.0
        self._eef_target = self.robot.eef_pose
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), self._get_info(success=False)

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=float), self.action_space.low, self.action_space.high
        )
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self._steps += 1

        success = self.task.success(self.model, self.data)
        terminated = success
        truncated = self._steps >= self.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(success), terminated, truncated, self._get_info(success)

    def _apply_action(self, action: np.ndarray) -> None:
        robot = self.robot
        ik_error = 0.0
        if self.control_mode == "joint_pos":
            robot.set_joint_target(action[:6])
        elif self.control_mode == "joint_delta":
            robot.set_joint_target(robot.joint_target + action[:6])
        elif self.control_mode == "eef_pose":
            result = robot.set_eef_target(action[:3], action[3:7])
            ik_error = result.pos_error
        else:
            pos, quat = self._eef_target
            delta_quat = np.empty(4)
            mujoco.mju_axisAngle2Quat(
                delta_quat,
                action[3:6] / max(np.linalg.norm(action[3:6]), 1e-12),
                np.linalg.norm(action[3:6]),
            )
            target_quat = np.empty(4)
            mujoco.mju_mulQuat(target_quat, delta_quat, quat)
            # Keep the target near the measured pose so it cannot run away when unreachable.
            measured = robot.eef_pose[0]
            offset = pos + action[:3] - measured
            max_offset = 2 * self.max_delta_pos
            if np.linalg.norm(offset) > max_offset:
                offset *= max_offset / np.linalg.norm(offset)
            target_pos = measured + offset
            result = robot.set_eef_target(target_pos, target_quat)
            ik_error = result.pos_error
            self._eef_target = (target_pos, target_quat)
        robot.set_gripper(action[-1])
        self._last_ik_error = ik_error

    def _get_obs(self) -> dict:
        pos, quat = self.robot.eef_pose
        obs = {
            "state": {
                "qpos": self.robot.qpos.astype(np.float32),
                "gripper": np.array([self.robot.gripper], dtype=np.float32),
                "eef_pos": pos.astype(np.float32),
                "eef_quat": quat.astype(np.float32),
            }
        }
        if self.obs_mode == "rgb":
            if self._camera_renderer is None:
                self._camera_renderer = CameraRenderer(self.model, *self.image_size)
            obs["images"] = {
                name: self._camera_renderer.render(self.data, name) for name in self.camera_names
            }
        return obs

    def _get_info(self, success: bool) -> dict:
        return {
            "instruction": self.instruction,
            "success": success,
            "ik_error": self._last_ik_error,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            if self._render_renderer is None:
                self._render_renderer = CameraRenderer(self.model, *self.render_size)
            return self._render_renderer.render(self.data, "front")
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer

                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        for renderer in (self._camera_renderer, self._render_renderer):
            if renderer is not None:
                renderer.close()
        self._camera_renderer = self._render_renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
