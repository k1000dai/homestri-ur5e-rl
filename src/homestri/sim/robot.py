"""High-level access to the UR5e arm and Robotiq 2F-85 gripper.

All poses are expressed in the robot base frame: origin at the arm mount on the
pedestal, axes aligned with the world (x points toward the table). Quaternions
are wxyz.
"""

import mujoco
import numpy as np

from homestri.sim.ik import IKResult, solve_ik

ARM_JOINTS = (
    "ur5e:shoulder_pan_joint",
    "ur5e:shoulder_lift_joint",
    "ur5e:elbow_joint",
    "ur5e:wrist_1_joint",
    "ur5e:wrist_2_joint",
    "ur5e:wrist_3_joint",
)
ARM_ACTUATORS = (
    "ur5e:shoulder_pan",
    "ur5e:shoulder_lift",
    "ur5e:elbow",
    "ur5e:wrist_1",
    "ur5e:wrist_2",
    "ur5e:wrist_3",
)
GRIPPER_ACTUATOR = "2f85:fingers_actuator"
GRIPPER_JOINT = "2f85:right_driver_joint"
GRIPPER_CTRL_MAX = 255.0
GRIPPER_JOINT_MAX = 0.8
EEF_SITE = "eef"
BASE_SITE = "base"

# Gripper pointing straight down above the table.
HOME_QPOS = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])


class Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        joint_ids = [model.joint(name).id for name in ARM_JOINTS]
        self.qpos_adr = model.jnt_qposadr[joint_ids]
        self.dof_adr = model.jnt_dofadr[joint_ids]
        self.actuator_ids = np.array([model.actuator(name).id for name in ARM_ACTUATORS])
        self.gripper_actuator_id = model.actuator(GRIPPER_ACTUATOR).id
        self.gripper_qpos_adr = model.jnt_qposadr[model.joint(GRIPPER_JOINT).id]
        self.eef_site_id = model.site(EEF_SITE).id
        self.base_site_id = model.site(BASE_SITE).id
        self.joint_low = model.jnt_range[joint_ids, 0].copy()
        self.joint_high = model.jnt_range[joint_ids, 1].copy()
        self._scratch = mujoco.MjData(model)

    # ---- state -------------------------------------------------------------

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    @property
    def qvel(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].copy()

    @property
    def joint_target(self) -> np.ndarray:
        return self.data.ctrl[self.actuator_ids].copy()

    @property
    def gripper(self) -> float:
        """Measured gripper closure in [0, 1] (0 = open)."""
        value = self.data.qpos[self.gripper_qpos_adr] / GRIPPER_JOINT_MAX
        return float(np.clip(value, 0.0, 1.0))

    @property
    def eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.world_to_base(
            self.data.site_xpos[self.eef_site_id],
            _mat_to_quat(self.data.site_xmat[self.eef_site_id]),
        )

    # ---- frames ------------------------------------------------------------

    def _base(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.site_xpos[self.base_site_id]
        rot = self.data.site_xmat[self.base_site_id].reshape(3, 3)
        return pos, rot

    def base_to_world(self, pos, quat=None):
        base_pos, base_rot = self._base()
        world_pos = base_pos + base_rot @ np.asarray(pos, dtype=float)
        if quat is None:
            return world_pos, None
        return world_pos, _mul_quat(_mat_to_quat(base_rot.ravel()), quat)

    def world_to_base(self, pos, quat=None):
        base_pos, base_rot = self._base()
        local_pos = base_rot.T @ (np.asarray(pos, dtype=float) - base_pos)
        if quat is None:
            return local_pos, None
        return local_pos, _mul_quat(_mat_to_quat(base_rot.T.ravel()), quat)

    # ---- commands ----------------------------------------------------------

    def reset(self, qpos: np.ndarray = HOME_QPOS, gripper: float = 0.0) -> None:
        """Teleport the arm to `qpos` and hold it there. Call `mj_forward` afterwards."""
        self.data.qpos[self.qpos_adr] = qpos
        self.data.qvel[self.dof_adr] = 0.0
        self.set_joint_target(qpos)
        self.set_gripper(gripper)

    def set_joint_target(self, qpos) -> None:
        q = np.clip(np.asarray(qpos, dtype=float), self.joint_low, self.joint_high)
        self.data.ctrl[self.actuator_ids] = q

    def solve_ik(self, pos, quat=None, q_init=None) -> IKResult:
        """IK for an EEF pose in the base frame. `quat=None` keeps the current orientation."""
        if quat is None:
            quat = self.eef_pose[1]
        world_pos, world_quat = self.base_to_world(pos, _normalize(quat))
        self._scratch.qpos[:] = self.data.qpos
        return solve_ik(
            self.model,
            self._scratch,
            self.eef_site_id,
            self.qpos_adr,
            self.dof_adr,
            world_pos,
            world_quat,
            q_init=self.qpos if q_init is None else q_init,
            q_home=HOME_QPOS,
        )

    def set_eef_target(self, pos, quat=None) -> IKResult:
        result = self.solve_ik(pos, quat)
        self.set_joint_target(result.qpos)
        return result

    def set_gripper(self, value: float) -> None:
        self.data.ctrl[self.gripper_actuator_id] = (
            float(np.clip(value, 0.0, 1.0)) * GRIPPER_CTRL_MAX
        )

    # ---- blocking helpers ----------------------------------------------------

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    def move_joints(self, qpos, duration: float = 2.0, settle: float = 0.5) -> None:
        """Linearly interpolate the joint target to `qpos` while stepping the simulation."""
        start = self.joint_target
        goal = np.asarray(qpos, dtype=float)
        n = max(1, int(duration / self.model.opt.timestep))
        for i in range(1, n + 1):
            self.set_joint_target(start + (goal - start) * i / n)
            self.step()
        self.step(int(settle / self.model.opt.timestep))

    def move_eef(
        self, pos, quat=None, duration: float = 2.0, settle: float = 0.5, ik_hz: float = 50.0
    ) -> IKResult:
        """Move the EEF along a straight line (slerp for orientation) to the target pose."""
        start_pos, start_quat = self.eef_pose
        goal_pos = np.asarray(pos, dtype=float)
        goal_quat = start_quat if quat is None else _normalize(quat)
        substeps = max(1, int(1.0 / (ik_hz * self.model.opt.timestep)))
        n = max(1, int(duration * ik_hz))
        q = self.joint_target
        result = None
        for i in range(1, n + 1):
            alpha = i / n
            waypoint_quat = np.empty(4)
            _slerp(waypoint_quat, start_quat, goal_quat, alpha)
            result = self.solve_ik(
                start_pos + (goal_pos - start_pos) * alpha, waypoint_quat, q_init=q
            )
            q = result.qpos
            self.set_joint_target(q)
            self.step(substeps)
        self.step(int(settle / self.model.opt.timestep))
        return result


def _mat_to_quat(mat) -> np.ndarray:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=float).ravel())
    return quat


def _mul_quat(a, b) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def _normalize(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=float)
    return quat / np.linalg.norm(quat)


def _slerp(out, q0, q1, t) -> None:
    q1 = q1 if np.dot(q0, q1) >= 0 else -q1
    diff = np.empty(4)
    conj = np.empty(4)
    mujoco.mju_negQuat(conj, q0)
    mujoco.mju_mulQuat(diff, conj, q1)
    vel = np.empty(3)
    mujoco.mju_quat2Vel(vel, diff, 1.0)
    out[:] = q0
    mujoco.mju_quatIntegrate(out, vel, t)
