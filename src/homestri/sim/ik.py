"""Damped-least-squares inverse kinematics on the MuJoCo site Jacobian."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    qpos: np.ndarray
    pos_error: float
    rot_error: float


def orientation_error(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    """Rotation vector that rotates `current_quat` onto `target_quat` (world frame)."""
    conj = np.empty(4)
    mujoco.mju_negQuat(conj, current_quat)
    diff = np.empty(4)
    mujoco.mju_mulQuat(diff, target_quat, conj)
    err = np.empty(3)
    mujoco.mju_quat2Vel(err, diff, 1.0)
    return err


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    qpos_adr: np.ndarray,
    dof_adr: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_init: np.ndarray,
    q_home: np.ndarray | None = None,
    iterations: int = 20,
    tolerance: float = 1e-4,
    damping: float = 1e-3,
    max_step: float = 0.2,
    nullspace_gain: float = 0.05,
) -> IKResult:
    """Solve for arm joint positions that put `site_id` at the target pose (world frame).

    `data` is used as scratch space; only its `qpos` and kinematics are modified.
    Never raises for unreachable targets: returns the best effort and its error.
    """
    low = model.jnt_range[model.dof_jntid[dof_adr], 0]
    high = model.jnt_range[model.dof_jntid[dof_adr], 1]
    q = np.clip(np.asarray(q_init, dtype=float).copy(), low, high)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err = np.zeros(6)
    n = len(dof_adr)

    for _ in range(iterations):
        data.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        site_quat = np.empty(4)
        mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id])
        err[:3] = target_pos - data.site_xpos[site_id]
        err[3:] = orientation_error(target_quat, site_quat)
        if np.linalg.norm(err) < tolerance:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp[:, dof_adr], jacr[:, dof_adr]])
        jjt = jac @ jac.T + damping * np.eye(6)
        dq = jac.T @ np.linalg.solve(jjt, err)
        if q_home is not None:
            null = np.eye(n) - np.linalg.pinv(jac) @ jac
            dq += null @ (nullspace_gain * (q_home - q))
        norm = np.linalg.norm(dq)
        if norm > max_step:
            dq *= max_step / norm
        q = np.clip(q + dq, low, high)

    data.qpos[qpos_adr] = q
    mujoco.mj_kinematics(model, data)
    site_quat = np.empty(4)
    mujoco.mju_mat2Quat(site_quat, data.site_xmat[site_id])
    pos_error = float(np.linalg.norm(target_pos - data.site_xpos[site_id]))
    rot_error = float(np.linalg.norm(orientation_error(target_quat, site_quat)))
    return IKResult(qpos=q, pos_error=pos_error, rot_error=rot_error)
