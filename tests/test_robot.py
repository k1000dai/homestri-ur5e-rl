import mujoco
import numpy as np
import pytest

from homestri.scripted import top_down_quat
from homestri.sim import Robot, build_model
from homestri.sim.ik import orientation_error


@pytest.fixture
def robot():
    model = build_model()
    data = mujoco.MjData(model)
    robot = Robot(model, data)
    robot.reset()
    mujoco.mj_forward(model, data)
    return robot


@pytest.mark.parametrize(
    "pos, yaw",
    [([0.5, 0.0, 0.05], 0.0), ([0.45, -0.2, 0.15], 0.5), ([0.65, 0.2, 0.0], -0.7)],
)
def test_ik_reaches_target(robot, pos, yaw):
    quat = top_down_quat(yaw)
    result = robot.solve_ik(pos, quat)
    assert result.pos_error < 5e-3
    assert result.rot_error < 0.05


def test_move_eef_tracks_target_in_sim(robot):
    target = np.array([0.5, 0.1, 0.05])
    quat = top_down_quat(0.3)
    robot.move_eef(target, quat, duration=1.5)
    pos, current = robot.eef_pose
    assert np.linalg.norm(pos - target) < 5e-3
    assert np.linalg.norm(orientation_error(quat, current)) < 0.05


def test_move_joints(robot):
    goal = robot.qpos + np.array([0.2, -0.1, 0.1, 0.0, 0.1, 0.3])
    robot.move_joints(goal, duration=1.0)
    np.testing.assert_allclose(robot.qpos, goal, atol=5e-3)


def test_unreachable_target_does_not_raise(robot):
    result = robot.solve_ik([3.0, 0.0, 0.0])
    assert result.pos_error > 1.0
    assert np.all(result.qpos >= robot.joint_low) and np.all(result.qpos <= robot.joint_high)


def test_gripper_open_close(robot):
    robot.set_gripper(1.0)
    robot.step(500)
    assert robot.gripper > 0.9
    robot.set_gripper(0.0)
    robot.step(500)
    assert robot.gripper < 0.05


def test_base_frame_round_trip(robot):
    pos, quat = robot.eef_pose
    world_pos, world_quat = robot.base_to_world(pos, quat)
    back_pos, back_quat = robot.world_to_base(world_pos, world_quat)
    np.testing.assert_allclose(back_pos, pos, atol=1e-9)
    np.testing.assert_allclose(np.abs(np.dot(back_quat, quat)), 1.0, atol=1e-9)
