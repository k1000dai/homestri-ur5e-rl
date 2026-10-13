import numpy as np
import pytest

from homestri.envs import CONTROL_MODES, ENV_IDS
from homestri.scripted import ScriptedPickPlace

from .conftest import requires_gl


@pytest.mark.parametrize("env_id", list(ENV_IDS))
@pytest.mark.parametrize("control_mode", CONTROL_MODES)
def test_reset_step_spaces(make_env, env_id, control_mode):
    env = make_env(env_id, control_mode=control_mode, obs_mode="state")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info["instruction"], str) and info["instruction"]
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert reward in (0.0, 1.0)
        assert set(info) >= {"instruction", "success", "ik_error"}


def test_reset_is_deterministic_for_seed(make_env):
    env = make_env("homestri/StackCubes-v0", obs_mode="state")
    env.reset(seed=3)
    first = (env.unwrapped.instruction, env.unwrapped.data.qpos.copy())
    env.reset(seed=3)
    assert env.unwrapped.instruction == first[0]
    np.testing.assert_allclose(env.unwrapped.data.qpos, first[1])


def test_eef_delta_moves_eef(make_env):
    env = make_env(control_mode="eef_delta", obs_mode="state")
    obs, _ = env.reset(seed=0)
    start = obs["state"]["eef_pos"].copy()
    for _ in range(5):
        obs, *_ = env.step(np.array([0.0, 0.0, 0.02, 0, 0, 0, 0], dtype=np.float32))
    assert obs["state"]["eef_pos"][2] - start[2] > 0.07


def test_out_of_range_action_is_clipped(make_env):
    env = make_env(control_mode="eef_delta", obs_mode="state")
    obs, _ = env.reset(seed=0)
    start = obs["state"]["eef_pos"].copy()
    obs, *_ = env.step(np.array([0.0, 0.0, 10.0, 0, 0, 0, 0], dtype=np.float32))
    assert obs["state"]["eef_pos"][2] - start[2] < 0.06


def test_truncation(make_env):
    env = make_env(obs_mode="state", max_episode_steps=2)
    env.reset(seed=0)
    hold = np.zeros(7, dtype=np.float32)
    assert not env.step(hold)[3]
    assert env.step(hold)[3]


def test_invalid_arguments(make_env):
    with pytest.raises(ValueError):
        make_env(control_mode="torque")
    with pytest.raises(ValueError):
        make_env(obs_mode="pointcloud")
    with pytest.raises(KeyError):
        make_env(camera_names=("nope",))


@pytest.mark.parametrize("env_id", list(ENV_IDS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scripted_policy_succeeds(make_env, env_id, seed):
    env = make_env(env_id, control_mode="eef_pose", obs_mode="state")
    obs, _ = env.reset(seed=seed)
    policy = ScriptedPickPlace(env.unwrapped)
    policy.reset()
    for _ in range(150):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        if terminated or truncated:
            break
    assert info["success"], info["instruction"]


@requires_gl
def test_rgb_observation_and_render(make_env):
    env = make_env(image_size=(96, 128), render_mode="rgb_array", render_size=(120, 160))
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs["images"]["front"].shape == (96, 128, 3)
    assert obs["images"]["wrist"].std() > 0
    assert env.render().shape == (120, 160, 3)
