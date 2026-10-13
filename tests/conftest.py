import gymnasium as gym
import mujoco
import pytest

import homestri  # noqa: F401


def _gl_available() -> bool:
    try:
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        mujoco.Renderer(model, 8, 8).close()
        return True
    except Exception:
        return False


requires_gl = pytest.mark.skipif(not _gl_available(), reason="no OpenGL backend")


@pytest.fixture
def make_env():
    envs = []

    def _make(env_id="homestri/PickCube-v0", **kwargs):
        env = gym.make(env_id, **kwargs)
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.close()
