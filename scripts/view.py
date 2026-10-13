"""Open an environment in the interactive MuJoCo viewer and run the scripted policy.

uv run mjpython scripts/view.py --env homestri/PlaceInBin-v0   # macOS needs mjpython
"""

import argparse
import time

import gymnasium as gym

import homestri  # noqa: F401
from homestri.scripted import ScriptedPickPlace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="homestri/PickCube-v0")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = gym.make(args.env, control_mode="eef_pose", obs_mode="state", render_mode="human")
    policy = ScriptedPickPlace(env.unwrapped)
    for episode in range(args.episodes):
        obs, info = env.reset(seed=episode)
        policy.reset()
        print(info["instruction"])
        while True:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            time.sleep(1.0 / env.metadata["render_fps"])
            if terminated or truncated or policy.done:
                break
        print(f"  success={info['success']}")
    env.close()


if __name__ == "__main__":
    main()
