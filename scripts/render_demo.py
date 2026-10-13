"""Run the scripted pick-and-place policy on a task and save a video.

uv run python scripts/render_demo.py --env homestri/StackCubes-v0 --out stack.mp4
"""

import argparse

import gymnasium as gym
import imageio.v3 as iio
import numpy as np

import homestri  # noqa: F401  (registers environments)
from homestri.scripted import ScriptedPickPlace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="homestri/PickCube-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out", default="demo.mp4")
    args = parser.parse_args()

    env = gym.make(args.env, control_mode="eef_pose", obs_mode="rgb", render_mode="rgb_array")
    policy = ScriptedPickPlace(env.unwrapped)
    frames = []
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        policy.reset()
        print(f"episode {episode}: {info['instruction']}")
        while True:
            wrist = obs["images"]["wrist"]
            front = env.render()
            # Show the wrist camera as an inset in the top-left corner.
            front[: wrist.shape[0], : wrist.shape[1]] = wrist
            frames.append(front)
            obs, _, terminated, truncated, info = env.step(policy(obs))
            if terminated or truncated or (policy.done and not info["success"]):
                break
        frames.extend([env.render()] * 5)
        print(f"  success={info['success']}")
    env.close()
    iio.imwrite(args.out, np.stack(frames), fps=env.metadata["render_fps"])
    print(f"saved {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()
