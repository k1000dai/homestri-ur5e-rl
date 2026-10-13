from gymnasium.envs.registration import register, registry

ENV_IDS = {
    "homestri/PickCube-v0": "pick_cube",
    "homestri/PlaceInBin-v0": "place_in_bin",
    "homestri/StackCubes-v0": "stack_cubes",
}


def register_envs() -> None:
    for env_id, task in ENV_IDS.items():
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point="homestri.envs.manip_env:ManipEnv",
            kwargs={"task": task},
            disable_env_checker=True,
        )
