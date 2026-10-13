"""Offscreen RGB rendering of named cameras."""

import mujoco
import numpy as np


class CameraRenderer:
    def __init__(self, model: mujoco.MjModel, height: int, width: int):
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData, camera: str | int | mujoco.MjvCamera) -> np.ndarray:
        self._renderer.update_scene(data, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        self._renderer.close()
