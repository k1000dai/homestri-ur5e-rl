"""Compose the robot, the table scene and task objects into one MuJoCo model."""

from pathlib import Path

import mujoco

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

TIMESTEP = 0.002


def build_model(objects_xml: str = "", scene: str = "table") -> mujoco.MjModel:
    """Build a model from the robot, a scene file and extra MJCF for task objects.

    `objects_xml` may contain any top-level MJCF sections (`<asset>`, `<worldbody>`, ...).
    """
    xml = f"""
<mujoco model="homestri">
  <compiler angle="radian" meshdir="{ASSETS_DIR / "meshes"}" autolimits="true" />
  <option integrator="implicitfast" timestep="{TIMESTEP}" impratio="10" cone="elliptic" />
  <include file="{ASSETS_DIR / "robot" / "ur5e_2f85.xml"}" />
  <include file="{ASSETS_DIR / "scenes" / f"{scene}.xml"}" />
  {objects_xml}
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)
