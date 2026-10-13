from homestri.tasks.base import Task
from homestri.tasks.pick_cube import PickCube
from homestri.tasks.place_in_bin import PlaceInBin
from homestri.tasks.stack_cubes import StackCubes

TASKS = {cls.name: cls for cls in (PickCube, PlaceInBin, StackCubes)}

__all__ = ["TASKS", "PickCube", "PlaceInBin", "StackCubes", "Task"]
