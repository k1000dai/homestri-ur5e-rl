# VLA Sim Modernization — Design

Date: 2026-10-13
Status: Approved in conversation, pending written-spec review

## Goal

Turn this fork into a modern, lightweight MuJoCo simulator for VLA experiments:

- Several pick-and-place tasks usable from Gymnasium.
- Installable with a single `uv sync`.
- A simple API to command the robot by joint positions or end-effector (EEF) pose.

Out of scope: demo collection, dataset writers (LeRobot, HDF5), RL training code,
the legacy cable (wire) task.

## Decisions

- Rewrite as a new package `src/homestri/`. Reuse the existing UR5e, Robotiq 2F-85 and
  pedestal meshes and the MJCF kinematics. Delete the legacy `homestri_ur5e_rl/` package
  (OSC controllers, wire env, vendored MujocoEnv); it remains in git history.
- Arm actuators become joint **position** actuators. EEF commands are converted to joint
  targets with damped-least-squares IK using the MuJoCo Jacobian.
- Dependencies: `mujoco>=3.2`, `gymnasium>=1.0`, `numpy`. Dev: `pytest`, `ruff`,
  `imageio[ffmpeg]`. No `gymnasium-robotics`, no `pynput`.
- Python `>=3.10`, build backend `hatchling`, `uv.lock` committed.

## Layout

```
pyproject.toml
uv.lock
.pre-commit-config.yaml        # ruff format + ruff check
src/homestri/
  __init__.py                  # registers envs on import
  assets/
    meshes/                    # moved from legacy assets (ur5e, 2f85, pedestal)
    robot/ur5e_2f85.xml        # arm + gripper, position actuators, wrist camera, eef site
    scenes/table.xml           # floor, table, lights, front camera
    objects/                   # cube, bin
  sim/
    scene.py                   # builds the model with mujoco.MjSpec from scene + robot + objects
    robot.py                   # Robot: state, joint/EEF/gripper commands, blocking move helpers
    ik.py                      # DLS IK
    camera.py                  # offscreen RGB (and optional depth) via mujoco.Renderer
  tasks/
    base.py                    # Task interface
    pick_cube.py
    place_in_bin.py
    stack_cubes.py
  envs/
    manip_env.py               # ManipEnv(gymnasium.Env)
    registration.py
scripts/
  render_demo.py               # runs a scripted policy and writes a GIF/MP4
  view.py                      # interactive mujoco.viewer
tests/
```

## Robot API (`homestri.sim.robot.Robot`)

Usable without the gym env. All poses are in the robot base frame; quaternions are
`wxyz` (MuJoCo convention).

```python
robot.qpos  # (6,) arm joint positions [rad]
robot.gripper  # float in [0, 1], 0 = open, 1 = closed
robot.eef_pose  # (pos (3,), quat (4,))
robot.set_joint_target(q)  # absolute targets for the 6 arm joints
robot.set_eef_target(pos, quat=None)  # IK -> joint targets; quat=None keeps current orientation
robot.set_gripper(g)  # 0..1
robot.move_joints(q, duration=2.0)  # interpolate and step the sim (blocking)
robot.move_eef(pos, quat=None, duration=2.0)
```

`set_eef_target` returns the IK position/orientation error; it never raises for
unreachable targets.

### IK

- Damped least squares on the 6×6 EEF Jacobian (`mj_jacSite`) on a scratch `MjData`
  copy, warm-started from the current joint positions.
- A few iterations per call (default 20, tol 1e-4), step clipped, joint limits enforced.
- Null-space bias toward the home configuration.

## Gym env (`homestri.envs.ManipEnv`)

Constructor arguments: `task`, `control_mode`, `obs_mode`, `camera_names`,
`image_size`, `control_hz` (default 10), `render_mode`, `max_episode_steps`.

Physics timestep 2 ms; substeps = `1 / (control_hz * timestep)`.

### Action (`control_mode`)

| mode | shape | meaning |
|---|---|---|
| `joint_pos` | 7 | 6 absolute joint positions + gripper |
| `joint_delta` | 7 | 6 joint deltas + gripper |
| `eef_pose` | 8 | pos (3) + quat wxyz (4) absolute + gripper |
| `eef_delta` (default) | 7 | Δpos (3) + Δrotvec (3) + gripper |

Gripper is in `[0, 1]`. Deltas are clipped to per-step limits (default 5 cm, 0.25 rad).

### Observation

```
{
  "images": {"front": (H, W, 3) uint8, "wrist": (H, W, 3) uint8},   # omitted when obs_mode="state"
  "state": {"qpos": (6,), "gripper": (1,), "eef_pos": (3,), "eef_quat": (4,)},
}
```

Default `image_size=(224, 224)`, `camera_names=("front", "wrist")`.

### Info, reward, termination

- `info = {"instruction": str, "success": bool, "ik_error": float}`;
  `env.unwrapped.instruction` also exposes the instruction.
- Reward: `1.0` on success, else `0.0`.
- `terminated` on success; `truncated` at `max_episode_steps` (default 200).

### Rendering

- `render_mode="rgb_array"` returns the front camera image.
- `render_mode="human"` uses `mujoco.viewer.launch_passive`.

## Tasks

A `Task` defines the objects it adds to the scene, randomizes them on `reset(rng)`,
produces an instruction string, and checks `success(model, data)`.

| env id | description | success |
|---|---|---|
| `homestri/PickCube-v0` | lift a colored cube | cube z > table + 0.10 m and in contact with both pads |
| `homestri/PlaceInBin-v0` | put the cube into the bin | cube center inside bin bounds, below bin rim |
| `homestri/StackCubes-v0` | stack the red cube on the blue cube | red above blue, xy offset < 2 cm, both slow |

Cube positions and the target cube color are randomized from the reset seed. A
`num_distractors` argument adds extra cubes of other colors.

## Testing

- Every registered env: `reset`/`step` work and observations match `observation_space`
  (`gymnasium.utils.env_checker.check_env` where applicable).
- IK reaches reachable targets within 5 mm / 0.05 rad.
- For each task, a scripted IK pick-and-place policy succeeds on several seeds.
- Rendering returns images of the requested size (skipped if no GL backend).

## Tooling

- `.pre-commit-config.yaml` with `ruff format` and `ruff check`.
- README rewritten: install (`uv sync`), quick start, API, tasks.

## Implementation notes

- "Robot base frame" is a world-aligned frame at the arm mount (`base` site); x points
  toward the table. This avoids the 90° yaw of the UR5e base body.
- Arm bodies use `gravcomp="1"` so position targets are tracked without sag.
- Delta actions (`joint_delta`, `eef_delta`) accumulate on the previous commanded target,
  not the measured state; the EEF target is kept within `2 * max_delta_pos` of the
  measured pose.
- `StackCubes` randomizes the color pair ("stack the X cube on the Y cube") instead of
  always red-on-blue.
- Each task provides `scripted_targets()` used by `homestri.scripted.ScriptedPickPlace`.
