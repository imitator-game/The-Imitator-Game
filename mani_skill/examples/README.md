# ManiSkill Examples

This folder contains runnable examples for **The Imitator Game** simulation
framework. Everything runs from the repository root.

```
mani_skill/examples/
├── motionplanning/    scripted dual/single-arm solutions that collect demos
│   ├── dual/          two-panda benchmark tasks (the 50 released tasks)
│   ├── panda/         single panda arm + the shared Panda motion planner
│   ├── realman/       realman arm examples
│   ├── xarm6/         xArm6 examples
│   └── ...
└── teleoperation/     interactive (mouse/keyboard) and VR teleop collection
```

## What you can do here

1. **Visualize a task** and watch a scripted motion-planning solution run.
2. **Collect one demonstration** (H5 + JSON, plus an optional video) for a task.
3. **Batch-collect** all 50 tasks x 4 levels with `scripts/collect_data.py`.
4. **Teleoperate** a panda arm interactively to record demos by hand.

## Motion planning (dual-arm, the benchmark tasks)

The main entry is `mani_skill.examples.motionplanning.dual.two_robot_run`.

```bash
# Visualize the scene + run one solution live
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickCubeYCB-v1 -n 1 --vis --save-video

# Collect one successful trajectory and its video (no viewer)
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickCubeYCB-v1 --l0 -n 1 --only-count-success --save-video \
  --traj-name L0_TwoRobotPickCubeYCB-v1 --record-dir demos
```

- `-e` selects the environment (must have a registered solution).
- `-n` number of trajectories.
- `--only-count-success` keeps only successful episodes (used for the dataset).
- `--l0/--l1/--l2/--l3` select the mismatch level.
- `--vis` opens the interactive viewer; `--save-video` writes an MP4.
- `--traj-name` names the H5; default is `<level>_<env_id>`.

Outputs:

```text
demos/{env_id}/motionplanning/{trajectory_name}.h5     # actions + states + obs
demos/{env_id}/motionplanning/{trajectory_name}.json   # env args, seeds, results
demos/{env_id}/motionplanning/videos/                  # optional MP4s
```

The H5/JSON pair feeds `examples/baselines/lerobot_dataset/h5_to_lerobot.py`
to become LeRobot training data (see `mani_skill/README.md`).

### Adding a solution for a new task

1. Create the task environment (see `mani_skill/envs/tasks/_template/README.md`).
2. Copy `mani_skill/examples/motionplanning/dual/solutions/two_robot_pick_cube_ycb.py`
   (or `_001_stir_spoon.py`) as `solutions/your_task.py`.
3. Implement `solve(env, seed=None, debug=False, vis=False) -> (left_res, right_res)`.
4. Export it in `solutions/__init__.py` and register it in `MP_SOLUTIONS` in
   `two_robot_run.py`.

See `mani_skill/examples/motionplanning/dual/README.md` for the full guide.

## Teleoperation

Interactive collection with a panda arm (click + drag a ghost arm, or use
keyboard shortcuts):

```bash
python -m mani_skill.examples.teleoperation.interactive_panda \
  -e TwoRobotPickCubeYCB-v1 -r panda --save-video
```

Controls: `g` toggle gripper, `n` execute the ghost pose, `c` next episode,
`q` quit, arrow keys move the ghost arm. See
`mani_skill/examples/teleoperation/README.md`.

## Single-arm motion planning (other robots)

Per-robot folders (`panda/`, `realman/`, `xarm6/`, ...) contain their own
`run.py` + `motionplanner.py`. They share the same interface:

```bash
python -m mani_skill.examples.motionplanning.panda.run -e PickCube-v1 -n 1 --vis
```

`panda/motionplanner.py` provides `PandaArmMotionPlanningSolver` (move to pose
with screw motion, open/close gripper, follow_path), which the dual solutions
also reuse.
