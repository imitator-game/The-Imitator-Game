# Dual-Arm Motion Planning (Imitator Game tasks)

This folder contains the scripted **motion-planning solutions** that generate
the dual-panda demonstration dataset for the Imitator Game benchmark.

```
mani_skill/examples/motionplanning/dual/
├── two_robot_run.py               # runner CLI (the entry point)
├── utils.py                       # shared helpers
├── solutions/                     # one solve() per base (L0/L1/L2) task
│   ├── two_robot_pick_cube_ycb.py # simple reference solution (start here)
│   ├── _001_stir_spoon.py         # full task solution example
│   └── ...
├── solutions_l3/                  # one solve() per L3 task
└── README.md
```

## Run one solution

```bash
# visualize the scene and data-collection process
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickCubeYCB-v1 -n 1 --vis --save-video

# collect 1 demo + its video (saved under demos/)
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotPickCubeYCB-v1 -n 1 --only-count-success --save-video \
  --traj-name L0_TwoRobotPickCubeYCB-v1 --record-dir demos
```

The command:
1. opens the scene (and the viewer with `--vis`);
2. runs the registered `solve()` until `-n` successful trajectories are recorded;
3. saves `demos/{env_id}/motionplanning/{traj_name}.h5` (actions + states +
   RGB-D obs) and the companion `.json`; `--save-video` additionally writes
   an MP4 of the collection run.

## How to add a solution for a new task

Every released task is four components: base env (`...-v1`), L3 env
(`...L3-v1`), base solution, and L3 solution. Only the solution is described
here; see `mani_skill/envs/tasks/_template/README.md` for the environments.

### Step 1: create the solution file

Copy the reference solution:

```bash
cp mani_skill/examples/motionplanning/dual/solutions/two_robot_pick_cube_ycb.py \
   mani_skill/examples/motionplanning/dual/solutions/_NNN_your_task.py
```

### Step 2: implement `solve()`

The runner calls `solve(env, seed=None, debug=False, vis=False)` and expects the
tuple `(left_res, right_res)` (each result from the last planner action, or
`-1` on failure).

Minimal skeleton:

```python
from mani_skill.envs.tasks import YourTaskEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: YourTaskEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)

    left_planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.agents[0].robot.pose,
        visualize_target_grasp_pose=vis, print_env_info=False,
        multi_robot_id=0,                      # left robot
    )
    right_planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.agents[1].robot.pose,
        visualize_target_grasp_pose=vis, print_env_info=False,
        multi_robot_id=1,                      # right robot
    )
    env = env.unwrapped

    # Always pass the other robot's gripper state to avoid collisions.
    left_planner.move_to_pose_with_screw(
        my_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    ...
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    return left_res, left_res
```

Key planner API (`PandaArmMotionPlanningSolver`, see `../panda/motionplanner.py`):

| Method | Purpose |
| --- | --- |
| `move_to_pose_with_screw(pose, other_gripper_state=...)` | plan + execute a screw-motion path to `pose` |
| `close_gripper(...)` / `open_gripper(...)` | gripper commands (returns `(obs, reward, terminated, truncated, info)`) |
| `follow_path(result)` | execute a returned plan |
| `gripper_state` | current gripper state (share it with the other planner) |
| `move_to_pose_with_screw(..., dry_run=True)` | plan without executing |

Useful helpers: `get_actor_obb(actor)`, `compute_grasp_info_by_obb(...)` from
`../panda/utils.py` to build grasp poses from an object's oriented bounding box.

### Step 3: register the solution

In `solutions/__init__.py`:

```python
from ._NNN_your_task import solve as solveYourTask
```

In `two_robot_run.py`, add to `MP_SOLUTIONS`:

```python
MP_SOLUTIONS = {
    ...,
    "TwoRobotYourTask-v1": solveYourTask,
}
```

If the task is L0/L1/L2, its L3 variant goes into `solutions_l3/` and is
registered with the `...L3-v1` key.

### Step 4: verify

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotYourTask-v1 -n 1 --only-count-success --record-dir demos
```

## Tips

- **Level hooks**: the base task environment (L0/L1/L2) is the same class; use
  `is_l2_enabled()` from `envs/tasks/tabletop/utils/L0_L3_utils.py` inside the
  solution to adapt waypoints when needed (see `_001_stir_spoon.py`).
- **L3 solutions** are separate files because the object affordances and
  waypoints differ structurally from the base task.
- **Determinism**: `env.reset(seed=seed)` at the start keeps episodes
  reproducible; the runner increments `seed` per trajectory.
- Always pass `other_gripper_state` — the two planners share one scene and
  would otherwise collide.
