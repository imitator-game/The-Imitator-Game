# Teleoperation Examples

Interactive demonstration collection for the Imitator Game. Two families of
entry points:

| File | Method | Robot |
| --- | --- | --- |
| `interactive_panda.py` | click + drag ghost arm, keyboard shortcuts | panda / panda_wristcam / panda_stick |
| `interactive_realman.py`, `interactive_realman_inspire.py`, ... | same pattern | realman variants |
| `vr_realman*.py` | VR teleoperation | realman variants |

## Interactive (mouse + keyboard) teleop — panda

Reference implementation for panda-type arms:

```bash
python -m mani_skill.examples.teleoperation.interactive_panda \
  -e TwoRobotPickCubeYCB-v1 -r panda --save-video
```

How it works:

1. A **ghost panda arm** is shown in the viewer. Drag its gizmo with the mouse
   to place a target pose.
2. Keyboard commands:

   | Key | Action |
   | --- | --- |
   | `g` | toggle gripper open/close |
   | `u` / `j` | move ghost arm up / down |
   | arrow keys | move ghost arm in the arrow direction |
   | `n` | run motion planning and move the real robot to the ghost pose |
   | `c` | finish the current episode and continue to the next one |
   | `q` | quit (saves trajectories and optionally videos) |

3. Each finished episode is recorded under
   `demos/{env_id}/teleop/trajectory.h5` (+ companion `.json`). With
   `--save-video`, the H5 is replayed to render an MP4 per episode.

## How to extend to a new robot

`interactive_panda.py` is written around a generic `solve(env, ...)` loop.
To support another robot:

1. Add a branch in `_make_planner(env)` returning the matching motion-planning
   solver (the solver must expose `move_to_pose_with_screw(dry_run=True)`,
   `follow_path`, `close_gripper`, `open_gripper`).
2. Adjust the gizmo->TCP z-offset constant used when executing the ghost pose
   (hardcoded per robot type).
3. The recording/replay logic in `main()` is robot-agnostic and can be reused
   unchanged.

## VR teleop

`vr_realman*.py` use the same RecordEpisode wrapper; see the top of each file
for the required VR / headset setup.
