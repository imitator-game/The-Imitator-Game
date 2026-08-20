# L0-L3 Environment Template

Minimal templates for creating new **Imitator Game** tabletop tasks. Two levels
of abstraction:

| File | Purpose |
| --- | --- |
| `template_task.py` | L0/L1/L2 environment (one shared registered class) |
| `object_loader.py` | Helpers for loading YCB / RoboTwin / PartNet / sketchfab assets |
| `README.md` | This guide |

The L0/L1/L2 class is registered as `TwoRobotTemplateTask-v1`; the L3 class as
`TwoRobotTemplateTaskL3-v1`. Rename both before use.

## How levels work

| Level | What changes | Where it is implemented |
| --- | --- | --- |
| L0 | base scene (nothing) | the environment class itself |
| L1 | same objects, rearranged layout | `apply_l1_offset_xy(...)` in `_initialize_episode` |
| L2 | same semantics, different instances | `apply_l2_ycb_model_id(...)` in `__init__` |
| L3 | different semantics, reusable affordances | a separate `...L3-v1` class |

The level is chosen **before** `gym.make` via
`configure_dual_task_level(level)` (from
`mani_skill.envs.tasks.tabletop.utils.dual_task_camera_utils`), or via the
`--l0/--l1/--l2` flags of `two_robot_run`. Do not add level logic to the policy.

## Creating a new task

1. Copy `template_task.py` to `mani_skill/envs/tasks/tabletop/dual_tasks/_NNN_your_task.py`.
2. Change the `@register_env` id, class name, and docstring.
3. Import them in `mani_skill/envs/tasks/tabletop/__init__.py`.
4. Replace the assets in `_load_scene()` — use `object_loader.py`:

```python
from mani_skill.envs.tasks._template.object_loader import (
    load_ycb_object, load_robotwin_object, load_partnet_object,
    load_sketchfab_object, z_offset_to_table,
)

# YCB object (dynamic)
self.apple, _ = load_ycb_object(self.scene, "013_apple", position=[0, 0.5, 0],
                                scale=0.7, mass=0.5)

# RoboTwin container (static goal)
self.basket, _ = load_robotwin_object(self.scene, "076_breadbasket",
                                      position=[0, -0.5, 0],
                                      rotation=(np.pi / 2, 0, 0),
                                      is_static=True)

# PartNet articulated object (microwave, cabinet, ...)
self.microwave, _ = load_partnet_object(self.scene, "microwave", "7119",
                                        position=[0.3, 0.1, 0],
                                        robot_base_position=[0, -1, 0])

# External GLB asset
self.scale, _ = load_sketchfab_object(self.scene, "balance_scale",
                                      position=[0.2, 0.0, 0.0])
```

5. Update `_initialize_episode()` (randomize poses, apply L1 offsets).
6. Update `evaluate()` (success predicates) and `compute_dense_reward()`.

## Required method contract

Every task environment must implement:

- `_load_scene(options)` — build table + load all actors/articulations
- `_initialize_episode(env_idx, options)` — randomize per-episode state
- `_after_reconfigure(options)` — compute z-offsets for table contact
- `evaluate()` — return dict with a `success` tensor (rule-based)
- `_get_obs_extra(info)` — extra observation fields
- `compute_dense_reward(obs, action, info)` — reward
- `compute_normalized_dense_reward(obs, action, info)`

Camera configuration (`_default_sensor_configs`, `hi_res`, `wrist_sensor`) and
the dual-agent setup are identical to the released tasks; keep them as-is so the
data pipeline (two_robot_run, collect_data, h5_to_lerobot) works unchanged.

## Using different dataset assets

Assets live under `~/.maniskill/data` (see `mani_skill/README.md`):

| Asset set | Loader | Download |
| --- | --- | --- |
| YCB | `load_ycb_object` | `python -m mani_skill.utils.download_asset ycb` |
| RoboTwin | `load_robotwin_object` | `python -m mani_skill.utils.download_robotwin` |
| PartNet-Mobility | `load_partnet_object` | `python -m mani_skill.utils.download_partnet` |
| sketchfab GLB | `load_sketchfab_object` | manual (see `assets/sketchfab_README.md`) |

PartNet object ids per category can be listed from the downloaded PartNet-Mobility
dataset metadata (see `mani_skill/utils/building/articulations/partnet_mobility.py`
and `~/.maniskill/data/partnet_mobility/`).

## Verify

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotTemplateTask-v1 -n 1 --only-count-success --record-dir demos
```
