# Template Baseline (minimal)

A minimal example that shows how to add a **new imitation-learning model** to the Imitator Game benchmark. Copy this folder, rename `template` to your model name, and replace `TemplatePolicy` with your own network. Everything else (dataset, encoders, simulator evaluation) can be reused unchanged.

```
examples/baselines/_template/
├── train_template_imitator.py   # training entry (dataset + policy + loop)
├── eval_template_imitator.py    # simulator evaluation entry
└── README.md
```

## The data contract (how to use the dataset & simulator)

Training data is LeRobot-format and lives at:

| Path | Contents |
| --- | --- |
| `demos/demo_data` | Human demonstration videos |
| `demos/imitator_data` | Paired simulation (robot) trajectories |

Config JSON files under `examples/baselines/lerobot_dataset/config/exp_configs/`
select which tasks are used:

- `<kind>_train_config_{15,30,45}.json` — training splits
- `<kind>_test_config_{seen,unseen}.json` — evaluation splits

`task_mapping.json` pairs each human demo with the corresponding simulation
levels (L0-L3). The `HumanSimPairedDataset` loader combines both sides and
returns one sample per call:

```
sample["robot_obs"]       # robot RGB image + robot state (dict)
sample["robot_actions"]   # (pred_horizon, action_dim) normalized action chunk
sample["human_video"]     # human demonstration video (video_only mode)
sample["human_repo_id"]   # optional cache key for the task encoder
```

The **simulator evaluation** is fully handled by the shared pipeline
(`HumanVideoSimEvaluateProcessor` + `evaluate_with_task_encoder`). Your agent
only needs to answer `get_action(obs)` at every control step.

## Required agent interface

```python
class YourAgent(nn.Module):
    def compute_loss(self, batch) -> dict:      # training: return {"loss": tensor}
    def prepare_for_eval(self, human_video, robot_obs, human_desc=None, human_vl_ids=None):
        ...                                     # per episode: cache task features
    @torch.no_grad()
    def get_action(self, obs) -> Tensor:        # return (B, pred_horizon, action_dim)
    def clear_cache(self): ...                  # between episodes
```

## Train

```bash
export PYTHONPATH=$PWD:$PYTHONPATH

python -m examples.baselines.template.train_template_imitator \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-dataset-file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json \
  --sim-dataset-file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_15.json \
  --task-mapping-file examples/baselines/lerobot_dataset/task_mapping.json \
  --input-mode video_only \
  --frozen-backbone-type dinov2_vitl14 \
  --pred-horizon 16 --obs-horizon 1 \
  --batch-size 256 --total-epochs 10 --lr 1e-4
```

Checkpoint: `runs/<run_name>/checkpoints/final_model.pt`

## Evaluate

```bash
python -m examples.baselines.template.eval_template_imitator \
  --eval-config examples/baselines/lerobot_dataset/eval/exp_list/seen_5tasks_env_list.txt \
  --checkpoint runs/xxx/checkpoints/final_model.pt \
  --output-dir runs/template_eval \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-config examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json \
  --sim-config examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json \
  --task-mapping examples/baselines/lerobot_dataset/task_mapping.json \
  --human-task-desc examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim-task-desc examples/baselines/lerobot_dataset/task_desc/sim_desc.json
```

## How to add your own model

1. Open `train_template_imitator.py`, find `TemplatePolicy`.
2. Replace it with your network (same I/O: conditioning vector in,
   `(B, pred_horizon, action_dim)` action chunk out).
3. Update `TemplateAgent.compute_loss()` for your training objective.
4. Keep `prepare_for_eval` / `get_action` / `clear_cache` — these are what the
   simulator evaluator calls.

As long as `get_action()` returns normalized actions of shape
`(B, pred_horizon, action_dim)`, the existing simulator evaluation code is reused
as-is.

## Notes

- `action_dim=16` and `state_dim=18` are the dual-panda defaults (7 arm joints +
  1 gripper per robot). Change them only if you use a different robot.
- The frozen video backbone (`dinov2_vitl14`) is the shared task encoder; see
  `examples/baselines/encoders/` for alternatives.
