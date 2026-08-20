# XSkill

XSkill is a two-stage cross-embodiment imitation-learning baseline. It first learns a shared visual skill space from paired human and robot demonstrations, then trains a diffusion policy that uses the learned skill representation to generate robot actions.

Use XSkill when you have task-aligned human videos and robot demonstrations, and want to transfer the task progress visible in the human video to a robot policy.

## At a glance

```text
Paired human and robot demonstrations
        │
        ▼
Stage 1 — skill discovery
Learn a visual-motion encoder and skill prototypes
        │
        ▼
Stage 2 — skill-conditioned imitation
Condition a diffusion action policy on the learned representation
        │
        ▼
Evaluation
Roll out the policy in ManiSkill with a human demonstration as task context
```

Stage 1 learns an embodiment-shared representation from short video windows. Stage 2 combines the robot observation, robot state, and the representation inferred from the human demonstration to predict a receding-horizon action sequence.

## Before you start

XSkill uses the shared project Python environment; it does not maintain a separate environment in this directory. From the repository root, install and activate the project environment according to the top-level [README](../../../README.md). The launchers expect the project virtual environment at `$PROJECT_ROOT/.venv`.

Set the paths that are specific to your machine and dataset before launching an experiment:

```bash
export PROJECT_ROOT=/path/to/repository
export SERVER_HOME=/path/to/your/home
export CONDA_PREFIX=/path/to/your/conda

export HUMAN_ROOT=/path/to/human/demonstrations
export SIM_ROOT=/path/to/simulation/demonstrations
# Required only by robot-data launchers:
export ROBOT_ROOT=/path/to/robot/demonstrations

# Optional cache locations:
export HF_HOME="$PROJECT_ROOT/.hf_home"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/hf_datasets"
```

Keep WandB credentials outside the repository, for example in your shell environment or the secret manager used by your cluster.

## Data requirements

XSkill relies on the project's LeRobot dataset integration. The human and target-embodiment datasets must be paired through a shared task mapping. The standard metadata files are:

```text
examples/baselines/lerobot_dataset/task_mapping.json
examples/baselines/lerobot_dataset/task_desc/human_desc.json
examples/baselines/lerobot_dataset/task_desc/sim_desc.json
examples/baselines/lerobot_dataset/task_desc/robot_desc.json
examples/baselines/lerobot_dataset/config/exp_configs/*_train_config_*.json
examples/baselines/lerobot_dataset/config/exp_configs/*_test_config_*.json
```

Stage 1 reads matched human and robot videos. Stage 2 additionally requires robot state and action trajectories. Evaluation requires the human demonstration configuration, the simulator configuration, and an environment list.

The task mapping, data split, camera names, state representation, and action dimensionality must agree across the two stages. A checkpoint trained on one task split should not be evaluated with metadata from another split.

## Training workflow

### 1. Train the skill encoder

`scripts/stage1_pretrain_encoder.py` trains the Stage 1 visual-motion encoder. Its model uses visual features, temporal attention, prototype assignments, and cross-embodiment video pairs to learn a shared skill space.

| Launcher | Data source |
| --- | --- |
| [`run_exp_xskill_stage1.sh`](../../../exp_scripts/xskill/run_exp_xskill_stage1.sh) | Human + simulation pairs |

Stage 1 artifacts are normally written below:

```text
examples/baselines/xskill/logs/stage1_<tag>/xskill/experiment/pretrain/
```

Preserve both the Lightning checkpoint and the Hydra run configuration. Stage 2 loads the Stage 1 configuration from this experiment directory.

### 2. Train the diffusion policy

`scripts/stage2_skill_transfer.py` trains the action policy. The policy takes a robot observation window and state, obtains a skill condition from the human demonstration, and uses a conditional 1D UNet to denoise an action sequence.

| Launcher | Purpose |
| --- | --- |
| [`run_exp_xskill_stage2.sh`](../../../exp_scripts/xskill/run_exp_xskill_stage2.sh) | Simulation skill transfer |

Most launchers select the task-count experiments through `RUN_ONLY`:

```bash
cd "$PROJECT_ROOT"
RUN_ONLY=15 bash exp_scripts/xskill/run_exp_xskill_stage2.sh
```

Use `RUN_ONLY=15,30,45` or `RUN_ONLY=all` when the matching GPU assignments and checkpoints have been configured in the script. Stage 2 artifacts are normally written below:

```text
examples/baselines/xskill/logs/stage2_<tag>/xskill/experiment/transfer/
```

The policy checkpoint, the Stage 1 path, the data split, and the Hydra configuration form one inseparable experiment record.

## Evaluation

Use [`run_eval_xskill.sh`](../../../exp_scripts/xskill/run_eval_xskill.sh) for parallel evaluation. Each entry in its `JOBS` array has the format `name|checkpoint|hydra_config|eval_config|sim_config|human_config|output_dir|gpu_ids`.

Before launching, verify that every referenced checkpoint and configuration exists and give every job a unique `output_dir`.

```bash
cd "$PROJECT_ROOT"
bash exp_scripts/xskill/run_eval_xskill.sh
```

Useful evaluation controls include `INPUT_MODE`, `NUM_EPISODES`, `NUM_ENVS`, `SIM_BACKEND`, `CONTROL_MODE`, `OBS_MODE`, `COMPUTE_DTW`, and `DTW_BAND_RATIO`. The evaluator writes metrics, rollout videos, and logs under each job's output directory.

## What to keep after a run

For each reproducible result, keep:

1. the train/test dataset configuration JSON files and task mapping;
2. the Stage 1 checkpoint and Hydra configuration;
3. the Stage 2 checkpoint and Hydra configuration;
4. the environment list, evaluation settings, and output directory;
5. the repository revision, seed, GPU configuration, and any launcher overrides.

## Configuration

The active Hydra configurations are stored directly in `config/`:

- `stage1_pretrain_encoder.yaml` is used by the Stage 1 launchers.
- `stage2_skill_transfer.yaml` is used by the Stage 2 launchers.

The launchers select these defaults through their corresponding training modules. Pass standard Hydra command-line overrides after the launcher command when an experiment needs different settings.

## Citation

```bibtex
@inproceedings{
  xu2023xskill,
  title={{XS}kill: Cross Embodiment Skill Discovery},
  author={Mengda Xu and Zhenjia Xu and Cheng Chi and Manuela Veloso and Shuran Song},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023},
  url={https://openreview.net/forum?id=8L6pHd9aS6w}
}
```