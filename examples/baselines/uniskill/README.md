# UniSkill

UniSkill learns cross-embodiment skill representations from human and robot video, then uses those representations to condition a robot diffusion policy. The intended use case is imitation from a human demonstration: a human video provides task-progress context, while the learned policy produces robot actions in the target environment.

## At a glance

```text
Human and robot video pairs
        │
        ▼
Stage 1 — inverse dynamics model (IDM)
Encode motion between consecutive frames as a skill latent
        │
        ▼
Stage 2 — conditional diffusion policy
Align human IDM latents with the current robot observation
        │
        ▼
Evaluation
Use the human-video latent sequence to condition robot rollouts
```

The IDM combines visual and depth features with a spatiotemporal Transformer. The policy combines robot visual features, robot state, and an aligned human skill latent, then uses a conditional 1D UNet to predict an action horizon.

## Important release status

The current checkout is fully runnable end to end for UniSkill:

- `diffusion/train_uniskill.py` consumes `diffusion/pipeline_dynamics.py` (the image-to-image diffusion pipeline) and `diffusion/policy_model.py` (the shared conditional UNet, alignment transformer, and ResNet backbones).
- `diffusion/train_cond_dp.py` and `diffusion/eval_uniskill.py` reuse the same modules, together with the parallel scheduler `diffusion/parallel_eval_uniskill.py`.
- Training launchers live in `scripts/` and `examples/baselines/exp_scripts/uniskill/`.

The information below documents the expected workflow, data contract, artifacts, and launchers.

## Environment and dataset paths

UniSkill uses the shared project environment; it does not have a directory-local environment. Install and activate the project environment from the repository root as described in the top-level [README](../../../README.md).

Set the machine-specific paths before using a launcher:

```bash
export PROJECT_ROOT=/path/to/repository
export SERVER_HOME=/path/to/your/home
export CONDA_PREFIX=/path/to/your/conda

export HUMAN_ROOT=/path/to/human/demonstrations
export SIM_ROOT=/path/to/simulation/demonstrations

# Optional cache locations:
export HF_HOME="$PROJECT_ROOT/.hf_home"
export UNISKILL_HF_DATASETS_CACHE_ROOT="$PROJECT_ROOT/.cache/hf_datasets"
export CACHE_ROOT="$PROJECT_ROOT/feature_cache/uniskill_cache"
export PREDECODE_CACHE_DIR="$CACHE_ROOT/human_video_predecode"
```

`HUMAN_ROOT` and `SIM_ROOT` must point to the same task universe represented by the dataset JSON files. Keep WandB credentials outside the repository.

## Data contract

The project integration uses LeRobot task metadata:

```text
examples/baselines/lerobot_dataset/task_mapping.json
examples/baselines/lerobot_dataset/task_desc/human_desc.json
examples/baselines/lerobot_dataset/task_desc/sim_desc.json
examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_*.json
examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_*.json
examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_*.json
examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_*.json
```

Stage 1 expects consecutive human and robot video frames, together with depth features. Stage 2 expects robot RGB observations, robot states, robot actions, and the associated human videos. Task mapping, split, camera configuration, state representation, and action dimensionality must remain consistent across training and evaluation.

## Intended workflow

### 1. Train the IDM

`diffusion/train_uniskill.py` trains the IDM jointly with an image-to-image diffusion objective. The output needed by later stages is:

```text
<stage1-output>/checkpoint-<step>/idm.pth
```

The intended multi-GPU launcher is [`run_exp_uniskill.sh`](../../../exp_scripts/uniskill/run_exp_uniskill.sh). It configures task-count runs such as 15, 30, and 45, and writes outputs under `examples/baselines/uniskill/outputs/uniskill_<tag>/`.

### 2. Train the conditional diffusion policy

`diffusion/train_cond_dp.py` freezes the IDM and trains a policy that aligns the human latent sequence to the current robot observation. The key artifacts are:

```text
<policy-output>/checkpoint-<step>/model*.safetensors
<policy-output>/stats.pickle
```

`stats.pickle` contains the state/action normalization statistics and must remain with the policy checkpoint.

| Launcher | Intended purpose |
| --- | --- |
| [`run_exp_cond_dp.sh`](../../../exp_scripts/uniskill/run_exp_cond_dp.sh) | Cond-DP training (45/30/15-task parallel) |

The launchers expose cache controls such as `USE_PREDECODE_CACHE`, `USE_HUMAN_IDM_CACHE`, `USE_ROBOT_IDM_CACHE`, `BUILD_HUMAN_IDM_CACHE`, and `BUILD_ROBOT_IDM_CACHE`. Use a distinct cache directory for different task splits or checkpoints to avoid mixing incompatible features.

### 3. Evaluate

Evaluation needs all of the following:

```text
Stage 2 policy checkpoint directory
Stage 1 idm.pth
stats.pickle from the Stage 2 output directory
human and simulator dataset configurations
task or environment list
an empty output directory
```

The intended parallel launcher is [`run_eval_uniskill.sh`](../../../exp_scripts/uniskill/run_eval_uniskill.sh). A rollout encodes the human video into an IDM latent sequence, aligns that sequence to the current robot observation, samples an action sequence, executes the first action horizon, and repeats.

## Reproducibility checklist

Keep the following together for every reported result:

1. dataset JSON files, task mapping, and task descriptions;
2. IDM checkpoint and its training configuration;
3. policy checkpoint and `stats.pickle`;
4. cache policy, task split, seed, GPU setup, and launcher overrides;
5. evaluation environment list, episode count, and result directory.

## Citation

```bibtex
@article{kim2025uniskill,
  title={Uniskill: Imitating human videos via cross-embodiment skill representations},
  author={Kim, Hanjung and Kang, Jaehyun and Kang, Hyolim and Cho, Meedeum and Kim, Seon Joo and Lee, Youngwoon},
  journal={arXiv preprint arXiv:2505.08787},
  year={2025}
}
```