# OpenVLA (OpenFlamingo Transformer head)

OpenVLA is a 7B open-source Vision-Language-Action model built on a Prismatic VLM base with a continuous action-head, fine-tuned for robot policy learning (see [Citation](#citation)). This directory contains the project integration of **openvla_oft** (the OpenVLA-Open-Flamingo-Transformer implementation) with a ManiSkill / LeRobot data pipeline.

## Entrypoints

- Training: `train_openvla.py`
- Evaluation: `eval_openvla.py` (single checkpoint), `eval_openvla_batch.py` (batch evaluation)
- VLA feature precompute (cache mode): `precompute_vla_features.py`
- Shared dependencies: `examples/baselines/lerobot_dataset` and `examples/baselines/encoders`

## Expected execution context

Run from the repository root (see top-level [`README.md`](../../../README.md) for the shared environment):

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

On NVIDIA 40-series GPUs, disable NCCL P2P/IB before training:

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

## Action modes

The action head supports three output modes selected with `--action_mode`:

- `discrete` — OpenVLA-style discrete action tokenization
- `l1_regression` — continuous L1 regression head (recommended)
- `diffusion` — diffusion action head (supports `--num_diffusion_steps_train` and `--num_diffusion_steps_inference`, both default 50)

## Training

### Legacy ManiSkill h5 demo path

```bash
python -m examples.baselines.openvla_oft.train_openvla \
  --demo_path "demos/PickCubeYCB-v1/motionplanning/multi_task_4.rgbd.pd_joint_delta_pos.physx_cpu.h5" \
  --batch_size 2 \
  --learning_rate 2e-5 \
  --total_iters 1000000 \
  --eval_freq 10000 \
  --num_eval_episodes 10 \
  --use_lora \
  --lora_rank 256 \
  --lora_alpha 128 \
  --use_gradient_checkpointing \
  --action_mode l1_regression
```

### LeRobot paired dataset path (reported experiments)

The reported experiments precompute OpenVLA V/L hidden states (`vla_cache_dir`) once per dataset, then train an epoch-based L1-regression LoRA model that reloads the cached features:

1. Precompute features, e.g. with [`run_precompute_vla_cache_45.sh`](../../../exp_scripts/precompute_cache/run_precompute_vla_cache_45.sh) (`RUN_ID=428`, cache under `feature_cache/openvla_vl_cache_45_h200_run${RUN_ID}`):

```bash
python -m examples.baselines.openvla_oft.precompute_vla_features \
  --cache_dir <CACHE_DIR> \
  --model_name_or_path <openvla--openvla-7b> \
  --use_lerobot \
  --human_root demos/demo_data \
  --sim_root demos/imitator_data \
  --human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json \
  --sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json \
  --task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
  --human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --lerobot_camera zed2i \
  --pred_horizon 16 --action_dim 16 --obs_horizon 1 \
  --batch_size 256 --num_workers 8 --shard_size 25600
```

2. Train in cache mode (see [`run_exp_openvla_cache_45.sh`](../../../exp_scripts/openvla/run_exp_openvla_cache_45.sh)):

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m examples.baselines.openvla_oft.train_openvla \
  --use_lerobot \
  --human_root demos/demo_data \
  --sim_root demos/imitator_data \
  --human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json \
  --sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json \
  --task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
  --human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --lerobot_camera zed2i \
  --model_name_or_path <openvla--openvla-7b> \
  --vla_cache_dir <CACHE_DIR> \
  --batch_size 512 --effective_batch_size 512 --pin_memory --num_workers 4 \
  --action_mode l1_regression --learning_rate 1e-4 \
  --output_dir <OUTPUT_DIR> --exp_name <EXP_NAME> \
  --use_proprio \
  --use_epoch_training --total_epochs 10 --save_epoch_freq 1 --eval_epoch_freq 100 \
  --env_id TwoRobotPourCup-v1 --eval_camera zed2i \
  --num_eval_episodes 1 --num_eval_envs 1 \
  --eval_sim_task_ids L0_TwoRobotPourCup-v1 \
  --eval_lr_mirror auto --eval_lr_mirror_robot_pose false \
  --capture_video --control_mode pd_joint_pos --no_eval
```

The training script redirects `HF_DATASETS_CACHE` to a node-local cache directory (to avoid distributed-FS flock deadlock) and cleans it up after the run.

## Evaluation

### LeRobot online eval (LoRA checkpoint)

```bash
python -m examples.baselines.openvla_oft.eval_openvla \
  --checkpoint_dir runs/<exp>/final \
  --model_name_or_path <openvla--openvla-7b> \
  --use_lerobot \
  --sim_root demos/imitator_data \
  --sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_eval_config_seen_plus_unseen_10tasks.json \
  --task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
  --human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --env_id TwoRobotPickAppleToScale-v1 \
  --eval_camera zed2i \
  --num_eval_episodes 10 --num_eval_envs 1 \
  --capture_video \
  --control_mode pd_joint_pos \
  --action_mode l1_regression --action_dim 16 \
  --eval_sim_task_ids TwoRobotPickAppleToScale-v1
```

### Parallel eval + other expansions

The parallel eval orchestrator [`run_eval_openvla.sh`](../../../exp_scripts/openvla/run_eval_openvla.sh) launches per-tag (15/30/45) schedulers on GPUs 0/1/2 using `examples.baselines.openvla_oft.eval_openvla` against the checkpoint directories pinned by `CHECKPOINT_DIR_{15,30,45}`. Defaults: seen+unseen 10-task eval env list, `NUM_EVAL_EPISODES=10`, `NUM_EVAL_ENVS=5`, `MAX_EPISODE_STEPS=600`, `ACTION_MODE=l1_regression`. Results go under `runs/openvla_eval_results/`.

Related exp_scripts: [`run_exp_openvla_cache_45.sh`](../../../exp_scripts/openvla/run_exp_openvla_cache_45.sh) (cache-mode training), [`run_eval_openvla.sh`](../../../exp_scripts/openvla/run_eval_openvla.sh) (parallel eval), and [`run_precompute_vla_cache_45.sh`](../../../exp_scripts/precompute_cache/run_precompute_vla_cache_45.sh) (VLA feature precompute).

## Notes

- Config paths historically referenced as `config/human_config.json` / `config/sim_config.json` now live under `examples/baselines/lerobot_dataset/config/exp_configs/` (e.g. `human_train_config_45.json`, `sim_eval_config_seen_plus_unseen_10tasks.json`).
- The cache-mode run sets a large `batch_size` because feature generation is served from the precomputed V/L cache rather than the frozen VLM.
- Eval LR-mirror handling (`--eval_lr_mirror auto`) matches the other ManiSkill baselines for the dual-arm convention.

## Citation

```bibtex
@article{kim2024openvla,
  title={Openvla: An open-source vision-language-action model, 2024},
  author={Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and Xiao, Ted and Balakrishna, Ashwin and Nair, Suraj and Rafailov, Rafael and Foster, Ethan and Lam, Grace and Sanketi, Pannag and others},
  journal={URL https://arxiv. org/abs/2406.09246},
  volume={1},
  number={2},
  pages={4},
  year={2024}
}
```