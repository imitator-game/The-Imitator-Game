#!/bin/bash
# OpenVLA train (cache mode): 45-task run, vla_cache_dir must contain precomputed features.
# MODEL_NAME_OR_PATH must point to a local openvla-7b checkpoint.

set -euo pipefail

GPU="${GPU:-0}"
CACHE_DIR="${CACHE_DIR:-runs/openvla_vl_cache_45}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/openvla_45_cache}"
EXP_NAME="${EXP_NAME:-openvla_45_h200_cache}"

export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES="$GPU" python -m examples.baselines.openvla_oft.train_openvla \
  --use_lerobot \
  --human_root demos/demo_data \
  --sim_root demos/imitator_data \
  --human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json \
  --sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json \
  --task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
  --human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --lerobot_camera zed2i \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --vla_cache_dir "$CACHE_DIR" \
  --batch_size 512 \
  --effective_batch_size 512 \
  --pin_memory \
  --num_workers 4 \
  --action_mode l1_regression \
  --learning_rate 1e-4 \
  --output_dir "$OUTPUT_DIR" \
  --exp_name "$EXP_NAME" \
  --use_proprio \
  --use_epoch_training \
  --total_epochs 10 \
  --save_epoch_freq 1 \
  --eval_epoch_freq 100 \
  --env_id TwoRobotPourCup-v1 \
  --eval_camera zed2i \
  --num_eval_episodes 1 \
  --num_eval_envs 1 \
  --eval_sim_task_ids L0_TwoRobotPourCup-v1 \
  --eval_lr_mirror auto \
  --eval_lr_mirror_robot_pose false \
  --capture_video \
  --control_mode pd_joint_pos \
  --no_eval