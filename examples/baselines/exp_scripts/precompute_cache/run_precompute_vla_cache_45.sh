#!/bin/bash
# OpenVLA feature precompute (45 tasks): precompute VLA hidden states for cache training.
# MODEL_NAME_OR_PATH and CACHE_DIR must be set.

set -euo pipefail

export PYTHONUNBUFFERED=1

CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR (e.g. feature_cache/openvla_vl_cache_45)}"
mkdir -p "$CACHE_DIR"

python -m examples.baselines.openvla_oft.precompute_vla_features \
  --cache_dir "$CACHE_DIR" \
  --model_name_or_path "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH (openvla-7b)}" \
  --use_lerobot \
  --human_root demos/demo_data \
  --sim_root demos/imitator_data \
  --human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json \
  --sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json \
  --task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
  --human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --lerobot_camera zed2i \
  --pred_horizon 16 \
  --action_dim 16 \
  --obs_horizon 1 \
  --batch_size 256 \
  --num_workers 8 \
  --shard_size 25600