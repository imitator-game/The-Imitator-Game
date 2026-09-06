#!/bin/bash
# GR00T train: 45/30/15-task human-desc training, one GPU per tag.
# GR00T_BASE_MODEL must point to a local GR00T-N1.6-3B checkpoint.

set -euo pipefail

GPU_45="${GPU_45:-0}"
GPU_30="${GPU_30:-1}"
GPU_15="${GPU_15:-2}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SIM_ROOT="demos/imitator_data"

run_gr00t() {
  local gpu="$1" tag="$2" human_dataset_file="$3" sim_dataset_file="$4" epochs="$5"
  local num_shards
  num_shards="$(python scripts/gr00t_num_shards_per_epoch.py \
    --human-config-path "$human_dataset_file" \
    --sim-config-path "$sim_dataset_file" \
    --dataset-path "$SIM_ROOT" \
    --lerobot-version v3 \
    --shard-size 1024 \
    --action-horizon 16)"
  CUDA_VISIBLE_DEVICES="$gpu" python -m gr00t.experiment.launch_finetune \
    --embodiment-tag NEW_EMBODIMENT \
    --human-config-path "$human_dataset_file" \
    --sim-config-path "$sim_dataset_file" \
    --lerobot-version v3 \
    --language-source human_desc \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 32 \
    --num-gpus 1 \
    --epoch-based-training \
    --num-epochs "$epochs" \
    --num-shards-per-epoch "$num_shards" \
    --save-epochs 1 \
    --save-total-limit 30 \
    --logging-steps 100 \
    --output-dir "runs/gr00t_${tag}" \
    --no-tune-visual \
    --no-tune-llm \
    --tune-top-llm-layers 0 \
    --tune-projector \
    --use-wandb \
    --base-model-path "$GR00T_BASE_MODEL" \
    --dataset-path "$SIM_ROOT"
}

python scripts/gr00t_prepare_stats.py \
  --human-config-path examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json \
  --sim-config-path examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json \
  --human-config-path examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_30.json \
  --sim-config-path examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_30.json \
  --human-config-path examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json \
  --sim-config-path examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_15.json \
  --dataset-path "$SIM_ROOT" \
  --lerobot-version v3 \
  --embodiment-tag NEW_EMBODIMENT

run_gr00t "$GPU_45" 45 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json 10 &
run_gr00t "$GPU_30" 30 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_30.json examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_30.json 10 &
run_gr00t "$GPU_15" 15 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_15.json 10 &

wait