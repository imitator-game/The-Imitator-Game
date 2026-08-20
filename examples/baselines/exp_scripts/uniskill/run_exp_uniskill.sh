#!/usr/bin/env bash
# UniSkill stage1: 45/30/15-task human-desc IDM pretraining, one GPU per tag.
# Requires HUMAN_ROOT/SIM_ROOT env vars.

set -euo pipefail

GPU_30="${GPU_30:-0}"
GPU_45="${GPU_45:-1}"
HUMAN_ROOT="${HUMAN_ROOT:?Set HUMAN_ROOT (e.g. demos/demo_data)}"
SIM_ROOT="${SIM_ROOT:?Set SIM_ROOT (e.g. demos/imitator_data)}"

run_task() {
  local gpu="$1" tag="$2" human_cfg="$3" sim_cfg="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python examples/baselines/uniskill/diffusion/train_uniskill.py \
    --pretrained_model_name_or_path timbrooks/instruct-pix2pix \
    --dataset_name lerobot \
    --human_root "$HUMAN_ROOT" \
    --sim_root "$SIM_ROOT" \
    --task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
    --human_dataset_file "$human_cfg" \
    --sim_dataset_file "$sim_cfg" \
    --human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
    --sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
    --state_type qpos \
    --cameras zed2i \
    --image_size 144 144 \
    --num_video_frames 10 \
    --video_backend torchcodec \
    --fps 30 \
    --train_batch_size 192 \
    --dataloader_num_workers 20 \
    --num_train_epochs 10 \
    --checkpointing_steps 1000 \
    --checkpoints_total_limit 3 \
    --validation_steps 0 \
    --report_name "uniskill_${tag}" \
    --report_to none \
    --mixed_precision bf16 \
    --output_dir "examples/baselines/uniskill/outputs/uniskill_${tag}" \
    --pre_decode \
    --resume_from_checkpoint latest
}

run_task "$GPU_30" 30 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_30.json examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_30.json &
PID_30=$!

run_task "$GPU_45" 45 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json &

wait "$PID_30"
wait