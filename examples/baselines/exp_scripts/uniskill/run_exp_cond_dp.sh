#!/usr/bin/env bash
# UniSkill stage2 (Cond-DP): 45/30/15-task skill transfer on top of stage1 IDMs.
# IDM_CKPT_45/30/15 point to the stage1 idm.pth checkpoints.
# Requires HUMAN_ROOT/SIM_ROOT env vars.

set -euo pipefail

GPU_15="${GPU_15:-0}"
GPU_30="${GPU_30:-1}"
GPU_45="${GPU_45:-2}"
HUMAN_ROOT="${HUMAN_ROOT:?Set HUMAN_ROOT (e.g. demos/demo_data)}"
SIM_ROOT="${SIM_ROOT:?Set SIM_ROOT (e.g. demos/imitator_data)}"

IDM_CKPT_15="${IDM_CKPT_15:?Set IDM_CKPT_15}"
IDM_CKPT_30="${IDM_CKPT_30:?Set IDM_CKPT_30}"
IDM_CKPT_45="${IDM_CKPT_45:?Set IDM_CKPT_45}"

PREDECODE_CACHE_DIR="${PREDECODE_CACHE_DIR:-feature_cache/uniskill_cache/human_video_predecode}"
mkdir -p "$PREDECODE_CACHE_DIR"

run_task() {
  local gpu="$1" tag="$2" idm_ckpt="$3" human_cfg="$4" sim_cfg="$5"
  CUDA_VISIBLE_DEVICES="$gpu" python examples/baselines/uniskill/diffusion/train_cond_dp.py \
    --idm_ckpt_path "$idm_ckpt" \
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
    --train_batch_size 128 \
    --dataloader_num_workers 20 \
    --num_train_epochs 10 \
    --checkpointing_steps 1000 \
    --checkpoints_total_limit 3 \
    --vis_interval 1000 \
    --report_to none \
    --report_name "cond_dp_${tag}" \
    --mixed_precision bf16 \
    --output_dir "examples/baselines/uniskill/outputs/cond_dp_${tag}" \
    --pre_decode \
    --pre_decode_cache_dir "$PREDECODE_CACHE_DIR" \
    --pre_decode_num_workers 20
}

run_task "$GPU_15" 15 "$IDM_CKPT_15" examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_15.json &
run_task "$GPU_30" 30 "$IDM_CKPT_30" examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_30.json examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_30.json &
run_task "$GPU_45" 45 "$IDM_CKPT_45" examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json &

wait