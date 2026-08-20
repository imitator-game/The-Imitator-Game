#!/bin/bash
# Diffusion Policy train: 45/30/15-task human-desc paired training, one GPU per tag.
# BACKBONE=<dinov2_vitl14|siglip2_so400m|videomae_large> default dinov2_vitl14

set -euo pipefail

GPU_45="${GPU_45:-0}"
GPU_30="${GPU_30:-1}"
GPU_15="${GPU_15:-2}"
BACKBONE="${BACKBONE:-dinov2_vitl14}"

run_dp() {
  local gpu="$1" tag="$2" human_dataset_file="$3" epochs="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -m examples.baselines.diffusion_policy.train_dp_imitator \
    --human-root demos/demo_data \
    --sim-root demos/imitator_data \
    --human-dataset-file "$human_dataset_file" \
    --sim-dataset-file "${human_dataset_file/human_/sim_}" \
    --task-mapping-file examples/baselines/lerobot_dataset/task_mapping.json \
    --human-task-description-file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
    --sim-task-description-file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
    --input-mode video_only \
    --task-encoder-type frozen_backbone \
    --frozen-backbone-type "$BACKBONE" \
    --frozen-backbone-num-frames 10 \
    --frozen-backbone-adapter-layers 1 \
    --frozen-backbone-seq-patches 32 \
    --control-mode pd_joint_pos \
    --batch-size 256 \
    --total-epochs "$epochs" \
    --lr 1e-4 \
    --num-dataload-workers 24 \
    --env-id TwoRobotPourCup-v1 \
    --max-episode-steps 500 \
    --unet-dims 352 704 1408 2816
}

run_dp "$GPU_45" 45 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json 10 &
run_dp "$GPU_30" 30 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_30.json 10 &
run_dp "$GPU_15" 15 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json 10 &

wait