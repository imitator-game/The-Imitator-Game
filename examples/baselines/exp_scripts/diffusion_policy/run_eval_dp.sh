#!/bin/bash
# Diffusion Policy eval on seen/unseen splits, one GPU per checkpoint.
# RUN_DIR_45/30/15 point to run dirs containing checkpoints/final_model.pt.
# BACKBONE=<dinov2_vitl14|siglip2_so400m|videomae_large> default dinov2_vitl14

set -euo pipefail

GPU_45="${GPU_45:-0}"
GPU_30="${GPU_30:-1}"
GPU_15="${GPU_15:-2}"
EVAL_TAGS=(${EVAL_TAGS:-45 30 15})
BACKBONE="${BACKBONE:-dinov2_vitl14}"
EVAL_CONFIG="${EVAL_CONFIG:-examples/baselines/lerobot_dataset/eval/exp_list/seen_plus_unseen_10tasks_env_list.txt}"
HUMAN_CONFIG="examples/baselines/lerobot_dataset/config/exp_configs/human_eval_config_seen_plus_unseen_10tasks.json"
SIM_CONFIG="${SIM_CONFIG:-examples/baselines/lerobot_dataset/config/exp_configs/sim_eval_config_seen_plus_unseen_10tasks.json}"
RESULT_ROOT="${RESULT_ROOT:-runs/dp_eval}"/"$(date +%Y%m%d_%H%M%S)"

RUN_DIR_45="${RUN_DIR_45:-}"
RUN_DIR_30="${RUN_DIR_30:-}"
RUN_DIR_15="${RUN_DIR_15:-}"

run_dp_eval() {
  local gpu="$1" tag="$2" run_dir="$3" out_dir="$4"
  local ckpt="$run_dir/checkpoints/final_model.pt"
  CUDA_VISIBLE_DEVICES="$gpu" python -m examples.baselines.diffusion_policy.parallel_eval_dp \
    --eval-config "$EVAL_CONFIG" \
    --checkpoint "$ckpt" \
    --output-dir "$out_dir" \
    --input-mode video_only \
    --task-encoder-type frozen_backbone \
    --frozen-backbone-type "$BACKBONE" \
    --frozen-backbone-num-frames 10 \
    --action-dim 16 \
    --state-dim 18 \
    --obs-horizon 1 \
    --pred-horizon 16 \
    --obs-latent-dim 256 \
    --task-latent-dim 256 \
    --num-diffusion-iters 100 \
    --num-episodes 10 \
    --num-envs 1 \
    --max-episode-steps 500 \
    --control-mode pd_joint_pos \
    --obs-mode rgb \
    --sim-backend physx_cpu \
    --shader rt-fast \
    --use-ddim \
    --num-ddim-steps 10 \
    --human-root demos/demo_data \
    --sim-root demos/imitator_data \
    --human-config "$HUMAN_CONFIG" \
    --sim-config "$SIM_CONFIG" \
    --task-mapping examples/baselines/lerobot_dataset/task_mapping.json \
    --human-task-desc examples/baselines/lerobot_dataset/task_desc/human_desc.json \
    --sim-task-desc examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
    --num-gpus 1 \
    --gpu-ids "$gpu" \
    --max-procs-per-gpu 1
}

run_dir_for_tag() {
  case "$1" in
    45) echo "$RUN_DIR_45" ;;
    30) echo "$RUN_DIR_30" ;;
    15) echo "$RUN_DIR_15" ;;
    *) echo "" ;;
  esac
}

gpu_for_tag() {
  case "$1" in
    45) echo "$GPU_45" ;;
    30) echo "$GPU_30" ;;
    15) echo "$GPU_15" ;;
  esac
}

declare -A PIDS=()
for tag in "${EVAL_TAGS[@]}"; do
  run_dir="$(run_dir_for_tag "$tag")"
  [[ -n "$run_dir" ]] || { echo "Set RUN_DIR_${tag}"; exit 1; }
  (
    run_dp_eval "$(gpu_for_tag "$tag")" "$tag" "$run_dir" "$RESULT_ROOT/dp_${tag}"
  ) &
  PIDS["$tag"]=$!
done

for tag in "${EVAL_TAGS[@]}"; do
  wait "${PIDS[$tag]}"
done