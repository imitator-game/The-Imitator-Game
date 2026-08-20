#!/bin/bash
# OpenVLA eval: evaluate checkpoints for 45/30/15 in parallel.
# CHECKPOINT_DIR_45/30/15 point to dirs containing the model (OpenVLA run/final).

set -euo pipefail

GPU_45="${GPU_45:-0}"
GPU_30="${GPU_30:-1}"
GPU_15="${GPU_15:-2}"
EVAL_TAGS=(${EVAL_TAGS:-45 30 15})

CHECKPOINT_DIR_45="${CHECKPOINT_DIR_45:-}"
CHECKPOINT_DIR_30="${CHECKPOINT_DIR_30:-}"
CHECKPOINT_DIR_15="${CHECKPOINT_DIR_15:-}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}"
EVAL_CONFIG="${EVAL_CONFIG:-examples/baselines/lerobot_dataset/eval/exp_list/seen_plus_unseen_10tasks_env_list.txt}"
HUMAN_DATASET_FILE="examples/baselines/lerobot_dataset/config/exp_configs/human_eval_config_seen_plus_unseen_10tasks.json"
SIM_DATASET_FILE="${SIM_DATASET_FILE:-examples/baselines/lerobot_dataset/config/exp_configs/sim_eval_config_seen_plus_unseen_10tasks.json}"
RESULT_ROOT="${RESULT_ROOT:-runs/openvla_eval}"/"$(date +%Y%m%d_%H%M%S)"

NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-5}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-600}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-1}"

checkpoint_dir_for_tag() {
  case "$1" in
    45) echo "$CHECKPOINT_DIR_45" ;;
    30) echo "$CHECKPOINT_DIR_30" ;;
    15) echo "$CHECKPOINT_DIR_15" ;;
  esac
}

gpu_for_tag() {
  case "$1" in
    45) echo "$GPU_45" ;;
    30) echo "$GPU_30" ;;
    15) echo "$GPU_15" ;;
  esac
}

run_openvla_eval() {
  local gpu="$1" tag="$2" checkpoint_dir="$3" out_dir="$4"
  local video_args=()
  [[ "${CAPTURE_VIDEO,,}" =~ ^(1|true|yes|on)$ ]] && video_args=(--capture_video)
  CUDA_VISIBLE_DEVICES="$gpu" python -m examples.baselines.openvla_oft.eval_openvla_batch \
    --eval-config "$EVAL_CONFIG" \
    --output-dir "$out_dir" \
    --checkpoint_dir "$checkpoint_dir" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --use_lerobot \
    --human_root demos/demo_data \
    --sim_root demos/imitator_data \
    --human_dataset_file "$HUMAN_DATASET_FILE" \
    --sim_dataset_file "$SIM_DATASET_FILE" \
    --task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
    --human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
    --sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
    --lerobot_camera zed2i \
    --eval_camera zed2i \
    --num_eval_episodes "$NUM_EVAL_EPISODES" \
    --num_eval_envs "$NUM_EVAL_ENVS" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --sim_backend physx_cpu \
    --shader rt-fast \
    --action_mode l1_regression \
    --action_dim 16 \
    --use_proprio \
    --proprio_fusion_mode output \
    --eval_lr_mirror auto \
    --eval_lr_mirror_robot_pose false \
    --control_mode pd_joint_pos \
    "${video_args[@]}"
}

declare -A PIDS=()
for tag in "${EVAL_TAGS[@]}"; do
  ckpt_dir="$(checkpoint_dir_for_tag "$tag")"
  [[ -n "$ckpt_dir" ]] || { echo "Set CHECKPOINT_DIR_${tag}"; exit 1; }
  (
    run_openvla_eval "$(gpu_for_tag "$tag")" "$tag" "$ckpt_dir" "$RESULT_ROOT/openvla_${tag}"
  ) &
  PIDS["$tag"]=$!
done

for tag in "${EVAL_TAGS[@]}"; do
  wait "${PIDS[$tag]}"
done