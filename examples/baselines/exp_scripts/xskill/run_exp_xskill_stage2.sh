#!/usr/bin/env bash
# XSkill stage2: skill transfer on top of stage1 pretrained encoders.
# PRETRAIN_PATH_15/30/45 override the default stage1 dirs; RUN_ONLY=15,30,45 (default all).

set -euo pipefail

mkdir -p examples/baselines/xskill/logs
RUN_ONLY="${RUN_ONLY:-15 30 45}"

should_run() {
  local tag="$1"
  for selected in ${RUN_ONLY}; do
    [[ "$selected" == "$tag" ]] && return 0
  done
  return 1
}

PRETRAIN_PATH_15="${PRETRAIN_PATH_15:-examples/baselines/xskill/logs/stage1_15/xskill/experiment/pretrain}"
PRETRAIN_PATH_30="${PRETRAIN_PATH_30:-examples/baselines/xskill/logs/stage1_30/xskill/experiment/pretrain}"
PRETRAIN_PATH_45="${PRETRAIN_PATH_45:-examples/baselines/xskill/logs/stage1_45/xskill/experiment/pretrain}"

run_task() {
  local gpu="$1" tag="$2" pretrain_path="$3"
  CUDA_VISIBLE_DEVICES="$gpu" HYDRA_FULL_ERROR=1 \
    python examples/baselines/xskill/scripts/stage2_skill_transfer.py \
    "base_dev_dir=examples/baselines/xskill/logs/stage2_${tag}_sim" \
    "pretrain_path=${pretrain_path}" \
    "pretrain_ckpt=04" \
    "human_dataset_file=examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_${tag}.json" \
    "sim_dataset_file=examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_${tag}.json" \
    "pre_decode=true" \
    > "examples/baselines/xskill/logs/stage2_${tag}_sim.log" 2>&1
}

gpu_for_tag() {
  case "$1" in
    15) echo 0 ;;
    30) echo 1 ;;
    45) echo 2 ;;
  esac
}

pretrain_path_for_tag() {
  case "$1" in
    15) echo "$PRETRAIN_PATH_15" ;;
    30) echo "$PRETRAIN_PATH_30" ;;
    45) echo "$PRETRAIN_PATH_45" ;;
  esac
}

declare -A PIDS=()
for tag in ${RUN_ONLY}; do
  if should_run "$tag"; then
    run_task "$(gpu_for_tag "$tag")" "$tag" "$(pretrain_path_for_tag "$tag")" &
    PIDS["$tag"]=$!
  fi
done

for tag in ${RUN_ONLY}; do
  [[ -n "${PIDS[$tag]:-}" ]] && wait "${PIDS[$tag]}"
done