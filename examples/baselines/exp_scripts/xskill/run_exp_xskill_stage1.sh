#!/usr/bin/env bash
# XSkill stage1: pretrain the skill-transfer encoder, running 15/30/45 if selected.
# RUN_ONLY=15,30,45 (default all).

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

run_task() {
  local gpu="$1" tag="$2" human_cfg="$3" sim_cfg="$4"
  CUDA_VISIBLE_DEVICES="$gpu" HYDRA_FULL_ERROR=1 \
    python -m examples.baselines.xskill.scripts.stage1_pretrain_encoder \
    "base_dev_dir=examples/baselines/xskill/logs/stage1_${tag}" \
    num_workers=8 \
    persistent_workers=False \
    pin_memory=False \
    video_backend="pyav" \
    human_dataset_file="$human_cfg" \
    sim_dataset_file="$sim_cfg" \
    > "examples/baselines/xskill/logs/stage1_${tag}.log" 2>&1
}

gpu_for_tag() {
  case "$1" in
    15) echo 0 ;;
    30) echo 1 ;;
    45) echo 2 ;;
  esac
}

declare -A PIDS=()
for tag in ${RUN_ONLY}; do
  if should_run "$tag"; then
    run_task "$(gpu_for_tag "$tag")" "$tag" \
      "examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_${tag}.json" \
      "examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_${tag}.json" &
    PIDS["$tag"]=$!
  fi
done

for tag in ${RUN_ONLY}; do
  [[ -n "${PIDS[$tag]:-}" ]] && wait "${PIDS[$tag]}"
done