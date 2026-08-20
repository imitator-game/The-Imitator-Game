#!/bin/bash
# Pi05 JAX eval: evaluate checkpoints for each tag on seen/unseen splits.
# RUN_DIR_45/30/15 override the default runs/pi_lerobot_jax_{45,30,15}_h200 dirs.

set -euo pipefail

GPU_45="${GPU_45:-0}"
GPU_30="${GPU_30:-1}"
GPU_15="${GPU_15:-2}"
EVAL_TAGS=(${EVAL_TAGS:-45 30 15})
EVAL_SPLITS=(${EVAL_SPLITS:-seen unseen})
EVAL_CHECKPOINT_EPOCH="${EVAL_CHECKPOINT_EPOCH:-10}"
STEPS_PER_EPOCH_45="${STEPS_PER_EPOCH_45:-189968}"
STEPS_PER_EPOCH_30="${STEPS_PER_EPOCH_30:-132752}"
STEPS_PER_EPOCH_15="${STEPS_PER_EPOCH_15:-64607}"

RUN_DIR_45="${RUN_DIR_45:-runs/pi_lerobot_jax_45_h200}"
RUN_DIR_30="${RUN_DIR_30:-runs/pi_lerobot_jax_30_h200}"
RUN_DIR_15="${RUN_DIR_15:-runs/pi_lerobot_jax_15_h200}"

RESULT_ROOT="${RESULT_ROOT:-runs/pi_eval}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
REWARD_MODE="${REWARD_MODE:-dense}"
CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
COMPUTE_DTW="${COMPUTE_DTW:-1}"
DTW_BAND_RATIO="${DTW_BAND_RATIO:-0.15}"

SIM_TEST_CONFIG_SEEN="examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json"
HUMAN_TEST_CONFIG_SEEN="examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json"
SIM_TEST_CONFIG_UNSEEN="examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_unseen.json"
HUMAN_TEST_CONFIG_UNSEEN="examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_unseen.json"

run_dir_for_tag() {
  case "$1" in
    45) echo "$RUN_DIR_45" ;;
    30) echo "$RUN_DIR_30" ;;
    15) echo "$RUN_DIR_15" ;;
  esac
}

gpu_for_tag() {
  case "$1" in
    45) echo "$GPU_45" ;;
    30) echo "$GPU_30" ;;
    15) echo "$GPU_15" ;;
  esac
}

steps_per_epoch_for_tag() {
  case "$1" in
    45) echo "$STEPS_PER_EPOCH_45" ;;
    30) echo "$STEPS_PER_EPOCH_30" ;;
    15) echo "$STEPS_PER_EPOCH_15" ;;
  esac
}

split_configs() {
  case "$1" in
    seen) echo "$SIM_TEST_CONFIG_SEEN $HUMAN_TEST_CONFIG_SEEN" ;;
    unseen) echo "$SIM_TEST_CONFIG_UNSEEN $HUMAN_TEST_CONFIG_UNSEEN" ;;
  esac
}

latest_checkpoint_step() {
  find "$1" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; 2>/dev/null \
    | awk '/^[0-9]+$/ { print }' | sort -n | tail -n1
}

run_pi_eval() {
  local gpu="$1" tag="$2" split="$3" run_dir="$4" checkpoint_step="$5"
  read -r sim_config human_config <<< "$(split_configs "$split")"
  local out_dir="$RESULT_ROOT/pi_jax_${tag}_step${checkpoint_step}_${split}_$(date +%Y%m%d_%H%M%S)"
  local dtw_args=()
  [[ "${COMPUTE_DTW,,}" =~ ^(1|true|yes|on)$ ]] && dtw_args=(--compute_dtw --dtw_band_ratio "$DTW_BAND_RATIO")

  CUDA_VISIBLE_DEVICES="$gpu" python -m examples.baselines.pi.parallel_eval_pi_lerobot_jax \
    --checkpoint_path "$run_dir" \
    --checkpoint_step "$checkpoint_step" \
    --result_root "$out_dir" \
    --sim_dataset_file "$sim_config" \
    --sim_state_type qpos \
    --human_dataset_file "$human_config" \
    --human_task_desc_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
    --human_root demos/demo_data \
    --sim_root demos/imitator_data \
    --processor_name_or_path "$PALIGEMMA_PROCESSOR" \
    --processor_local_files_only \
    --l3_eval_l_level L0 \
    --eval_lr_mirror auto \
    --eval_lr_mirror_robot_pose false \
    "${dtw_args[@]}" \
    --reward_mode "$REWARD_MODE" \
    --num_eval_episodes "$NUM_EVAL_EPISODES" \
    --num_eval_envs "$NUM_EVAL_ENVS" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --control_mode "$CONTROL_MODE" \
    --capture_video \
    --num_gpus 1 \
    --gpu_ids "$gpu" \
    --max_procs_per_gpu 1 \
    --pi05 \
    --skip_masked_cameras \
    --use_prefix_kv_cache
}

declare -A PIDS=()
for tag in "${EVAL_TAGS[@]}"; do
  ( 
    run_dir="$(run_dir_for_tag "$tag")"
    [[ -n "$run_dir" ]] || { echo "Set RUN_DIR_${tag}"; exit 1; }
    if [[ "${EVAL_CHECKPOINT_EPOCH,,}" == "latest" ]]; then
      checkpoint_step="$(latest_checkpoint_step "$run_dir")"
    else
      checkpoint_step=$(( $(steps_per_epoch_for_tag "$tag") * EVAL_CHECKPOINT_EPOCH ))
    fi
    for split in "${EVAL_SPLITS[@]}"; do
      run_pi_eval "$(gpu_for_tag "$tag")" "$tag" "$split" "$run_dir" "$checkpoint_step"
    done
  ) &
  PIDS["$tag"]=$!
done

for tag in "${EVAL_TAGS[@]}"; do
  wait "${PIDS[$tag]}"
done