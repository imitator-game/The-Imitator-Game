#!/bin/bash
# GR00T eval: evaluate latest checkpoints for each tag on seen/unseen splits.
# CHECKPOINT_DIR_45/30/15 override the default runs/gr00t_{45,30,15} dirs.

set -euo pipefail

GPU_45="${GPU_45:-0}"
GPU_30="${GPU_30:-1}"
GPU_15="${GPU_15:-2}"
EVAL_TAGS=(${EVAL_TAGS:-45 30 15})
EVAL_SPLITS=(${EVAL_SPLITS:-seen unseen})
SIM_ROOT="demos/imitator_data"
RESULT_ROOT="${RESULT_ROOT:-runs/gr00t_eval}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-5}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
REWARD_MODE="${REWARD_MODE:-dense}"
CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
OBS_MODE="${OBS_MODE:-rgb}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
SHADER="${SHADER:-rt-fast}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-1}"
COMPUTE_DTW="${COMPUTE_DTW:-1}"
DTW_BAND_RATIO="${DTW_BAND_RATIO:-0.15}"
DTW_ACTION_KEY="${DTW_ACTION_KEY:-action.qpos_gripper_actions}"
EVAL_LR_MIRROR="${EVAL_LR_MIRROR:-auto}"

CHECKPOINT_DIR_45="${CHECKPOINT_DIR_45:-}"
CHECKPOINT_DIR_30="${CHECKPOINT_DIR_30:-}"
CHECKPOINT_DIR_15="${CHECKPOINT_DIR_15:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"

SIM_TEST_CONFIG_SEEN="examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json"
HUMAN_TEST_CONFIG_SEEN="examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json"
SIM_TEST_CONFIG_UNSEEN="examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_unseen.json"
HUMAN_TEST_CONFIG_UNSEEN="examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_unseen.json"

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

split_configs() {
  case "$1" in
    seen) echo "$SIM_TEST_CONFIG_SEEN $HUMAN_TEST_CONFIG_SEEN" ;;
    unseen) echo "$SIM_TEST_CONFIG_UNSEEN $HUMAN_TEST_CONFIG_UNSEEN" ;;
  esac
}

latest_checkpoint() {
  local run_dir="$1"
  local latest=""
  local entry
  while IFS= read -r entry; do
    [[ -d "$entry/checkpoint" ]] && latest="$entry"
  done < <(find "$run_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort)
  if [[ -z "$latest" ]]; then
    find "$run_dir" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n1
  else
    echo "$latest"
  fi
}

run_gr00t_eval() {
  local gpu="$1" tag="$2" split="$3" checkpoint="$4"
  read -r sim_config human_config <<< "$(split_configs "$split")"
  local out_dir="$RESULT_ROOT/gr00t_${tag}_${split}_$(date +%Y%m%d_%H%M%S)"
  local dtw_args=()
  [[ "${COMPUTE_DTW,,}" =~ ^(1|true|yes|on)$ ]] && dtw_args=(--compute_dtw --dtw_band_ratio "$DTW_BAND_RATIO" --sim_root "$SIM_ROOT" --dtw_action_key "$DTW_ACTION_KEY")
  local video_args=()
  [[ "${CAPTURE_VIDEO,,}" =~ ^(1|true|yes|on)$ ]] && video_args=(--capture_video)
  local processor_args=()
  [[ -n "$PROCESSOR_PATH" ]] && processor_args=(--processor_path "$PROCESSOR_PATH")

  CUDA_VISIBLE_DEVICES="$gpu" python -m gr00t.eval.parallel_eval_imitator \
    --checkpoint_path "$checkpoint" \
    "${processor_args[@]}" \
    --result_root "$out_dir" \
    --sim_dataset_file "$sim_config" \
    --human_dataset_file "$human_config" \
    --embodiment_tag NEW_EMBODIMENT \
    --language_source human_desc \
    --task_mapping_path examples/baselines/lerobot_dataset/task_mapping.json \
    --human_desc_path examples/baselines/lerobot_dataset/task_desc/human_desc.json \
    --sim_desc_path examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
    --reward_mode "$REWARD_MODE" \
    --num_eval_episodes "$NUM_EVAL_EPISODES" \
    --num_eval_envs "$NUM_EVAL_ENVS" \
    --max_episode_steps "$MAX_EPISODE_STEPS" \
    --control_mode "$CONTROL_MODE" \
    --obs_mode "$OBS_MODE" \
    --sim_backend "$SIM_BACKEND" \
    --shader "$SHADER" \
    --dtype bf16 \
    --eval_lr_mirror "$EVAL_LR_MIRROR" \
    --num_gpus 1 \
    --gpu_ids "$gpu" \
    --max_procs_per_gpu 1 \
    --min_free_mem_gb 20 \
    "${video_args[@]}" \
    "${dtw_args[@]}"
}

declare -A PIDS=()
for tag in "${EVAL_TAGS[@]}"; do
  ( 
    ckpt_dir="$(checkpoint_dir_for_tag "$tag")"
    [[ -n "$ckpt_dir" ]] || ckpt_dir="$(find runs -maxdepth 1 -type d -name "gr00t_${tag}_h200*" | sort | tail -n1)"
    [[ -n "$ckpt_dir" ]] || { echo "No run dir for tag $tag; set CHECKPOINT_DIR_${tag}"; exit 1; }
    checkpoint="$(latest_checkpoint "$ckpt_dir")"
    [[ -n "$checkpoint" ]] || { echo "No checkpoint under $ckpt_dir"; exit 1; }
    for split in "${EVAL_SPLITS[@]}"; do
      run_gr00t_eval "$(gpu_for_tag "$tag")" "$tag" "$split" "$checkpoint"
    done
  ) &
  PIDS["$tag"]=$!
done

for tag in "${EVAL_TAGS[@]}"; do
  wait "${PIDS[$tag]}"
done