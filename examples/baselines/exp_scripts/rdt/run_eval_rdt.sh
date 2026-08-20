#!/bin/bash
# RDT LoRA eval: evaluate 10-epoch checkpoints for each tag on seen/unseen splits.
# RUN_DIR_45/30/15 point to run dirs with checkpoints/epoch_{N}.pt.

set -euo pipefail

GPU_45="${GPU_45:-0}"
GPU_30="${GPU_30:-1}"
GPU_15="${GPU_15:-2}"
EVAL_TAGS=(${EVAL_TAGS:-45 30 15})
EVAL_SPLITS=(${EVAL_SPLITS:-seen unseen})
EVAL_CHECKPOINT_EPOCH="${EVAL_CHECKPOINT_EPOCH:-10}"

RUN_DIR_45="${RUN_DIR_45:-}"
RUN_DIR_30="${RUN_DIR_30:-}"
RUN_DIR_15="${RUN_DIR_15:-}"

RESULT_ROOT="${RESULT_ROOT:-runs/rdt_eval}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-10}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
REWARD_MODE="${REWARD_MODE:-dense}"
CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
SHADER="${SHADER:-rt-fast}"
MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU:-1}"
RDT_COMPUTE_DTW="${RDT_COMPUTE_DTW:-1}"
DTW_BAND_RATIO="${DTW_BAND_RATIO:-0.15}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-1}"
CONTROL_FREQUENCY="${CONTROL_FREQUENCY:-30.0}"
EVAL_LR_MIRROR="${EVAL_LR_MIRROR:-auto}"

PRETRAINED_PATH="${PRETRAINED_PATH:?Set PRETRAINED_PATH}"
VISION_ENCODER="${VISION_ENCODER:?Set VISION_ENCODER}"
TEXT_ENCODER="${TEXT_ENCODER:?Set TEXT_ENCODER}"

SIM_CONFIG_SEEN="examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json"
HUMAN_CONFIG_SEEN="examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json"
SIM_CONFIG_UNSEEN="examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_unseen.json"
HUMAN_CONFIG_UNSEEN="examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_unseen.json"
LANG_CACHE_SEEN="${LANG_CACHE_SEEN:-}"
LANG_CACHE_UNSEEN="${LANG_CACHE_UNSEEN:-}"

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

split_configs() {
  case "$1" in
    seen) echo "$SIM_CONFIG_SEEN $HUMAN_CONFIG_SEEN $LANG_CACHE_SEEN" ;;
    unseen) echo "$SIM_CONFIG_UNSEEN $HUMAN_CONFIG_UNSEEN $LANG_CACHE_UNSEEN" ;;
  esac
}

resolve_checkpoint() {
  local run_dir="$1"
  local epoch_idx=$((EVAL_CHECKPOINT_EPOCH - 1))
  local path="$run_dir/checkpoints/epoch_${epoch_idx}.pt"
  if [[ -f "$path" ]]; then
    echo "$path"
    return
  fi
  find "$run_dir/checkpoints" -maxdepth 1 -name '*.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -n1 | cut -d' ' -f2-
}

run_rdt_eval() {
  local gpu="$1" tag="$2" split="$3" checkpoint="$4"
  read -r sim_config human_config lang_cache <<< "$(split_configs "$split")"
  local out_name="rdt_lora_${tag}_${split}"
  local dtw_args=()
  [[ "${RDT_COMPUTE_DTW,,}" =~ ^(1|true|yes|on)$ ]] && dtw_args=(--compute_dtw --dtw_band_ratio "$DTW_BAND_RATIO")
  local video_args=()
  [[ "${CAPTURE_VIDEO,,}" =~ ^(1|true|yes|on)$ ]] || video_args=(--no-capture-video)

  local cmd=(
    python -m examples.baselines.rdt.parallel_eval_rdt
    --checkpoint_path "$checkpoint"
    --eval_module examples.baselines.rdt.eval_rdt_lora
    --output_name "$out_name"
    --num_eval_episodes "$NUM_EVAL_EPISODES"
    --num_eval_envs "$NUM_EVAL_ENVS"
    --sim_backend "$SIM_BACKEND"
    --control_mode "$CONTROL_MODE"
    --shader "$SHADER"
    --vision_encoder "$VISION_ENCODER"
    --text_encoder "$TEXT_ENCODER"
    --t5_version t5-v1_1-xxl
    --max_lang_len 1024
    --pretrained_path "$PRETRAINED_PATH"
    --lora_r 576
    --lora_alpha 1152
    --lora_dropout 0.05
    --lerobot_human_root demos/demo_data
    --lerobot_sim_root demos/imitator_data
    --lerobot_human_dataset_file "$human_config"
    --lerobot_sim_dataset_file "$sim_config"
    --lerobot_task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json
    --lerobot_human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json
    --lerobot_sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json
    --lerobot_state_type qpos
    --lerobot_image_size 224 224
    --num_gpus 1
    --gpu_ids "$gpu"
    --max_procs_per_gpu "$MAX_PROCS_PER_GPU"
    --max_retries 1
    --min_free_mem_gb 8
    --max_cpu_workers 1
  )
  [[ -n "$lang_cache" && -d "$lang_cache" ]] && cmd+=(--precomputed_vl_dir "$lang_cache" --expected_precomputed_vl_mode language_only)
  [[ -n "$lang_cache" && -d "$lang_cache" ]] || cmd+=(--allow_online_text_encoder)
  [[ ${#dtw_args[@]} -gt 0 ]] && cmd+=("${dtw_args[@]}")
  cmd+=(
    --
    --max-episode-steps "$MAX_EPISODE_STEPS"
    --reward-mode "$REWARD_MODE"
    --control-frequency "$CONTROL_FREQUENCY"
    --rdt-slot-mapping official
    --eval-lr-mirror "$EVAL_LR_MIRROR"
    --eval-lr-mirror-robot-pose false
    "${video_args[@]}"
  )
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
}

declare -A PIDS=()
for tag in "${EVAL_TAGS[@]}"; do
  (
    run_dir="$(run_dir_for_tag "$tag")"
    [[ -n "$run_dir" ]] || { echo "Set RUN_DIR_${tag}"; exit 1; }
    checkpoint="$(resolve_checkpoint "$run_dir")"
    [[ -n "$checkpoint" ]] || { echo "No checkpoint under $run_dir"; exit 1; }
    for split in "${EVAL_SPLITS[@]}"; do
      run_rdt_eval "$(gpu_for_tag "$tag")" "$tag" "$split" "$checkpoint"
    done
  ) &
  PIDS["$tag"]=$!
done

for tag in "${EVAL_TAGS[@]}"; do
  wait "${PIDS[$tag]}"
done