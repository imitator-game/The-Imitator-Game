#!/usr/bin/env bash
# UniSkill eval: one JOB per checkpoint, format:
#   name|checkpoint|idm_ckpt|eval_config|sim_config|human_config|output_dir|gpu_ids
# Requires HUMAN_ROOT/SIM_ROOT env vars.

set -euo pipefail

HUMAN_ROOT="${HUMAN_ROOT:?Set HUMAN_ROOT (e.g. demos/demo_data)}"
SIM_ROOT="${SIM_ROOT:?Set SIM_ROOT (e.g. demos/imitator_data)}"
mkdir -p examples/baselines/uniskill/logs

PARALLEL_SCHEDULER="${PARALLEL_SCHEDULER:-examples/baselines/uniskill/diffusion/parallel_eval_uniskill.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-examples/baselines/uniskill/diffusion/eval_uniskill.py}"
OUTPUT_DIR="${OUTPUT_DIR:-examples/baselines/uniskill/logs/eval_results}"

INPUT_MODE="${INPUT_MODE:-video_only}"
IDM_CKPT_45="${IDM_CKPT_45:?Set IDM_CKPT_45}"
NUM_EPISODES="${NUM_EPISODES:-10}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-500}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
OBS_MODE="${OBS_MODE:-rgb}"
SHADER="${SHADER:-rt-fast}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU:-1}"
COMPUTE_DTW="${COMPUTE_DTW:-0}"

JOBS=(
  "uniskill_45_seen|examples/baselines/uniskill/outputs/cond_dp_45/checkpoint-237460|$IDM_CKPT_45|examples/baselines/lerobot_dataset/eval/exp_list/seen_5tasks_env_list.txt|examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json|examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json|$OUTPUT_DIR/cond_dp_45_seen|0"
  "uniskill_45_unseen|examples/baselines/uniskill/outputs/cond_dp_45/checkpoint-237460|$IDM_CKPT_45|examples/baselines/lerobot_dataset/eval/exp_list/unseen_5tasks_env_list.txt|examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_unseen.json|examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_unseen.json|$OUTPUT_DIR/cond_dp_45_unseen|1"
)

declare -A PIDS=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r name checkpoint idm_ckpt eval_config sim_config human_config output_dir gpu_ids_csv <<< "$spec"
  mkdir -p "$output_dir"
  dtw_args=()
  [[ "${COMPUTE_DTW,,}" =~ ^(1|true|yes|on)$ ]] && dtw_args=(--compute-dtw) || dtw_args=(--no-compute-dtw)
  python3 -u "$PARALLEL_SCHEDULER" \
    --eval-script "$EVAL_SCRIPT" \
    --eval-config "$eval_config" \
    --checkpoint "$checkpoint" \
    --idm-ckpt-path "$idm_ckpt" \
    --output-dir "$output_dir" \
    --input-mode "$INPUT_MODE" \
    --human-root "$HUMAN_ROOT" \
    --sim-root "$SIM_ROOT" \
    --sim-config "$sim_config" \
    --human-config "$human_config" \
    --task-mapping examples/baselines/lerobot_dataset/task_mapping.json \
    --human-task-desc examples/baselines/lerobot_dataset/task_desc/human_desc.json \
    --sim-task-desc examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
    --num-episodes "$NUM_EPISODES" \
    --num-envs "$NUM_ENVS" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --sim-backend "$SIM_BACKEND" \
    --control-mode "$CONTROL_MODE" \
    --obs-mode "$OBS_MODE" \
    --shader "$SHADER" \
    --action-dim 16 \
    --obs-dim 18 \
    --obs-horizon 2 \
    --policy-pred-horizon 16 \
    --vision-feature-dim 256 \
    --idm-feature-dim 128 \
    --num-diffusion-iters 100 \
    --resolution 112 \
    --idm-resolution 224 \
    --image-size 224 224 \
    --cameras zed2i \
    --num-video-frames 10 \
    --state-type qpos \
    --vocab-size 32000 \
    --max-text-len 500 \
    --task-seq-len 10 \
    --mano-dim 14 \
    --num-gpus 1 \
    --gpu-ids "$gpu_ids_csv" \
    --max-procs-per-gpu "$MAX_PROCS_PER_GPU" \
    --min-free-mem-gb 16 \
    --max-retries 1 \
    --status-interval-sec 60 \
    --launch-interval-sec 2 \
    --eval_lr_mirror auto \
    --eval_lr_mirror_robot_pose false \
    "${dtw_args[@]}" \
    --dtw-band-ratio 0.15 > "examples/baselines/uniskill/logs/${name}.log" 2>&1 &
  PIDS["$name"]=$!
done

EXIT_CODE=0
for name in "${!PIDS[@]}"; do
  wait "${PIDS[$name]}" || EXIT_CODE=1
done
exit "$EXIT_CODE"