#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
UNISKILL_ROOT="${REPO_ROOT}/examples/baselines/uniskill"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to your interpreter (e.g. .venv/bin/python)." >&2
  exit 1
fi

# export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache}"
# export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export WANDB_MODE="${WANDB_MODE:-online}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

mkdir -p "${UNISKILL_ROOT}/outputs"

"${PYTHON_BIN}" examples/baselines/uniskill/diffusion/train_uniskill.py \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH:-timbrooks/instruct-pix2pix}" \
  --dataset_name "${DATASET_NAME:-lerobot}" \
  --human_root "${HUMAN_ROOT:-}" \
  --sim_root "${SIM_ROOT:-}" \
  --task_mapping_file "${TASK_MAPPING_FILE:-examples/baselines/lerobot_dataset/task_mapping.json}" \
  --human_dataset_file "${HUMAN_DATASET_FILE:-examples/baselines/lerobot_dataset/config/debug_configs/human_train_config.json}" \
  --sim_dataset_file "${SIM_DATASET_FILE:-examples/baselines/lerobot_dataset/config/debug_configs/sim_train_config.json}" \
  --human_task_description_file "${HUMAN_TASK_DESCRIPTION_FILE:-examples/baselines/lerobot_dataset/task_desc/human_desc.json}" \
  --sim_task_description_file "${SIM_TASK_DESCRIPTION_FILE:-examples/baselines/lerobot_dataset/task_desc/sim_desc.json}" \
  --state_type "${STATE_TYPE:-qpos}" \
  --cameras "${CAMERAS:-zed2i}" \
  --image_size ${IMAGE_SIZE:-144 144} \
  --num_video_frames "${NUM_VIDEO_FRAMES:-10}" \
  --video_backend "${VIDEO_BACKEND:-torchcodec}" \
  --fps "${FPS:-30}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-32}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-48}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-500}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-1000}" \
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT:-5}" \
  --validation_steps "${VALIDATION_STEPS:-0}" \
  --report_name "${REPORT_NAME:-uniskill}" \
  --report_to "${REPORT_TO:-wandb}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --output_dir "${OUTPUT_DIR:-${UNISKILL_ROOT}/outputs/uniskill}" \
  --pre_decode \
  "$@"
