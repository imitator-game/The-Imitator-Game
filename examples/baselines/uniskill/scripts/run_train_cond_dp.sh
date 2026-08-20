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

if [[ -z "${IDM_CKPT_PATH:-}" ]]; then
  echo "Please set IDM_CKPT_PATH to your IDM checkpoint path." >&2
  echo "Example: IDM_CKPT_PATH=examples/baselines/uniskill/outputs/checkpoint-15000/idm.pth $0" >&2
  exit 1
fi

if [[ ! -f "${IDM_CKPT_PATH}" ]]; then
  echo "IDM checkpoint not found: ${IDM_CKPT_PATH}" >&2
  exit 1
fi

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_home}"
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "${UNISKILL_ROOT}/outputs"

"${PYTHON_BIN}" examples/baselines/uniskill/diffusion/train_cond_dp.py \
  --idm_ckpt_path "${IDM_CKPT_PATH}" \
  --human_root "${HUMAN_ROOT:-demo_data}" \
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
  --train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-10000}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-1000}" \
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT:-5}" \
  --vis_interval "${VIS_INTERVAL:-1000}" \
  --report_to "${REPORT_TO:-wandb}" \
  --report_name "${REPORT_NAME:-uniskill_policy}" \
  --mixed_precision "${MIXED_PRECISION:-fp16}" \
  --output_dir "${OUTPUT_DIR:-${UNISKILL_ROOT}/outputs/cond_dp}" \
  "$@"
