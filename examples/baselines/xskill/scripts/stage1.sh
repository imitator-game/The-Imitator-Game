#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/examples/baselines/xskill:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XSKILL_LOG_DIR="${ROOT_DIR}/examples/baselines/xskill/logs"
export WANDB_DIR="${WANDB_DIR:-${XSKILL_LOG_DIR}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/config}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DISABLED="${WANDB_DISABLED:-false}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${XSKILL_LOG_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
python3 -m examples.baselines.xskill.scripts.stage1_pretrain_encoder \
  "base_dev_dir=examples/baselines/xskill/logs" \
  num_workers="${STAGE1_NUM_WORKERS:-16}" \
  persistent_workers="${STAGE1_PERSISTENT_WORKERS:-false}" \
  pin_memory="${STAGE1_PIN_MEMORY:-false}" \
  video_backend="${STAGE1_VIDEO_BACKEND:-'pyav'}"
