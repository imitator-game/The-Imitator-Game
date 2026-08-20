#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${ROOT_DIR}"
source "${ROOT_DIR}/.venv/bin/activate"
export PYTHONPATH="${ROOT_DIR}/examples/baselines/uniskill/diffusion:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

# ── Paths (adjust to your setup) ─────────────────────────────────────────────
UNISKILL_LOG_DIR="${ROOT_DIR}/examples/baselines/uniskill/logs"
POLICY_TRAIN_DIR="${UNISKILL_LOG_DIR}/outputs_policy"          # stage2 training output dir
CHECKPOINT="${POLICY_TRAIN_DIR}/checkpoint-50000"               # accelerator checkpoint dir
IDM_CKPT="${UNISKILL_LOG_DIR}/outputs_idm/idm.pth"            # frozen IDM from stage1
EVAL_CONFIG="examples/baselines/lerobot_dataset/eval/sim_eval.txt"
OUTPUT_DIR="${UNISKILL_LOG_DIR}/eval_results"

# ── Data paths ────────────────────────────────────────────────────────────────
HUMAN_ROOT="demo_data"
SIM_ROOT=""
SIM_CONFIG="examples/baselines/lerobot_dataset/config/sim_config.json"
HUMAN_CONFIG="examples/baselines/lerobot_dataset/config/human_config.json"
TASK_MAPPING="examples/baselines/lerobot_dataset/task_mapping.json"
HUMAN_TASK_DESC="examples/baselines/lerobot_dataset/task_desc/human_desc.json"
SIM_TASK_DESC="examples/baselines/lerobot_dataset/task_desc/sim_desc.json"

# ── Run ───────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
python3 -m examples.baselines.uniskill.diffusion.eval_uniskill \
  --eval-config "${EVAL_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --idm-ckpt-path "${IDM_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --input-mode video_only \
  --human-root "${HUMAN_ROOT}" \
  --sim-root "${SIM_ROOT}" \
  --sim-config "${SIM_CONFIG}" \
  --human-config "${HUMAN_CONFIG}" \
  --task-mapping "${TASK_MAPPING}" \
  --human-task-desc "${HUMAN_TASK_DESC}" \
  --sim-task-desc "${SIM_TASK_DESC}" \
  --num-episodes 10 \
  --num-envs 1 \
  --max-episode-steps 500 \
  --sim-backend physx_cpu \
  --device cuda \
  --temporal-agg \
  --policy-pred-horizon 16 \
  --obs-horizon 2 \
  --vision-feature-dim 256 \
  --idm-feature-dim 128 \
  --num-diffusion-iters 100 \
  --resolution 112 \
  --idm-resolution 224
