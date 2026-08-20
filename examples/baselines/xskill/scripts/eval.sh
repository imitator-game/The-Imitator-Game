#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

SERVER_HOME="${SERVER_HOME:-}"
IMITATOR_ROOT="${IMITATOR_ROOT:-}"
MANISKILL_ROOT="${MANISKILL_ROOT:-}"

cd "${ROOT_DIR}"
export CONDA_PREFIX="${CONDA_PREFIX:-}"
if [[ -n "${CONDA_PREFIX}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  eval "$(${CONDA_PREFIX}/bin/conda shell.bash hook)"
  conda activate base
fi
source "${ROOT_DIR}/.venv/bin/activate"
export PYTHONPATH="${ROOT_DIR}/examples/baselines/xskill:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
# export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HOME="${HOME:-}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-${MANISKILL_ROOT}}"

# ── Paths (adjust to your setup) ─────────────────────────────────────────────
XSKILL_LOG_DIR="${ROOT_DIR}/examples/baselines/xskill/logs"
TRAIN_DIR="${XSKILL_LOG_DIR}/stage2_15_sim/xskill/experiment/transfer"    # stage2 training output dir
CHECKPOINT="${TRAIN_DIR}/ckpt_9.pt"                       # stage2 checkpoint
HYDRA_CONFIG="${TRAIN_DIR}/hydra_config.yaml"               # saved during training
EVAL_CONFIG="examples/baselines/lerobot_dataset/eval/exp_list/seen_5tasks_env_list.txt"
OUTPUT_DIR="${TRAIN_DIR}/eval_result_seen"

LOG_FILE="${XSKILL_LOG_DIR}/eval_xskill_$(date +%Y%m%d_%H%M%S).log"



exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================================"
echo "Logging to: ${LOG_FILE}"
echo "Start time: $(date)"
echo "ROOT_DIR=${ROOT_DIR}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "HYDRA_CONFIG=${HYDRA_CONFIG}"
echo "EVAL_CONFIG=${EVAL_CONFIG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "============================================================"

# ── Data paths ────────────────────────────────────────────────────────────────
HUMAN_ROOT="${HUMAN_ROOT:-}"
SIM_ROOT="${SIM_ROOT:-}"
SIM_CONFIG="examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json"
HUMAN_CONFIG="examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json"
TASK_MAPPING="examples/baselines/lerobot_dataset/task_mapping.json"
HUMAN_TASK_DESC="examples/baselines/lerobot_dataset/task_desc/human_desc.json"
SIM_TASK_DESC="examples/baselines/lerobot_dataset/task_desc/sim_desc.json"

# ── Run ───────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python3 examples/baselines/xskill/scripts/eval_xskill.py \
  --eval-config "${EVAL_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --hydra-config "${HYDRA_CONFIG}" \
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
  --compute-dtw \
  --dtw-band-ratio 0.15
