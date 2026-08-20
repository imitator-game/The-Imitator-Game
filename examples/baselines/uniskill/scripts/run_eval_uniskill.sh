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

export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${REPO_ROOT}:${UNISKILL_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" -m examples.baselines.uniskill.diffusion.eval_uniskill \
  --eval-config "${EVAL_CONFIG:-examples/baselines/lerobot_dataset/eval/sim_eval.txt}" \
  --checkpoint "${CKPT_PATH:-${UNISKILL_ROOT}/outputs/cond_dp/checkpoint-225000}" \
  --idm-ckpt-path "${IDM_CKPT_PATH:-${UNISKILL_ROOT}/outputs/uniskill/checkpoint-542000/idm.pth}" \
  --output-dir "${OUTPUT_DIR:-${UNISKILL_ROOT}/eval_results}" \
  --input-mode video_only \
  --human-root "${HUMAN_ROOT:-demo_data}" \
  --sim-root "${SIM_ROOT:-}" \
  --sim-config "${SIM_CONFIG:-examples/baselines/lerobot_dataset/config/test_configs/sim_test_config.json}" \
  --human-config "${HUMAN_CONFIG:-examples/baselines/lerobot_dataset/config/test_configs/human_test_config.json}" \
  --task-mapping "${TASK_MAPPING:-examples/baselines/lerobot_dataset/task_mapping.json}" \
  --human-task-desc "${HUMAN_TASK_DESC:-examples/baselines/lerobot_dataset/task_desc/human_desc.json}" \
  --sim-task-desc "${SIM_TASK_DESC:-examples/baselines/lerobot_dataset/task_desc/sim_desc.json}" \
  --num-episodes "${NUM_EVAL_EPISODES:-2}" \
  --num-envs "${NUM_EVAL_ENVS:-1}" \
  --max-episode-steps 500 \
  --sim-backend "${SIM_BACKEND:-physx_cpu}" \
  --shader "${SHADER:-rt-fast}" \
  --device "${DEVICE:-cuda}" \
  --image-size 144 144 \
  --temporal-agg \
  --policy-pred-horizon 16 \
  --obs-horizon 2 \
  --vision-feature-dim 256 \
  --idm-feature-dim 128 \
  --num-diffusion-iters 100 \
  --resolution 112 \
  --idm-resolution 224 \
  "$@"
