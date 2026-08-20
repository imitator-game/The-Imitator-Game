#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IDM_CKPT_PATH="${IDM_CKPT_PATH:-examples/baselines/uniskill/outputs/uniskill/checkpoint-542000/idm.pth}"

exec "${SCRIPT_DIR}/run_train_cond_dp.sh" \
  --resume_from_checkpoint "examples/baselines/uniskill/outputs/cond_dp/checkpoint-55000" \
  "$@"
