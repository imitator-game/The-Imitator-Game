#!/usr/bin/env bash
set -euo pipefail
: "${ASTRA_RUNTIME_CONFIG:?Set ASTRA_RUNTIME_CONFIG to an external private runtime JSON}"
python -m examples.baselines.astra.eval_astra_imitator --runtime-config "$ASTRA_RUNTIME_CONFIG" "$@"
