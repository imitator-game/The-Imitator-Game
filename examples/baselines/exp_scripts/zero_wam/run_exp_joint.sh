#!/usr/bin/env bash
set -euo pipefail
exec torchrun \
  --nnodes="${NNODES:-1}" --nproc-per-node="${GPUS_PER_NODE:-4}" \
  --node-rank="${NODE_RANK:-0}" \
  --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port="${MASTER_PORT:-29500}" \
  --module examples.baselines.zero_wam.train_zero_wam_imitator \
  --train-scope joint "$@"
