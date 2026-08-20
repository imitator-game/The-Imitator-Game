#!/bin/bash
# RDT T5-XXL language-only feature precompute for a given tag (RDT_TASK_TAG=45|30|15).
# RDT_TEXT_ENCODER and RDT_LANG_OUTPUT_DIR must be set.

set -euo pipefail

TASK_TAG="${RDT_TASK_TAG:-45}"
RDT_TEXT_ENCODER="${RDT_TEXT_ENCODER:?Set RDT_TEXT_ENCODER (google/t5-v1_1-xxl)}"
RDT_LANG_OUTPUT_DIR="${RDT_LANG_OUTPUT_DIR:?Set RDT_LANG_OUTPUT_DIR (e.g. feature_cache/rdt_lang_cache_45_h200_t5xxl)}"
RDT_MAX_LANG_LEN="${RDT_MAX_LANG_LEN:-1024}"
RDT_LANG_BATCH_SIZE="${RDT_LANG_BATCH_SIZE:-4}"

HUMAN_DATASET_FILE="examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_${TASK_TAG}.json"
SIM_DATASET_FILE="examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_${TASK_TAG}.json"

mkdir -p "$RDT_LANG_OUTPUT_DIR"

python -m examples.baselines.rdt.precompute_vl_features \
  --output_dir "$RDT_LANG_OUTPUT_DIR" \
  --feature_mode language_only \
  --use_lerobot \
  --lerobot_use_paired_dataset \
  --lerobot_human_root demos/demo_data \
  --lerobot_sim_root demos/imitator_data \
  --lerobot_human_dataset_file "$HUMAN_DATASET_FILE" \
  --lerobot_sim_dataset_file "$SIM_DATASET_FILE" \
  --lerobot_task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
  --lerobot_human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --lerobot_sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --text_encoder "$RDT_TEXT_ENCODER" \
  --t5_version t5-v1_1-xxl \
  --max_lang_len "$RDT_MAX_LANG_LEN" \
  --batch_size "$RDT_LANG_BATCH_SIZE" \
  --num_dataload_workers 0 \
  --save_dtype bfloat16