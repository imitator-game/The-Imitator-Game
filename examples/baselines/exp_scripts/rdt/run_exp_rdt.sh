#!/bin/bash
# RDT LoRA train: 45/30/15-task human-desc paired training, one GPU per tag.
# Requires language-only precompute caches (see exp_scripts/precompute_cache).
# RDT_LANG_CACHE_45/30/15 and RDT_SIM_PREDECODE_CACHE_DIR must be set.

set -euo pipefail

GPU_45="${GPU_45:-0}"
GPU_30="${GPU_30:-1}"
GPU_15="${GPU_15:-2}"
HUMAN_ROOT="demos/demo_data"
SIM_ROOT="demos/imitator_data"
RDT_LANG_CACHE_45="${RDT_LANG_CACHE_45:?Set RDT_LANG_CACHE_45}"
RDT_LANG_CACHE_30="${RDT_LANG_CACHE_30:-$RDT_LANG_CACHE_45}"
RDT_LANG_CACHE_15="${RDT_LANG_CACHE_15:-$RDT_LANG_CACHE_45}"
RDT_SIM_PREDECODE_CACHE_DIR="${RDT_SIM_PREDECODE_CACHE_DIR:?Set RDT_SIM_PREDECODE_CACHE_DIR}"
RDT_TEXT_ENCODER="${RDT_TEXT_ENCODER:?Set RDT_TEXT_ENCODER}"
PRETRAINED_PATH="${PRETRAINED_PATH:?Set PRETRAINED_PATH (robotics-diffusion-transformer/rdt-1b)}"
VISION_ENCODER="${VISION_ENCODER:?Set VISION_ENCODER (google/siglip-so400m-patch14-384)}"

run_rdt() {
  local gpu="$1" tag="$2" human_dataset_file="$3" lang_cache="$4" epochs="$5"
  CUDA_VISIBLE_DEVICES="$gpu" python -m examples.baselines.rdt.train_rdt_lora \
    --use_lerobot \
    --lerobot_use_paired_dataset \
    --no-eval \
    --lerobot_human_dataset_file "$human_dataset_file" \
    --lerobot_sim_dataset_file "${human_dataset_file/human_/sim_}" \
    --env_id L0_TwoRobotPourCup-v1 \
    --pretrained_path "$PRETRAINED_PATH" \
    --batch_size 128 \
    --exp_name "rdt_lora_${tag}_h200" \
    --num_eval_episodes 1 \
    --num_eval_envs 1 \
    --t5_version t5-v1_1-xxl \
    --max_lang_len 1024 \
    --control_frequency 30.0 \
    --total_epochs "$epochs" \
    --use_epoch_training \
    --log_freq 200 \
    --num_dataload_workers 4 \
    --dataloader_prefetch_factor 1 \
    --dataloader_multiprocessing_context fork \
    --eval_epoch_freq 100 \
    --save_epoch_freq 1 \
    --vision_encoder "$VISION_ENCODER" \
    --text_encoder "$RDT_TEXT_ENCODER" \
    --lora_r 576 \
    --lora_alpha 1152 \
    --lerobot_root "$SIM_ROOT" \
    --lerobot_human_root "$HUMAN_ROOT" \
    --lerobot_sim_root "$SIM_ROOT" \
    --use_precomputed_vl_features \
    --precomputed_vl_dir "$lang_cache" \
    --expected_precomputed_vl_mode language_only \
    --lerobot_sim_pre_decode \
    --lerobot_sim_pre_decode_cache_dir "$RDT_SIM_PREDECODE_CACHE_DIR" \
    --lerobot_sim_pre_decode_num_workers 8 \
    --no-lerobot-enable-augmentation \
    --track
}

run_rdt "$GPU_45" 45 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json "$RDT_LANG_CACHE_45" 10 &
run_rdt "$GPU_30" 30 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_30.json "$RDT_LANG_CACHE_30" 15 &
run_rdt "$GPU_15" 15 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json "$RDT_LANG_CACHE_15" 30 &

wait