#!/bin/bash
# Pi05 JAX train: 45/30/15-task human-desc paired training, one GPU per tag.
# PI_PRETRAIN_PATH and PALIGEMMA_PROCESSOR must point to local checkpoints.

set -euo pipefail

GPU_45="${GPU_45:-0}"
GPU_30="${GPU_30:-1}"
GPU_15="${GPU_15:-2}"

run_pi() {
  local gpu="$1" tag="$2" human_dataset_file="$3" epochs="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -m examples.baselines.pi.train_pi_lerobot_jax \
    --pretrained_model_path "$PI_PRETRAIN_PATH" \
    --output_dir "runs/pi_lerobot_jax_${tag}_h200" \
    --exp_name "pi_jax_${tag}_h200" \
    --human_root demos/demo_data/ \
    --sim_root demos/imitator_data/ \
    --human_dataset_file "$human_dataset_file" \
    --human_task_desc_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
    --sim_dataset_file "${human_dataset_file/human_/sim_}" \
    --num_epochs "$epochs" \
    --batch_size 16 \
    --eval_freq_epochs 100 \
    --save_interval_epochs 1 \
    --log_interval 200 \
    --overwrite \
    --num_workers 32 \
    --max_episode_steps 500 \
    --num_eval_episodes 1 \
    --use_epoch_training \
    --persistent_workers \
    --prefetch_factor 16 \
    --keep_all_checkpoints \
    --env_id L0_TwoRobotPourCup-v1 \
    --processor_name_or_path "$PALIGEMMA_PROCESSOR" \
    --processor_local_files_only \
    --trainable_scope action_expert_full \
    --no_eval \
    --disable_dataset_augmentation \
    --skip_masked_cameras \
    --use_prefix_kv_cache \
    --disable_async_checkpointing \
    --disable_ocdbt_checkpoint \
    --cleanup_tmp_checkpoint_dirs \
    --disable_tensorstore_file_locking \
    --use_wandb
}

run_pi "$GPU_45" 45 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json 10 &
run_pi "$GPU_30" 30 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_30.json 15 &
run_pi "$GPU_15" 15 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json 30 &

wait