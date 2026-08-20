#!/bin/bash
# VQ-BeT train: VQ-VAE pretrain then VQ-BeT train per tag, one GPU each.
# VQVAE_CKPT overrides the freshly-pretrained checkpoint to resume training.

set -euo pipefail

GPU_45="${GPU_45:-0}"
GPU_30="${GPU_30:-1}"
GPU_15="${GPU_15:-2}"
VQVAE_GLOBAL_CKPT="${VQVAE_GLOBAL_CKPT:-}"

run_vqvae() {
  local gpu="$1" tag="$2" human_dataset_file="$3" save_path="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -m examples.baselines.vqbet.pretrain_vqvae_imitator \
    --human_root demos/demo_data \
    --sim_root demos/imitator_data \
    --human_dataset_file "$human_dataset_file" \
    --sim_dataset_file "${human_dataset_file/human_/sim_}" \
    --human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
    --sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
    --task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
    --save_path "$save_path" \
    --exp_name "vqvae_${tag}" \
    --epochs 100 \
    --batch_size 256 \
    --num_dataload_workers 24 \
    --action_dim 16 \
    --act_horizon 10 \
    --n_latent_dims 512 \
    --vqvae_n_embed 32 \
    --vqvae_groups 2 \
    --act_scale 1.0 \
    --pred_horizon 16 \
    --obs_horizon 1 \
    --image_size 224 224 \
    --num_video_frames 10 \
    --state_type qpos
}

run_vqbet() {
  local gpu="$1" tag="$2" human_dataset_file="$3" vqvae_ckpt="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -m examples.baselines.vqbet.train_vqbet_imitator \
    --human_root demos/demo_data \
    --sim_root demos/imitator_data \
    --human_dataset_file "$human_dataset_file" \
    --sim_dataset_file "${human_dataset_file/human_/sim_}" \
    --human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
    --sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
    --task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
    --vqvae_ckpt "$vqvae_ckpt" \
    --epochs 200 \
    --batch_size 32 \
    --num_dataload_workers 24 \
    --lr 1e-4 \
    --action_dim 16 \
    --act_horizon 10 \
    --n_latent_dims 512 \
    --vqvae_n_embed 32 \
    --vqvae_groups 2 \
    --act_scale 1.0 \
    --pred_horizon 16 \
    --obs_horizon 1 \
    --image_size 224 224 \
    --num_video_frames 10 \
    --video_encoder_type seq \
    --temporal_agg \
    --eval_freq 5 \
    --num_eval_episodes 10 \
    --num_eval_envs 1 \
    --save_freq 20 \
    --env_id TwoRobotPourCup-v1 \
    --control_mode pd_joint_pos \
    --max_episode_steps 500
}

run_job() {
  local gpu="$1" tag="$2" human_dataset_file="$3"
  local save_path="examples/baselines/vqbet/vqvae_ckpt_${tag}"
  if [[ -n "${VQVAE_GLOBAL_CKPT}" ]]; then
    run_vqbet "$gpu" "$tag" "$human_dataset_file" "$VQVAE_GLOBAL_CKPT"
    return
  fi
  run_vqvae "$gpu" "$tag" "$human_dataset_file" "$save_path"
  run_vqbet "$gpu" "$tag" "$human_dataset_file" "$save_path/vqvae_${tag}/vqvae_best.pt"
}

run_job "$GPU_45" 45 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json &
run_job "$GPU_30" 30 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_30.json &
run_job "$GPU_15" 15 examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json &

wait