# Pi Model Training on ManiSkill Dataset

This training script adapts the Pi model (Pi0/Pi0.5) to work with ManiSkill demonstrations, following OpenVLA's data loading pipeline while maintaining Pi's training methodology. Pi0 is the open-source generalist robot foundation model; Pi0.5 (Pi05) is the faster, smaller variant (see [Citation](#citation)).

## Overview

- **Data Loading**: Uses OpenVLA's ManiSkill data loader format.
- **Model**: Pi0/Pi05 models with flow matching training.
- **LoRA Support**: Efficient fine-tuning with LoRA adapters.
- **Action Modes**: Supports discrete tokenization (OpenVLA style) or continuous flow matching (Pi style).
- **JAX entrypoints**: `train_pi_lerobot_jax.py`, `eval_pi_lerobot_jax.py`, `parallel_eval_pi_lerobot_jax.py`, `eval_pi_real_robot_jax.py`.

## Key Features

1. **Data Format Alignment**: Converts ManiSkill observations to Pi's expected format.
2. **Flexible Action Spaces**: `continuous` — Pi's native flow matching (recommended); `discrete` — OpenVLA-style action tokenization.
3. **Memory Optimization**: Gradient checkpointing, mixed precision training.
4. **Multi-GPU Support**: DDP training out of the box.

## Pi05 JAX LeRobot Paired Training

For the current Pi05 JAX LeRobot human-sim paired multi-task training pipeline, use:

- [`train_pi_lerobot_jax.py`](train_pi_lerobot_jax.py)
- [`run_exp_pi.sh`](../../../exp_scripts/pi/run_exp_pi.sh) for 45/30/15-task Pi05 JAX runs on three GPUs.
- [`run_eval_pi.sh`](../../../exp_scripts/pi/run_eval_pi.sh) for parallel online eval (10-epoch checkpoints, SEEN/UNSEEN splits).

The Pi05 JAX path now has opt-in speed switches:

- `--skip_masked_cameras`: only run SigLIP on real camera views instead of padded masked zero cameras.
- `--use_prefix_kv_cache`: train with prefix prefill/KV cache followed by suffix action-expert forward.
- `--disable_dataset_augmentation`: disable LeRobot paired dataset-side Albumentations.
- `--disable_jax_image_augmentation`: disable OpenPI/JAX image augmentation inside `compute_loss`.

These switches default to off in `train_pi_lerobot_jax.py`, so omitting them preserves the previous runnable path. `run_exp_pi.sh` enables the first three for Pi05 paired training.

### Training (45-task example from `run_exp_pi.sh`)

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m examples.baselines.pi.train_pi_lerobot_jax \
  --pretrained_model_path <pi05_droid JAX checkpoint> \
  --output_dir "runs/pi_lerobot_jax_${TAG}_h200_opt" \
  --exp_name "pi_jax_${TAG}_h200_opt" \
  --human_root demos/demo_data/ \
  --sim_root demos/imitator_data/ \
  --human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_${TAG}.json \
  --sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_${TAG}.json \
  --num_epochs 10 --batch_size 16 \
  --eval_freq_epochs 100 --save_interval_epochs 1 \
  --log_interval 200 \
  --num_workers 32 --persistent_workers --prefetch_factor 16 \
  --max_episode_steps 500 --num_eval_episodes 1 \
  --use_epoch_training --overwrite --keep_all_checkpoints \
  --env_id L0_TwoRobotPourCup-v1 \
  --processor_name_or_path <google--paligemma-3b-pt-224> \
  --processor_local_files_only \
  --trainable_scope action_expert_full \
  --no_eval \
  --disable_dataset_augmentation --skip_masked_cameras --use_prefix_kv_cache \
  --disable_async_checkpointing --disable_ocdbt_checkpoint --cleanup_tmp_checkpoint_dirs \
  --use_wandb
```

`run_exp_pi.sh` launches the 45/30/15-task runs in parallel on GPUs 0/1/2 with 10/15/30 epochs.

### Evaluation (`run_eval_pi.sh`)

The eval orchestrator resolves the 10-epoch checkpoint step (using the `STEPS_PER_EPOCH_{45,30,15}` constants, e.g. 189968 for the 45-task split), then launches one scheduler per tag/split pinned to a GPU via `parallel_eval_pi_lerobot_jax`:

```bash
python -m examples.baselines.pi.parallel_eval_pi_lerobot_jax \
  --checkpoint_path <run dir> --checkpoint_step <step> \
  --result_root <output dir> \
  --sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_${SPLIT}.json \
  --human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_${SPLIT}.json \
  --human_root demos/demo_data --sim_root demos/imitator_data \
  --sim_state_type qpos \
  --processor_name_or_path <google--paligemma-3b-pt-224> --processor_local_files_only \
  --l3_eval_l_level L0 --eval_lr_mirror auto \
  --dtw_band_ratio 0.15 --compute_dtw \
  --reward_mode dense --num_eval_episodes 10 --num_eval_envs 1 \
  --max_episode_steps 500 --control_mode pd_joint_pos \
  --capture_video --num_gpus 1 --gpu_ids <gpu> --max_procs_per_gpu 1 \
  --pi05 --skip_masked_cameras --use_prefix_kv_cache
```

Default eval targets are `EVAL_TAGS="15 30 45"`, `EVAL_SPLITS="seen unseen"`. Results go under `runs/pi_eval_results/`.

## Other exp_scripts

- [`exp_scripts/precompute_cache/run_precompute_rdt_lang.sh`](../../../exp_scripts/precompute_cache/run_precompute_rdt_lang.sh) and [`exp_scripts/openvla/run_exp_openvla_cache_45.sh`](../../../exp_scripts/openvla/run_exp_openvla_cache_45.sh) — related V/L feature or cache exports used by the paired training paths.

## Notes

- `--pretrained_model_path` must point to a **JAX/Orbax-style checkpoint**, not a PyTorch safetensors file. Pi0 vs Pi0.5 is auto-detected from model params (`time_mlp_in`/`time_mlp_out`), with `--pi05` as an explicit override.
- Reuse a single cache directory across 45/30/15 runs only when the splits are compatible (the JAX dataloader caches video/features in a node-local cache directory, cleaned up by an EXIT trap in the launchers).

## Citation

```bibtex
@article{black2024pi_0,
  title={$$\backslash$pi\_0 $: A Vision-Language-Action Flow Model for General Robot Control},
  author={Black, Kevin and Brown, Noah and Driess, Danny and Esmail, Adnan and Equi, Michael and Finn, Chelsea and Fusai, Niccolo and Groom, Lachy and Hausman, Karol and Ichter, Brian and others},
  journal={arXiv preprint arXiv:2410.24164},
  year={2024}
}
```