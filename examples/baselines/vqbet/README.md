# VQ-BeT

3-stage pipeline: **VQ-VAE pretrain → VQ-BeT train → Eval**.

VQ-BeT is the discretized Behavior Transformer that learns a shared action codebook via VQ-VAE and predicts code indices with a transformer (see [Citation](#citation)). The task representation is provided by the shared task encoder package under `examples/baselines/encoders`.

## 1. Pretrain VQ-VAE

Learns the discrete action codebook from LeRobot action sequences.

```bash
# Minimal
python -m examples.baselines.vqbet.pretrain_vqvae_imitator \
    --human_root "demos/imitator_hevc" \
    --sim_root "demos/imitator_data" \
    --save_path "examples/baselines/vqbet/vqvae_ckpt"

# Full
python -m examples.baselines.vqbet.pretrain_vqvae_imitator \
    --human_root "demos/demo_data" \
    --sim_root "demos/imitator_data" \
    --human_dataset_file "examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json" \
    --sim_dataset_file "examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json" \
    --human_task_description_file "examples/baselines/lerobot_dataset/task_desc/human_desc.json" \
    --sim_task_description_file "examples/baselines/lerobot_dataset/task_desc/sim_desc.json" \
    --task_mapping_file "examples/baselines/lerobot_dataset/task_mapping.json" \
    --save_path "examples/baselines/vqbet/vqvae_ckpt" \
    --exp_name "vqvae-v1" \
    --epochs 100 \
    --batch_size 256 \
    --num_dataload_workers 24 \
    --seed 42 \
    --action_dim 16 \
    --act_horizon 10 \
    --n_latent_dims 512 \
    --vqvae_n_embed 32 \
    --vqvae_groups 2 \
    --encoder_loss_multiplier 1.0 \
    --act_scale 1.0 \
    --pred_horizon 16 \
    --obs_horizon 1 \
    --state_type "qpos" \
    --fps 30 \
    --cameras "zed2i" \
    --image_size 224 224 \
    --num_video_frames 10 \
    --video_backend "torchcodec" \
    --save_freq 10 \
    --track \
    --wandb_project_name "VQVAEPretrain"
```

| Arg               | Default | Description                       |
| ----------------- | ------- | --------------------------------- |
| `--epochs`        | 100     | Total training epochs             |
| `--batch_size`    | 256     | Batch size                        |
| `--action_dim`    | 16      | Action dimension                  |
| `--act_horizon`   | 10      | Action chunk length fed to VQ-VAE |
| `--n_latent_dims` | 512     | Latent space dimensions           |
| `--vqvae_n_embed` | 32      | Codebook size                     |
| `--vqvae_groups`  | 2       | Number of VQ groups               |
| `--save_freq`     | 10      | Checkpoint save interval (epochs) |
| `--track`         | False   | Enable W&B logging                |

## 2. Train VQ-BeT

Trains the full VQ-BeT policy using a pretrained VQ-VAE checkpoint. Training is **epoch-based** (`--epochs`); `--total_iters` is ignored if `--epochs` is set.

```bash
# Minimal
python -m examples.baselines.vqbet.train_vqbet_imitator \
    --human_root "demos/imitator_hevc" \
    --sim_root "demos/imitator_data" \
    --vqvae_ckpt "examples/baselines/vqbet/vqvae_ckpt/your_run/vqvae_best.pt"

# Full
python -m examples.baselines.vqbet.train_vqbet_imitator \
    --human_root "demos/demo_data" \
    --sim_root "demos/imitator_data" \
    --human_dataset_file "examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json" \
    --sim_dataset_file "examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json" \
    --human_task_description_file "examples/baselines/lerobot_dataset/task_desc/human_desc.json" \
    --sim_task_description_file "examples/baselines/lerobot_dataset/task_desc/sim_desc.json" \
    --task_mapping_file "examples/baselines/lerobot_dataset/task_mapping.json" \
    --vqvae_ckpt "examples/baselines/vqbet/vqvae_ckpt/your_run/vqvae_best.pt" \
    --exp_name "vqbet-lerobot" \
    --epochs 200 \
    --batch_size 32 \
    --num_dataload_workers 24 \
    --seed 1 \
    --lr 1e-4 \
    --action_dim 16 \
    --act_horizon 10 \
    --n_latent_dims 512 \
    --vqvae_n_embed 32 \
    --vqvae_groups 2 \
    --encoder_loss_multiplier 1.0 \
    --act_scale 1.0 \
    --pred_horizon 16 \
    --obs_horizon 1 \
    --state_type "qpos" \
    --fps 30 \
    --cameras "zed2i" \
    --image_size 224 224 \
    --num_video_frames 10 \
    --video_backend "torchcodec" \
    --obs_mode "rgb" \
    --video_encoder_type "seq" \
    --temporal_agg \
    --eval_freq 5 \
    --num_eval_episodes 10 \
    --num_eval_envs 1 \
    --save_freq 20 \
    --log_freq 10 \
    --env_id "TwoRobotStirSpoon-v1" \
    --control_mode "pd_joint_pos" \
    --shader "rt-fast" \
    --max_episode_steps 500 \
    --track \
    --wandb_project_name "vqbet-lerobot"
```

| Arg                    | Default  | Description                          |
| ---------------------- | -------- | ------------------------------------ |
| `--epochs`             | 200      | Total training epochs                |
| `--lr`                 | 1e-4     | Learning rate                        |
| `--vqvae_ckpt`         | required | Path to pretrained VQ-VAE checkpoint |
| `--video_encoder_type` | `seq`    | Video encoder type (`seq` / `ego4d`) |
| `--temporal_agg`       | False    | Enable temporal aggregation          |
| `--eval_freq`          | 5        | Eval interval (epochs)               |
| `--save_freq`          | 20       | Checkpoint save interval (epochs)    |
| `--num_eval_episodes`  | 10       | Episodes per eval                    |
| `--obs_mode`           | `rgb`    | Observation mode                     |

## 3. Evaluate

```bash
# Minimal
python -m examples.baselines.vqbet.eval_vqbet_imitator \
    --checkpoint "runs/your_run/checkpoints/best_model.pt" \
    --eval_config "examples/baselines/lerobot_dataset/eval/sim_eval.txt"

# Full
python -m examples.baselines.vqbet.eval_vqbet_imitator \
    --checkpoint "runs/your_run/checkpoints/best_model.pt" \
    --eval_config "examples/baselines/lerobot_dataset/eval/sim_eval.txt" \
    --output_dir "runs/your_run/batch_eval_results" \
    --human_root "demos/imitator_hevc" \
    --sim_root "demos/imitator_data" \
    --human_config "examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json" \
    --sim_config "examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json" \
    --num_episodes 10 \
    --seed 42 \
    --env_id "TwoRobotStirSpoon-v1" \
    --control_mode "pd_joint_pos" \
    --shader "rt-fast" \
    --max_episode_steps 500
```

| Arg              | Default  | Description                       |
| ---------------- | -------- | --------------------------------- |
| `--checkpoint`   | required | Path to trained VQ-BeT checkpoint |
| `--eval_config`  | required | Eval task config file             |
| `--num_episodes` | 10       | Episodes per task                 |
| `--output_dir`   | —        | Directory for eval result logs    |

## Experiment scripts

The reported VQ-BeT experiments are reproduced by the launchers under [`../exp_scripts/`](../exp_scripts/):

- [`exp_scripts/vqbet/run_exp_vqbet.sh`](../exp_scripts/vqbet/run_exp_vqbet.sh) — parallel 45/30/15-task runs, each covering the VQ-VAE pretraining stage and the VQ-BeT training stage on GPUs 0/1/2. Set `VQVAE_GLOBAL_CKPT` to skip pretraining and reuse an existing VQ-VAE checkpoint.
- [`exp_scripts/vqbet/run_eval_vqbet.sh`](../exp_scripts/vqbet/run_eval_vqbet.sh) — parallel evaluation on the seen/unseen benchmark tasks. Point `RUN_DIR_{45,30,15}` at the training run directories.

Run from the repository root after `export PYTHONPATH=$PWD:$PYTHONPATH`. Task counts 15/30/45 map to `human_train_config_{15,30,45}.json` / `sim_train_config_{15,30,45}.json`, and the SEEN-N/UNSEEN-N eval setups to `human_test_config_seen.json` / `human_test_config_unseen.json` + `sim_test_config_seen.json` / `sim_test_config_unseen.json`.

## Citation

```bibtex
@inproceedings{
  lee2025vqbet,
  title={VQ-BeT: Behavior Generation with Latent Actions},
  author={Seungjae Lee and Yibin Wang and Haritheja Etukuru and H. Jin Kim and Nur Muhammad Mahi Shafiullah and Lerrel Pinto},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025},
  url={https://github.com/seunggabi/vq-beat}
}
@inproceedings{
  shafiullah2022behavior,
  title={Behavior Transformers: Cloning Transformer Modes and Learned Behaviors},
  author={Nur Muhammad Mahi Shafiullah and Zichen Jeff Cui and Ariuntuya Arty Altanzaya and Lerrel Pinto},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```