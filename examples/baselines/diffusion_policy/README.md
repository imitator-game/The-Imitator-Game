# Diffusion Policy Baseline

This directory contains the diffusion-policy baseline adapted to the Imitator Game interface. The model conditions a 1D action diffusion model (UNet) on robot observations together with a task representation extracted from the human demonstration video.

Diffusion Policy is the visuomotor policy-learning framework proposed by Chi et al. (see [Citation](#citation)).

## Entrypoints

- Training: `train_dp_imitator.py`
- Evaluation: `eval_dp_imitator.py`, `parallel_eval_dp.py`, `eval_dp_single.py`
- Shared dependencies: `examples/baselines/lerobot_dataset` and `examples/baselines/encoders`

## Expected execution context

Run from the repository root (see top-level [`README.md`](../../../README.md) for environment setup):

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## Reference training command

The training script uses `tyro` over a dataclass configuration. The command below shows the frozen task-video (video-only) configuration used in the reported experiments:

```bash
TAG=<15|30|45>
BACKBONE=<dinov2_vitl14|siglip2_so400m|videomae_large>

python -m examples.baselines.diffusion_policy.train_dp_imitator \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-dataset-file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_${TAG}.json \
  --sim-dataset-file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_${TAG}.json \
  --task-mapping-file examples/baselines/lerobot_dataset/task_mapping.json \
  --human-task-description-file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim-task-description-file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --input-mode video_only \
  --task-encoder-type frozen_backbone \
  --frozen-backbone-type ${BACKBONE} \
  --frozen-backbone-num-frames 10 \
  --frozen-backbone-adapter-layers 1 \
  --frozen-backbone-seq-patches 32 \
  --control-mode pd_joint_pos \
  --batch-size 256 \
  --total-epochs 10 \
  --lr 1e-4 \
  --save-epoch-freq 5 \
  --num-dataload-workers 24 \
  --env-id TwoRobotPourCup-v1 \
  --max-episode-steps 500 \
  --unet-dims 352 704 1408 2816
```

Checkpoints are written to `runs/<run_name>/checkpoints/`, including `final_model.pt`.

## Reference evaluation command

```bash
BACKBONE=<dinov2_vitl14|siglip2_so400m|videomae_large>

python -m examples.baselines.diffusion_policy.eval_dp_imitator \
  --eval-config path/to/eval_config.txt \
  --checkpoint path/to/final_model.pt \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-config path/to/human_eval_config.json \
  --sim-config path/to/sim_eval_config.json \
  --task-mapping examples/baselines/lerobot_dataset/task_mapping.json \
  --human-task-desc examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --sim-task-desc examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --input-mode video_only \
  --task-encoder-type frozen_backbone \
  --frozen-backbone-type ${BACKBONE} \
  --frozen-backbone-num-frames 10 \
  --action-dim 16 \
  --state-dim 18 \
  --obs-horizon 1 \
  --pred-horizon 16 \
  --obs-latent-dim 256 \
  --task-latent-dim 256 \
  --task-num-frames 10 \
  --num-diffusion-iters 100 \
  --num-episodes 10 \
  --num-envs 1 \
  --max-episode-steps 500 \
  --control-mode pd_joint_pos \
  --obs-mode rgb \
  --sim-backend physx_cpu \
  --shader rt-fast \
  --use-ddim \
  --num-ddim-steps 10
```

## Experiment scripts

The reported Diffusion Policy experiments are reproduced by the launchers under [`../exp_scripts/`](../exp_scripts/):

- [`exp_scripts/diffusion_policy/run_exp_dp.sh`](../exp_scripts/diffusion_policy/run_exp_dp.sh) — parallel 45/30/15-task frozen-backbone training on GPUs 0/1/2 (UNet channels `--unet-dims 352 704 1408 2816`).
- [`exp_scripts/diffusion_policy/run_eval_dp.sh`](../exp_scripts/diffusion_policy/run_eval_dp.sh) — parallel evaluation on the seen/unseen benchmark tasks. Point `RUN_DIR_{45,30,15}` at the training run directories.

## Notes

- The task encoder settings must match between training and evaluation.
- `--use-ddim` is an evaluation-time acceleration option; omit it if you want to reproduce the default DDPM rollout path exactly.
- As with the other benchmark baselines, L1 and L2 are realized as environment-side scene changes rather than policy-side switches.

## Citation

```bibtex
@inproceedings{
  chi2023diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Cheng Chi and Siyuan Feng and Yilun Du and Zhenjia Xu and Eric Cousineau and Benjamin Burchfiel and Shuran Song},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023},
  url={https://diffusion-policy.cs.columbia.edu/}
}
```