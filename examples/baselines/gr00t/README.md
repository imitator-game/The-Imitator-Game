# GR00T Baseline

This directory vendors the minimal GR00T N1.6 code needed for the project (the upstream package under `gr00t/`, the `scripts/lerobot_conversion/` helper scripts, and a dedicated `pyproject.toml` for its own uv environment). Imports still expect the package name `gr00t`.

GR00T N1.6 fuses multi-modal observations (RGB video, joint/qpos state) with a language task embedding and generates actions with a flow-matching / diffusion action head. Here it is used as a vision-language-action baseline conditioned on human task descriptions; the reported experiments fine-tune the model with LoRA on the shared LeRobot-style bimanual dataset (see `examples/baselines/lerobot_dataset`).

## Dedicated environment

```bash
cd examples/baselines/gr00t
uv sync
source .venv/bin/activate
```

After activating this environment, run everything from the repository root. Ensure the environment can import `gr00t.configs.base_config` and `gr00t.model.gr00t_n1d6.setup`, and that Hugging Face access to `nvidia/GR00T-N1.6-3B` works.

## Training

The 45/30/15-task runs are launched in parallel on GPUs 0/1/2 by [`exp_scripts/gr00t/run_exp_gr00t.sh`](../exp_scripts/gr00t/run_exp_gr00t.sh). It prepares dataset statistics first, then for each tag runs `gr00t.experiment.launch_finetune` (all CLI arguments come from `gr00t/configs/finetune_config.py` through `tyro.cli`):

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m gr00t.experiment.launch_finetune \
  --embodiment-tag NEW_EMBODIMENT \
  --human-config-path examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_${TAG}.json \
  --sim-config-path examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_${TAG}.json \
  --lerobot-version v3 \
  --language-source human_desc \
  --batch-size 256 \
  --dataloader-num-workers 32 \
  --num-gpus 1 \
  --epoch-based-training \
  --num-epochs 10 \
  --save-epochs 1 \
  --save-total-limit 30 \
  --logging-steps 100 \
  --output-dir "runs/gr00t_${TAG}_h200_$(date +%Y%m%d_%H%M%S)" \
  --use-backbone-lora --backbone-lora-rank 64 \
  --use-llm-lora --llm-lora-rank 64 \
  --tune-projector \
  --use-wandb \
  --base-model-path <GR00T-N1.6-3B cache path> \
  --dataset-path demos/imitator_data
```

The intended training setup for the current bimanual datasets is: LeRobot v3 data with `--lerobot-version v3`, `--language-source human_desc` (cols `task_mapping.json -> human_desc.json`), and `--embodiment-tag NEW_EMBODIMENT`. Defaults for `--task-mapping-path`, `--human-desc-path`, and `--sim-desc-path` already point at the shared files under `examples/baselines/lerobot_dataset/`. `--human-config-path` combined with `--sim-config-path` (or `--robot-config-path`) enables parent-directory multi-task training on the intersection of the two configs.

Minimal real-training command (single GPU, VRAM-friendly mixed LoRA regime):

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m gr00t.experiment.launch_finetune \
  --base-model-path /path/to/GR00T-N1.6-3B \
  --dataset-path demos/imitator_data/L0_TwoRobotCleanCup-v1 \
  --embodiment-tag NEW_EMBODIMENT \
  --lerobot-version v3 \
  --language-source human_desc \
  --batch-size 1 \
  --gradient-checkpointing \
  --dataloader-num-workers 0 \
  --num-gpus 1 \
  --max-steps 30000 \
  --save-steps 1000 \
  --save-total-limit 3 \
  --output-dir runs/gr00t_test_lora \
  --use-backbone-lora --backbone-lora-rank 16 \
  --use-llm-lora --llm-lora-rank 16 \
  --use-action-head-diffusion-lora --action-head-diffusion-lora-rank 16
```

This mode leaves the Eagle vision/language base weights and the diffusion base weights frozen, LoRA-tunes the Eagle backbone, the language model, and the action-head diffusion transformer, and fully fine-tunes the projector modules (`state_encoder`, `action_encoder`, `action_decoder`, `position_embedding`, `vlln`). LoRA is currently supported only for the Eagle language model, the Eagle vision backbone, and the action-head diffusion transformer/output head.

Resume by setting `GR00T_RESUME_CHECKPOINT` or `GR00T_RESUME_CHECKPOINT_{45,30,15}` to a `checkpoint-*` directory. A `WANDB_API_KEY` must be exported; set `HF_ENDPOINT=https://hf-mirror.com` and unset proxies when using the mirror. For a larger effective batch size use `--gradient-accumulation-steps`; on GPUs without bf16 use `--no-bf16 --fp16`.

Training-time single-task ManiSkill online evaluation is supported with `--enable-online-eval --online-eval-env-id <env> --online-eval-epochs <n> --online-eval-num-episodes <n>`.

## Evaluation

[`exp_scripts/gr00t/run_eval_gr00t.sh`](../exp_scripts/gr00t/run_eval_gr00t.sh) evaluates the latest (or `CHECKPOINT_DIR_{45,30,15}`-pinned) checkpoints on the SEEN and UNSEEN splits on GPUs 0/1/2:

```bash
python -m gr00t.eval.parallel_eval_imitator \
  --checkpoint_path <checkpoint> \
  --result_root <output dir> \
  --sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_${SPLIT}.json \
  --human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_${SPLIT}.json \
  --embodiment_tag NEW_EMBODIMENT \
  --language_source human_desc \
  --task_mapping_path examples/baselines/lerobot_dataset/task_mapping.json \
  --reward_mode dense --num_eval_episodes 10 --num_eval_envs 5 \
  --max_episode_steps 500 --control_mode pd_joint_pos --obs_mode rgb \
  --sim_backend physx_cpu --shader rt-fast --dtype bf16 \
  --num_gpus 1 --gpu_ids <gpu> --max_procs_per_gpu 1 --min_free_mem_gb 20 \
  --capture_video --compute_dtw --dtw_band_ratio 0.15
```

`--eval_lr_mirror auto` handles the ManiSkill left/right arm mirroring convention. Results are written under `runs/gr00t_eval_results/`.

## Notes

- The vendored training data path supports LeRobot v2 and v3 layouts behind `SingleDatasetConfig.lerobot_version` (`v2`, `v3`, or `auto`). The v3 loader reads `meta/episodes/chunk-*/file-*.parquet`, `meta/tasks.parquet`, shared `data/chunk-*/file-*.parquet`, and shared `videos/<video_key>/chunk-*/file-*.mp4`, deriving the state/action dimensionality from `meta/info.json` feature names.
- `language_source` supports `task` (original task string), `human_desc` (sampled from `human_desc.json`), and `sim_desc`. `human_desc` sampling matches `examples/baselines/lerobot_dataset/lerobot_paired_dataset.py`.
- Training is step-based by default (`--max-steps`, `--save-steps`); use `--epoch-based-training --num-epochs --save-epochs` for epoch-based runs.
- This project integration is not LoRA-first by default; the projector modules are full-finetuned even in the LoRA regime above.

## Citation

```bibtex
@article{bjorck2025gr00t,
  title={Gr00t n1: An open foundation model for generalist humanoid robots},
  author={Bjorck, Johan and Casta{\~n}eda, Fernando and Cherniadev, Nikita and Da, Xingye and Ding, Runyu and Fan, Linxi and Fang, Yu and Fox, Dieter and Hu, Fengyuan and Huang, Spencer and others},
  journal={arXiv preprint arXiv:2503.14734},
  year={2025}
}
```