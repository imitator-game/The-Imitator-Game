# RDT

RDT is a diffusion-based 1B-parameter dual-arm robot foundation model. Given `obs_horizon` RGB observations and a language instruction, a diffusion transformer (DiT) autoregressively denoises a receding-horizon action chunk. This directory adapts RDT-1B to the Imitator Game interface: a LeRobot-style paired dataset provides human-demonstration task context (via `task_mapping.json` + task descriptions), and the model is fine-tuned with LoRA before evaluation in ManiSkill.

The reproduction uses `robotics-diffusion-transformer/rdt-1b`, a SigLIP vision encoder, and a T5 text encoder (see [Citation](#citation)).

## Entrypoints

- Training (scratch): `train_rdt_scratch.py`
- Training (LoRA): `train_rdt_lora.py`
- Evaluation (PyTorch checkpoint): `eval_rdt_scratch.py`, `eval_rdt_lora.py`
- Parallel multi-task online eval: `parallel_eval_rdt.py`
- Offline V/L feature export: `precompute_vl_features.py`
- Shared dependencies: `examples/baselines/lerobot_dataset` and `examples/baselines/encoders`

## Expected execution context

Run from the repository root (see top-level [`README.md`](../../../README.md) for the shared environment):

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

The launchers activate the root `.venv`, set `HF_ENDPOINT=https://hf-mirror.com`, unset proxies, cap BLAS/thread env vars (`OMP_NUM_THREADS=1`, etc.), and route the Hugging Face datasets cache to a node-local cache directory (`NO_ALBUMENTATIONS_UPDATE=1`).

## Precomputed V/L features

Training attention is on online `SigLIP + T5` encoding cost. RDT supports exporting `img_tokens` / `lang_embeds` / `lang_mask` once per dataset and reloading them during training, keyed by `sample_id`:

- paired dataset: `"<sim_task_id>::<actual_sim_idx>"`
- non-paired h5 dataset: `"traj<traj_idx>::<start>::<end>"`

Export layout: `metadata.json` (validated at training start), `feature_index.json` (`sample_id -> {path, offset}`) plus `feature_index_part*.json` for parallel exports, and `shard_00000.pt, ...`. Language embeddings are deduplicated per-string; default `save_dtype=float16`; shard with `--dataset_num_shards / --dataset_shard_id`.

Two modes:

- `--feature_mode vl` — full V/L export (SigLIP + T5).
- `--feature_mode language_only` — T5-only export (`lang_features.pt`); no robot images are read and SigLIP is never initialized. Paired descriptions are computed over the whole description list (not pinned to the first one), so it matches training's per-sample random description sampling. RDT-1B defaults to T5-XXL with `--max_lang_len 1024`.

LoRA online eval loads no T5 unless explicitly allowed: it looks up embeddings from the language-only cache by the current task prompt string and raises on a cache miss instead of silently falling back (pass `--allow_online_text_encoder` only if you really want to load T5 at eval). Use `--verify_precomputed_vl_features` to re-compute a few batches at startup and compare against the cache (language-only mode compares only `lang_embeds` / `lang_mask`). Random augmentation is a known limitation: augmented online images may not match off-line exported `img_tokens`.

Example export (single process):

```bash
python -m examples.baselines.rdt.precompute_vl_features \
  --output_dir /path/to/rdt_vl_cache_45_h200 \
  --use_lerobot --lerobot_use_paired_dataset \
  --lerobot_human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json \
  --lerobot_sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json \
  --env_id L0_TwoRobotPourCup-v1 \
  --batch_size 128 --num_dataload_workers 32 \
  --feature_mode vl --shard_size 2048 --save_dtype float16 \
  --t5_version t5-v1_1-base \
  --vision_encoder <google--siglip-so400m-patch14-384> \
  --text_encoder <google--t5-v1_1-base> \
  --lerobot_root demos/imitator_data \
  --lerobot_human_root demos/demo_data \
  --lerobot_sim_root demos/imitator_data
```

For `language_only` (T5-XXL, 1024):

```bash
python -m examples.baselines.rdt.precompute_vl_features \
  --output_dir /path/to/rdt_lang_cache_45_h200_t5xxl \
  --feature_mode language_only \
  --use_lerobot --lerobot_use_paired_dataset \
  --lerobot_human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json \
  --lerobot_sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json \
  --env_id L0_TwoRobotPourCup-v1 \
  --batch_size 4 --num_dataload_workers 0 \
  --shard_size 8192 --save_dtype bfloat16 \
  --t5_version t5-v1_1-xxl --max_lang_len 1024 \
  --vision_encoder <google--siglip-so400m-patch14-384> \
  --text_encoder <google--t5-v1_1-xxl> \
  --lerobot_root demos/imitator_data \
  --lerobot_human_root demos/demo_data \
  --lerobot_sim_root demos/imitator_data
```

## Training

Two schedules are supported: iteration-based (`--total_iters`) and epoch-based (`--use_epoch_training --total_epochs`). The reported experiments use epoch-based LoRA training (10 / 15 / 30 epochs for the 45/30/15-task splits respectively).

Single-GPU minimum example (scratch, legacy ManiSkill h5 path):

```bash
python -m examples.baselines.rdt.train_rdt_scratch \
  --env_id PickCubeYCB-v1 \
  --demo_path "demos/PickCubeYCB-v1/motionplanning/multi_task_4.rgbd.pd_joint_delta_pos.physx_cpu.h5" \
  --batch_size 32 --eval_freq 1 --total_iters 1000000
```

The reported paired LeRobot LoRA run (representative of `run_exp_rdt.sh`):

```bash
python -m examples.baselines.rdt.train_rdt_lora \
  --use_lerobot --lerobot_use_paired_dataset --no-eval \
  --lerobot_human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_${TAG}.json \
  --lerobot_sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_${TAG}.json \
  --env_id L0_TwoRobotPourCup-v1 \
  --pretrained_path <robotics-diffusion-transformer--rdt-1b> \
  --batch_size 128 --exp_name "rdt_lora_${TAG}_h200_langcache_orig" \
  --num_eval_episodes 1 --num_eval_envs 1 \
  --t5_version t5-v1_1-xxl --max_lang_len 1024 \
  --control_frequency 30.0 \
  --use_epoch_training --total_epochs 10 --eval_epoch_freq 100 --save_epoch_freq 1 \
  --log_freq 200 --num_dataload_workers 4 \
  --vision_encoder <google--siglip-so400m-patch14-384> \
  --text_encoder <google--t5-v1_1-xxl> \
  --lora_r 576 --lora_alpha 1152 \
  --lerobot_root demos/imitator_data \
  --lerobot_human_root demos/demo_data \
  --lerobot_sim_root demos/imitator_data \
  --use_precomputed_vl_features \
  --precomputed_vl_dir ${RDT_LANG_CACHE_${TAG}} \
  --expected_precomputed_vl_mode language_only \
  --lerobot_sim_pre_decode \
  --lerobot_sim_pre_decode_cache_dir /path/to/rdt_sim_video_cache \
  --no-lerobot-enable-augmentation --track
```

Additional architecture flags used by earlier runs: `--obs_horizon 2 --pred_horizon 16 --depth 16 --num_heads 16 --max_episode_steps 600`.

## Evaluation

### Single checkpoint (online eval, LoRA)

Online eval does not load T5 by default; point `--precomputed_vl_dir` at the language-only cache used for training:

```bash
python -m examples.baselines.rdt.eval_rdt_lora \
  --checkpoint_path runs/<exp>/checkpoints/epoch_9.pt \
  --env_id L0_TwoRobotPourCup-v1 \
  --pretrained_path <robotics-diffusion-transformer--rdt-1b> \
  --use_lerobot --lerobot_use_paired_dataset --lerobot_eval_online \
  --lerobot_human_root demos/demo_data \
  --lerobot_sim_root demos/imitator_data \
  --lerobot_human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_45.json \
  --lerobot_sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_45.json \
  --lerobot_task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
  --lerobot_human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --lerobot_sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --vision_encoder <google--siglip-so400m-patch14-384> \
  --text_encoder <google--t5-v1_1-xxl> \
  --t5_version t5-v1_1-xxl --max_lang_len 1024 \
  --precomputed_vl_dir /path/to/rdt_lang_cache_45_h200_t5xxl \
  --expected_precomputed_vl_mode language_only \
  --lora_r 576 --lora_alpha 1152 \
  --num_eval_episodes 5 --num_eval_envs 1
```

### Parallel multi-task online eval

`parallel_eval_rdt.py` reads the full task list from `lerobot_sim_dataset_file` and dispatches each complete `env_id` (e.g. `L0_TwoRobotStirSpoon-v1`) to a GPU running `eval_rdt_scratch` / `eval_rdt_lora`:

```bash
python -m examples.baselines.rdt.parallel_eval_rdt \
  --checkpoint_path runs/<exp>/checkpoints/iter_100000.pt \
  --eval_module examples.baselines.rdt.eval_rdt_lora \
  --pretrained_path <robotics-diffusion-transformer--rdt-1b> \
  --lora_r 576 --lora_alpha 1152 \
  --num_eval_episodes 5 --num_eval_envs 1 \
  --vision_encoder <google--siglip-so400m-patch14-384> \
  --text_encoder <google--t5-v1_1-xxl> \
  --t5_version t5-v1_1-xxl --max_lang_len 1024 \
  --precomputed_vl_dir /path/to/rdt_lang_cache_45_h200_t5xxl \
  --expected_precomputed_vl_mode language_only \
  --lerobot_human_root demos/demo_data \
  --lerobot_sim_root demos/imitator_data \
  --lerobot_human_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json \
  --lerobot_sim_dataset_file examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json \
  --lerobot_task_mapping_file examples/baselines/lerobot_dataset/task_mapping.json \
  --lerobot_human_task_description_file examples/baselines/lerobot_dataset/task_desc/human_desc.json \
  --lerobot_sim_task_description_file examples/baselines/lerobot_dataset/task_desc/sim_desc.json \
  --gpu_ids 0 1 --max_procs_per_gpu 1
```

Results accumulate under `runs/<exp_name>/<checkpoint_tag>_parallel_eval/`: `parallel_eval_summary.json`, `parallel_eval_results.json`, `parallel_eval_results.csv`.

## Experiment scripts

- [`exp_scripts/rdt/run_exp_rdt.sh`](../exp_scripts/rdt/run_exp_rdt.sh) — trains the 45/30/15-task LoRA runs in parallel on GPUs 0/1/2 (10/15/30 epochs), reusing a single T5-XXL language-only cache across subsets, with sim-video predecode.
- [`exp_scripts/rdt/run_eval_rdt.sh`](../exp_scripts/rdt/run_eval_rdt.sh) — parallel online eval of the 10-epoch (`EVAL_CHECKPOINT_EPOCH`) checkpoints on the SEEN and UNSEEN splits, with DTW/TSS metrics and video capture.
- [`exp_scripts/precompute_cache/run_precompute_rdt_lang.sh`](../exp_scripts/precompute_cache/run_precompute_rdt_lang.sh) — language-only V/L cache export used by training.

## Citation

```bibtex
@inproceedings{liu2025rdt,
  title={Rdt-1b: a diffusion foundation model for bimanual manipulation},
  author={Liu, Songming and Wu, Lingxuan and Li, Bangguo and Tan, Hengkai and Chen, Huayu and Wang, Zhengyi and Xu, Ke and Su, Hang and Zhu, Jun},
  booktitle={International Conference on Learning Representations},
  volume={2025},
  pages={29982--30009},
  year={2025}
}
```

## Notes

- Change the T5 version by updating both `--t5_version` and the matching `--text_encoder` path.
- Checkpoint / V/L cache keys validated by `metadata.json`: `vision_encoder`, `text_encoder`, `t5_version`, `max_lang_len`, `obs_horizon`, `pred_horizon`, `use_lerobot`, `lerobot_use_paired_dataset`, and the paired dataset config/description/mapping paths. A mismatch aborts training.
- `config/test_configs/` referenced by older examples maps to the current `config/exp_configs/` files.