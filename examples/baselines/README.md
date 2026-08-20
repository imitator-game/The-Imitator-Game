# Imitator Game Baselines

This folder contains the code for all baselines in the Imitator Game benchmark. Each baseline is provided as a self-contained directory with its own `README.md` describing the method, the training/roll-out entry points, the experiment launchers in [`exp_scripts/`](exp_scripts/), and the key hyperparameters used to reproduce the paper's experiments.

## Baselines

| Directory | Method | Initiation | Eval entry |
| --- | --- | --- | --- |
| [`_template/`](_template/) | **New-model template** (start here to add your own baseline) | `train_template_imitator.py` | `eval_template_imitator.py` |
| [`act/`](act/) | ACT (Action Chunking with Transformers) | `train_act_imitator.py` | `eval_act_imitator.py` / `parallel_eval_act.py` |
| [`diffusion_policy/`](diffusion_policy/) | Diffusion Policy (1D UNet) | `train_dp_imitator.py` | `eval_dp_imitator.py` / `parallel_eval_dp.py` |
| [`vqbet/`](vqbet/) | VQ-BeT (VQ-VAE + behavior Transformer) | `pretrain_vqvae_imitator.py` → `train_vqbet_imitator.py` | `eval_vqbet_imitator.py` / `parallel_eval_vqbet.py` |
| [`uniskill/`](uniskill/) | UniSkill (IDM + conditional diffusion) | `diffusion/train_uniskill.py` → `diffusion/train_cond_dp.py` | `diffusion/eval_uniskill.py` |
| [`xskill/`](xskill/) | XSkill (skill discovery + diffusion) | `scripts/stage1_pretrain_encoder.py` → `scripts/stage2_skill_transfer.py` | `scripts/eval_xskill.py` |
| [`gr00t/`](gr00t/) | NVIDIA GR00T N1.6 (flow-matching VLA) | `gr00t.experiment.launch_finetune` | `gr00t.eval.parallel_eval_imitator` / `eval_imitator` |
| [`rdt/`](rdt/) | RDT-1B (diffusion transformer, LoRA finetune) | `train_rdt_lora.py` / `train_rdt_scratch.py` | `eval_rdt_lora.py` / `parallel_eval_rdt.py` |
| [`pi/`](pi/) | pi0 / pi0.5 (flow-matching VLA, JAX) | `train_pi_lerobot_jax.py` | `eval_pi_lerobot_jax.py` / `parallel_eval_pi_lerobot_jax.py` |
| [`openvla_oft/`](openvla_oft/) | OpenVLA (LoRA finetune, discrete/regression/diffusion head) | `train_openvla.py` | `eval_openvla.py` / `eval_openvla_batch.py` |

## Shared infrastructure

| Directory | Purpose |
| --- | --- |
| [`lerobot_dataset/`](lerobot_dataset/) | LeRobot dataset loaders, human/sim/robot dataloaders, detection/learning-in/configs, evaluation processor, mask-gen tools |
| [`encoders/`](encoders/) | Shared observation/state encoders and the frozen task-video backbone used by ACT/Diffusion Policy/VQ-BeT |
| [`hand_estimation/`](hand_estimation/) | HaMeR and WiLoR hand-tracking tools used to estimate MANO hand features for the human dataset |
| [`exp_scripts/`](exp_scripts/) | Experiment launchers that reproduce the paper's training and evaluation runs |

## Common conventions

- **Data**: human demonstrations and simulation/robot trajectories are consumed through the LeRobot interface under `lerobot_dataset/`. See the repository [`README.md`](lerobot_dataset/README.md) for the dataset layout, `lerobot_dataset/maskgen/` for the mask annotations, and `lerobot_dataset/maskgen/MASK.md` for mask loading/visualization.
- **Environment**: baselines share the repository Python environment (`uv sync --active`, see the top-level [`README.md`](../../README.md)). GR00T additionally provides a dedicated `uv` environment in `gr00t/pyproject.toml`; Pi (JAX) additionally depends on the `openpi` package pinned in `pi/pyproject.toml`.
- **Experiment scripts**: every reported number in the paper is produced by a launcher under `exp_scripts/`. Set the environment variables and paths listed at the top of each launcher before running it.
- **Splits**: configs named `<kind>_train_config_{15,30,45}.json` are the 15/30/45-task training splits; `*_test_config_seen.json`, `*_test_config_unseen.json`, and the `eval/exp_list/*.txt` env lists are the seen/unseen evaluation splits.

## Citation

If you use any of these baselines, cite the original paper in addition to the Imitator Game paper. Each baseline directory provides its own BibTeX entry under the "Citation" section of its `README.md`.