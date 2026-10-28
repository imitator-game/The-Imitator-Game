# Zero-WAM Baseline

[Zero-WAM](https://github.com/robbyant-research/Zero-WAM) adapted to Imitator Game. The policy conditions on a human demonstration video and robot observations to generate video and dual-arm actions. Required upstream sources and license are included in `zero_wam/wan_va/`; see [UPSTREAM.md](zero_wam/wan_va/UPSTREAM.md).

## Scripts

- `precompute_vae.py`: prepare causal streaming VAE caches.
- `train_zero_wam_imitator.py`: Scratch, C15/C30/C45 pretraining, target-task fine-tuning and resume.
- `eval_zero_wam_imitator.py`: evaluate checkpoints from this baseline.

Run commands from the repository root in a Zero-WAM environment. Keep foundation weights, datasets, caches and run outputs outside the repository. You can keep final checkpoints in the gitignored `zerowam_checkpoint/` directory. Command-line arguments override script defaults. The commands below use the final settings.

## Configuration

| Setting | Value |
|---|---|
| Robot input | Complete episode from frame 0; causal streaming VAE |
| Robot frame spacing | One RGB every 4 controls; 16 controls per latent after the initial latent |
| Robot / human RGB | 224×288 / 320×480; full human clip at 12 fps |
| Human pairing | Random same-task demonstration on each training read |
| Training chunk / history window | Native random 1–4 latents / 4–64 chunk-stage units |
| Actions and proprioception | Episode-relative joints; joint slots 14–27, absolute grippers in 28/29 |
| Normalization | Global q01/q99 from the selected training split; separate action/state statistics |
| Initial action group | Normalized zeros, included in the loss as in the original code |
| Trainable scope | Joint video/action + MCP; LoRA rank 106, alpha 212; 499,447,326 parameters |
| Frozen modules | VAE, T5 and foundation weights; activation checkpointing off |
| Batch / optimizer | 1 episode/GPU × 4 GPUs, effective batch 4; native AdamW |
| Training | 60 epochs; cosine LR 1e-4 → 1e-6 after warmup |
| Evaluation | 500 controls, CFG 5, video/action 50 denoising steps; first 16 actions, then 32 |

## Training budget

We follow Zero-WAM's chunk-based training, using one complete IG robot episode per sample. One forward pass supervises several chunks in parallel. Each latent after the initial one covers 16 control steps. Many IG baselines instead create a sliding-window sample at each control step, so their epochs contain more samples and parameter updates.

For the same training splits and effective batch size of 4:

| Protocol | Robot episodes | Control steps | Zero-WAM updates/epoch | Sliding-window updates/epoch (estimate) |
|---|---:|---:|---:|---:|
| Scratch | 200 | 60,246 | 50 | 15,062 |
| C15 | 3,000 | 1,033,711 | 750 | 258,428 |
| C30 | 6,000 | 2,124,030 | 1,500 | 531,008 |
| C45 | 9,000 | 3,039,478 | 2,250 | 759,870 |

At the same batch size, the two training methods differ by about 301 to 354 times in updates per epoch. Over 60 epochs, our training budget is 3,000 optimizer updates for Scratch, 45,000 for C15, 90,000 for C30 and 135,000 for C45.

## Scratch pilot experiments

We used Scratch pilots to select the training duration and learning rate.

### 1. Constant learning rate

The following checkpoints come from the same run, using `1e-4` after 10 warmup updates.

| Start epoch | End epoch | Learning rate | Success rate |
|---:|---:|---|---:|
| 0 | 10 | Constant `1e-4` after warmup | 17.0% |
| 0 | 20 | Constant `1e-4` after warmup | 25.5% |
| 0 | 30 | Constant `1e-4` after warmup | 43.0% |

### 2. Learning-rate comparisons


| Start epoch | End epoch | Learning rate | Success rate |
|---:|---:|---|---:|
| 30 | 40 | Constant `1e-4` | 27.6% |
| 30 | 40 | Constant `1e-5` | **53.5%** |

| Start epoch | End epoch | Learning rate | Success rate |
|---:|---:|---|---:|
| 50 | 60 | Constant `1e-5` | 62.0% |
| 50 | 60 | Cosine `1e-5 → 1e-6` | **69.0%** |

| Start epoch | End epoch | Learning rate | Success rate |
|---:|---:|---|---:|
| 0 | 60 | Cosine `3e-5 → 1e-6` | 44.5% |
| 0 | 60 | Cosine `1e-4 → 1e-6` | **68.0%** |


### 3. Training beyond 60 epochs

| Start epoch | End epoch | Learning rate | Success rate |
|---:|---:|---|---:|
| 60 | 70 | Constant `1e-6` | 60.5% |
| 60 | 70 | Constant `1e-7` | 65.0% |

Neither tested schedule improved on epoch 60. We use 60 epochs to balance success rate and training time.

## Prepare data and train

Set external paths:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
export ZERO_WAM_MODEL=/external/zero-wam-posttrain-robotwin
export ZERO_WAM_EMPTY_TEXT=/external/empty_text_emb.pt
export HUMAN_ROOT=/external/imitator_human_v1
export SIM_ROOT=/external/imitator_sim_v1_zed2i
export ZERO_WAM_CACHE=/external/cache/c15
RUN=/external/runs/c15
PROTOCOL=c15
WARMUP=100
DECAY=44900
```

| Protocol | Robot / human episodes | Updates in 60 epochs | Warmup updates | `DECAY` |
|---|---:|---:|---:|---:|
| Scratch / target FT | 200 / 50 | 3,000 | 10 | 2,990 |
| C15 | 3,000 / 750 | 45,000 | 100 | 44,900 |
| C30 | 6,000 / 1,500 | 90,000 | 100 | 89,900 |
| C45 | 9,000 / 2,400 cached (2,250 eligible) | 135,000 | 100 | 134,900 |

```bash
python -m examples.baselines.zero_wam.precompute_vae \
  --protocol "$PROTOCOL" --manifest "$ZERO_WAM_CACHE/manifests/train.json" --action-per-frame 16 --robot-frame-stride 4 \
  --image-size 224 --robot-width 288 --human-height 320 --human-width 480

# Fit train-only statistics once for this split.
export STATS=/external/stats/c15.json
python - <<'PY'
import json, os
from pathlib import Path
from examples.baselines.zero_wam.zero_wam.action_space import fit_statistics
manifest = json.loads((Path(os.environ["ZERO_WAM_CACHE"]) / "manifests/train.json").read_text())
fit_statistics(manifest, "relative_episode", os.environ["STATS"])
PY

torchrun --standalone --nproc-per-node 4 \
  -m examples.baselines.zero_wam.train_zero_wam_imitator \
  --protocol "$PROTOCOL" --manifest "$ZERO_WAM_CACHE/manifests/train.json" --run-dir "$RUN" \
  --sampling-mode episode --action-per-frame 16 --robot-frame-stride 4 \
  --image-size 224 --robot-width 288 --human-height 320 --human-width 480 \
  --action-representation relative_episode --action-channel-offset 14 \
  --normalization-path "$STATS" --action-history-mode zero_history_supervised_v1 \
  --train-scope joint --adapter-rank 106 --enable-mcp --no-activation-checkpointing \
  --batch-size 1 --effective-batch-size 4 --total-epochs 60 \
  --learning-rate 1e-4 --min-learning-rate 1e-6 --lr-schedule cosine \
  --warmup-steps "$WARMUP" --lr-decay-start-step "$WARMUP" --lr-decay-steps "$DECAY" \
  --save-epoch-freq 10 --save-step-freq 1000 --save-seconds 1200
```

For fine-tuning (**FT**), use `PROTOCOL=scratch`, the target split's cache/statistics, `WARMUP=10`, `DECAY=2990`, and add `--initialize-from /external/pretrain/checkpoints/epoch_060.pt`. This loads the trained weights with a fresh optimizer and learning-rate schedule. To **resume**, keep the original settings and add `--resume-from "$RUN/checkpoints/latest.pt"` to restore the optimizer, scheduler, RNG and sampler progress. Training saves checkpoints; evaluation is launched separately.

## Evaluation

```bash
python -m examples.baselines.zero_wam.eval_zero_wam_imitator \
  --checkpoint "$RUN/checkpoints/epoch_060.pt" \
  --output-dir /external/evaluation/c15 \
  --eval-config examples/baselines/lerobot_dataset/eval/exp_list/seen_plus_unseen_10tasks_env_list.txt \
  --num-episodes 10 --max-episode-steps 500 \
  --icl-guidance-scale 5 --no-trajectory-metrics
```

The checkpoint restores architecture, action representation and normalization. Select the appropriate environment list and `--human-config` / `--sim-config` for seen or target-task evaluation. No video is exported by default. Parallel evaluation uses four independent workers per GPU. Workers request individual rollouts through `zero_wam/dispatch.py`. Each worker needs its own output directory and the shared `--dispatch-url` / `--case-root`. MCP is used only during training.
