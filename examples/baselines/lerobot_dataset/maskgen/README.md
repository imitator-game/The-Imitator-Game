# Mask Generation Development Guide

This directory contains the Grounded-SAM-2 annotation and LeRobot postprocessing tools used for IG-10K.

For released dataset locations, mask format, loading examples, visualization, generation settings, postprocessing, and quality-control notes, see [`MASK.md`](MASK.md).

## Active pipeline

```text
object_list_all.json
        │
        ▼
mask_gen_whole.py
        │  per-frame packbits masks + episode-camera JSONL
        ▼
postprocess_maskgen_whole_to_lerobot.py
        │
        ▼
LeRobot tasks with optional observation.images.*_mask video features
```

The source LeRobot dataset is never modified. Postprocessing writes to a separate output root.

## Environment

Two tested dependency configurations are provided:

| Target | Project file | PyTorch stack |
| --- | --- | --- |
| RTX 4090 / CUDA 12.1 | `pyproject.cuda121.toml` | PyTorch 2.5.1, TorchVision 0.20.1, TorchCodec 0.1.1 |
| RTX 5090 / CUDA 13.0 | `pyproject.cuda130.toml` | PyTorch 2.10.0, TorchVision 0.25.0, TorchCodec 0.10.0 |

Select the configuration and build the environment:

```bash
cd examples/baselines/lerobot_dataset/maskgen

# RTX 4090
bash use_pyproject.sh 4090
MASKGEN_CUDA_HOME=/usr/local/cuda-12.1 bash setup_maskgen_env.sh

# RTX 5090
bash use_pyproject.sh 5090
MASKGEN_CUDA_HOME=/usr/local/cuda-13.0 bash setup_maskgen_env.sh
```

`setup_maskgen_env.sh` installs the local Grounded-SAM-2 and GroundingDINO packages and runs import/version smoke tests.

If TorchCodec cannot find FFmpeg, provide a directory containing the FFmpeg shared libraries:

```bash
MASKGEN_FFMPEG_LIB_DIR=/path/to/ffmpeg/lib \
MASKGEN_CUDA_HOME=/usr/local/cuda-12.1 \
bash setup_maskgen_env.sh
```

## Required models

The active pipeline uses:

```text
<Grounded-SAM-2>/checkpoints/sam2.1_hiera_large.pt
<Grounded-SAM-2>/gdino_checkpoints/groundingdino_swint_ogc.pth
<bert-base-uncased snapshot>/config.json
```

The corresponding configs are:

```text
configs/sam2.1/sam2.1_hiera_l.yaml
grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

## Main tools

| File | Purpose |
| --- | --- |
| `mask_gen_whole.py` | Generate full-episode Grounded-SAM-2 masks |
| `object_list_all.json` | Canonical human/robot/simulation object prompts |
| `postprocess_maskgen_whole_to_lerobot.py` | Add mask videos and metadata to a copied LeRobot dataset |
| `overlay_withmask_video.py` | Overlay one mask video on its RGB video |
| `batch_overlay_sample_frames.py` | Export sampled overlay frames for QC |
| `generate_vis_from_mask_videos.py` | Generate mask-only color videos |

Run any Python tool with `--help` for its complete CLI. Reproduction commands for the released annotations are kept in [`MASK.md`](MASK.md).