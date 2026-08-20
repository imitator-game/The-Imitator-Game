#!/usr/bin/env python

import argparse
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import diffusers
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

from dataset import LeRobotPolicyConfig, LeRobotPolicyDataset
from policy_model import ConditionalUnet1D, AlignmentTransformer, get_resnet, replace_bn_with_gn

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dynamics.idm import IDM


logger = get_logger(__name__)


def sanitize_tracker_config(config):
    sanitized = {}
    for key, value in config.items():
        if isinstance(value, (bool, int, float, str, torch.Tensor)):
            sanitized[key] = value
        elif value is None:
            sanitized[key] = "None"
        elif isinstance(value, Path):
            sanitized[key] = str(value)
        elif isinstance(value, (list, tuple, dict)):
            sanitized[key] = json.dumps(value)
        else:
            sanitized[key] = str(value)
    return sanitized


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train Conditional Diffusion Policy for Uniskill.")

    # Standard training args
    parser.add_argument("--output_dir", type=str, default="outputs_policy", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_train_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=2)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--finetune_from_checkpoint",type=str,default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--report_name", type=str, default="uniskill_policy")

    # Dataset args
    parser.add_argument("--human_root", type=str, default="demos")
    parser.add_argument("--sim_root", type=str, default="demos")
    parser.add_argument("--task_mapping_file", type=str, default="examples/baselines/lerobot_dataset/task_mapping.json")
    parser.add_argument("--human_dataset_file", type=str, default="examples/baselines/lerobot_dataset/config/human_config.json")
    parser.add_argument("--sim_dataset_file", type=str, default="examples/baselines/lerobot_dataset/config/sim_config.json")
    parser.add_argument("--human_task_description_file", type=str, default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    parser.add_argument("--sim_task_description_file", type=str, default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")
    parser.add_argument("--state_type", type=str, default="qpos")
    parser.add_argument("--single_arm", action="store_true")
    parser.add_argument("--include_depth", action="store_true", default=False)
    parser.add_argument("--cameras", nargs="+", default=["zed2i"])
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--num_video_frames", type=int, default=10)
    parser.add_argument("--video_backend", type=str, default="torchcodec")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--enable_augmentation", action="store_true", default=False)
    parser.add_argument("--resolution", type=int, default=112, help="Image resolution for policy")
    parser.add_argument("--idm_resolution", type=int, default=224, help="Image resolution for IDM")

    # Model args
    parser.add_argument("--idm_ckpt_path", type=str, required=True, help="Path to frozen IDM checkpoint")
    parser.add_argument("--vision_feature_dim", type=int, default=256)
    parser.add_argument("--obs_dim", type=int, default=18)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--idm_feature_dim", type=int, default=128, help="Dimension to project IDM features to")

    # Policy Horizon args
    parser.add_argument("--policy_pred_horizon", type=int, default=16)
    parser.add_argument("--policy_obs_horizon", type=int, default=2)
    parser.add_argument("--policy_action_horizon", type=int, default=8)

    # Diffusion args
    parser.add_argument("--num_diffusion_iters", type=int, default=100)

    # Visualization args
    parser.add_argument("--vis_interval", type=int, default=1000, help="Visualization interval")

    # Alignment args
    parser.add_argument("--alignment_loss_weight", type=float, default=1.0)
    parser.add_argument("--pre_decode", action="store_true", help=("Pre Decoding Videos to ./tmp/"))
    parser.add_argument("--pre_decode_cache_dir", type=str, default="tmp/human_video_cache")
    parser.add_argument("--pre_decode_num_workers", type=int, default=4)
    parser.add_argument("--human_idm_cache_dir", type=str, default=None)
    parser.add_argument("--build_human_idm_cache", action="store_true")
    parser.add_argument("--robot_idm_cache_dir", type=str, default=None)
    parser.add_argument("--build_robot_idm_cache", action="store_true")
    parser.add_argument("--robot_frame_gap", type=int, default=35, help="Sim obs_horizon for robot IDM frame gap")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def collate_fn(examples):
    # Filter out None samples (if any)
    examples = [e for e in examples if e is not None]

    batch = {}

    # Stack Tensors
    batch["curr_images"] = torch.stack([example["curr_images"] for example in examples])
    batch["actions"] = torch.stack([torch.from_numpy(example["actions"]) for example in examples])
    batch["state"] = torch.stack([torch.from_numpy(example["state"]) for example in examples])

    batch["idm_curr_images"] = torch.stack([example["idm_curr_images"] for example in examples])
    batch["idm_next_images"] = torch.stack([example["idm_next_images"] for example in examples])

    batch["curr_depth_features"] = torch.stack([example["curr_depth_features"] if torch.is_tensor(example["curr_depth_features"]) else torch.from_numpy(example["curr_depth_features"]) for example in examples])
    batch["next_depth_features"] = torch.stack([example["next_depth_features"] if torch.is_tensor(example["next_depth_features"]) else torch.from_numpy(example["next_depth_features"]) for example in examples])

    batch["human_video_id"] = [example["human_video_id"] for example in examples]
    batch["task_name"] = [example["task_name"] for example in examples]
    batch["human_demo_idm_images_raw"] = [example.get("human_demo_idm_images") for example in examples]
    batch["human_demo_depth_features_raw"] = [example.get("human_demo_depth_features") for example in examples]
    batch["human_demo_idm_latents_raw"] = [example.get("human_demo_idm_latents") for example in examples]
    batch["robot_idm_cache_path"] = [example.get("robot_idm_cache_path", "") for example in examples]

    robot_idm_gt = [example.get("robot_idm_gt") for example in examples]
    batch["robot_idm_gt"] = (
        torch.stack(robot_idm_gt) if robot_idm_gt and all(t is not None for t in robot_idm_gt) else None
    )

    # Human data padding
    human_idm_imgs = [
        example.get("human_demo_idm_images")
        for example in examples
        if example.get("human_demo_idm_images") is not None
    ] # List of (T, ...)
    human_depth_feats = [
        example.get("human_demo_depth_features")
        for example in examples
        if example.get("human_demo_depth_features") is not None
    ] # List of (T, ...)

    max_len = max([t.shape[0] for t in human_idm_imgs]) if human_idm_imgs else 0

    padded_human_idm = []
    padded_human_depth = []
    human_masks = []

    if max_len > 0:
        for h_idm, h_depth in zip(human_idm_imgs, human_depth_feats):
            curr_len = h_idm.shape[0]
            pad_len = max_len - curr_len

            # Mask: True for padding (Torch convention for TransformerDecoder memory_key_padding_mask)
            mask = torch.zeros(max_len, dtype=torch.bool)

            if pad_len > 0:
                mask[curr_len:] = True

                # Pad IDM images
                # h_idm: (T, 2, 3, H, W)
                pad_shape_idm = (pad_len,) + h_idm.shape[1:]
                h_idm_padded = torch.cat([h_idm, torch.zeros(pad_shape_idm, dtype=h_idm.dtype)], dim=0)

                # Pad Depth features
                # h_depth: (T, 2, C, H, W)
                pad_shape_depth = (pad_len,) + h_depth.shape[1:]
                h_depth_padded = torch.cat([h_depth, torch.zeros(pad_shape_depth, dtype=h_depth.dtype)], dim=0)
            else:
                h_idm_padded = h_idm
                h_depth_padded = h_depth

            padded_human_idm.append(h_idm_padded)
            padded_human_depth.append(h_depth_padded)
            human_masks.append(mask)

    batch["human_demo_idm_images"] = torch.stack(padded_human_idm) if padded_human_idm else None
    batch["human_demo_depth_features"] = torch.stack(padded_human_depth) if padded_human_depth else None
    batch["human_masks"] = torch.stack(human_masks) if human_masks else None

    return batch


def _atomic_save_tensor(tensor: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.cpu(), tmp_path)
    os.replace(tmp_path, path)


def _human_cache_path(cache_dir: str, task_name: str, episode_idx: int) -> Path:
    return Path(cache_dir) / task_name / f"episode_{episode_idx:06d}.pt"


def _compute_human_idm_latents(
    video: torch.Tensor,
    dataset: LeRobotPolicyDataset,
    depth_processor,
    depth_estimator,
    idm,
    device,
    idm_resolution: int,
):
    human_frames = [dataset._to_chw_float(video[i]) for i in range(video.shape[0])]
    human_frames = [dataset._resize(x, idm_resolution) for x in human_frames]
    if len(human_frames) < 2:
        return torch.zeros((0, 768), dtype=torch.float32)

    visual_pairs = []
    depth_inputs = []
    for i in range(len(human_frames) - 1):
        f0 = human_frames[i]
        f1 = human_frames[i + 1]
        visual_pairs.append(torch.stack([f0, f1], dim=0))
        depth_inputs.append(
            torch.as_tensor(
                depth_processor(f0, do_rescale=False)["pixel_values"][0],
                dtype=torch.float32,
            )
        )
        depth_inputs.append(
            torch.as_tensor(
                depth_processor(f1, do_rescale=False)["pixel_values"][0],
                dtype=torch.float32,
            )
        )

    visual_pair = torch.stack(visual_pairs).to(device)
    depth_input = torch.stack(depth_inputs).to(device)

    with torch.inference_mode():
        depth_out = depth_estimator(depth_input).predicted_depth
        depth_min = depth_out.flatten(1).min(dim=1)[0]
        depth_max = depth_out.flatten(1).max(dim=1)[0]
        depth_denom = depth_max - depth_min
        depth_denom[depth_denom == 0] = 1.0
        depth_out = (depth_out - depth_min[..., None, None]) / depth_denom[..., None, None]
        depth_pair = depth_out.view(-1, 2, *depth_out.shape[1:])
        depth_pair = F.interpolate(
            depth_pair,
            size=(idm_resolution, idm_resolution),
            mode="bilinear",
            align_corners=False,
        )
        latents = idm(depth_pair, visual_pair).squeeze(1)

    return latents.float().cpu()


def build_human_idm_cache_if_needed(
    dataset: LeRobotPolicyDataset,
    depth_processor,
    depth_estimator,
    idm,
    args,
    accelerator,
):
    if not args.human_idm_cache_dir or not args.build_human_idm_cache:
        dataset.refresh_human_idm_cache_index()
        if args.human_idm_cache_dir:
            if dataset.human_idm_cache_complete:
                logger.info(
                    "Using existing complete human IDM cache at %s",
                    args.human_idm_cache_dir,
                )
            else:
                logger.warning(
                    "Human IDM cache at %s is missing or incomplete; "
                    "falling back to raw human video decoding and online frozen IDM when needed.",
                    args.human_idm_cache_dir,
                )
        else:
            logger.info("Human IDM cache disabled; using raw human video conditioning.")
        return

    human_dataset = dataset.paired_dataset.human_dataset
    if human_dataset is None:
        logger.warning(
            "Human IDM cache build requested, but human dataset is unavailable; "
            "falling back to the current dataset path."
        )
        return

    logger.info("Building human IDM cache at %s", args.human_idm_cache_dir)
    cache_dir = Path(args.human_idm_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cam_name = args.cameras[0]

    depth_estimator.eval()
    idm.eval()
    total_jobs = sum(repos[0]["num_episodes"] for repos in human_dataset.task_to_repos.values())
    progress = tqdm(
        total=total_jobs,
        desc="Building human IDM cache",
        disable=not accelerator.is_local_main_process,
    )

    for task_name, repos in human_dataset.task_to_repos.items():
        num_episodes = repos[0]["num_episodes"]
        for episode_idx in range(num_episodes):
            out_path = _human_cache_path(args.human_idm_cache_dir, task_name, episode_idx)
            if out_path.exists():
                progress.update(1)
                continue

            rgb_cache = human_dataset._episode_cache_path(
                task_name, episode_idx, cam_name, is_depth=False
            )
            if not rgb_cache.exists():
                ok = human_dataset._pre_decode_episode(task_name, episode_idx, cam_name)
                if not ok:
                    logger.warning(
                        "Failed to predecode human video for task=%s episode=%s; "
                        "that episode will be absent from human IDM cache.",
                        task_name,
                        episode_idx,
                    )
                    progress.update(1)
                    continue
            video = torch.load(rgb_cache, map_location="cpu", weights_only=True)
            latents = _compute_human_idm_latents(
                video,
                dataset,
                depth_processor,
                depth_estimator,
                idm,
                accelerator.device,
                args.idm_resolution,
            )
            _atomic_save_tensor(latents, out_path)
            progress.update(1)

    progress.close()
    dataset.refresh_human_idm_cache_index()
    if not dataset.human_idm_cache_index:
        raise RuntimeError(f"No human IDM cache entries were built in {args.human_idm_cache_dir}")
    if dataset.human_idm_cache_complete:
        logger.info("Human IDM cache is complete; raw human video decoding will be skipped.")
    else:
        missing_tasks = sorted({
            human_task_id for human_task_id, _ in dataset.paired_dataset.paired_tasks
        } - set(dataset.human_idm_cache_index))
        logger.warning(
            "Human IDM cache is incomplete after build; missing tasks=%s. "
            "Training will fall back to raw human video decoding for missing tasks.",
            missing_tasks,
        )


def build_robot_idm_cache_if_needed(
    dataset: LeRobotPolicyDataset,
    depth_estimator,
    idm,
    args,
    accelerator,
):
    if not args.robot_idm_cache_dir or not args.build_robot_idm_cache:
        if args.robot_idm_cache_dir:
            logger.info(
                "Robot IDM GT cache directory configured at %s; existing entries will be used when present.",
                args.robot_idm_cache_dir,
            )
        else:
            logger.info("Robot IDM GT cache disabled; computing robot IDM targets online.")
        return

    logger.info("Building robot IDM GT cache at %s", args.robot_idm_cache_dir)
    Path(args.robot_idm_cache_dir).mkdir(parents=True, exist_ok=True)
    depth_estimator.eval()
    idm.eval()

    cache_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    for batch in tqdm(
        cache_loader,
        desc="Building robot IDM cache",
        disable=not accelerator.is_local_main_process,
    ):
        cache_paths = [Path(p) for p in batch["robot_idm_cache_path"]]
        if cache_paths and all(path.exists() for path in cache_paths):
            continue

        with torch.inference_mode():
            depth_features = torch.cat([
                batch["curr_depth_features"],
                batch["next_depth_features"],
            ]).to(accelerator.device)

            depth_outputs = depth_estimator(depth_features).predicted_depth
            depth_min = depth_outputs.flatten(1).min(dim=1)[0]
            depth_max = depth_outputs.flatten(1).max(dim=1)[0]
            depth_denom = depth_max - depth_min
            depth_denom[depth_denom == 0] = 1.0
            depth_outputs = (depth_outputs - depth_min[..., None, None]) / depth_denom[..., None, None]
            curr_depth, next_depth = torch.chunk(depth_outputs, 2, dim=0)
            depth_pair = torch.stack([curr_depth, next_depth], dim=1)
            depth_pair = F.interpolate(
                depth_pair,
                size=(args.idm_resolution, args.idm_resolution),
                mode="bilinear",
                align_corners=False,
            )
            visual_pair = torch.stack([
                batch["idm_curr_images"],
                batch["idm_next_images"],
            ], dim=1).to(accelerator.device)
            robot_idm_gt = idm(depth_pair, visual_pair).squeeze(1).float().cpu()

        for idx, path in enumerate(cache_paths):
            if path.exists():
                continue
            _atomic_save_tensor(robot_idm_gt[idx], path)

def load_model_weights_from_accelerate_checkpoint(
    checkpoint_dir: str,
    vision_encoder,
    vision_projection,
    idm_projection,
    noise_pred_net,
    alignment_net,
    strict: bool = False,
):
    """
    Load model weights only from an Accelerate checkpoint directory.

    This does NOT load:
    - optimizer
    - lr_scheduler
    - random states
    - global_step
    - first_epoch

    It assumes the model order is the same as accelerator.prepare(...):
        0. vision_encoder
        1. vision_projection
        2. idm_projection
        3. noise_pred_net
        4. alignment_net
    """

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise ValueError(f"finetune checkpoint path is not a directory: {checkpoint_dir}")

    try:
        from safetensors.torch import load_file as safe_load_file
    except ImportError as e:
        raise ImportError(
            "safetensors is required to load Accelerate checkpoints. "
            "Please install it with: pip install safetensors"
        ) from e

    modules = [
        ("vision_encoder", vision_encoder),
        ("vision_projection", vision_projection),
        ("idm_projection", idm_projection),
        ("noise_pred_net", noise_pred_net),
        ("alignment_net", alignment_net),
    ]

    def _find_model_file(idx: int):
        if idx == 0:
            candidates = [
                checkpoint_dir / "model.safetensors",
                checkpoint_dir / "pytorch_model.bin",
            ]
        else:
            candidates = [
                checkpoint_dir / f"model_{idx}.safetensors",
                checkpoint_dir / f"pytorch_model_{idx}.bin",
            ]

        for p in candidates:
            if p.exists():
                return p

        return None

    for idx, (name, module) in enumerate(modules):
        model_file = _find_model_file(idx)

        if model_file is None:
            # Some modules, such as Identity, may have no parameters.
            if len(module.state_dict()) == 0:
                print(f"[Finetune] Skip {name}: empty state_dict and no checkpoint file.")
                continue

            raise FileNotFoundError(
                f"Cannot find checkpoint file for {name} in {checkpoint_dir}. "
                f"Expected model index {idx}."
            )

        if model_file.suffix == ".safetensors":
            state_dict = safe_load_file(str(model_file), device="cpu")
        else:
            state_dict = torch.load(model_file, map_location="cpu")

        msg = module.load_state_dict(state_dict, strict=strict)
        print(f"[Finetune] Loaded {name} from {model_file.name}: {msg}")

def main(args):
    logging_dir = Path(args.output_dir, "logs")

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Models

    # IDM (Frozen)
    idm = IDM(
        num_layers=8,
        num_heads=4,
        hidden_dim=256,
        skill_dim=64,
        out_dim=768,
        idm_resolution=args.idm_resolution,
    )

    logger.info(f"Loading IDM from {args.idm_ckpt_path}")
    idm_state_dict = torch.load(args.idm_ckpt_path, map_location='cpu')
    msg = idm.load_state_dict(idm_state_dict, strict=False) # strict=False in case of minor mismatches, but ideally True
    logger.info(f"IDM Load result: {msg}")
    idm.requires_grad_(False)
    idm.eval()

    # Vision Encoder
    # Using ResNet18 adapted for 1D Unet conditioning
    if args.vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18")
    else:
        # If not 512, we might need a custom encoder or linear projection.
        # xskill uses a custom ResnetConv if dim != 512.
        # For simplicity, let's use ResNet18 and project if needed, or just set dim=512.
        # But args default is 64. Let's use a small encoder or modify ResNet.
        # XSkill eval script says: if dim=512 use resnet18, else ResnetConv.
        # Let's implement a simple wrapper or force 512.
        # For now, let's use Resnet18 and a projection layer.
        vision_encoder = get_resnet("resnet18")
        # Resnet18 outputs 512. We can project to 64 if needed.
        # Or we just use 512 as vision dim.
        pass

    vision_encoder = replace_bn_with_gn(vision_encoder)

    # Projection for vision if needed
    vision_projection = torch.nn.Linear(512, args.vision_feature_dim) if args.vision_feature_dim != 512 else torch.nn.Identity()

    # Projection for IDM
    # Shrink IDM features (768) to match vision/state dimensions roughly
    idm_projection = torch.nn.Linear(768, args.idm_feature_dim)

    # Noise Prediction Net (Policy)
    # Global Cond = (Vision * Obs_Horizon) + (State * Obs_Horizon) + (IDM_Latent)
    # IDM Latent from Uniskill IDM is likely 768.
    global_cond_dim = (args.vision_feature_dim * args.policy_obs_horizon) + \
                      (args.obs_dim * args.policy_obs_horizon) + \
                      args.idm_feature_dim # IDM projected dim

    noise_pred_net = ConditionalUnet1D(
        input_dim=args.action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=256,
    )

    # Noise Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # Depth Processor
    depth_estimator = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_estimator.requires_grad_(False)

    # Alignment Module
    robot_dim = (args.vision_feature_dim * args.policy_obs_horizon) + (args.obs_dim * args.policy_obs_horizon)
    alignment_net = AlignmentTransformer(
        robot_dim=robot_dim,
        idm_dim=768, # IDM latent dim
    )
    alignment_net.to(accelerator.device)

    # Move to device
    idm.to(accelerator.device)
    vision_encoder.to(accelerator.device)
    vision_projection.to(accelerator.device)
    idm_projection.to(accelerator.device)
    noise_pred_net.to(accelerator.device)
    depth_estimator.to(accelerator.device)
    alignment_net.to(accelerator.device)

    if args.resume_from_checkpoint and args.finetune_from_checkpoint:
        raise ValueError(
            "Do not use --resume_from_checkpoint and --finetune_from_checkpoint together. "
            "--resume_from_checkpoint restores full training state; "
            "--finetune_from_checkpoint loads model weights only."
        )

    if args.finetune_from_checkpoint:
        load_model_weights_from_accelerate_checkpoint(
            checkpoint_dir=args.finetune_from_checkpoint,
            vision_encoder=vision_encoder,
            vision_projection=vision_projection,
            idm_projection=idm_projection,
            noise_pred_net=noise_pred_net,
            alignment_net=alignment_net,
            strict=True,
        )
        logger.info(
            "Loaded finetune weights from %s. Optimizer/scheduler/global_step are initialized from scratch.",
            args.finetune_from_checkpoint,
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(vision_encoder.parameters()) + list(vision_projection.parameters()) + list(idm_projection.parameters()) + list(noise_pred_net.parameters()) + list(alignment_net.parameters()),
        lr=args.learning_rate,
        weight_decay=args.adam_weight_decay,
    )

    # Dataset (LeRobot-only)
    dataset_cfg = LeRobotPolicyConfig(
        human_root=args.human_root,
        sim_root=args.sim_root,
        split="train",
        task_mapping_file=args.task_mapping_file,
        human_dataset_file=args.human_dataset_file,
        sim_dataset_file=args.sim_dataset_file,
        human_task_description_file=args.human_task_description_file,
        sim_task_description_file=args.sim_task_description_file,
        cameras=tuple(args.cameras),
        include_depth=args.include_depth,
        image_size=tuple(args.image_size),
        num_frames=args.num_video_frames,
        sampling_strategy="uniform_jitter",
        video_backend=args.video_backend,
        fps=args.fps,
        state_type=args.state_type,
        single_arm=args.single_arm,
        enable_augmentation=args.enable_augmentation,
        resolution=args.resolution,
        idm_resolution=args.idm_resolution,
        pred_horizon=args.policy_pred_horizon,
        obs_horizon=args.policy_obs_horizon,
        pre_decode=args.pre_decode,
        pre_decode_cache_dir=args.pre_decode_cache_dir,
        pre_decode_num_workers=args.pre_decode_num_workers,
        human_idm_cache_dir=args.human_idm_cache_dir,
        robot_idm_cache_dir=args.robot_idm_cache_dir,
        robot_frame_gap=args.robot_frame_gap,
    )
    dataset = LeRobotPolicyDataset(
        cfg=dataset_cfg,
        train=True,
        depth_processor=depth_processor,
    )

    build_human_idm_cache_if_needed(
        dataset=dataset,
        depth_processor=depth_processor,
        depth_estimator=depth_estimator,
        idm=idm,
        args=args,
        accelerator=accelerator,
    )
    build_robot_idm_cache_if_needed(
        dataset=dataset,
        depth_estimator=depth_estimator,
        idm=idm,
        args=args,
        accelerator=accelerator,
    )

    print(f'Dataset complete')

    # Save sim normalizer for eval-time denormalization
    if accelerator.is_main_process:
        import pickle
        sim_ds = dataset.paired_dataset.sim_dataset
        normalizer_info = {
            "normalizer": sim_ds.normalizer,
            "normalization_method": sim_ds.config.normalization_method,
        }
        # Build repo_id -> dataset_idx mapping for eval task resolution
        repo_id_to_idx = {}
        for idx, info in sim_ds.normalizer.dataset_info.items():
            repo_id_to_idx[info["repo_id"]] = idx
        normalizer_info["repo_id_to_dataset_idx"] = repo_id_to_idx
        with open(os.path.join(args.output_dir, "normalizer.pickle"), "wb") as f:
            pickle.dump(normalizer_info, f)

    print(f'Saving normalizer complete')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_train_epochs * len(dataloader),
    )

    # Prepare
    vision_encoder, vision_projection, idm_projection, noise_pred_net, alignment_net, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        vision_encoder, vision_projection, idm_projection, noise_pred_net, alignment_net, optimizer, dataloader, lr_scheduler
    )

    print(f'Preparing complete')

    # Training Loop
    global_step = 0
    first_epoch = 0

    # Cache for Human IDM Latents
    # Key: (task_name, human_video_id)
    # Value: Tensor (T, 768)
    IDM_LATENT_CACHE = {}
    warned_robot_idm_cache_fallback = False
    warned_human_idm_cache_fallback = False

    def find_latest_checkpoint(output_dir):
        if not os.path.isdir(output_dir):
            return None

        # Prefer restoring an epoch-boundary checkpoint first
        epoch_checkpoints = [
            d for d in os.listdir(output_dir)
            if d.startswith("epoch-") and d.split("-")[-1].isdigit()
        ]
        if epoch_checkpoints:
            epoch_checkpoints = sorted(epoch_checkpoints, key=lambda x: int(x.split("-")[-1]))
            return os.path.join(output_dir, epoch_checkpoints[-1])

        # Fall back to legacy step-based checkpoints
        step_checkpoints = [
            d for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()
        ]
        if not step_checkpoints:
            return None

        step_checkpoints = sorted(step_checkpoints, key=lambda x: int(x.split("-")[-1]))
        return os.path.join(output_dir, step_checkpoints[-1])

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            ckpt_path = find_latest_checkpoint(args.output_dir)
            if ckpt_path is None:
                logger.info("No checkpoint found for resume_from_checkpoint=latest. Starting from scratch.")
                args.resume_from_checkpoint = None
            else:
                args.resume_from_checkpoint = ckpt_path

        if args.resume_from_checkpoint:
            accelerator.load_state(args.resume_from_checkpoint)
            ckpt_name = os.path.basename(args.resume_from_checkpoint.rstrip("/"))

            if ckpt_name.startswith("epoch-"):
                meta_path = os.path.join(args.resume_from_checkpoint, "epoch_state.json")
                if not os.path.exists(meta_path):
                    raise RuntimeError(f"Missing epoch_state.json in {args.resume_from_checkpoint}")

                with open(meta_path, "r") as f:
                    epoch_state = json.load(f)

                global_step = int(epoch_state["global_step"])
                first_epoch = int(epoch_state["next_epoch"])

                logger.info(
                    f"Resumed from epoch checkpoint {args.resume_from_checkpoint}, "
                    f"global_step={global_step}, next_epoch={first_epoch}"
                )

            else:
                raise ValueError(f"Unsupported checkpoint format for resume_from_checkpoint: {args.resume_from_checkpoint}. Expected a directory starting with 'epoch-'.")
    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.report_name,
            config=sanitize_tracker_config(vars(args)),
        )

    print(f'Accelerator prepare complete')

    logger.info("***** Running training *****")

    for epoch in range(first_epoch, args.num_train_epochs):
        noise_pred_net.train()
        vision_encoder.train()
        vision_projection.train()
        idm_projection.train()
        alignment_net.train()

        is_epoch_resume = (
            args.resume_from_checkpoint is not None
            and os.path.basename(str(args.resume_from_checkpoint).rstrip("/")).startswith("epoch-")
        )

        resume_step = 0
        if (
            epoch == first_epoch
            and args.resume_from_checkpoint
            and not is_epoch_resume
        ):
            resume_step = global_step % len(dataloader)

        progress_bar = tqdm(
            total=len(dataloader),
            initial=resume_step,
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            if resume_step > 0 and step < resume_step:
                progress_bar.update(1)
                continue
            # Filter for robot data only
            # The Policy Dataset guarantees all items are robot data for policy learning.
            # IDM conditioning might be mixed (inside getitem), but targets are robot.
            # So no need to filter by 'data_type' == 'robot' anymore for loss calculation,
            # as the dataset only yields robot trajectories for the main loop.

            with accelerator.accumulate(noise_pred_net):
                # 1. Compute Robot IDM GT Latents
                if batch.get("robot_idm_gt") is not None:
                    robot_idm_gt = batch["robot_idm_gt"].to(accelerator.device)
                else:
                    if args.robot_idm_cache_dir and not warned_robot_idm_cache_fallback:
                        logger.warning(
                            "Robot IDM GT cache is unavailable for at least one batch; "
                            "falling back to online Depth-Anything + frozen IDM target computation. "
                            "Cache dir: %s",
                            args.robot_idm_cache_dir,
                        )
                        warned_robot_idm_cache_fallback = True
                    with torch.no_grad():
                        depth_features = torch.cat([
                            batch["curr_depth_features"],
                            batch["next_depth_features"],
                        ]) # (2B, C, H, W) or similar

                        # Normalize depth
                        depth_outputs = depth_estimator(depth_features.to(accelerator.device)).predicted_depth
                        depth_min, depth_max = depth_outputs.flatten(1).min(dim=1)[0], depth_outputs.flatten(1).max(dim=1)[0]
                        # Avoid division by zero
                        depth_denom = (depth_max - depth_min)
                        depth_denom[depth_denom == 0] = 1.0
                        depth_outputs = (depth_outputs - depth_min[..., None, None]) / depth_denom[..., None, None]

                        curr_depth, next_depth = torch.chunk(depth_outputs, 2, dim=0)

                        depth_pair = torch.stack([curr_depth, next_depth], dim=1)
                        # Interpolate to IDM resolution
                        depth_pair = F.interpolate(depth_pair, size=(args.idm_resolution, args.idm_resolution), mode="bilinear", align_corners=False)

                        visual_pair = torch.stack([
                            batch["idm_curr_images"],
                            batch["idm_next_images"]
                        ], dim=1).to(accelerator.device)

                        # (B, 1, 768) -> (B, 768)
                        robot_idm_gt = idm(depth_pair, visual_pair).squeeze(1)

                # 2. Compute Human IDM Latents
                human_idm_latents = None
                human_mask = None

                precomputed_human_latents = batch.get("human_demo_idm_latents_raw") or []
                if precomputed_human_latents and any(t is not None for t in precomputed_human_latents):
                    batch_human_latents = precomputed_human_latents
                else:
                    if args.human_idm_cache_dir and not warned_human_idm_cache_fallback:
                        logger.warning(
                            "Human IDM latent cache is unavailable for at least one batch; "
                            "falling back to raw human video tensors and online frozen IDM. "
                            "Cache dir: %s",
                            args.human_idm_cache_dir,
                        )
                        warned_human_idm_cache_fallback = True
                    # Collect batch requirements
                    batch_human_latents = []
                    miss_indices = []
                    miss_inputs_visual = []
                    miss_inputs_depth = []

                    # Check in-process cache. This only avoids repeated frozen IDM work;
                    # persistent human_idm_cache_dir avoids data-loader video decoding too.
                    for i, (vid_id, task_n) in enumerate(zip(batch["human_video_id"], batch["task_name"])):
                        # vid_id = -1 means no human video
                        if vid_id == -1:
                            batch_human_latents.append(None)
                            continue

                        cache_key = (task_n, vid_id)
                        if cache_key in IDM_LATENT_CACHE:
                            batch_human_latents.append(IDM_LATENT_CACHE[cache_key])
                        else:
                            batch_human_latents.append(None) # Placeholder
                            miss_indices.append(i)
                            miss_inputs_visual.append(batch["human_demo_idm_images_raw"][i])
                            miss_inputs_depth.append(batch["human_demo_depth_features_raw"][i])

                    # Process Cache Misses
                    if miss_indices:
                        with torch.no_grad():
                             miss_sizes = [t.shape[0] for t in miss_inputs_visual]

                             # Flatten all miss videos
                             # Visual: (Total_T, 2, 3, H, W)
                             all_visual = torch.cat(miss_inputs_visual, dim=0).to(accelerator.device)
                             # Depth Feats: (Total_T, 2, C, H, W)
                             all_depth_feats = torch.cat(miss_inputs_depth, dim=0).to(accelerator.device)

                             # --- Depth Estimation ---
                             # Flatten for depth model: (Total_T * 2, C, H, W)
                             h_depth_input = all_depth_feats.view(-1, *all_depth_feats.shape[2:])

                             h_depth_out = depth_estimator(h_depth_input).predicted_depth

                             # Normalize
                             h_d_min, h_d_max = h_depth_out.flatten(1).min(dim=1)[0], h_depth_out.flatten(1).max(dim=1)[0]
                             h_d_denom = (h_d_max - h_d_min)
                             h_d_denom[h_d_denom == 0] = 1.0
                             h_depth_out = (h_depth_out - h_d_min[..., None, None]) / h_d_denom[..., None, None]

                             # Reshape to (Total_T, 2, H, W)
                             h_depth_pair = h_depth_out.view(-1, 2, *h_depth_out.shape[1:])
                             h_depth_pair = F.interpolate(h_depth_pair, size=(args.idm_resolution, args.idm_resolution), mode="bilinear", align_corners=False)

                             h_idm_out = idm(h_depth_pair, all_visual).squeeze(1) # (Total_T, 768)

                             # Store back to cache and batch list
                             curr_idx = 0
                             for idx, size in zip(miss_indices, miss_sizes):
                                 latents = h_idm_out[curr_idx : curr_idx + size].cpu() # Store on CPU

                                 task_n = batch["task_name"][idx]
                                 vid_id = batch["human_video_id"][idx]
                                 key = (task_n, vid_id)

                                 IDM_LATENT_CACHE[key] = latents
                                 batch_human_latents[idx] = latents

                                 curr_idx += size

                # Construct Padded Batch
                valid_latents = [l for l in batch_human_latents if l is not None]

                if not valid_latents:
                     # Fallback
                     human_idm_latents = torch.zeros(robot_idm_gt.shape[0], 1, 768).to(accelerator.device)
                     human_mask = torch.zeros(robot_idm_gt.shape[0], 1, dtype=torch.bool).to(accelerator.device)
                else:
                     max_len = max([l.shape[0] for l in valid_latents])

                     padded_latents = []
                     masks = []

                     for l_or_none in batch_human_latents:
                         if l_or_none is None:
                             # Pad with zeros, mask all
                             padded_latents.append(torch.zeros(max_len, 768))
                             masks.append(torch.ones(max_len, dtype=torch.bool))
                         else:
                             l = l_or_none
                             curr_len = l.shape[0]
                             pad_len = max_len - curr_len

                             mask = torch.zeros(max_len, dtype=torch.bool)
                             if pad_len > 0:
                                 mask[curr_len:] = True
                                 pad_l = torch.cat([l, torch.zeros(pad_len, 768)], dim=0)
                             else:
                                 pad_l = l

                             padded_latents.append(pad_l)
                             masks.append(mask)

                     human_idm_latents = torch.stack(padded_latents).to(accelerator.device)
                     human_mask = torch.stack(masks).to(accelerator.device)

                # 3. Encode Vision
                # (B, C, H, W) -> (B, 512)
                curr_images = batch["curr_images"].to(accelerator.device)
                visual_feat = vision_encoder(curr_images).flatten(1)
                visual_feat = vision_projection(visual_feat) # (B, 64)

                # Expand visual feat for obs_horizon if needed
                visual_feat = visual_feat.repeat(1, args.policy_obs_horizon) # (B, 64*T)

                # 4. State
                state = batch["state"].to(accelerator.device) # (B, T, D)
                state_flat = state.flatten(1) # (B, T*D)

                # Alignment Input
                robot_feat = torch.cat([visual_feat, state_flat], dim=-1) # (B, Robot_Dim)

                # Predict IDM latent
                pred_idm_latent = alignment_net(robot_feat, human_idm_latents, human_mask) # (B, 768)

                # Alignment Loss
                align_loss = F.mse_loss(pred_idm_latent, robot_idm_gt)

                # Project for Policy
                cond_idm_latent = idm_projection(pred_idm_latent) # (B, 128)

                # 5. Global Cond
                global_cond = torch.cat([visual_feat, state_flat, cond_idm_latent], dim=-1)

                # 6. Diffusion Loss
                actions = batch["actions"].to(accelerator.device) # (B, T, D)
                noise = torch.randn_like(actions)
                bsz = actions.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=actions.device
                ).long()

                noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)

                # print(f'noisy actions: {noisy_actions.shape}, timesteps: {timesteps.shape}, global_cond: {global_cond.shape}')
                noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=global_cond)

                diff_loss = F.mse_loss(noise_pred, noise, reduction="none").mean()

                loss = diff_loss + args.alignment_loss_weight * align_loss
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(vision_encoder.parameters()) + list(vision_projection.parameters()) +
                        list(idm_projection.parameters()) + list(noise_pred_net.parameters()) +
                        list(alignment_net.parameters()),
                        args.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()

            if accelerator.sync_gradients:
                global_step += 1

                if accelerator.is_main_process and global_step % args.vis_interval == 0:
                    vis_dir = os.path.join(args.output_dir, "vis")
                    os.makedirs(vis_dir, exist_ok=True)

                    # Visualize first element in batch
                    idx = 0
                    t_name = batch["task_name"][idx]

                    # 1. Obs Image
                    # curr_images: (B, C, H, W)
                    obs_img = batch["curr_images"][idx]
                    torchvision.utils.save_image(obs_img, os.path.join(vis_dir, f"step_{global_step}_{t_name}_obs.png"))

                    # 2. Robot IDM Image (Curr)
                    # idm_curr_images: (B, 3, H, W)
                    r_idm_img = batch["idm_curr_images"][idx]
                    torchvision.utils.save_image(r_idm_img, os.path.join(vis_dir, f"step_{global_step}_{t_name}_robot_idm.png"))

                    # 3. Human IDM Image
                    # human_demo_idm_images: (B, T, 2, 3, H, W) or None
                    if batch.get("human_demo_idm_images") is not None:
                        h_imgs = batch["human_demo_idm_images"]
                        h_masks = batch.get("human_masks")

                        if h_imgs is not None and len(h_imgs) > idx:
                             # shape (T, 2, 3, H, W)
                             h_seq = h_imgs[idx]

                             valid_len = h_seq.shape[0]
                             if h_masks is not None:
                                 mask = h_masks[idx]
                                 # mask: True is padding
                                 valid_len = (~mask).sum().item()
                                 if valid_len == 0:
                                     valid_len = 1 # Fallback to show at least one frame if something is wrong

                             if h_seq.dim() >= 4:
                                 h_frames = h_seq[:valid_len, 0] # (T_valid, 3, H, W)
                                 torchvision.utils.save_image(h_frames, os.path.join(vis_dir, f"step_{global_step}_{t_name}_human_idm.png"), nrow=8)

                def cleanup_checkpoints(output_dir, total_limit):
                    if total_limit is None or total_limit <= 0:
                        return

                    checkpoints = [
                        d for d in os.listdir(output_dir)
                        if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()
                    ]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

                    if len(checkpoints) <= total_limit:
                        return

                    num_to_remove = len(checkpoints) - total_limit
                    for ckpt in checkpoints[:num_to_remove]:
                        rm_path = os.path.join(output_dir, ckpt)
                        logger.info(f"Removing old checkpoint: {rm_path}")
                        shutil.rmtree(rm_path, ignore_errors=True)


                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    if accelerator.is_main_process:
                        cleanup_checkpoints(args.output_dir, args.checkpoints_total_limit)

            logs = {
                "loss": loss.detach().item(),
                "diff_loss": diff_loss.detach().item(),
                "align_loss": align_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            progress_bar.update(1)

            if args.max_train_steps and global_step >= args.max_train_steps:
                break

        # ===== epoch-end checkpoint =====
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            epoch_ckpt_path = os.path.join(args.output_dir, f"epoch-{epoch + 1}")
            accelerator.save_state(epoch_ckpt_path)

            epoch_state = {
                "global_step": int(global_step),
                "finished_epoch": int(epoch),
                "next_epoch": int(epoch + 1),
                "len_dataloader": int(len(dataloader)),
                "train_batch_size": int(args.train_batch_size),
                "seed": int(args.seed) if args.seed is not None else None,
                "checkpoint_type": "epoch_boundary",
            }

            with open(os.path.join(epoch_ckpt_path, "epoch_state.json"), "w") as f:
                json.dump(epoch_state, f, indent=2)

            logger.info(
                f"Saved epoch checkpoint to {epoch_ckpt_path}: "
                f"finished_epoch={epoch}, next_epoch={epoch + 1}, global_step={global_step}"
            )

        accelerator.wait_for_everyone()

        if args.max_train_steps and global_step >= args.max_train_steps:
            break
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        if not os.path.exists(final_path):
            accelerator.save_state(final_path)
            logger.info(f"Saved final state to {final_path}")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
