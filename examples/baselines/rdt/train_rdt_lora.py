"""
RDT LoRA Training Script - Simplified Version (Language-only, no video)
Directly uses language from dataset without video conditioning
"""

import os
import random
import time
import json
import re
import math
from copy import copy
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import tyro

# NOTE: mani_skill imports moved to lazy inside train() when --no-eval is NOT set.

# NOTE: make_eval_envs imported lazily inside train()
from examples.baselines.rdt.utils.utils import (
    IterationBasedBatchSampler,
    build_state_obs_extractor,
    convert_obs,
    worker_init_fn
)
from examples.baselines.rdt.utils.precomputed_lang_loader import HybridLangEmbedding
from examples.baselines.rdt.utils.precomputed_vl_loader import PrecomputedVLFeatures
from examples.baselines.rdt.utils.precomputed_vl_metadata import build_precomputed_vl_expected_metadata


def _extract_l_level_from_task_id(task_id: Optional[str]) -> Optional[str]:
    if not task_id:
        return None
    match = re.match(r"^(L[0-3])_", str(task_id))
    return match.group(1) if match else None


def _extract_base_env_id(task_id: str) -> str:
    if isinstance(task_id, str) and re.match(r"^L[0-3]_.+", task_id):
        return task_id.split("_", 1)[1]
    return task_id


def _set_l_level_flags(level: Optional[str]) -> None:
    if level is None:
        return
    level = str(level).upper()
    if level not in ("L0", "L1", "L2", "L3"):
        raise ValueError(f"l_level must be one of L0/L1/L2/L3, got: {level}")
    try:
        from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils as lutils
        lutils.set_l1_enabled(level == "L1")
        lutils.set_l2_enabled(level == "L2")
        lutils.set_l3_enabled(level == "L3")
    except Exception:
        os.environ["MANI_SKILL_L1"] = "1" if level == "L1" else "0"
        os.environ["MANI_SKILL_L2"] = "1" if level == "L2" else "0"
        os.environ["MANI_SKILL_L3"] = "1" if level == "L3" else "0"


def _lerobot_default_language(text):
    if text is None:
        return "pick red cube and place on plate."
    if isinstance(text, list):
        for item in text:
            if isinstance(item, str) and item.strip():
                return item
        return "pick red cube and place on plate."
    if isinstance(text, str) and text.strip() == "":
        return "pick red cube and place on plate."
    if isinstance(text, str) and text.strip().lower() == "none":
        return "pick red cube and place on plate."
    return text


def _lerobot_resolve_language(
    item,
    *,
    use_paired_dataset: bool,
    sim_repo_ids: Optional[tuple[str, ...]] = None,
    sim_task_descriptions: Optional[dict] = None,
):
    text = item.get("language")
    if isinstance(text, list):
        for item_text in text:
            if isinstance(item_text, str) and item_text.strip():
                return item_text
    if isinstance(text, str) and text.strip():
        return text
    if text is None or not str(text).strip():
        if use_paired_dataset and sim_repo_ids and sim_task_descriptions:
            dataset_idx = item.get("dataset_idx")
            if dataset_idx is not None:
                try:
                    idx = int(dataset_idx)
                    repo_id = sim_repo_ids[idx]
                    fallback = sim_task_descriptions.get(repo_id, "")
                    if fallback:
                        return fallback
                except Exception:
                    pass
        return _lerobot_default_language(text)
    return _lerobot_default_language(text)


def collate_fn_lerobot(
    batch,
    *,
    use_paired_dataset: bool,
    sim_repo_ids: Optional[tuple[str, ...]] = None,
    sim_task_descriptions: Optional[dict] = None,
    precomputed_vl: Optional[PrecomputedVLFeatures] = None,
):
    observations = {}
    sample_ids: list[str] = []
    if use_paired_dataset:
        states = torch.stack([item["robot_obs"]["states"] for item in batch])
        observations["state"] = states

        view_keys = sorted(
            [k for k in batch[0]["robot_obs"].keys() if k.startswith("view_")],
            key=lambda x: int(x.split("_")[1]),
        )
        view_key = view_keys[0] if view_keys else None
        if view_key is not None:
            view = torch.stack([item["robot_obs"][view_key] for item in batch])
            if view.dim() == 5 and view.shape[-1] in (3, 4):
                view = view.permute(0, 1, 4, 2, 3)
            observations["rgb"] = view[:, :, :3]

        actions = torch.stack([item["robot_actions"] for item in batch])
        languages = [
            _lerobot_resolve_language(
                item,
                use_paired_dataset=use_paired_dataset,
                sim_repo_ids=sim_repo_ids,
                sim_task_descriptions=sim_task_descriptions,
            )
            for item in batch
        ]
        sample_ids = [str(item.get("sample_id", "")) for item in batch]
    else:
        states = torch.stack([item["states"] for item in batch])
        observations["state"] = states

        view_keys = sorted(
            [k for k in batch[0].keys() if k.startswith("view_")],
            key=lambda x: int(x.split("_")[1]),
        )
        view_key = view_keys[0] if view_keys else None
        if view_key is not None:
            view = torch.stack([item[view_key] for item in batch])
            observations["rgb"] = view[:, :, :3]

        actions = torch.stack([item["actions"] for item in batch])
        languages = [_lerobot_default_language(item.get("task_descriptions")) for item in batch]
        sample_ids = [str(item.get("sample_id", "")) for item in batch]

    collated = {
        "observations": observations,
        "actions": actions,
        "language": languages,
        "sample_id": sample_ids,
    }
    return collated


@dataclass
class LoRAArgs:
    # Experiment
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "RDT-LoRA"
    wandb_entity: Optional[str] = None
    capture_video: bool = True

    # Environment
    env_id: str = "PickCubeYCB-v1"
    demo_path: str = "demos/PickCubeYCB-v1/motionplanning/multi_task_4.rgbd.pd_joint_delta_pos.physx_cpu.h5"
    num_demos: Optional[int] = None
    obs_mode: str = "rgb"
    control_mode: str = "pd_joint_pos"
    reward_mode: str = "dense"
    max_episode_steps: int = 600
    sim_backend: str = "physx_cpu"
    shader: str = "rt-fast"

    # Training
    total_iters: int = 100000
    total_epochs: int = 100
    use_epoch_training: bool = False
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    num_dataload_workers: int = 24
    dataloader_prefetch_factor: int = 1
    dataloader_multiprocessing_context: Optional[str] = "fork"
    log_freq: int = 5
    eval_freq: int = 5000
    save_freq: int = 10000
    eval_epoch_freq: int = 1
    save_epoch_freq: int = 10
    num_eval_episodes: int = 10
    num_eval_envs: int = 1

    # ── Eval muting ──────────────────────────────────────────────────────────
    # Set no_eval=True (--no-eval) to skip ALL evaluation and avoid creating
    # any simulation environment (mani_skill will NOT be imported).
    no_eval: bool = True
    gripper_binary_action: bool = True
    gripper_threshold: float = 0.4
    gripper_indices: Tuple[int, ...] = (7, 15)

    # RDT specific
    obs_horizon: int = 2
    pred_horizon: int = 16
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    img_size: Tuple[int, int] = (224, 224)

    # Encoders
    vision_encoder: str = "/path/to/google--siglip-so400m-patch14-384"
    text_encoder: str = "/path/to/google--t5-v1_1-xxl"
    t5_version: str = "t5-v1_1-xxl"
    max_lang_len: int = 1024
    control_frequency: float = 30.0
    rdt_slot_mapping: str = "official"
    """Runner slot mapping for local actions: 'official' or legacy pre-4473996 'legacy'."""

    # Diffusion
    # Kept for backward-compatible CLI parsing. Pretrained RDT LoRA training now
    # reuses the scheduler and prediction type stored in RDTRunner.
    num_diffusion_iters: int = 100
    num_inference_steps: int = 10

    # LoRA specific arguments
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q", "kv", "proj", "ffn"]
    )
    lora_bias: str = "none"

    # Pretrained model
    pretrained_path: Optional[str] = None
    """Path to pretrained RDT model for LoRA fine-tuning"""

    # Language embedding options
    use_precomputed_lang: bool = False
    """Whether to use precomputed language embeddings"""
    precomputed_lang_dir: Optional[str] = None
    """Directory containing precomputed language embeddings"""
    use_precomputed_vl_features: bool = False
    """Whether to load precomputed image/language features for training batches"""
    precomputed_vl_dir: Optional[str] = None
    """Directory containing precomputed per-sample img_tokens/lang_embeds/lang_mask"""
    precomputed_vl_preload: bool = False
    """Preload precomputed V/L features into host memory before training"""
    precomputed_vl_shard_cache_size: int = 8
    """Number of shard files kept in memory when using sharded precomputed V/L features"""
    expected_precomputed_vl_mode: Optional[str] = None
    """Expected precomputed feature mode: 'vl' or 'language_only'. If set, metadata must match."""
    precomputed_img_features_dir: Optional[str] = None
    """Optional directory containing image-only img_tokens keyed by sample_id"""
    precomputed_img_preload: bool = False
    """Preload image-only feature shards into host memory before training"""
    precomputed_img_shard_cache_size: int = 8
    """Number of image-only shard files kept in memory"""
    expected_precomputed_img_mode: str = "image_only"
    """Expected precomputed image feature mode"""
    precomputed_img_sequential_sampling: bool = True
    """Use sequential dataset order when image cache is enabled to avoid random shard thrashing"""
    verify_precomputed_vl_features: bool = False
    """Verify cached V/L features by recomputing a few batches online before training"""
    verify_precomputed_vl_num_batches: int = 1
    """Number of batches to verify before training when cached V/L features are enabled"""
    verify_precomputed_vl_atol: float = 1e-4
    """Absolute tolerance used when comparing cached and online V/L features"""
    verify_precomputed_vl_rtol: float = 1e-4
    """Relative tolerance used when comparing cached and online V/L features"""

    # Checkpoint
    resume_from: Optional[str] = None
    resume_start_epoch: int = 0
    """Epoch index to start from when resuming epoch-based training"""
    resume_load_optimizer: bool = True
    """Load optimizer and LR scheduler states from resume_from"""

    # Language handling
    use_dummy_language: bool = False
    """If true, skip text encoder and use zero language embeddings"""
    gripper_threshold: float = 0.4
    """Threshold for binarizing gripper state when compressing 2D gripper"""

    # LeRobot data (optional)
    use_lerobot: bool = False
    """If true, load training data from LeRobot sim dataset"""
    lerobot_repo_id: Optional[str] = None
    """LeRobot dataset repo id (required if use_lerobot is True)"""
    lerobot_root: Optional[str] = None
    """LeRobot dataset root dir"""
    lerobot_cameras: Tuple[str, ...] = ("zed2i",)
    """Camera list for LeRobot inputs (single camera only)"""
    lerobot_image_size: Tuple[int, int] = (224, 224)
    """Resize (H, W) for LeRobot images"""
    lerobot_state_type: str = "qpos"
    """LeRobot state type: eepos, qpos, mixpos"""
    lerobot_include_depth: bool = False
    """Whether to include depth from LeRobot (TODO: wire into RDT)"""
    lerobot_depth_mode: str = "sim"
    """Depth decoding mode for LeRobot"""
    lerobot_video_backend: str = "torchcodec"
    """Video backend for LeRobot"""
    lerobot_tolerance_s: float = 0.05
    """LeRobot video timestamp tolerance (seconds)"""
    lerobot_task_description_file: Optional[str] = None
    """Optional task description mapping for LeRobot language prompts"""
    lerobot_eval_online: bool = False
    """Run online rollout eval when training with LeRobot data"""
    lerobot_use_paired_dataset: bool = False
    """Use HumanSimPairedDataset to provide language prompts (VLA mode)"""
    lerobot_human_root: Optional[str] = "/path/to/human_root"
    """Root directory for human LeRobot datasets when using paired dataset"""
    lerobot_sim_root: Optional[str] = "/path/to/sim_root"
    """Root directory for sim LeRobot datasets when using paired dataset"""
    lerobot_task_mapping_file: str = "examples/baselines/lerobot_dataset/task_mapping.json"
    """Task mapping file for paired dataset"""
    lerobot_human_dataset_file: str = "examples/baselines/lerobot_dataset/config/human_config.json"
    """Human dataset config file for paired dataset"""
    lerobot_sim_dataset_file: str = "examples/baselines/lerobot_dataset/config/sim_config.json"
    """Sim dataset config file for paired dataset"""
    lerobot_human_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    """Human task description file for paired dataset"""
    lerobot_sim_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"
    """Sim task description file for paired dataset"""
    lerobot_use_eval_processor: bool = True
    """Use HumanVideoSimEvaluateProcessor for task-based normalization"""
    lerobot_sim_pre_decode: bool = False
    """Pre-decode sim videos into an RDT-only cache before training"""
    lerobot_sim_pre_decode_cache_dir: str = "tmp/rdt_sim_video_cache"
    """Cache directory for pre-decoded sim videos"""
    lerobot_sim_pre_decode_num_workers: int = 8
    """Worker count used when building the sim video cache"""
    lerobot_enable_augmentation: bool = True
    """Enable LeRobot sim-side image augmentation during training"""


class FlattenRGBDSelectWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        camera_names,
        rgb=True,
        depth=True,
        state=True,
        sep_depth=True,
        state_type: str | None = None,
        expected_state_dim: int | None = None,
        gripper_threshold: float = 0.4,
        state_normalizer=None,
        normalization_method: str = "bounds_q99",
        dataset_idx: int = 0,
        image_size: tuple[int, int] | None = None,
        processor=None,
        sim_task_id: Optional[str] = None,
    ):
        self.base_env = env.unwrapped
        super().__init__(env)
        self.camera_names = list(camera_names)
        self.include_rgb = rgb
        self.include_depth = depth
        self.sep_depth = sep_depth
        self.include_state = state
        self.state_type = state_type
        self.expected_state_dim = expected_state_dim
        self.gripper_threshold = gripper_threshold
        self.state_normalizer = state_normalizer
        self.normalization_method = normalization_method
        self.dataset_idx = dataset_idx
        self.image_size = image_size
        self.processor = processor
        self.sim_task_id = sim_task_id or self._infer_sim_task_id()

        first_cam = None
        for name in self.camera_names:
            if name in self.base_env._init_raw_obs["sensor_data"]:
                first_cam = self.base_env._init_raw_obs["sensor_data"][name]
                break
        if first_cam is None:
            raise ValueError(f"No matching cameras found in env: {self.camera_names}")
        if "depth" not in first_cam:
            self.include_depth = False
        if "rgb" not in first_cam:
            self.include_rgb = False
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def _infer_sim_task_id(self) -> str:
        spec = getattr(self.base_env, "spec", None)
        if spec is not None and getattr(spec, "id", None):
            return spec.id
        for attr in ("env_id", "task_id", "task_name"):
            value = getattr(self.base_env, attr, None)
            if value:
                return str(value)
        return ""

    def _collect_state_by_key(self, state_dict, key):
        values = []
        if isinstance(state_dict, dict):
            for k, v in state_dict.items():
                if k == key:
                    values.append(v)
                elif isinstance(v, dict):
                    values.extend(self._collect_state_by_key(v, key))
        return values

    def _stack_state(self, values):
        if not values:
            return torch.empty((0,), device=self.base_env.device)
        chunks = []
        batch_dim = None
        for value in values:
            tensor = torch.as_tensor(value, device=self.base_env.device)
            if (
                self.expected_state_dim in (8, 16)
                and tensor.dim() >= 1
                and tensor.shape[-1] == 9
                and self.state_type == "qpos"
            ):
                arm = tensor[..., :7]
                gripper = tensor[..., 7:9].mean(dim=-1, keepdim=True)
                gripper = torch.where(
                    gripper > self.gripper_threshold,
                    torch.ones_like(gripper),
                    -torch.ones_like(gripper),
                )
                tensor = torch.cat([arm, gripper], dim=-1)
            if tensor.dim() == 0:
                tensor = tensor.view(1, 1)
            elif tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.reshape(tensor.shape[0], -1)
            if batch_dim is None:
                batch_dim = tensor.shape[0]
            elif tensor.shape[0] != batch_dim:
                tensor = tensor.reshape(batch_dim, -1)
            chunks.append(tensor)
        if not chunks:
            return torch.empty((batch_dim or 0, 0), device=self.base_env.device)
        return torch.cat(chunks, dim=1)

    def _extract_state(self, observation):
        if not self.state_type:
            return common.flatten_state_dict(
                observation, use_torch=True, device=self.base_env.device
            )
        state_chunks = []
        if self.state_type in ("qpos", "mixpos"):
            state_chunks.extend(self._collect_state_by_key(observation, "qpos"))
        if self.state_type in ("eepos", "mixpos"):
            state_chunks.extend(self._collect_state_by_key(observation, "eepos"))
        if not state_chunks:
            return common.flatten_state_dict(
                observation, use_torch=True, device=self.base_env.device
            )
        return self._stack_state(state_chunks)

    def _resize_hwc(self, img: torch.Tensor) -> torch.Tensor:
        if self.image_size is None:
            return img
        if img.dim() == 5 and img.shape[0] == 1:
            img = img[0]
        h, w = self.image_size
        if img.dim() == 2:
            img = img.unsqueeze(-1)
        if img.dim() == 3:
            img_chw = img.permute(2, 0, 1).unsqueeze(0)
            resized = torch.nn.functional.interpolate(
                img_chw, size=(h, w), mode="bilinear", align_corners=False
            )
            return resized.squeeze(0).permute(1, 2, 0)
        if img.dim() == 4:
            img_chw = img.permute(0, 3, 1, 2)
            resized = torch.nn.functional.interpolate(
                img_chw, size=(h, w), mode="bilinear", align_corners=False
            )
            return resized.permute(0, 2, 3, 1)
        return img

    def _process_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.float()
        rgb = self._resize_hwc(rgb)
        return rgb

    @staticmethod
    def _rgb_for_eval_processor(rgb: torch.Tensor) -> torch.Tensor:
        """HumanVideoSimEvaluateProcessor expects image input in 0-255 range."""
        rgb = rgb.float()
        if rgb.numel() > 0 and rgb.max() <= 1.5:
            rgb = rgb * 255.0
        return rgb

    @staticmethod
    def _rgb_for_agent(rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.float()
        if rgb.numel() > 0 and rgb.max() > 1.5:
            rgb = rgb / 255.0
        return rgb

    def _process_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.float()
        if depth.max() > 10.0:
            depth = depth / 1000.0
        depth = self._resize_hwc(depth)
        return depth

    def observation(self, observation):
        sensor_data = observation.pop("sensor_data")
        if "sensor_param" in observation:
            del observation["sensor_param"]
        rgb_images = []
        depth_images = []
        for cam_name in self.camera_names:
            if cam_name not in sensor_data:
                continue
            cam_data = sensor_data[cam_name]
            if self.include_rgb:
                rgb_images.append(self._process_rgb(cam_data["rgb"]))
            if self.include_depth:
                depth_images.append(self._process_depth(cam_data["depth"]))

        if len(rgb_images) > 0:
            rgb_images = torch.concat(rgb_images, axis=-1)
        if len(depth_images) > 0:
            depth_images = torch.concat(depth_images, axis=-1)
        observation = self._extract_state(observation)
        ret = {}
        if (
            self.processor is not None
            and self.include_state
            and self.include_rgb
            and isinstance(rgb_images, torch.Tensor)
        ):
            rgb_concat = rgb_images if isinstance(rgb_images, torch.Tensor) else None
            if rgb_concat is not None:
                rgb_concat = self._resize_hwc(rgb_concat.float())
                rgb_concat = self._rgb_for_eval_processor(rgb_concat)
            state_out, rgb_out = self.processor.normalize_state_rgb(
                observation, rgb_concat, sim_task_id=self.sim_task_id
            )
            ret["state"] = state_out
            if self.include_rgb and not self.include_depth:
                ret["rgb"] = rgb_out
            elif self.include_rgb and self.include_depth:
                if self.sep_depth:
                    ret["rgb"] = rgb_out
                    ret["depth"] = depth_images
                else:
                    ret["rgbd"] = torch.concat([rgb_out, depth_images], axis=-1)
            elif self.include_depth and not self.include_rgb:
                ret["depth"] = depth_images
        else:
            rgb_for_agent = (
                self._rgb_for_agent(rgb_images)
                if isinstance(rgb_images, torch.Tensor)
                else rgb_images
            )
            if self.include_state:
                if self.state_normalizer is not None:
                    ret["state"] = self.state_normalizer.normalize_state(
                        observation,
                        dataset_idx=self.dataset_idx,
                        method=self.normalization_method,
                    )
                else:
                    ret["state"] = observation
            if self.include_rgb and not self.include_depth:
                ret["rgb"] = rgb_for_agent
            elif self.include_rgb and self.include_depth:
                if self.sep_depth:
                    ret["rgb"] = rgb_for_agent
                    ret["depth"] = depth_images
                else:
                    ret["rgbd"] = torch.concat([rgb_for_agent, depth_images], axis=-1)
            elif self.include_depth and not self.include_rgb:
                ret["depth"] = depth_images
        return ret


class ActionDenormalizeWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env,
        normalizer,
        dataset_idx: int,
        method: str,
        act_dim: int,
        processor=None,
        sim_task_id: Optional[str] = None,
    ):
        super().__init__(env)
        self.normalizer = normalizer
        self.dataset_idx = dataset_idx
        self.method = method
        self.processor = processor
        self.sim_task_id = sim_task_id or self._infer_sim_task_id()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        self.single_action_space = self.action_space

    def _infer_sim_task_id(self) -> str:
        spec = getattr(self.env, "spec", None)
        if spec is not None and getattr(spec, "id", None):
            return spec.id
        for attr in ("env_id", "task_id", "task_name"):
            value = getattr(self.env, attr, None)
            if value:
                return str(value)
        return ""

    def action(self, action):
        action_tensor = torch.as_tensor(action)
        if self.processor is not None:
            denorm = self.processor.denormalize_action(
                action_tensor, sim_task_id=self.sim_task_id
            )
        else:
            denorm = self.normalizer.denormalize_action(
                action_tensor, dataset_idx=self.dataset_idx, method=self.method
            )
        return denorm.detach().cpu().numpy()


class RDTSimplifiedDataset(Dataset):
    """Simplified RDT Dataset for LoRA - Language-only, no video"""

    def __init__(
        self,
        demo_path: str,
        obs_process_fn,
        obs_space,
        obs_horizon: int,
        pred_horizon: int,
        include_rgb: bool,
        include_depth: bool,
        num_traj: Optional[int] = None,
    ):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.include_rgb = include_rgb
        self.include_depth = include_depth

        # Load trajectory data
        from examples.baselines.rdt.utils.utils import load_demo_dataset_with_lan
        trajectories = load_demo_dataset_with_lan(demo_path, num_traj=num_traj, concat=False)

        # Process observations
        print("Processing observations...")
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = self._reorder_keys(obs_traj_dict, obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(_obs_traj_dict["depth"].astype(np.float16))
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"])
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"])
            obs_traj_dict_list.append(_obs_traj_dict)

        trajectories["observations"] = obs_traj_dict_list

        # Process actions
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i])

        # Get language for each trajectory
        if "language" in trajectories:
            self.trajectory_language = trajectories["language"]
        else:
            # Fallback to default language
            self.trajectory_language = ["pick red cube and place on plate."] * len(trajectories["actions"])
            print("Warning: No language field in trajectories, using default language")

        # Create action padding for terminal states
        if "delta" in demo_path or "vel" in demo_path:
            self.pad_action = torch.zeros(trajectories["actions"][0].shape[1])
        else:
            raise NotImplementedError(f"Control mode not supported")

        # Create slices for data indexing
        self.slices = []
        for traj_idx in range(len(trajectories["actions"])):
            L = trajectories["actions"][traj_idx].shape[0]
            pad_before = obs_horizon - 1
            pad_after = pred_horizon - obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]

        self.trajectories = trajectories
        print(f"Dataset loaded: {len(self.slices)} samples from {len(trajectories['actions'])} trajectories")

    def _reorder_keys(self, d, ref_dict):
        """Reorder dictionary keys to match reference"""
        out = dict()
        for k, v in ref_dict.items():
            if k not in ['prompt']:
                if isinstance(v, dict) or isinstance(v, spaces.Dict):
                    out[k] = self._reorder_keys(d[k], ref_dict[k])
                else:
                    out[k] = d[k]
        return out

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L = self.trajectories["actions"][traj_idx].shape[0]

        # Get observations
        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start):start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)

        # Get actions
        act_seq = self.trajectories["actions"][traj_idx][max(0, start):end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            act_seq = torch.cat([act_seq, self.pad_action.repeat(end - L, 1)], dim=0)

        # Get language - directly use trajectory language
        language = self.trajectory_language[traj_idx]

        return {
            "observations": obs_seq,
            "actions": act_seq,
            "language": language,  # Directly use trajectory language
            "sample_id": f"traj{traj_idx}::{start}::{end}",
        }


def collate_fn_cpu(batch, *, precomputed_vl: Optional[PrecomputedVLFeatures] = None):
    """Collate function that keeps data on CPU"""
    collated = {}

    # Stack observations (keep on CPU)
    observations = {}
    for key in batch[0]["observations"].keys():
        observations[key] = torch.stack([item["observations"][key] for item in batch])
    collated["observations"] = observations

    # Stack actions (keep on CPU)
    collated["actions"] = torch.stack([item["actions"] for item in batch])

    # Language (text)
    collated["language"] = [item["language"] for item in batch]
    collated["sample_id"] = [str(item.get("sample_id", "")) for item in batch]

    return collated


def attach_precomputed_vl_to_batch(
    batch,
    precomputed_vl: Optional[PrecomputedVLFeatures],
    precomputed_img: Optional[PrecomputedVLFeatures] = None,
):
    """Attach cached V/L features in the main process.

    Passing a large PrecomputedVLFeatures object into worker-side collate
    functions is unsafe with forkserver/spawn because Python tries to pickle the
    full cache into every worker. T5-XXL language-only caches are large enough
    that worker startup can stall for many minutes and end with BrokenPipeError.
    """
    if precomputed_vl is None and precomputed_img is None:
        return batch

    sample_ids = [str(x) for x in batch.get("sample_id", [])]
    languages = batch.get("language", [])
    if not sample_ids or not all(sample_ids):
        return batch

    if precomputed_vl is not None:
        if not (
            "lang_embeds" in batch
            and "lang_mask" in batch
            and (precomputed_vl.metadata.get("feature_mode", "vl") != "vl" or "img_tokens" in batch)
        ):
            cached = precomputed_vl.get_batch(sample_ids, languages)
            if cached is not None:
                batch.update(cached)

    if precomputed_img is not None and "img_tokens" not in batch:
        missing_img_ids = [
            sample_id for sample_id in sample_ids if not precomputed_img.has_sample(sample_id)
        ]
        if missing_img_ids:
            preview = ", ".join(missing_img_ids[:5])
            raise FileNotFoundError(
                "Precomputed image cache is enabled but missing sample_id(s): "
                f"{preview}{' ...' if len(missing_img_ids) > 5 else ''}"
            )
        cached = precomputed_img.get_batch(sample_ids, languages)
        if cached is None or not (
            "img_tokens" in cached
            or ("img_tokens_q" in cached and "img_tokens_scale" in cached)
        ):
            raise KeyError("Precomputed image cache did not return image tokens for this batch.")
        if "img_tokens" in cached:
            batch["img_tokens"] = cached["img_tokens"]
        else:
            batch["img_tokens_q"] = cached["img_tokens_q"]
            batch["img_tokens_scale"] = cached["img_tokens_scale"]

    return batch


def move_batch_to_device(batch, device):
    """Move batch from CPU to device"""
    batch_device = {}

    # Move observations to device
    observations = {}
    for key, value in batch["observations"].items():
        observations[key] = value.to(device, non_blocking=True)
    batch_device["observations"] = observations

    # Move actions to device
    batch_device["actions"] = batch["actions"].to(device, non_blocking=True)

    # Language stays as list of strings
    batch_device["language"] = batch["language"]
    batch_device["sample_id"] = batch.get("sample_id", [])
    if "img_tokens" in batch:
        batch_device["img_tokens"] = batch["img_tokens"].to(device, non_blocking=True)
    if "img_tokens_q" in batch:
        batch_device["img_tokens_q"] = batch["img_tokens_q"].to(device, non_blocking=True)
    if "img_tokens_scale" in batch:
        batch_device["img_tokens_scale"] = batch["img_tokens_scale"].to(device, non_blocking=True)
    if "lang_embeds" in batch:
        batch_device["lang_embeds"] = batch["lang_embeds"].to(device, non_blocking=True)
    if "lang_mask" in batch:
        batch_device["lang_mask"] = batch["lang_mask"].to(device, non_blocking=True)

    return batch_device


class LoRARDTAgent(nn.Module):
    """RDT Agent with LoRA for efficient fine-tuning - Simplified version"""

    def __init__(self, args: LoRAArgs, env, device, precomputed_lang=None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from examples.baselines.rdt.models.multimodal_encoder.siglip_encoder import SiglipVisionTower
        from examples.baselines.rdt.models.multimodal_encoder.t5_encoder import T5Embedder
        from examples.baselines.rdt.models.rdt_runner import RDTRunner

        self.args = args
        self.device = device
        self.obs_horizon = args.obs_horizon
        self.pred_horizon = args.pred_horizon
        self.use_lerobot = args.use_lerobot

        # Get dimensions (fallback to vector env spaces if single_* is missing)
        action_space = getattr(env, "single_action_space", None) or env.action_space
        obs_space = getattr(env, "single_observation_space", None) or env.observation_space
        self.act_dim = action_space.shape[0]
        self.obs_state_dim = obs_space["state"].shape[1]

        # Visual channels
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()

        # Language embedding setup
        self.precomputed_lang = precomputed_lang
        self.use_dummy_language = args.use_dummy_language
        self.gripper_threshold = args.gripper_threshold
        self.rdt_slot_mapping = str(getattr(args, "rdt_slot_mapping", "official")).lower()
        if self.rdt_slot_mapping not in {"official", "legacy"}:
            raise ValueError(
                f"rdt_slot_mapping must be 'official' or 'legacy', got: {self.rdt_slot_mapping}"
            )
        print(f"Using RDT slot mapping: {self.rdt_slot_mapping}")

        # Initialize encoders (frozen)
        print("Initializing frozen SiGLIP vision encoder...")
        self.vision_encoder = SiglipVisionTower(
            vision_tower=args.vision_encoder,
            args=None
        )
        self.vision_encoder.vision_tower.to(device)
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        if self.use_dummy_language:
            print("Using dummy language embeddings (all zeros)...")
            self.text_embedder = None
        elif self.precomputed_lang is None:
            print("Initializing frozen T5 text encoder...")
            self.text_embedder = T5Embedder(
                from_pretrained=args.text_encoder,
                model_max_length=args.max_lang_len,
                device=device,
                local_files_only=True,
            )
            self.text_embedder.model.eval()
            for param in self.text_embedder.model.parameters():
                param.requires_grad = False
        else:
            print("Using precomputed language embeddings...")
            self.text_embedder = None

        img_token_dim = self.vision_encoder.hidden_size
        lang_dim_map = {
            "t5-v1_1-xxl": 4096,
            "t5-v1_1-base": 768,
            "t5-v1_1-small": 512,
        }
        if args.t5_version not in lang_dim_map:
            raise ValueError(f"Unsupported t5_version: {args.t5_version}")
        lang_token_dim = lang_dim_map[args.t5_version]
        self.lang_token_dim = lang_token_dim

        # Load pretrained model or initialize from scratch. Keep the local
        # RDTRunner wrapper so training and sampling use the original RDT
        # adaptor, action-mask, and diffusion-target logic.
        if args.pretrained_path and "/" in args.pretrained_path:
            print(f"Loading pretrained RDT model from {args.pretrained_path}...")

            config_path = os.path.join(args.pretrained_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    pretrained_config = json.load(f)
                pretrained_runner = RDTRunner.from_pretrained(
                    args.pretrained_path,
                    local_files_only=True,
                    action_dim=pretrained_config["action_dim"],
                    pred_horizon=pretrained_config["pred_horizon"],
                    config=pretrained_config,
                    lang_token_dim=pretrained_config["lang_token_dim"],
                    img_token_dim=pretrained_config["img_token_dim"],
                    state_token_dim=pretrained_config["state_token_dim"],
                    max_lang_cond_len=pretrained_config["max_lang_cond_len"],
                    img_cond_len=pretrained_config["img_cond_len"],
                    lang_pos_embed_config=pretrained_config.get("lang_pos_embed_config"),
                    img_pos_embed_config=pretrained_config.get("img_pos_embed_config"),
                    dtype=torch.bfloat16,
                )
            else:
                raise FileNotFoundError(
                    f"Missing config.json in pretrained_path: {args.pretrained_path}"
                )
            self.rdt_runner = pretrained_runner.to(device)
            self.rdt = self.rdt_runner.model

            self.pretrained_action_dim = self.rdt_runner.action_dim
            self.pretrained_hidden_size = self.rdt.hidden_size

            # Resize action positional embeddings if needed
            if self.rdt.horizon != args.pred_horizon:
                print(f"Resizing action positional embeddings from horizon {self.rdt.horizon} to {args.pred_horizon}")

                old_x_pos_embed = self.rdt.x_pos_embed.data
                new_horizon = args.pred_horizon

                # Keep timestep, freq, state embeddings (first 3)
                fixed_embed = old_x_pos_embed[:, :3, :]

                # Interpolate action embeddings
                old_action_embed = old_x_pos_embed[:, 3:, :].permute(0, 2, 1)
                new_action_embed = F.interpolate(
                    old_action_embed,
                    size=new_horizon,
                    mode='linear',
                    align_corners=True
                ).permute(0, 2, 1)

                new_x_pos_embed = torch.cat([fixed_embed, new_action_embed], dim=1)
                self.rdt.x_pos_embed = nn.Parameter(new_x_pos_embed)
                self.rdt.horizon = new_horizon
                self.rdt_runner.pred_horizon = new_horizon

                print(f"✓ Resized x_pos_embed from {old_x_pos_embed.shape} to {new_x_pos_embed.shape}")

            # Resize image positional embeddings if needed
            expected_img_tokens = args.obs_horizon * self.vision_encoder.num_patches
            pretrained_img_tokens = self.rdt.img_cond_pos_embed.shape[1]

            if pretrained_img_tokens != expected_img_tokens:
                print(f"Resizing img positional embeddings from {pretrained_img_tokens} to {expected_img_tokens} tokens")

                old_img_pos_embed = self.rdt.img_cond_pos_embed.data.permute(0, 2, 1)
                new_img_pos_embed = F.interpolate(
                    old_img_pos_embed,
                    size=expected_img_tokens,
                    mode='linear',
                    align_corners=True
                ).permute(0, 2, 1)

                self.rdt.img_cond_pos_embed = nn.Parameter(new_img_pos_embed)
                self.rdt.img_cond_len = expected_img_tokens

                print(f"✓ Resized img_cond_pos_embed from {pretrained_img_tokens} to {expected_img_tokens} tokens")

            print(f"✓ Loaded pretrained RDT: hidden_size={self.pretrained_hidden_size}, "
                  f"depth={len(self.rdt.blocks)}, action_dim={self.pretrained_action_dim}")

        else:
            # Initialize from scratch
            print("Initializing RDT model from scratch...")
            img_cond_len = args.obs_horizon * self.vision_encoder.num_patches

            scratch_config = {
                "lang_adaptor": "linear",
                "img_adaptor": "linear",
                "state_adaptor": "mlp3x_gelu",
                "rdt": {
                    "hidden_size": args.hidden_size,
                    "depth": args.depth,
                    "num_heads": args.num_heads,
                },
                "noise_scheduler": {
                    "num_train_timesteps": args.num_diffusion_iters,
                    "num_inference_timesteps": args.num_inference_steps,
                    "beta_schedule": "squaredcos_cap_v2",
                    "prediction_type": "epsilon",
                    "clip_sample": True,
                },
            }
            self.rdt_runner = RDTRunner(
                action_dim=self.act_dim,
                pred_horizon=args.pred_horizon,
                config=scratch_config,
                lang_token_dim=lang_token_dim,
                img_token_dim=img_token_dim,
                state_token_dim=self.act_dim,
                max_lang_cond_len=args.max_lang_len,
                img_cond_len=img_cond_len,
                lang_pos_embed_config=[("lang", -args.max_lang_len)],
                img_pos_embed_config=[("image", (args.obs_horizon, 1, -self.vision_encoder.num_patches))],
                dtype=torch.bfloat16,
            )
            self.rdt_runner = self.rdt_runner.to(device)
            self.rdt = self.rdt_runner.model
            self.pretrained_action_dim = self.act_dim
            self.pretrained_hidden_size = args.hidden_size

        self._align_runner_adapters(lang_token_dim, img_token_dim)

        # Apply LoRA to RDT
        print(f"Applying LoRA with rank={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=self._get_lora_target_modules(),
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
        )

        self.rdt = get_peft_model(self.rdt, lora_config)
        self.rdt_runner.model = self.rdt
        self.rdt.print_trainable_parameters()

        # Legacy modules kept only so old full checkpoints with these keys can be
        # loaded. The active train/inference path uses RDTRunner.state_adaptor for
        # both state and action tokens, matching the original RDT implementation.
        self.action_proj = nn.Linear(self.pretrained_action_dim, self.act_dim).to(device)
        self.action_embed = nn.Linear(self.pretrained_action_dim, self.pretrained_hidden_size).to(device)
        for legacy_module in (self.action_proj, self.action_embed):
            for param in legacy_module.parameters():
                param.requires_grad = False

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    @staticmethod
    def _first_linear_in_features(module: nn.Module) -> Optional[int]:
        if hasattr(module, "in_features"):
            return int(module.in_features)
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                return int(submodule.in_features)
        return None

    def _align_runner_adapters(self, lang_token_dim: int, img_token_dim: int) -> None:
        """Keep RDTRunner adapters as the single source of truth."""
        lang_adapter_in = self._first_linear_in_features(self.rdt_runner.lang_adaptor)
        if lang_adapter_in is None or lang_adapter_in != lang_token_dim:
            self.rdt_runner.lang_adaptor = nn.Linear(
                lang_token_dim, self.pretrained_hidden_size
            ).to(self.device)
            print(
                "Pretrained lang_adaptor dim mismatch; initialized new lang_adaptor "
                f"({lang_token_dim} -> {self.pretrained_hidden_size})."
            )
        else:
            self.rdt_runner.lang_adaptor = self.rdt_runner.lang_adaptor.to(self.device)

        img_adapter_in = self._first_linear_in_features(self.rdt_runner.img_adaptor)
        if img_adapter_in is None or img_adapter_in != img_token_dim:
            self.rdt_runner.img_adaptor = nn.Linear(
                img_token_dim, self.pretrained_hidden_size
            ).to(self.device)
            print(
                "Pretrained img_adaptor dim mismatch; initialized new img_adaptor "
                f"({img_token_dim} -> {self.pretrained_hidden_size})."
            )
        else:
            self.rdt_runner.img_adaptor = self.rdt_runner.img_adaptor.to(self.device)

        self.rdt_runner.state_adaptor = self.rdt_runner.state_adaptor.to(self.device)
        self.lang_adapter = self.rdt_runner.lang_adaptor
        self.img_adapter = self.rdt_runner.img_adaptor
        self.state_adapter = self.rdt_runner.state_adaptor
        print(
            "✓ Using RDTRunner adapters "
            f"(prediction_type={self.rdt_runner.prediction_type}, "
            f"train_timesteps={self.rdt_runner.num_train_timesteps}, "
            f"inference_timesteps={self.rdt_runner.num_inference_timesteps})"
        )

    def _get_lora_target_modules(self):
        """Get target modules for LoRA based on args"""
        target_modules = []

        module_patterns = {
            "q": ["q"],
            "kv": ["kv"],
            "proj": ["proj"],
            "ffn": ["fc1", "fc2"],
        }

        for module_name in self.args.lora_target_modules:
            if module_name in module_patterns:
                target_modules.extend(module_patterns[module_name])

        if not target_modules:
            target_modules = ["q", "kv", "proj"]

        print(f"LoRA target modules: {target_modules}")
        return target_modules

    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def encode_images_batch(self, rgb):
        """Optimized image encoding"""
        B = rgb.shape[0]
        rgb_flat = rgb.flatten(end_dim=1)

        # Use channels_last memory format for acceleration
        rgb_flat = rgb_flat.to(memory_format=torch.channels_last)

        # Resize
        if rgb_flat.shape[-2:] != (384, 384):
            rgb_flat = F.interpolate(
                rgb_flat, size=(384, 384),
                mode='bicubic',
                align_corners=False,
                antialias=False
            )

        img_tokens = self.vision_encoder(rgb_flat)
        return img_tokens.reshape(B, -1, img_tokens.shape[-1])

    def encode_language(self, lang_texts):
        """Encode language using precomputed or real-time encoding"""
        if self.use_dummy_language:
            embeddings = torch.zeros(
                len(lang_texts), self.args.max_lang_len, self.lang_token_dim, device=self.device
            )
            masks = torch.zeros(
                len(lang_texts), self.args.max_lang_len, dtype=torch.bool, device=self.device
            )
            return embeddings, masks
        if self.precomputed_lang:
            # Use precomputed/hybrid embedder
            return self.precomputed_lang.get_text_embeddings(lang_texts)
        else:
            # Use T5 directly
            with torch.no_grad():
                lang_embeds, lang_mask = self.text_embedder.get_text_embeddings(lang_texts)
            return lang_embeds, lang_mask

    def compute_condition_features(self, batch):
        obs_seq = batch["observations"]
        lang_texts = batch["language"]
        actions = batch["actions"]
        bsz = actions.shape[0]
        device = actions.device

        with torch.no_grad():
            lang_embeds, lang_mask = self.encode_language(lang_texts)
            lang_embeds = lang_embeds.float()
            lang_mask = lang_mask.bool()

            if self.include_rgb:
                rgb = obs_seq["rgb"].float()
                if not self.use_lerobot:
                    rgb = rgb / 255.0
                img_tokens = self.encode_images_batch(rgb).float()
            else:
                img_tokens = torch.zeros(
                    bsz,
                    self.args.obs_horizon * self.vision_encoder.num_patches,
                    self.vision_encoder.hidden_size,
                    device=device,
                )

        return img_tokens, lang_embeds, lang_mask

    @staticmethod
    def _dequantize_img_tokens(batch: dict, device: torch.device) -> Optional[torch.Tensor]:
        if "img_tokens_q" not in batch:
            return None
        if "img_tokens_scale" not in batch:
            raise KeyError("img_tokens_q is present but img_tokens_scale is missing.")
        q = batch["img_tokens_q"].to(device, non_blocking=True)
        scale = batch["img_tokens_scale"].to(device, non_blocking=True)
        return q.float() * scale.float().unsqueeze(1)

    def _compress_state_to_16(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] == 18:
            left_arm = state[..., :7]
            left_grip = state[..., 7:9].mean(dim=-1, keepdim=True)
            right_arm = state[..., 9:16]
            right_grip = state[..., 16:18].mean(dim=-1, keepdim=True)
            left_grip = torch.where(
                left_grip > self.gripper_threshold,
                torch.ones_like(left_grip),
                -torch.ones_like(left_grip),
            )
            right_grip = torch.where(
                right_grip > self.gripper_threshold,
                torch.ones_like(right_grip),
                -torch.ones_like(right_grip),
            )
            return torch.cat([left_arm, left_grip, right_arm, right_grip], dim=-1)
        return state

    def _map_actions_to_runner_space(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] == self.pretrained_action_dim:
            return actions

        unified_actions = torch.zeros(
            actions.shape[0],
            actions.shape[1],
            self.pretrained_action_dim,
            device=actions.device,
            dtype=actions.dtype,
        )
        if self.rdt_slot_mapping == "legacy":
            if actions.shape[-1] == 8 and self.pretrained_action_dim >= 24:
                unified_actions[:, :, 16:24] = actions
            elif actions.shape[-1] == 16 and self.pretrained_action_dim >= 24:
                unified_actions[:, :, 0:8] = actions[:, :, :8]
                unified_actions[:, :, 16:24] = actions[:, :, 8:16]
            else:
                raise ValueError(f"Unsupported action dim for legacy RDT mapping: {actions.shape[-1]}")
        elif actions.shape[-1] == 8 and self.pretrained_action_dim >= 11:
            unified_actions[:, :, 0:7] = actions[:, :, :7]
            unified_actions[:, :, 10] = actions[:, :, 7]
        elif actions.shape[-1] == 16 and self.pretrained_action_dim >= 61:
            unified_actions[:, :, 50:57] = actions[:, :, :7]
            unified_actions[:, :, 60] = actions[:, :, 7]
            unified_actions[:, :, 0:7] = actions[:, :, 8:15]
            unified_actions[:, :, 10] = actions[:, :, 15]
        else:
            raise ValueError(f"Unsupported action dim for RDT mapping: {actions.shape[-1]}")
        return unified_actions

    def _map_state_to_runner_space(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._compress_state_to_16(state)
        if state.shape[-1] == self.pretrained_action_dim:
            state_unified = state[:, -1:, :]
            state_mask = torch.ones_like(state_unified)
            return state_unified, state_mask

        state_unified = torch.zeros(
            state.shape[0], 1, self.pretrained_action_dim, device=state.device, dtype=state.dtype
        )
        state_mask = torch.zeros_like(state_unified)
        if self.rdt_slot_mapping == "legacy":
            if state.shape[-1] == 8 and self.pretrained_action_dim >= 24:
                state_unified[:, :, 16:24] = state[:, -1:, :8]
                state_mask[:, :, 16:24] = 1.0
            elif state.shape[-1] == 16 and self.pretrained_action_dim >= 24:
                state_unified[:, :, 0:8] = state[:, -1:, :8]
                state_unified[:, :, 16:24] = state[:, -1:, 8:16]
                state_mask[:, :, 0:8] = 1.0
                state_mask[:, :, 16:24] = 1.0
            else:
                raise ValueError(f"Unsupported state dim for legacy RDT mapping: {state.shape[-1]}")
        elif state.shape[-1] == 8 and self.pretrained_action_dim >= 11:
            state_unified[:, :, 0:7] = state[:, -1:, :7]
            state_unified[:, :, 10] = state[:, -1:, 7]
            state_mask[:, :, 0:7] = 1.0
            state_mask[:, :, 10] = 1.0
        elif state.shape[-1] == 16 and self.pretrained_action_dim >= 61:
            state_unified[:, :, 50:57] = state[:, -1:, :7]
            state_unified[:, :, 60] = state[:, -1:, 7]
            state_unified[:, :, 0:7] = state[:, -1:, 8:15]
            state_unified[:, :, 10] = state[:, -1:, 15]
            state_mask[:, :, 50:57] = 1.0
            state_mask[:, :, 60] = 1.0
            state_mask[:, :, 0:7] = 1.0
            state_mask[:, :, 10] = 1.0
        else:
            raise ValueError(f"Unsupported state dim for RDT mapping: {state.shape[-1]}")
        return state_unified, state_mask

    def _extract_actions_from_runner_space(self, actions: torch.Tensor) -> torch.Tensor:
        if self.act_dim == self.pretrained_action_dim:
            return actions
        if self.rdt_slot_mapping == "legacy":
            if self.act_dim == 8 and self.pretrained_action_dim >= 24:
                return actions[:, :, 16:24]
            if self.act_dim == 16 and self.pretrained_action_dim >= 24:
                left = actions[:, :, 0:8]
                right = actions[:, :, 16:24]
                return torch.cat([left, right], dim=-1)
            raise ValueError(f"Unsupported action dim for legacy RDT extraction: {self.act_dim}")
        if self.act_dim == 8 and self.pretrained_action_dim >= 11:
            return torch.cat([actions[:, :, 0:7], actions[:, :, 10:11]], dim=-1)
        if self.act_dim == 16 and self.pretrained_action_dim >= 61:
            left = torch.cat([actions[:, :, 50:57], actions[:, :, 60:61]], dim=-1)
            right = torch.cat([actions[:, :, 0:7], actions[:, :, 10:11]], dim=-1)
            return torch.cat([left, right], dim=-1)
        raise ValueError(f"Unsupported action dim for RDT extraction: {self.act_dim}")

    def _control_freqs(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch_size,), float(self.args.control_frequency), device=device)

    def compute_loss(self, batch):
        """Compute training loss through the original RDTRunner path."""
        obs_seq = batch["observations"]
        actions = batch["actions"]
        lang_texts = batch["language"]  # Directly use trajectory language

        B = actions.shape[0]
        device = actions.device

        action_gt = self._map_actions_to_runner_space(actions)

        if all(key in batch for key in ("lang_embeds", "lang_mask")):
            lang_embeds = batch["lang_embeds"].to(device).float()
            lang_mask = batch["lang_mask"].to(device).bool()
        else:
            with torch.no_grad():
                lang_embeds, lang_mask = self.encode_language(lang_texts)
                lang_embeds = lang_embeds.float()
                lang_mask = lang_mask.bool()

        quantized_img_tokens = self._dequantize_img_tokens(batch, device)
        if quantized_img_tokens is not None:
            img_tokens = quantized_img_tokens
        elif "img_tokens" in batch:
            img_tokens = batch["img_tokens"].to(device).float()
        else:
            with torch.no_grad():
                if self.include_rgb:
                    rgb = obs_seq["rgb"].float()
                    if not self.use_lerobot:
                        rgb = rgb / 255.0
                    img_tokens = self.encode_images_batch(rgb).float()
                else:
                    img_tokens = torch.zeros(
                        B,
                        self.args.obs_horizon * self.vision_encoder.num_patches,
                        self.vision_encoder.hidden_size,
                        device=device,
                    )

        state_tokens, action_mask = self._map_state_to_runner_space(obs_seq["state"])
        ctrl_freqs = self._control_freqs(B, device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return self.rdt_runner.compute_loss(
                lang_tokens=lang_embeds,
                lang_attn_mask=lang_mask,
                img_tokens=img_tokens,
                state_tokens=state_tokens,
                action_gt=action_gt,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
            )

    @torch.no_grad()
    def get_action(self, obs, language_prompt=None):
        """Generate action based on observation and language prompt"""
        if isinstance(obs, dict):
            state = obs["state"]
            rgb = obs.get("rgb", None)
        else:
            state = obs
            rgb = None

        B = state.shape[0]
        device = state.device

        # Prepare language
        if language_prompt is None:
            language_prompt = "pick red cube and place on plate."
        
        if isinstance(language_prompt, str):
            lang_texts = [language_prompt] * B
        else:
            lang_texts = language_prompt

        with torch.no_grad():
            # Encode language
            lang_embeds, lang_mask = self.encode_language(lang_texts)
            lang_embeds = lang_embeds.float()
            lang_mask = lang_mask.bool()

            # Encode images through SiGLIP
            if self.include_rgb and rgb is not None:
                rgb_processed = rgb.float()
                if not self.use_lerobot:
                    rgb_processed = rgb_processed / 255.0

                if rgb_processed.shape[-1] > 3:
                    rgb_processed = rgb_processed[..., :3]

                rgb_processed = rgb_processed.permute(0, 1, 4, 2, 3)
                img_tokens = self.encode_images_batch(rgb_processed)
            else:
                img_tokens = torch.zeros(
                    B, self.args.obs_horizon * self.vision_encoder.num_patches,
                    self.vision_encoder.hidden_size
                ).to(device)

            state_tokens, action_mask = self._map_state_to_runner_space(state)
            ctrl_freqs = self._control_freqs(B, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                unified_actions = self.rdt_runner.predict_action(
                    lang_tokens=lang_embeds,
                    lang_attn_mask=lang_mask,
                    img_tokens=img_tokens,
                    state_tokens=state_tokens,
                    action_mask=action_mask,
                    ctrl_freqs=ctrl_freqs,
                )

            return self._extract_actions_from_runner_space(unified_actions.float())

    def save_lora_weights(self, save_path):
        """Save only the LoRA weights"""
        os.makedirs(save_path, exist_ok=True)

        self.rdt.save_pretrained(save_path)

        torch.save({
            'lang_adapter': self.lang_adapter.state_dict(),
            'img_adapter': self.img_adapter.state_dict(),
            'state_adapter': self.state_adapter.state_dict(),
        }, os.path.join(save_path, 'custom_adapters.pt'))

        print(f"✓ Saved LoRA weights to {save_path}")


def save_checkpoint(run_name, tag, agent, optimizer, lr_scheduler, iteration, best_metrics, args=None):
    """Save training checkpoint"""
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save({
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "iteration": iteration,
        "best_metrics": dict(best_metrics),
        "args": vars(args) if args is not None else None,
    }, f"runs/{run_name}/checkpoints/{tag}.pt")


def load_checkpoint(path, agent, optimizer, lr_scheduler, device, load_optimizer: bool = True):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    agent.load_state_dict(checkpoint["agent"])
    if load_optimizer and optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if load_optimizer and lr_scheduler is not None and "lr_scheduler" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    iteration = checkpoint.get("iteration", 0)
    best_metrics = checkpoint.get("best_metrics", defaultdict(float))

    print(f"Loaded checkpoint from iteration {iteration}")
    if not load_optimizer:
        print("Skipped optimizer/LR scheduler state loading; using fresh optimizer state.")
    return iteration, best_metrics


def verify_precomputed_vl_features(
    agent,
    dataloader,
    device,
    *,
    precomputed_vl: Optional[PrecomputedVLFeatures],
    num_batches: int,
    atol: float,
    rtol: float,
) -> None:
    if num_batches <= 0:
        return

    print(
        f"Verifying precomputed V/L features on {num_batches} batch(es) with atol={atol} rtol={rtol}..."
    )
    checked = 0
    agent.eval()
    for raw_batch in dataloader:
        raw_batch = attach_precomputed_vl_to_batch(raw_batch, precomputed_vl)
        batch = move_batch_to_device(raw_batch, device)
        if not all(key in batch for key in ("lang_embeds", "lang_mask")):
            raise ValueError(
                "Verification requested but dataloader batch does not contain cached lang_embeds/lang_mask."
            )
        if "img_tokens" in batch:
            online_img_tokens, online_lang_embeds, online_lang_mask = agent.compute_condition_features(batch)
        else:
            with torch.no_grad():
                online_lang_embeds, online_lang_mask = agent.encode_language(batch["language"])
            online_lang_embeds = online_lang_embeds.float()
            online_lang_mask = online_lang_mask.bool()
            online_img_tokens = None
        cached_lang_embeds = batch["lang_embeds"].float()
        cached_lang_mask = batch["lang_mask"].bool()

        if cached_lang_embeds.shape != online_lang_embeds.shape:
            raise ValueError(
                f"Cached lang_embeds shape mismatch: cached={tuple(cached_lang_embeds.shape)} "
                f"online={tuple(online_lang_embeds.shape)}"
            )
        if cached_lang_mask.shape != online_lang_mask.shape:
            raise ValueError(
                f"Cached lang_mask shape mismatch: cached={tuple(cached_lang_mask.shape)} "
                f"online={tuple(online_lang_mask.shape)}"
            )
        if not torch.equal(cached_lang_mask, online_lang_mask):
            raise ValueError("Cached lang_mask does not match online-computed lang_mask.")
        if not torch.allclose(cached_lang_embeds, online_lang_embeds, atol=atol, rtol=rtol):
            diff = (cached_lang_embeds - online_lang_embeds).abs().max().item()
            raise ValueError(f"Cached lang_embeds mismatch detected, max_abs_diff={diff:.6g}")
        if "img_tokens" in batch:
            cached_img_tokens = batch["img_tokens"].float()
            if online_img_tokens is None:
                raise ValueError("Internal verification error: online img_tokens were not computed.")
            if cached_img_tokens.shape != online_img_tokens.shape:
                raise ValueError(
                    f"Cached img_tokens shape mismatch: cached={tuple(cached_img_tokens.shape)} "
                    f"online={tuple(online_img_tokens.shape)}"
                )
            if not torch.allclose(cached_img_tokens, online_img_tokens, atol=atol, rtol=rtol):
                diff = (cached_img_tokens - online_img_tokens).abs().max().item()
                raise ValueError(f"Cached img_tokens mismatch detected, max_abs_diff={diff:.6g}")

        checked += 1
        if checked >= num_batches:
            break

    if checked < num_batches:
        raise ValueError(f"Requested {num_batches} verification batches, but only checked {checked}.")
    agent.train()
    print(f"Verified precomputed V/L features on {checked} batch(es).")


if __name__ == "__main__":
    args = tyro.cli(LoRAArgs)
    sim_task_id_for_mapping = args.env_id
    base_env_id = _extract_base_env_id(args.env_id)
    requested_level = _extract_l_level_from_task_id(sim_task_id_for_mapping)
    _set_l_level_flags(requested_level)

    # Setup
    if args.exp_name is None:
        args.exp_name = f"rdt_lora_simplified_{args.env_id}"
    run_name = f"{args.exp_name}_{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    lerobot_dataset = None
    action_denorm = None
    eval_processor = None
    if args.use_lerobot:
        include_depth = bool(args.lerobot_include_depth)
        if include_depth:
            # TODO: wire depth into RDT (vision encoder expects RGB-only today).
            include_depth = False

        if args.lerobot_use_paired_dataset:
            from examples.baselines.rdt.utils.rdt_lerobot_dataset import (
                RDTHumanSimPairedDataset,
                RDTPairedDatasetConfig,
            )

            paired_config = RDTPairedDatasetConfig(
                human_root=args.lerobot_human_root or args.lerobot_root or "demos",
                sim_root=args.lerobot_sim_root or args.lerobot_root or "demos",
                task_mapping_file=args.lerobot_task_mapping_file,
                human_dataset_file=args.lerobot_human_dataset_file,
                sim_dataset_file=args.lerobot_sim_dataset_file,
                human_task_description_file=args.lerobot_human_task_description_file,
                sim_task_description_file=args.lerobot_sim_task_description_file,
                split="train",
                cameras=["zed2i"],
                include_depth=include_depth,
                image_size=tuple(args.lerobot_image_size),
                horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                state_type=args.lerobot_state_type,
                fps=30,
                video_backend=args.lerobot_video_backend,
                input_mode="language_only",
                include_first_frame=False,
                enable_augmentation=args.lerobot_enable_augmentation,
                sim_pre_decode=args.lerobot_sim_pre_decode,
                sim_pre_decode_cache_dir=args.lerobot_sim_pre_decode_cache_dir,
                sim_pre_decode_num_workers=args.lerobot_sim_pre_decode_num_workers,
                sim_skip_image_loading=bool(args.precomputed_img_features_dir),
            )
            lerobot_dataset = RDTHumanSimPairedDataset(paired_config)
            dataset = lerobot_dataset
            sim_dataset = lerobot_dataset.sim_dataset
            action_denorm = {
                "normalizer": sim_dataset.normalizer,
                "dataset_idx": 0,
                "method": sim_dataset.config.normalization_method,
                "act_dim": sim_dataset.action_dim,
            }
        else:
            if args.lerobot_repo_id is None:
                raise ValueError("lerobot_repo_id is required when use_lerobot is True")
            from examples.baselines.rdt.utils.rdt_lerobot_dataset import (
                RDTLeRobotSimDataConfig,
                RDTLeRobotSimDataset,
            )

            lerobot_cameras = list(args.lerobot_cameras) if args.lerobot_cameras else ["zed2i"]
            lerobot_cameras = ["zed2i"]  # enforce single camera as requested

            lerobot_config = RDTLeRobotSimDataConfig(
                root=args.lerobot_root or "./data/lerobot",
                repo_id=args.lerobot_repo_id,
                split="train",
                image_size=tuple(args.lerobot_image_size),
                state_type=args.lerobot_state_type,
                include_depth=include_depth,
                cameras=lerobot_cameras,
                horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                tolerance_s=args.lerobot_tolerance_s,
                depth_mode=args.lerobot_depth_mode,
                video_backend=args.lerobot_video_backend,
                dataset_file=None,
                task_description_file=args.lerobot_task_description_file,
                enable_augmentation=args.lerobot_enable_augmentation,
                pre_decode=args.lerobot_sim_pre_decode,
                pre_decode_cache_dir=args.lerobot_sim_pre_decode_cache_dir,
                pre_decode_num_workers=args.lerobot_sim_pre_decode_num_workers,
                skip_image_loading=bool(args.precomputed_img_features_dir),
            )
            lerobot_dataset = RDTLeRobotSimDataset(lerobot_config)
            dataset = lerobot_dataset
            action_denorm = {
                "normalizer": lerobot_dataset.normalizer,
                "dataset_idx": 0,
                "method": lerobot_dataset.config.normalization_method,
                "act_dim": lerobot_dataset.action_dim,
            }
        lerobot_sim_dataset = (
            lerobot_dataset.sim_dataset if args.lerobot_use_paired_dataset else lerobot_dataset
        )
        if args.lerobot_eval_online and args.lerobot_use_eval_processor:
            from examples.baselines.lerobot_dataset.evaluate_processor import (
                HumanVideoSimEvaluateProcessor,
                HumanVideoSimEvaluateProcessorConfig,
            )

            eval_processor = HumanVideoSimEvaluateProcessor(
                HumanVideoSimEvaluateProcessorConfig(
                    human_root=args.lerobot_human_root or args.lerobot_root or "demos",
                    human_split="train",
                    human_dataset_file=args.lerobot_human_dataset_file,
                    human_task_description_file=args.lerobot_human_task_description_file,
                    human_cameras=["zed2i"],
                    human_include_depth=include_depth,
                    human_image_size=tuple(args.lerobot_image_size),
                    human_video_backend=args.lerobot_video_backend,
                    sim_root=args.lerobot_sim_root or args.lerobot_root or "demos",
                    sim_split="train",
                    sim_dataset_file=args.lerobot_sim_dataset_file,
                    sim_task_description_file=args.lerobot_sim_task_description_file,
                    sim_state_type=args.lerobot_state_type,
                    sim_single_arm=False,
                    normalization_method=lerobot_sim_dataset.config.normalization_method,
                    task_mapping_file=args.lerobot_task_mapping_file,
                )
            )
        include_rgb = True
        include_depth = False
    elif not args.no_eval:
        # Setup environment
        from mani_skill.utils import common
        from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
        from examples.baselines.rdt.utils.make_env import make_eval_envs
        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode=args.reward_mode,
            obs_mode=args.obs_mode,
            render_mode="rgb_array",
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            max_episode_steps=args.max_episode_steps,
        )

        tmp_env = gym.make(args.env_id, **env_kwargs)
        original_obs_space = tmp_env.observation_space
        include_rgb = "rgb" in args.obs_mode
        include_depth = "depth" in args.obs_mode
        tmp_env.close()

        obs_process_fn = partial(
            convert_obs,
            concat_fn=partial(np.concatenate, axis=-1),
            transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
            state_obs_extractor=build_state_obs_extractor(args.env_id),
            depth=include_depth,
        )

        dataset = RDTSimplifiedDataset(
            demo_path=args.demo_path,
            obs_process_fn=obs_process_fn,
            obs_space=original_obs_space,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            include_rgb=include_rgb,
            include_depth=include_depth,
            num_traj=args.num_demos,
        )

    # Setup environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)

    class _DummyVecEnv:
        def __init__(self, obs_horizon, state_dim, act_dim, image_h, image_w, num_views=1):
            self.single_observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        -float("inf"),
                        float("inf"),
                        shape=(obs_horizon, state_dim),
                        dtype=np.float32,
                    ),
                    "rgb": spaces.Box(
                        0,
                        255,
                        shape=(obs_horizon, image_h, image_w, 3 * num_views),
                        dtype=np.uint8,
                    ),
                }
            )
            self.single_action_space = spaces.Box(
                -1.0,
                1.0,
                shape=(act_dim,),
                dtype=np.float32,
            )

        def close(self):
            pass

    if args.use_lerobot and not args.lerobot_eval_online:
        num_views = 1
        image_h, image_w = args.lerobot_image_size

        envs = _DummyVecEnv(
            args.obs_horizon,
            lerobot_sim_dataset.state_dim,
            lerobot_sim_dataset.action_dim,
            image_h,
            image_w,
            num_views,
        )
    else:
        if not args.no_eval:
            from mani_skill.utils import common  # noqa: F401
            from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
            from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
            from examples.baselines.rdt.utils.make_env import make_eval_envs

        wrappers = []
        if not args.no_eval and not (args.use_lerobot and args.lerobot_eval_online):
            wrappers = [FlattenRGBDObservationWrapper]
        if not args.no_eval and args.use_lerobot and args.lerobot_eval_online:
            dataset_idx_for_eval = 0
            if not args.lerobot_use_eval_processor:
                try:
                    for idx, info in lerobot_sim_dataset.normalizer.dataset_info.items():
                        if info.get("repo_id", "").endswith(args.env_id):
                            dataset_idx_for_eval = idx
                            break
                except Exception:
                    pass
            wrappers = [
                lambda env: FlattenRGBDSelectWrapper(
                    env,
                    ["zed2i"],
                    state_type=args.lerobot_state_type,
                    expected_state_dim=lerobot_sim_dataset.state_dim,
                    state_normalizer=lerobot_sim_dataset.normalizer,
                    normalization_method=lerobot_sim_dataset.config.normalization_method,
                    dataset_idx=dataset_idx_for_eval,
                    image_size=tuple(args.lerobot_image_size),
                    depth=include_depth,
                    processor=eval_processor,
                    sim_task_id=sim_task_id_for_mapping,
                ),
                FlattenActionSpaceWrapper,
            ]
            if action_denorm is not None:
                wrappers.append(
                    lambda env: ActionDenormalizeWrapper(
                        env,
                        action_denorm["normalizer"],
                        dataset_idx=dataset_idx_for_eval,
                        method=action_denorm["method"],
                        act_dim=action_denorm["act_dim"],
                        processor=eval_processor,
                        sim_task_id=sim_task_id_for_mapping,
                    )
                )

        envs = make_eval_envs(
            base_env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs,
            video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
            wrappers=wrappers,
            l_level=requested_level,
        ) if not args.no_eval else _DummyVecEnv(
            args.obs_horizon,
            lerobot_sim_dataset.state_dim,
            lerobot_sim_dataset.action_dim,
            args.lerobot_image_size[0],
            args.lerobot_image_size[1],
        )
        if action_denorm is not None:
            action_space = gym.spaces.Box(
                -1.0, 1.0, shape=(action_denorm["act_dim"],), dtype=np.float32
            )
            envs.single_action_space = action_space
            envs.action_space = action_space
            if hasattr(envs, "envs") and envs.envs:
                for sub_env in envs.envs:
                    if getattr(sub_env, "single_action_space", None) is None:
                        sub_env.single_action_space = action_space
        if getattr(envs, "single_observation_space", None) is None:
            envs.single_observation_space = envs.observation_space

    # Setup precomputed language embeddings (optional)
    precomputed_lang = None
    if args.use_precomputed_lang and args.precomputed_lang_dir:
        precomputed_lang = HybridLangEmbedding(
            precomputed_dir=args.precomputed_lang_dir,
            text_encoder_path=args.text_encoder,
            device=device,
            max_length=args.max_lang_len,
            t5_version=args.t5_version,
        )
        print(f"Precomputed language embedding stats: {precomputed_lang.get_stats()}")

    precomputed_vl = None
    if args.use_precomputed_vl_features and args.precomputed_vl_dir:
        precomputed_vl = PrecomputedVLFeatures(
            args.precomputed_vl_dir,
            device="cpu",
            preload_to_memory=args.precomputed_vl_preload,
            shard_cache_size=args.precomputed_vl_shard_cache_size,
        )
        precomputed_vl.validate_metadata(build_precomputed_vl_expected_metadata(args))
        if args.expected_precomputed_vl_mode is not None:
            expected_mode = str(args.expected_precomputed_vl_mode)
            if expected_mode not in ("vl", "language_only"):
                raise ValueError(
                    "expected_precomputed_vl_mode must be 'vl' or 'language_only', "
                    f"got {expected_mode!r}"
                )
            actual_mode = precomputed_vl.metadata.get("feature_mode", "vl")
            if actual_mode != expected_mode:
                raise ValueError(
                    "Precomputed V/L mode mismatch: "
                    f"actual={actual_mode!r} expected={expected_mode!r} "
                    f"for directory {args.precomputed_vl_dir}"
                )
        print(f"Precomputed V/L feature stats: {precomputed_vl.get_stats()}")

    precomputed_img = None
    if args.precomputed_img_features_dir:
        precomputed_img = PrecomputedVLFeatures(
            args.precomputed_img_features_dir,
            device="cpu",
            preload_to_memory=args.precomputed_img_preload,
            shard_cache_size=args.precomputed_img_shard_cache_size,
        )
        img_expected_args = copy(args)
        img_expected_args.expected_precomputed_vl_mode = None
        precomputed_img.validate_metadata(build_precomputed_vl_expected_metadata(img_expected_args))
        expected_img_mode = str(args.expected_precomputed_img_mode)
        if expected_img_mode != "image_only":
            raise ValueError(
                "expected_precomputed_img_mode must be 'image_only', "
                f"got {expected_img_mode!r}"
            )
        actual_img_mode = precomputed_img.metadata.get("feature_mode", "vl")
        if actual_img_mode != expected_img_mode:
            raise ValueError(
                "Precomputed image mode mismatch: "
                f"actual={actual_img_mode!r} expected={expected_img_mode!r} "
                f"for directory {args.precomputed_img_features_dir}"
            )
        print(f"Precomputed image feature stats: {precomputed_img.get_stats()}")

    def _load_task_desc_prompts(desc_path: str | None) -> list[str]:
        if not desc_path:
            return []
        try:
            with open(desc_path, "r") as f:
                data = json.load(f)
        except Exception:
            return []
        prompts: list[str] = []
        for value in data.values():
            if isinstance(value, str):
                prompts.append(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        prompts.append(item)
                    elif isinstance(item, list):
                        prompts.extend([v for v in item if isinstance(v, str)])
        return [p for p in prompts if p]

    def _load_paired_eval_prompts_for_env(sim_task_id: str) -> list[str]:
        if eval_processor is not None:
            return eval_processor.get_language(
                sim_task_id,
                num_envs=max(args.num_eval_envs, args.num_eval_episodes),
            )

        mapping_path = args.lerobot_task_mapping_file
        desc_path = (
            args.lerobot_human_task_description_file
            or args.lerobot_task_description_file
        )
        if not mapping_path or not desc_path:
            return []

        try:
            with open(mapping_path, "r") as f:
                mapping_data = json.load(f)
            with open(desc_path, "r") as f:
                desc_data = json.load(f)
        except Exception:
            return []

        human_task_id = None
        for mapping in mapping_data.get("task_mappings", []):
            if sim_task_id in mapping.get("sim_task_id", []):
                human_task_id = mapping.get("human_task_id")
                break

        if human_task_id is None:
            return []

        prompts = desc_data.get(human_task_id, [])
        if isinstance(prompts, str):
            prompts = [prompts]
        return [p for p in prompts if isinstance(p, str) and p]

    eval_language_prompts = None
    if args.use_lerobot and args.lerobot_eval_online:
        if args.lerobot_use_paired_dataset:
            # Resolve prompts from the current eval env id so L0/L1/L2/L3 stay distinct.
            eval_language_prompts = _load_paired_eval_prompts_for_env(sim_task_id_for_mapping)
        else:
            desc_path = args.lerobot_task_description_file
            eval_language_prompts = _load_task_desc_prompts(desc_path)

    # Setup logging
    writer = SummaryWriter(f"runs/{run_name}")

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
            tags=["lora", "parameter-efficient", "simplified"],
        )

    use_img_cache_sequential_sampling = (
        precomputed_img is not None and args.precomputed_img_sequential_sampling
    )
    if use_img_cache_sequential_sampling:
        print("Using sequential dataloader sampling for image-cache locality.")

    # Create dataloader
    sampler = (
        SequentialSampler(dataset)
        if use_img_cache_sequential_sampling
        else RandomSampler(dataset, replacement=False)
    )
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    if not args.use_epoch_training:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters, start_iter=0)

    sim_repo_ids = None
    sim_task_descriptions = None
    if args.use_lerobot:
        if args.lerobot_use_paired_dataset:
            sim_repo_ids = tuple(
                ds.repo_id for ds in lerobot_dataset.sim_dataset.lerobot_dataset.datasets
            )
            sim_task_descriptions = dict(lerobot_dataset.sim_dataset.task_descriptions)
        else:
            sim_task_descriptions = dict(getattr(lerobot_dataset, "task_descriptions", {}))

    collate_lerobot = partial(
        collate_fn_lerobot,
        use_paired_dataset=args.lerobot_use_paired_dataset,
        sim_repo_ids=sim_repo_ids,
        sim_task_descriptions=sim_task_descriptions,
    )
    collate_cpu = collate_fn_cpu
    dataloader_worker_init = partial(worker_init_fn, base_seed=args.seed)
    multiprocessing_context = (
        args.dataloader_multiprocessing_context
        if args.num_dataload_workers > 0 and args.dataloader_multiprocessing_context
        else None
    )
    prefetch_factor = (
        max(1, int(args.dataloader_prefetch_factor))
        if args.num_dataload_workers > 0
        else None
    )

    if args.use_epoch_training:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=not use_img_cache_sequential_sampling,
            num_workers=args.num_dataload_workers,
            worker_init_fn=dataloader_worker_init,
            persistent_workers=(args.num_dataload_workers > 0),
            collate_fn=collate_lerobot if args.use_lerobot else collate_cpu,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_dataload_workers,
            worker_init_fn=dataloader_worker_init,
            persistent_workers=(args.num_dataload_workers > 0),
            collate_fn=collate_lerobot if args.use_lerobot else collate_cpu,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
        )

    # Create LoRA agent
    print("Creating LoRA RDT agent...")
    agent = LoRARDTAgent(args, envs, device, precomputed_lang)
    if precomputed_vl is not None and args.verify_precomputed_vl_features:
        verify_precomputed_vl_features(
            agent,
            dataloader,
            device,
            precomputed_vl=precomputed_vl,
            num_batches=args.verify_precomputed_vl_num_batches,
            atol=args.verify_precomputed_vl_atol,
            rtol=args.verify_precomputed_vl_rtol,
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, agent.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if args.use_epoch_training:
        total_train_batches = len(dataloader) * args.total_epochs
    else:
        total_train_batches = args.total_iters
    total_optimizer_steps = max(1, math.ceil(total_train_batches / args.gradient_accumulation_steps))
    warmup_steps = min(args.lr_warmup_steps, total_optimizer_steps)
    print(
        f"LR scheduler: {args.lr_scheduler}, optimizer_steps={total_optimizer_steps}, "
        f"warmup_steps={warmup_steps}, lr={args.lr}"
    )
    from diffusers.optimization import get_scheduler

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    # Resume
    start_iter = 0
    best_metrics = defaultdict(float)

    if args.resume_from and os.path.exists(args.resume_from):
        start_iter, best_metrics = load_checkpoint(
            args.resume_from,
            agent,
            optimizer,
            lr_scheduler,
            device,
            load_optimizer=args.resume_load_optimizer,
        )

    # Training loop with gradient accumulation
    agent.train()

    # Checkpoint save function for evaluation
    def save_checkpoint_fn(run_name, tag, iteration, best_metrics):
        save_checkpoint(run_name, tag, agent, optimizer, lr_scheduler, iteration, best_metrics, args=args)
        # Also save LoRA weights
        agent.save_lora_weights(f"runs/{run_name}/lora_weights/{tag}")

    if args.use_epoch_training:
        global_step = start_iter
        start_epoch = int(args.resume_start_epoch)
        if start_epoch < 0:
            raise ValueError(f"resume_start_epoch must be non-negative, got {start_epoch}")
        if start_epoch > args.total_epochs:
            raise ValueError(
                f"resume_start_epoch ({start_epoch}) cannot exceed total_epochs ({args.total_epochs})"
            )
        for epoch in tqdm(range(start_epoch, args.total_epochs), desc="Epochs"):
            epoch_loss = 0.0
            num_batches = 0
            accumulation_step = 0
            optimizer.zero_grad()

            for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
                batch = attach_precomputed_vl_to_batch(batch, precomputed_vl, precomputed_img)
                batch = move_batch_to_device(batch, device)

                loss = agent.compute_loss(batch)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

                accumulation_step += 1
                if accumulation_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, agent.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    accumulation_step = 0

                epoch_loss += loss.item()
                num_batches += 1

                if global_step % args.log_freq == 0:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    if args.track:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "iteration": global_step,
                        }, step=global_step)

                global_step += 1

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            writer.add_scalar("train_epoch/loss", avg_epoch_loss, epoch)

            if (not args.use_lerobot or args.lerobot_eval_online) and args.eval_epoch_freq > 0 and not args.no_eval:
                if epoch % args.eval_epoch_freq == 0 and epoch > 0:
                    from examples.baselines.rdt.utils.eval import evaluate_and_save_best

                    best_metrics = evaluate_and_save_best(
                        iteration=epoch,
                        agent=agent,
                        eval_envs=envs,
                        device=device,
                        sim_backend=args.sim_backend,
                        precomputed_lang=precomputed_lang,
                        language_prompts=eval_language_prompts,
                        gripper_binary_action=args.gripper_binary_action,
                        gripper_threshold=args.gripper_threshold,
                        gripper_indices=list(args.gripper_indices),
                        num_eval_episodes=args.num_eval_episodes,
                        eval_freq=1,
                        best_eval_metrics=best_metrics,
                        save_checkpoint_fn=save_checkpoint_fn,
                        run_name=run_name,
                        writer=writer,
                    )

            if args.save_epoch_freq > 0 and (epoch % args.save_epoch_freq == 0 or epoch == args.total_epochs - 1):
                save_checkpoint(
                    run_name,
                    f"epoch_{epoch}",
                    agent,
                    optimizer,
                    lr_scheduler,
                    global_step,
                    best_metrics,
                    args=args,
                )
                agent.save_lora_weights(f"runs/{run_name}/lora_weights/epoch_{epoch}")

        save_checkpoint(
            run_name,
            "final",
            agent,
            optimizer,
            lr_scheduler,
            global_step,
            best_metrics,
            args=args,
        )
        agent.save_lora_weights(f"runs/{run_name}/lora_weights/final")
    else:
        pbar = tqdm(total=args.total_iters, initial=start_iter)
        accumulation_step = 0
        optimizer.zero_grad()

        for iteration, batch in enumerate(dataloader, start=start_iter):
            # Move batch from CPU to device
            batch = attach_precomputed_vl_to_batch(batch, precomputed_vl, precomputed_img)
            batch = move_batch_to_device(batch, device)

            # Compute loss (don't clear gradients)
            loss = agent.compute_loss(batch)
            loss = loss / args.gradient_accumulation_steps  # Normalize
            loss.backward()

            accumulation_step += 1

            # Update every N steps
            if accumulation_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, agent.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accumulation_step = 0

            # Logging
            if iteration % args.log_freq == 0:
                writer.add_scalar("train/loss", loss.item(), iteration)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)

                if args.track:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "iteration": iteration,
                    }, step=iteration)

            # Evaluation and best model saving
            if (not args.use_lerobot or args.lerobot_eval_online) and not args.no_eval:
                from examples.baselines.rdt.utils.eval import evaluate_and_save_best

                best_metrics = evaluate_and_save_best(
                    iteration=iteration,
                    agent=agent,
                    eval_envs=envs,
                    device=device,
                    sim_backend=args.sim_backend,
                    precomputed_lang=precomputed_lang,
                    language_prompts=eval_language_prompts,
                    gripper_binary_action=args.gripper_binary_action,
                    gripper_threshold=args.gripper_threshold,
                    gripper_indices=list(args.gripper_indices),
                    num_eval_episodes=args.num_eval_episodes,
                    eval_freq=args.eval_freq,
                    best_eval_metrics=best_metrics,
                    save_checkpoint_fn=save_checkpoint_fn,
                    run_name=run_name,
                    writer=writer,
                )

            # Periodic checkpoint saving
            if args.save_freq and iteration % args.save_freq == 0 and iteration > 0:
                save_checkpoint(
                    run_name,
                    f"iter_{iteration}",
                    agent,
                    optimizer,
                    lr_scheduler,
                    iteration,
                    best_metrics,
                    args=args,
                )
                agent.save_lora_weights(f"runs/{run_name}/lora_weights/iter_{iteration}")

            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Final save
        save_checkpoint(
            run_name,
            "final",
            agent,
            optimizer,
            lr_scheduler,
            args.total_iters,
            best_metrics,
            args=args,
        )
        agent.save_lora_weights(f"runs/{run_name}/lora_weights/final")

    if envs is not None and not args.no_eval:
        envs.close()
    writer.close()
    print("✓ LoRA training completed!")
