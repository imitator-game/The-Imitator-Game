"""
LeRobot-compatible ManiSkill Simulation dataloader with ActionNormalizer integration.
This dataloader loads data from LeRobot format (Parquet/Videos) which was converted
using the utils/real2lerobot.py script.
"""

import os
import logging
import json
import bisect
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
import re

from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset
from examples.baselines.lerobot_dataset.utils import (
    create_output_dir,
    print_header,
    print_dataset_info,
    print_sample_info,
    visualize_images,
    visualize_states,
    visualize_actions,
    visualize_batch_grid,
    print_summary,
)
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer

logger = logging.getLogger(__name__)

def get_episdoes_idx(sub_episodes: str) -> List[int]:
    """
    Get episode indices from string like "0: 45"
    """
    if ':' not in sub_episodes:
        return eval(sub_episodes)

    start, end = map(int, sub_episodes.split(":"))
    return list(range(start, end))

def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB to HSV"""
    ftype = np.float32
    output = np.zeros_like(rgb, dtype=ftype)
    h, s, v = np.split(output, 3, -1)
    h = h.squeeze(-1)
    s = s.squeeze(-1)
    v = v.squeeze(-1)

    rgb_amax = rgb.argmax(-1)
    rgb_max = rgb.max(-1)
    rgb_min = rgb.min(-1)

    r = (rgb_max - rgb_min).astype(ftype)
    ok = r > 0

    m = ok & (rgb_amax == 0); h[m] = 0+(rgb[m, 1] - rgb[m, 2]) / r[m]
    m = ok & (rgb_amax == 1); h[m] = 2+(rgb[m, 2] - rgb[m, 0]) / r[m]
    m = ok & (rgb_amax == 2); h[m] = 4+(rgb[m, 0] - rgb[m, 1]) / r[m]

    h[:] *= 60
    h[h < 0] += 360
    s[ok] = r[ok] / rgb_max[ok]
    v[:] = rgb_max

    return np.stack((h, s, v), -1)

def decode_depth(rgb: np.ndarray, cam_name: str, depth_mode: str) -> np.ndarray:
    """Decompress RGB to depth based on zip2lerobot_single.py"""
    if np.issubdtype(rgb.dtype, np.unsignedinteger):
        rgb = rgb.astype(np.float32) / 255.0

    hsv = rgb2hsv(rgb)

    max_hue = 300.0
    min_s = 0.1
    min_v = 0.1
    err_depth = 0.0 # or np.nan
    if depth_mode == "robot":
        if "zed" in cam_name.lower():
            zrange = (0.0, 0.5)
        else:
            zrange = (0.0, 4.0)
    else:
        if "wrist" in cam_name.lower() or "wristcam" in cam_name.lower():
            zrange = (0.0, 1.0)
        elif "cam2" in cam_name.lower():
            zrange = (0.0, 2.0)
        else:
            zrange = (0.0, 3.0)
    ok = (hsv[..., 1] >= min_s) & (hsv[..., 2] >= min_v) & (hsv[..., 0] <= max_hue)

    depth = np.full(rgb.shape[:-1], err_depth, dtype=np.float32)
    depth[ok] = hsv[ok, 0] / max_hue
    depth = depth * (zrange[1] - zrange[0]) + zrange[0]

    return depth




@dataclass
class LeRobotSimDataConfig:
    """Configuration for LeRobot-format ManiSkill simulation dataset loading."""

    root: str = "./data/lerobot"
    split: str = "train"
    debug: bool = False

    # Image processing
    image_size: Tuple[int, int] = (224, 224)

    # Robot configuration
    state_type: str = "qpos"  # "eepos", "qpos", or "mixpos"
    include_depth: bool = False
    cameras: List[str] = field(default_factory=lambda: ["cam1", "cam2"])
    single_arm: bool = False
    depth_mode: str = "sim"
    # Action sequence parameters
    horizon: int = 16
    obs_horizon: int = 1

    # Dataset dimensions
    state_dim: int = 18
    action_dim: int = 16

    # LeRobot-specific / additional parameters
    repo_id: Optional[str] = None
    fps: int = 30  # FPS for calculating time offsets
    video_backend: str = "torchcodec"
    tolerance_s: float = 0.0001

    # Normalization
    normalization_method: str = "bounds_q99"  # "bounds" or "bounds_q99"

    # Compatibility with RealDataConfig
    split_file: Optional[str] = None
    task_description_file: Optional[str] = None
    dataset_file: Optional[str] = None

    enable_augmentation: bool = True

    # For *Skill Model Training
    skill: bool = False 
    xskill: bool = False
    robot_frame_gap: int = 35

class LeRobotSimDataset(Dataset):
    """LeRobot-compatible ManiSkill simulation dataset with ActionNormalizer integration."""

    def __init__(self, config: LeRobotSimDataConfig):
        self.config = config

        if LeRobotDataset is None:
            raise ImportError("lerobot is not installed. Cannot use LeRobotSimDataset.")

        # Determine root for LeRobotDataset
        self.root = config.root

        self.horizon = config.horizon
        self.cameras = config.cameras
        self.image_size = config.image_size

        if config.state_type == "eepos":
            self.state_key = "observation.eepos_gripper_states"
            self.action_key = "action.eepos_gripper_actions"
        elif config.state_type == "qpos":
            self.state_key = "observation.qpos_gripper_states"
            self.action_key = "action.qpos_gripper_actions"
        elif config.state_type == "mixpos":
            self.state_key = "observation.eepos_gripper_states"
            self.action_key = "action.qpos_gripper_actions"
        else:
            raise ValueError(f"Unknown state_type: {config.state_type}")

        # Define delta_timestamps to avoid loading unnecessary data
        action_offsets = [i / config.fps for i in range(self.horizon)]
        delta_timestamps = {
            self.action_key: action_offsets,
        }

        # Define used camera keys for stripping
        used_cam_keys = []
        for cam in self.cameras:
            used_cam_keys.append(f"observation.images.{cam}")
            if config.include_depth:
                used_cam_keys.append(f"observation.images.{cam}_depth")

        sub_datasets = []
        self.task_descriptions = {}
        if self.config.dataset_file:
            with open(self.config.dataset_file, "r") as f:
                dataset_configs = json.load(f)
            if self.config.task_description_file:
                with open(self.config.task_description_file, "r") as f:
                    self.task_descriptions = json.load(f)
            for ds_cfg in tqdm(dataset_configs):
                repo_id = ds_cfg.get("repo_id")
                ds_root = os.path.join(self.root, ds_cfg.get("root"))
                sub_episodes = ds_cfg.get(self.config.split)  # Load all if not specified
                sub_episodes = get_episdoes_idx(sub_episodes)

                print(f'\033[94mloading episodes {sub_episodes} from {ds_root} with split {self.config.split}\033[0m')

                sub_ds = LeRobotDataset(
                    repo_id=repo_id,
                    root=ds_root,
                    delta_timestamps=delta_timestamps,
                    video_backend=config.video_backend,
                    tolerance_s=config.tolerance_s,
                    episodes=sub_episodes,
                )
                sub_datasets.append(sub_ds)
        else:
            if not self.config.repo_id:
                raise ValueError("repo_id is required when dataset_file is not provided")
            sub_ds = LeRobotDataset(
                repo_id=self.config.repo_id,
                root=self.root,
                delta_timestamps=delta_timestamps,
                video_backend=config.video_backend,
                tolerance_s=config.tolerance_s,
            )
            sub_datasets.append(sub_ds)

        for sub_ds in sub_datasets:
            # remove camera features from meta so they won't be fetched/decoded, but keep the ones we use
            for k in list(sub_ds.meta.features.keys()):
                if sub_ds.meta.features[k]["dtype"] in ("image", "video", "depth"):
                    if k not in used_cam_keys:
                        sub_ds.meta.features.pop(k)

            cols_to_remove = [
                c for c in sub_ds.hf_dataset.column_names
                if c.startswith("observation.images.") and c not in used_cam_keys
            ]
            if cols_to_remove:
                print(f'removing columns {cols_to_remove}')
                sub_ds.hf_dataset = sub_ds.hf_dataset.remove_columns(cols_to_remove)

        self.lerobot_dataset = ConcatDataset(sub_datasets)
        self.main_dataset = sub_datasets[0]  # For dimensions and stats fallback

        # Initialize ActionNormalizer and populate with dataset stats
        self.normalizer = ActionNormalizer()
        for idx, sub_ds in enumerate(sub_datasets):
            self.normalizer.add_dataset_stats(
                dataset_idx=idx,
                repo_id=sub_ds.repo_id,
                stats=sub_ds.meta.stats,
                state_key=self.state_key,
                action_key=self.action_key,
                single_arm=config.single_arm
            )

        # Augmentation pipeline (matching RealRobotDataset)
        self._setup_transforms_sim(config)

        # LeRobotDataset handles indexing and padding automatically
        self.total_transitions = len(self.lerobot_dataset)
        self.target_length = self.total_transitions

        # Stats
        self.stats = self.main_dataset.meta.stats

        self._episode_start_cache = {}

        for ds_idx, ds in enumerate(self.lerobot_dataset.datasets):
            episode_indices = ds.hf_dataset["episode_index"]

            last_ep = None
            for local_idx, ep in enumerate(episode_indices):
                ep = int(ep)
                if ep != last_ep:
                    self._episode_start_cache[(ds_idx, ep)] = local_idx
                    last_ep = ep

        # Discover dimensions
        try:
            state_dim = self.main_dataset.features[self.state_key]["shape"][0]
            action_dim = self.main_dataset.features[self.action_key]["shape"][0]
            if self.config.single_arm:
                state_dim = state_dim // 2
                action_dim = action_dim // 2
            self.state_dim = state_dim
            self.action_dim = action_dim
            logger.info(f"Discovered dimensions - state: {self.state_dim}, action: {self.action_dim}")
        except (KeyError, AttributeError):
            self.state_dim = config.state_dim
            self.action_dim = config.action_dim
            logger.warning(f"Could not discover dimensions, using config: state={self.state_dim}, action={self.action_dim}")

        logger.info(f"Initialized LeRobotSimDataset with {self.total_transitions} valid transitions")

    def __len__(self) -> int:
        return self.target_length

    def _setup_transforms_sim(self, config):
        """Call this inside LeRobotSimDataset.__init__ instead of old transform setup."""
        additional_targets = {}
        if config.include_depth:
            additional_targets['depth'] = 'image'

        if config.enable_augmentation:
            self.transform = A.Compose([
                A.OneOf([
                    A.Compose([
                        A.ShiftScaleRotate(
                            shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
                        ),
                        A.Resize(height=self.image_size[0], width=self.image_size[1], p=1.0),
                    ]),
                ], p=1.0),
            ], additional_targets=additional_targets)

            self.rgb_transform = A.Compose([
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5
                ),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1], p=1.0),
            ], additional_targets=additional_targets)

            self.rgb_transform = A.Compose([
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                ToTensorV2(),
            ])

        self.depth_transform = A.Compose([
            ToTensorV2(),
        ])

    def _extract_right_arm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts right arm data from [right_arm, left_arm, right_gripper, left_gripper] layout.
        """
        if not self.config.single_arm:
            return x
        M = x.shape[-1] // 2
        # Right arm: indices 0 to M-2, and 2*M-2
        return torch.cat([x[..., :M-1], x[..., 2*M-2 : 2*M-1]], dim=-1)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = idx % self.total_transitions

        # 1. Determine which dataset this sample comes from
        dataset_idx = bisect.bisect_right(self.lerobot_dataset.cumulative_sizes, global_idx)
        if dataset_idx == 0:
            sample_idx = global_idx
        else:
            sample_idx = global_idx - self.lerobot_dataset.cumulative_sizes[dataset_idx - 1]
        curr_dataset = self.lerobot_dataset.datasets[dataset_idx]
        obs_horizon = max(int(self.config.obs_horizon), 1)
        start_idx = max(sample_idx - obs_horizon + 1, 0)
        frame_indices = list(range(start_idx, sample_idx + 1))
        if len(frame_indices) < obs_horizon:
            frame_indices = [frame_indices[0]] * (obs_horizon - len(frame_indices)) + frame_indices
        frames = [curr_dataset[i] for i in frame_indices]

        # Task description from task_desc.json using repo_id
        repo_id = frames[-1]['task']
        task_description = self.task_descriptions.get(repo_id)

        skill_frames = None
        if self.config.skill:
            robot_frame_gap = max(int(self.config.robot_frame_gap), 1)

            skill_first_idx = sample_idx
            skill_first_frame = curr_dataset[skill_first_idx]
            skill_first_ep = (
                skill_first_frame["episode_index"].item()
                if hasattr(skill_first_frame["episode_index"], "item")
                else skill_first_frame["episode_index"]
            )

            # ----- New xskill mode -----
            if self.config.xskill:
                # 1. Find the start index of the current episode
                ep_start_idx = self._episode_start_cache[(dataset_idx, int(skill_first_ep))]

                # 2. Go back 30 frames (but not past the start of the episode)
                start_idx = max(ep_start_idx, sample_idx - 30)

                # 3. Evenly sample 4 frames (including start_idx and sample_idx)
                indices = np.linspace(start_idx, sample_idx, 4).round().astype(int)
                # Deduplicate (e.g., duplicates occur when start_idx == sample_idx)
                unique_indices = np.unique(indices)
                if len(unique_indices) < 4:
                    # Pad with the first frame to meet the required count (satisfies the "repeat first frame" requirement)
                    first_idx = unique_indices[0]
                    indices = np.concatenate([unique_indices, [first_idx] * (4 - len(unique_indices))])
                else:
                    indices = unique_indices

                # Fetch frames
                skill_frames = [curr_dataset[idx] for idx in indices]

            # ----- Original skill logic (two-frame sampling) -----
            else:
                skill_last_idx = min(sample_idx + robot_frame_gap, len(curr_dataset) - 1)
                skill_last_frame = curr_dataset[skill_last_idx]
                skill_last_ep = (
                    skill_last_frame["episode_index"].item()
                    if hasattr(skill_last_frame["episode_index"], "item")
                    else skill_last_frame["episode_index"]
                )

                while skill_last_idx > skill_first_idx and skill_last_ep != skill_first_ep:
                    skill_last_idx -= 1
                    skill_last_frame = curr_dataset[skill_last_idx]
                    skill_last_ep = (
                        skill_last_frame["episode_index"].item()
                        if hasattr(skill_last_frame["episode_index"], "item")
                        else skill_last_frame["episode_index"]
                    )
                skill_frames = [skill_first_frame, skill_last_frame]

        # 2. Process images and depth
        all_view_tensors = []
        skill_view_tensors = []
        for cam_name in self.cameras:
            rgb_key = f"observation.images.{cam_name}"
            depth_key = f"observation.images.{cam_name}_depth"
            if rgb_key not in frames[-1]:
                continue

            per_frame_views = []
            for frame in frames:
                rgb_tensor = frame[rgb_key]
                if rgb_tensor.dim() == 4:
                    rgb_tensor = rgb_tensor[0]
                rgb_img = (rgb_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                depth_img = None
                if self.config.include_depth and depth_key in frame:
                    depth_rgb_tensor = frame[depth_key]
                    if depth_rgb_tensor.dim() == 4:
                        depth_rgb_tensor = depth_rgb_tensor[0]
                    depth_rgb_img = (depth_rgb_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    depth_img = decode_depth(depth_rgb_img, cam_name, self.config.depth_mode)
                    depth_img = depth_img[..., None]

                if depth_img is not None:
                    res = self.transform(image=rgb_img, depth=depth_img)
                    aug_rgb = res["image"]
                    aug_depth = res["depth"]
                else:
                    res = self.transform(image=rgb_img)
                    aug_rgb = res["image"]
                    aug_depth = None

                res_rgb = self.rgb_transform(image=aug_rgb)
                tensor_rgb = res_rgb["image"]

                if aug_depth is not None:
                    tensor_depth = self.depth_transform(image=aug_depth)["image"]
                    view_tensor = torch.cat([tensor_rgb, tensor_depth], dim=0)
                else:
                    view_tensor = tensor_rgb

                per_frame_views.append(view_tensor)

            all_view_tensors.append(torch.stack(per_frame_views, dim=0))

            # -------- extra skill frames processing --------
            if self.config.skill:
                per_skill_views = []
                for frame in skill_frames:
                    if rgb_key not in frame:
                        continue

                    rgb_tensor = frame[rgb_key]
                    if rgb_tensor.dim() == 4:
                        rgb_tensor = rgb_tensor[0]
                    rgb_img = (rgb_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    depth_img = None
                    if self.config.include_depth and depth_key in frame:
                        depth_rgb_tensor = frame[depth_key]
                        if depth_rgb_tensor.dim() == 4:
                            depth_rgb_tensor = depth_rgb_tensor[0]
                        depth_rgb_img = (depth_rgb_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        depth_img = decode_depth(depth_rgb_img, cam_name, self.config.depth_mode)
                        depth_img = depth_img[..., None]

                    if depth_img is not None:
                        res = self.transform(image=rgb_img, depth=depth_img)
                        aug_rgb = res["image"]
                        aug_depth = res["depth"]
                    else:
                        res = self.transform(image=rgb_img)
                        aug_rgb = res["image"]
                        aug_depth = None

                    res_rgb = self.rgb_transform(image=aug_rgb)
                    tensor_rgb = res_rgb["image"]

                    if aug_depth is not None:
                        tensor_depth = self.depth_transform(image=aug_depth)["image"]
                        view_tensor = torch.cat([tensor_rgb, tensor_depth], dim=0)
                    else:
                        view_tensor = tensor_rgb

                    per_skill_views.append(view_tensor)

                if len(per_skill_views) > 0:
                    skill_view_tensors.append(torch.stack(per_skill_views, dim=0))

        # 3. State
        state_seq = []
        for frame in frames:
            state = frame[self.state_key]
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state)
            if state.dim() == 2:
                state = state[0]
            state_seq.append(self._extract_right_arm(state))
        state = torch.stack(state_seq, dim=0)

        # 4. Action Sequence
        action_sequence = frames[-1][self.action_key]
        if not isinstance(action_sequence, torch.Tensor):
            action_sequence = torch.tensor(action_sequence)
        # Extract right arm if needed
        action_sequence = self._extract_right_arm(action_sequence)

        # 5. Normalization using ActionNormalizer
        state = self.normalizer.normalize_state(
            state,
            dataset_idx,
            method=self.config.normalization_method
        )
        action_sequence = self.normalizer.normalize_action(
            action_sequence,
            dataset_idx,
            method=self.config.normalization_method
        )

        ret_dict = {
            "states": state.float(),
            "actions": action_sequence.float(),
            "task_name": re.sub(r"^L[0-9]_", "", repo_id),
            "repo_id": repo_id,
            "task_descriptions": str(task_description),
            # "task_ids": torch.tensor(0, dtype=torch.long),
            # Add metadata for denormalization
            "dataset_idx": torch.tensor(dataset_idx, dtype=torch.long),
        }

        for i, tensor in enumerate(all_view_tensors):
            ret_dict[f"view_{i+1}"] = tensor

        if self.config.skill and len(skill_view_tensors) > 0:
            ret_dict["skill_frames"] = {
                f"view_{i+1}": tensor for i, tensor in enumerate(skill_view_tensors)
            }

        return ret_dict

    def denormalize_action(
        self,
        normalized_action: torch.Tensor,
        dataset_idx: int,
        method: Optional[str] = None
    ) -> torch.Tensor:
        """
        Denormalize action back to original scale.

        Args:
            normalized_action: Normalized action tensor in [-1, 1]
            dataset_idx: Which dataset's stats to use
            method: Denormalization method (defaults to config.normalization_method)

        Returns:
            Denormalized action in original scale
        """
        if method is None:
            method = self.config.normalization_method
        return self.normalizer.denormalize_action(normalized_action, dataset_idx, method)

    def denormalize_state(
        self,
        normalized_state: torch.Tensor,
        dataset_idx: int,
        method: Optional[str] = None
    ) -> torch.Tensor:
        """
        Denormalize state back to original scale.

        Args:
            normalized_state: Normalized state tensor in [-1, 1]
            dataset_idx: Which dataset's stats to use
            method: Denormalization method (defaults to config.normalization_method)

        Returns:
            Denormalized state in original scale
        """
        if method is None:
            method = self.config.normalization_method
        return self.normalizer.denormalize_state(normalized_state, dataset_idx, method)

    def __del__(self):
        pass
