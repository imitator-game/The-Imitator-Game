"""
LeRobot-compatible real-robot dataloader aligned with LeRobotSimDataset.

Key alignment changes vs the original:
  1. Added obs_horizon support (manual consecutive-frame stacking, same as sim)
  2. Added tolerance_s config field
  3. Fixed task_description_file: None-guarded open()
  4. Fixed self.main_dataset → self.train_dataset attribute name
  5. Fixed repo_id = frame['task'] → curr_dataset.repo_id
  6. Metadata-based indexing in __init__ (no O(N) __getitem__ scan)
  7. decode_depth now uses depth_mode field (same signature as sim)
  8. Dataset feature stripping matches sim pattern
  9. Fixed self.root usage inside __init__: use config.root directly

Robot-specific behaviour preserved:
  - State/action key mapping differs from sim:
      qpos  : observation.qpos_gripper_states  / action.qpos_gripper_actions
      eepos : observation.eepos_gripper_states / action.eepos_gripper_actions
      mixpos: observation.eepos_gripper_states / action.qpos_gripper_actions
  - depth_mode default "robot" (different zrange from sim)
  - No 'task' field in LeRobot frames for real-robot data; uses task_descriptions file
"""

import bisect
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Depth helpers (shared with sim dataset, depth_mode switches zrange)
# ---------------------------------------------------------------------------

def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    ftype = np.float32
    output = np.zeros_like(rgb, dtype=ftype)
    h, s, v = np.split(output, 3, -1)
    h = h.squeeze(-1); s = s.squeeze(-1); v = v.squeeze(-1)

    rgb_amax = rgb.argmax(-1)
    rgb_max  = rgb.max(-1)
    rgb_min  = rgb.min(-1)
    r        = (rgb_max - rgb_min).astype(ftype)
    ok       = r > 0

    m = ok & (rgb_amax == 0); h[m] = 0 + (rgb[m, 1] - rgb[m, 2]) / r[m]
    m = ok & (rgb_amax == 1); h[m] = 2 + (rgb[m, 2] - rgb[m, 0]) / r[m]
    m = ok & (rgb_amax == 2); h[m] = 4 + (rgb[m, 0] - rgb[m, 1]) / r[m]

    h[:] *= 60
    h[h < 0] += 360
    s[ok] = r[ok] / rgb_max[ok]
    v[:] = rgb_max
    return np.stack((h, s, v), -1)


def decode_depth(rgb: np.ndarray, cam_name: str, depth_mode: str) -> np.ndarray:
    """Decode hue-encoded depth image.  Same API as lerobot_sim_dataset."""
    if np.issubdtype(rgb.dtype, np.unsignedinteger):
        rgb = rgb.astype(np.float32) / 255.0

    hsv = rgb2hsv(rgb)
    max_hue, min_s, min_v = 300.0, 0.1, 0.1
    err_depth = 0.0

    if depth_mode == "robot":
        zrange = (0.0, 0.5) if "zed" in cam_name.lower() else (0.0, 4.0)
    else:  # sim
        if "wrist" in cam_name.lower() or "wristcam" in cam_name.lower():
            zrange = (0.0, 1.0)
        elif "cam2" in cam_name.lower():
            zrange = (0.0, 2.0)
        else:
            zrange = (0.0, 3.0)

    ok    = (hsv[..., 1] >= min_s) & (hsv[..., 2] >= min_v) & (hsv[..., 0] <= max_hue)
    depth = np.full(rgb.shape[:-1], err_depth, dtype=np.float32)
    depth[ok] = hsv[ok, 0] / max_hue
    depth = depth * (zrange[1] - zrange[0]) + zrange[0]
    return depth


def get_episodes_idx(sub_episodes: str) -> Optional[List[int]]:
    """Parse episode spec: "0:45", "[0,1,2]", "0:" (→ None = load all)."""
    s = sub_episodes.strip()
    if ":" in s:
        parts = s.split(":", 1)
        start_s, end_s = parts[0].strip(), parts[1].strip()
        start = int(start_s) if start_s else 0
        if not end_s:
            return None            # "0:" or ":" → load all
        return list(range(start, int(end_s)))
    return eval(s)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LeRobotRobotDataConfig:
    """Configuration for LeRobot-format real-robot dataset loading."""

    root: str = "demos"
    split: str = "train"
    debug: bool = False

    # Image processing
    image_size: Tuple[int, int] = (224, 224)

    # Robot-specific state type
    state_type: str = "qpos"           # "eepos" | "qpos" | "mixpos"
    include_depth: bool = True
    depth_mode: str = "robot"          # "robot" | "sim"
    cameras: List[str] = field(default_factory=lambda: ["cam1", "cam2"])
    single_arm: bool = False

    # Action / observation sequence parameters
    horizon: int = 16                  # action prediction horizon
    obs_horizon: int = 1               # number of stacked observation frames

    # Dataset dimensions (fallback when auto-detection fails)
    state_dim: int = 16
    action_dim: int = 16

    # LeRobot loading parameters
    repo_id: Optional[str] = None
    fps: int = 30
    video_backend: str = "torchcodec"
    tolerance_s: float = 0.0001

    # Normalization
    normalization_method: str = "bounds_q99"

    # Dataset file (JSON list of sub-dataset configs)
    dataset_file: Optional[str] = None
    task_description_file: Optional[str] = None

    enable_augmentation: bool = True

    skill: bool = False
    xskill: bool = False
    robot_frame_gap: int = 35


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LeRobotRobotDataset(Dataset):
    """LeRobot-compatible real-robot dataset aligned with LeRobotSimDataset."""

    def __init__(self, config: LeRobotRobotDataConfig):
        self.config = config

        self.horizon = config.horizon
        self.cameras = config.cameras
        self.image_size = config.image_size

        # ── State / action key mapping (robot-specific, kept as-is) ─────────
        if config.state_type == "eepos":
            self.state_key  = "observation.eepos_gripper_states"
            self.action_key = "action.eepos_gripper_actions"
        elif config.state_type == "qpos":
            self.state_key  = "observation.qpos_gripper_states"
            self.action_key = "action.qpos_gripper_actions"
        elif config.state_type == "mixpos":
            self.state_key  = "observation.eepos_gripper_states"
            self.action_key = "action.qpos_gripper_actions"
        else:
            raise ValueError(f"Unknown state_type: {config.state_type!r}")

        # ── delta_timestamps: only action horizon; state is loaded manually ─
        action_offsets   = [i / config.fps for i in range(self.horizon)]
        delta_timestamps = {self.action_key: action_offsets}

        # ── Camera keys to keep (strip the rest from meta & HF dataset) ─────
        used_cam_keys: List[str] = []
        for cam in self.cameras:
            used_cam_keys.append(f"observation.images.{cam}")
            if config.include_depth:
                used_cam_keys.append(f"observation.images.{cam}_depth")

        # ── Load task descriptions ────────────────────────────────────────────
        self.task_descriptions: Dict[str, str] = {}
        if config.task_description_file:
            with open(config.task_description_file, "r") as f:
                self.task_descriptions = json.load(f)

        # ── Build sub-datasets ────────────────────────────────────────────────
        sub_datasets: List[LeRobotDataset] = []

        # Use config.root directly inside __init__ to avoid property lookup
        # before __init__ completes (the @property works fine, but being
        # explicit avoids any confusion).
        root = config.root

        if config.dataset_file:
            with open(config.dataset_file, "r") as f:
                dataset_configs = json.load(f)

            for ds_cfg in tqdm(dataset_configs, desc="Loading robot sub-datasets"):
                repo_id      = ds_cfg["repo_id"]
                ds_root      = os.path.join(root, ds_cfg["root"])
                episodes_raw = ds_cfg.get(config.split)
                episodes     = get_episodes_idx(episodes_raw) if episodes_raw is not None else None

                print(f"\033[94mloading episodes {episodes} from {ds_root} "
                      f"(split={config.split})\033[0m")

                sub_ds = LeRobotDataset(
                    repo_id=repo_id,
                    root=ds_root,
                    delta_timestamps=delta_timestamps,
                    video_backend=config.video_backend,
                    tolerance_s=config.tolerance_s,
                    episodes=episodes,
                )
                
                self._strip_unused_camera_features(sub_ds, used_cam_keys)
                sub_datasets.append(sub_ds)

        elif config.repo_id:
            sub_ds = LeRobotDataset(
                repo_id=config.repo_id,
                root=root,
                delta_timestamps=delta_timestamps,
                video_backend=config.video_backend,
                tolerance_s=config.tolerance_s,
            )
            self._strip_unused_camera_features(sub_ds, used_cam_keys)
            sub_datasets.append(sub_ds)

        else:
            raise ValueError(
                "Either dataset_file or repo_id must be set in LeRobotRobotDataConfig"
            )

        self.lerobot_dataset = ConcatDataset(sub_datasets)
        self.train_dataset   = sub_datasets[0]   # for stats / dimension fallback

        self._episode_start_cache = {}

        for ds_idx, ds in enumerate(self.lerobot_dataset.datasets):
            episode_indices = ds.hf_dataset["episode_index"]

            last_ep = None
            for local_idx, ep in enumerate(episode_indices):
                ep = int(ep)
                if ep != last_ep:
                    self._episode_start_cache[(ds_idx, ep)] = local_idx
                    last_ep = ep

        # ── ActionNormalizer ─────────────────────────────────────────────────
        self.normalizer = ActionNormalizer()
        for idx, sub_ds in enumerate(sub_datasets):
            self.normalizer.add_dataset_stats(
                dataset_idx=idx,
                repo_id=sub_ds.repo_id,
                stats=sub_ds.meta.stats,
                state_key=self.state_key,
                action_key=self.action_key,
                single_arm=config.single_arm,
            )

        self._repo_id_per_subdataset = [ds.repo_id for ds in sub_datasets]

        # ── Augmentation transforms ──────────────────────────────────────────
        self._setup_transforms(config)

        self.total_transitions = len(self.lerobot_dataset)
        self.target_length     = self.total_transitions
        self.stats             = self.train_dataset.meta.stats
        self.include_depth     = config.include_depth

        # ── Dimension discovery ──────────────────────────────────────────────
        try:
            feats      = self.train_dataset.features
            state_dim  = feats[self.state_key]["shape"][0]
            action_dim = feats[self.action_key]["shape"][0]
            if config.single_arm:
                state_dim  = state_dim  // 2
                action_dim = action_dim // 2
            self.state_dim  = state_dim
            self.action_dim = action_dim
        except (KeyError, AttributeError):
            self.state_dim  = config.state_dim
            self.action_dim = config.action_dim
            logger.warning(
                f"Could not auto-detect dims; using config defaults "
                f"state={self.state_dim}, action={self.action_dim}"
            )

        logger.info(
            f"LeRobotRobotDataset: {self.total_transitions} transitions, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}"
        )

    # ── Setup helpers ────────────────────────────────────────────────────────

    @property
    def root(self) -> str:
        return self.config.root

    @staticmethod
    def _strip_unused_camera_features(
        sub_ds: LeRobotDataset,
        used_cam_keys: List[str],
    ) -> None:
        """Remove image/video features not in used_cam_keys (faster loading)."""
        for k in list(sub_ds.meta.features.keys()):
            if sub_ds.meta.features[k]["dtype"] in ("image", "video", "depth"):
                if k not in used_cam_keys:
                    sub_ds.meta.features.pop(k)

        cols_to_remove = [
            c for c in sub_ds.hf_dataset.column_names
            if c.startswith("observation.images.") and c not in used_cam_keys
        ]
        if cols_to_remove:
            print(f"Stripping unused columns: {cols_to_remove}")
            sub_ds.hf_dataset = sub_ds.hf_dataset.remove_columns(cols_to_remove)

    def _setup_transforms(self, config: LeRobotRobotDataConfig) -> None:
        additional_targets = {"depth": "image"} if config.include_depth else {}

        if config.enable_augmentation:
            self.transform = A.Compose([
                A.OneOf([
                    A.Compose([
                        A.ShiftScaleRotate(
                            shift_limit=0.1, scale_limit=0.1,
                            rotate_limit=15, p=0.5,
                        ),
                        A.Resize(height=config.image_size[0],
                                 width=config.image_size[1], p=1.0),
                    ]),
                ], p=1.0),
            ], additional_targets=additional_targets)

            self.rgb_transform = A.Compose([
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5
                ),
                A.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.], max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=config.image_size[0],
                         width=config.image_size[1], p=1.0),
            ], additional_targets=additional_targets)

            self.rgb_transform = A.Compose([
                A.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.], max_pixel_value=255.0),
                ToTensorV2(),
            ])

        self.depth_transform = A.Compose([ToTensorV2()])

    # ── Core helpers ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.target_length

    def _extract_right_arm(self, x: torch.Tensor) -> torch.Tensor:
        """Extract right arm from [right_arm, left_arm, right_gripper, left_gripper]."""
        if not self.config.single_arm:
            return x
        M = x.shape[-1] // 2
        return torch.cat([x[..., :M - 1], x[..., 2 * M - 2: 2 * M - 1]], dim=-1)

    def _process_frame_image(
        self,
        frame: Dict[str, Any],
        cam_name: str,
    ) -> Optional[torch.Tensor]:
        """
        Process a single frame's image (and optional depth) for one camera.
        Returns (C, H, W) tensor or None if the camera key is absent.
        """
        rgb_key   = f"observation.images.{cam_name}"
        depth_key = f"observation.images.{cam_name}_depth"

        if rgb_key not in frame:
            return None

        rgb_tensor = frame[rgb_key]
        if rgb_tensor.dim() == 4:          # (T, C, H, W) → take first
            rgb_tensor = rgb_tensor[0]
        rgb_img = (rgb_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        depth_img = None
        if self.config.include_depth and depth_key in frame:
            d_tensor = frame[depth_key]
            if d_tensor.dim() == 4:
                d_tensor = d_tensor[0]
            depth_rgb = (d_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            depth_img = decode_depth(depth_rgb, cam_name, self.config.depth_mode)
            depth_img = depth_img[..., None]

        if depth_img is not None:
            res      = self.transform(image=rgb_img, depth=depth_img)
            aug_rgb  = res["image"]
            aug_dep  = res["depth"]
        else:
            res     = self.transform(image=rgb_img)
            aug_rgb = res["image"]
            aug_dep = None

        t_rgb = self.rgb_transform(image=aug_rgb)["image"]   # (C, H, W)

        if aug_dep is not None:
            t_dep = self.depth_transform(image=aug_dep)["image"]
            return torch.cat([t_rgb, t_dep], dim=0)
        return t_rgb

    # ── __getitem__ ──────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        global_idx = idx % self.total_transitions

        # Which sub-dataset?
        dataset_idx = bisect.bisect_right(
            self.lerobot_dataset.cumulative_sizes, global_idx
        )
        sample_idx = (
            global_idx
            if dataset_idx == 0
            else global_idx - self.lerobot_dataset.cumulative_sizes[dataset_idx - 1]
        )
        curr_dataset = self.lerobot_dataset.datasets[dataset_idx]

        # repo_id comes from sub-dataset metadata, NOT from frame['task']
        repo_id = curr_dataset.repo_id

        # ── Skill frames for skill / xskill training ─────────────────────
        skill_frames_raw = None

        if self.config.skill:
            robot_frame_gap = max(int(self.config.robot_frame_gap), 1)

            skill_first_idx = sample_idx
            skill_first_frame = curr_dataset[skill_first_idx]
            skill_first_ep = (
                skill_first_frame["episode_index"].item()
                if hasattr(skill_first_frame["episode_index"], "item")
                else skill_first_frame["episode_index"]
            )

            if self.config.xskill:
                # xskill: take 4 frames from [max(ep_start, sample_idx-30), sample_idx]
                ep_start_idx = self._episode_start_cache[(dataset_idx, int(skill_first_ep))]
                xskill_back_gap = 30
                start_idx = max(ep_start_idx, sample_idx - xskill_back_gap)

                indices = np.linspace(start_idx, sample_idx, 4).round().astype(int)
                unique_indices = np.unique(indices)

                if len(unique_indices) < 4:
                    first_idx = unique_indices[0]
                    indices = np.concatenate(
                        [unique_indices, [first_idx] * (4 - len(unique_indices))]
                    )
                else:
                    indices = unique_indices

                skill_frames_raw = [curr_dataset[int(i)] for i in indices]

            else:
                # skill: current frame + future frame, clipped inside same episode
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

                skill_frames_raw = [skill_first_frame, skill_last_frame]

        # ── obs_horizon frame stacking (same logic as LeRobotSimDataset) ────
        obs_horizon   = max(int(self.config.obs_horizon), 1)
        start_idx     = max(sample_idx - obs_horizon + 1, 0)
        frame_indices = list(range(start_idx, sample_idx + 1))

        # Pad at episode start (repeat first frame)
        if len(frame_indices) < obs_horizon:
            pad          = obs_horizon - len(frame_indices)
            frame_indices = [frame_indices[0]] * pad + frame_indices

        frames = [curr_dataset[i] for i in frame_indices]

        # Task description
        task_description = self.task_descriptions.get(repo_id, repo_id)

        # ── Image processing ──────────────────────────────────────────────
        all_view_tensors: List[torch.Tensor] = []
        skill_view_tensors: List[torch.Tensor] = []

        for cam_name in self.cameras:
            per_frame_views = []
            for frame in frames:
                view = self._process_frame_image(frame, cam_name)
                if view is None:
                    break
                per_frame_views.append(view)

            if per_frame_views:
                all_view_tensors.append(torch.stack(per_frame_views, dim=0))

            # extra skill frames processing
            if self.config.skill and skill_frames_raw is not None:
                per_skill_views = []
                for frame in skill_frames_raw:
                    view = self._process_frame_image(frame, cam_name)
                    if view is None:
                        break
                    per_skill_views.append(view)

                if per_skill_views:
                    skill_view_tensors.append(torch.stack(per_skill_views, dim=0))

        # ── State (obs_horizon, state_dim) ───────────────────────────────
        state_seq = []
        for frame in frames:
            s = frame[self.state_key]
            if not isinstance(s, torch.Tensor):
                s = torch.tensor(s)
            if s.dim() == 2:
                s = s[0]
            state_seq.append(self._extract_right_arm(s))
        state = torch.stack(state_seq, dim=0)   # (obs_horizon, state_dim)

        # ── Action sequence from the last / current frame ────────────────
        action_sequence = frames[-1][self.action_key]
        if not isinstance(action_sequence, torch.Tensor):
            action_sequence = torch.tensor(action_sequence)
        action_sequence = self._extract_right_arm(action_sequence)

        # ── Normalization ────────────────────────────────────────────────
        state = self.normalizer.normalize_state(
            state, dataset_idx, method=self.config.normalization_method
        )
        action_sequence = self.normalizer.normalize_action(
            action_sequence, dataset_idx, method=self.config.normalization_method
        )

        ret: Dict[str, Any] = {
            "states":            state.float(),
            "actions":           action_sequence.float(),
            "task_descriptions": str(task_description),
            "repo_id":           repo_id,
            "dataset_idx":       torch.tensor(dataset_idx, dtype=torch.long),
            "episode_index":     frames[-1].get("episode_index", torch.tensor(-1)),
            "frame_index":       frames[-1].get("index", torch.tensor(global_idx)),
        }

        for i, tensor in enumerate(all_view_tensors):
            ret[f"view_{i + 1}"] = tensor   # (obs_horizon, C, H, W)
        if self.config.skill:
            skill_frames = {}
            for view_idx, view_tensor in enumerate(skill_view_tensors, start=1):
                skill_frames[f"view_{view_idx}"] = view_tensor
            ret["skill_frames"] = skill_frames
        return ret

    # ── Denormalization helpers (public API) ─────────────────────────────────

    def denormalize_action(
        self,
        normalized_action: torch.Tensor,
        dataset_idx: int,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        if method is None:
            method = self.config.normalization_method
        return self.normalizer.denormalize_action(normalized_action, dataset_idx, method)

    def denormalize_state(
        self,
        normalized_state: torch.Tensor,
        dataset_idx: int,
        method: Optional[str] = None,
    ) -> torch.Tensor:
        if method is None:
            method = self.config.normalization_method
        return self.normalizer.denormalize_state(normalized_state, dataset_idx, method)