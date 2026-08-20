import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
        HumanSimPairedDataset,
        PairedDatasetConfig,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[5]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
        HumanSimPairedDataset,
        PairedDatasetConfig,
    )


@dataclass
class LeRobotPolicyConfig:
    human_root: str = "demos"
    sim_root: str = "demos"
    split: str = "train"
    task_mapping_file: str = "examples/baselines/lerobot_dataset/task_mapping.json"
    human_dataset_file: str = "examples/baselines/lerobot_dataset/config/human_config.json"
    sim_dataset_file: str = "examples/baselines/lerobot_dataset/config/sim_config.json"
    human_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    sim_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"
    cameras: tuple[str, ...] = ("zed2i",)
    include_depth: bool = False
    image_size: tuple[int, int] = (224, 224)
    num_frames: int = 10
    sampling_strategy: str = "uniform_jitter"
    video_backend: str = "torchcodec"
    fps: int = 30
    state_type: str = "qpos"
    single_arm: bool = False
    enable_augmentation: bool = True
    resolution: int = 112
    idm_resolution: int = 224
    pred_horizon: int = 16
    obs_horizon: int = 2
    pre_decode: bool = False
    pre_decode_cache_dir: str = "tmp/human_video_cache"
    pre_decode_num_workers: int = 4
    human_idm_cache_dir: Optional[str] = None
    robot_idm_cache_dir: Optional[str] = None
    skill: bool = True
    robot_frame_gap: int = 35


class LeRobotPolicyDataset(Dataset):
    """
    LeRobot-only policy dataset for UniSkill conditional policy training.
    Emits the same keys expected by train_cond_dp.py.
    """

    def __init__(
        self,
        cfg: LeRobotPolicyConfig,
        train: bool,
        depth_processor=None,
    ):
        self.cfg = cfg
        self.train = train
        self.depth_processor = depth_processor
        split = "train" if train else "test"

        paired_cfg = PairedDatasetConfig(
            human_root=cfg.human_root,
            sim_root=cfg.sim_root,
            task_mapping_file=cfg.task_mapping_file,
            human_dataset_file=cfg.human_dataset_file,
            sim_dataset_file=cfg.sim_dataset_file,
            human_task_description_file=cfg.human_task_description_file,
            sim_task_description_file=cfg.sim_task_description_file,
            split=split,
            cameras=list(cfg.cameras),
            include_depth=cfg.include_depth,
            image_size=cfg.image_size,
            num_frames=max(cfg.num_frames, 2),
            sampling_strategy=cfg.sampling_strategy,
            video_backend=cfg.video_backend,
            horizon=cfg.pred_horizon,
            obs_horizon=cfg.obs_horizon,
            state_type=cfg.state_type,
            single_arm=cfg.single_arm,
            fps=cfg.fps,
            input_mode="video_and_language",
            include_first_frame=False,
            enable_augmentation=cfg.enable_augmentation and train,
            pre_decode=cfg.pre_decode,
            pre_decode_cache_dir=cfg.pre_decode_cache_dir,
            pre_decode_num_workers=cfg.pre_decode_num_workers,
            skill=cfg.skill,
            robot_frame_gap=cfg.robot_frame_gap,
        )
        self.paired_dataset = HumanSimPairedDataset(paired_cfg)
        self.resolution = cfg.resolution
        self.idm_resolution = cfg.idm_resolution
        self.pred_horizon = cfg.pred_horizon
        self.obs_horizon = cfg.obs_horizon
        self.human_idm_cache_index = {}
        self.human_idm_cache_complete = False
        self.refresh_human_idm_cache_index()
        if self.human_idm_cache_complete:
            self.paired_dataset.skip_human_video = True

    def refresh_human_idm_cache_index(self):
        self.human_idm_cache_index = {}
        self.human_idm_cache_complete = False
        if not self.cfg.human_idm_cache_dir:
            return
        cache_root = Path(self.cfg.human_idm_cache_dir)
        if not cache_root.exists():
            return
        for task_dir in cache_root.iterdir():
            if not task_dir.is_dir():
                continue
            files = sorted(task_dir.glob("*.pt"))
            if files:
                self.human_idm_cache_index[task_dir.name] = files
        required_human_tasks = {
            human_task_id for human_task_id, _ in self.paired_dataset.paired_tasks
        }
        cached_human_tasks = set(self.human_idm_cache_index)
        self.human_idm_cache_complete = (
            bool(required_human_tasks)
            and required_human_tasks.issubset(cached_human_tasks)
        )
        if self.human_idm_cache_complete:
            self.paired_dataset.skip_human_video = True

    def _robot_cache_path(self, idx: int) -> Optional[Path]:
        if not self.cfg.robot_idm_cache_dir:
            return None
        return Path(self.cfg.robot_idm_cache_dir) / f"{idx:09d}.pt"

    def __len__(self):
        return len(self.paired_dataset)

    @staticmethod
    def _to_chw_float(frame: torch.Tensor) -> torch.Tensor:
        x = frame
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.float()
        if x.ndim != 3:
            raise ValueError(f"Expected frame rank 3, got {x.shape}")
        if x.shape[-1] in (3, 4):
            x = x[..., :3].permute(2, 0, 1)
        elif x.shape[0] in (3, 4):
            x = x[:3]
        else:
            raise ValueError(f"Unrecognized image shape: {x.shape}")
        if x.max() > 1.0:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    def _resize(self, image: torch.Tensor, size: int) -> torch.Tensor:
        return F.interpolate(
            image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        ).squeeze(0)

    @staticmethod
    def _depth_to_tensor(depth_value) -> torch.Tensor:
        if torch.is_tensor(depth_value):
            return depth_value.float()
        return torch.as_tensor(depth_value, dtype=torch.float32)

    def __getitem__(self, idx):
        sample = self.paired_dataset[idx]
        dataset_idx = int(sample["dataset_idx"])
        robot_cache_path = self._robot_cache_path(idx)
        robot_idm_gt = None
        if robot_cache_path is not None and robot_cache_path.exists():
            robot_idm_gt = torch.load(robot_cache_path, map_location="cpu", weights_only=True)

        # Actions and states are already normalized by the sim dataset (bounds_q99).
        actions = sample["robot_actions"].cpu().numpy()
        states = sample["robot_obs"]["states"].cpu().numpy()
        states = states[-self.obs_horizon:]

        # Robot image features.
        robot_video = sample["skill_frames"]["view_1"]  # [T, H, W, C]
        assert robot_video.shape[0] == 2, f"Expected 2 frames, got {robot_video.shape[0]}"
        curr_frame = robot_video[0]
        next_frame = robot_video[1]

        curr_chw = self._to_chw_float(curr_frame)

        curr_images = self._resize(curr_chw, self.resolution)
        if self.train:
            curr_images = (curr_images - 0.5) / 0.5

        idm_curr = self._resize(curr_chw, self.idm_resolution)
        next_chw = self._to_chw_float(next_frame)
        idm_next = self._resize(next_chw, self.idm_resolution)
        if self.depth_processor is not None:
            curr_depth = self._depth_to_tensor(
                self.depth_processor(idm_curr, do_rescale=False)["pixel_values"][0]
            )
            next_depth = self._depth_to_tensor(
                self.depth_processor(idm_next, do_rescale=False)["pixel_values"][0]
            )
        else:
            curr_depth = idm_curr
            next_depth = idm_next

        # Human demo windows for IDM sequence conditioning.
        human_demo_latents = None
        cached_human_paths = self.human_idm_cache_index.get(sample["human_task_id"], [])
        if cached_human_paths:
            human_cache_path = random.choice(cached_human_paths)
            human_demo_latents = torch.load(human_cache_path, map_location="cpu", weights_only=True)
            human_demo_idm_t = None
            human_demo_depth_t = None
            human_video_id = str(human_cache_path)
        else:
            human_video = sample["human_video"]  # [T, H, W, C]
            human_frames = [self._to_chw_float(human_video[i]) for i in range(human_video.shape[0])]
            human_frames = [self._resize(x, self.idm_resolution) for x in human_frames]

            human_demo_idm = []
            human_demo_depth = []
            for i in range(max(0, len(human_frames) - 1)):
                f0 = human_frames[i]
                f1 = human_frames[i + 1]
                human_demo_idm.append(torch.stack([f0, f1], dim=0))
                if self.depth_processor is not None:
                    d0 = self._depth_to_tensor(
                        self.depth_processor(f0, do_rescale=False)["pixel_values"][0]
                    )
                    d1 = self._depth_to_tensor(
                        self.depth_processor(f1, do_rescale=False)["pixel_values"][0]
                    )
                    human_demo_depth.append(torch.stack([d0, d1], dim=0))
                else:
                    human_demo_depth.append(torch.stack([f0, f1], dim=0))

            if human_demo_idm:
                human_demo_idm_t = torch.stack(human_demo_idm, dim=0)
                human_demo_depth_t = torch.stack(human_demo_depth, dim=0)
                human_video_id = int(dataset_idx)
            else:
                human_demo_idm_t = torch.zeros((0, 2, 3, self.idm_resolution, self.idm_resolution))
                human_demo_depth_t = torch.zeros((0, 2, 3, self.idm_resolution, self.idm_resolution))
                human_video_id = -1

        return {
            "curr_images": curr_images,  # [3, H, W]
            "actions": actions.astype(np.float32),  # [pred_horizon, action_dim]
            "state": states.astype(np.float32),  # [obs_horizon, state_dim]
            "idm_curr_images": idm_curr,
            "idm_next_images": idm_next,
            "curr_depth_features": curr_depth,
            "next_depth_features": next_depth,
            "human_demo_idm_images": human_demo_idm_t,
            "human_demo_depth_features": human_demo_depth_t,
            "human_demo_idm_latents": human_demo_latents,
            "robot_idm_gt": robot_idm_gt,
            "robot_idm_cache_path": str(robot_cache_path) if robot_cache_path is not None else "",
            "task_name": sample["sim_task_id"],
            "data_type": "robot",
            "human_video_id": human_video_id,
            "next_images": curr_images,  # placeholder for compatibility
        }
