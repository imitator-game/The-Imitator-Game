import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
class LeRobotDynamicsConfig:
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
    horizon: int = 16
    obs_horizon: int = 2
    state_type: str = "qpos"
    single_arm: bool = False
    enable_augmentation: bool = True
    resolution: int = 256
    idm_resolution: int = 224
    pre_decode: bool = False
    pre_decode_cache_dir: str = "tmp/human_video_cache"
    pre_decode_num_workers: int = 4  # Parallel worker count for the decode step
    skill: bool = True
    robot_frame_gap: int = 35


class LeRobotDynamicsDataset(Dataset):
    """
    LeRobot-only dynamics dataset for UniSkill FSD/ISD training.
    Emits the same keys expected by train_uniskill.py.
    """

    def __init__(
        self,
        cfg: LeRobotDynamicsConfig,
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
            horizon=cfg.horizon,
            obs_horizon=max(cfg.obs_horizon, 2),
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
        
        self.resolution = cfg.resolution
        self.idm_resolution = cfg.idm_resolution

        self.paired_dataset = HumanSimPairedDataset(paired_cfg)
            
    def __len__(self):
        return len(self.paired_dataset) * 2

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

    def _format_pair(self, curr: torch.Tensor, nxt: torch.Tensor):
        curr_idm = self._resize(curr, self.idm_resolution)
        next_idm = self._resize(nxt, self.idm_resolution)

        curr_policy = self._resize(curr, self.resolution)
        next_policy = self._resize(nxt, self.resolution)
        if self.train:
            curr_policy = (curr_policy - 0.5) / 0.5
            next_policy = (next_policy - 0.5) / 0.5

        if self.depth_processor is not None:
            curr_depth = self.depth_processor(curr_idm, do_rescale=False)["pixel_values"][0]
            next_depth = self.depth_processor(next_idm, do_rescale=False)["pixel_values"][0]
        else:
            curr_depth = curr_idm
            next_depth = next_idm

        return curr_policy, next_policy, curr_idm, next_idm, curr_depth, next_depth
    
    def __getitem__(self, idx):
        sample = self.paired_dataset[idx // 2]
        use_robot = (idx % 2) == 0
        if use_robot:
            robot_video = sample["skill_frames"]["view_1"]  # [T, H, W, C]
            assert robot_video.shape[0]==2, "Robot Video is NOT 2 frames."
            curr_i = 0
            next_i = 1
            curr = self._to_chw_float(robot_video[curr_i])
            nxt = self._to_chw_float(robot_video[next_i])
            task_name = sample["sim_task_id"]
            data_type = "robot"
        else:
            human_video = sample["human_video"]  # [T, H, W, C]
            t = human_video.shape[0]
            curr_i = random.randint(0, max(0, t - 2))
            next_i = min(curr_i + 1, t - 1)
            curr = self._to_chw_float(human_video[curr_i])
            nxt = self._to_chw_float(human_video[next_i])
            task_name = sample["human_task_id"]
            data_type = "human"

        (
            curr_policy,
            next_policy,
            curr_idm,
            next_idm,
            curr_depth,
            next_depth,
        ) = self._format_pair(curr, nxt)
        return {    
            "curr_images": curr_policy,
            "next_images": next_policy,
            "idm_curr_images": curr_idm,
            "idm_next_images": next_idm,
            "curr_depth_features": curr_depth,
            "next_depth_features": next_depth,
            "data_type": data_type,
            "task_name": task_name,
        }
