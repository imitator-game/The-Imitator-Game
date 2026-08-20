"""
LeRobot-backed video datasets for XSkill Stage1/Stage2.

This replaces the old HDF5 kitchen loader while preserving the original
xskill dataset interface:
- sample type: IndexBatch(im_q, index, info)
- im_q shape: (T, C, H, W), float32 in [0, 1]
- info keys: task_idx, task_name, vid_idx
"""

from __future__ import annotations

import collections
import random
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanSimPairedDataset,
    PairedDatasetConfig,
)

IndexBatch = namedtuple("IndexBatch", "im_q index info")


@dataclass
class _LeRobotVideoConfig:
    # Paths
    human_root: str
    sim_root: str
    task_mapping_file: str

    # Dataset configs
    human_dataset_file: str
    sim_dataset_file: str
    human_task_description_file: str
    sim_task_description_file: str

    # Common settings
    split: str
    cameras: List[str]
    include_depth: bool
    image_size: Tuple[int, int]
    num_frames: int
    video_backend: str
    fps: int
    state_type: str
    single_arm: bool
    sampling_strategy: str
    enable_augmentation: bool


class XskillVideoDataset(torch.utils.data.Dataset):
    """
    `video_type='human'` returns human clips from paired LeRobot data.
    `video_type='robot'` returns sim clips from paired LeRobot data.
    """

    def __init__(
        self,
        # New LeRobot params (required)
        human_root: str,
        sim_root: str,
        task_mapping_file: str,
        human_dataset_file: str,
        sim_dataset_file: str,
        human_task_description_file: str,
        sim_task_description_file: str,
        split: str = "train",
        cameras: Optional[List[str]] = None,
        include_depth: bool = False,
        image_size: List[int] | Tuple[int, int] = (224, 224),
        num_frames: int = 10,
        video_backend: str = "torchcodec",
        fps: int = 30,
        state_type: str = "qpos",
        single_arm: bool = False,
        sampling_strategy: str = "uniform_jitter",
        enable_augmentation: bool = True,
        # Kept for compatibility with old configs/callers
        frame_sampler=None,
        task_names: Optional[List[str]] = None,
        max_videos_per_task: Optional[int] = None,
        seed: Optional[int] = None,
        max_get_threads: int = 4,
        resize_shape: Optional[List[int]] = None,
        video_type: str = "human",
        slide: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        if video_type not in {"human", "robot"}:
            raise ValueError(f"video_type must be 'human' or 'robot', got {video_type}")

        self._frame_sampler = frame_sampler
        self.max_get_threads = max_get_threads
        self.resize_shape = resize_shape
        self.video_type = video_type
        self.max_videos_per_task = max_videos_per_task
        self.slide = slide
        self.kwargs = kwargs
        self.task_names_filter = set(task_names) if task_names is not None else None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if cameras is None:
            cameras = ["zed2i"]

        cfg = _LeRobotVideoConfig(
            human_root=human_root,
            sim_root=sim_root,
            task_mapping_file=task_mapping_file,
            human_dataset_file=human_dataset_file,
            sim_dataset_file=sim_dataset_file,
            human_task_description_file=human_task_description_file,
            sim_task_description_file=sim_task_description_file,
            split=split,
            cameras=list(cameras),
            include_depth=include_depth,
            image_size=tuple(image_size),
            num_frames=int(num_frames),
            video_backend=video_backend,
            fps=int(fps),
            state_type=state_type,
            single_arm=single_arm,
            sampling_strategy=sampling_strategy,
            enable_augmentation=enable_augmentation,
        )

        # Single paired backend for both video types.
        paired_cfg = PairedDatasetConfig(
            human_root=cfg.human_root,
            sim_root=cfg.sim_root,
            task_mapping_file=cfg.task_mapping_file,
            human_dataset_file=cfg.human_dataset_file,
            sim_dataset_file=cfg.sim_dataset_file,
            human_task_description_file=cfg.human_task_description_file,
            sim_task_description_file=cfg.sim_task_description_file,
            split=cfg.split,
            cameras=cfg.cameras,
            include_depth=cfg.include_depth,
            image_size=cfg.image_size,
            num_frames=cfg.num_frames,
            sampling_strategy=cfg.sampling_strategy,
            video_backend=cfg.video_backend,
            horizon=16,
            # Robot clip length for stage1/2: follow vqbet num_frames.
            obs_horizon=cfg.num_frames,
            state_type=cfg.state_type,
            single_arm=cfg.single_arm,
            fps=cfg.fps,
            input_mode="video_only",
            include_first_frame=False,
            enable_augmentation=cfg.enable_augmentation,
        )
        self.paired_dataset = HumanSimPairedDataset(paired_cfg)

        self._indexfile: Dict[int, Tuple[int, str, int]] = {}
        self._sample_positions: List[int] = []
        self._build_video_index()

    def _build_video_index(self):
        """Build task-indexed sample mapping from paired dataset indices."""
        valid_positions = list(range(len(self.paired_dataset.valid_indices)))
        actual_to_pos = {actual: pos for pos, actual in enumerate(self.paired_dataset.valid_indices)}

        task_to_positions: Dict[str, List[int]] = {}

        for sim_task_id, actual_indices in self.paired_dataset.task_to_sim_indices.items():
            human_task_id = self.paired_dataset.task_mapper.get_human_task_from_sim(sim_task_id)
            if human_task_id is None:
                continue
            if self.task_names_filter is not None and human_task_id not in self.task_names_filter:
                continue
            if human_task_id not in task_to_positions:
                task_to_positions[human_task_id] = []
            for actual_idx in actual_indices:
                if actual_idx in actual_to_pos:
                    task_to_positions[human_task_id].append(actual_to_pos[actual_idx])

        if self.task_names_filter is not None:
            task_to_positions = {
                k: v for k, v in task_to_positions.items() if k in self.task_names_filter
            }

        if not task_to_positions:
            raise ValueError("No paired tasks available after filtering")

        self.trajectory_order = []
        self._video_index = collections.OrderedDict()

        global_idx = 0
        for task_idx, task_name in enumerate(sorted(task_to_positions.keys())):
            positions = task_to_positions[task_name]
            if self.max_videos_per_task is not None:
                positions = positions[: self.max_videos_per_task]
            if len(positions) == 0:
                continue

            self._video_index[task_name] = {
                "task_idx": task_idx,
                "num_videos": len(positions),
            }

            for vid_idx, pos in enumerate(positions):
                self._indexfile[global_idx] = (task_idx, task_name, vid_idx)
                self._sample_positions.append(pos)
                self.trajectory_order.append((task_name, vid_idx, str(pos)))
                global_idx += 1

        if len(self._indexfile) == 0:
            raise ValueError("No videos indexed from LeRobot paired dataset")

        print(f"LeRobot {self.video_type} videos indexed: {len(self._indexfile)}")
        print(f"LeRobot matched tasks: {sorted(self._video_index.keys())}")

    def _extract_clip_from_paired_sample(self, sample: Dict[str, object]) -> np.ndarray:
        """Return clip as (T,H,W,C) numpy array."""
        if self.video_type == "human":
            clip = sample["human_video"]
        else:
            robot_obs = sample["robot_obs"]
            view_keys = [k for k in robot_obs.keys() if k != "states"]
            if not view_keys:
                raise ValueError("No robot views found in paired sample")
            # Follow vqbet convention: first configured view.
            clip = robot_obs[sorted(view_keys)[0]]

        if isinstance(clip, torch.Tensor):
            clip_np = clip.detach().cpu().numpy()
        else:
            clip_np = np.asarray(clip)

        # Expect clip in (T,H,W,C). If channel-first, convert.
        if clip_np.ndim != 4:
            raise ValueError(f"Expected 4D clip tensor, got shape {clip_np.shape}")
        if clip_np.shape[-1] not in (1, 3, 4) and clip_np.shape[1] in (1, 3, 4):
            clip_np = np.transpose(clip_np, (0, 2, 3, 1))

        # Ensure uint8-like range before normalization.
        if clip_np.dtype != np.uint8:
            if clip_np.max() <= 1.0:
                clip_np = (clip_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                clip_np = clip_np.clip(0, 255).astype(np.uint8)

        return clip_np

    def _subsample_or_full(self, clip: np.ndarray) -> np.ndarray:
        """Optionally subsample frames with legacy frame_sampler API."""
        if self._frame_sampler is None:
            return clip

        sampled = self._frame_sampler.sample(clip)
        frame_indices = np.array(sampled["ctx_idxs"]).flatten()
        return clip[frame_indices]

    def _resize_clip(self, clip: np.ndarray) -> np.ndarray:
        if self.resize_shape is None:
            return clip
        resized = [cv2.resize(frame, tuple(self.resize_shape)) for frame in clip]
        return np.stack(resized, axis=0)

    def transform(self, sequence_data: np.ndarray) -> np.ndarray:
        """(T,H,W,C)->(T,C,H,W), float32 in [0,1]."""
        sequence_data = np.transpose(sequence_data, (0, 3, 1, 2)).astype(np.float32)
        sequence_data = sequence_data / 255.0
        return sequence_data

    def __len__(self):
        return len(self._indexfile)

    def __getitem__(self, idx):
        task_idx, task_name, vid_idx = self._indexfile[idx]
        paired_pos = self._sample_positions[idx]

        assert paired_pos < len(self.paired_dataset), f"paired_pos: {paired_pos} is out of range, len(paired_dataset): {len(self.paired_dataset)}"
        paired_sample = self.paired_dataset[paired_pos]
        clip = self._extract_clip_from_paired_sample(paired_sample)
        clip = self._subsample_or_full(clip)
        clip = self._resize_clip(clip)

        im_q = self.transform(clip)
        info = {
            "task_idx": task_idx,
            "task_name": task_name,
            "vid_idx": vid_idx,
        }
        return IndexBatch(im_q, idx, info)

    def print_trajectory_order(self):
        if self.video_type == "robot" and self.trajectory_order:
            print("Robot dataset trajectory order:")
            for i, (task_name, vid_idx, pseudo_key) in enumerate(self.trajectory_order):
                print(f"  Index {i}: sample_{pseudo_key} -> task: {task_name}, vid_idx: {vid_idx}")
        else:
            print(f"Trajectory order not available for {self.video_type} videos")

    @property
    def task_names(self):
        return list(self._video_index.keys())


class XskillConcatDataset(torch.utils.data.Dataset):
    """
    Concatenates robot and human datasets with random same-task pairing.
    """

    def __init__(self, *datasets, target_dataset_size=int(1e3)):
        self.datasets = datasets
        self.target_dataset_size = target_dataset_size
        self._build_task_index()

    def _build_task_index(self):
        if len(self.datasets) != 2:
            raise ValueError("XskillConcatDataset expects exactly 2 datasets (robot and human)")

        robot_dataset, human_dataset = self.datasets
        if robot_dataset.video_type != "robot" or human_dataset.video_type != "human":
            raise ValueError("First dataset must be robot, second must be human")

        self.robot_task_indices = {}
        self.human_task_indices = {}

        for global_idx, (_task_idx, task_name, _vid_idx) in robot_dataset._indexfile.items():
            self.robot_task_indices.setdefault(task_name, []).append(global_idx)

        for global_idx, (_task_idx, task_name, _vid_idx) in human_dataset._indexfile.items():
            self.human_task_indices.setdefault(task_name, []).append(global_idx)

        self.common_tasks = set(self.human_task_indices.keys()).intersection(self.robot_task_indices.keys())
        if not self.common_tasks:
            raise ValueError("No common tasks found between human and robot datasets")

        self.robot_indices = []
        for task_name in sorted(self.common_tasks):
            self.robot_indices.extend(self.robot_task_indices[task_name])

        print(f"Found {len(self.common_tasks)} common tasks: {sorted(self.common_tasks)}")
        print(f"Total robot videos available for sampling: {len(self.robot_indices)}")

    def __getitem__(self, i):
        robot_idx = self.robot_indices[i]
        robot_sample = self.datasets[0][robot_idx]
        robot_task = robot_sample.info["task_name"]

        human_idx = np.random.choice(self.human_task_indices[robot_task])
        human_sample = self.datasets[1][human_idx]

        return (robot_sample, human_sample)

    def __len__(self):
        return len(self.robot_indices)

    @property
    def matched_tasks(self):
        return sorted(self.common_tasks)


def create_xskill_dataset(
    human_video_path: str,
    robot_video_path: str,
    frame_sampler,
    task_names: Optional[List[str]] = None,
    max_videos_per_task: int = 50,
    seed: Optional[int] = None,
    max_get_threads: int = 4,
    resize_shape: List[int] = [135, 135],
) -> XskillConcatDataset:
    """
    Legacy convenience function intentionally kept for compatibility.

    Note: this function is no longer path-based for LeRobot. Use direct class
    instantiation from Hydra configs for stage1/stage2.
    """
    raise RuntimeError(
        "create_xskill_dataset is deprecated in LeRobot mode. "
        "Instantiate XskillVideoDataset via Hydra config instead."
    )
