"""
Human video dataloader for video prompting.
Loads human demonstration videos from LeRobot format with random temporal subsampling.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gc

from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_episodes, load_info
from examples.baselines.lerobot_dataset.utils import (
    create_output_dir,
    print_header,
    print_dataset_info,
    print_summary,
)
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_episodes_idx(sub_episodes: str) -> List[int]:
    """
    Get episode indices from string like "0:45"
    """
    if ':' not in sub_episodes:
        return eval(sub_episodes)

    start, end = map(int, sub_episodes.split(":"))
    return list(range(start, end))


@dataclass
class HumanVideoDataConfig:
    """Configuration for human video dataset loading."""

    root: str = "demos"  # Root directory containing human demo repos
    split: str = "train"  # Split to use (train/test)

    # Dataset configuration files
    dataset_file: Optional[str] = None  # JSON file listing datasets to load
    task_description_file: Optional[str] = None  # JSON file mapping repo_id to task description

    # Camera configuration
    cameras: List[str] = field(default_factory=lambda: ["cam1", "cam2"])
    include_depth: bool = True

    # Video sampling parameters
    num_frames: int = 10  # Number of frames to sample from each episode
    image_size: Tuple[int, int] = (224, 224)

    # Sampling strategy
    sampling_strategy: str = "uniform_jitter"  # "uniform_jitter" or "random_interval"
    max_jitter: int = 3  # Maximum jitter for uniform sampling (in frames)

    # Video backend
    video_backend: str = "torchcodec"  # "torchcodec", "decord", or "opencv"

    # FPS
    fps: int = 30

    # Debug
    debug: bool = False

    # if vla == True, load language description instead of video frames
    vla: bool = False

    enable_augmentation: bool = True

    # Pre-decode: decode all video files in parallel into a local cache folder.
    # Once enabled, all subsequent frame reads are served from the cache (fast numpy
    # mmap loads) instead of going through the video decoder.
    pre_decode: bool = False
    pre_decode_cache_dir: str = "tmp/human_video_cache"
    pre_decode_num_workers: int = 16  # Parallel worker count for the decode step


class VideoFrameReader:
    """Efficient video frame reader with multiple backend support."""

    def __init__(self, backend: str = "torchcodec"):
        self.backend = backend
        self._init_backend()

    def _init_backend(self):
        """Initialize the appropriate backend."""
        if self.backend == "torchcodec":
            try:
                from torchcodec.decoders import VideoDecoder
                self.VideoDecoder = VideoDecoder
            except ImportError:
                print("⚠️  torchcodec not available, falling back to decord")
                self.backend = "decord"
                self._init_backend()

        if self.backend == "decord":
            try:
                from decord import VideoReader, cpu
                self.VideoReader = VideoReader
                self.decord_cpu = cpu
            except ImportError:
                print("⚠️  decord not available, falling back to opencv")
                self.backend = "opencv"

    def read_frames(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """
        Read specific frames from video.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to read

        Returns:
            np.ndarray: [N, H, W, C] RGB frames in uint8 format
        """
        if self.backend == "torchcodec":
            return self._read_torchcodec(video_path, frame_indices)
        elif self.backend == "decord":
            return self._read_decord(video_path, frame_indices)
        else:
            return self._read_opencv(video_path, frame_indices)

    def read_all_frames(self, video_path: str) -> np.ndarray:
        """
        Decode every frame in a video file.

        Args:
            video_path: Path to video file

        Returns:
            np.ndarray: [N, H, W, C] RGB frames in uint8 format
        """
        if self.backend == "torchcodec":
            return self._read_all_torchcodec(video_path)
        elif self.backend == "decord":
            return self._read_all_decord(video_path)
        else:
            return self._read_all_opencv(video_path)

    def _read_all_torchcodec(self, video_path: str) -> np.ndarray:
        decoder = self.VideoDecoder(video_path)
        num_frames = decoder.metadata.num_frames
        return self._read_torchcodec(video_path, list(range(num_frames)))

    def _read_all_decord(self, video_path: str) -> np.ndarray:
        vr = self.VideoReader(video_path, ctx=self.decord_cpu(0))
        indices = list(range(len(vr)))
        return vr.get_batch(indices).asnumpy()

    def _read_all_opencv(self, video_path: str) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return self._read_opencv(video_path, list(range(num_frames)))

    def _read_torchcodec(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Read using torchcodec (fastest, best for LeRobot)."""
        decoder = self.VideoDecoder(video_path)
        max_frame = decoder.metadata.num_frames - 1

        # Clip indices to valid range
        valid_indices = [min(idx, max_frame) for idx in frame_indices]

        try:
            # Batch read frames
            frame_batch = decoder.get_frames_at(indices=valid_indices)
            frames_tensor = frame_batch.data  # [N, C, H, W]

            # Convert to numpy: [N, C, H, W] -> [N, H, W, C]
            frames_np = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()

            # Ensure uint8
            if frames_np.dtype != np.uint8:
                frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

            return frames_np

        except Exception as e:
            print(f"⚠️  Torchcodec failed for {video_path}: {e}")
            # Fallback to frame-by-frame
            frames = []
            for idx in valid_indices:
                try:
                    frame_tensor = decoder[idx]  # [C, H, W]
                    frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
                    if frame_np.dtype != np.uint8:
                        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
                    frames.append(frame_np)
                except:
                    # Use black frame on error
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

            return np.array(frames)

    def _read_decord(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Read using decord (very fast)."""
        vr = self.VideoReader(video_path, ctx=self.decord_cpu(0))
        max_frame = len(vr) - 1

        # Clip indices
        valid_indices = [min(idx, max_frame) for idx in frame_indices]

        # Batch read
        frames = vr.get_batch(valid_indices).asnumpy()  # [N, H, W, C] RGB uint8


        return frames

    def _read_opencv(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Read using OpenCV (slowest but most compatible)."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        valid_indices = [min(idx, max_frame) for idx in frame_indices]

        frames = []
        sorted_indices = sorted(enumerate(valid_indices), key=lambda x: x[1])
        result_frames = [None] * len(valid_indices)

        current_pos = 0
        for original_idx, frame_idx in sorted_indices:
            if frame_idx != current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_pos = frame_idx

            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_frames[original_idx] = frame_rgb
                current_pos += 1
            else:
                # Use last valid frame or black frame
                if original_idx > 0 and result_frames[original_idx - 1] is not None:
                    result_frames[original_idx] = result_frames[original_idx - 1]
                else:
                    result_frames[original_idx] = np.zeros((480, 640, 3), dtype=np.uint8)

        cap.release()

        return np.array(result_frames)


def rgb2hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB to HSV conversion."""
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

    m = ok & (rgb_amax == 0); h[m] = 0 + (rgb[m, 1] - rgb[m, 2]) / r[m]
    m = ok & (rgb_amax == 1); h[m] = 2 + (rgb[m, 2] - rgb[m, 0]) / r[m]
    m = ok & (rgb_amax == 2); h[m] = 4 + (rgb[m, 0] - rgb[m, 1]) / r[m]

    h[:] *= 60
    h[h < 0] += 360
    s[ok] = r[ok] / rgb_max[ok]
    v[:] = rgb_max

    return np.stack((h, s, v), -1)


def decode_depth(rgb: np.ndarray, cam_name: str) -> np.ndarray:
    """Decompress RGB-encoded depth image."""
    if np.issubdtype(rgb.dtype, np.unsignedinteger):
        rgb = rgb.astype(np.float32) / 255.0

    hsv = rgb2hsv(rgb)

    max_hue = 300.0
    min_s = 0.1
    min_v = 0.1
    err_depth = 0.0

    # Determine depth range based on camera name
    if "zed" in cam_name.lower():
        zrange = (0.0, 0.5)
    else:
        zrange = (0.0, 4.0)

    ok = (hsv[..., 1] >= min_s) & (hsv[..., 2] >= min_v) & (hsv[..., 0] <= max_hue)

    depth = np.full(rgb.shape[:-1], err_depth, dtype=np.float32)
    depth[ok] = hsv[ok, 0] / max_hue
    depth = depth * (zrange[1] - zrange[0]) + zrange[0]

    return depth


class HumanVideoDataset(Dataset):
    """
    Dataset for loading human demonstration videos with random temporal subsampling.

    Each __getitem__ returns a randomly sampled video clip from a random episode
    of the specified task.

    Supports two modes:
    1. With dataset_file: Loads specific repos and episodes from config
    2. Without dataset_file: Recursively discovers all repos in root

    When config.pre_decode is True the dataset will, during __init__, decode every
    video file in the collection once (in parallel) and persist the raw frames as
    memory-mapped .npy files under config.pre_decode_cache_dir.  All subsequent
    frame reads are then served from those numpy files, completely bypassing the
    video decoder.  The public interface is identical regardless of this flag.
    """

    def __init__(self, config: HumanVideoDataConfig):
        self.config = config

        if LeRobotDataset is None or load_episodes is None or load_info is None:
            raise ImportError("lerobot is not installed. Cannot use HumanVideoDataset.")

        # Load task descriptions if provided
        self.task_descriptions = {}
        if config.task_description_file and os.path.exists(config.task_description_file):
            with open(config.task_description_file, 'r') as f:
                self.task_descriptions = json.load(f)
                print(f"✓ Loaded {len(self.task_descriptions)} task descriptions")

        # Determine which camera keys to keep
        self.used_cam_keys = []
        for cam in config.cameras:
            self.used_cam_keys.append(f"observation.images.{cam}")
            if config.include_depth:
                self.used_cam_keys.append(f"observation.images.{cam}_depth")

        # Load datasets based on configuration
        if config.dataset_file and os.path.exists(config.dataset_file):
            print(f"�� Loading datasets from config: {config.dataset_file}")
            self._load_from_config()
        else:
            print(f"�� Scanning for human demo repos in: {config.root}")
            self._discover_repos()

        if not self.all_tasks:
            raise ValueError("No valid tasks found. Check your dataset configuration.")

        print(f"✓ Loaded {len(self.all_tasks)} unique tasks")

        # Setup video reader
        self.video_reader = VideoFrameReader(backend=config.video_backend)


        # Setup image transforms
        self._setup_transforms_human(config)

        # Pre-decode all videos into the cache folder (parallel) if requested.
        # self._pre_decode_cache maps absolute video path -> cache .npy file path.
        if config.pre_decode:
            self._pre_decode_all_episodes()

        self.vla = config.vla

        print(f"✓ HumanVideoDataset initialized")
        print(f"  - Tasks: {len(self.all_tasks)}")
        print(f"  - Cameras: {config.cameras}")
        print(f"  - Frames per sample: {config.num_frames}")
        print(f"  - Sampling strategy: {config.sampling_strategy}")
        if config.pre_decode:
            print(f"  - Pre-decode cache: {config.pre_decode_cache_dir}")

    # ------------------------------------------------------------------
    # Pre-decode helpers
    # ------------------------------------------------------------------

    
    def _episode_cache_path(
        self,
        task_name: str,
        episode_idx: int,
        cam_name: str,
        is_depth: bool = False,
    ) -> Path:
        """
        Stable .pt path for one decoded (task, episode, camera) tuple.
        Key encodes everything that affects the stored tensor so it's
        safe to re-use across runs.
        """
        H, W = self.config.image_size
        suffix = "|depth" if is_depth else ""
        raw = (
            f"{task_name}|{episode_idx}|{cam_name}{suffix}"
            f"|{self.config.num_frames}|{H}|{W}|{self.config.split}"
        )
        key = hashlib.md5(raw.encode()).hexdigest()
        return Path(self.config.pre_decode_cache_dir) / f"{key}.pt"


    def _sample_frames_deterministic(
        self, start_frame: int, end_frame: int, num_frames: int
    ) -> List[int]:
        """
        Uniform sampling with NO jitter.
        Used during pre-decode and on cache hits so video and any
        downstream state loading stay in sync.
        """
        total = end_frame - start_frame + 1
        if total <= num_frames:
            return list(range(start_frame, end_frame + 1))
        return list(np.linspace(start_frame, end_frame, num_frames).astype(int))


    def _pre_decode_episode(
        self,
        task_name: str,
        episode_idx: int,
        cam_name: str,
    ) -> bool:
        """
        Decode `num_frames` frames for one (task, episode, camera) tuple,
        apply spatial + RGB/depth transforms, and persist as .pt files.

        RGB  → _episode_cache_path(..., is_depth=False)   [T, H, W, C]
        Depth→ _episode_cache_path(..., is_depth=True)    [T, H, W, 1]  (if include_depth)

        Returns True on success, False on any error.
        Idempotent: already-cached files are skipped.
        """
        rgb_cache   = self._episode_cache_path(task_name, episode_idx, cam_name, is_depth=False)
        depth_cache = self._episode_cache_path(task_name, episode_idx, cam_name, is_depth=True)

        need_rgb   = not rgb_cache.exists()
        need_depth = self.config.include_depth and not depth_cache.exists()

        if not need_rgb and not need_depth:
            return True  # already fully cached

        try:
            # ── Resolve repo metadata ──────────────────────────────────────
            repo_info_entry = self.task_to_repos[task_name][0]  # one repo per task
            repo_id   = repo_info_entry['repo_id']
            repo_path = Path(repo_info_entry['repo_path'])

            info               = self.repo_info[repo_id]
            episodes_dataset   = self.repo_episodes[repo_id]
            video_path_format  = info.get('video_path')
            fps                = info['fps']

            if video_path_format is None:
                return False

            episode_row    = episodes_dataset[episode_idx]
            episode_length = int(episode_row['length'])

            # ── Deterministic frame indices ────────────────────────────────
            frame_indices = self._sample_frames_deterministic(
                0, episode_length - 1, self.config.num_frames
            )

            # ── Resolve RGB video path ─────────────────────────────────────
            video_key      = f"observation.images.{cam_name}"
            chunk_idx_key  = f"videos/{video_key}/chunk_index"
            file_idx_key   = f"videos/{video_key}/file_index"

            if chunk_idx_key not in episode_row or file_idx_key not in episode_row:
                return False

            chunk_idx = int(episode_row[chunk_idx_key])
            file_idx  = int(episode_row[file_idx_key])

            rgb_video_path = repo_path / video_path_format.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            if not rgb_video_path.exists():
                return False

            from_ts_key = f"videos/{video_key}/from_timestamp"
            if from_ts_key in episode_row:
                video_start = round(float(episode_row[from_ts_key]) * fps)
                abs_indices = [video_start + idx for idx in frame_indices]
            else:
                abs_indices = list(frame_indices)

            # Each worker gets its own reader (not thread-safe to share)
            reader = VideoFrameReader(backend=self.config.video_backend)

            # Always read RGB (also needed as spatial-transform anchor for depth)
            rgb_frames = reader.read_frames(str(rgb_video_path), abs_indices)

            # ── Cache RGB ──────────────────────────────────────────────────
            if need_rgb:
                processed_rgb = []
                for frame in rgb_frames:
                    t = self.spatial_transform(image=frame)['image']
                    t = self.rgb_transform(image=t)['image']       # [C, H, W]
                    processed_rgb.append(t)

                # [T, C, H, W] → [T, H, W, C]  (matches _get_target_item convention)
                video_tensor = torch.stack(processed_rgb, dim=0).permute(0, 2, 3, 1)

                tmp = rgb_cache.with_suffix('.tmp.pt')
                torch.save(video_tensor, tmp)
                tmp.rename(rgb_cache)

            # ── Cache depth ────────────────────────────────────────────────
            if need_depth:
                depth_video_key  = f"observation.images.{cam_name}_depth"
                depth_chunk_key  = f"videos/{depth_video_key}/chunk_index"
                depth_file_key   = f"videos/{depth_video_key}/file_index"

                if depth_chunk_key in episode_row and depth_file_key in episode_row:
                    depth_chunk_idx = int(episode_row[depth_chunk_key])
                    depth_file_idx  = int(episode_row[depth_file_key])

                    depth_video_path = repo_path / video_path_format.format(
                        video_key=depth_video_key,
                        chunk_index=depth_chunk_idx,
                        file_index=depth_file_idx,
                    )

                    if depth_video_path.exists():
                        depth_from_ts_key = f"videos/{depth_video_key}/from_timestamp"
                        if depth_from_ts_key in episode_row:
                            depth_start = round(float(episode_row[depth_from_ts_key]) * fps)
                            depth_abs   = [depth_start + idx for idx in frame_indices]
                        else:
                            depth_abs = abs_indices

                        depth_rgb_frames = reader.read_frames(str(depth_video_path), depth_abs)

                        processed_depth = []
                        for i, depth_rgb in enumerate(depth_rgb_frames):
                            depth_map   = decode_depth(depth_rgb, cam_name)
                            depth_frame = depth_map[..., None]                # [H, W, 1]
                            # spatial_transform needs an anchor RGB image
                            transformed = self.spatial_transform(
                                image=rgb_frames[i], depth=depth_frame
                            )
                            depth_t     = transformed['depth']
                            depth_final = self.depth_transform(image=depth_t)['image']  # [1, H, W]
                            processed_depth.append(depth_final)

                        # [T, 1, H, W] → [T, H, W, 1]
                        depth_tensor = torch.stack(processed_depth, dim=0).permute(0, 2, 3, 1)

                        tmp = depth_cache.with_suffix('.tmp.pt')
                        torch.save(depth_tensor, tmp)
                        tmp.rename(depth_cache)

            return True

        except Exception as exc:
            print(f"⚠️  Pre-decode failed for {task_name} ep{episode_idx} cam={cam_name}: {exc}")
            return False


    def _pre_decode_all_episodes(self) -> None:
        """
        Iterate every unique (task, episode, camera) combination and
        pre-decode + cache the required frames.

        Idempotent — safe to re-run after a crash or partial run.
        """
        cache_dir = Path(self.config.pre_decode_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Collect unique decode jobs
        seen: set = set()
        jobs: List[Tuple[str, int, str]] = []

        for task_name, repos in self.task_to_repos.items():
            num_episodes = repos[0]['num_episodes']
            for episode_idx in range(num_episodes):
                for cam_name in self.config.cameras:
                    key = (task_name, episode_idx, cam_name)
                    if key not in seen:
                        seen.add(key)
                        jobs.append(key)

        print(
            f"🔄 Pre-decoding {len(jobs)} unique (task, episode, camera) jobs "
            f"with {self.config.pre_decode_num_workers} workers …"
        )

        success = 0
        with ThreadPoolExecutor(max_workers=self.config.pre_decode_num_workers) as executor:
            futures = {
                executor.submit(self._pre_decode_episode, t, e, c): (t, e, c)
                for t, e, c in jobs
            }
            for future in tqdm(as_completed(futures), total=len(jobs), desc="Pre-decoding"):
                ok = future.result()
                if ok:
                    success += 1

        print(f"✓ Pre-decode complete: {success}/{len(jobs)} episodes cached")

    # ------------------------------------------------------------------
    # Existing methods (unchanged)
    # ------------------------------------------------------------------

    def _setup_transforms_human(self, config):
        """Call this inside HumanVideoDataset.__init__ instead of the old transform setup."""
        additional_targets = {}
        if config.include_depth:
            additional_targets['depth'] = 'image'

        self.spatial_transform = A.Compose([
            A.Resize(height=config.image_size[0], width=config.image_size[1], p=1.0),
        ], additional_targets=additional_targets)

        if config.enable_augmentation:
            self.rgb_transform = A.Compose([
                A.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5
                ),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                ToTensorV2(),
            ])
        else:
            self.rgb_transform = A.Compose([
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                ToTensorV2(),
            ])

        self.depth_transform = A.Compose([
            ToTensorV2(),
        ])

    def _load_from_config(self):
        """Load datasets from dataset_file configuration (similar to LeRobotDataset)."""
        with open(self.config.dataset_file, 'r') as f:
            dataset_configs = json.load(f)

        self.task_to_repos = {}
        self.repo_episodes = {}
        self.repo_info = {}
        self.all_tasks = []
        self.num_videos = 0

        for ds_cfg in tqdm(dataset_configs, desc="Loading datasets"):
            repo_id = ds_cfg.get("repo_id")
            ds_root = os.path.join(self.config.root, ds_cfg.get("root"))

            # Get episodes for this split
            sub_episodes = ds_cfg.get(self.config.split)
            if sub_episodes is None:
                # Load all episodes if not specified
                sub_episodes = None
            else:
                sub_episodes = get_episodes_idx(sub_episodes)

            print(f'\033[94mLoading episodes {sub_episodes if sub_episodes else "all"} from {ds_root} with split {self.config.split}\033[0m')

            try:
                # Load repo info and episodes
                repo_path = Path(ds_root)
                info = load_info(repo_path)
                episodes_dataset = load_episodes(repo_path)

                # Filter episodes if needed
                if sub_episodes is not None:
                    # Filter the dataset to only include specified episodes
                    episodes_dataset = episodes_dataset.select(sub_episodes)

                num_episodes = len(episodes_dataset)

                if num_episodes == 0:
                    print(f"  ⚠️  Skipping {repo_id}: no episodes in split {self.config.split}")
                    continue

                self.repo_info[repo_id] = info
                self.repo_episodes[repo_id] = episodes_dataset

                # Get task name from task_descriptions or use repo_id
                task_name = repo_id

                if task_name not in self.task_to_repos:
                    self.task_to_repos[task_name] = []

                self.task_to_repos[task_name].append({
                    'repo_id': repo_id,
                    'repo_path': str(repo_path),
                    'num_episodes': num_episodes,
                    'original_episode_indices': sub_episodes  # Store for later use
                })

                self.num_videos += num_episodes

                if task_name not in self.all_tasks:
                    self.all_tasks.append(task_name)

            except Exception as e:
                print(f"  ⚠️  Failed to load {repo_id} from {ds_root}: {e}")
                import traceback
                traceback.print_exc()
                continue

            self.task_name_to_idx = {
                task_name: i
                for i, task_name in enumerate(self.all_tasks)
            }

    def _discover_repos(self):
        """Recursively discover all LeRobot repos (fallback mode)."""
        root_path = Path(self.config.root)

        if not root_path.exists():
            raise ValueError(f"Root directory does not exist: {self.config.root}")

        # Look for directories with meta/info.json
        repos = {}
        for path in root_path.rglob("meta/info.json"):
            repo_path = path.parent.parent
            repo_id = repo_path.name
            repos[repo_id] = repo_path

        if not repos:
            raise ValueError(f"No LeRobot repos found in {self.config.root}")

        print(f"✓ Found {len(repos)} human demo repos")

        self.task_to_repos = {}
        self.repo_episodes = {}
        self.repo_info = {}
        self.all_tasks = []

        for repo_id, repo_path in tqdm(repos.items(), desc="Indexing repos"):
            try:
                info = load_info(repo_path)
                episodes_dataset = load_episodes(repo_path)

                self.repo_info[repo_id] = info
                self.repo_episodes[repo_id] = episodes_dataset

                num_episodes = len(episodes_dataset)

                # Get task name from task_descriptions or use repo_id
                task_name = self.task_descriptions.get(repo_id, repo_id)

                if task_name not in self.task_to_repos:
                    self.task_to_repos[task_name] = []

                self.task_to_repos[task_name].append({
                    'repo_id': repo_id,
                    'repo_path': str(repo_path),
                    'num_episodes': num_episodes,
                    'original_episode_indices': None
                })

                if task_name not in self.all_tasks:
                    self.all_tasks.append(task_name)

            except Exception as e:
                print(f"  ⚠️  Failed to index {repo_id}: {e}")
                continue

    def _sample_frames_uniform_jitter(self, start_frame: int, end_frame: int,
                                     num_frames: int) -> List[int]:
        """
        Sample frames with uniform spacing and random jitter.

        Strategy: Create uniform grid, then add random jitter to each point.
        This maintains overall temporal coverage while adding randomness.
        """
        total_frames = end_frame - start_frame + 1

        if total_frames <= num_frames:
            # If not enough frames, return all available
            return list(range(start_frame, end_frame + 1))

        # Create uniform grid
        indices = np.linspace(start_frame, end_frame, num_frames).astype(int)

        # Add random jitter
        max_jitter = min(self.config.max_jitter, total_frames // (num_frames * 2))
        if max_jitter > 0:
            jitter = np.random.randint(-max_jitter, max_jitter + 1, size=num_frames)
            indices = np.clip(indices + jitter, start_frame, end_frame)

        # Ensure monotonically increasing and unique
        indices = np.unique(indices)

        # If we lost frames due to uniqueness, resample
        if len(indices) < num_frames:
            # Fill in missing frames
            while len(indices) < num_frames:
                # Find largest gaps and insert frames there
                gaps = np.diff(indices)
                if len(gaps) == 0:
                    break
                max_gap_idx = np.argmax(gaps)
                new_frame = (indices[max_gap_idx] + indices[max_gap_idx + 1]) // 2
                indices = np.sort(np.append(indices, new_frame))

        return indices[:num_frames].tolist()

    def _sample_frames_random_interval(self, start_frame: int, end_frame: int,
                                      num_frames: int) -> List[int]:
        """
        Sample frames with random intervals.

        Strategy: Start from a random point, then use random intervals to select frames.
        """
        total_frames = end_frame - start_frame + 1

        if total_frames <= num_frames:
            return list(range(start_frame, end_frame + 1))

        # Calculate average interval
        avg_interval = total_frames / num_frames

        # Random starting point (with margin)
        margin = int(avg_interval)
        start_point = random.randint(start_frame, start_frame + margin)

        indices = [start_point]
        current = start_point

        for _ in range(num_frames - 1):
            # Random interval around average
            interval = max(1, int(np.random.normal(avg_interval, avg_interval * 0.3)))
            current = min(current + interval, end_frame)
            indices.append(current)

            if current >= end_frame:
                break

        # Ensure we have exactly num_frames
        while len(indices) < num_frames:
            indices.append(end_frame)

        return indices[:num_frames]

    def __len__(self) -> int:
        return self.num_videos

    def _get_target_item(self, task_name: str, task_idx: int = None) -> Dict[str, Any]:
        if task_idx is None:
            task_idx = self.task_name_to_idx[task_name]

        repo_info = self.task_to_repos[task_name][0] # Actually only one repo per task for human demos
        repo_id = repo_info['repo_id']
        repo_path = Path(repo_info['repo_path'])
        num_episodes = repo_info['num_episodes']

        episode_idx = random.randint(0, num_episodes - 1)

        if self.vla:
            ret_dict = {
                'language': self.task_descriptions[task_name][episode_idx],  # str
                'task': task_name,
                'task_id': torch.tensor(task_idx, dtype=torch.long),
                'episode_idx': torch.tensor(episode_idx, dtype=torch.long),
            }
            return ret_dict

        episodes_dataset = self.repo_episodes[repo_id]
        episode_row = episodes_dataset[episode_idx]

        # Get episode length
        episode_length = int(episode_row['length'])

        # For video file, frame indices from 0 to episode_length - 1
        start_frame = 0
        end_frame = episode_length - 1

        # Sample frames
        if self.config.sampling_strategy == "uniform_jitter":
            frame_indices = self._sample_frames_uniform_jitter(
                start_frame, end_frame, self.config.num_frames
            )
        else:  # random_interval
            frame_indices = self._sample_frames_random_interval(
                start_frame, end_frame, self.config.num_frames
            )

        # Get video file path and timestamps for this episode
        info = self.repo_info[repo_id]
        video_path_format = info.get('video_path')

        if video_path_format is None:
            raise ValueError(f"video_path not found in info for {repo_id}")

        all_view_tensors = []
        depth_videos = {}

        if self.config.pre_decode:
            # ── Cache path: directly load the pre-transformed tensor, using deterministic indices ──
            frame_indices = self._sample_frames_deterministic(
                start_frame, end_frame, self.config.num_frames
            )
            for cam_name in self.config.cameras:
                rgb_cache = self._episode_cache_path(task_name, episode_idx, cam_name, is_depth=False)
                if not rgb_cache.exists():
                    raise RuntimeError(
                        f"[CACHE MISS] {rgb_cache} — "
                        f"run pre_decode=True first for {task_name} ep{episode_idx} cam={cam_name}"
                    )
                video = torch.load(rgb_cache, map_location='cpu', weights_only=True)
                all_view_tensors.append(video)  # already [T, H, W, C]

                if self.config.include_depth:
                    depth_cache = self._episode_cache_path(
                        task_name, episode_idx, cam_name, is_depth=True
                    )
                    if depth_cache.exists():
                        depth_videos[cam_name] = torch.load(
                            depth_cache, map_location='cpu', weights_only=True
                        )  # [T, H, W, 1]

        else:
            # ── pre_decode=False: original logic, unchanged ─────────────────────
            # Sample frames
            if self.config.sampling_strategy == "uniform_jitter":
                frame_indices = self._sample_frames_uniform_jitter(
                    start_frame, end_frame, self.config.num_frames
                )
            else:  # random_interval
                frame_indices = self._sample_frames_random_interval(
                    start_frame, end_frame, self.config.num_frames
                )

            info = self.repo_info[repo_id]
            video_path_format = info.get('video_path')

            if video_path_format is None:
                raise ValueError(f"video_path not found in info for {repo_id}")

            fps = info['fps']

            for cam_name in self.config.cameras:
                video_key     = f"observation.images.{cam_name}"
                chunk_idx_key = f"videos/{video_key}/chunk_index"
                file_idx_key  = f"videos/{video_key}/file_index"

                if chunk_idx_key not in episode_row or file_idx_key not in episode_row:
                    continue

                chunk_idx = int(episode_row[chunk_idx_key])
                file_idx  = int(episode_row[file_idx_key])

                rgb_video_path = repo_path / video_path_format.format(
                    video_key=video_key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )

                if not rgb_video_path.exists():
                    continue

                from_ts_key = f"videos/{video_key}/from_timestamp"
                if from_ts_key in episode_row:
                    from_timestamp        = float(episode_row[from_ts_key])
                    video_start_frame     = round(from_timestamp * fps)
                    absolute_frame_indices = [video_start_frame + idx for idx in frame_indices]
                else:
                    absolute_frame_indices = frame_indices

                rgb_frames = self.video_reader.read_frames(str(rgb_video_path), absolute_frame_indices)

                depth_frames = None
                if self.config.include_depth:
                    depth_video_key  = f"observation.images.{cam_name}_depth"
                    depth_chunk_key  = f"videos/{depth_video_key}/chunk_index"
                    depth_file_key   = f"videos/{depth_video_key}/file_index"

                    if depth_chunk_key in episode_row and depth_file_key in episode_row:
                        depth_chunk_idx = int(episode_row[depth_chunk_key])
                        depth_file_idx  = int(episode_row[depth_file_key])

                        depth_video_path = repo_path / video_path_format.format(
                            video_key=depth_video_key,
                            chunk_index=depth_chunk_idx,
                            file_index=depth_file_idx,
                        )

                        if depth_video_path.exists():
                            depth_from_ts_key = f"videos/{depth_video_key}/from_timestamp"
                            if depth_from_ts_key in episode_row:
                                depth_from_timestamp    = float(episode_row[depth_from_ts_key])
                                depth_video_start_frame = round(depth_from_timestamp * fps)
                                depth_absolute_frame_indices = [
                                    depth_video_start_frame + idx for idx in frame_indices
                                ]
                            else:
                                depth_absolute_frame_indices = frame_indices

                            depth_rgb_frames = self.video_reader.read_frames(
                                str(depth_video_path), depth_absolute_frame_indices
                            )

                            depth_frames_list = []
                            for depth_rgb in depth_rgb_frames:
                                depth = decode_depth(depth_rgb, cam_name)
                                depth_frames_list.append(depth[..., None])
                            depth_frames = np.stack(depth_frames_list, axis=0)

                processed_rgb   = []
                processed_depth = []

                for t in range(len(rgb_frames)):
                    rgb_frame = rgb_frames[t]

                    if depth_frames is not None:
                        depth_frame = depth_frames[t]
                        transformed = self.spatial_transform(image=rgb_frame, depth=depth_frame)
                        rgb_t   = transformed['image']
                        depth_t = transformed['depth']
                    else:
                        transformed = self.spatial_transform(image=rgb_frame)
                        rgb_t   = transformed['image']
                        depth_t = None

                    rgb_final = self.rgb_transform(image=rgb_t)['image']
                    processed_rgb.append(rgb_final)

                    if depth_t is not None:
                        depth_final = self.depth_transform(image=depth_t)['image']
                        processed_depth.append(depth_final)

                rgb_video = torch.stack(processed_rgb, dim=0)
                all_view_tensors.append(rgb_video.permute(0, 2, 3, 1))  # [T, H, W, C]

                if processed_depth:
                    depth_video = torch.stack(processed_depth, dim=0)
                    depth_videos[cam_name] = depth_video.permute(0, 2, 3, 1)  # [T, H, W, 1]

        # ── The code below is completely unchanged (multi-view concat, ret_dict construction, etc.) ─────────
        if len(all_view_tensors) > 1:
            video = all_view_tensors[0]
        elif len(all_view_tensors) == 1:
            video = all_view_tensors[0]
        else:
            raise RuntimeError(f"No valid camera data for task {task_name}")

        # video is already [T, H, W, C]; both branches guarantee this,
        # so the old video = video.permute(0, 2, 3, 1) line was removed.
        # (The append calls in each branch already perform the permute.)

        ret_dict = {
            'video':       video,
            'task':        task_name,
            'task_id':     torch.tensor(task_idx, dtype=torch.long),
            'episode_idx': torch.tensor(episode_idx, dtype=torch.long),
        }

        if depth_videos:
            for cam_name, depth_video in depth_videos.items():
                ret_dict[f'depth_{cam_name}'] = depth_video

        return ret_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a random video clip from a random episode.

        Returns:
            dict: {
                'video': torch.Tensor [T, H, W, C],  # RGB video clip
                'depth_videos': Dict[str, torch.Tensor],  # Optional depth videos
                'task': str,  # Task name
                'task_id': int,  # Task ID
                'episode_idx': int,  # Episode index within dataset
            }
        """
        # 1. Select task (cycle through tasks based on idx)
        task_idx = idx % len(self.all_tasks)
        task_name = self.all_tasks[task_idx]

        return self._get_target_item(task_name, task_idx)

    def get_task_list(self) -> List[str]:
        """Get list of all available tasks."""
        return self.all_tasks.copy()

    def get_num_tasks(self) -> int:
        """Get number of unique tasks."""
        return len(self.all_tasks)