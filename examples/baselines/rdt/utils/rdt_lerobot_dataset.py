"""
RDT-specific LeRobot dataset helpers.

This module adds a sim-video pre-decode path for RDT training without touching
the shared LeRobot dataset implementation used by other baselines.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from examples.baselines.lerobot_dataset.lerobot_dataloader import (
    LeRobotDataConfig,
    build_lerobot_dataset,
)
from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer
from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanSimPairedDataset,
    InputMode,
    PairedDatasetConfig,
    TaskMapper,
    filter_dataset_config,
)
from examples.baselines.lerobot_dataset.lerobot_sim_dataset import (
    LeRobotSimDataConfig,
    LeRobotSimDataset,
    get_episdoes_idx,
)
from examples.baselines.lerobot_dataset.video_utils import FrameTimestampError


def _rdt_log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[RDT-DATA {timestamp} pid={os.getpid()}] {message}", flush=True)


@dataclass
class RDTLeRobotSimDataConfig(LeRobotSimDataConfig):
    pre_decode: bool = False
    pre_decode_cache_dir: str = "tmp/rdt_sim_video_cache"
    pre_decode_num_workers: int = 8
    skip_image_loading: bool = False


@dataclass
class RDTPairedDatasetConfig(PairedDatasetConfig):
    sim_pre_decode: bool = False
    sim_pre_decode_cache_dir: str = "tmp/rdt_sim_video_cache"
    sim_pre_decode_num_workers: int = 8
    sim_skip_image_loading: bool = False


class RDTPredecodedLeRobotDataset(LeRobotDataset):
    """LeRobotDataset with an optional pre-decoded sim-video cache."""

    def __init__(
        self,
        *args,
        pre_decode: bool = False,
        pre_decode_cache_dir: str = "tmp/rdt_sim_video_cache",
        pre_decode_num_workers: int = 8,
        pre_decode_video_keys: list[str] | None = None,
        **kwargs,
    ):
        self.pre_decode = bool(pre_decode)
        self.pre_decode_cache_dir = Path(pre_decode_cache_dir)
        self.pre_decode_num_workers = max(1, int(pre_decode_num_workers))
        self.pre_decode_video_keys = list(pre_decode_video_keys) if pre_decode_video_keys else None
        self.pre_decode_lock_timeout_s = float(
            os.getenv("RDT_SIM_PREDECODE_LOCK_TIMEOUT_S", "600")
        )
        self.dataset_init_heartbeat_s = float(
            os.getenv("RDT_DATASET_INIT_HEARTBEAT_S", "30")
        )
        self._pre_decode_cache: dict[str, tuple[Path, Path]] = {}
        self._runtime_video_cache: "OrderedDict[str, tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._runtime_video_cache_size = 16

        repo_id = kwargs.get("repo_id", args[0] if args else "<unknown>")
        root = kwargs.get("root", "<unknown>")
        episodes = kwargs.get("episodes")
        video_backend = kwargs.get("video_backend")
        _rdt_log(
            "LeRobotDataset init START "
            f"repo_id={repo_id} root={root} episodes={episodes} "
            f"pre_decode={self.pre_decode} cache_dir={self.pre_decode_cache_dir} "
            f"video_backend={video_backend}"
        )
        stop_heartbeat = threading.Event()

        def _heartbeat() -> None:
            start = time.monotonic()
            while not stop_heartbeat.wait(self.dataset_init_heartbeat_s):
                elapsed = time.monotonic() - start
                _rdt_log(
                    "LeRobotDataset init still running "
                    f"repo_id={repo_id} root={root} elapsed={elapsed:.1f}s"
                )

        heartbeat_thread = None
        if self.dataset_init_heartbeat_s > 0:
            heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
            heartbeat_thread.start()
        init_start = time.monotonic()
        try:
            super().__init__(*args, **kwargs)
        finally:
            stop_heartbeat.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
        init_elapsed = time.monotonic() - init_start
        _rdt_log(
            "LeRobotDataset init END "
            f"repo_id={repo_id} root={root} elapsed={init_elapsed:.2f}s "
            f"num_frames={getattr(self, 'num_frames', '<unknown>')} "
            f"num_episodes={getattr(self, 'num_episodes', '<unknown>')} "
            f"video_keys={list(getattr(self.meta, 'video_keys', []))}"
        )
        if self.pre_decode and len(self.meta.video_keys) > 0:
            predecode_start = time.monotonic()
            _rdt_log(f"RDT sim pre-decode START repo_id={repo_id} root={root}")
            self._pre_decode_all_videos()
            _rdt_log(
                "RDT sim pre-decode END "
                f"repo_id={repo_id} root={root} elapsed={time.monotonic() - predecode_start:.2f}s"
            )

    def _cache_paths_for(self, video_path: str) -> tuple[Path, Path]:
        import hashlib

        key = hashlib.md5(os.path.abspath(video_path).encode()).hexdigest()
        frames_path = self.pre_decode_cache_dir / f"{key}_frames.npy"
        pts_path = self.pre_decode_cache_dir / f"{key}_pts.npy"
        return frames_path, pts_path

    def _lock_path_for(self, video_path: str) -> Path:
        import hashlib

        key = hashlib.md5(os.path.abspath(video_path).encode()).hexdigest()
        return self.pre_decode_cache_dir / f"{key}.lock"

    def _collect_all_video_paths(self) -> list[str]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        video_keys = self.pre_decode_video_keys or list(self.meta.video_keys)
        paths = {
            str(self.root / self.meta.get_video_file_path(ep_idx, vid_key))
            for ep_idx in episodes
            for vid_key in video_keys
        }
        return sorted(paths)

    def _decode_all_frames_with_torchcodec(self, video_path: str) -> tuple[np.ndarray, np.ndarray]:
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(video_path)
        num_frames = decoder.metadata.num_frames
        frame_indices = list(range(num_frames))
        frame_batch = decoder.get_frames_at(indices=frame_indices)
        frames_np = frame_batch.data.permute(0, 2, 3, 1).cpu().numpy()
        if frames_np.dtype != np.uint8:
            frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)
        pts = frame_batch.pts_seconds.cpu().numpy().astype(np.float32)
        return frames_np, pts

    def _decode_all_frames(self, video_path: str) -> tuple[np.ndarray, np.ndarray]:
        if self.video_backend == "torchcodec":
            return self._decode_all_frames_with_torchcodec(video_path)

        from examples.baselines.lerobot_dataset.lerobot_human_dataset import VideoFrameReader

        reader = VideoFrameReader(backend=self.video_backend)
        frames_np = reader.read_all_frames(video_path)
        pts = (np.arange(len(frames_np), dtype=np.float32) / float(self.fps)).astype(np.float32)
        return frames_np, pts

    def _cleanup_incomplete_cache(self, frames_path: Path, pts_path: Path, lock_path: Path) -> None:
        with contextlib.suppress(OSError):
            lock_path.unlink()
        for path in (
            frames_path,
            pts_path,
            frames_path.with_suffix(".tmp.npy"),
            pts_path.with_suffix(".tmp.npy"),
        ):
            with contextlib.suppress(OSError):
                path.unlink()

    def _wait_for_cache(self, frames_path: Path, pts_path: Path, lock_path: Path) -> bool:
        start = time.monotonic()
        while lock_path.exists():
            if frames_path.exists() and pts_path.exists():
                return True
            elapsed = time.monotonic() - start
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
            except OSError:
                lock_age = 0.0
            if elapsed >= self.pre_decode_lock_timeout_s or lock_age >= self.pre_decode_lock_timeout_s:
                print(
                    f"Warning: removing stale sim pre-decode lock after {elapsed:.1f}s: {lock_path}"
                )
                self._cleanup_incomplete_cache(frames_path, pts_path, lock_path)
                return False
            time.sleep(0.2)
        return frames_path.exists() and pts_path.exists()

    def _decode_and_cache_one(self, video_path: str) -> tuple[str, bool]:
        frames_path, pts_path = self._cache_paths_for(video_path)
        lock_path = self._lock_path_for(video_path)

        if frames_path.exists() and pts_path.exists():
            _rdt_log(f"RDT sim pre-decode cache hit before lock video={video_path}")
            return video_path, True

        self.pre_decode_cache_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = None
        try:
            while True:
                if frames_path.exists() and pts_path.exists():
                    return video_path, True
                try:
                    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(lock_fd, f"{os.getpid()} {time.time()} {video_path}\n".encode())
                    _rdt_log(f"RDT sim pre-decode lock acquired video={video_path} lock={lock_path}")
                    break
                except FileExistsError:
                    _rdt_log(f"RDT sim pre-decode waiting for lock video={video_path} lock={lock_path}")
                    if self._wait_for_cache(frames_path, pts_path, lock_path):
                        return video_path, True

            frames_np, pts = self._decode_all_frames(video_path)
            frames_tmp = frames_path.with_suffix(".tmp.npy")
            pts_tmp = pts_path.with_suffix(".tmp.npy")
            np.save(str(frames_tmp), frames_np)
            np.save(str(pts_tmp), pts)
            frames_tmp.rename(frames_path)
            pts_tmp.rename(pts_path)
            _rdt_log(f"RDT sim pre-decode wrote cache video={video_path}")
            return video_path, True
        except Exception as exc:
            print(f"Warning: sim pre-decode failed for {video_path}: {exc}")
            return video_path, False
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
                if lock_path.exists():
                    with contextlib.suppress(OSError):
                        lock_path.unlink()

    def _pre_decode_all_videos(self) -> None:
        video_paths = self._collect_all_video_paths()
        self.pre_decode_cache_dir.mkdir(parents=True, exist_ok=True)

        cached_paths = []
        missing_paths = []
        for video_path in video_paths:
            frames_path, pts_path = self._cache_paths_for(video_path)
            if frames_path.exists() and pts_path.exists():
                self._pre_decode_cache[video_path] = (frames_path, pts_path)
                cached_paths.append(video_path)
            else:
                missing_paths.append(video_path)

        if not missing_paths:
            _rdt_log(
                f"Sim pre-decode cache hit: {len(cached_paths)}/{len(video_paths)} files already cached "
                f"in {self.pre_decode_cache_dir}"
            )
            return

        _rdt_log(
            f"Pre-decoding {len(missing_paths)}/{len(video_paths)} missing sim video files into "
            f"{self.pre_decode_cache_dir} with {self.pre_decode_num_workers} workers "
            f"({len(cached_paths)} cache hits) ..."
        )
        success = len(cached_paths)
        with ThreadPoolExecutor(max_workers=self.pre_decode_num_workers) as executor:
            futures = {executor.submit(self._decode_and_cache_one, path): path for path in missing_paths}
            for future in tqdm(as_completed(futures), total=len(missing_paths), desc="Sim pre-decode"):
                video_path, ok = future.result()
                frames_path, pts_path = self._cache_paths_for(video_path)
                if ok and frames_path.exists() and pts_path.exists():
                    self._pre_decode_cache[video_path] = (frames_path, pts_path)
                    success += 1
        _rdt_log(
            f"Sim pre-decode complete: {success}/{len(video_paths)} files cached "
            f"({len(cached_paths)} pre-existing hits)"
        )

    def _query_predecoded_video(
        self,
        video_path: str,
        timestamps: list[float],
        tolerance_s: float,
    ) -> torch.Tensor:
        frames_path, pts_path = self._cache_paths_for(video_path)
        if not frames_path.exists() or not pts_path.exists():
            raise FileNotFoundError(f"Missing pre-decoded cache for {video_path}")

        cached = self._runtime_video_cache.pop(video_path, None)
        if cached is None:
            frames = np.load(str(frames_path), mmap_mode="r")
            pts = np.load(str(pts_path), mmap_mode="r")
        else:
            frames, pts = cached
        self._runtime_video_cache[video_path] = (frames, pts)
        while len(self._runtime_video_cache) > self._runtime_video_cache_size:
            _, evicted = self._runtime_video_cache.popitem(last=False)
            del evicted

        if len(pts) == 0:
            raise FrameTimestampError(f"Pre-decoded cache has no timestamps: {video_path}")

        query_ts = np.asarray(timestamps, dtype=np.float32)
        right_idx = np.searchsorted(pts, query_ts, side="left")
        right_idx = np.clip(right_idx, 0, len(pts) - 1)
        left_idx = np.clip(right_idx - 1, 0, len(pts) - 1)
        left_dist = np.abs(query_ts - pts[left_idx])
        right_dist = np.abs(query_ts - pts[right_idx])
        use_right = right_dist < left_dist
        argmin = np.where(use_right, right_idx, left_idx)
        min_dist = np.where(use_right, right_dist, left_dist)
        if np.any(min_dist >= tolerance_s):
            raise FrameTimestampError(
                f"One or several query timestamps violate the tolerance ({min_dist[min_dist >= tolerance_s]} >= {tolerance_s})."
                f"\nqueried timestamps: {query_ts}"
                f"\nloaded timestamps: {pts[argmin]}"
                f"\nvideo: {video_path}"
            )

        selected = np.asarray(frames[argmin])
        tensor = torch.from_numpy(selected).permute(0, 3, 1, 2).float() / 255.0
        return tensor

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        if not self.pre_decode:
            return super()._query_videos(query_timestamps, ep_idx)

        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = str(self.root / self.meta.get_video_file_path(ep_idx, vid_key))
            try:
                frames = self._query_predecoded_video(video_path, shifted_query_ts, self.tolerance_s)
            except Exception:
                frames = super()._query_videos({vid_key: query_ts}, ep_idx)[vid_key].unsqueeze(0)
            item[vid_key] = frames.squeeze(0)
        return item


class RDTLeRobotSimDataset(LeRobotSimDataset):
    """RDT-only sim dataset with optional pre-decoded video loading."""

    def __init__(self, config: RDTLeRobotSimDataConfig):
        self.config = config

        if LeRobotDataset is None:
            raise ImportError("lerobot is not installed. Cannot use RDTLeRobotSimDataset.")

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

        action_offsets = [i / config.fps for i in range(self.horizon)]
        delta_timestamps = {self.action_key: action_offsets}

        used_cam_keys = []
        if not config.skip_image_loading:
            for cam in self.cameras:
                used_cam_keys.append(f"observation.images.{cam}")
                if config.include_depth:
                    used_cam_keys.append(f"observation.images.{cam}_depth")
        else:
            _rdt_log("image loading disabled by precomputed image cache")

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
                sub_episodes = ds_cfg.get(self.config.split)
                sub_episodes = get_episdoes_idx(sub_episodes)
                idx = len(sub_datasets) + 1
                total = len(dataset_configs)
                _rdt_log(
                    f"sub-dataset START {idx}/{total} repo_id={repo_id} "
                    f"root={ds_root} split={self.config.split} episodes={sub_episodes}"
                )
                sub_start = time.monotonic()
                sub_ds = RDTPredecodedLeRobotDataset(
                    repo_id=repo_id,
                    root=ds_root,
                    delta_timestamps=delta_timestamps,
                    video_backend=config.video_backend,
                    tolerance_s=config.tolerance_s,
                    episodes=sub_episodes,
                    pre_decode=config.pre_decode and not config.skip_image_loading,
                    pre_decode_cache_dir=config.pre_decode_cache_dir,
                    pre_decode_num_workers=config.pre_decode_num_workers,
                    pre_decode_video_keys=used_cam_keys,
                )
                sub_datasets.append(sub_ds)
                _rdt_log(
                    f"sub-dataset END {idx}/{total} repo_id={repo_id} "
                    f"elapsed={time.monotonic() - sub_start:.2f}s"
                )
        else:
            if not self.config.repo_id:
                raise ValueError("repo_id is required when dataset_file is not provided")
            sub_ds = RDTPredecodedLeRobotDataset(
                repo_id=self.config.repo_id,
                root=self.root,
                delta_timestamps=delta_timestamps,
                video_backend=config.video_backend,
                tolerance_s=config.tolerance_s,
                pre_decode=config.pre_decode and not config.skip_image_loading,
                pre_decode_cache_dir=config.pre_decode_cache_dir,
                pre_decode_num_workers=config.pre_decode_num_workers,
                pre_decode_video_keys=used_cam_keys,
            )
            sub_datasets.append(sub_ds)

        for sub_ds in sub_datasets:
            for k in list(sub_ds.meta.features.keys()):
                if sub_ds.meta.features[k]["dtype"] in ("image", "video", "depth") and k not in used_cam_keys:
                    sub_ds.meta.features.pop(k)

            cols_to_remove = [
                c for c in sub_ds.hf_dataset.column_names
                if c.startswith("observation.images.") and c not in used_cam_keys
            ]
            if cols_to_remove:
                _rdt_log(f"removing image columns repo_id={sub_ds.repo_id}: {cols_to_remove}")
                sub_ds.hf_dataset = sub_ds.hf_dataset.remove_columns(cols_to_remove)
            if config.skip_image_loading:
                remaining_image_cols = [
                    c for c in sub_ds.hf_dataset.column_names
                    if c.startswith("observation.images.")
                ]
                if remaining_image_cols:
                    raise RuntimeError(
                        "skip_image_loading=True but image columns remain "
                        f"for repo_id={sub_ds.repo_id}: {remaining_image_cols}"
                    )
                _rdt_log(f"image columns removed repo_id={sub_ds.repo_id}")

        self.lerobot_dataset = ConcatDataset(sub_datasets)
        self.main_dataset = sub_datasets[0]

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

        self._setup_transforms_sim(config)
        self.total_transitions = len(self.lerobot_dataset)
        self.target_length = self.total_transitions
        self.stats = self.main_dataset.meta.stats

        try:
            state_dim = self.main_dataset.features[self.state_key]["shape"][0]
            action_dim = self.main_dataset.features[self.action_key]["shape"][0]
            if self.config.single_arm:
                state_dim = state_dim // 2
                action_dim = action_dim // 2
            self.state_dim = state_dim
            self.action_dim = action_dim
        except (KeyError, AttributeError):
            self.state_dim = config.state_dim
            self.action_dim = config.action_dim


class RDTHumanSimPairedDataset(HumanSimPairedDataset):
    """Paired dataset that routes sim loading through the RDT-only predecode path."""

    def __init__(self, config: RDTPairedDatasetConfig):
        self.config = config
        self.input_mode = InputMode(config.input_mode)
        self.skip_human_video = False

        print("\n" + "=" * 80)
        print("🔄 INITIALIZING RDT HUMAN-SIM PAIRED DATASET")
        print("=" * 80)
        print(f"   Input mode: {self.input_mode.value}")
        print(f"   Include first frame: {config.include_first_frame}")

        print(f"\n📖 Loading task mapping from: {config.task_mapping_file}")
        self.task_mapper = TaskMapper(
            config.task_mapping_file,
            config.human_task_description_file,
            None,
            config.sim_task_description_file,
        )

        valid_human_ids = self.task_mapper.get_all_human_task_ids()
        valid_sim_ids = self.task_mapper.get_all_sim_task_ids()

        filtered_human_configs = filter_dataset_config(
            config.human_dataset_file, valid_human_ids, "human", config.split
        )
        filtered_sim_configs = filter_dataset_config(
            config.sim_dataset_file, valid_sim_ids, "sim", config.split
        )

        if not filtered_human_configs or not filtered_sim_configs:
            raise ValueError(f"No valid paired tasks found for split '{config.split}'")

        human_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(filtered_human_configs, human_temp, indent=2)
        human_temp.close()

        sim_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(filtered_sim_configs, sim_temp, indent=2)
        sim_temp.close()

        self.human_dataset = None
        if self.input_mode in [InputMode.VIDEO_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
            human_config = LeRobotDataConfig(
                source_type="human",
                root=config.human_root,
                split=config.split,
                dataset_file=human_temp.name,
                task_description_file=None,
                cameras=config.cameras,
                include_depth=config.include_depth,
                num_frames=config.num_frames,
                image_size=config.image_size,
                sampling_strategy=config.sampling_strategy,
                video_backend=config.video_backend,
                fps=config.fps,
                debug=config.debug,
                vla=False,
                enable_augmentation=config.enable_augmentation,
                pre_decode=config.pre_decode,
                pre_decode_cache_dir=config.pre_decode_cache_dir,
                pre_decode_num_workers=config.pre_decode_num_workers,
            )
            self.human_dataset = build_lerobot_dataset(human_config)

        sim_config = RDTLeRobotSimDataConfig(
            root=config.sim_root,
            split=config.split,
            dataset_file=sim_temp.name,
            task_description_file=None,
            cameras=config.cameras,
            include_depth=config.include_depth,
            image_size=config.image_size,
            state_type=config.state_type,
            single_arm=config.single_arm,
            horizon=config.horizon,
            obs_horizon=config.obs_horizon,
            fps=config.fps,
            video_backend=config.video_backend,
            debug=config.debug,
            enable_augmentation=config.enable_augmentation,
            pre_decode=config.sim_pre_decode,
            pre_decode_cache_dir=config.sim_pre_decode_cache_dir,
            pre_decode_num_workers=config.sim_pre_decode_num_workers,
            skip_image_loading=config.sim_skip_image_loading,
        )
        self.sim_dataset = RDTLeRobotSimDataset(sim_config)
        self._cached_language_by_sim_task: dict[str, str] = {}

        os.unlink(human_temp.name)
        os.unlink(sim_temp.name)

        self._build_and_validate_pairing()
        for sim_task_id in self.task_mapper.get_all_sim_task_ids():
            language = self.task_mapper.get_description("human", self.task_mapper.get_human_task_from_sim(sim_task_id))
            if language is None:
                language = self.task_mapper.get_description("sim", sim_task_id)
            if language is None:
                language = f"Task {sim_task_id}"
            self._cached_language_by_sim_task[sim_task_id] = language

    def __getitem__(self, idx: int) -> Dict[str, object]:
        actual_sim_idx = self.valid_indices[idx]
        sim_sample = self.sim_dataset[actual_sim_idx]
        sim_task_id = sim_sample.get("repo_id", sim_sample.get("task_name", ""))

        human_task_id = self.task_mapper.get_human_task_from_sim(sim_task_id)
        if human_task_id is None or human_task_id not in self.task_to_human_indices:
            raise ValueError(f"No human pairing for sim task: {sim_task_id}")

        robot_obs = {"states": sim_sample["states"]}
        view_idx = 1
        while f"view_{view_idx}" in sim_sample:
            robot_obs[f"view_{view_idx}"] = sim_sample[f"view_{view_idx}"]
            view_idx += 1

        result: Dict[str, object] = {
            "robot_obs": robot_obs,
            "robot_actions": sim_sample["actions"],
            "dataset_idx": sim_sample.get("dataset_idx", torch.tensor(0)),
            "human_task_id": human_task_id,
            "sim_task_id": sim_task_id,
            "sample_id": f"{sim_task_id}::{actual_sim_idx}",
        }

        if self.config.include_first_frame:
            result["robot_first_frame_obs"] = self._get_first_frame_obs(sim_sample)

        if self.input_mode in [InputMode.VIDEO_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
            human_task_info = self.task_to_human_indices[human_task_id]
            human_repo_id = human_task_info["repos"][0]["repo_id"]
            result["human_repo_id"] = human_repo_id

            if not self.skip_human_video:
                result["human_video"] = self.human_dataset._get_target_item(human_repo_id)["video"]

        if self.input_mode in [InputMode.LANGUAGE_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
            result["language"] = self._cached_language_by_sim_task.get(sim_task_id, f"Task {human_task_id}")

        return result
