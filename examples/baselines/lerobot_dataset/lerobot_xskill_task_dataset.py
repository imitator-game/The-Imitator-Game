"""
Integrated Task Pair Dataset
============================
Extends the original TaskPairDataset with:
- Multiple pairing modes (all_pairs, one_to_one, random_sample)
- First frame observation support
- Modality dropout for robust training
- Memory-efficient data loading

Pre-decode strategy (v2):
  Instead of decoding entire video files (10s of GB), we pre-decode only the
  `num_frames` frames needed per episode and store each as a small .pt file.
  Cache key is derived from (task_id, episode_idx, camera, num_frames, H, W)
  so it is stable across runs.  When a cache hit occurs, deterministic
  (no-jitter) frame indices are used for both video and state loading so they
  stay in sync.
"""

import os
import re
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

# Import from original
from lerobot.datasets.utils import load_episodes, load_info
from torchcodec.decoders import VideoDecoder

import hashlib

class PairingMode(Enum):
    ALL_PAIRS = "all_pairs"
    ONE_TO_ONE = "one_to_one"
    RANDOM_SAMPLE = "random_sample"


@dataclass
class IntegratedDatasetConfig:
    """Configuration for the integrated dataset.

    Supports three target domains:
      - "sim"   : ManiSkill simulation data (default)
      - "robot" : Real-robot LeRobot-format data (same column schema as sim)

    For robot domain:
      - Set ``target_domain = "robot"``
      - Set ``robot_root`` / ``robot_dataset_file`` / ``robot_task_description_file``
      - Set ``cameras`` to the camera names used in the robot dataset
        (e.g. ``["cam1", "cam2"]`` rather than the default ``["zed2i"]``)
      - Set ``depth_mode = "robot"`` if depth images are ever used
        (currently task encoder training uses RGB only — include_depth=False)
      - State / action keys are the same for sim and robot:
        qpos  → observation.qpos_gripper_states / action.qpos_gripper_actions
        eepos → observation.eepos_gripper_states / action.eepos_gripper_actions

    Task mapping JSON must contain ``robot_task_id`` entries alongside
    ``sim_task_id`` entries when using robot domain.
    """
    # Paths
    human_root: str = "demos/demo_data"
    robot_root: str = "demos/robot_data"
    sim_root: str = "demos/imitator_data"
    task_mapping_file: str = "examples/baselines/lerobot_dataset/task_mapping.json"

    human_dataset_file: Optional[str] = "examples/baselines/lerobot_dataset/config/human_config.json"
    robot_dataset_file: Optional[str] = "examples/baselines/lerobot_dataset/config/robot_config.json"
    sim_dataset_file: Optional[str] = "examples/baselines/lerobot_dataset/config/sim_config.json"

    human_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    robot_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/robot_desc.json"
    sim_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"

    # Domain settings
    # "sim" or "robot"
    target_domain: str = "sim"
    split: str = "train"

    # ── Camera settings ───────────────────────────────────────────────
    cameras: List[str] = field(default_factory=lambda: ["zed2i"])
    human_camera: str = "zed2i"
    include_depth: bool = False
    depth_mode: str = "sim"

    # Frame settings
    image_size: Tuple[int, int] = (224, 224)
    num_frames: int = 10
    sampling_strategy: str = "uniform_jitter"
    max_jitter: int = 3
    fps: int = 30

    # ── State settings ────────────────────────────────────────────────
    state_type: str = "qpos"
    single_arm: bool = False

    # MANO state mode
    mano_state_mode: str = "compact_aa"
    mano_interpolate_missing: bool = False
    mano_ema_alpha: float = 0.0
    mano_cross_view_camera: Optional[str] = None

    # Pairing settings
    pairing_mode: str = "random_sample"
    max_pairs_per_mapping: Optional[int] = 1000
    random_seed: int = 42

    # New features
    include_first_frame: bool = True
    modality_dropout_prob: float = 0.0

    # Debug
    debug: bool = False
    max_samples: Optional[int] = None

    enable_augmentation: bool = False
    use_dummy_mano_states: bool = False

    # ── Episode-level pre-decode ──────────────────────────────────────
    # When True, __init__ will iterate every unique (task, episode, camera)
    # tuple, sample `num_frames` frames deterministically, decode + transform
    # them, and save as a small .pt file under `pre_decode_cache_dir`.
    #
    # During __getitem__ a cache hit skips live decoding and uses the stored
    # tensor directly.  Frame indices for state loading are also derived
    # deterministically (no jitter) on a cache hit so video and states stay
    # in sync.
    pre_decode: bool = False
    pre_decode_cache_dir: str = "tmp/episode_frame_cache"
    pre_decode_num_workers: int = 8

    skip_states: bool = False


class TaskMapper:
    """Maps human tasks to robot/sim tasks with descriptions."""

    def __init__(
        self,
        mapping_file: str,
        human_desc_file: Optional[str] = None,
        robot_desc_file: Optional[str] = None,
        sim_desc_file: Optional[str] = None,
    ):
        with open(mapping_file, 'r') as f:
            data = json.load(f)

        self.task_mappings = data["task_mappings"]

        self.human_to_robot = {}
        self.human_to_sim = {}
        self.robot_to_human = {}
        self.sim_to_human = {}

        for m in self.task_mappings:
            h_id = m["human_task_id"]
            r_ids = m.get("robot_task_id", [])
            s_ids = m.get("sim_task_id", [])

            self.human_to_robot[h_id] = r_ids
            self.human_to_sim[h_id] = s_ids

            for r_id in r_ids:
                self.robot_to_human[r_id] = h_id
            for s_id in s_ids:
                self.sim_to_human[s_id] = h_id

        self.human_descriptions = self._load_json(human_desc_file)
        self.robot_descriptions = self._load_json(robot_desc_file)
        self.sim_descriptions = self._load_json(sim_desc_file)

    def _load_json(self, path: Optional[str]) -> Dict:
        if path:
            with open(path) as f:
                return json.load(f)
        return {}

    def get_description(self, task_id: str, domain: str) -> str:
        descs = {'human': self.human_descriptions,
                 'robot': self.robot_descriptions,
                 'sim': self.sim_descriptions}
        desc = descs.get(domain, {}).get(task_id, f"Task: {task_id}")
        return random.choice(desc) if isinstance(desc, list) else desc

    def get_all_ids(self, domain: str) -> Set[str]:
        if domain == "human":
            return set(self.human_to_robot.keys())
        elif domain == "sim":
            return set(self.sim_to_human.keys())
        return set(self.robot_to_human.keys())


class VideoFrameReader:
    """
    Efficient video frame reader using torchcodec, with LRU Decoder cache.
    """

    def __init__(self, cache_size: int = 16):
        from collections import OrderedDict
        self._cache: "OrderedDict" = OrderedDict()
        self._cache_size = cache_size

    def _get_decoder(self, video_path: str):
        if video_path in self._cache:
            self._cache.move_to_end(video_path)
            return self._cache[video_path]
        decoder = VideoDecoder(video_path)
        self._cache[video_path] = decoder
        self._cache.move_to_end(video_path)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return decoder

    def read_frames(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        decoder = self._get_decoder(video_path)
        max_frame = decoder.metadata.num_frames - 1
        valid_indices = [min(idx, max_frame) for idx in frame_indices]

        frame_batch = decoder.get_frames_at(indices=valid_indices)
        frames_tensor = frame_batch.data
        frames_np = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()

        if frames_np.dtype != np.uint8:
            frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

        return frames_np

    def read_single_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        """Read a single frame efficiently."""
        return self.read_frames(video_path, [frame_idx])[0]


# ============ MANO State Conversion Utilities ============

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    if axis_angle.ndim == 1:
        rot = R.from_rotvec(axis_angle)
        return rot.as_quat()
    else:
        rot = R.from_rotvec(axis_angle)
        return rot.as_quat()


def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    if axis_angle.ndim == 1:
        rot = R.from_rotvec(axis_angle)
        rotmat = rot.as_matrix()
        return np.concatenate([rotmat[:, 0], rotmat[:, 1]])
    else:
        rot = R.from_rotvec(axis_angle)
        rotmat = rot.as_matrix()
        col1 = rotmat[:, :, 0]
        col2 = rotmat[:, :, 1]
        return np.concatenate([col1, col2], axis=1)


def compute_gripper_from_joints(joints_3d: np.ndarray) -> float:
    if joints_3d.ndim == 1:
        joints_3d = joints_3d.reshape(21, 3)
    thumb_tip = joints_3d[THUMB_TIP]
    index_tip = joints_3d[INDEX_TIP]
    middle_tip = joints_3d[MIDDLE_TIP]
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
    thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
    return (thumb_index_dist + thumb_middle_dist) / 2.0


def get_mano_state_dim(mode: str, per_hand: bool = True) -> int:
    dims = {
        "compact_aa": 7,
        "compact_quat": 8,
        "original": 48,
        "full": 162,
    }
    dim = dims.get(mode, 48)
    return dim if per_hand else dim * 2


MANO_BIAS = np.array([0.09566993, 0.00638343, 0.00618631])


def is_all_zero(x: np.ndarray, eps: float = 1e-6) -> bool:
    return np.all(np.abs(x) < eps)


class EMAFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.state = None

    def reset(self):
        self.state = None

    def update(self, x: np.ndarray) -> np.ndarray:
        if is_all_zero(x):
            self.reset()
            return x
        if self.state is None:
            self.state = x.copy()
        else:
            self.state = self.alpha * x + (1 - self.alpha) * self.state
        return self.state.copy()


def cubic_interpolate_params(
    valid_indices: List[int],
    valid_values: np.ndarray,
    target_indices: List[int],
) -> np.ndarray:
    from scipy.interpolate import CubicSpline

    squeeze = False
    if valid_values.ndim == 1:
        valid_values = valid_values[:, None]
        squeeze = True

    D = valid_values.shape[1]

    if len(valid_indices) < 2 or len(target_indices) == 0:
        out = np.zeros((len(target_indices), D))
        return out.squeeze(-1) if squeeze else out

    vi = np.array(valid_indices)
    order = np.argsort(vi)
    vi = vi[order]
    vv = valid_values[order]

    targets = np.clip(np.array(target_indices, dtype=float), vi[0], vi[-1])
    result = np.zeros((len(target_indices), D))

    try:
        use_cubic = len(vi) >= 4
        for d in range(D):
            if use_cubic:
                cs = CubicSpline(vi, vv[:, d], bc_type='natural')
                result[:, d] = cs(targets)
            else:
                result[:, d] = np.interp(targets, vi, vv[:, d])
    except Exception:
        for d in range(D):
            result[:, d] = np.interp(targets, vi, vv[:, d])

    return result.squeeze(-1) if squeeze else result


def compute_cross_view_transform(
    orient_src: np.ndarray,
    cam_t_src: np.ndarray,
    orient_dst: np.ndarray,
    cam_t_dst: np.ndarray,
) -> np.ndarray:
    R_src = R.from_rotvec(orient_src).as_matrix()
    R_dst = R.from_rotvec(orient_dst).as_matrix()
    R_rel = R_dst @ R_src.T
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R_rel
    T[:3, 3] = (cam_t_dst - MANO_BIAS) - R_rel @ (cam_t_src - MANO_BIAS)
    return T


def apply_cross_view_to_params(
    global_orient: np.ndarray,
    cam_t: np.ndarray,
    transform: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    R_rel = transform[:3, :3]
    t_rel = transform[:3, 3]
    R_src = R.from_rotvec(global_orient).as_matrix()
    R_dst = R_rel @ R_src
    orient_dst = R.from_matrix(R_dst).as_rotvec().astype(np.float32)
    cam_t_dst = (R_rel @ (cam_t - MANO_BIAS) + t_rel + MANO_BIAS).astype(np.float32)
    return orient_dst, cam_t_dst


def mano_params_to_state_vector(
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    cam_t: np.ndarray,
    joints_3d: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "compact_aa":
        gripper = compute_gripper_from_joints(joints_3d)
        return np.concatenate([cam_t, global_orient, [gripper]])
    elif mode == "compact_quat":
        quat = axis_angle_to_quaternion(global_orient)
        gripper = compute_gripper_from_joints(joints_3d)
        return np.concatenate([cam_t, quat, [gripper]])
    elif mode == "original":
        return np.concatenate([global_orient, hand_pose])
    elif mode == "full":
        g_rot6d = axis_angle_to_rot6d(global_orient)
        p_rot6d = axis_angle_to_rot6d(hand_pose.reshape(15, 3)).flatten()
        return np.concatenate([g_rot6d, p_rot6d, cam_t, joints_3d])
    else:
        raise ValueError(f"Unknown mano_state_mode: {mode}")


def get_episodes_idx(sub_episodes: str) -> List[int]:
    if ':' not in sub_episodes:
        return eval(sub_episodes)
    start, end = map(int, sub_episodes.split(":"))
    return list(range(start, end))


_LEVEL_RE = re.compile(r'(?:^|_)L([0-3])(?:_|-v\d|$)')


def extract_level_id(task_id: str) -> int:
    m = _LEVEL_RE.search(task_id)
    return int(m.group(1)) if m else -1


class IntegratedTaskPairDataset(Dataset):
    """
    Integrated dataset supporting:
    - Multiple pairing modes
    - First frame observations
    - Modality dropout
    - Episode-level pre-decode cache (new in v2)
    """

    MANO_GLOBAL_ORIENT_DIM = 3
    MANO_HAND_POSE_DIM = 45
    MANO_BETAS_DIM = 10
    MANO_PER_HAND_DIM = 58
    MANO_TOTAL_DIM = 116

    MANO_STATE_DIMS = {
        "compact_aa": 7,
        "compact_quat": 8,
        "original": 48,
        "full": 162,
    }

    def __init__(self, config: IntegratedDatasetConfig):
        self.config = config
        self.rng = random.Random(config.random_seed)
        np.random.seed(config.random_seed)

        print("\n" + "=" * 80)
        print("INITIALIZING INTEGRATED TASK PAIR DATASET")
        print("=" * 80)

        self.video_reader = VideoFrameReader()
        self._setup_transforms_task(config)

        print(f"\nLoading task mapping from: {config.task_mapping_file}")
        self.task_mapper = TaskMapper(
            config.task_mapping_file,
            config.human_task_description_file,
            config.robot_task_description_file,
            config.sim_task_description_file,
        )

        print(f"\nLoading HUMAN dataset...")
        self._load_human_data()

        print(f"\nLoading {config.target_domain.upper()} dataset...")
        self._load_target_data()

        print(f"\nBuilding pairs with mode: {config.pairing_mode}")
        self._build_pair_index()

        self._build_level_index()

        all_target_tasks = sorted(set(p['target_task'] for p in self.pair_index))
        self.target_task_to_idx = {name: idx for idx, name in enumerate(all_target_tasks)}
        self.num_target_tasks = len(all_target_tasks)
        print(f"  Target task ID mapping ({self.num_target_tasks} tasks):")
        for name, idx in self.target_task_to_idx.items():
            print(f"    {idx}: {name}")

        self._print_summary()

        from collections import OrderedDict as _OD
        self._pq_cache: "_OD" = _OD()
        self._pq_cache_size: int = 64

        if config.pre_decode:
            self._pre_decode_all_episodes()

    # ====================================================================
    #  Episode-level pre-decode cache  (v2)
    # ====================================================================

    def _episode_cache_path(
        self,
        task_id: str,
        episode_idx: int,
        camera: str,
        is_first_frame: bool = False,
    ) -> Path:
        """
        Return a stable, filesystem-safe .pt path for one decoded episode.

        The key encodes everything that affects the stored tensor:
          task_id | episode_idx | camera | num_frames | H | W
        First-frame caches get a separate key (different transforms applied).
        """
        H, W = self.config.image_size
        tag = "first" if is_first_frame else f"T{self.config.num_frames}"
        raw = f"{task_id}|{episode_idx}|{camera}|{self.config.split}|{self.config.target_domain}"
        key = hashlib.md5(raw.encode()).hexdigest()
        suffix = "_first.pt" if is_first_frame else ".pt"
        return Path(self.config.pre_decode_cache_dir) / f"{key}{suffix}"

    def _sample_frame_indices_deterministic(
        self, start: int, end: int, num: int
    ) -> List[int]:
        """
        Uniform frame sampling with NO jitter.
        Used during pre-decode and on cache hits (keeps video and states in sync).
        """
        total = end - start + 1
        if total <= num:
            indices = list(range(start, end + 1))
            while len(indices) < num:
                indices.append(indices[-1])
            return indices
        return list(np.linspace(start, end, num).astype(int))

    def _pre_decode_episode(
        self,
        task_id: str,
        episode_idx: int,
        camera: str,
        repo_info: Dict,
        repo_episodes,
        is_human: bool,
    ) -> bool:
        """
        Decode `num_frames` frames for a single (task, episode, camera) tuple,
        apply spatial + RGB transforms, and persist as a .pt file.

        Human episodes also get a separate first-frame .pt file when
        `include_first_frame` is True (target domain uses the same logic).

        Returns True on success, False on any error.
        """
        reader = VideoFrameReader(cache_size=1)
        video_cache_path = self._episode_cache_path(task_id, episode_idx, camera)
        first_cache_path = self._episode_cache_path(task_id, episode_idx, camera, is_first_frame=True)

        need_video = not video_cache_path.exists()
        need_first = self.config.include_first_frame and not first_cache_path.exists()

        if not need_video and not need_first:
            return True  # already fully cached

        try:
            episode_row = repo_episodes[episode_idx]
            episode_len = int(episode_row["length"])

            info = repo_info["info"]
            repo_path = repo_info["repo_path"]
            fps = info["fps"]
            video_path_format = info["video_path"]

            video_key = f"observation.images.{camera}"
            chunk_idx = int(episode_row[f"videos/{video_key}/chunk_index"])
            file_idx = int(episode_row[f"videos/{video_key}/file_index"])

            video_path = Path(repo_path) / video_path_format.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )

            from_ts = float(episode_row[f"videos/{video_key}/from_timestamp"])
            video_start = round(from_ts * fps)

            # ── Decode and cache multi-frame clip ──────────────────────────
            if need_video:
                frame_indices = self._sample_frame_indices_deterministic(
                    0, episode_len - 1, self.config.num_frames
                )
                abs_indices = [video_start + i for i in frame_indices]
                rgb_frames = reader.read_frames(str(video_path), abs_indices)

                processed = []
                for frame in rgb_frames:
                    t = self.spatial_transform(image=frame)["image"]
                    t = self.rgb_transform(image=t)["image"]
                    processed.append(t)

                # [T, C, H, W] → [T, H, W, C]  (matches _load_video_frames output)
                video_tensor = torch.stack(processed, dim=0).permute(0, 2, 3, 1)

                tmp = video_cache_path.with_suffix(".tmp.pt")
                torch.save(video_tensor, tmp)
                tmp.rename(video_cache_path)

            # ── Decode and cache first frame ───────────────────────────────
            if need_first:
                first_frame_raw = reader.read_frames(str(video_path), [video_start])[0]
                first_frame = self.spatial_transform(image=first_frame_raw)["image"]
                # First frame always uses eval_transform (no augmentation)
                first_frame = self.eval_transform(image=first_frame)["image"]
                # Result is [C, H, W] tensor

                tmp = first_cache_path.with_suffix(".tmp.pt")
                torch.save(first_frame, tmp)
                tmp.rename(first_cache_path)

            return True

        except Exception as exc:
            print(f"  ⚠️  Pre-decode failed for {task_id} ep{episode_idx} cam={camera}: {exc}")
            return False

    def _pre_decode_all_episodes(self) -> None:
        """
        Iterate every unique (task, episode, camera) combination in the pair
        index and pre-decode + cache the required frames.

        Runs with a thread pool for parallel I/O.  Already-cached episodes are
        skipped instantly (idempotent — safe to re-run after a crash).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cache_dir = Path(self.config.pre_decode_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Collect unique decode jobs ─────────────────────────────────────────
        # job = (task_id, episode_idx, camera, repo_info, repo_episodes, is_human)
        seen: Set[Tuple[str, int, str]] = set()
        jobs = []

        for pair in self.pair_index:
            # Human side
            h_key = (pair["human_task"], pair["human_episode"], self.config.human_camera)
            if h_key not in seen:
                seen.add(h_key)
                jobs.append((
                    pair["human_task"],
                    pair["human_episode"],
                    self.config.human_camera,
                    self.human_repo_info[pair["human_task"]],
                    self.human_repo_episodes[pair["human_task"]],
                    True,
                ))

            # Target side (primary camera only — used in __getitem__)
            target_cam = self.config.cameras[0]
            t_key = (pair["target_task"], pair["target_episode"], target_cam)
            if t_key not in seen:
                seen.add(t_key)
                jobs.append((
                    pair["target_task"],
                    pair["target_episode"],
                    target_cam,
                    self.target_repo_info[pair["target_task"]],
                    self.target_repo_episodes[pair["target_task"]],
                    False,
                ))

        n_workers = self.config.pre_decode_num_workers
        print(f"\n[pre-decode] {len(jobs)} unique (task, episode, camera) jobs "
              f"→ {cache_dir}  (workers={n_workers})")

        success = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._pre_decode_episode, *job): job
                for job in jobs
            }
            for future in tqdm(
                as_completed(futures), total=len(jobs), desc="Pre-decoding episodes"
            ):
                ok = future.result()
                if ok:
                    success += 1

        print(f"[pre-decode] Done: {success}/{len(jobs)} episodes cached.\n")

    # ====================================================================
    #  Core video loading  (cache-aware)
    # ====================================================================

    def _load_video_frames_cached(
        self,
        task_id: str,
        episode_idx: int,
        camera: str,
        repo_info: Dict,
        episode_row: Dict,
        episode_len: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Load video frames for one episode, returning both the tensor and the
        frame indices used (needed for state loading).

        Cache hit  → load .pt, return deterministic indices (no jitter).
        Cache miss → live decode with optional jitter, same as before.

        Returns:
            video  : [T, H, W, C] float32 tensor
            frames : List[int] of episode-relative frame indices
        """
        cache_path = self._episode_cache_path(task_id, episode_idx, camera)

        if self.config.pre_decode:
            if not cache_path.exists():
                raise RuntimeError(
                    f"[CACHE MISS] {cache_path} for {task_id} ep={episode_idx} cam={camera}"
                )

            video = torch.load(cache_path, map_location="cpu")
            frames = self._sample_frame_indices_deterministic(
                0, episode_len - 1, self.config.num_frames
            )
            return video, frames

        # Live decode only when pre_decode is disabled
        frames = self._sample_frame_indices(0, episode_len - 1, self.config.num_frames)
        video = self._load_video_frames(repo_info, episode_row, frames, camera)
        return video, frames

    def _load_first_frame_cached(
        self,
        task_id: str,
        episode_idx: int,
        camera: str,
        repo_info: Dict,
        episode_row: Dict,
    ) -> torch.Tensor:
        """
        Load the first frame, using the episode-level cache when available.

        Returns:
            [C, H, W] float32 tensor
        """
        cache_path = self._episode_cache_path(task_id, episode_idx, camera, is_first_frame=True)

        if self.config.pre_decode and cache_path.exists():
            return torch.load(cache_path, weights_only=True)

        return self._load_first_frame(repo_info, episode_row, camera)

    # ====================================================================
    #  Transforms
    # ====================================================================

    def _setup_transforms_task(self, config):
        H, W = config.image_size

        self.spatial_transform = A.Compose([
            A.Resize(height=H, width=W),
        ])

        if config.enable_augmentation:
            self.train_transform = A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            if self.config.skip_states:
                self.train_transform = A.Compose([ToTensorV2()])
            else:
                self.train_transform = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ])

        self.eval_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.rgb_transform = (
            self.train_transform if config.split == 'train' else self.eval_transform
        )

    # ====================================================================
    #  Data loading helpers (unchanged from original)
    # ====================================================================

    def _load_human_data(self):
        with open(self.config.human_dataset_file, 'r') as f:
            dataset_configs = json.load(f)

        self.human_repo_info = {}
        self.human_repo_episodes = {}
        self.human_tasks = []

        valid_ids = self.task_mapper.get_all_ids('human')

        for ds_cfg in tqdm(dataset_configs, desc="Loading human repos"):
            repo_id = ds_cfg["repo_id"]
            if repo_id not in valid_ids:
                continue

            ds_root = os.path.join(self.config.human_root, ds_cfg["root"])
            sub_episodes = get_episodes_idx(ds_cfg[self.config.split])

            repo_path = Path(ds_root)
            info = load_info(repo_path)
            episodes = load_episodes(repo_path).select(sub_episodes)

            self.human_repo_info[repo_id] = {
                'info': info,
                'repo_path': repo_path,
                'num_episodes': len(episodes),
            }
            self.human_repo_episodes[repo_id] = episodes
            self.human_tasks.append(repo_id)

            if self.config.debug:
                print(f"  {repo_id}: {len(episodes)} episodes")

    def _load_target_data(self):
        domain = self.config.target_domain
        dataset_file = self.config.sim_dataset_file if domain == "sim" else self.config.robot_dataset_file
        root = self.config.sim_root if domain == "sim" else self.config.robot_root

        with open(dataset_file, 'r') as f:
            dataset_configs = json.load(f)

        self.target_repo_info = {}
        self.target_repo_episodes = {}
        self.target_tasks = []

        valid_ids = self.task_mapper.get_all_ids(domain)

        for ds_cfg in tqdm(dataset_configs, desc=f"Loading {domain} repos"):
            repo_id = ds_cfg["repo_id"]
            if repo_id not in valid_ids:
                continue

            ds_root = os.path.join(root, ds_cfg["root"])
            sub_episodes = get_episodes_idx(ds_cfg[self.config.split])

            repo_path = Path(ds_root)
            info = load_info(repo_path)
            episodes = load_episodes(repo_path).select(sub_episodes)

            self.target_repo_info[repo_id] = {
                'info': info,
                'repo_path': repo_path,
                'num_episodes': len(episodes),
            }
            self.target_repo_episodes[repo_id] = episodes
            self.target_tasks.append(repo_id)

            if self.config.debug:
                print(f"  {repo_id}: {len(episodes)} episodes")

    def _build_pair_index(self):
        self.pair_index = []
        mode = PairingMode(self.config.pairing_mode)

        for mapping in self.task_mapper.task_mappings:
            human_id = mapping["human_task_id"]
            if human_id not in self.human_tasks:
                continue

            target_key = "sim_task_id" if self.config.target_domain == "sim" else "robot_task_id"
            target_ids = mapping.get(target_key, [])

            human_info = self.human_repo_info[human_id]
            h_num = human_info['num_episodes']

            for target_id in target_ids:
                if target_id not in self.target_tasks:
                    continue

                target_info = self.target_repo_info[target_id]
                t_num = target_info['num_episodes']

                pairs = self._generate_pairs_for_task(human_id, h_num, target_id, t_num, mode)
                self.pair_index.extend(pairs)

        if self.config.max_samples and len(self.pair_index) > self.config.max_samples:
            self.pair_index = self.rng.sample(self.pair_index, self.config.max_samples)

        print(f"  Built {len(self.pair_index)} pairs")

    def _generate_pairs_for_task(
        self, human_id: str, h_num: int,
        target_id: str, t_num: int,
        mode: PairingMode
    ) -> List[Dict]:
        pairs = []

        if mode == PairingMode.ALL_PAIRS:
            for h in range(h_num):
                for t in range(t_num):
                    pairs.append({
                        'human_task': human_id, 'human_episode': h,
                        'target_task': target_id, 'target_episode': t,
                    })

        elif mode == PairingMode.ONE_TO_ONE:
            for i in range(min(h_num, t_num)):
                pairs.append({
                    'human_task': human_id, 'human_episode': i,
                    'target_task': target_id, 'target_episode': i,
                })

        elif mode == PairingMode.RANDOM_SAMPLE:
            all_pairs = [(h, t) for h in range(h_num) for t in range(t_num)]
            max_pairs = self.config.max_pairs_per_mapping or len(all_pairs)

            if len(all_pairs) > max_pairs:
                all_pairs = self.rng.sample(all_pairs, max_pairs)

            for h, t in all_pairs:
                pairs.append({
                    'human_task': human_id, 'human_episode': h,
                    'target_task': target_id, 'target_episode': t,
                })

        return pairs

    def _build_level_index(self) -> None:
        self.level_index: Dict[int, List[int]] = defaultdict(list)
        self.task_level_index: Dict[Tuple[str, int], List[int]] = defaultdict(list)

        for i, pair in enumerate(self.pair_index):
            level = extract_level_id(pair["target_task"])
            self.level_index[level].append(i)
            self.task_level_index[(pair["human_task"], level)].append(i)

        total = len(self.pair_index)
        print(f"\n[V4] Level index built ({total} total pairs):")
        for lv in sorted(k for k in self.level_index if k >= 0):
            n = len(self.level_index[lv])
            print(f"     L{lv}: {n:>6} pairs  ({n / total * 100:5.1f}%)")
        n_unk = len(self.level_index.get(-1, []))
        if n_unk:
            print(f"     [warn] {n_unk} pairs have no level tag — "
                  "check task_mapping.json naming convention")

    def get_cross_level_pair(self, idx: int) -> Optional[Dict[str, Any]]:
        pair = self.pair_index[idx]
        current_level = extract_level_id(pair["target_task"])
        human_task = pair["human_task"]

        other_levels = [
            lv for lv in range(4)
            if lv != current_level
            and (human_task, lv) in self.task_level_index
        ]

        if not other_levels:
            return None

        target_level = self.rng.choice(other_levels)
        other_idx = self.rng.choice(self.task_level_index[(human_task, target_level)])
        return self[other_idx]

    def _sample_frame_indices(self, start: int, end: int, num: int) -> List[int]:
        """Sample frame indices with optional jitter (used on cache miss / no pre-decode)."""
        total = end - start + 1

        if total <= num:
            indices = list(range(start, end + 1))
            while len(indices) < num:
                indices.append(indices[-1])
            return indices

        indices = np.linspace(start, end, num).astype(int)

        if self.config.sampling_strategy == "uniform_jitter" and self.config.split == "train":
            max_j = min(self.config.max_jitter, total // (num * 2))
            if max_j > 0:
                jitter = np.random.randint(-max_j, max_j + 1, size=num)
                indices = np.clip(indices + jitter, start, end)

        return list(np.sort(indices))

    def _load_video_frames(
        self,
        repo_info: Dict,
        episode_row: Dict,
        frame_indices: List[int],
        camera: str,
    ) -> torch.Tensor:
        """Live decode + transform for a list of episode-relative frame indices."""
        repo_path = repo_info["repo_path"]
        info = repo_info["info"]
        fps = info["fps"]
        video_path_format = info["video_path"]

        video_key = f"observation.images.{camera}"
        chunk_idx = int(episode_row[f"videos/{video_key}/chunk_index"])
        file_idx = int(episode_row[f"videos/{video_key}/file_index"])

        video_path = Path(repo_path) / video_path_format.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )

        from_ts = float(episode_row[f"videos/{video_key}/from_timestamp"])
        video_start = round(from_ts * fps)
        abs_indices = [video_start + idx for idx in frame_indices]

        rgb_frames = self.video_reader.read_frames(str(video_path), abs_indices)

        processed = []
        for frame in rgb_frames:
            t = self.spatial_transform(image=frame)["image"]
            t = self.rgb_transform(image=t)["image"]
            processed.append(t)

        video = torch.stack(processed, dim=0)      # [T, C, H, W]
        return video.permute(0, 2, 3, 1)           # [T, H, W, C]

    def _load_first_frame(
        self,
        repo_info: Dict,
        episode_row: Dict,
        camera: str,
    ) -> torch.Tensor:
        """Live decode + eval transform for the episode's first frame."""
        repo_path = repo_info["repo_path"]
        info = repo_info["info"]
        fps = info["fps"]
        video_path_format = info["video_path"]

        video_key = f"observation.images.{camera}"
        chunk_idx = int(episode_row[f"videos/{video_key}/chunk_index"])
        file_idx = int(episode_row[f"videos/{video_key}/file_index"])

        video_path = Path(repo_path) / video_path_format.format(
            video_key=video_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )

        from_ts = float(episode_row[f"videos/{video_key}/from_timestamp"])
        frame_idx = round(from_ts * fps)

        frame = self.video_reader.read_frames(str(video_path), [frame_idx])[0]
        frame = self.spatial_transform(image=frame)["image"]
        frame = self.eval_transform(image=frame)["image"]
        return frame  # [C, H, W]

    def _read_parquet_cached(self, parquet_path) -> "pd.DataFrame":
        key = str(parquet_path)
        if key in self._pq_cache:
            self._pq_cache.move_to_end(key)
            return self._pq_cache[key]
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        self._pq_cache[key] = df
        self._pq_cache.move_to_end(key)
        if len(self._pq_cache) > self._pq_cache_size:
            self._pq_cache.popitem(last=False)
        return df

    # ====================================================================
    #  MANO state loading (unchanged from original)
    # ====================================================================

    def _load_episode_parquet(self, task_id: str, episode_idx: int) -> "pd.DataFrame":
        repo_info = self.human_repo_info[task_id]
        repo_path = repo_info['repo_path']
        info = repo_info['info']
        data_path_format = info['data_path']

        episodes = self.human_repo_episodes[task_id]
        episode_row = episodes[episode_idx]

        chunk_idx = int(episode_row["data/chunk_index"])
        file_idx = int(episode_row["data/file_index"])

        parquet_path = repo_path / data_path_format.format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        df = self._read_parquet_cached(parquet_path)

        ep_index = int(episode_row['episode_index'])
        return df[df['episode_index'] == ep_index]

    @staticmethod
    def _extract_raw_hand_params(
        ep_df: "pd.DataFrame",
        cam: str,
        hand_type: str,
    ) -> Tuple[Dict[str, np.ndarray], List[bool]]:
        prefix = f"observation.hand.{hand_type}.{cam}"
        n = len(ep_df)

        orient_key   = f"{prefix}.mano_global_orient"
        pose_key     = f"{prefix}.mano_hand_pose"
        cam_t_key    = f"{prefix}.pred_cam_t_full"
        joints_key   = f"{prefix}.pred_keypoints_3d"
        betas_key    = f"{prefix}.mano_betas"
        is_right_key = f"{prefix}.is_right"

        has_joints   = joints_key   in ep_df.columns
        has_betas    = betas_key    in ep_df.columns
        has_is_right = is_right_key in ep_df.columns

        orient_list, pose_list, cam_t_list = [], [], []
        joints_list, betas_list = [], []
        valid: List[bool] = []

        for i in range(n):
            row = ep_df.iloc[i]

            orient = np.array(row[orient_key]).flatten()
            pose   = np.array(row[pose_key]).flatten()
            cam_t  = np.array(row[cam_t_key]).flatten()

            orient_list.append(orient)
            pose_list.append(pose)
            cam_t_list.append(cam_t)
            joints_list.append(
                np.array(row[joints_key]).flatten() if has_joints else np.zeros(63)
            )
            betas_list.append(
                np.array(row[betas_key]).flatten() if has_betas else np.zeros(10)
            )

            is_right_flag = float(row[is_right_key]) if has_is_right else 1.0
            valid.append(is_right_flag >= 0 and not is_all_zero(orient))

        params = {
            'global_orient': np.stack(orient_list),
            'hand_pose':     np.stack(pose_list),
            'cam_t':         np.stack(cam_t_list),
            'joints_3d':     np.stack(joints_list),
            'betas':         np.stack(betas_list),
        }
        return params, valid

    def _interpolate_cam_t(self, params, valid, hand_type=""):
        valid_idx   = [i for i, v in enumerate(valid) if v]
        invalid_idx = [i for i, v in enumerate(valid) if not v]

        if not invalid_idx or len(valid_idx) < 2:
            return 0

        cam_ts = params['cam_t']
        interp = cubic_interpolate_params(valid_idx, cam_ts[valid_idx], invalid_idx)
        for j, inv in enumerate(invalid_idx):
            cam_ts[inv] = interp[j]

        if self.config.debug and len(invalid_idx) > 0:
            print(f"      interpolated cam_t for {len(invalid_idx)} frames"
                  f"{f' ({hand_type})' if hand_type else ''}")
        return len(invalid_idx)

    def _fill_pose_from_cross_view(self, primary_params, primary_valid, xv_sources, primary_cam, hand_type):
        n = len(primary_valid)
        if not xv_sources:
            return 0

        transforms: List[Optional[np.ndarray]] = []
        for xv_params, xv_valid, _cam in xv_sources:
            calib_idx = None
            for i in range(n):
                if primary_valid[i] and xv_valid[i]:
                    calib_idx = i
                    break
            if calib_idx is not None:
                transforms.append(compute_cross_view_transform(
                    xv_params['global_orient'][calib_idx],
                    xv_params['cam_t'][calib_idx],
                    primary_params['global_orient'][calib_idx],
                    primary_params['cam_t'][calib_idx],
                ))
            else:
                transforms.append(None)

        filled = 0
        for i in range(n):
            if primary_valid[i]:
                continue

            candidates = []
            for s, (xv_p, xv_v, _) in enumerate(xv_sources):
                if xv_v[i] and transforms[s] is not None:
                    orient_dst, _ = apply_cross_view_to_params(
                        xv_p['global_orient'][i], xv_p['cam_t'][i], transforms[s],
                    )
                    candidates.append((s, orient_dst))

            if not candidates:
                continue

            if len(candidates) == 1:
                best_s, best_orient = candidates[0]
            else:
                prev = primary_params['global_orient'][i - 1] if i > 0 else None
                if prev is None or is_all_zero(prev):
                    best_s, best_orient = candidates[0]
                else:
                    best_s, best_orient = min(
                        candidates, key=lambda c: float(np.linalg.norm(c[1] - prev)),
                    )

            xv_best = xv_sources[best_s][0]
            primary_params['global_orient'][i] = best_orient
            primary_params['hand_pose'][i]  = xv_best['hand_pose'][i]
            primary_params['betas'][i]      = xv_best['betas'][i]
            primary_params['joints_3d'][i]  = xv_best['joints_3d'][i]
            filled += 1

        if filled > 0 and self.config.debug:
            cam_names = [s[2] for s in xv_sources]
            print(f"      cross-view filled pose for {filled} frames "
                  f"({hand_type}: {cam_names} → {primary_cam})")
        return filled

    @staticmethod
    def _apply_ema(params, valid, alpha):
        n = len(valid)
        for key in params:
            ema = EMAFilter(alpha)
            for i in range(n):
                if valid[i]:
                    params[key][i] = ema.update(params[key][i])
                else:
                    ema.reset()

    def _load_mano_states_dummy(self, task_id, episode_idx, frame_indices):
        repo_info = self.human_repo_info[task_id]
        repo_path = repo_info['repo_path']
        info = repo_info['info']
        cam = self.config.human_camera
        data_path_format = info['data_path']

        episodes = self.human_repo_episodes[task_id]
        episode_row = episodes[episode_idx]

        chunk_idx = int(episode_row["data/chunk_index"])
        file_idx = int(episode_row["data/file_index"])

        parquet_path = repo_path / data_path_format.format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        df = self._read_parquet_cached(parquet_path)

        ep_index = int(episode_row['episode_index'])
        ep_df = df[df['episode_index'] == ep_index]

        max_idx = len(ep_df) - 1
        frame_indices = [min(idx, max_idx) for idx in frame_indices]

        states = []
        for idx in frame_indices:
            row = ep_df.iloc[idx]
            left_orient = np.array(row[f"observation.hand.left.{cam}.mano_global_orient"])
            left_pose = np.array(row[f"observation.hand.left.{cam}.mano_hand_pose"])
            left_betas = np.array(row[f"observation.hand.left.{cam}.mano_betas"])
            left = np.concatenate([left_orient, left_pose, left_betas])
            right_orient = np.array(row[f"observation.hand.right.{cam}.mano_global_orient"])
            right_pose = np.array(row[f"observation.hand.right.{cam}.mano_hand_pose"])
            right_betas = np.array(row[f"observation.hand.right.{cam}.mano_betas"])
            right = np.concatenate([right_orient, right_pose, right_betas])
            states.append(np.concatenate([left, right]))

        return torch.tensor(np.stack(states), dtype=torch.float32)

    def _load_mano_states(self, task_id, episode_idx, frame_indices):
        if self.config.use_dummy_mano_states:
            return self._load_mano_states_dummy(task_id, episode_idx, frame_indices)

        ep_df = self._load_episode_parquet(task_id, episode_idx)
        max_idx = len(ep_df) - 1
        cam  = self.config.human_camera
        mode = self.config.mano_state_mode

        need_xv     = bool(self.config.mano_cross_view_camera)
        need_interp = self.config.mano_interpolate_missing
        need_ema    = self.config.mano_ema_alpha > 0

        if not need_xv and not need_interp and not need_ema:
            clamped = [min(idx, max_idx) for idx in frame_indices]
            states = []
            for idx in clamped:
                row = ep_df.iloc[idx]
                hand_states = []
                for hand_type in ['left', 'right']:
                    prefix = f"observation.hand.{hand_type}.{cam}"
                    orient = np.array(row[f"{prefix}.mano_global_orient"]).flatten()
                    pose   = np.array(row[f"{prefix}.mano_hand_pose"]).flatten()
                    ct     = np.array(row[f"{prefix}.pred_cam_t_full"]).flatten()
                    jk     = f"{prefix}.pred_keypoints_3d"
                    j3d    = (np.array(row[jk]).flatten()
                              if jk in ep_df.columns else np.zeros(63))
                    hand_states.append(
                        mano_params_to_state_vector(orient, pose, ct, j3d, mode)
                    )
                states.append(np.concatenate(hand_states))
            return torch.tensor(np.stack(states), dtype=torch.float32)

        hand_params: Dict[str, Dict[str, np.ndarray]] = {}
        hand_valid:  Dict[str, List[bool]] = {}

        for ht in ['left', 'right']:
            params, valid = self._extract_raw_hand_params(ep_df, cam, ht)
            hand_params[ht] = params
            hand_valid[ht]  = valid

        if need_ema:
            for ht in ['left', 'right']:
                self._apply_ema(hand_params[ht], hand_valid[ht], self.config.mano_ema_alpha)

        if need_interp:
            for ht in ['left', 'right']:
                self._interpolate_cam_t(hand_params[ht], hand_valid[ht], ht)

        if need_xv:
            xv_cams = self.config.mano_cross_view_camera
            if isinstance(xv_cams, str):
                xv_cams = [xv_cams]
            for ht in ['left', 'right']:
                xv_sources = [
                    (*self._extract_raw_hand_params(ep_df, c, ht), c)
                    for c in xv_cams
                ]
                self._fill_pose_from_cross_view(
                    hand_params[ht], hand_valid[ht], xv_sources, cam, ht,
                )

        clamped = [min(idx, max_idx) for idx in frame_indices]
        states = []
        for idx in clamped:
            hand_states = []
            for ht in ['left', 'right']:
                p = hand_params[ht]
                hand_states.append(
                    mano_params_to_state_vector(
                        p['global_orient'][idx], p['hand_pose'][idx],
                        p['cam_t'][idx], p['joints_3d'][idx], mode,
                    )
                )
            states.append(np.concatenate(hand_states))

        return torch.tensor(np.stack(states), dtype=torch.float32)

    @staticmethod
    def _resolve_state_key(state_type: str, available_columns: List[str]) -> Optional[str]:
        _PREFERRED = {
            "qpos":   "observation.qpos_gripper_states",
            "eepos":  "observation.eepos_gripper_states",
            "mixpos": "observation.eepos_gripper_states",
        }
        _FALLBACK = {
            "qpos":   "observation.eepos_gripper_states",
            "eepos":  "observation.qpos_gripper_states",
            "mixpos": "observation.qpos_gripper_states",
        }
        preferred = _PREFERRED.get(state_type, "observation.qpos_gripper_states")
        if preferred in available_columns:
            return preferred
        fallback = _FALLBACK.get(state_type, preferred)
        if fallback in available_columns:
            return fallback
        for col in available_columns:
            if col.startswith("observation.") and col.endswith("_states"):
                return col
        return None

    def _load_target_states(self, task_id, episode_idx, frame_indices):
        repo_info = self.target_repo_info[task_id]
        repo_path = repo_info['repo_path']
        info = repo_info['info']

        data_path_format = info['data_path']
        episodes = self.target_repo_episodes[task_id]
        episode_row = episodes[episode_idx]

        chunk_idx = int(episode_row["data/chunk_index"])
        file_idx = int(episode_row["data/file_index"])

        parquet_path = repo_path / data_path_format.format(
            chunk_index=chunk_idx, file_index=file_idx,
        )

        df = self._read_parquet_cached(parquet_path)

        ep_index = int(episode_row['episode_index'])
        ep_df = df[df['episode_index'] == ep_index]

        max_idx = len(ep_df) - 1
        frame_indices = [min(idx, max_idx) for idx in frame_indices]

        state_key = self._resolve_state_key(self.config.state_type, list(ep_df.columns))
        if state_key is None:
            raise KeyError(
                f"[{self.config.target_domain}] task={task_id}: cannot find a "
                f"state column for state_type={self.config.state_type!r}. "
                f"Available columns: {[c for c in ep_df.columns if 'observation' in c]}"
            )

        states = []
        for idx in frame_indices:
            row = ep_df.iloc[idx]
            state = np.array(row[state_key]).flatten().astype(np.float32)

            if self.config.single_arm:
                M = len(state) // 2
                state = np.concatenate([state[:M - 1], state[2 * M - 2: 2 * M - 1]])

            states.append(state)

        return torch.tensor(np.stack(states), dtype=torch.float32)

    def _load_robot_states(self, task_id, episode_idx, frame_indices):
        """Deprecated alias → use _load_target_states()."""
        return self._load_target_states(task_id, episode_idx, frame_indices)

    def _apply_modality_dropout(self) -> Dict[str, bool]:
        if self.config.modality_dropout_prob <= 0:
            return {'human_video': True, 'human_states': True, 'human_desc': True}

        mask = {
            'human_video':  self.rng.random() >= self.config.modality_dropout_prob,
            'human_states': self.rng.random() >= self.config.modality_dropout_prob,
            'human_desc':   self.rng.random() >= self.config.modality_dropout_prob,
        }

        if not any(mask.values()):
            key = self.rng.choice(list(mask.keys()))
            mask[key] = True

        return mask

    def _print_summary(self):
        mode = self.config.mano_state_mode
        per_hand_dim = self.MANO_STATE_DIMS.get(mode, 48)

        target_state_col = {
            "qpos":   "observation.qpos_gripper_states",
            "eepos":  "observation.eepos_gripper_states",
            "mixpos": "observation.eepos_gripper_states",
        }.get(self.config.state_type, "observation.qpos_gripper_states")

        print("\n" + "=" * 80)
        print("DATASET READY")
        print("=" * 80)
        print(f"  Human tasks: {len(self.human_tasks)}")
        print(f"  Target domain: {self.config.target_domain.upper()}")
        print(f"  Target tasks ({self.config.target_domain}): {len(self.target_tasks)}")
        print(f"  Total pairs: {len(self.pair_index)}")
        print(f"  Pairing mode: {self.config.pairing_mode}")
        print(f"  Frames per sample: {self.config.num_frames}")
        print(f"  Image size: {self.config.image_size}")
        print(f"  Target cameras: {self.config.cameras}")
        print(f"  Human camera: {self.config.human_camera}")
        print(f"  Include first frame: {self.config.include_first_frame}")
        print(f"  State type: {self.config.state_type!r}  →  column: {target_state_col}")
        print(f"  Single arm: {self.config.single_arm}")
        print(f"  MANO state mode: {mode} ({per_hand_dim}D per hand, {per_hand_dim * 2}D total)")
        if self.config.mano_interpolate_missing:
            print(f"  MANO interpolation: cubic-spline (enabled)")
        if self.config.mano_ema_alpha > 0:
            print(f"  MANO EMA alpha: {self.config.mano_ema_alpha}")
        if self.config.mano_cross_view_camera:
            print(f"  MANO cross-view source: {self.config.mano_cross_view_camera}")
        if self.config.pre_decode:
            print(f"  Pre-decode: ENABLED  →  {self.config.pre_decode_cache_dir}")
        if self.config.target_domain == "robot":
            print(f"  [ROBOT] depth_mode: {self.config.depth_mode}")
        print("=" * 80 + "\n")

    def __len__(self) -> int:
        return len(self.pair_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pair_index[idx]

        human_task = pair['human_task']
        human_ep   = pair['human_episode']
        target_task = pair['target_task']
        target_ep   = pair['target_episode']

        human_info  = self.human_repo_info[human_task]
        target_info = self.target_repo_info[target_task]

        human_episodes  = self.human_repo_episodes[human_task]
        target_episodes = self.target_repo_episodes[target_task]

        human_row  = human_episodes[human_ep]
        target_row = target_episodes[target_ep]

        human_len  = int(human_row['length'])
        target_len = int(target_row['length'])

        target_camera = self.config.cameras[0]

        # ── Video loading (cache-aware) ───────────────────────────────────────
        # _load_video_frames_cached returns BOTH the tensor AND the frame indices
        # used, so states are always sampled at the same positions as the video.
        human_video, human_frames = self._load_video_frames_cached(
            human_task, human_ep, self.config.human_camera,
            human_info, human_row, human_len,
        )
        robot_video, target_frames = self._load_video_frames_cached(
            target_task, target_ep, target_camera,
            target_info, target_row, target_len,
        )

        # ── State loading (uses same frame indices as video) ──────────────────
        if not self.config.skip_states:
            human_states = self._load_mano_states(human_task, human_ep, human_frames)
            robot_states = self._load_target_states(target_task, target_ep, target_frames)

        # ── Descriptions ──────────────────────────────────────────────────────
        human_desc = self.task_mapper.get_description(human_task, 'human')
        robot_desc = self.task_mapper.get_description(target_task, self.config.target_domain)

        if not self.config.skip_states:
            sample = {
                'human_video':       human_video,
                'human_states':      human_states,
                'human_desc':        human_desc,
                'robot_video':       robot_video,
                'robot_states':      robot_states,
                'robot_desc':        robot_desc,
                'human_task_id':     human_task,
                'target_task_id':    target_task,
                'human_episode_idx': human_ep,
                'target_episode_idx': target_ep,
                'task_id':           self.target_task_to_idx[target_task],
                'level_id':          torch.tensor(extract_level_id(target_task), dtype=torch.long),
                'human_video_path':  f"{human_task}__ep{human_ep}",
            }
        else:
            sample = {
                'human_video':       human_video,
                'human_desc':        human_desc,
                'robot_video':       robot_video,
                'robot_desc':        robot_desc,
                'human_task_id':     human_task,
                'target_task_id':    target_task,
                'human_episode_idx': human_ep,
                'target_episode_idx': target_ep,
                'task_id':           self.target_task_to_idx[target_task],
                'level_id':          torch.tensor(extract_level_id(target_task), dtype=torch.long),
                'human_video_path':  f"{human_task}__ep{human_ep}",
            }

        # ── First frame ───────────────────────────────────────────────────────
        if self.config.include_first_frame:
            sample['robot_first_frame'] = self._load_first_frame_cached(
                target_task, target_ep, target_camera, target_info, target_row,
            )

        # ── Modality dropout ──────────────────────────────────────────────────
        if self.config.split == 'train':
            sample['modality_mask'] = self._apply_modality_dropout()

        return sample

    def get_statistics(self) -> Dict[str, Any]:
        mode = self.config.mano_state_mode
        per_hand_dim = self.MANO_STATE_DIMS.get(mode, 48)
        return {
            'num_human_tasks':         len(self.human_tasks),
            'num_target_tasks':        len(self.target_tasks),
            'num_pairs':               len(self.pair_index),
            'num_frames':              self.config.num_frames,
            'mano_state_mode':         mode,
            'mano_state_dim_per_hand': per_hand_dim,
            'mano_state_dim_total':    per_hand_dim * 2,
            'target_domain':           self.config.target_domain,
            'pairing_mode':            self.config.pairing_mode,
            'include_first_frame':     self.config.include_first_frame,
            'pre_decode':              self.config.pre_decode,
        }


def collate_task_pair_batch(batch: List[Dict]) -> Dict[str, Any]:
    result = {
        'human_video':    torch.stack([b['human_video'] for b in batch]),
        'human_states':   torch.stack([b['human_states'] for b in batch]),
        'human_desc':     [b['human_desc'] for b in batch],
        'robot_video':    torch.stack([b['robot_video'] for b in batch]),
        'robot_states':   torch.stack([b['robot_states'] for b in batch]),
        'robot_desc':     [b['robot_desc'] for b in batch],
        'human_task_id':  [b['human_task_id'] for b in batch],
        'target_task_id': [b['target_task_id'] for b in batch],
        'task_ids':       torch.tensor([b['task_id'] for b in batch], dtype=torch.long),
        'level_ids':      torch.stack([b['level_id'] for b in batch]),
        'human_video_path': [b['human_video_path'] for b in batch],
    }

    if 'robot_first_frame' in batch[0]:
        result['robot_first_frame'] = torch.stack([b['robot_first_frame'] for b in batch])

    if 'modality_mask' in batch[0]:
        result['modality_mask'] = batch[0]['modality_mask']

    return result


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    config = IntegratedDatasetConfig(
        human_root="demos/demo_data",
        sim_root="demos/imitator_data",
        task_mapping_file="examples/baselines/lerobot_dataset/task_mapping.json",
        human_dataset_file="examples/baselines/lerobot_dataset/config/human_train_config.json",
        sim_dataset_file="examples/baselines/lerobot_dataset/config/sim_train_config.json",
        human_task_description_file="examples/baselines/lerobot_dataset/task_desc/human_desc.json",
        sim_task_description_file="examples/baselines/lerobot_dataset/task_desc/sim_desc.json",
        target_domain="sim",
        split="train",
        pairing_mode="random_sample",
        max_pairs_per_mapping=10,
        include_first_frame=True,
        modality_dropout_prob=0.2,
        debug=True,
        pre_decode=True,
        pre_decode_cache_dir="tmp/episode_frame_cache",
        pre_decode_num_workers=8,
    )

    dataset = IntegratedTaskPairDataset(config)

    print("\nDataset Statistics:")
    for k, v in dataset.get_statistics().items():
        print(f"  {k}: {v}")

    print("\nTesting sample loading...")
    sample = dataset[0]

    print(f"\nSample contents:")
    print(f"  human_video: {sample['human_video'].shape}")
    if 'human_states' in sample:
        print(f"  human_states: {sample['human_states'].shape}")
    print(f"  human_desc: {sample['human_desc'][:50]}...")
    print(f"  robot_video: {sample['robot_video'].shape}")
    if 'robot_states' in sample:
        print(f"  robot_states: {sample['robot_states'].shape}")
    print(f"  robot_desc: {sample['robot_desc'][:50]}...")

    if 'robot_first_frame' in sample:
        print(f"  robot_first_frame: {sample['robot_first_frame'].shape}")

    if 'modality_mask' in sample:
        print(f"  modality_mask: {sample['modality_mask']}")


class RobotVideoOnlyDataset(Dataset):

    def __init__(self, base_dataset: "IntegratedTaskPairDataset"):
        self.base = base_dataset
        self.pair_index = base_dataset.pair_index
        self.config = base_dataset.config

    def __len__(self) -> int:
        return len(self.pair_index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        pair = self.pair_index[idx]
        target_task = pair["target_task"]
        target_ep   = pair["target_episode"]

        target_info     = self.base.target_repo_info[target_task]
        target_episodes = self.base.target_repo_episodes[target_task]
        target_row      = target_episodes[target_ep]
        target_len      = int(target_row["length"])

        target_camera = self.config.cameras[0]

        # Use cache-aware loader
        video, _ = self.base._load_video_frames_cached(
            target_task, target_ep, target_camera,
            target_info, target_row, target_len,
        )
        return video  # [T, H, W, C]


def collate_robot_video_only(batch: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)
