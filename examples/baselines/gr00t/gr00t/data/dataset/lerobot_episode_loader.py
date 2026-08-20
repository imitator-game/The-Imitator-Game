#!/usr/bin/env python
"""
LeRobot Dataset Loader

A simplified, clean implementation for loading LeRobot datasets with video support.
This module provides the core functionality for loading episodes from LeRobot format datasets,
handling metadata parsing, video decoding, and data preprocessing for VLA training.

The LeRobotEpisodeLoader serves as the foundation for higher-level dataset classes,
providing episode-level data access with support for multi-modal data including:
- Video frames from multiple camera views
- Proprioceptive state information
- Action sequences
- Language instructions/annotations

Returns messages with VLAStepData as defined in types.py.
"""

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from gr00t.data.types import ModalityConfig
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, load_initial_actions
from gr00t.utils.video_utils import get_frames_by_indices


# LeRobot standard metadata filenames
LEROBOT_META_DIR_NAME = "meta"
LEROBOT_INFO_FILENAME = "info.json"
LEROBOT_EPISODES_FILENAME = "episodes.jsonl"
LEROBOT_TASKS_FILENAME = "tasks.jsonl"
LEROBOT_MODALITY_FILENAME = "modality.json"
LEROBOT_STATS_FILE_NAME = "stats.json"
LEROBOT_RELATIVE_STATS_FILE_NAME = "relative_stats.json"
LEROBOT_V3_EPISODES_DIR = "episodes"
LEROBOT_V3_TASKS_FILENAME = "tasks.parquet"

ALLOWED_MODALITIES = ["video", "state", "action", "language", "mask"]
DEFAULT_COLUMN_NAMES = {
    "state": "observation.state",
    "action": "action",
}

LANG_KEYS = ["task", "sub_task"]


def _rec_defaultdict() -> defaultdict:
    """Factory that creates an infinitely nestable defaultdict."""
    return defaultdict(_rec_defaultdict)


def _to_plain_dict(tree):
    """Recursively turn a (nested) defaultdict into a regular dict."""
    if isinstance(tree, defaultdict):
        return {k: _to_plain_dict(v) for k, v in tree.items()}
    return tree


class LeRobotEpisodeLoader:
    """
    Episode-level data loader for LeRobot format datasets.

    This class handles the loading and preprocessing of individual episodes from LeRobot datasets.
    It manages metadata parsing, video decoding, and data extraction across multiple modalities
    (video, state, action, language) while maintaining compatibility with the VLA training pipeline.

    Key responsibilities:
    - Parse LeRobot metadata files (info.json, episodes.jsonl, etc.)
    - Load and decode video data using configurable backends
    - Extract and process multi-modal data according to modality configurations
    - Provide dataset statistics for normalization
    - Handle initial action loading for policy initialization

    Args:
        dataset_path: Path to dataset root directory containing meta/ and data files
        modality_configs: Dictionary mapping modality names to ModalityConfig objects
                         that specify temporal sampling and data keys to load
        video_backend: Video decoding backend ('torchcodec', 'decord', etc.)
        video_backend_kwargs: Additional arguments for the video backend

    Example:
        >>> loader = LeRobotEpisodeLoader(
        ...     dataset_path="/path/to/lerobot_dataset",
        ...     modality_configs={
        ...         "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        ...         "state": ModalityConfig(delta_indices=[0], modality_keys=["joint_positions"]),
        ...         "action": ModalityConfig(
        ...             delta_indices=list(range(16)), modality_keys=["joint_velocities"]
        ...         ),
        ...     },
        ... )
        >>> episode_data = loader[0]  # Load first episode as DataFrame
    """

    def __init__(
        self,
        dataset_path: str | Path,
        modality_configs: dict[str, ModalityConfig],
        lerobot_version: str = "auto",
        language_source: str = "task",
        task_mapping_path: str | Path | None = None,
        human_desc_path: str | Path | None = None,
        sim_desc_path: str | Path | None = None,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict[str, Any] | None = None,
        episode_indices: list[int] | None = None,
    ) -> None:
        """
        Initialize LeRobot episode loader with dataset path and modality configurations.

        The initialization process involves:
        1. Loading all metadata files from the dataset
        2. Parsing and validating modality configurations
        3. Computing effective episode lengths based on action horizon
        """
        self.dataset_path = Path(dataset_path)
        self.requested_lerobot_version = str(lerobot_version).lower()
        self.language_source = str(language_source).lower()
        self.task_mapping_path = Path(task_mapping_path) if task_mapping_path else None
        self.human_desc_path = Path(human_desc_path) if human_desc_path else None
        self.sim_desc_path = Path(sim_desc_path) if sim_desc_path else None
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs
        self.requested_episode_indices = episode_indices

        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        # Load metadata files and parse dataset structure
        self._load_metadata()
        self._filter_episode_metadata()
        self._load_language_metadata()

        # Set up modality configs after metadata is loaded
        self.modality_configs = self._parse_and_validate_modality_configs(modality_configs)

        # Compute effective episode lengths accounting for action horizon
        self.episode_lengths = self.get_episode_lengths()

    def _filter_episode_metadata(self) -> None:
        if self.requested_episode_indices is None:
            return

        requested = {int(idx) for idx in self.requested_episode_indices}
        filtered = [
            rec
            for rec in self.episodes_metadata
            if int(rec.get("episode_index", -1)) in requested
        ]
        if not filtered:
            raise ValueError(
                f"No requested episodes {sorted(requested)} found in {self.dataset_path}"
            )
        missing = requested - {int(rec["episode_index"]) for rec in filtered}
        if missing:
            raise ValueError(
                f"Requested episodes not found in {self.dataset_path}: {sorted(missing)}"
            )
        self.episodes_metadata = filtered

    def _default_lerobot_dataset_dir(self) -> Path:
        return Path(__file__).resolve().parents[4] / "lerobot_dataset"

    def _load_json_if_exists(self, path: Path | None) -> dict[str, Any]:
        if path is None or not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _load_language_metadata(self) -> None:
        if self.language_source not in {"task", "human_desc", "sim_desc"}:
            raise ValueError(
                f"Unsupported language_source={self.language_source}. "
                "Expected one of: task, human_desc, sim_desc."
            )

        default_dir = self._default_lerobot_dataset_dir()
        task_mapping_path = self.task_mapping_path or (default_dir / "task_mapping.json")
        human_desc_path = self.human_desc_path or (default_dir / "task_desc" / "human_desc.json")
        sim_desc_path = self.sim_desc_path or (default_dir / "task_desc" / "sim_desc.json")

        self.task_to_human_task_map: dict[str, str] = {}
        self.human_desc_map: dict[str, list[str]] = {}
        self.sim_desc_map: dict[str, list[str]] = {}

        if self.language_source == "task":
            return

        task_mapping_data = self._load_json_if_exists(task_mapping_path)
        for mapping in task_mapping_data.get("task_mappings", []):
            human_task_id = mapping.get("human_task_id")
            if not human_task_id:
                continue
            for sim_task_id in mapping.get("sim_task_id", []):
                self.task_to_human_task_map[str(sim_task_id)] = str(human_task_id)
            for robot_task_id in mapping.get("robot_task_id", []):
                self.task_to_human_task_map[str(robot_task_id)] = str(human_task_id)

        human_desc_data = self._load_json_if_exists(human_desc_path)
        for task_id, descs in human_desc_data.items():
            if isinstance(descs, list):
                self.human_desc_map[str(task_id)] = [str(desc) for desc in descs if str(desc)]

        sim_desc_data = self._load_json_if_exists(sim_desc_path)
        for task_id, descs in sim_desc_data.items():
            if isinstance(descs, list):
                self.sim_desc_map[str(task_id)] = [str(desc) for desc in descs if str(desc)]

    def _load_metadata(self) -> None:
        """
        Load all metadata files including dataset statistics.

        Parses the standard LeRobot metadata structure:
        - info.json: Dataset configuration and file patterns
        - episodes.jsonl: Per-episode metadata (length, timestamps, etc.)
        - tasks.jsonl: Task descriptions and mappings
        - modality.json: Modality structure and data layout
        - stats.json: Dataset statistics for normalization
        """
        meta_dir = self.dataset_path / LEROBOT_META_DIR_NAME

        if self.requested_lerobot_version not in {"auto", "v2", "v3"}:
            raise ValueError(
                f"Unsupported lerobot_version={self.requested_lerobot_version}. "
                "Expected one of: auto, v2, v3."
            )

        # Load dataset configuration
        info_path = meta_dir / LEROBOT_INFO_FILENAME
        with open(info_path, "r") as f:
            self.info_meta = json.load(f)

        self.lerobot_version = self._resolve_lerobot_version(meta_dir)
        if self.lerobot_version == "v3":
            self._load_metadata_v3(meta_dir)
        else:
            self._load_metadata_v2(meta_dir)

        # Load dataset statistics for normalization
        stats_path = meta_dir / LEROBOT_STATS_FILE_NAME
        assert stats_path.exists(), (
            f"{stats_path} does not exist for {self.dataset_path}, please use gr00t/data/stats.py to generate it"
        )
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        relative_stats_path = meta_dir / LEROBOT_RELATIVE_STATS_FILE_NAME
        if relative_stats_path.exists():
            with open(relative_stats_path, "r") as f:
                self.stats["relative_action"] = json.load(f)

        # Extract key configuration parameters
        self.feature_config = self.info_meta.get("features", {})
        self.data_path_pattern = self.info_meta["data_path"]
        self.video_path_pattern = self.info_meta.get("video_path")
        self.mask_path_pattern = self.info_meta.get("mask_path")
        self.chunk_size = self.info_meta["chunks_size"]
        self.fps = self.info_meta.get("fps", 30)

    def _resolve_lerobot_version(self, meta_dir: Path) -> str:
        if self.requested_lerobot_version != "auto":
            return self.requested_lerobot_version

        version = str(self.info_meta.get("codebase_version", "")).lower()
        if version.startswith("v3"):
            return "v3"
        if version.startswith("v2"):
            return "v2"

        if (meta_dir / LEROBOT_EPISODES_FILENAME).exists():
            return "v2"
        if (meta_dir / LEROBOT_V3_EPISODES_DIR).exists():
            return "v3"
        return "v2"

    def _load_metadata_v2(self, meta_dir: Path) -> None:
        # Load episode metadata (one episode per line)
        episodes_path = meta_dir / LEROBOT_EPISODES_FILENAME
        with open(episodes_path, "r") as f:
            self.episodes_metadata = [json.loads(line) for line in f]

        # Load task descriptions and create mapping
        tasks_path = meta_dir / LEROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks_data = [json.loads(line) for line in f]
            self.tasks_map = {task["task_index"]: task["task"] for task in tasks_data}

        # Load modality structure information
        modality_path = meta_dir / LEROBOT_MODALITY_FILENAME
        with open(modality_path, "r") as f:
            self.modality_meta = json.load(f)

        self._data_file_offsets = {}

    def _load_metadata_v3(self, meta_dir: Path) -> None:
        episodes_dir = meta_dir / LEROBOT_V3_EPISODES_DIR
        episode_tables = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
        if not episode_tables:
            raise FileNotFoundError(f"No v3 episode parquet files found under {episodes_dir}")

        self.episodes_metadata = []
        for path in episode_tables:
            table = pd.read_parquet(path)
            self.episodes_metadata.extend(table.to_dict(orient="records"))
        self.episodes_metadata.sort(key=lambda rec: int(rec["episode_index"]))

        tasks_path = meta_dir / LEROBOT_V3_TASKS_FILENAME
        tasks_df = pd.read_parquet(tasks_path)
        self.tasks_map = {}
        for idx, row in tasks_df.iterrows():
            task_name = str(idx)
            self.tasks_map[int(row["task_index"])] = task_name

        self.modality_meta = self._build_v3_modality_meta()

        self._data_file_offsets = {}
        for rec in self.episodes_metadata:
            key = (int(rec["data/chunk_index"]), int(rec["data/file_index"]))
            start = int(rec["dataset_from_index"])
            prev = self._data_file_offsets.get(key)
            self._data_file_offsets[key] = start if prev is None else min(prev, start)

    def _build_v3_modality_meta(self) -> dict[str, dict[str, dict[str, Any]]]:
        modality_meta: dict[str, dict[str, dict[str, Any]]] = {
            "video": {},
            "state": {},
            "action": {},
            "annotation": {},
            "mask": {},
        }

        for feature_name, feature_info in self.info_meta.get("features", {}).items():
            dtype = str(feature_info.get("dtype", ""))
            shape = feature_info.get("shape") or []
            names = feature_info.get("names") or []

            if feature_name.startswith("observation.images.") and dtype == "video":
                view_key = feature_name.removeprefix("observation.images.")
                modality_meta["video"][view_key] = {"original_key": feature_name}
                modality_meta["video"][feature_name] = {"original_key": feature_name}
                continue

            if feature_name.startswith("observation.") and "images" not in feature_name:
                key = feature_name.removeprefix("observation.")
                dim = int(shape[0]) if shape else 1
                entry = {"original_key": feature_name, "start": 0, "end": dim}
                modality_meta["state"][key] = entry
                modality_meta["state"][feature_name] = entry
                self._register_v3_named_groups(
                    modality_meta["state"],
                    feature_name,
                    names,
                )
                continue

            if feature_name.startswith("action."):
                key = feature_name.removeprefix("action.")
                dim = int(shape[0]) if shape else 1
                entry = {"original_key": feature_name, "start": 0, "end": dim}
                modality_meta["action"][key] = entry
                modality_meta["action"][feature_name] = entry
                self._register_v3_named_groups(
                    modality_meta["action"],
                    feature_name,
                    names,
                )

        return modality_meta

    def _register_v3_named_groups(
        self,
        modality_bucket: dict[str, dict[str, Any]],
        feature_name: str,
        names: list[str],
    ) -> None:
        if not names:
            return

        named_groups = {
            "left_arm": self._find_named_slice(names, lambda name: name.startswith("left_joint_")),
            "left_gripper": self._find_named_slice(
                names, lambda name: name.startswith("left_gripper")
            ),
            "right_arm": self._find_named_slice(names, lambda name: name.startswith("right_joint_")),
            "right_gripper": self._find_named_slice(
                names, lambda name: name.startswith("right_gripper")
            ),
        }

        for group_name, group_slice in named_groups.items():
            if group_slice is None:
                continue
            modality_bucket[group_name] = {
                "original_key": feature_name,
                "start": group_slice[0],
                "end": group_slice[1],
            }

    def _find_named_slice(
        self,
        names: list[str],
        predicate,
    ) -> tuple[int, int] | None:
        matching_indices = [idx for idx, name in enumerate(names) if predicate(str(name))]
        if not matching_indices:
            return None
        return min(matching_indices), max(matching_indices) + 1

    def get_episode_lengths(self):
        """
        Compute original episode lengths.

        Returns:
            List of original episode lengths
        """
        episode_lengths = []
        for ep_meta in self.episodes_metadata:
            episode_lengths.append(int(ep_meta["length"]))
        return episode_lengths

    def get_episode_length(self, idx: int) -> int:
        """Get the length of a specific episode."""
        return self.episode_lengths[idx]

    def _parse_and_validate_modality_configs(
        self,
        modality_configs: dict[str, ModalityConfig],
    ) -> dict[str, ModalityConfig]:
        """
        Parse and validate modality configurations, filling in defaults where needed.

        For missing modality configs, creates default configurations:
        - video: All available camera views with single timestep
        - state: All available state keys with single timestep
        - action: All available action keys with 16-step horizon
        - language: Must be explicitly configured if needed

        Args:
            modality_configs: User-provided modality configurations

        Returns:
            Complete and validated modality configurations

        Raises:
            ValueError: If invalid modalities are specified
            AssertionError: If language modality configuration is invalid
        """
        # Validate all modality configurations
        for modality in modality_configs:
            if modality not in ALLOWED_MODALITIES:
                raise ValueError(f"Invalid modality: {modality}")
            if modality == "language":
                # Language modality has special constraints
                assert len(modality_configs[modality].modality_keys) == 1, (
                    "Language modality must have exactly one key"
                )
                assert modality_configs[modality].delta_indices == [0], (
                    "Only single timestep is supported for language modality"
                )

        # Validate video modality_keys against modality.json.
        # Each key in modality_configs["video"].modality_keys must exist in
        # modality.json["video"], otherwise _load_video_data will fail with
        # a confusing KeyError when trying to resolve the original video key.
        if "video" in modality_configs and "video" in self.modality_meta:
            config_keys = set(modality_configs["video"].modality_keys)
            meta_keys = set(self.modality_meta["video"].keys())
            missing_keys = config_keys - meta_keys
            if missing_keys:
                raise ValueError(
                    f"Video modality_keys {sorted(missing_keys)} in modality_config "
                    f"not found in modality.json. "
                    f"modality_config expects: {sorted(config_keys)}, "
                    f"modality.json defines: {sorted(meta_keys)}. "
                    f"Please ensure modality.json and your modality_config use the "
                    f"same video key names."
                )

        return modality_configs

    def __len__(self) -> int:
        """Return number of episodes in dataset."""
        return len(self.episodes_metadata)

    def _extract_joint_groups(
        self,
        df: pd.DataFrame,
        joint_groups: list[str],
        modality_type: str = "state",
    ) -> pd.DataFrame:
        """
        Extract specific joint groups from data arrays based on modality metadata.

        Uses the modality metadata to slice the appropriate indices from the raw data arrays,
        allowing for flexible joint group extraction (e.g., arm joints, gripper state).

        Args:
            df: DataFrame containing the raw episode data
            joint_groups: List of joint group names to extract (e.g., ["arm", "gripper"])
            modality_type: Type of modality ("state" or "action")

        Returns:
            DataFrame with columns for each requested joint group containing sliced arrays
        """
        modality_info = self.modality_meta.get(modality_type, {})
        joint_data = pd.DataFrame()

        for group_name in joint_groups:
            if group_name in modality_info:
                group_info = modality_info[group_name]
                start_idx = group_info["start"]
                end_idx = group_info["end"]
                original_key = group_info.get("original_key", DEFAULT_COLUMN_NAMES[modality_type])
                # Slice the array data for this joint group
                if isinstance(df[original_key].iloc[0], np.ndarray):
                    joint_data[group_name] = df[original_key].map(lambda x: x[start_idx:end_idx])
                else:
                    joint_data[group_name] = df[original_key]  # for strings and scalars
            else:
                print(
                    f"Warning: Joint group '{group_name}' not found in {modality_type} modality. Available groups: {list(modality_info.keys())}"
                )

        return joint_data

    def _load_parquet_data(self, episode_index: int) -> pd.DataFrame:
        """
        Load and process parquet data for a specific episode.

        Handles the complete data loading pipeline:
        1. Load raw parquet file based on chunking structure
        2. Process language annotations (convert task indices to strings)
        3. Extract state and action joint groups

        Args:
            episode_index: Index of the episode to load

        Returns:
            Processed DataFrame with all modality data
        """
        # Load raw parquet data using the appropriate layout.
        episode_meta = self.episodes_metadata[episode_index]
        episode_id = int(episode_meta["episode_index"])

        if self.lerobot_version == "v3":
            data_chunk = int(episode_meta["data/chunk_index"])
            data_file = int(episode_meta["data/file_index"])
            parquet_filename = self.data_path_pattern.format(
                chunk_index=data_chunk,
                file_index=data_file,
            )
            parquet_path = self.dataset_path / parquet_filename
            original_df = pd.read_parquet(parquet_path)
            file_offset = self._data_file_offsets[(data_chunk, data_file)]
            start = int(episode_meta["dataset_from_index"]) - file_offset
            stop = int(episode_meta["dataset_to_index"]) - file_offset
            original_df = original_df.iloc[start:stop].reset_index(drop=True)
        else:
            chunk_idx = episode_id // self.chunk_size
            parquet_filename = self.data_path_pattern.format(
                episode_chunk=chunk_idx, episode_index=episode_id
            )
            parquet_path = self.dataset_path / parquet_filename
            original_df = pd.read_parquet(parquet_path)
        loaded_df = pd.DataFrame()

        # Process language annotations (convert task indices to task strings)
        if "language" in self.modality_configs:
            for key in self.modality_configs["language"].modality_keys:
                # these keys will be loaded separately from episodes.jsonl
                if key in LANG_KEYS:
                    continue
                assert key.startswith("annotation.")
                subkey = key.replace("annotation.", "")
                assert subkey in self.modality_meta["annotation"], (
                    f"Key {subkey} not found in language modality"
                )
                original_key = self.modality_meta["annotation"][subkey].get("original_key", key)
                loaded_df[f"language.{key}"] = original_df[original_key].apply(
                    lambda x: self.tasks_map[x]
                )

        # Extract joint groups for state and action modalities
        for modality_type in ["state", "action"]:
            if modality_type not in self.modality_configs:
                continue
            joint_groups_df = self._extract_joint_groups(
                original_df,
                self.modality_configs[modality_type].modality_keys,
                modality_type,
            )
            for joint_group in joint_groups_df.columns:
                loaded_df[f"{modality_type}.{joint_group}"] = joint_groups_df[joint_group]

        return loaded_df

    def _load_video_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        """
        Load video data for all configured camera views at specified indices.

        Uses the configured video backend to decode video frames at the exact indices
        needed for the episode, supporting multiple camera views simultaneously.

        Args:
            episode_index: Index of the episode to load videos for
            indices: Array of indices to extract frames at

        Returns:
            Dictionary mapping camera view names to arrays of decoded frames
        """
        video_data = {}

        if not self.video_path_pattern or "video" not in self.modality_configs:
            return video_data

        episode_meta = self.episodes_metadata[episode_index]
        episode_id = int(episode_meta["episode_index"])
        chunk_idx = episode_id // self.chunk_size
        image_keys = self.modality_configs["video"].modality_keys

        for image_key in image_keys:
            # Resolve the original key used in video file naming
            original_key = self.modality_meta["video"][image_key].get(
                "original_key", f"observation.images.{image_key}"
            )
            assert original_key in self.feature_config, (
                f"Original key {original_key} not found in feature config"
            )

            # Construct video file path using pattern
            if self.lerobot_version == "v3":
                video_chunk = int(episode_meta[f"videos/{original_key}/chunk_index"])
                video_file = int(episode_meta[f"videos/{original_key}/file_index"])
                video_filename = self.video_path_pattern.format(
                    video_key=original_key,
                    chunk_index=video_chunk,
                    file_index=video_file,
                )
                start_frame = int(round(float(episode_meta[f"videos/{original_key}/from_timestamp"]) * self.fps))
                frame_indices = indices + start_frame
            else:
                video_filename = self.video_path_pattern.format(
                    episode_chunk=chunk_idx,
                    video_key=original_key,
                    episode_index=episode_id,
                )
                frame_indices = indices
            video_path = self.dataset_path / video_filename

            # Decode video frames at specified timestamps
            video_data[image_key] = get_frames_by_indices(
                str(video_path),
                frame_indices,
                video_backend=self.video_backend,
                video_backend_kwargs=self.video_backend_kwargs or {},
            )

        return video_data

    def _load_mask_file(self, mask_path: Path, indices: np.ndarray) -> np.ndarray:
        """Load masks from npz/npy file at specified indices."""
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")
        suffix = mask_path.suffix.lower()
        if suffix not in {".npz", ".npy"}:
            raise ValueError(f"Only .npz or .npy mask files are supported: {mask_path}")

        if suffix == ".npy":
            masks = np.load(mask_path)
        else:
            npz_data = np.load(mask_path)
            if "arr_0" in npz_data:
                masks = npz_data["arr_0"]
            elif len(npz_data.files) == 1:
                masks = npz_data[npz_data.files[0]]
            else:
                raise ValueError(f"Mask npz must contain a single array or 'arr_0': {mask_path}")

        if masks.ndim == 2:
            masks = masks[None, ...]

        return masks[indices]

    def _load_mask_data(self, episode_index: int, indices: np.ndarray) -> dict[str, np.ndarray]:
        """
        Load mask data for all configured mask views at specified indices.
        """
        mask_data = {}

        if not self.mask_path_pattern or "mask" not in self.modality_configs:
            return mask_data

        chunk_idx = episode_index // self.chunk_size
        mask_keys = self.modality_configs["mask"].modality_keys

        for mask_key in mask_keys:
            mask_meta = self.modality_meta.get("mask", {}).get(mask_key, {})
            original_key = mask_meta.get("original_key", mask_key)
            mask_filename = self.mask_path_pattern.format(
                episode_chunk=chunk_idx,
                episode_index=episode_index,
                mask_key=original_key,
                video_key=original_key,
            )
            mask_path = self.dataset_path / mask_filename
            mask_data[mask_key] = self._load_mask_file(mask_path, indices)

        return mask_data

    def get_dataset_statistics(self) -> dict[str, Any]:
        """
        Extract dataset statistics for normalization from loaded metadata.

        Constructs a nested dictionary containing statistics (mean, std, min, max, q01, q99)
        for each joint group in state and action modalities. These statistics are used
        by processors for data normalization during training.

        Returns:
            Nested dictionary: {modality: {joint_group: {stat_type: values}}}
        """
        mapping = {"state": "observation.state", "action": "action"}
        dataset_statistics = _rec_defaultdict()

        for modality in mapping.keys():  # state, action
            for joint_key in self.modality_configs[modality].modality_keys:
                # Determine which statistics key to use
                if self.modality_meta[modality][joint_key].get("original_key", None) is not None:
                    stats_key = self.modality_meta[modality][joint_key]["original_key"]
                else:
                    stats_key = mapping[modality]

                # Extract the relevant slice of statistics
                start_idx, end_idx = (
                    self.modality_meta[modality][joint_key]["start"],
                    self.modality_meta[modality][joint_key]["end"],
                )
                for stat_type in self.stats[stats_key].keys():  # mean, std, min, max, q01, q99
                    dataset_statistics[modality][joint_key][stat_type] = self.stats[stats_key][
                        stat_type
                    ][start_idx:end_idx]
        stats = _to_plain_dict(dataset_statistics)
        # Directly add relative action stats
        if "relative_action" in self.stats:
            stats["relative_action"] = self.stats["relative_action"]
        return stats

    def create_language_from_meta(
        self, episode_meta: dict, nframes: int, lang_key: str
    ) -> list[str]:
        if lang_key == "task":
            meta_language = self._sample_task_language(episode_meta)
            new_languages = [meta_language] * nframes
        elif lang_key == "sub_task":
            action_delta_indices = self.modality_configs["action"].delta_indices
            action_horizon = max(action_delta_indices) - min(action_delta_indices) + 1
            new_languages = [[] for _ in range(nframes)]
            sub_tasks = episode_meta["sub_tasks"]
            for sub_task in sub_tasks:
                start_idx, end_idx, sub_text = (
                    sub_task["start"],
                    sub_task["end"],
                    sub_task["text"],
                )
                horizon = action_horizon // 2
                for i in range(start_idx - horizon, end_idx):
                    if i < 0:
                        continue
                    new_languages[i].append(sub_text)
            new_languages = [i if len(i) > 0 else [""] for i in new_languages]
            new_languages = [random.choice(i) for i in new_languages]
        else:
            raise ValueError(f"Language key {lang_key} not supported")
        return new_languages

    def _sample_task_language(self, episode_meta: dict) -> str:
        task_candidates = [str(task) for task in episode_meta.get("tasks", []) if str(task)]
        dataset_task_id = self.dataset_path.name
        if dataset_task_id and dataset_task_id not in task_candidates:
            task_candidates.append(dataset_task_id)
        fallback_task = task_candidates[0] if task_candidates else ""

        if self.language_source == "task":
            return random.choice(task_candidates) if task_candidates else ""

        if self.language_source == "sim_desc":
            for task_id in task_candidates:
                sim_descs = self.sim_desc_map.get(task_id, [])
                if sim_descs:
                    return random.choice(sim_descs)
            return random.choice(task_candidates) if task_candidates else ""

        for task_id in task_candidates:
            human_task_id = self.task_to_human_task_map.get(task_id)
            if not human_task_id:
                continue
            human_descs = self.human_desc_map.get(human_task_id, [])
            if human_descs:
                return random.choice(human_descs)

        for task_id in task_candidates:
            sim_descs = self.sim_desc_map.get(task_id, [])
            if sim_descs:
                return random.choice(sim_descs)

        return fallback_task

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """
        Load complete episode data as a processed DataFrame.

        Combines parquet data loading and video decoding to create a unified DataFrame
        containing all modality data for the episode. Video frames are converted to
        PIL Images and stored in the DataFrame.

        Args:
            idx: Episode index to load

        Returns:
            DataFrame with columns for all modalities and timestamps, with video frames
            as PIL Images ready for further processing

        Raises:
            IndexError: If episode index is out of bounds
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Episode index {idx} out of bounds")

        episode_meta = self.episodes_metadata[idx]
        episode_id = int(episode_meta["episode_index"])
        nominal_length = episode_meta["length"]

        # Load and parse the parquet data
        df = self._load_parquet_data(idx)

        if "language" in self.modality_configs:
            lang_key = self.modality_configs["language"].modality_keys[0]
            if lang_key in LANG_KEYS:
                new_languages = self.create_language_from_meta(episode_meta, len(df), lang_key)
                df["language." + lang_key] = new_languages

        # Use actual dataframe length (might be less than nominal)
        actual_length = min(len(df), nominal_length)
        df = df.iloc[:actual_length]

        # Load synchronized video data
        video_data = self._load_video_data(idx, np.arange(actual_length))

        # Add video frames to dataframe as PIL Images
        for key in video_data.keys():
            assert len(video_data[key]) == len(df), (
                f"Video data for {key} has length {len(video_data[key])} but dataframe has length {len(df)}"
            )
            df[f"video.{key}"] = [frame for frame in video_data[key]]

        # Load synchronized mask data
        mask_data = self._load_mask_data(episode_id, np.arange(actual_length))
        for key in mask_data.keys():
            assert len(mask_data[key]) == len(df), (
                f"Mask data for {key} has length {len(mask_data[key])} but dataframe has length {len(df)}"
            )
            df[f"mask.{key}"] = [mask for mask in mask_data[key]]

        return df

    def get_initial_actions(self):
        """
        Load initial actions for policy initialization if available.

        Returns:
            List containing initial action dictionaries, or empty list if not available
        """
        meta_dirpath = self.dataset_path / LEROBOT_META_DIR_NAME
        initial_actions_path = meta_dirpath / INITIAL_ACTIONS_FILENAME
        if initial_actions_path.exists():
            initial_actions = load_initial_actions(initial_actions_path)
            return initial_actions  # a single-element list of dict[str, dict[str, np.ndarray]]
        else:
            return []
