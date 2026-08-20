"""
Human-Sim-Robot Paired Dataset
===============================
- Supports first frame observation for task conditioning
- Supports three input modes: video_only, language_only, video_and_language
- skip_human_video mode: when the L3 TaskEmbeddingCache is ready, the dataset
  can skip video decoding entirely — only human_repo_id is returned.
  This eliminates the dominant per-sample I/O cost during training.

Bug fixes vs original:
  HumanRobotPairedDataset._build_and_validate_pairing no longer calls
  __getitem__ for every sample (O(N) full image-decode scan).  Instead it
  uses the sub-dataset metadata (repo_id, cumulative_sizes) — same approach
  as HumanSimPairedDataset.
"""

import json
import random
import tempfile
import os
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from examples.baselines.lerobot_dataset.lerobot_dataloader import (
    LeRobotDataConfig,
    build_lerobot_dataset,
)


class InputMode(Enum):
    """Input mode for task conditioning."""
    VIDEO_ONLY         = "video_only"
    LANGUAGE_ONLY      = "language_only"
    VIDEO_AND_LANGUAGE = "video_and_language"


@dataclass
class PairedDatasetConfig:
    """Configuration for paired human-sim/robot dataset."""

    # Data paths
    human_root: str = "demos"
    robot_root: str = "demos"
    sim_root:   str = "demos"
    task_mapping_file: str = "examples/baselines/lerobot_dataset/task_mapping.json"

    # Dataset configs
    human_dataset_file: Optional[str] = "examples/baselines/lerobot_dataset/config/human_config.json"
    robot_dataset_file: Optional[str] = "examples/baselines/lerobot_dataset/config/robot_config.json"
    sim_dataset_file:   Optional[str] = "examples/baselines/lerobot_dataset/config/sim_config.json"

    # Description files
    human_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    robot_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/robot_desc.json"
    sim_task_description_file:   Optional[str] = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"

    # Common settings
    split:         str  = "train"
    cameras:       List[str] = None
    include_depth: bool = True
    image_size:    Tuple[int, int] = (224, 224)

    # Human video settings
    num_frames:        int = 10
    sampling_strategy: str = "uniform_jitter"
    video_backend:     str = "torchcodec"

    # Robot/Sim trajectory settings
    horizon:     int = 16
    obs_horizon: int = 1
    state_type:  str = "qpos"
    single_arm:  bool = False

    # Training settings
    fps:   int  = 30
    debug: bool = False

    # Input mode
    input_mode: str = "video_only"  # video_only | language_only | video_and_language

    # First frame observation
    include_first_frame: bool = True

    enable_augmentation: bool = True

    # Don't Use it!
    pre_decode: bool = False
    pre_decode_cache_dir: str = "tmp/human_video_cache"
    pre_decode_num_workers: int = 16

    # For Skill Model Training
    skill: bool = False
    xskill: bool = False
    robot_frame_gap: int = 35

    def __post_init__(self):
        if self.cameras is None:
            self.cameras = ["zed2i"]


# ---------------------------------------------------------------------------
# TaskMapper
# ---------------------------------------------------------------------------

class TaskMapper:
    """Maps human tasks to robot/sim tasks using task_mappings.json."""

    def __init__(
        self,
        mapping_file: str,
        human_task_description_file: Optional[str] = None,
        robot_task_description_file: Optional[str] = None,
        sim_task_description_file:   Optional[str] = None,
    ):
        with open(mapping_file, "r") as f:
            data = json.load(f)

        self.task_mappings  = data.get("task_mappings", [])
        self.human_to_robots: dict = {}
        self.human_to_sims:   dict = {}
        self.robot_to_human:  dict = {}
        self.sim_to_human:    dict = {}

        for mapping in self.task_mappings:
            h_id  = mapping["human_task_id"]
            r_ids = mapping["robot_task_id"]
            s_ids = mapping["sim_task_id"]

            self.human_to_robots[h_id] = r_ids
            self.human_to_sims[h_id]   = s_ids
            for r_id in r_ids:
                self.robot_to_human[r_id] = h_id
            for s_id in s_ids:
                self.sim_to_human[s_id] = h_id

        self.human_descriptions: dict = {}
        self.robot_descriptions: dict = {}
        self.sim_descriptions:   dict = {}

        def _load_desc(path):
            if path and os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
            return {}

        self.human_descriptions = _load_desc(human_task_description_file)
        self.robot_descriptions  = _load_desc(robot_task_description_file)
        self.sim_descriptions    = _load_desc(sim_task_description_file)

        if self.human_descriptions:
            print(f"   ✓ {len(self.human_descriptions)} human task descriptions")
        if self.robot_descriptions:
            print(f"   ✓ {len(self.robot_descriptions)} robot task descriptions")
        if self.sim_descriptions:
            print(f"   ✓ {len(self.sim_descriptions)} sim task descriptions")

    def get_robot_tasks(self, human_task_id: str) -> List[str]:
        return self.human_to_robots.get(human_task_id, [])

    def get_sim_tasks(self, human_task_id: str) -> List[str]:
        return self.human_to_sims.get(human_task_id, [])

    def get_human_task_from_robot(self, robot_task_id: str) -> Optional[str]:
        return self.robot_to_human.get(robot_task_id)

    def get_human_task_from_sim(self, sim_task_id: str) -> Optional[str]:
        return self.sim_to_human.get(sim_task_id)

    def get_all_human_task_ids(self) -> Set[str]:
        return set(self.human_to_robots.keys())

    def get_all_robot_task_ids(self) -> Set[str]:
        return set(self.robot_to_human.keys())

    def get_all_sim_task_ids(self) -> Set[str]:
        return set(self.sim_to_human.keys())

    def get_description(self, task_type: str, task_id: str) -> Optional[str]:
        if task_type == "human":
            desc = self.human_descriptions.get(task_id)
            if isinstance(desc, list) and desc:
                return desc[random.randint(0, len(desc) - 1)]
            return desc
        elif task_type == "robot":
            return self.robot_descriptions.get(task_id)
        elif task_type == "sim":
            return self.sim_descriptions.get(task_id)
        return None


def get_human_video_id(sim_key: str, task_mappings: List[Dict]) -> Optional[str]:
    for task in task_mappings:
        if sim_key in task["sim_task_id"]:
            return task["human_task_id"]
        for sim_id in task["sim_task_id"]:
            if sim_id.endswith(sim_key) or sim_key in sim_id:
                return task["human_task_id"]
    return None


def filter_dataset_config(
    config_file: str,
    valid_task_ids: Set[str],
    dataset_type: str,
    split: str,
) -> List[Dict]:
    with open(config_file, "r") as f:
        all_configs = json.load(f)

    filtered, skipped, loaded = [], [], []
    for cfg in all_configs:
        repo_id   = cfg.get("repo_id", cfg.get("task_id", ""))
        cfg_split = cfg.get("split", "train")
        if cfg_split != split:
            continue
        matched = any(
            repo_id == tid or repo_id.endswith(tid) or tid in repo_id
            for tid in valid_task_ids
        )
        if matched:
            filtered.append(cfg)
            loaded.append(repo_id)
        else:
            skipped.append(repo_id)

    print(f"\n✓ Filtered {dataset_type} configs: {len(filtered)} loaded, "
          f"{len(skipped)} skipped")
    return filtered


# ---------------------------------------------------------------------------
# HumanSimPairedDataset
# ---------------------------------------------------------------------------

class HumanSimPairedDataset(Dataset):
    """Paired dataset combining human videos with sim trajectories.

    skip_human_video mode
    ---------------------
    When ``self.skip_human_video = True``, __getitem__ skips the call to
    ``human_dataset._get_target_item()`` and returns only ``human_repo_id``
    without the video tensor.
    """

    def __init__(self, config: PairedDatasetConfig):
        self.config     = config
        self.input_mode = InputMode(config.input_mode)
        self.skip_human_video: bool = False

        print("\n" + "=" * 80)
        print("🔄 INITIALIZING HUMAN-SIM PAIRED DATASET")
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
        valid_sim_ids   = self.task_mapper.get_all_sim_task_ids()

        print(f"\n✓ Mapping loaded:")
        print(f"  - {len(valid_human_ids)} human tasks: {sorted(valid_human_ids)}")
        print(f"  - {len(valid_sim_ids)} sim tasks: {sorted(valid_sim_ids)}")

        filtered_human_configs = filter_dataset_config(
            config.human_dataset_file, valid_human_ids, "human", config.split)
        filtered_sim_configs   = filter_dataset_config(
            config.sim_dataset_file, valid_sim_ids, "sim", config.split)

        if not filtered_human_configs or not filtered_sim_configs:
            raise ValueError(f"No valid paired tasks found for split '{config.split}'")

        human_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(filtered_human_configs, human_temp, indent=2)
        human_temp.close()

        sim_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(filtered_sim_configs, sim_temp, indent=2)
        sim_temp.close()

        # Build human dataset (only needed for video modes)
        self.human_dataset = None
        if self.input_mode in [InputMode.VIDEO_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
            print("\n" + "=" * 80)
            print("🎥 LOADING HUMAN VIDEO DATASET")
            print("=" * 80)
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

        # Build sim dataset
        print("\n" + "=" * 80)
        print("🤖 LOADING SIM TRAJECTORY DATASET")
        print("=" * 80)
        sim_config = LeRobotDataConfig(
            source_type="sim",
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
            skill=config.skill,
            xskill=config.xskill,
            robot_frame_gap=config.robot_frame_gap,
        )
        self.sim_dataset = build_lerobot_dataset(sim_config)

        os.unlink(human_temp.name)
        os.unlink(sim_temp.name)

        self._build_and_validate_pairing()

        print("\n" + "=" * 80)
        print("✅ HUMAN-SIM PAIRED DATASET READY")
        print("=" * 80)
        print(f"📊 Summary:")
        print(f"   - Total sim samples: {len(self.sim_dataset)}")
        print(f"   - Valid paired samples: {len(self)}")
        print(f"   - Skipped samples: {len(self.sim_dataset) - len(self)}")
        print(f"   - Valid paired tasks: {len(self.paired_tasks)}")
        print(f"   - Input mode: {self.input_mode.value}")
        print(f"   - Include first frame: {config.include_first_frame}")
        print("=" * 80 + "\n")

    def _build_and_validate_pairing(self):
        print("\n" + "=" * 80)
        print("🔗 VALIDATING HUMAN-SIM TASK PAIRINGS")
        print("=" * 80)

        self.task_to_human_indices: dict = {}
        self.task_to_sim_indices:   dict = {}
        self.paired_tasks:          list = []
        self.valid_indices:         list = []

        # Index human videos
        print("\n🔑 Indexing human videos...")
        if self.human_dataset is not None:
            if hasattr(self.human_dataset, "all_tasks") and hasattr(self.human_dataset, "task_to_repos"):
                for task in self.human_dataset.all_tasks:
                    repos = self.human_dataset.task_to_repos[task]
                    total_episodes = sum(r["num_episodes"] for r in repos)
                    self.task_to_human_indices[task] = {
                        "num_episodes": total_episodes,
                        "repos": repos,
                        "task_name": task,
                    }
                    print(f"   ✓ {task} - {total_episodes} episodes")
        else:
            for h_id in self.task_mapper.get_all_human_task_ids():
                self.task_to_human_indices[h_id] = {"task_name": h_id}

        # O(k) metadata-based indexing — build contiguous ranges from
        # cumulative_sizes directly instead of looping over all N indices.
        print("\n🔑 Indexing sim trajectories (O(sub-datasets))...")
        lerobot_ds = self.sim_dataset.lerobot_dataset

        prev = 0
        for ds_idx, curr_ds in enumerate(lerobot_ds.datasets):
            repo_id = curr_ds.repo_id
            end     = lerobot_ds.cumulative_sizes[ds_idx]
            self.task_to_sim_indices[repo_id] = list(range(prev, end))
            print(f"   {repo_id}: {end - prev} samples")
            prev = end

        # Validate pairings
        print("\n🔍 Validating pairings...")
        for sim_task_id, indices in self.task_to_sim_indices.items():
            human_task_id = self.task_mapper.get_human_task_from_sim(sim_task_id)
            if human_task_id is None:
                continue
            if human_task_id not in self.task_to_human_indices:
                continue
            self.paired_tasks.append((human_task_id, sim_task_id))
            self.valid_indices.extend(indices)

        print(f"\n✅ Successfully paired: {len(self.paired_tasks)} task pair(s)\n")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _get_first_frame_obs(self, sim_sample: dict) -> dict:
        first_frame_obs = {}
        if "states" in sim_sample:
            first_frame_obs["states"] = sim_sample["states"][0:1]
        view_idx = 1
        while f"view_{view_idx}" in sim_sample:
            view = sim_sample[f"view_{view_idx}"]
            first_frame_obs[f"view_{view_idx}"] = (
                view[0:1] if view.dim() == 4 else view.unsqueeze(0)
            )
            view_idx += 1
        return first_frame_obs

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual_sim_idx = self.valid_indices[idx]
        sim_sample     = self.sim_dataset[actual_sim_idx]
        sim_task_id    = sim_sample.get("repo_id", sim_sample.get("task_name", ""))

        human_task_id = self.task_mapper.get_human_task_from_sim(sim_task_id)
        if human_task_id is None or human_task_id not in self.task_to_human_indices:
            raise ValueError(f"No human pairing for sim task: {sim_task_id}")

        robot_obs = {"states": sim_sample["states"]}
        view_idx = 1
        while f"view_{view_idx}" in sim_sample:
            robot_obs[f"view_{view_idx}"] = sim_sample[f"view_{view_idx}"].permute(0, 2, 3, 1)
            view_idx += 1

        result = {
            "robot_obs":     robot_obs,
            "robot_actions": sim_sample["actions"],
            "dataset_idx":   sim_sample.get("dataset_idx", torch.tensor(0)),
            "human_task_id": human_task_id,
            "sim_task_id":   sim_task_id,
            "sample_id":     f"{sim_task_id}::{actual_sim_idx}",
        }

        if self.config.include_first_frame:
            result["robot_first_frame_obs"] = self._get_first_frame_obs(sim_sample)

        if self.config.skill:
            skill_frames = {}
            view_idx = 1
            while f"view_{view_idx}" in sim_sample.get("skill_frames", {}):
                skill_frames[f"view_{view_idx}"] = sim_sample["skill_frames"][f"view_{view_idx}"].permute(0, 2, 3, 1)
                view_idx += 1
            result["skill_frames"] = skill_frames

        if self.input_mode in [InputMode.VIDEO_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
            human_task_info = self.task_to_human_indices[human_task_id]
            human_repo_id   = human_task_info["repos"][0]["repo_id"]
            result["human_repo_id"] = human_repo_id

            if not self.skip_human_video:
                human_video = self.human_dataset._get_target_item(human_repo_id)["video"]
                result["human_video"] = human_video

        if self.input_mode in [InputMode.LANGUAGE_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
            language = self.task_mapper.get_description("human", human_task_id)
            if language is None:
                language = self.task_mapper.get_description("sim", sim_task_id)
            if language is None:
                language = f"Task {human_task_id}"
            result["language"] = language

        return result


# ---------------------------------------------------------------------------
# HumanRobotPairedDataset  (fixed: fast metadata indexing)
# ---------------------------------------------------------------------------

class HumanRobotPairedDataset(Dataset):
    """Paired dataset combining human videos with real robot trajectories.

    Bug fix vs original: _build_and_validate_pairing uses sub-dataset metadata
    (repo_id, cumulative_sizes) instead of calling __getitem__ for every sample,
    reducing initialization from O(N * decode_cost) to O(N) index arithmetic.
    """

    def __init__(self, config: PairedDatasetConfig):
        self.config     = config
        self.input_mode = InputMode(config.input_mode)
        self.skip_human_video: bool = False

        print("\n" + "=" * 80)
        print("🔄 INITIALIZING HUMAN-ROBOT PAIRED DATASET")
        print("=" * 80)
        print(f"   Input mode: {self.input_mode.value}")
        print(f"   Include first frame: {config.include_first_frame}")

        print(f"\n📖 Loading task mapping from: {config.task_mapping_file}")
        self.task_mapper = TaskMapper(
            config.task_mapping_file,
            config.human_task_description_file,
            config.robot_task_description_file,
            config.sim_task_description_file,
        )

        valid_human_ids = self.task_mapper.get_all_human_task_ids()
        valid_robot_ids = self.task_mapper.get_all_robot_task_ids()

        print(f"\n✓ Mapping loaded:")
        print(f"  - {len(valid_human_ids)} human tasks")
        print(f"  - {len(valid_robot_ids)} robot tasks")

        filtered_human_configs = filter_dataset_config(
            config.human_dataset_file, valid_human_ids, "human", config.split)
        filtered_robot_configs  = filter_dataset_config(
            config.robot_dataset_file, valid_robot_ids, "robot", config.split)

        if not filtered_human_configs or not filtered_robot_configs:
            raise ValueError(f"No valid paired tasks found for split '{config.split}'")

        human_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(filtered_human_configs, human_temp, indent=2)
        human_temp.close()

        robot_temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(filtered_robot_configs, robot_temp, indent=2)
        robot_temp.close()

        # Build human dataset
        self.human_dataset = None
        if self.input_mode in [InputMode.VIDEO_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
            print("\n" + "=" * 80)
            print("🎥 LOADING HUMAN VIDEO DATASET")
            print("=" * 80)
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

        # Build robot dataset
        print("\n" + "=" * 80)
        print("🦾 LOADING ROBOT TRAJECTORY DATASET")
        print("=" * 80)
        robot_config = LeRobotDataConfig(
            source_type="robot",
            root=config.robot_root,
            split=config.split,
            dataset_file=robot_temp.name,
            task_description_file=config.robot_task_description_file,
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
            skill=config.skill,
            xskill=config.xskill,
            robot_frame_gap=config.robot_frame_gap,
        )
        self.robot_dataset = build_lerobot_dataset(robot_config)

        os.unlink(human_temp.name)
        os.unlink(robot_temp.name)

        self._build_and_validate_pairing()

        print("\n" + "=" * 80)
        print("✅ HUMAN-ROBOT PAIRED DATASET READY")
        print("=" * 80)
        print(f"📊 Summary:")
        print(f"   - Total robot samples: {len(self.robot_dataset)}")
        print(f"   - Valid paired samples: {len(self)}")
        print(f"   - Skipped samples: {len(self.robot_dataset) - len(self)}")
        print(f"   - Valid paired tasks: {len(self.paired_tasks)}")
        print(f"   - Input mode: {self.input_mode.value}")
        print(f"   - Include first frame: {config.include_first_frame}")
        print("=" * 80 + "\n")

    def _build_and_validate_pairing(self):
        """Build index using sub-dataset metadata (O(N) index arithmetic only).

        Previously called self.robot_dataset[idx] for every sample, which
        decoded images and applied transforms just to read the repo_id —
        O(N * ~100 ms) for a large dataset.  Now we read repo_id directly
        from curr_dataset.repo_id via the cumulative_sizes index.
        """
        print("\n" + "=" * 80)
        print("🔗 VALIDATING HUMAN-ROBOT TASK PAIRINGS")
        print("=" * 80)

        self.task_to_human_indices: dict = {}
        self.task_to_robot_indices: dict = {}
        self.paired_tasks:          list = []
        self.valid_indices:         list = []

        # Index human videos
        print("\n🔑 Indexing human videos...")
        if self.human_dataset is not None:
            if hasattr(self.human_dataset, "all_tasks") and hasattr(self.human_dataset, "task_to_repos"):
                for task in self.human_dataset.all_tasks:
                    repos = self.human_dataset.task_to_repos[task]
                    total_episodes = sum(r["num_episodes"] for r in repos)
                    self.task_to_human_indices[task] = {
                        "num_episodes": total_episodes,
                        "repos": repos,
                        "task_name": task,
                    }
                    print(f"   ✓ {task} - {total_episodes} episodes")
        else:
            for h_id in self.task_mapper.get_all_human_task_ids():
                self.task_to_human_indices[h_id] = {"task_name": h_id}

        # O(k) metadata-based indexing — build contiguous ranges from
        # cumulative_sizes directly (same approach as HumanSimPairedDataset).
        print("\n🔑 Indexing robot trajectories (O(sub-datasets))...")
        lerobot_ds = self.robot_dataset.lerobot_dataset

        prev = 0
        for ds_idx, curr_ds in enumerate(lerobot_ds.datasets):
            robot_task_id = curr_ds.repo_id
            end           = lerobot_ds.cumulative_sizes[ds_idx]
            self.task_to_robot_indices[robot_task_id] = list(range(prev, end))
            print(f"   {robot_task_id}: {end - prev} samples")
            prev = end

        # Validate pairings
        print("\n🔍 Validating pairings...")
        for robot_task_id, indices in self.task_to_robot_indices.items():
            human_task_id = self.task_mapper.get_human_task_from_robot(robot_task_id)
            if human_task_id is None or human_task_id not in self.task_to_human_indices:
                continue
            self.paired_tasks.append((human_task_id, robot_task_id))
            self.valid_indices.extend(indices)

        print(f"\n✅ Successfully paired: {len(self.paired_tasks)} task pair(s)\n")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _get_first_frame_obs(self, robot_sample: dict) -> dict:
        first_frame_obs = {}
        if "states" in robot_sample:
            first_frame_obs["states"] = robot_sample["states"][0:1]
        view_idx = 1
        while f"view_{view_idx}" in robot_sample:
            view = robot_sample[f"view_{view_idx}"]
            first_frame_obs[f"view_{view_idx}"] = (
                view[0:1] if view.dim() == 4 else view.unsqueeze(0)
            )
            view_idx += 1
        return first_frame_obs

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        actual_robot_idx = self.valid_indices[idx]
        robot_sample     = self.robot_dataset[actual_robot_idx]
        robot_task_id    = robot_sample.get("repo_id", robot_sample.get("task_name", ""))

        human_task_id = self.task_mapper.get_human_task_from_robot(robot_task_id)
        if human_task_id is None or human_task_id not in self.task_to_human_indices:
            raise ValueError(f"No human pairing for robot task: {robot_task_id}")

        robot_obs = {"states": robot_sample["states"]}
        view_idx = 1
        while f"view_{view_idx}" in robot_sample:
            view_data = robot_sample[f"view_{view_idx}"]
            # view_data shape: (obs_horizon, C, H, W) → (obs_horizon, H, W, C)
            if view_data.dim() == 4:
                view_data = view_data.permute(0, 2, 3, 1)
            robot_obs[f"view_{view_idx}"] = view_data
            view_idx += 1

        result = {
            "robot_obs":     robot_obs,
            "robot_actions": robot_sample["actions"],
            "dataset_idx":   robot_sample.get("dataset_idx", torch.tensor(0)),
            "human_task_id": human_task_id,
            "robot_task_id": robot_task_id,
            "sample_id":     f"{robot_task_id}::{actual_robot_idx}",
        }
        
        if self.config.skill:
            skill_frames = {}
            view_idx = 1
            while f"view_{view_idx}" in robot_sample.get("skill_frames", {}):
                sf = robot_sample["skill_frames"][f"view_{view_idx}"]

                # robot dataset returns (T, C, H, W), paired dataset wants (T, H, W, C)
                if sf.dim() == 4:
                    sf = sf.permute(0, 2, 3, 1)

                skill_frames[f"view_{view_idx}"] = sf
                view_idx += 1

            result["skill_frames"] = skill_frames

        if self.config.include_first_frame:
            result["robot_first_frame_obs"] = self._get_first_frame_obs(robot_sample)

        if self.input_mode in [InputMode.VIDEO_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
            human_task_info = self.task_to_human_indices[human_task_id]
            human_repo_id   = human_task_info["repos"][0]["repo_id"]
            result["human_repo_id"] = human_repo_id

            if not self.skip_human_video:
                human_video = self.human_dataset._get_target_item(human_repo_id)["video"]
                result["human_video"] = human_video

        if self.input_mode in [InputMode.LANGUAGE_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
            language = self.task_mapper.get_description("human", human_task_id)
            if language is None:
                language = self.task_mapper.get_description("robot", robot_task_id)
            if language is None:
                language = f"Task {human_task_id}"
            result["language"] = language

        return result

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "num_human_videos":             len(self.human_dataset) if self.human_dataset else 0,
            "num_robot_trajectories_total": len(self.robot_dataset),
            "num_valid_paired_samples":     len(self.valid_indices),
            "num_skipped_samples":          len(self.robot_dataset) - len(self.valid_indices),
            "paired_tasks":                 self.paired_tasks,
            "num_paired_tasks":             len(self.paired_tasks),
            "input_mode":                   self.input_mode.value,
            "include_first_frame":          self.config.include_first_frame,
        }


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_paired_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate for video_only mode.  human_video optional (skip_human_video)."""
    collated: Dict[str, Any] = {
        "robot_actions": torch.stack([item["robot_actions"] for item in batch]),
        "dataset_idx":   torch.stack([item["dataset_idx"]   for item in batch]),
        "human_repo_id": [item["human_repo_id"] for item in batch],
    }

    if "human_video" in batch[0]:
        collated["human_video"] = torch.stack([item["human_video"] for item in batch])

    robot_obs_keys = batch[0]["robot_obs"].keys()
    collated["robot_obs"] = {
        key: torch.stack([item["robot_obs"][key] for item in batch])
        for key in robot_obs_keys
    }

    if "robot_first_frame_obs" in batch[0]:
        first_frame_keys = batch[0]["robot_first_frame_obs"].keys()
        collated["robot_first_frame_obs"] = {
            key: torch.stack([item["robot_first_frame_obs"][key] for item in batch])
            for key in first_frame_keys
        }

    return collated


def collate_paired_batch_vla(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate for language_only mode."""
    collated: Dict[str, Any] = {
        "language":      [item["language"]      for item in batch],
        "robot_actions": torch.stack([item["robot_actions"] for item in batch]),
        "dataset_idx":   torch.stack([item["dataset_idx"]   for item in batch]),
    }

    robot_obs_keys = batch[0]["robot_obs"].keys()
    collated["robot_obs"] = {
        key: torch.stack([item["robot_obs"][key] for item in batch])
        for key in robot_obs_keys
    }

    if "robot_first_frame_obs" in batch[0]:
        first_frame_keys = batch[0]["robot_first_frame_obs"].keys()
        collated["robot_first_frame_obs"] = {
            key: torch.stack([item["robot_first_frame_obs"][key] for item in batch])
            for key in first_frame_keys
        }

    return collated


def collate_paired_batch_video_and_language(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate for video_and_language mode.  human_video optional."""
    collated: Dict[str, Any] = {
        "language":      [item["language"]      for item in batch],
        "robot_actions": torch.stack([item["robot_actions"] for item in batch]),
        "dataset_idx":   torch.stack([item["dataset_idx"]   for item in batch]),
        "human_task_id": [item["human_task_id"] for item in batch],
        "human_repo_id": [item["human_repo_id"] for item in batch],
    }

    if "human_video" in batch[0]:
        collated["human_video"] = torch.stack([item["human_video"] for item in batch])

    if "sim_task_id" in batch[0]:
        collated["sim_task_id"]   = [item["sim_task_id"]   for item in batch]
    if "robot_task_id" in batch[0]:
        collated["robot_task_id"] = [item["robot_task_id"] for item in batch]

    robot_obs_keys = batch[0]["robot_obs"].keys()
    collated["robot_obs"] = {
        key: torch.stack([item["robot_obs"][key] for item in batch])
        for key in robot_obs_keys
    }

    if "robot_first_frame_obs" in batch[0]:
        first_frame_keys = batch[0]["robot_first_frame_obs"].keys()
        collated["robot_first_frame_obs"] = {
            key: torch.stack([item["robot_first_frame_obs"][key] for item in batch])
            for key in first_frame_keys
        }

    return collated


def get_collate_fn(input_mode: str):
    """Return the collate function for the given input mode."""
    mode = InputMode(input_mode)
    if mode == InputMode.VIDEO_ONLY:
        return collate_paired_batch
    elif mode == InputMode.LANGUAGE_ONLY:
        return collate_paired_batch_vla
    else:
        return collate_paired_batch_video_and_language