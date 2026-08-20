import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch, random
import numpy as np
import albumentations as A

from examples.baselines.lerobot_dataset.lerobot_human_dataset import (
    HumanVideoDataConfig,
    HumanVideoDataset,
    decode_depth
)
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer

try:
    from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDatasetMetadata
except ImportError:
    LeRobotDatasetMetadata = None


@dataclass
class HumanVideoSimEvaluateProcessorConfig:
    # Human dataset config
    human_root: str = "data"
    human_split: str = "train"
    human_dataset_file: Optional[str] = "examples/baselines/lerobot_dataset/config/human_config.json"
    human_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    human_cameras: List[str] = field(default_factory=lambda: ["zed2i"])
    human_include_depth: bool = True
    human_num_frames: int = 10
    human_image_size: Tuple[int, int] = (224, 224)
    human_sampling_strategy: str = "uniform_jitter"
    human_max_jitter: int = 3
    human_video_backend: str = "torchcodec"
    human_fps: int = 30
    human_debug: bool = False

    # Sim dataset config (for normalization stats)
    sim_root: str = "data"
    sim_split: str = "train"
    sim_dataset_file: Optional[str] = "examples/baselines/lerobot_dataset/config/sim_config.json"
    sim_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"
    sim_state_type: str = "qpos"
    sim_single_arm: bool = False
    normalization_method: str = "bounds_q99"

    task_mapping_file: Optional[str] = "examples/baselines/lerobot_dataset/task_mapping.json"

    vla: bool = False  


class HumanVideoSimEvaluateProcessor:
    """
    Processor that loads HumanVideoDataset and uses sim normalization stats.
    It maps human_task_id -> sim_task_id -> dataset_idx to normalize/denormalize.
    """

    def __init__(self, config: Optional[HumanVideoSimEvaluateProcessorConfig] = None):
        self.config = config or HumanVideoSimEvaluateProcessorConfig()

        # Load human dataset (required)
        human_config = HumanVideoDataConfig(
            root=self.config.human_root,
            split=self.config.human_split,
            dataset_file=self.config.human_dataset_file,
            task_description_file=self.config.human_task_description_file,
            cameras=self.config.human_cameras,
            include_depth=self.config.human_include_depth,
            num_frames=self.config.human_num_frames,
            image_size=self.config.human_image_size,
            sampling_strategy=self.config.human_sampling_strategy,
            max_jitter=self.config.human_max_jitter,
            video_backend=self.config.human_video_backend,
            fps=self.config.human_fps,
            debug=self.config.human_debug,
            vla=self.config.vla,
        )
        self.human_dataset = HumanVideoDataset(human_config)

        # Load sim stats and build ActionNormalizer
        self.normalizer = ActionNormalizer()
        # repo_id -> dataset_idx (for sim stats lookup)
        self.repo_id_to_dataset_idx: Dict[str, int] = {}
        # sim_task_id -> dataset_idx (for task-level lookup)
        self.sim_task_to_dataset_idx: Dict[str, int] = {}
        self._load_sim_normalizer()

        with open(config.task_mapping_file, 'r') as f:
            task_mappings = json.load(f)
        self.task_mappings = task_mappings.get("task_mappings", [])

        # Build robot_repo_id → human_task_id reverse map so that
        # get_video / get_language / get_task_description also accept robot
        # repo_ids (e.g. "robot_H1_L0") as the lookup key.
        self.robot_task_id_to_human: Dict[str, str] = {}
        for mapping in self.task_mappings:
            human_id = mapping.get("human_task_id", "")
            for robot_id in mapping.get("robot_task_id", []):
                self.robot_task_id_to_human[robot_id] = human_id

        additional_targets = {}
        if self.human_dataset.config.include_depth:
            additional_targets['depth'] = 'image'

        self.spatial_transform = A.Compose([
            A.Resize(height=self.config.human_image_size[0], width=config.human_image_size[1], p=1.0),
        ], additional_targets=additional_targets)

        self.rgb_transform = A.Compose([
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ])

    def _resolve_state_action_keys(self) -> Tuple[str, str]:
        if self.config.sim_state_type == "eepos":
            return "observation.eepos_gripper_states", "action.eepos_gripper_actions"
        if self.config.sim_state_type == "qpos":
            return "observation.qpos_gripper_states", "action.qpos_gripper_actions"
        if self.config.sim_state_type == "mixpos":
            return "observation.eepos_gripper_states", "action.qpos_gripper_actions"
        raise ValueError(f"Unknown sim_state_type: {self.config.sim_state_type}")

    def _load_sim_normalizer(self) -> None:
        if LeRobotDatasetMetadata is None:
            raise ImportError("LeRobotDatasetMetadata is not available. Please install lerobot.")
        if not self.config.sim_dataset_file or not os.path.exists(self.config.sim_dataset_file):
            raise FileNotFoundError(f"sim_dataset_file not found: {self.config.sim_dataset_file}")

        with open(self.config.sim_dataset_file, "r") as f:
            dataset_configs = json.load(f)

        state_key, action_key = self._resolve_state_action_keys()

        for idx, ds_cfg in enumerate(dataset_configs):
            repo_id = ds_cfg.get("repo_id")
            ds_root = os.path.join(self.config.sim_root, ds_cfg.get("root"))

            # Load metadata only (stats/info/tasks) without loading dataset frames
            meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)

            self.normalizer.add_dataset_stats(
                dataset_idx=idx,
                repo_id=repo_id,
                stats=meta.stats,
                state_key=state_key,
                action_key=action_key,
                single_arm=self.config.sim_single_arm,
            )
            self.repo_id_to_dataset_idx[repo_id] = idx

            # Build additional mapping: strip leading "Lx_" prefix if exists
            if "_" in repo_id:
                base_name = repo_id.split("_", 1)[1]
                self.sim_task_to_dataset_idx[base_name] = idx

    def _resolve_dataset_idx(self, sim_task_id) -> int:
        if isinstance(sim_task_id, int):
            if sim_task_id < 0 or sim_task_id >= len(self.repo_id_to_dataset_idx):
                raise ValueError(f"Invalid dataset_idx: {sim_task_id}")
            return sim_task_id
        if sim_task_id in self.repo_id_to_dataset_idx:
            return self.repo_id_to_dataset_idx[sim_task_id]

        if sim_task_id in self.sim_task_to_dataset_idx:
            return self.sim_task_to_dataset_idx[sim_task_id]

        for repo_id, idx in self.repo_id_to_dataset_idx.items():
            if repo_id.endswith(sim_task_id):
                return idx

        raise ValueError(f"Cannot find dataset_idx for sim_task_id: {sim_task_id}")

    def normalize_state_rgb(
        self,
        state: torch.Tensor,
        rgb,
        sim_task_id: Optional[str] = None,
    ) -> torch.Tensor:
        if sim_task_id is None:
            raise ValueError("sim_task_id is required (mapping disabled).")
        dataset_idx = self._resolve_dataset_idx(sim_task_id)

        # TODO: get real state here
        if state.shape[-1] > 27:
            qpos_state = torch.concatenate([state[..., 0:9], state[..., 18:27]], dim=-1)
        else:
            qpos_state = state

        state_out = self.normalizer.normalize_state(
            qpos_state,
            dataset_idx,
            method=self.config.normalization_method,
        )
        if isinstance(rgb, torch.Tensor):
            rgb_out = rgb.float() / 255.0
        elif isinstance(rgb, dict):
            rgb_out = {k: v.float() / 255.0 for k, v in rgb.items()}
        elif isinstance(rgb, list):
            rgb_out = [v.float() / 255.0 for v in rgb]
        return state_out, rgb_out

    def denormalize_action(
        self,
        normalized_action: torch.Tensor,
        sim_task_id: Optional[str] = None,
    ) -> torch.Tensor:
        if sim_task_id is None:
            raise ValueError("sim_task_id is required (mapping disabled).")
        dataset_idx = self._resolve_dataset_idx(sim_task_id)
        return self.normalizer.denormalize_action(
            normalized_action,
            dataset_idx,
            method=self.config.normalization_method,
        )

    def _resolve_human_task_id(self, key: str) -> str:
        """
        Resolve any lookup key to a human_task_id.

        Priority:
          1. Direct match in sim_task_id (via get_human_video_id)
          2. Direct match in robot_task_id (e.g. "robot_H1_L0")
          3. Raise ValueError

        This lets callers pass either a sim env-id like
        ``"L0_TwoRobotPourCup-v1"`` or a robot repo-id like
        ``"robot_H1_L0"`` without needing to know the mapping.
        """
        from examples.baselines.lerobot_dataset.lerobot_paired_dataset import get_human_video_id
        human_id = get_human_video_id(key, self.task_mappings)
        if human_id is not None:
            return human_id
        # Fall back to robot_task_id lookup
        if key in self.robot_task_id_to_human:
            return self.robot_task_id_to_human[key]
        raise ValueError(
            f"Cannot resolve human_task_id for key '{key}'. "
            f"It was not found in sim_task_id or robot_task_id lists."
        )

    def get_video(self, sim_task_id: Optional[str] = None, num_envs: int = 1) -> torch.Tensor:
        """
        Fetch a single video sample from the human dataset.
        Returns a 4D tensor: [T, H, W, C].

        ``sim_task_id`` may be a sim env-id (e.g. "L0_TwoRobotPourCup-v1"),
        a bare env name ("TwoRobotPourCup-v1"), OR a robot repo-id
        (e.g. "robot_H1_L0").  All three forms are resolved via
        ``_resolve_human_task_id``.
        """
        if sim_task_id is None:
            raise ValueError("sim_task_id is required (mapping disabled).")
        human_video_id = self._resolve_human_task_id(sim_task_id)

        videos = []
        for n in range(num_envs):
            video = self._get_video(human_video_id)
            assert video.ndim == 4
            videos.append(video.float())

        return torch.stack(videos)
    
    def get_language(self, sim_task_id: Optional[str] = None, num_envs: int = 1) -> List[str]:
        """
        Fetch a list of language descriptions from the human dataset.

        ``sim_task_id`` accepts the same forms as ``get_video``.
        Returns a list of strings of length ``num_envs``.
        """
        if sim_task_id is None:
            raise ValueError("sim_task_id is required (mapping disabled).")
        human_video_id = self._resolve_human_task_id(sim_task_id)

        languages = []
        for n in range(num_envs):
            language = self._get_language(human_video_id)
            languages.append(language)

        return languages

    def get_task_description(self, sim_task_id: Optional[str] = None) -> str:
        """
        Fetch a single language description string for the given task.

        ``sim_task_id`` accepts the same forms as ``get_video`` (sim env-id,
        bare env name, or robot repo-id).  Returns one randomly sampled
        description string.
        """
        descriptions = self.get_language(sim_task_id, num_envs=1)
        return descriptions[0]

    def _get_video(self, repo_id: str) -> torch.Tensor:
        episodes_dataset = self.human_dataset.repo_episodes[repo_id]
        episode_idx = random.randint(0, len(episodes_dataset) - 1)
        episode_row = episodes_dataset[episode_idx]

        episode_length = int(episode_row['length'])

        start_frame = 0
        end_frame = episode_length - 1

        if self.human_dataset.config.sampling_strategy == "uniform_jitter":
            frame_indices = self.human_dataset._sample_frames_uniform_jitter(
                start_frame, end_frame, self.human_dataset.config.num_frames
            )
        else:  # random_interval
            frame_indices = self.human_dataset._sample_frames_random_interval(
                start_frame, end_frame, self.human_dataset.config.num_frames
            )

        info = self.human_dataset.repo_info[repo_id]
        video_path_format = info.get('video_path')

        if video_path_format is None:
            raise ValueError(f"video_path not found in info for {repo_id}")

        all_view_tensors = []
        depth_videos = {}
        fps = info['fps']

        for cam_name in self.human_dataset.config.cameras:
            video_key = f"observation.images.{cam_name}"
            chunk_idx_key = f"videos/{video_key}/chunk_index"
            file_idx_key = f"videos/{video_key}/file_index"

            if chunk_idx_key not in episode_row or file_idx_key not in episode_row:
                continue

            chunk_idx = int(episode_row[chunk_idx_key])
            file_idx = int(episode_row[file_idx_key])

            repo_path = Path(self.human_dataset.task_to_repos[repo_id][0]['repo_path'])

            rgb_video_path = repo_path / video_path_format.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx
            )

            if not rgb_video_path.exists():
                continue

            from_ts_key = f"videos/{video_key}/from_timestamp"

            if from_ts_key in episode_row:
                from_timestamp = float(episode_row[from_ts_key])
                video_start_frame = int(from_timestamp * fps)
                absolute_frame_indices = [video_start_frame + idx for idx in frame_indices]
            else:
                absolute_frame_indices = frame_indices

            # Read RGB frames
            rgb_frames = self.human_dataset.video_reader.read_frames(str(rgb_video_path), absolute_frame_indices)

            # Process depth if needed
            depth_frames = None
            if self.human_dataset.config.include_depth:
                depth_video_key = f"observation.images.{cam_name}_depth"
                depth_chunk_key = f"videos/{depth_video_key}/chunk_index"
                depth_file_key = f"videos/{depth_video_key}/file_index"

                if depth_chunk_key in episode_row and depth_file_key in episode_row:
                    depth_chunk_idx = int(episode_row[depth_chunk_key])
                    depth_file_idx = int(episode_row[depth_file_key])

                    depth_video_path = repo_path / video_path_format.format(
                        video_key=depth_video_key,
                        chunk_index=depth_chunk_idx,
                        file_index=depth_file_idx
                    )

                    if depth_video_path.exists():
                        # Depth video uses same frame index calculation
                        depth_from_ts_key = f"videos/{depth_video_key}/from_timestamp"
                        if depth_from_ts_key in episode_row:
                            depth_from_timestamp = float(episode_row[depth_from_ts_key])
                            depth_video_start_frame = int(depth_from_timestamp * fps)
                            depth_absolute_frame_indices = [depth_video_start_frame + idx for idx in frame_indices]
                        else:
                            depth_absolute_frame_indices = frame_indices

                        depth_rgb_frames = self.human_dataset.video_reader.read_frames(
                            str(depth_video_path), depth_absolute_frame_indices
                        )

                        # Decode depth for each frame
                        depth_frames_list = []
                        for depth_rgb in depth_rgb_frames:
                            depth = decode_depth(depth_rgb, cam_name)
                            depth_frames_list.append(depth[..., None])  # Add channel dim
                        depth_frames = np.stack(depth_frames_list, axis=0)  # [T, H, W, 1]

            processed_rgb = []
            processed_depth = []

            for t in range(len(rgb_frames)):
                rgb_frame = rgb_frames[t]

                if depth_frames is not None:
                    depth_frame = depth_frames[t]
                    # Apply spatial transform
                    transformed = self.spatial_transform(image=rgb_frame, depth=depth_frame)
                    rgb_t = transformed['image']
                    depth_t = transformed['depth']
                else:
                    transformed = self.spatial_transform(image=rgb_frame)
                    rgb_t = transformed['image']
                    depth_t = None
                rgb_final = self.rgb_transform(image=rgb_t)['image']  # [H, W, C]
                rgb_final = torch.from_numpy(rgb_final)
                processed_rgb.append(rgb_final)

                # Apply depth transform
                if depth_t is not None:  
                    depth_t = torch.from_numpy(depth_t)
                    processed_depth.append(depth_t) # [H, W, 1]

            # Stack frames: [T, C, H, W]
            rgb_video = torch.stack(processed_rgb, dim=0)
            all_view_tensors.append(rgb_video)

            if processed_depth:
                depth_video = torch.stack(processed_depth, dim=0)
                depth_videos[cam_name] = depth_video

        # TODO: What's this? Copied from lerobot_human_dataloader.
        if len(all_view_tensors) > 1:
            video = all_view_tensors[0]
        elif len(all_view_tensors) == 1:
            video = all_view_tensors[0]
        else:
            raise RuntimeError(f"No valid camera data for task {repo_id}")

        if not isinstance(video, torch.Tensor):
            video = torch.as_tensor(video)
        if video.shape[-1] != 3:
            video = video.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]

        # TODO
        #if depth_videos:
            # for cam_name, depth_video in depth_videos.items():
            #     depth_video = depth_video.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]
            #     ret_dict[f'depth_{cam_name}'] = depth_video


        return video

    def _get_language(self, repo_id: str) -> str:
        task_name = repo_id  # For human dataset, repo_id is task_name
        descriptions = self.human_dataset.task_descriptions[task_name]
        if isinstance(descriptions, list):
            if not descriptions:
                raise ValueError(f"No task descriptions found for {task_name}")
            return random.choice(descriptions)
        if isinstance(descriptions, str):
            return descriptions
        raise TypeError(f"Unsupported task description type for {task_name}: {type(descriptions)}")