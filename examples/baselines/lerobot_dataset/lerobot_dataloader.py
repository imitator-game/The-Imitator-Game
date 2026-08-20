"""
Unified LeRobot dataloader entry point.
Selects human/sim/robot dataloaders via a single config.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from examples.baselines.lerobot_dataset.lerobot_human_dataset import (
    HumanVideoDataConfig,
    HumanVideoDataset,
)
from examples.baselines.lerobot_dataset.lerobot_robot_dataset import (
    LeRobotRobotDataConfig,
    LeRobotRobotDataset,
)
from examples.baselines.lerobot_dataset.lerobot_sim_dataset import (
    LeRobotSimDataConfig,
    LeRobotSimDataset,
)


@dataclass
class LeRobotDataConfig:
    source_type: str
    """human, sim, or robot"""

    root: str = "demos"
    split: str = "train"
    repo_id: Optional[str] = None
    dataset_file: Optional[str] = None
    task_description_file: Optional[str] = None

    cameras: List[str] = field(default_factory=lambda: ["cam1", "cam2"])
    include_depth: bool = False
    image_size: Tuple[int, int] = (224, 224)
    horizon: int = 16
    obs_horizon: int = 1
    state_type: str = "qpos"
    single_arm: bool = False
    depth_mode: str = "sim"

    fps: int = 30
    video_backend: str = "pyav"
    tolerance_s: float = 0.05

    # Human-video specific
    num_frames: int = 10
    sampling_strategy: str = "uniform_jitter"
    max_jitter: int = 3

    debug: bool = False
    vla: bool = False

    enable_augmentation: bool = True

    pre_decode: bool = False
    pre_decode_cache_dir: str = "tmp/human_video_cache"
    pre_decode_num_workers: int = 4

    # For Skill Model Training
    skill: bool = False
    xskill: bool = False
    robot_frame_gap: int = 35



def build_lerobot_dataset(config: LeRobotDataConfig):
    source = config.source_type.lower()
    if source == "human":
        cfg = HumanVideoDataConfig(
            root=config.root,
            split=config.split,
            dataset_file=config.dataset_file,
            task_description_file=config.task_description_file,
            cameras=config.cameras,
            include_depth=config.include_depth,
            num_frames=config.num_frames,
            image_size=config.image_size,
            sampling_strategy=config.sampling_strategy,
            max_jitter=config.max_jitter,
            video_backend=config.video_backend,
            fps=config.fps,
            debug=config.debug,
            vla=config.vla,
            enable_augmentation=config.enable_augmentation,
            pre_decode=config.pre_decode,
            pre_decode_cache_dir=config.pre_decode_cache_dir,
            pre_decode_num_workers=config.pre_decode_num_workers,
        )
        return HumanVideoDataset(cfg)

    if source == "sim":
        cfg = LeRobotSimDataConfig(
            root=config.root,
            split=config.split,
            image_size=config.image_size,
            state_type=config.state_type,
            include_depth=config.include_depth,
            cameras=config.cameras,
            single_arm=config.single_arm,
            horizon=config.horizon,
            obs_horizon=config.obs_horizon,
            repo_id=config.repo_id,
            fps=config.fps,
            video_backend=config.video_backend,
            tolerance_s=config.tolerance_s,
            dataset_file=config.dataset_file,
            task_description_file=config.task_description_file,
            depth_mode=config.depth_mode,
            debug=config.debug,
            enable_augmentation=config.enable_augmentation,
            skill=config.skill,
            xskill=config.xskill,
            robot_frame_gap=config.robot_frame_gap,
        )
        return LeRobotSimDataset(cfg)

    if source == "robot":
        cfg = LeRobotRobotDataConfig(
            root=config.root,
            split=config.split,
            image_size=config.image_size,
            state_type=config.state_type,
            include_depth=config.include_depth,
            cameras=config.cameras,
            single_arm=config.single_arm,
            horizon=config.horizon,
            obs_horizon=config.obs_horizon,         # ← now properly forwarded
            repo_id=config.repo_id,
            fps=config.fps,
            video_backend=config.video_backend,
            tolerance_s=config.tolerance_s,
            dataset_file=config.dataset_file,
            task_description_file=config.task_description_file,
            depth_mode="robot",                     # robot depth mode default
            debug=config.debug,
            enable_augmentation=config.enable_augmentation,
            skill=config.skill,
            xskill=config.xskill,
            robot_frame_gap=config.robot_frame_gap,
        )
        return LeRobotRobotDataset(cfg)

    raise ValueError(f"Unsupported source_type: {config.source_type!r}")