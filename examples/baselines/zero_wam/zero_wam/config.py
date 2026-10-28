"""Portable baseline configuration; importing this module does not load CUDA."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = REPO_ROOT / "examples/baselines/lerobot_dataset"
PACKAGE_ROOT = Path(__file__).resolve().parent
CODE_RESOURCES = {
    "task_mapping_file": "task_mapping.json",
    "human_task_description_file": "task_desc/human_desc.json",
    "sim_task_description_file": "task_desc/sim_desc.json",
}


def outside_repository(value: str, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_relative_to(REPO_ROOT):
        raise ValueError(f"{name} must be outside the public source repository: {path}")
    return path


@dataclass
class ZeroWAMConfig:
    protocol: Literal["scratch", "c15", "c30", "c45"] = "scratch"
    train_scope: Literal["joint"] = "joint"
    model_path: str = os.environ.get("ZERO_WAM_MODEL", "")
    empty_text_embedding: str = os.environ.get("ZERO_WAM_EMPTY_TEXT", "")
    human_root: str = os.environ.get("HUMAN_ROOT", "")
    sim_root: str = os.environ.get("SIM_ROOT", "")
    cache_dir: str = os.environ.get("ZERO_WAM_CACHE", "")
    run_dir: str = ""
    manifest: str | None = None
    human_dataset_file: str | None = None
    sim_dataset_file: str | None = None
    task_mapping_file: str = str(CONFIG_ROOT / "task_mapping.json")
    human_task_description_file: str = str(CONFIG_ROOT / "task_desc/human_desc.json")
    sim_task_description_file: str = str(CONFIG_ROOT / "task_desc/sim_desc.json")
    task: str | None = None
    episode_count: int | None = None  # explicit single-task pilot may extend the protocol slice
    batch_size: int = 1
    effective_batch_size: int = 4
    total_epochs: int = 60
    learning_rate: float = 1e-4
    lr_schedule: Literal["cosine"] = "cosine"
    min_learning_rate: float = 1e-6
    lr_decay_start_step: int = 0
    lr_decay_steps: int = 0
    warmup_steps: int | None = None
    enable_mcp: bool = True
    activation_checkpointing: bool = False
    adapter_rank: int | None = None
    num_workers: int = 4
    seed: int = 1
    sampling_mode: Literal["episode"] = "episode"
    action_history_mode: Literal["zero_history_supervised_v1"] = "zero_history_supervised_v1"
    save_epoch_freq: int = 10
    save_step_freq: int = 1000
    save_seconds: int = 1200
    resume_from: str | None = None
    initialize_from: str | None = None
    local_sharding: bool = True
    dry_run: bool = False
    # Episode recipe: train over all latents, random chunks1..4; infer with2.
    # Retained in checkpoint metadata; episode length comes from the cache.
    robot_frames: int = 4
    frame_chunk_size: int = 2
    action_per_frame: int = 16
    robot_frame_stride: int = 4
    human_fps: float = 12.0
    image_size: int = 224
    fps: int = 30
    camera: str = "zed2i"
    state_dim: int = 18
    action_dim: int = 16
    action_representation: Literal["relative_episode"] = "relative_episode"
    action_channel_offset: int = 14
    normalization_path: str | None = None  # unified train-only state/action statistics
    robot_width: int = 288
    human_height: int = 320
    human_width: int = 480
    video_inference_steps: int = 50
    action_inference_steps: int = 50
    # Shared by native training and policy reconstruction.
    video_snr_shift: float = 5.0
    action_snr_shift: float = 1.0
    icl_rope_h: int = 24
    device: str = "cuda"

    @property
    def pred_horizon(self) -> int:
        return self.frame_chunk_size * self.action_per_frame

    @property
    def lora_rank(self) -> int:
        if self.adapter_rank is not None:
            return self.adapter_rank
        return 106

    def dataset_files(self) -> tuple[Path, Path]:
        suffix = "test_config_unseen" if self.protocol == "scratch" else f"train_config_{self.protocol[1:]}"
        folder = CONFIG_ROOT / "config/exp_configs"
        return (
            Path(self.human_dataset_file or folder / f"human_{suffix}.json"),
            Path(self.sim_dataset_file or folder / f"sim_{suffix}.json"),
        )

    def selections(self) -> tuple[list[dict], list[dict]]:
        human_file, sim_file = self.dataset_files()
        humans, robots = json.loads(human_file.read_text()), json.loads(sim_file.read_text())
        mapping = json.loads(Path(self.task_mapping_file).read_text())["task_mappings"]
        pairing = {s: m["human_task_id"] for m in mapping for s in m["sim_task_id"]}
        if self.task:
            robots = [r for r in robots if r["repo_id"] == self.task]
            if not robots:
                raise ValueError(f"Task is absent from protocol {self.protocol}: {self.task}")
            used = {pairing[r["repo_id"]] for r in robots}
            humans = [r for r in humans if r["repo_id"] in used]
        for rows in [humans, robots]:
            if not rows or len({r["repo_id"] for r in rows}) != len(rows):
                raise ValueError("Empty or duplicate dataset selection")
            for row in rows:
                ids = selected_episodes(row["train"])
                if self.episode_count is not None:
                    # A named single-task pilot explicitly chooses its episode
                    # count (e.g.50 vs Scratch's10). Full benchmark protocols
                    # never expand their declared split. make_records verifies
                    # every requested episode exists before VAE/training.
                    if self.episode_count <= 0 or (self.episode_count > len(ids) and not self.task):
                        raise ValueError("episode_count exceeds the configured training split")
                    row["train"] = f"{ids[0]}:{ids[0] + self.episode_count}"
        human_ids = {r["repo_id"] for r in humans}
        for row in robots:
            if pairing.get(row["repo_id"]) not in human_ids:
                raise ValueError(f"Robot/human task mapping is incomplete: {row['repo_id']}")
        return humans, robots

    def selection_identity(self) -> str:
        return hashlib.sha256(json.dumps(self.selections(), sort_keys=True).encode()).hexdigest()

    def cache_request_identity(self) -> str:
        """Identify source selection and encoding settings, independent of training."""
        request = {
            "selection": self.selection_identity(),
            "model_path": str(Path(self.model_path).expanduser().resolve()),
            "human_root": str(Path(self.human_root).expanduser().resolve()),
            "sim_root": str(Path(self.sim_root).expanduser().resolve()),
            "task_mapping": hashlib.sha256(Path(self.task_mapping_file).read_bytes()).hexdigest(),
            "task_mapping_path": str(Path(self.task_mapping_file).expanduser().resolve()),
            "record_schema": 3,
            "camera": self.camera,
            "image_size": self.image_size,
            "human_fps": self.human_fps,
            "robot_frame_stride": self.robot_frame_stride,
            "encoding": "causal_full_episode_first1_then4_mu_mean_std_v4",
        }
        request["robot_action_tail_policy"] = "all_actions_v1"
        if (self.robot_width, self.human_height, self.human_width) != (224, 224, 224):
            request["resize_hw"] = {"sim": [self.image_size, self.robot_width],
                                    "human": [self.human_height, self.human_width]}
        return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()

    def code_resources(self) -> dict:
        """Portable defaults and immutable custom JSON resources for checkpoints."""
        resources = {}
        for name, relative in CODE_RESOURCES.items():
            path = Path(getattr(self, name)).expanduser().resolve()
            raw = path.read_bytes()
            descriptor = {"sha256": hashlib.sha256(raw).hexdigest()}
            if path == (CONFIG_ROOT / relative).resolve():
                descriptor.update(kind="repository", relative_path=str(path.relative_to(REPO_ROOT)))
            else:
                descriptor.update(kind="custom", original_path=str(path), text=raw.decode("utf-8"))
            resources[name] = descriptor
        return resources

    def manifest_path(self) -> Path:
        return (
            Path(self.manifest)
            if self.manifest
            else Path(self.cache_dir) / "manifests" / f"{self.protocol}_{self.cache_request_identity()[:16]}.json"
        )

    def validate(self, *, training: bool = False) -> None:
        if self.lr_schedule == "cosine" and (
            self.lr_decay_steps < 0 or self.lr_decay_start_step < 0
            or not 0 <= self.min_learning_rate <= self.learning_rate
        ):
            raise ValueError("Cosine LR requires nonnegative decay steps (0 = full training horizon), nonnegative start, and min LR <= base LR")
        if self.action_channel_offset != 14:
            raise ValueError("Joint channels are14–27")
        if training and not self.normalization_path:
            raise ValueError("Episode-relative coordinates require their train-split statistics")
        for name in ["model_path", "human_root", "sim_root", "cache_dir"]:
            value = getattr(self, name)
            if not value:
                raise ValueError(f"--{name.replace('_', '-')} is required")
            outside_repository(value, name)
        if training:
            for name in ["run_dir", "empty_text_embedding"]:
                if not getattr(self, name):
                    raise ValueError(f"--{name.replace('_', '-')} is required")
                outside_repository(getattr(self, name), name)
        if self.resume_from and self.initialize_from:
            raise ValueError("Resume and new-stage initialization are separate operations")
        if (self.train_scope, self.sampling_mode, self.action_history_mode, self.action_representation) != (
            "joint", "episode", "zero_history_supervised_v1", "relative_episode"
        ) or not self.enable_mcp:
            raise ValueError("Use the joint/MCP complete-episode recipe with relative joints and supervised zero prefix")
        if (self.action_per_frame, self.robot_frame_stride, self.frame_chunk_size) != (16, 4, 2):
            raise ValueError("Robot cadence is16 controls/latent, RGB stride4, inference chunk2")
        if training:
            world = int(os.environ.get("WORLD_SIZE", "4"))
            if self.batch_size != 1 or self.effective_batch_size != world:
                raise ValueError("One episode/rank, no accumulation; effective batch = world size")
        if self.lora_rank <= 0 or (self.warmup_steps is not None and self.warmup_steps < 0):
            raise ValueError("LoRA rank must be positive; warmup steps nonnegative")
        if (
            min(
                self.batch_size,
                self.effective_batch_size,
                self.total_epochs,
                self.save_epoch_freq,
                self.save_step_freq,
                self.save_seconds,
            )
            <= 0
        ):
            raise ValueError("Batch, epoch and checkpoint settings must be positive")
        if self.num_workers < 0:
            raise ValueError("Worker count must be nonnegative")
        if any(not math.isfinite(v) or v <= 0 for v in [self.video_snr_shift, self.action_snr_shift]):
            raise ValueError("Video/action SNR shifts must be finite and positive")
        if not isinstance(self.icl_rope_h, int) or self.icl_rope_h < 0:
            raise ValueError("icl_rope_h must be a nonnegative integer")
        self.selections()


def selected_episodes(selection: str) -> list[int]:
    start, stop = map(int, selection.split(":"))
    if not 0 <= start < stop:
        raise ValueError(f"Invalid episode range: {selection}")
    return list(range(start, stop))
