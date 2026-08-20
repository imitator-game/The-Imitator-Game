from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gc
import json
import logging
import os
import random
import time
from collections import defaultdict, deque
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import flax.nnx as nnx
import flax.traverse_util
from flax.training import common_utils
import jax
import jax.numpy as jnp
# NOTE: mani_skill imports moved to lazy inside setup_eval()
import gymnasium as gym
import numpy as np
import optax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoProcessor

import examples.baselines.pi.src.openpi.shared.array_typing as at
import examples.baselines.pi.src.openpi.shared.nnx_utils as nnx_utils
from examples.baselines.pi.src.openpi.models import model as model_lib
from examples.baselines.pi.src.openpi.models.pi0_config import Pi0Config
from examples.baselines.pi.src.openpi.training import optimizer as openpi_optimizer
from examples.baselines.pi.src.openpi.training import utils as training_utils
# NOTE: HumanVideoSimEvaluateProcessor imported lazily inside setup_eval()
from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanRobotPairedDataset,
    HumanSimPairedDataset,
    PairedDatasetConfig,
)
from examples.baselines.lerobot_dataset.trajectory_metrics import EpisodeActionBuffer
from examples.baselines.pi.utils.utils import worker_init_fn

try:
    import wandb
except ImportError:
    wandb = None


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)


def extract_base_env_name(env_id: str) -> str:
    env_name = os.path.basename(env_id)
    if env_name.startswith("L") and "_" in env_name:
        parts = env_name.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2"]:
            return parts[1]
        if len(parts) == 2 and parts[0] == "L3":
            base_name = parts[1]
            version_idx = base_name.rfind("-v")
            if version_idx != -1:
                return f"{base_name[:version_idx]}L3{base_name[version_idx:]}"
            return base_name
    return env_id


def extract_level(env_id: str) -> str:
    env_name = os.path.basename(env_id)
    if env_name.startswith("L") and "_" in env_name:
        parts = env_name.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            return parts[0]
    if env_name.rsplit("-v", 1)[0].endswith("L3"):
        return "L3"
    return "L0"


def _clean_prompt(text: str) -> str:
    return text.strip().replace("_", " ").replace("\n", " ")


def _format_pi05_prompt(text: str, state: Any) -> str:
    state_np = _torch_to_numpy(state).astype(np.float32).reshape(-1)
    discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(str(int(x)) for x in discretized_state)
    return f"Task: {_clean_prompt(text)}, State: {state_str};\nAction: "


def tokenize_pi_prompts(
    processor: AutoProcessor,
    prompts: list[str],
    *,
    max_token_len: int,
    pi05: bool,
    states: Any = None,
    return_tensors: str = "np",
):
    if pi05:
        if states is None:
            raise ValueError("Pi05 prompt tokenization requires normalized state.")
        states_np = _torch_to_numpy(states).astype(np.float32)
        if states_np.ndim == 1:
            states_np = states_np[None]
        if len(prompts) != states_np.shape[0]:
            raise ValueError(f"Prompt/state batch mismatch: {len(prompts)} prompts vs {states_np.shape[0]} states.")
        prompts = [_format_pi05_prompt(prompt, state) for prompt, state in zip(prompts, states_np)]
    else:
        prompts = [_clean_prompt(prompt) for prompt in prompts]

    return processor.tokenizer(
        prompts,
        max_length=max_token_len,
        padding="max_length",
        truncation=True,
        return_tensors=return_tensors,
    )


def set_l_level(level: str):
    from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils

    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)

    if level == "L1":
        L0_L3_utils.set_l1_enabled(True)
    elif level == "L2":
        L0_L3_utils.set_l2_enabled(True)
    elif level == "L3":
        L0_L3_utils.set_l3_enabled(True)


def make_eval_envs(*args, **kwargs):
    from examples.baselines.pi.utils.make_env import make_eval_envs as _make_eval_envs

    return _make_eval_envs(*args, **kwargs)


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, rgb=True, depth=True, state=True) -> None:
        self.base_env = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.transforms = T.Compose([T.Resize((224, 224), antialias=True)])
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images_rgb = []
        images_depth = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                resized_rgb = self.transforms(cam_data["rgb"].permute(0, 3, 1, 2))
                images_rgb.append(resized_rgb)
            if self.include_depth:
                depth = (cam_data["depth"].to(torch.float32) / 1024).to(torch.float16)
                resized_depth = self.transforms(depth.permute(0, 3, 1, 2))
                images_depth.append(resized_depth)

        rgb = torch.stack(images_rgb, dim=1)
        if self.include_depth:
            depth = torch.stack(images_depth, dim=1)

        from mani_skill.utils import common

        observation = common.flatten_state_dict(observation, use_torch=True)
        ret = {}
        if self.include_state:
            ret["state"] = observation
        if self.include_rgb:
            ret["rgb"] = rgb
        if self.include_depth:
            ret["depth"] = depth
        return ret


class _LeRobotJaxDatasetMixin:
    def _init_jax_dataset(
        self,
        processor: AutoProcessor | None = None,
        max_token_len: int = 200,
        pi05: bool = False,
    ):
        self.processor = processor
        self.max_token_len = max_token_len
        self.pi05 = pi05

    def _tokenize_language(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokenize_pi_prompts(
            self.processor,
            [text],
            max_token_len=self.max_token_len,
            pi05=False,
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0).bool()

    def _tokenize_prompt(self, text: str, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_state = state[-1] if state.ndim > 1 else state
        tokens = tokenize_pi_prompts(
            self.processor,
            [text],
            max_token_len=self.max_token_len,
            pi05=self.pi05,
            states=prompt_state if self.pi05 else None,
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0).bool()

    @staticmethod
    def _choose_description(desc):
        if isinstance(desc, list) and desc:
            return desc[random.randint(0, len(desc) - 1)]
        return desc

    def _convert_paired_sample(self, idx, sample):
        obs = sample["robot_obs"]
        state = obs["states"]
        actions = sample["robot_actions"]
        target_task_id = sample.get("sim_task_id") or sample.get("robot_task_id") or ""
        language = sample.get("language")
        if not language:
            logging.warning(
                "Missing paired language for idx=%s target_task_id=%s; falling back to target task id/default prompt.",
                idx,
                target_task_id or "unknown",
            )
            language = target_task_id or "Complete the task."

        cameras = {}
        for i in range(1, len(self.config.cameras) + 1):
            cameras[f"view_{i}"] = obs[f"view_{i}"]

        return {
            "camera_images": cameras,
            "state": state,
            "actions": actions,
            "language": language,
            "target_task_id": target_task_id,
            "sim_task_id": sample.get("sim_task_id", ""),
            "robot_task_id": sample.get("robot_task_id", ""),
        }


class LeRobotJaxDataset(_LeRobotJaxDatasetMixin, HumanSimPairedDataset):
    def __init__(
        self,
        config: PairedDatasetConfig,
        processor: AutoProcessor | None = None,
        max_token_len: int = 200,
        pi05: bool = False,
    ):
        super().__init__(config)
        self._init_jax_dataset(processor, max_token_len, pi05)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        return self._convert_paired_sample(idx, sample)


class LeRobotRobotJaxDataset(_LeRobotJaxDatasetMixin, HumanRobotPairedDataset):
    def __init__(
        self,
        config: PairedDatasetConfig,
        processor: AutoProcessor | None = None,
        max_token_len: int = 200,
        pi05: bool = False,
    ):
        super().__init__(config)
        self._init_jax_dataset(processor, max_token_len, pi05)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        human_task_id = sample.get("human_task_id")
        robot_task_id = sample.get("robot_task_id")
        if human_task_id and robot_task_id:
            language = self._choose_description(self.task_mapper.human_descriptions.get(human_task_id))
            if language is None:
                language = self._choose_description(self.task_mapper.robot_descriptions.get(robot_task_id))
                if language is not None:
                    logging.warning(
                        "Missing human description for human_task_id=%s robot_task_id=%s; falling back to robot description.",
                        human_task_id,
                        robot_task_id,
                    )
            if language is None:
                language = f"Task {human_task_id}"
                logging.warning(
                    "Missing human and robot descriptions for human_task_id=%s robot_task_id=%s; falling back to default task prompt.",
                    human_task_id,
                    robot_task_id,
                )
            sample["language"] = language
        return self._convert_paired_sample(idx, sample)


def collate_fn(
    batch,
    *,
    processor: AutoProcessor,
    max_token_len: int,
    pi05: bool,
):
    all_camera_keys = set()
    for item in batch:
        all_camera_keys.update(item["camera_images"].keys())

    camera_images = {}
    for key in sorted(
        all_camera_keys,
        key=lambda item: (0, int(item.split("_")[1])) if item.startswith("view_") else (1, item),
    ):
        camera_images[key] = torch.stack([item["camera_images"][key] for item in batch])

    state = torch.stack([item["state"] for item in batch])
    languages = [item["language"] for item in batch]
    prompt_states = state[:, -1] if pi05 and state.ndim > 2 else state
    tokens = tokenize_pi_prompts(
        processor,
        languages,
        max_token_len=max_token_len,
        pi05=pi05,
        states=prompt_states if pi05 else None,
        return_tensors="pt",
    )

    return {
        "camera_images": camera_images,
        "state": state,
        "actions": torch.stack([item["actions"] for item in batch]),
        "tokenized_prompt": tokens["input_ids"],
        "tokenized_prompt_mask": tokens["attention_mask"].bool(),
        "language": languages,
        "target_task_id": [item["target_task_id"] for item in batch],
        "sim_task_id": [item["sim_task_id"] for item in batch],
        "robot_task_id": [item["robot_task_id"] for item in batch],
    }


def _torch_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def convert_lerobot_to_jax_observation(
    camera_images: Dict[str, torch.Tensor],
    state: torch.Tensor,
    tokenized_prompt: torch.Tensor,
    tokenized_prompt_mask: torch.Tensor,
    *,
    pad_missing_cameras: bool = True,
) -> model_lib.Observation:
    batch_size = state.shape[0]
    images = {}
    image_masks = {}

    for cam_name, img in camera_images.items():
        assert img.ndim == 5, f"Expected 5D tensor for images, got {img.ndim}D"
        frame = _torch_to_numpy(img[:, -1]).astype(np.float32)
        if frame.shape[1] in (3, 4):
            frame = np.transpose(frame[:, :3], (0, 2, 3, 1))
        elif frame.shape[-1] > 3:
            frame = frame[..., :3]

        if frame.max() <= 1.0:
            frame = frame * 2.0 - 1.0
        else:
            frame = frame / 255.0 * 2.0 - 1.0

        images[cam_name] = frame.astype(np.float32)
        image_masks[cam_name] = np.ones(batch_size, dtype=bool)

    images, image_masks = canonicalize_pi_images(images, image_masks, pad_missing=pad_missing_cameras)

    obs_dict = {
        "image": images,
        "image_mask": image_masks,
        "state": _torch_to_numpy(state[:, -1]).astype(np.float32),
        "tokenized_prompt": _torch_to_numpy(tokenized_prompt).astype(np.int32),
        "tokenized_prompt_mask": _torch_to_numpy(tokenized_prompt_mask).astype(bool),
    }
    return model_lib.Observation.from_dict(obs_dict)


def convert_lerobot_batch_to_jax(batch, *, pad_missing_cameras: bool = True):
    observation = convert_lerobot_to_jax_observation(
        camera_images=batch["camera_images"],
        state=batch["state"],
        tokenized_prompt=batch["tokenized_prompt"],
        tokenized_prompt_mask=batch["tokenized_prompt_mask"],
        pad_missing_cameras=pad_missing_cameras,
    )
    actions = jnp.asarray(_torch_to_numpy(batch["actions"]).astype(np.float32))
    return observation, actions


def resolve_params_path(pretrained_model_path: Optional[str]) -> Optional[Path]:
    if not pretrained_model_path:
        return None
    path = Path(pretrained_model_path)
    if path.is_file():
        if path.name == "model.safetensors":
            raise ValueError("JAX training does not support loading PyTorch model.safetensors checkpoints.")
        return path
    if (path / "model.safetensors").exists():
        raise ValueError("JAX training does not support loading PyTorch model.safetensors checkpoints.")
    if (path / "params").exists():
        return path / "params"
    return path


def detect_pi05_from_params(params_path: Optional[Path], default_pi05: bool) -> bool:
    if params_path is None:
        return default_pi05
    params = model_lib.restore_params(params_path, restore_type=np.ndarray)
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    return any("time_mlp_in" in k or "time_mlp_out" in k for k in flat)


def detect_pretrained_action_dim(params_path: Optional[Path]) -> Optional[int]:
    if params_path is None:
        return None
    params = model_lib.restore_params(params_path, restore_type=np.ndarray)
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    if "action_in_proj/kernel" not in flat:
        return None
    return int(flat["action_in_proj/kernel"].shape[0])


def load_partial_params(params_template: dict, params_path: Optional[Path]) -> tuple[Optional[dict], Optional[int]]:
    if params_path is None:
        return None, None

    loaded = model_lib.restore_params(params_path, restore_type=np.ndarray)
    flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
    flat_ref = flax.traverse_util.flatten_dict(params_template, sep="/")

    merged = {}
    for key, value in flat_loaded.items():
        if key not in flat_ref:
            continue
        if getattr(value, "shape", None) != getattr(flat_ref[key], "shape", None):
            continue
        merged[key] = value.astype(flat_ref[key].dtype) if value.dtype != flat_ref[key].dtype else value

    pretrained_action_dim = None
    if "action_in_proj/kernel" in flat_loaded:
        pretrained_action_dim = flat_loaded["action_in_proj/kernel"].shape[0]

    merged_tree = flax.traverse_util.unflatten_dict(merged, sep="/")
    return merged_tree, pretrained_action_dim


def _path_regex(pattern: str):
    return nnx_utils.PathRegex(pattern)


def _action_expert_filter():
    return _path_regex(r"PaliGemma/llm/.*_1.*")


def _action_dim_projection_filter(pi05: bool):
    heads = ["action_in_proj", "action_out_proj"]
    if not pi05:
        heads.append("state_proj")
    return _path_regex(rf"({'|'.join(heads)})/.*")


def _model_head_filter(pi05: bool):
    heads = ["action_in_proj", "action_out_proj"]
    if pi05:
        heads.extend(["time_mlp_in", "time_mlp_out"])
    else:
        heads.extend(["state_proj", "action_time_mlp_in", "action_time_mlp_out"])
    return _path_regex(rf"({'|'.join(heads)})/.*")


def _lora_filter():
    return _path_regex(r".*lora.*")


def build_trainable_filter(
    use_lora: bool,
    projections_trainable: bool,
    trainable_scope: str = "lora_only",
    paligemma_top_n_layers: int = 0,
    pi05: bool = False,
):
    if not use_lora or trainable_scope == "all":
        return nnx.Param

    filters = []

    if trainable_scope == "lora_only":
        filters.append(_lora_filter())

    elif trainable_scope in ("action_expert_full", "action_expert_and_paligemma_top"):
        filters.extend([_action_expert_filter(), _model_head_filter(pi05)])

        if trainable_scope == "action_expert_and_paligemma_top" and paligemma_top_n_layers > 0:
            logging.warning(
                "PaliGemma top-%d layer path filtering is not supported because Gemma layers are scanned into "
                "single tensors. Training action expert + projection/time heads only.",
                paligemma_top_n_layers,
            )

    else:
        filters.append(_lora_filter())

    if projections_trainable:
        filters.append(_action_dim_projection_filter(pi05))

    if not filters:
        filters.append(_lora_filter())

    return nnx.All(nnx.Param, nnx.Any(*filters))


def build_model_config(args, pi05: bool) -> Pi0Config:
    paligemma_variant = "gemma_2b"
    action_expert_variant = "gemma_300m_lora" if args.use_lora else "gemma_300m"
    if len(args.cameras) > len(model_lib.IMAGE_KEYS):
        raise ValueError(f"Pi supports at most {len(model_lib.IMAGE_KEYS)} image keys, got cameras={args.cameras}")
    image_keys = (
        tuple(model_lib.IMAGE_KEYS[: len(args.cameras)])
        if args.skip_masked_cameras
        else tuple(model_lib.IMAGE_KEYS)
    )
    return Pi0Config(
        dtype=args.precision,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
        action_dim=args.action_dim,
        action_horizon=args.pred_horizon,
        max_token_len=args.max_token_len if pi05 else 48,
        pi05=pi05,
        image_keys=image_keys,
        use_prefix_kv_cache=args.use_prefix_kv_cache,
    )


def build_lr_schedule_config(args, total_train_steps: int) -> openpi_optimizer.CosineDecaySchedule:
    return openpi_optimizer.CosineDecaySchedule(
        warmup_steps=args.warmup_steps,
        peak_lr=args.learning_rate,
        decay_steps=total_train_steps,
        decay_lr=max(args.learning_rate * 0.1, 1e-6),
    )


def canonicalize_pi_images(
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    *,
    pad_missing: bool = True,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    canonical_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    source_keys = list(images.keys())
    if not source_keys:
        raise ValueError("No camera images were provided to build a Pi observation.")

    first_image = images[source_keys[0]]
    batch_size = first_image.shape[0]
    zero_image = np.zeros_like(first_image)
    false_mask = np.zeros(batch_size, dtype=bool)

    mapped_images: Dict[str, np.ndarray] = {}
    mapped_masks: Dict[str, np.ndarray] = {}
    target_keys = canonical_keys if pad_missing else canonical_keys[: min(len(source_keys), len(canonical_keys))]
    for index, canonical_key in enumerate(target_keys):
        if index < len(source_keys):
            source_key = source_keys[index]
            mapped_images[canonical_key] = images[source_key]
            mapped_masks[canonical_key] = masks[source_key]
        else:
            mapped_images[canonical_key] = zero_image.copy()
            mapped_masks[canonical_key] = false_mask.copy()
    return mapped_images, mapped_masks


class CallbackHandler(ocp.AsyncCheckpointHandler):
    def save(self, directory: Path, args):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: Path, args):
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Any


def initialize_checkpoint_dir(
    checkpoint_dir: str,
    *,
    overwrite: bool,
    resume: bool,
    keep_all_checkpoints: bool,
    use_ocdbt: bool,
    enable_async_checkpointing: bool,
    cleanup_tmp_directories: bool,
    async_timeout_secs: int,
):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            import shutil

            shutil.rmtree(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info("Wiped checkpoint directory %s", checkpoint_dir)
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manager = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "train_state": ocp.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt),
            "params": ocp.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None if keep_all_checkpoints else 1,
            create=False,
            cleanup_tmp_directories=cleanup_tmp_directories,
            enable_async_checkpointing=enable_async_checkpointing,
            async_options=(
                ocp.AsyncOptions(timeout_secs=async_timeout_secs)
                if enable_async_checkpointing
                else None
            ),
        ),
    )
    if resuming and tuple(manager.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain restorable checkpoints. Disabling resume.")
        resuming = False
    return manager, resuming


def _checkpoint_train_state(state: training_utils.TrainState) -> dict:
    return {
        "step": state.step,
        "opt_state": state.opt_state,
    }


def _restore_checkpoint_train_state(state: training_utils.TrainState, restored: dict) -> training_utils.TrainState:
    return dataclasses.replace(
        state,
        step=restored["step"],
        opt_state=restored["opt_state"],
    )


def _split_params(state: training_utils.TrainState):
    if state.ema_params is not None:
        params = state.ema_params
    else:
        params = state.params
    train_state = _checkpoint_train_state(state)
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict):
    if train_state.ema_params is not None:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])


def save_state(checkpoint_manager: ocp.CheckpointManager, state: training_utils.TrainState, step: int):
    train_state, params = _split_params(state)
    checkpoint_manager.save(
        step,
        {
            "train_state": train_state,
            "params": {"params": params},
        },
    )


def restore_state(checkpoint_manager: ocp.CheckpointManager, state: training_utils.TrainState):
    train_state_template, params = _split_params(state)
    restored = checkpoint_manager.restore(
        None,
        items={
            "train_state": train_state_template,
            "params": {"params": params},
        },
    )
    state = _restore_checkpoint_train_state(state, restored["train_state"])
    return _merge_params(state, restored["params"])


def configure_tensorstore_checkpoint_io(
    *,
    disable_file_locking: bool,
):
    if not disable_file_locking:
        return
    try:
        from orbax.checkpoint._src.serialization import serialization as orbax_serialization
        from orbax.checkpoint._src.serialization import tensorstore_utils as ts_utils
    except Exception as exc:
        logging.warning("Could not import Orbax TensorStore internals for checkpoint I/O tuning: %s", exc)
        return

    def patch_context(context: dict):
        context["file_io_locking"] = {"mode": "none"}

    for context_name in ("_BASE_TS_CONTEXT", "_DEFAULT_OCDBT_TS_CONTEXT"):
        context = getattr(ts_utils, context_name, None)
        if isinstance(context, dict):
            patch_context(context)

    # Some older Orbax paths use this module-level context directly.
    orbax_serialization.TS_CONTEXT = ts_utils.get_ts_context(use_ocdbt=False)
    logging.info("Configured TensorStore checkpoint I/O: file_locking=disabled")


def count_params(state: nnx.State, filter_spec) -> int:
    count = 0
    for _, value in state.filter(filter_spec).flat_state().items():
        count += int(value.value.size)
    return count


def _filter_stats(state: nnx.State, filter_spec) -> tuple[int, int]:
    filtered_state = state.filter(filter_spec).flat_state()
    return len(filtered_state), sum(int(value.value.size) for value in filtered_state.values())


def validate_trainable_filter(state: nnx.State, trainable_filter, trainable_scope: str, pi05: bool):
    trainable_tensors, trainable_params = _filter_stats(state, trainable_filter)
    if trainable_tensors == 0 or trainable_params == 0:
        raise RuntimeError(f"Trainable filter for scope '{trainable_scope}' matched no parameters.")

    if trainable_scope in ("action_expert_full", "action_expert_and_paligemma_top"):
        action_tensors, action_params = _filter_stats(state, nnx.All(trainable_filter, _action_expert_filter()))
        if action_tensors == 0 or action_params == 0:
            raise RuntimeError(
                f"Trainable scope '{trainable_scope}' did not match action expert parameters. "
                "Expected paths like PaliGemma/llm/..._1/..."
            )
        if pi05:
            time_tensors, time_params = _filter_stats(
                state,
                nnx.All(trainable_filter, _path_regex(r"(time_mlp_in|time_mlp_out)/.*")),
            )
            if time_tensors == 0 or time_params == 0:
                raise RuntimeError(f"Pi05 trainable scope '{trainable_scope}' did not match time_mlp_in/out heads.")


def log_trainable_param_groups(state: nnx.State, trainable_filter, pi05: bool):
    groups = {
        "lora": _lora_filter(),
        "action_expert": _action_expert_filter(),
        "model_heads": _model_head_filter(pi05),
        "action_dim_projections": _action_dim_projection_filter(pi05),
    }
    logging.info("Trainable parameter groups:")
    for name, group_filter in groups.items():
        tensors, params = _filter_stats(state, nnx.All(trainable_filter, group_filter))
        logging.info("  %s: tensors=%s params=%s", name, tensors, f"{params:,}")


def log_trainable_params(state: nnx.State, trainable_filter, trainable_scope: str, pi05: bool):
    total_params = count_params(state, nnx.Param)
    trainable_state = state.filter(trainable_filter).flat_state()
    trainable_params = sum(int(value.value.size) for value in trainable_state.values())
    validate_trainable_filter(state, trainable_filter, trainable_scope, pi05)
    logging.info("Total params: %s", f"{total_params:,}")
    logging.info("Trainable params: %s", f"{trainable_params:,}")
    log_trainable_param_groups(state, trainable_filter, pi05)
    logging.info("Trainable parameter tensors:")
    for path, value in sorted(trainable_state.items(), key=lambda item: "/".join(str(x) for x in item[0])):
        path_str = "/".join(str(x) for x in path)
        logging.info("  %s shape=%s dtype=%s", path_str, tuple(value.value.shape), value.value.dtype)


def init_train_state(args, model_config: Pi0Config, params_path: Optional[Path], trainable_filter, tx):
    tx = tx
    
    rng = jax.random.key(args.seed)
    model = model_config.create(rng)
    params = nnx.state(model)
    partial_params, pretrained_action_dim = load_partial_params(params.to_pure_dict(), params_path)
    if partial_params is not None:
        graphdef, state = nnx.split(model)
        state.replace_by_pure_dict(ocp.transform_utils.intersect_trees(state.to_pure_dict(), partial_params))
        model = nnx.merge(graphdef, state)
        params = nnx.state(model)

    state = training_utils.TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(trainable_filter)),
        ema_decay=args.ema_decay,
        ema_params=None if args.ema_decay is None else params,
    )
    return state, pretrained_action_dim


def create_tx(args, total_train_steps: int):
    lr_schedule = build_lr_schedule_config(args, total_train_steps)
    return openpi_optimizer.create_optimizer(
        openpi_optimizer.AdamW(
            weight_decay=args.weight_decay,
            clip_gradient_norm=args.max_grad_norm,
        ),
        lr_schedule,
    )


def create_train_step(trainable_filter, lr_schedule, *, train_image_augmentation: bool = True):
    diff_state = nnx.DiffState(0, trainable_filter)

    def train_step(rng, state, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model, rng, observation, actions):
            chunked_loss = model.compute_loss(rng, observation, actions, train=train_image_augmentation)
            return jnp.mean(chunked_loss), chunked_loss

        observation, actions = batch
        train_rng = jax.random.fold_in(rng, state.step)
        (loss, chunked_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )

        trainable_params = state.params.filter(trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, trainable_params)
        new_trainable_params = optax.apply_updates(trainable_params, updates)

        nnx.update(model, new_trainable_params)
        new_params = nnx.state(model)
        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
        if state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_params,
                ),
            )

        info = {
            "loss": loss,
            "loss_step_0": jnp.mean(chunked_loss[:, 0]),
            "loss_step_mid": jnp.mean(chunked_loss[:, chunked_loss.shape[1] // 2]),
            "loss_step_last": jnp.mean(chunked_loss[:, -1]),
            "grad_norm": optax.global_norm(grads),
            "learning_rate": lr_schedule(state.step),
        }
        return new_state, info

    return jax.jit(train_step, donate_argnums=(1, 2))


class DummyDataLoaderForCheckpoint:
    @dataclasses.dataclass
    class _DataConfig:
        norm_stats: Any = None
        asset_id: Any = None

    def data_config(self):
        return self._DataConfig()


class JaxPiEvaluator:
    def __init__(self, args, evaluate_processor: HumanVideoSimEvaluateProcessor, processor: AutoProcessor):
        self.args = args
        self.evaluate_processor = evaluate_processor
        self.processor = processor
        self.rng = jax.random.key(args.seed + 1)

    def tokenize_language(self, languages: list[str]):
        tokens = tokenize_pi_prompts(
            self.processor,
            languages,
            max_token_len=self.args.max_token_len,
            pi05=False,
            return_tensors="np",
        )
        return tokens["input_ids"].astype(np.int32), tokens["attention_mask"].astype(bool)

    def tokenize_prompt(self, languages: list[str], state: np.ndarray):
        tokens = tokenize_pi_prompts(
            self.processor,
            languages,
            max_token_len=self.args.max_token_len,
            pi05=getattr(self.args, "pi05", False),
            states=state if getattr(self.args, "pi05", False) else None,
            return_tensors="np",
        )
        return tokens["input_ids"].astype(np.int32), tokens["attention_mask"].astype(bool)

    def process_observation(self, obs: Dict, env_id: str, num_envs: int):
        from mani_skill.utils import common

        obs = {k: common.to_tensor(v, device=torch.device("cpu")) for k, v in obs.items()}
        languages = self.evaluate_processor.get_language(env_id, num_envs)
        if not languages:
            logging.warning("Missing evaluation language for env_id=%s; falling back to default prompt.", env_id)
            languages = ["Complete the task."] * num_envs

        rgb = obs["rgb"]
        if rgb.ndim == 6:
            # [B, num_cams, obs_horizon, C, H, W] -> keep the latest observation.
            rgb = rgb[:, :, -1]
        image_dict = {}
        for i in range(rgb.shape[1]):
            img = rgb[:, i]
            if img.ndim == 5:
                img = img[:, -1]
            if img.ndim == 4 and img.shape[1] in (3, 4):
                img = img[:, :3].permute(0, 2, 3, 1)
            image_dict[f"view_{i+1}"] = img

        state = obs["state"]
        if state.ndim == 3:
            # [B, obs_horizon, S] -> keep the latest observation.
            state = state[:, -1]

        state, image_dict = self.evaluate_processor.normalize_state_rgb(state, image_dict, env_id)
        state = state.detach().cpu().numpy().astype(np.float32)
        tokenized_prompt, tokenized_prompt_mask = self.tokenize_prompt(languages, state)
        image_dict = {
            k: ((v.detach().cpu().numpy().astype(np.float32) * 2.0) - 1.0)
            for k, v in image_dict.items()
        }
        image_masks = {k: np.ones(num_envs, dtype=bool) for k in image_dict}
        image_dict, image_masks = canonicalize_pi_images(
            image_dict,
            image_masks,
            pad_missing=not self.args.skip_masked_cameras,
        )

        obs_dict = {
            "image": image_dict,
            "image_mask": image_masks,
            "state": state,
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask,
        }
        return model_lib.Observation.from_dict(obs_dict)

    def get_action(self, model, obs: Dict, env_id: str, num_envs: int, num_diffusion_steps: int = 10) -> np.ndarray:
        if hasattr(model, "eval"):
            model.eval()
        sample_actions = nnx_utils.module_jit(model.sample_actions)
        observation = self.process_observation(obs, env_id, num_envs)
        self.rng, sample_rng = jax.random.split(self.rng)
        actions = sample_actions(sample_rng, observation, num_steps=num_diffusion_steps)
        actions = np.asarray(jax.device_get(actions))
        actions = self.evaluate_processor.denormalize_action(torch.from_numpy(actions), env_id)
        return actions.cpu().numpy()


def evaluate(
    env_id: str,
    n: int,
    model,
    evaluator: JaxPiEvaluator,
    eval_envs,
    sim_backend: str,
    progress_bar: bool = True,
    num_diffusion_steps: int = 10,
    dtw_provider=None,
    traj_metrics=None,
):
    if hasattr(model, "eval"):
        model.eval()

    def _any(value) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(value.any().item())
        return bool(np.asarray(value).any())

    def _bool_array(value, size: int) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=bool)
        if arr.ndim == 0:
            return np.full((size,), bool(arr), dtype=bool)
        return arr.reshape(-1)[:size]

    def _final_success_array(final_info, size: int) -> np.ndarray | None:
        def _extract_success(item):
            if not isinstance(item, dict):
                return None
            for key in ("success", "success_once", "success_at_end"):
                if key in item:
                    return item[key]
            episode = item.get("episode")
            if isinstance(episode, dict):
                for key in ("success", "success_once", "success_at_end"):
                    if key in episode:
                        return episode[key]
            return None

        if isinstance(final_info, dict):
            return _bool_array(_extract_success(final_info), size)
        if isinstance(final_info, (list, tuple)):
            values = []
            found = False
            for item in final_info[:size]:
                success = _extract_success(item)
                if success is None:
                    values.append(False)
                    continue
                arr = _bool_array(success, 1)
                values.append(bool(arr[0]) if arr is not None and len(arr) else False)
                found = True
            if found:
                values.extend([False] * (size - len(values)))
                return np.asarray(values[:size], dtype=bool)
        return None

    num_envs = eval_envs.num_envs
    compute_dtw = dtw_provider is not None and traj_metrics is not None
    gt_traj = None
    action_buf = None
    success_steps = None
    if compute_dtw:
        gt_traj = dtw_provider.sample_gt_trajectory(env_id, seed=None)
        if gt_traj is None:
            logging.warning("No GT trajectory for %s; TSS metrics will be skipped.", env_id)
            compute_dtw = False
        else:
            action_buf = EpisodeActionBuffer(num_envs=num_envs)
            success_steps = np.full((num_envs,), -1, dtype=np.int64)

    if progress_bar:
        pbar = tqdm(total=n, desc=f"Evaluating Pi JAX{' [+TSS]' if compute_dtw else ''}")

    eval_metrics = defaultdict(list)
    obs, info = eval_envs.reset()
    eps_count = 0
    ts = 0
    while eps_count < n:
        action_seq = evaluator.get_action(
            model=model,
            obs=obs,
            env_id=env_id,
            num_envs=num_envs,
            num_diffusion_steps=num_diffusion_steps,
        )

        for i in range(action_seq.shape[1]):
            _action = action_seq[:, i]
            if compute_dtw and action_buf is not None:
                action_buf.append(_action)
            action = {
                "panda_wristcam-0": _action[:, :8],
                "panda_wristcam-1": _action[:, 8:16],
            }
            obs, rew, terminated, truncated, info = eval_envs.step(action)
            ts += 1
            if compute_dtw and success_steps is not None:
                success = _bool_array(info.get("success"), num_envs)
                if success is not None:
                    new_success = (success_steps == -1) & success
                    success_steps[new_success] = ts
            if _any(truncated):
                break

        if _any(truncated):
            if isinstance(info["final_info"], dict):
                for k, v in info["final_info"]["episode"].items():
                    eval_metrics[k].append(v.float().cpu().numpy() if isinstance(v, torch.Tensor) else v)
            else:
                for final_info in info["final_info"]:
                    if final_info is not None and "episode" in final_info:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v.float().cpu().numpy() if isinstance(v, torch.Tensor) else v)

            if compute_dtw and action_buf is not None and success_steps is not None and gt_traj is not None:
                final_success = _final_success_array(info.get("final_info"), num_envs)
                if final_success is not None:
                    success_steps[(success_steps == -1) & final_success] = ts
                pred_trajs = action_buf.get_and_reset()
                gt_len = len(gt_traj)
                for env_i, pred_traj in enumerate(pred_trajs):
                    success_step = int(success_steps[env_i])
                    if success_step >= 2:
                        try:
                            metric = traj_metrics.compute(pred_traj[:success_step], gt_traj)
                            eval_metrics["tss_success"].append(np.array(metric["tss"], dtype=np.float32))
                            eval_metrics["ndtw_success"].append(np.array(metric["ndtw"], dtype=np.float32))
                        except Exception as exc:
                            logging.warning("TSS(success) failed for %s env %s: %s", env_id, env_i, exc)
                    else:
                        pred_trimmed = pred_traj[:gt_len]
                        if len(pred_trimmed) >= 2:
                            try:
                                metric = traj_metrics.compute(pred_trimmed, gt_traj)
                                eval_metrics["tss_fail"].append(np.array(metric["tss"], dtype=np.float32))
                                eval_metrics["ndtw_fail"].append(np.array(metric["ndtw"], dtype=np.float32))
                            except Exception as exc:
                                logging.warning("TSS(fail) failed for %s env %s: %s", env_id, env_i, exc)
                success_steps[:] = -1
                new_gt = dtw_provider.sample_gt_trajectory(env_id, seed=None)
                if new_gt is not None:
                    gt_traj = new_gt

            eps_count += num_envs
            if progress_bar:
                pbar.update(num_envs)
            ts = 0
            obs, info = eval_envs.reset()

    if progress_bar:
        pbar.close()

    return {k: np.stack(v) for k, v in eval_metrics.items() if len(v) > 0}


class PiJaxFineTuner:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
        processor_source = args.processor_name_or_path
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        try:
            self.processor = AutoProcessor.from_pretrained(
                processor_source,
                trust_remote_code=True,
                use_auth_token=token,
                local_files_only=args.processor_local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the PaliGemma processor. "
                f"Requested source: {processor_source}. "
                "If the server cannot reach Hugging Face, pre-download the processor files and pass "
                "--processor_name_or_path /path/to/local/processor together with "
                "--processor_local_files_only."
            ) from exc
        if args.use_wandb and wandb is not None:
            wandb.init(project=args.wandb_project, name=args.exp_name or f"pi_jax_{int(time.time())}", config=vars(args))

        self.params_path = resolve_params_path(args.pretrained_model_path)
        self.checkpoint_is_pi05 = detect_pi05_from_params(self.params_path, args.pi05)
        self.pretrained_action_dim = detect_pretrained_action_dim(self.params_path)
        logging.info("Detected checkpoint/model type: %s", "Pi05" if self.checkpoint_is_pi05 else "Pi0")

        self.setup_data()
        self.steps_per_epoch = max(len(self.dataloader), 1)
        self.total_train_steps = self._resolve_total_train_steps()
        self.setup_model()
        self.setup_checkpointing()
        if self._epoch_eval_enabled() or self._step_eval_enabled():
            self.setup_eval()
        else:
            self.eval_envs = None
            self.evaluator = None

        self.recent_metrics = defaultdict(lambda: deque(maxlen=args.log_interval))
        self._last_eval_step = -1
        self._last_save_step = -1

    def _resolve_total_train_steps(self) -> int:
        if self.args.use_epoch_training:
            if self.args.num_epochs is None:
                raise ValueError("--use_epoch_training requires --num_epochs.")
            return self.steps_per_epoch * self.args.num_epochs
        return self.args.num_train_steps

    def _step_eval_enabled(self) -> bool:
        return (not self.args.use_epoch_training) and self.args.eval_freq > 0 and not getattr(self.args, "no_eval", False)

    def _step_save_enabled(self) -> bool:
        return (not self.args.use_epoch_training) and self.args.save_interval > 0

    def _epoch_eval_enabled(self) -> bool:
        return (
            self.args.use_epoch_training
            and self.args.eval_freq_epochs is not None
            and self.args.eval_freq_epochs > 0
            and not getattr(self.args, "no_eval", False)
        )

    def _epoch_save_enabled(self) -> bool:
        return (
            self.args.use_epoch_training
            and self.args.save_interval_epochs is not None
            and self.args.save_interval_epochs > 0
        )

    def _maybe_run_eval(self):
        step = int(self.train_state.step)
        if step == self._last_eval_step:
            return
        self.evaluate()
        self._last_eval_step = step

    def _maybe_save_checkpoint(self):
        step = int(self.train_state.step)
        if step == self._last_save_step:
            return
        self.save_checkpoint()
        self._last_save_step = step

    def setup_data(self):
        args = self.args
        if args.data_domain == "robot" and not getattr(args, "no_eval", False):
            raise ValueError("Robot-domain Pi JAX training does not support ManiSkill online eval. Use --no_eval.")

        cfg = PairedDatasetConfig(
            human_root=args.human_root,
            robot_root=args.robot_root,
            sim_root=args.sim_root,
            task_mapping_file=args.task_mapping_file,
            human_dataset_file=args.human_dataset_file,
            robot_dataset_file=args.robot_dataset_file,
            sim_dataset_file=args.sim_dataset_file,
            human_task_description_file=args.human_task_desc_file,
            robot_task_description_file=args.robot_task_desc_file,
            sim_task_description_file=args.sim_task_desc_file,
            cameras=list(args.cameras),
            include_depth=args.include_depth,
            horizon=args.pred_horizon,
            state_type=args.robot_state_type if args.data_domain == "robot" else args.sim_state_type,
            video_backend="torchcodec",
            input_mode="language_only",
            enable_augmentation=not args.disable_dataset_augmentation,
        )
        dataset_cls = LeRobotRobotJaxDataset if args.data_domain == "robot" else LeRobotJaxDataset
        logging.info(
            "Building %s paired dataset: human=%s target=%s root=%s video_backend=%s",
            args.data_domain,
            args.human_dataset_file,
            args.robot_dataset_file if args.data_domain == "robot" else args.sim_dataset_file,
            args.robot_root if args.data_domain == "robot" else args.sim_root,
            "torchcodec",
        )
        self.dataset = dataset_cls(
            cfg,
            max_token_len=args.max_token_len,
            pi05=self.checkpoint_is_pi05,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=partial(
                collate_fn,
                processor=self.processor,
                max_token_len=args.max_token_len,
                pi05=self.checkpoint_is_pi05,
            ),
            pin_memory=True,
            persistent_workers=args.persistent_workers if args.num_workers > 0 else False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
            multiprocessing_context=args.dataloader_context if args.num_workers > 0 else None,
        )

    def setup_model(self):
        args = self.args
        params_path = self.params_path
        checkpoint_is_pi05 = self.checkpoint_is_pi05
        pretrained_action_dim = self.pretrained_action_dim

        self.model_config = build_model_config(args, checkpoint_is_pi05)
        self.lr_schedule = build_lr_schedule_config(args, self.total_train_steps).create()
        self.tx = create_tx(args, self.total_train_steps)
        projections_trainable = pretrained_action_dim is not None and pretrained_action_dim != args.action_dim
        self.trainable_filter = build_trainable_filter(
            args.use_lora,
            projections_trainable,
            trainable_scope=getattr(args, "trainable_scope", "lora_only"),
            paligemma_top_n_layers=getattr(args, "paligemma_top_n_layers", 0),
            pi05=checkpoint_is_pi05,
        )
        self.train_state, _ = init_train_state(
            args,
            self.model_config,
            params_path,
            self.trainable_filter,
            self.tx,
        )

        log_trainable_params(
            self.train_state.params,
            self.trainable_filter,
            getattr(args, "trainable_scope", "lora_only"),
            checkpoint_is_pi05,
        )
        self.ptrain_step = create_train_step(
            self.trainable_filter,
            self.lr_schedule,
            train_image_augmentation=not args.disable_jax_image_augmentation,
        )
        self.rng = jax.random.key(args.seed)

    def setup_checkpointing(self):
        configure_tensorstore_checkpoint_io(
            disable_file_locking=self.args.disable_tensorstore_file_locking,
        )
        self.checkpoint_manager, resuming = initialize_checkpoint_dir(
            self.args.output_dir,
            overwrite=self.args.overwrite,
            resume=self.args.resume,
            keep_all_checkpoints=self.args.keep_all_checkpoints,
            use_ocdbt=not self.args.disable_ocdbt_checkpoint,
            enable_async_checkpointing=not self.args.disable_async_checkpointing,
            cleanup_tmp_directories=self.args.cleanup_tmp_checkpoint_dirs,
            async_timeout_secs=self.args.checkpoint_timeout_secs,
        )
        logging.info(
            "Checkpoint config: use_ocdbt=%s, async=%s, cleanup_tmp=%s, timeout_secs=%s",
            not self.args.disable_ocdbt_checkpoint,
            not self.args.disable_async_checkpointing,
            self.args.cleanup_tmp_checkpoint_dirs,
            self.args.checkpoint_timeout_secs,
        )
        if resuming:
            self.train_state = restore_state(self.checkpoint_manager, self.train_state)

    def setup_eval(self):
        from examples.baselines.lerobot_dataset.evaluate_processor import (
            HumanVideoSimEvaluateProcessor,
            HumanVideoSimEvaluateProcessorConfig,
        )
        args = self.args
        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode=args.reward_mode,
            obs_mode=args.obs_mode,
            render_mode="rgb_array",
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            max_episode_steps=args.max_episode_steps,
        )
        base_env_name = extract_base_env_name(args.env_id)
        level = extract_level(args.env_id)
        set_l_level(level)
        logging.info("Eval env mapping: env_id=%s -> gym_env_id=%s level=%s", args.env_id, base_env_name, level)
        self.eval_envs = make_eval_envs(
            base_env_name,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs=dict(obs_horizon=1),
            video_dir=os.path.join(args.output_dir, "videos") if args.capture_video else None,
            wrappers=[partial(FlattenRGBDObservationWrapper, depth=args.include_depth)],
            l_level=level,
        )
        evaluate_processor_config = HumanVideoSimEvaluateProcessorConfig(
            human_root=args.human_root,
            human_split=args.human_split,
            human_dataset_file=args.human_dataset_file,
            human_task_description_file=args.human_task_desc_file,
            human_cameras=args.human_cameras,
            human_fps=args.human_fps,
            human_image_size=args.human_image_size,
            human_num_frames=args.human_num_frames,
            human_sampling_strategy=args.human_sampling_strategy,
            human_max_jitter=args.human_max_jitter,
            sim_root=args.sim_root,
            sim_split=args.sim_split,
            sim_dataset_file=args.sim_dataset_file,
            sim_task_description_file=args.sim_task_desc_file,
            sim_state_type=args.sim_state_type,
            task_mapping_file=args.task_mapping_file,
            vla=args.vla,
        )
        self.evaluator = JaxPiEvaluator(
            args,
            HumanVideoSimEvaluateProcessor(evaluate_processor_config),
            self.processor,
        )

    def current_model(self):
        params = self.train_state.ema_params if self.train_state.ema_params is not None else self.train_state.params
        model = nnx.merge(self.train_state.model_def, params)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def evaluate(self):
        if self.eval_envs is None or self.evaluator is None:
            return
        logging.info("Starting evaluation at step %s", int(self.train_state.step))
        metrics = evaluate(
            env_id=self.args.env_id,
            n=self.args.num_eval_episodes,
            model=self.current_model(),
            evaluator=self.evaluator,
            eval_envs=self.eval_envs,
            sim_backend=self.args.sim_backend,
            num_diffusion_steps=self.args.num_diffusion_steps,
        )
        reduced = {f"eval/{k}": float(np.mean(v)) for k, v in metrics.items()}
        for k, v in reduced.items():
            self.writer.add_scalar(k, v, int(self.train_state.step))
        if self.args.use_wandb and wandb is not None:
            wandb.log(reduced, step=int(self.train_state.step))
        logging.info("Eval metrics: %s", reduced)

    def save_checkpoint(self):
        step = int(self.train_state.step)
        save_state(self.checkpoint_manager, self.train_state, step)

    def log_metrics(self):
        avg_metrics = {}
        for k, v in self.recent_metrics.items():
            if len(v) > 0:
                avg_metrics[k] = float(sum(v) / len(v))
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, int(self.train_state.step))
        if self.args.use_wandb and wandb is not None:
            wandb.log({f"train/{k}": v for k, v in avg_metrics.items()}, step=int(self.train_state.step))
        logging.info("Step %s: %s", int(self.train_state.step), ", ".join(f"{k}={v:.4f}" for k, v in avg_metrics.items()))

    def train(self):
        logging.info(
            "Starting training for %s steps%s",
            self.total_train_steps,
            "" if not self.args.use_epoch_training else f" ({self.args.num_epochs} epochs x {self.steps_per_epoch} steps/epoch)",
        )
        pbar = tqdm(total=self.total_train_steps)
        epoch = 0

        while int(self.train_state.step) < self.total_train_steps:
            for batch in self.dataloader:
                if int(self.train_state.step) >= self.total_train_steps:
                    break

                batch = convert_lerobot_batch_to_jax(batch, pad_missing_cameras=not self.args.skip_masked_cameras)
                self.rng, step_rng = jax.random.split(self.rng)
                self.train_state, info = self.ptrain_step(step_rng, self.train_state, batch)
                info = jax.device_get(info)

                for k, v in info.items():
                    self.recent_metrics[k].append(float(v))

                step = int(self.train_state.step)
                pbar.update(1)
                pbar.set_postfix({"loss": f"{float(info['loss']):.4f}", "epoch": epoch})

                if step % self.args.log_interval == 0:
                    self.log_metrics()
                if self._step_eval_enabled() and step % self.args.eval_freq == 0:
                    self._maybe_run_eval()
                    gc.collect()
                if self._step_save_enabled() and step % self.args.save_interval == 0:
                    self._maybe_save_checkpoint()

            completed_epoch = epoch + 1
            if self._epoch_eval_enabled() and completed_epoch % self.args.eval_freq_epochs == 0:
                self._maybe_run_eval()
                gc.collect()
            if self._epoch_save_enabled() and completed_epoch % self.args.save_interval_epochs == 0:
                self._maybe_save_checkpoint()
            epoch += 1

        if not getattr(self.args, "no_eval", False):
            self._maybe_run_eval()
        self._maybe_save_checkpoint()
        self.checkpoint_manager.wait_until_finished()
        pbar.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Pi JAX fine-tuning on LeRobot paired dataset")
    parser.add_argument("--pretrained_model_path", type=str, required=True)
    parser.add_argument("--processor_name_or_path", type=str, default="google/paligemma-3b-pt-224")
    parser.add_argument("--processor_local_files_only", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./runs/pi_lerobot_jax")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pi05", action="store_true")

    parser.add_argument("--env_id", type=str, default="L0_TwoRobotStirSpoon-v1")
    parser.add_argument("--obs_mode", type=str, default="rgb")
    parser.add_argument("--control_mode", type=str, default="pd_joint_pos")
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--sim_backend", type=str, default="physx_cpu")
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="dense",
        choices=["sparse", "dense", "normalized_dense", "none"],
    )
    parser.add_argument("--shader", default="rt-fast")

    parser.add_argument("--precision", type=str, default="bfloat16")
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--pred_horizon", type=int, default=50)
    parser.add_argument("--max_token_len", type=int, default=200)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--use_epoch_training", action="store_true")
    parser.add_argument("--num_train_steps", type=int, default=400)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dataloader_context",
        type=str,
        default="forkserver",
        choices=["forkserver", "fork", "spawn"],
    )
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--save_interval_epochs", type=int, default=None)
    parser.add_argument("--keep_all_checkpoints", action="store_true")
    parser.add_argument(
        "--disable_async_checkpointing",
        action="store_true",
        help="Disable Orbax async checkpoint writing to reduce filesystem-related checkpoint failures.",
    )
    parser.add_argument(
        "--disable_ocdbt_checkpoint",
        action="store_true",
        help="Disable Orbax OCDBT/TensorStore checkpoint format and use the simpler non-OCDBT layout.",
    )
    parser.add_argument(
        "--cleanup_tmp_checkpoint_dirs",
        action="store_true",
        help="Ask Orbax to clean up stale temporary checkpoint directories on startup.",
    )
    parser.add_argument(
        "--checkpoint_timeout_secs",
        type=int,
        default=7200,
        help="Orbax async checkpoint timeout in seconds when async checkpointing is enabled.",
    )
    parser.add_argument(
        "--disable_tensorstore_file_locking",
        action="store_true",
        help="Disable TensorStore local file locking for checkpoints. Useful on shared filesystems that return ENOLCK.",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="pi_lerobot_jax")

    parser.add_argument("--eval_freq", type=int, default=0)
    parser.add_argument("--eval_freq_epochs", type=int, default=None)
    parser.add_argument(
        "--no_eval", action="store_true", default=True,
        help="Skip ALL evaluation and avoid creating any simulation environment.",
    )
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--num_eval_envs", type=int, default=1)
    parser.add_argument("--num_diffusion_steps", type=int, default=10)
    parser.add_argument("--capture_video", action="store_true")

    parser.add_argument(
        "--data_domain",
        type=str,
        default="sim",
        choices=["sim", "robot"],
        help="Target trajectory domain for human-paired LeRobot training.",
    )
    parser.add_argument("--sim_root", type=str, default="/path/to/sim_root")
    parser.add_argument("--sim_split", type=str, default="train")
    parser.add_argument("--sim_dataset_file", type=str, default="examples/baselines/lerobot_dataset/config/sim_config.json")
    parser.add_argument("--sim_task_desc_file", type=str, default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")
    parser.add_argument("--sim_state_type", type=str, default="qpos")
    parser.add_argument("--cameras", type=str, nargs="+", default=["zed2i"])

    parser.add_argument("--human_root", type=str, default="/path/to/human_root")
    parser.add_argument("--human_split", type=str, default="train")
    parser.add_argument("--human_dataset_file", type=str, default="examples/baselines/lerobot_dataset/config/human_config.json")
    parser.add_argument("--human_task_desc_file", type=str, default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    parser.add_argument("--human_cameras", type=list, default=["zed2i"])
    parser.add_argument("--human_num_frames", type=int, default=10)
    parser.add_argument("--human_sampling_strategy", type=str, default="uniform_jitter")
    parser.add_argument("--human_max_jitter", type=int, default=3)
    parser.add_argument("--human_fps", type=int, default=30)
    parser.add_argument("--human_image_size", type=int, nargs=2, default=[224, 224])

    parser.add_argument("--robot_root", type=str, default="/path/to/robot_root")
    parser.add_argument("--robot_dataset_file", type=str, default="examples/baselines/lerobot_dataset/config/robot_config.json")
    parser.add_argument("--robot_task_desc_file", type=str, default="examples/baselines/lerobot_dataset/task_desc/robot_desc.json")
    parser.add_argument("--robot_state_type", type=str, default="qpos")

    parser.add_argument("--task_mapping_file", type=str, default="examples/baselines/lerobot_dataset/task_mapping.json")
    parser.add_argument("--include_depth", action="store_true", default=False)
    parser.add_argument("--vla", action="store_true", default=False)
    parser.add_argument(
        "--disable_dataset_augmentation",
        action="store_true",
        help="Disable Albumentations in the LeRobot dataset. Pi JAX still applies model-side image augmentation unless disabled separately.",
    )
    parser.add_argument(
        "--disable_jax_image_augmentation",
        action="store_true",
        help="Disable OpenPI/JAX model-side image augmentation inside compute_loss.",
    )
    parser.add_argument(
        "--skip_masked_cameras",
        action="store_true",
        help="Only feed real camera views to Pi instead of padding missing canonical cameras with masked zero images.",
    )
    parser.add_argument(
        "--use_prefix_kv_cache",
        action="store_true",
        help="Train Pi with a prefix prefill/KV-cache pass followed by a suffix action-expert pass.",
    )
    parser.add_argument("--trainable_scope", type=str, default="lora_only", choices=["lora_only", "action_expert_full", "action_expert_and_paligemma_top", "all"])
    parser.add_argument("--paligemma_top_n_layers", type=int, default=6)
    return parser.parse_args()


def main():
    init_logging()
    args = parse_args()
    if args.use_epoch_training:
        if args.num_epochs is None:
            raise ValueError("--use_epoch_training requires --num_epochs.")
        if args.num_train_steps != 400:
            logging.info("--num_train_steps is ignored because --use_epoch_training is set.")
        if args.eval_freq > 0:
            logging.info("--eval_freq is ignored because --use_epoch_training is set.")
        if args.save_interval > 0:
            logging.info("--save_interval is ignored because --use_epoch_training is set.")
    elif args.num_epochs is not None:
        logging.info("--num_epochs is ignored because --use_epoch_training is not set.")
    trainer = PiJaxFineTuner(args)
    trainer.train()


if __name__ == "__main__":
    main()
