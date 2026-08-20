"""
OpenVLA Training Script - Simplified (no video dependency)
Uses the language in the dataset directly as language input
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from gymnasium import spaces

from tqdm import tqdm
from functools import partial
import random
import time
import gc

# PyTorch and HuggingFace
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoProcessor, AutoModelForVision2Seq, AutoConfig, AutoImageProcessor,
    BitsAndBytesConfig, get_scheduler, set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator

# Logging
import wandb

# Image processing
import torchvision.transforms as transforms
from PIL import Image

# ManiSkill imports
import gymnasium as gym
# NOTE: FlattenRGBDObservationWrapper imported lazily inside _setup_eval_envs()

# OpenVLA imports
from examples.baselines.openvla_oft.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from examples.baselines.openvla_oft.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from examples.baselines.openvla_oft.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, \
    PrismaticProcessor
from examples.baselines.openvla_oft.prismatic.vla.action_tokenizer import ActionTokenizer
from examples.baselines.openvla_oft.prismatic.models.action_heads import L1RegressionActionHead, DiffusionActionHead
from examples.baselines.openvla_oft.prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from examples.baselines.openvla_oft.prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from examples.baselines.openvla_oft.prismatic.models.backbones.llm.prompting import PurePromptBuilder
from examples.baselines.openvla_oft.prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, IGNORE_INDEX, \
    ACTION_TOKEN_BEGIN_IDX
from examples.baselines.openvla_oft.prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
    compute_token_accuracy,
    compute_actions_l1_loss,
)
from examples.baselines.openvla_oft.stage_debug import (
    TrainingStageProfiler,
    activate_step_trace,
    get_active_step_trace,
)

# Import RDT's environment utilities
from examples.baselines.openvla_oft.utils.make_env import (
    SelectCameraObservationWrapper,
    configure_tabletop_task_flags,
    extract_base_env_name,
    extract_level,
    parse_optional_bool_flag,
)
# NOTE: make_eval_envs imported lazily inside _setup_eval_envs()
from examples.baselines.openvla_oft.utils.utils import (
    IterationBasedBatchSampler,
    build_state_obs_extractor,
    convert_obs,
    worker_init_fn
)

# Import evaluation utilities
from examples.baselines.openvla_oft.eval_openvla import (
    evaluate_and_save_best,
    _build_evaluate_processor,  # unified interface (required)
    _parse_sim_task_ids,
)
from examples.baselines.openvla_oft.normalizer import ActionNormalizer


# ============ Simplified: remove prompt2task_dict, it is no longer needed ============


def _parse_comma_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    if isinstance(s, (list, tuple)):
        return list(s)
    s = s.strip()
    if s == "":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _resolve_eval_env_spec(args) -> tuple[str, str]:
    sim_task_ids = _parse_sim_task_ids(args.eval_sim_task_ids, args.num_eval_envs, args.env_id)
    primary_id = sim_task_ids[0] if sim_task_ids else args.env_id
    return extract_base_env_name(primary_id), extract_level(primary_id)


def openvla_collate_fn(batch, pad_token_id: int):
    # Cache mode: items have "vla_feat" instead of pixel_values/input_ids/labels
    if "vla_feat" in batch[0]:
        out = {
            "vla_feat": torch.stack([item["vla_feat"] for item in batch]),   # (B, num_tokens, llm_dim)
            "actions":  torch.stack([item["actions"]  for item in batch]),
            "dataset_names": [item["dataset_name"] for item in batch],
        }
        if "proprio" in batch[0]:
            out["proprio"] = torch.stack([item["proprio"] for item in batch])
        return out

    # Normal mode
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    actions = torch.stack([item["actions"] for item in batch])
    dataset_names = [item["dataset_name"] for item in batch]
    proprios = None
    if "proprio" in batch[0]:
        proprios = torch.stack([item["proprio"] for item in batch])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(pad_token_id)

    out = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "actions": actions,
        "dataset_names": dataset_names,
    }
    if proprios is not None:
        out["proprio"] = proprios
    return out


def finalize_args(args):
    """Perform the same derived computations that TrainConfig.__post_init__ did."""
    if getattr(args, "total_epochs", None) is not None:
        args.num_train_epochs = args.total_epochs
        args.use_epoch_training = True

    if getattr(args, "vla_cache_dir", None) is not None:
        # Cache mode bypasses the VLA forward entirely; force-freeze so VLA
        # parameters are never added to the optimizer and no grad state is wasted.
        args.freeze_vla = True

    if getattr(args, "train_action_head_only", False):
        args.freeze_vla = True
        if args.action_mode not in ("l1_regression", "diffusion"):
            raise ValueError("--train_action_head_only requires --action_mode to be l1_regression or diffusion.")
        if args.use_proprio and not getattr(args, "resume_from", None):
            raise ValueError(
                "--train_action_head_only with --use_proprio requires --resume_from so a trained "
                "proprio_projector can be loaded before freezing."
            )
        if args.action_mode == "diffusion" and not getattr(args, "resume_from", None):
            raise ValueError(
                "--train_action_head_only with diffusion requires --resume_from so a trained "
                "noisy_action_projector can be loaded before freezing."
            )

    if getattr(args, "exp_name", None) is None:
        args.exp_name = f"openvla_{args.action_mode}_{args.env_id.split('-')[0]}_{int(time.time())}"
        args.output_dir = f"./runs/{args.exp_name}"

    if getattr(args, "effective_batch_size", None) is not None:
        args.gradient_accumulation_steps = max(1, int(args.effective_batch_size) // int(args.batch_size))
        print(
            f"[Args] Set gradient_accumulation_steps={args.gradient_accumulation_steps} for effective_batch_size={args.effective_batch_size}")

    if getattr(args, "num_workers", 0) == 0:
        args.persistent_workers = False

    # Match original OpenVLA defaults: use all-linear unless user specifies otherwise
    args.lora_target_modules = _parse_comma_list(getattr(args, "lora_target_modules", None)) or "all-linear"
    args.lora_modules_to_save = _parse_comma_list(getattr(args, "lora_modules_to_save", None))
    args.wandb_tags = _parse_comma_list(getattr(args, "wandb_tags", None))

    if isinstance(getattr(args, "image_size", None), (list, tuple)):
        args.image_size = tuple(args.image_size)

    if getattr(args, "use_lerobot", False):
        if not getattr(args, "demo_path", None):
            args.demo_path = ""
    else:
        if not getattr(args, "demo_path", None):
            raise ValueError("--demo_path is required unless --use_lerobot is set")

    if getattr(args, "use_epoch_training", False) and int(args.num_train_epochs) <= 0:
        raise ValueError("--num_train_epochs/--total_epochs must be > 0 when --use_epoch_training is enabled")

    return args


def _uses_hidden_state_l1_route(args) -> bool:
    return (
        getattr(args, "action_mode", None) == "l1_regression"
        and (
            getattr(args, "vla_cache_dir", None) is not None
            or getattr(args, "freeze_vla", False)
        )
    )


def _uses_output_side_proprio_fusion(args) -> bool:
    return _uses_hidden_state_l1_route(args) and getattr(args, "use_proprio", False)


def _describe_training_route(args, use_cache: bool) -> str:
    if use_cache:
        return "cache_vla_hidden_state_output_proprio_l1"
    if _uses_hidden_state_l1_route(args):
        return "freeze_vla_hidden_state_output_proprio_l1"
    if (
        getattr(args, "use_lora", False)
        and getattr(args, "use_proprio", False)
        and getattr(args, "action_mode", None) == "l1_regression"
    ):
        return "lora_input_proprio_l1"
    if getattr(args, "action_mode", None) == "l1_regression":
        return "online_vla_l1"
    if getattr(args, "action_mode", None) == "diffusion":
        return "online_vla_diffusion"
    return f"online_vla_{getattr(args, 'action_mode', 'unknown')}"


def _get_vla_vision_backbone(vla):
    """Return the vision backbone across current OpenVLA wrapper variants."""
    if hasattr(vla, "vision_backbone"):
        return vla.vision_backbone
    if hasattr(vla, "model") and hasattr(vla.model, "vision_backbone"):
        return vla.model.vision_backbone
    raise AttributeError(f"{type(vla).__name__} has no accessible vision_backbone")


def _set_vla_vision_backbone(vla, vision_backbone) -> None:
    """Set the vision backbone across current OpenVLA wrapper variants."""
    if hasattr(vla, "vision_backbone"):
        vla.vision_backbone = vision_backbone
        return
    if hasattr(vla, "model") and hasattr(vla.model, "vision_backbone"):
        vla.model.vision_backbone = vision_backbone
        return
    raise AttributeError(f"{type(vla).__name__} has no accessible vision_backbone")


def _module_param_dtype(module, fallback: torch.dtype) -> torch.dtype:
    if module is None:
        return fallback
    raw_module = getattr(module, "module", module)
    for param in raw_module.parameters():
        if param.is_floating_point():
            return param.dtype
    return fallback


def _is_hf_model_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    expected = (
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any(os.path.exists(os.path.join(path, name)) for name in expected)


# ==========================================================================
# DATASET (simplified - use language directly)
# ==========================================================================

class OpenVLADataset(Dataset):
    """Simplified dataset - uses the language in the dataset directly"""

    def __init__(
            self,
            demo_path: str,
            obs_process_fn,
            obs_space,
            include_rgb: bool,
            include_depth: bool,
            processor: AutoProcessor,
            num_traj: Optional[int] = None,
            pred_horizon: int = 16,
            action_dim: int = 8,
            use_hand_camera: bool = False,
            image_size: Tuple[int, int] = (224, 224),
            action_mode: str = "discrete",
            action_tokenizer: Optional[ActionTokenizer] = None,
            augmentation_config: Optional[Dict] = None,
            default_language: str = "pick red cube and place on plate.",  # default language
    ):
        super().__init__()
        self.demo_path = Path(demo_path)
        self.processor = processor
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.use_hand_camera = use_hand_camera
        self.image_size = image_size
        self.action_mode = action_mode
        self.action_tokenizer = action_tokenizer
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.obs_horizon = 1
        self.default_language = default_language

        # Action normalizer
        self.action_normalizer = ActionNormalizer(
            demo_path=demo_path,
            action_dim=action_dim,
            dataset_name="maniskill_pickcubeycb"
        )

        # ============ Simplified: no longer load lang_descriptions and videos ============

        # Load trajectories
        from examples.baselines.openvla_oft.utils.utils import load_demo_dataset_with_lan
        trajectories = load_demo_dataset_with_lan(demo_path, num_traj=num_traj, concat=False)

        print("Processing observations...")
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = self._reorder_keys(obs_traj_dict, obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.Tensor(_obs_traj_dict["depth"].astype(np.float16))
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"]).to(torch.uint8)
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"]).to(torch.float16)
            obs_traj_dict_list.append(_obs_traj_dict)

        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i]).to(torch.float16)

        # ============ Simplified: use trajectory_language directly ============
        if "language" in trajectories:
            self.trajectory_language = trajectories["language"]
        else:
            # If there is no language field, use the default value
            self.trajectory_language = [self.default_language] * len(trajectories["actions"])
            print(f"[Warning] No language field in dataset, using default: '{self.default_language}'")

        if "delta" in demo_path or "vel" in demo_path:
            self.pad_action = torch.zeros(trajectories["actions"][0].shape[1], dtype=torch.float16)
        else:
            raise NotImplementedError(f"Control mode not supported")

        assert action_mode in ["discrete", "l1_regression", "diffusion"]

        if action_mode == "discrete":
            assert action_tokenizer is not None

        # Create slices
        self.slices = []
        for traj_idx in range(len(trajectories["actions"])):
            L = trajectories["actions"][traj_idx].shape[0]
            pad_before = self.obs_horizon - 1
            pad_after = pred_horizon - self.obs_horizon
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]

        self.trajectories = trajectories
        self.obs_traj_dict_list = obs_traj_dict_list

        gc.collect()
        torch.cuda.empty_cache()

        print(f"Dataset loaded: {len(self.slices)} samples from {len(trajectories['actions'])} trajectories")

    def _reorder_keys(self, d, ref_dict):
        out = dict()
        for k, v in ref_dict.items():
            if k not in ['prompt']:
                if isinstance(v, dict) or isinstance(v, spaces.Dict):
                    out[k] = self._reorder_keys(d[k], ref_dict[k])
                else:
                    out[k] = d[k]
        return out

    def __len__(self):
        return len(self.slices)

    def _build_prompt_and_labels(self, lang_text: str, actions: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """Build the prompt and labels"""
        prompt_builder = PurePromptBuilder("openvla")
        flat_actions = actions.flatten().cpu().float().numpy()
        action_token_str = self.action_tokenizer(flat_actions)

        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang_text.lower()}?"},
            {"from": "gpt", "value": action_token_str},
        ]

        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        prompt = prompt_builder.get_prompt()
        input_ids = self.processor.tokenizer(prompt, add_special_tokens=True).input_ids
        labels = list(input_ids)

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        action_chunk_len = len(action_token_str)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX

        return prompt, input_ids, labels

    def __getitem__(self, index: int) -> Dict:
        traj_idx, start, end = self.slices[index]
        L = self.trajectories["actions"][traj_idx].shape[0]

        # Get observation sequence
        obs_traj = self.obs_traj_dict_list[traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[max(0, start):start + self.obs_horizon]
            if start < 0:
                pad_obs_seq = torch.stack([obs_seq[k][0]] * abs(start), dim=0)
                obs_seq[k] = torch.cat((pad_obs_seq, obs_seq[k]), dim=0)

        # Get RGB image
        rgb_image = obs_seq['rgb'][-1].float() / 255.0
        rgb_np = (rgb_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image = Image.fromarray(rgb_np)

        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.Resampling.BILINEAR)

        # Get action sequence
        act_seq = self.trajectories["actions"][traj_idx][max(0, start):end].float()
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            act_seq = torch.cat([act_seq, self.pad_action.float().repeat(end - L, 1)], dim=0)

        # Normalize actions
        act_seq = self.action_normalizer.normalize(act_seq, method="bounds_q99")

        # ============ Simplified: use the language in the dataset directly ============
        language = self.trajectory_language[traj_idx]

        # If language is empty, use the default value
        if not language or language.strip() == "":
            language = self.default_language

        lang_text = language  # use as-is, no conversion

        # Process image
        pixel_values = self.processor.image_processor.apply_transform(image)

        # Build prompt and labels
        prompt, input_ids, labels = self._build_prompt_and_labels(lang_text, act_seq)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "actions": act_seq,
            "dataset_name": "maniskill",  # simplified dataset_name
            "language": language,  # include the raw language for debugging
        }


class OpenVLATrainer:
    """Memory-optimized trainer"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Defaults; may be overwritten if resuming from checkpoint
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_metrics = defaultdict(float)
        self.evaluate_processor = None
        self.eval_sim_task_ids = None

        set_seed(args.seed)
        if args.torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            if torch.cuda.is_available():
                if not args.disable_tf32:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    if hasattr(torch.backends.cudnn, "allow_tf32"):
                        torch.backends.cudnn.allow_tf32 = True
                    try:
                        torch.set_float32_matmul_precision(args.matmul_precision)
                    except Exception:
                        pass
                torch.backends.cudnn.benchmark = True

        os.makedirs(args.output_dir, exist_ok=True)

        self.model_dtype = self._determine_model_dtype(args)
        print(f"[Trainer] Unified model dtype: {self.model_dtype}")
        if torch.cuda.is_available():
            print(
                f"[Trainer] Perf settings: attn={args.attn_implementation}, "
                f"pin_memory={args.pin_memory}, num_workers={args.num_workers}, "
                f"cleanup_interval={args.cleanup_interval}, "
                f"tf32={'off' if args.disable_tf32 else 'on'}"
            )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
        )
        self.stage_profiler = TrainingStageProfiler(
            output_dir=args.output_dir,
            enabled=getattr(args, "debug_stage_timing", False),
            process_index=self.accelerator.process_index,
            device=self.device,
            args=args,
        )

        self.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
        if args.use_wandb and self.accelerator.is_main_process:
            self._init_wandb()

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        if args.resume_from:
            if args.freeze_vla:
                print(
                    "[Trainer] Attempting optimizer/scheduler resume for frozen-VLA route. "
                    "If the trainable parameter set differs, loading will fall back to a fresh optimizer."
                )
            self._load_optimizer_scheduler(args.resume_from)
        self._setup_eval_env()


        self.recent_metrics = {
            "loss": deque(maxlen=args.gradient_accumulation_steps),
            "curr_action_accuracy": deque(maxlen=args.gradient_accumulation_steps),
            "curr_action_l1_loss": deque(maxlen=args.gradient_accumulation_steps),
            "next_actions_accuracy": deque(maxlen=args.gradient_accumulation_steps),
            "next_actions_l1_loss": deque(maxlen=args.gradient_accumulation_steps),
        }

    def _init_wandb(self):
        wandb.init(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            name=self.args.wandb_run_name or self.args.exp_name,
            config=vars(self.args),
            tags=self.args.wandb_tags,
        )

    def _determine_model_dtype(self, args):
        if args.load_in_8bit:
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32

        if args.mixed_precision == "bf16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            if torch.cuda.is_available():
                return torch.float16
            return torch.float32

        if args.mixed_precision == "fp16":
            if torch.cuda.is_available():
                return torch.float16
            return torch.float32

        return torch.float32

    def _safe_unwrap(self, model):
        if model is None:
            return None
        try:
            return self.accelerator.unwrap_model(model)
        except Exception as exc:
            print(f"[Trainer] unwrap_model failed ({type(exc).__name__}); using raw model.")
            return getattr(model, "module", model)

    def _load_base_vla_for_eval(self):
        """Load the frozen base VLA on-demand for cache-mode evaluation."""
        args = self.args
        quantization_config = None
        if self.use_kbit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        vla = AutoModelForVision2Seq.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.model_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            attn_implementation=args.attn_implementation,
            device_map="auto" if self.use_kbit else None,
            quantization_config=quantization_config,
            local_files_only=True,
        )
        if not self.use_kbit:
            vla = vla.to(self.device)
        vla.eval()
        return vla

    def _setup_model(self):
        """Setup VLA model with memory optimizations"""
        args = self.args
        self.use_kbit = bool(args.load_in_8bit)

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        model_load_path = args.model_name_or_path
        resume_adapter = False
        if args.resume_from:
            adapter_cfg = os.path.join(args.resume_from, "adapter_config.json")
            if os.path.exists(adapter_cfg):
                resume_adapter = True
            elif _is_hf_model_dir(args.resume_from):
                model_load_path = args.resume_from
            else:
                print(
                    "[Trainer] resume_from points to an auxiliary checkpoint directory; "
                    "loading base VLA from --model_name_or_path and restoring "
                    "action head/projectors/training state from --resume_from."
                )

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_load_path,
                trust_remote_code=False,
                local_files_only=True
            )
        except Exception:
            self.processor = AutoProcessor.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=False,
                local_files_only=True
            )

        self.action_tokenizer = ActionTokenizer(
            self.processor.tokenizer,
            bins=args.n_action_bins,
            min_action=args.action_min,
            max_action=args.action_max,
        )

        # ── Cache mode: VLA never called → skip loading 7B weights to save GPU memory ──
        if getattr(args, "vla_cache_dir", None) is not None:
            meta_path = os.path.join(args.vla_cache_dir, "vla_cache_meta.json")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"vla_cache_meta.json not found in {args.vla_cache_dir}. "
                    "Run precompute_vla_features.py first."
                )
            with open(meta_path) as f:
                _cache_meta = json.load(f)
            _llm_dim = _cache_meta["llm_dim"]
            self.vla = None
            print(
                f"[Trainer] Cache mode: skipping VLA load (saves ~14 GB GPU memory). "
                f"llm_dim={_llm_dim} read from cache meta."
            )

            self.action_head = None
            self.noisy_action_projector = None
            self.proprio_projector = None

            if args.action_mode == "l1_regression":
                self.action_head = L1RegressionActionHead(
                    input_dim=_llm_dim,
                    hidden_dim=_llm_dim,
                    action_dim=args.action_dim,
                ).to(self.model_dtype)
            elif args.action_mode == "diffusion":
                raise NotImplementedError(
                    "Diffusion action mode is not supported in VLA cache mode "
                    "(noisy_action_projector is on the VLA input side). "
                    "Use --action_mode l1_regression with --vla_cache_dir."
                )
            else:
                raise ValueError(
                    f"action_mode '{args.action_mode}' is not supported in cache mode. "
                    "Use --action_mode l1_regression."
                )

            if args.use_proprio:
                self.proprio_projector = ProprioProjector(
                    llm_dim=_llm_dim,
                    proprio_dim=args.action_dim,
                ).to(self.model_dtype)

            if args.resume_from:
                self._load_aux_from_checkpoint(args.resume_from)

            print(f"[Trainer] Model setup complete (cache mode). Action mode: {args.action_mode}")
            return
        # ── End cache mode early return ──────────────────────────────────────────────

        quantization_config = None
        if self.use_kbit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        if args.from_scratch:
            config = OpenVLAConfig.from_pretrained(args.model_name_or_path)
            self.vla = OpenVLAForActionPrediction(config).to(self.model_dtype)
        else:
            self.vla = AutoModelForVision2Seq.from_pretrained(
                model_load_path,
                torch_dtype=self.model_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                attn_implementation=args.attn_implementation,
                device_map="auto" if self.use_kbit else None,
                quantization_config=quantization_config,
                local_files_only=True
            )

        if self.use_kbit:
            self.vla = prepare_model_for_kbit_training(
                self.vla, use_gradient_checkpointing=args.use_gradient_checkpointing
            )

        if args.use_lora:
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules,
                modules_to_save=args.lora_modules_to_save,
                task_type=TaskType.CAUSAL_LM,
            )
            if args.resume_from and resume_adapter:
                try:
                    self.vla = PeftModel.from_pretrained(
                        self.vla,
                        args.resume_from,
                        is_trainable=True,
                    )
                except TypeError:
                    self.vla = PeftModel.from_pretrained(self.vla, args.resume_from)
                # Fallback: ensure LoRA params are trainable regardless of PEFT version
                _lora_activated = 0
                for name, param in self.vla.named_parameters():
                    if "lora_" in name and not param.requires_grad:
                        param.requires_grad = True
                        _lora_activated += param.numel()
                if _lora_activated > 0:
                    print(f"[Trainer] Manually enabled grad for {_lora_activated:,} LoRA params")
            else:
                self.vla = get_peft_model(self.vla, lora_config)
            if not args.freeze_vla:
                self.vla.print_trainable_parameters()
            else:
                print("[Trainer] LoRA adapters loaded but VLA will be frozen (--freeze_vla).")
            if args.resume_from and resume_adapter and not args.freeze_vla:
                vla_trainable_params = sum(p.numel() for p in self.vla.parameters() if p.requires_grad)
                if vla_trainable_params == 0:
                    raise RuntimeError(
                        "Resume loaded a LoRA adapter with 0 trainable VLA parameters. "
                        "This would silently disable LoRA fine-tuning; please check the PEFT environment "
                        "or adapter loading path."
                    )

        if args.use_film:
            vision_backbone = _get_vla_vision_backbone(self.vla)
            _set_vla_vision_backbone(
                self.vla,
                FiLMedPrismaticVisionBackbone(
                    vision_backbone=vision_backbone,
                    llm_dim=self.vla.llm_dim,
                ),
            )

        if args.freeze_vision_backbone:
            print("[Memory Opt] Freezing vision backbone...")
            for param in _get_vla_vision_backbone(self.vla).parameters():
                param.requires_grad = False

        if args.freeze_vla:
            print("[Memory Opt] Freezing full VLA...")
            for param in self.vla.parameters():
                param.requires_grad = False

        needs_vla_backward = (
            (not args.freeze_vla)
            or (
                args.use_proprio
                and not args.train_action_head_only
                and not _uses_hidden_state_l1_route(args)
            )
            or (args.action_mode == "diffusion" and not args.train_action_head_only)
        )
        if args.use_gradient_checkpointing and needs_vla_backward:
            print(
                "[Memory Opt] Enabling gradient checkpointing... "
                f"(freeze_vla={args.freeze_vla}, train_action_head_only={args.train_action_head_only}, "
                f"use_proprio={args.use_proprio}, action_mode={args.action_mode})"
            )
            self.vla.gradient_checkpointing_enable()
        elif args.use_gradient_checkpointing:
            print(
                "[Memory Opt] Skipping gradient checkpointing because VLA is frozen and no trainable "
                "input projector needs gradients through the VLA."
            )

        self.action_head = None
        self.noisy_action_projector = None
        self.proprio_projector = None

        if args.action_mode == "l1_regression":
            self.action_head = L1RegressionActionHead(
                input_dim=self.vla.llm_dim,
                hidden_dim=self.vla.llm_dim,
                action_dim=args.action_dim,
            ).to(self.model_dtype)
        elif args.action_mode == "diffusion":
            self.action_head = DiffusionActionHead(
                input_dim=self.vla.llm_dim,
                hidden_dim=self.vla.llm_dim,
                action_dim=args.action_dim,
                num_diffusion_steps_train=args.num_diffusion_steps_train,
            ).to(self.model_dtype)
            self.noisy_action_projector = NoisyActionProjector(
                llm_dim=self.vla.llm_dim,
            ).to(self.model_dtype)

        if args.use_proprio:
            self.proprio_projector = ProprioProjector(
                llm_dim=self.vla.llm_dim,
                proprio_dim=args.action_dim,
            ).to(self.model_dtype)

        if args.train_action_head_only:
            if args.use_proprio:
                proprio_ckpt = os.path.join(args.resume_from, "proprio_projector.pt")
                if not os.path.exists(proprio_ckpt):
                    raise ValueError(
                        "--train_action_head_only with --use_proprio requires a checkpoint containing "
                        f"'proprio_projector.pt', but it was not found in {args.resume_from}."
                    )
            if args.action_mode == "diffusion":
                noisy_ckpt = os.path.join(args.resume_from, "noisy_action_projector.pt")
                if not os.path.exists(noisy_ckpt):
                    raise ValueError(
                        "--train_action_head_only with diffusion requires a checkpoint containing "
                        f"'noisy_action_projector.pt', but it was not found in {args.resume_from}."
                    )

        if args.resume_from:
            self._load_aux_from_checkpoint(args.resume_from)

        if args.train_action_head_only:
            if self.proprio_projector is not None:
                print("[Memory Opt] Freezing proprio_projector (--train_action_head_only)...")
                for param in self.proprio_projector.parameters():
                    param.requires_grad = False
            if self.noisy_action_projector is not None:
                print("[Memory Opt] Freezing noisy_action_projector (--train_action_head_only)...")
                for param in self.noisy_action_projector.parameters():
                    param.requires_grad = False

        print(f"[Trainer] Model setup complete. Action mode: {args.action_mode}")

    @staticmethod
    def _torch_load_resume(path: str):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch versions do not support weights_only.
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _capture_rng_state() -> Dict[str, object]:
        rng_state = {
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return rng_state

    def _restore_rng_state(self, training_state: Dict[str, object]) -> None:
        try:
            if "python_random_state" in training_state:
                random.setstate(training_state["python_random_state"])
            if "numpy_random_state" in training_state:
                np.random.set_state(training_state["numpy_random_state"])
            if "torch_rng_state" in training_state:
                torch.random.set_rng_state(training_state["torch_rng_state"])
            if torch.cuda.is_available() and "torch_cuda_rng_state_all" in training_state:
                torch.cuda.set_rng_state_all(training_state["torch_cuda_rng_state_all"])
            if any(
                key in training_state
                for key in (
                    "python_random_state",
                    "numpy_random_state",
                    "torch_rng_state",
                    "torch_cuda_rng_state_all",
                )
            ):
                print("[Trainer] RNG state restored from checkpoint")
        except Exception as exc:
            print(f"[Trainer] Failed to restore RNG state ({type(exc).__name__}): {exc}")

    def _fallback_progress_from_checkpoint_name(self, checkpoint_dir: str) -> None:
        checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
        try:
            if checkpoint_name.startswith("step_"):
                self.global_step = max(self.global_step, int(checkpoint_name.split("_", 1)[1]))
                print(f"[Trainer] Fallback restored global_step={self.global_step} from checkpoint name")
            elif checkpoint_name.startswith("epoch_"):
                self.current_epoch = max(self.current_epoch, int(checkpoint_name.split("_", 1)[1]))
                print(f"[Trainer] Fallback restored current_epoch={self.current_epoch} from checkpoint name")
        except Exception:
            pass

    def _load_aux_from_checkpoint(self, checkpoint_dir: str) -> None:
        """Load auxiliary modules (action head/projectors) from a checkpoint directory."""
        _aux_loaded: list[str] = []
        _aux_missing: list[str] = []

        if self.action_head is not None:
            path = os.path.join(checkpoint_dir, "action_head.pt")
            if os.path.exists(path):
                self.action_head.load_state_dict(self._torch_load_resume(path))
                _aux_loaded.append("action_head")
            else:
                _aux_missing.append("action_head.pt")
        if self.noisy_action_projector is not None:
            path = os.path.join(checkpoint_dir, "noisy_action_projector.pt")
            if os.path.exists(path):
                self.noisy_action_projector.load_state_dict(self._torch_load_resume(path))
                _aux_loaded.append("noisy_action_projector")
            else:
                _aux_missing.append("noisy_action_projector.pt")
        if self.proprio_projector is not None:
            path = os.path.join(checkpoint_dir, "proprio_projector.pt")
            if os.path.exists(path):
                self.proprio_projector.load_state_dict(self._torch_load_resume(path))
                _aux_loaded.append("proprio_projector")
            else:
                _aux_missing.append("proprio_projector.pt")

        if _aux_loaded:
            print(f"[Trainer] Loaded auxiliary modules from checkpoint: {', '.join(_aux_loaded)}")
        if _aux_missing:
            print(
                f"[WARNING] resume_from={checkpoint_dir} is missing: {', '.join(_aux_missing)}. "
                f"These modules will use RANDOM initialization! Training results may be incorrect."
            )

        state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(state_path):
            try:
                training_state = self._torch_load_resume(state_path)
                self.global_step = int(training_state.get("global_step", 0))
                self.current_epoch = int(training_state.get("current_epoch", training_state.get("epoch", 0)))
                self.best_eval_metrics = defaultdict(float, training_state.get("best_eval_metrics", {}))
                self._restore_rng_state(training_state)
                print(
                    f"[Trainer] Resumed training_state: global_step={self.global_step}, "
                    f"current_epoch={self.current_epoch}"
                )
            except Exception as exc:
                print(f"[Trainer] Failed to load training_state ({type(exc).__name__}): {exc}")
                self._fallback_progress_from_checkpoint_name(checkpoint_dir)
        else:
            self._fallback_progress_from_checkpoint_name(checkpoint_dir)

    def _load_optimizer_scheduler(self, checkpoint_dir: str) -> None:
        """Load optimizer & scheduler state_dict for proper resume."""
        opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            try:
                self.optimizer.load_state_dict(self._torch_load_resume(opt_path))
                print(f"[Trainer] Optimizer state loaded from {opt_path}")
            except Exception as e:
                print(f"[Trainer] Optimizer load failed ({e}), using fresh optimizer")

        sched_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(sched_path):
            try:
                self.scheduler.load_state_dict(self._torch_load_resume(sched_path))
                print(f"[Trainer] Scheduler state loaded from {sched_path}")
            except Exception as e:
                print(f"[Trainer] Scheduler load failed ({e}), using fresh scheduler")

    def _refresh_trainable_handles(self) -> None:
        """Refresh trainable module/parameter handles after accelerator.prepare()."""
        self._trainable_modules = []
        self._trainable_params = []

        module_candidates = [
            self.vla,
            self.action_head,
            self.noisy_action_projector,
            self.proprio_projector,
        ]
        for module in module_candidates:
            if module is None:
                continue
            module_trainable_params = [p for p in module.parameters() if p.requires_grad]
            if not module_trainable_params:
                continue
            self._trainable_modules.append(module)
            self._trainable_params.extend(module_trainable_params)

    def _setup_data(self):
        """Setup training dataset with memory optimizations"""
        args = self.args

        if args.use_lerobot:
            from examples.baselines.openvla_oft.lerobot_dataset import OpenVLAPairedDataset
            from examples.baselines.lerobot_dataset.lerobot_paired_dataset import PairedDatasetConfig

            if args.sim_dataset_file is None:
                raise ValueError("--sim_dataset_file is required when --use_lerobot is set")
            if args.human_dataset_file is None:
                raise ValueError("--human_dataset_file is required when --use_lerobot is set")

            if args.human_split and args.human_split != args.sim_split:
                raise ValueError("--human_split and --sim_split must match for paired training")

            paired_config = PairedDatasetConfig(
                human_root=args.human_root,
                sim_root=args.sim_root,
                task_mapping_file=args.task_mapping_file,
                human_dataset_file=args.human_dataset_file,
                sim_dataset_file=args.sim_dataset_file,
                human_task_description_file=args.human_task_description_file,
                sim_task_description_file=args.sim_task_description_file,
                split=args.sim_split,
                cameras=[args.lerobot_camera],
                include_depth=False,
                image_size=tuple(args.image_size),
                horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                video_backend=args.lerobot_video_backend or "torchcodec",
                input_mode="language_only",
            )

            self.train_dataset = OpenVLAPairedDataset(
                paired_config=paired_config,
                processor=self.processor,
                action_tokenizer=self.action_tokenizer,
                default_language=args.default_language,
                action_dim=args.action_dim,
                pred_horizon=args.pred_horizon,
                image_size=tuple(args.image_size),
                vla_cache_dir=getattr(args, "vla_cache_dir", None),
            )
        else:
            env_kwargs = dict(
                control_mode=args.control_mode,
                reward_mode="dense",
                obs_mode=args.obs_mode,
                render_mode="rgb_array",
                sensor_configs=dict(shader_pack=args.shader),
                human_render_camera_configs=dict(shader_pack=args.shader),
                max_episode_steps=args.max_episode_steps,
            )

            tmp_env = gym.make(args.env_id, **env_kwargs)
            original_obs_space = tmp_env.observation_space
            include_rgb = "rgb" in args.obs_mode
            include_depth = "depth" in args.obs_mode
            tmp_env.close()

            obs_process_fn = partial(
                convert_obs,
                concat_fn=partial(np.concatenate, axis=-1),
                transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
                state_obs_extractor=build_state_obs_extractor(args.env_id),
                depth=False
            )

            # ============ Simplified: remove the lang_descriptions_path and video_path arguments ============
            self.train_dataset = OpenVLADataset(
                demo_path=args.demo_path,
                obs_process_fn=obs_process_fn,
                obs_space=original_obs_space,
                processor=self.processor,
                pred_horizon=args.pred_horizon,
                include_rgb=include_rgb,
                include_depth=include_depth,
                action_dim=args.action_dim,
                use_hand_camera=args.use_hand_camera,
                image_size=args.image_size,
                action_mode=args.action_mode,
                action_tokenizer=self.action_tokenizer,
                num_traj=args.num_demos,
                default_language=args.default_language,
            )

            if self.accelerator.is_main_process:
                stats_path = os.path.join(self.args.output_dir, "openvla_maniskill_pickcubeycb.json")
                self.train_dataset.action_normalizer.save(stats_path)
                with open(stats_path, 'r') as f:
                    norm_stats = json.load(f)
                if hasattr(self.vla, 'module'):
                    unwrapped_vla = self.vla.module
                else:
                    unwrapped_vla = self.vla
                unwrapped_vla.norm_stats = norm_stats

        prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=partial(openvla_collate_fn, pad_token_id=self.processor.tokenizer.pad_token_id),
            pin_memory=args.pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
            multiprocessing_context="forkserver" if args.num_workers > 0 else None,
        )

        print(f"[Trainer] Data setup complete. Training samples: {len(self.train_dataset)}")

    def _setup_optimizer(self):
        """Setup optimizer with memory optimizations"""
        args = self.args

        trainable_params = []
        if self.vla is not None:
            trainable_params += [p for p in self.vla.parameters() if p.requires_grad]
        if self.action_head is not None:
            trainable_params += [p for p in self.action_head.parameters() if p.requires_grad]
        if self.noisy_action_projector is not None:
            trainable_params += [p for p in self.noisy_action_projector.parameters() if p.requires_grad]
        if self.proprio_projector is not None:
            trainable_params += [p for p in self.proprio_projector.parameters() if p.requires_grad]

        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"[Trainer] Total trainable parameters in optimizer: {total_trainable:,}")
        if total_trainable == 0:
            raise RuntimeError("No trainable parameters remain after applying the current freeze/train settings.")

        if args.use_8bit_optimizer:
            print("[Memory Opt] Using 8bit AdamW optimizer...")
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    eps=args.adam_epsilon,
                    weight_decay=args.weight_decay,
                )
            except ImportError:
                print("[Warning] bitsandbytes not available, using standard AdamW")
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=args.learning_rate,
                    betas=(args.adam_beta1, args.adam_beta2),
                    eps=args.adam_epsilon,
                    weight_decay=args.weight_decay,
                )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_epsilon,
                weight_decay=args.weight_decay,
            )

        self.steps_per_epoch = len(self.train_dataloader)
        self.num_update_steps_per_epoch = max(
            (self.steps_per_epoch + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps,
            1,
        )
        if args.use_epoch_training:
            max_train_steps = args.num_train_epochs * self.num_update_steps_per_epoch
        else:
            max_train_steps = max(int(args.total_iters), 1)

        if args.warmup_steps is not None:
            num_warmup_steps = args.warmup_steps
        else:
            num_warmup_steps = int(max_train_steps * args.warmup_ratio)

        self.scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        device_placement = None
        if getattr(self, "use_kbit", False):
            device_placement = [False, True, True, True]
        if self.vla is not None:
            self.vla, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
                self.vla, self.optimizer, self.train_dataloader, self.scheduler,
                device_placement=device_placement
            )
        else:
            # Cache mode: no VLA to prepare; optimizer/dataloader/scheduler only.
            self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
                self.optimizer, self.train_dataloader, self.scheduler,
            )

        if self.action_head is not None:
            self.action_head = self.accelerator.prepare(self.action_head)
        if self.noisy_action_projector is not None:
            self.noisy_action_projector = self.accelerator.prepare(self.noisy_action_projector)
        if self.proprio_projector is not None:
            self.proprio_projector = self.accelerator.prepare(self.proprio_projector)

        self._refresh_trainable_handles()

        print(
            f"[Trainer] Optimizer setup complete. Max steps: {max_train_steps}, "
            f"Warmup: {num_warmup_steps}, Steps/epoch: {self.steps_per_epoch}"
        )

    def _setup_eval_env(self):
        args = self.args
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

        base_env_name, l_level = _resolve_eval_env_spec(args)
        lr_mirror_enabled = parse_optional_bool_flag(args.eval_lr_mirror)
        lr_mirror_robot_pose_enabled = parse_optional_bool_flag(args.eval_lr_mirror_robot_pose)

        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode="dense",
            obs_mode=args.obs_mode,
            render_mode="rgb_array",
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            max_episode_steps=args.max_episode_steps,
        )

        other_kwargs = dict(obs_horizon=args.obs_horizon)

        video_dir = None
        if args.capture_video:
            video_dir = os.path.join(args.output_dir, "videos")

        wrappers = []
        eval_camera = args.eval_camera
        if eval_camera is None and args.use_lerobot:
            eval_camera = args.lerobot_camera
        if eval_camera:
            wrappers.append(partial(SelectCameraObservationWrapper, camera_names=[eval_camera]))
        wrappers.append(FlattenRGBDObservationWrapper)

        configure_tabletop_task_flags(
            level=l_level,
            lr_mirror_enabled=lr_mirror_enabled,
            lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
        )

        if not getattr(args, "no_eval", False):
            from examples.baselines.openvla_oft.utils.make_env import make_eval_envs
            self.eval_envs = make_eval_envs(
                base_env_name,
                args.num_eval_envs,
                args.sim_backend,
                env_kwargs,
                other_kwargs,
                video_dir=video_dir,
                wrappers=wrappers,
                l_level=l_level,
                lr_mirror_enabled=lr_mirror_enabled,
                lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
            )
            # Build evaluation helpers once and reuse them across all eval calls.
            # Recreating these on every eval reloads the full human dataset and
            # causes long stalls before rollout even begins.
            self.evaluate_processor = _build_evaluate_processor(args)
            self.eval_sim_task_ids = _parse_sim_task_ids(
                args.eval_sim_task_ids,
                args.num_eval_envs,
                args.env_id,
            )
        else:
            self.eval_envs = None
            self.evaluate_processor = None
            self.eval_sim_task_ids = None
            print('\n⏭️  Simulation environments skipped (--no-eval)')

        print(
            f"[Trainer] Created evaluation environments: base={base_env_name}, "
            f"level={l_level}, lr_mirror={lr_mirror_enabled}, "
            f"lr_mirror_robot_pose={lr_mirror_robot_pose_enabled}"
        )

    def _resolve_epoch_freq(self, explicit_epoch_freq: Optional[int], step_freq: Optional[int]) -> int:
        if explicit_epoch_freq is not None:
            return max(int(explicit_epoch_freq), 0)
        if step_freq is None or int(step_freq) <= 0 or getattr(self, "steps_per_epoch", 0) <= 0:
            return 0
        return max(1, int(step_freq) // int(self.steps_per_epoch))

    def _run_training_step(
            self,
            batch,
            progress_bar,
            cleanup_interval: int,
            enable_step_eval: bool,
            enable_step_save: bool,
            batch_fetch_time: float | None = None,
    ) -> None:
        args = self.args
        batch_size = int(batch["actions"].shape[0]) if "actions" in batch else -1
        trace = self.stage_profiler.create_trace(
            route=_describe_training_route(args, "vla_feat" in batch),
            global_step=self.global_step,
            current_epoch=self.current_epoch,
            batch_size=batch_size,
        )
        if trace is not None:
            trace.set_metadata(
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                use_lora=bool(getattr(args, "use_lora", False)),
                freeze_vla=bool(getattr(args, "freeze_vla", False)),
                use_cache=bool("vla_feat" in batch),
                use_proprio=bool(getattr(args, "use_proprio", False)),
                action_mode=getattr(args, "action_mode", None),
            )

        if trace is not None and batch_fetch_time is not None:
            trace.add_timing("dataloader_batch_fetch_s", batch_fetch_time)

        with activate_step_trace(trace):
            with self.accelerator.accumulate(*self._trainable_modules):
                if trace is not None:
                    with trace.stage("train_forward_total_s"):
                        loss, metrics = self._forward_pass(batch)
                else:
                    loss, metrics = self._forward_pass(batch)

                if trace is not None:
                    with trace.stage("backward_s"):
                        self.accelerator.backward(loss)
                else:
                    self.accelerator.backward(loss)

                if args.max_grad_norm > 0:
                    if trace is not None:
                        with trace.stage("clip_grad_s", sync_cuda=False):
                            self.accelerator.clip_grad_norm_(self._trainable_params, args.max_grad_norm)
                    else:
                        self.accelerator.clip_grad_norm_(self._trainable_params, args.max_grad_norm)

                if trace is not None:
                    with trace.stage("optimizer_step_s", sync_cuda=False):
                        self.optimizer.step()
                    with trace.stage("scheduler_step_s", sync_cuda=False):
                        self.scheduler.step()
                    with trace.stage("optimizer_zero_grad_s", sync_cuda=False):
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                progress_bar.update(1)

                for key, value in metrics.items():
                    if key in self.recent_metrics:
                        self.recent_metrics[key].append(value)

                if self.global_step % args.log_interval == 0:
                    self._log_metrics()

                if enable_step_save and args.save_interval and self.global_step % args.save_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                if enable_step_eval and args.eval_freq > 0 and self.global_step % args.eval_freq == 0 and not getattr(args, "no_eval", False):
                    self.evaluate()

                if cleanup_interval > 0 and self.global_step % cleanup_interval == 0:
                    if trace is not None:
                        with trace.stage("cleanup_s", sync_cuda=False):
                            torch.cuda.empty_cache()
                            gc.collect()
                    else:
                        torch.cuda.empty_cache()
                        gc.collect()

        if trace is not None:
            trace.finalize()

    def train(self):
        """Train with aggressive memory management"""
        args = self.args

        mode = "epoch" if args.use_epoch_training else "iteration"
        print(f"[Trainer] Starting {mode}-based training")

        if args.eval_at_start and not getattr(args, "no_eval", False):
            self.evaluate()

        cleanup_interval = max(int(args.cleanup_interval), 0)
        if args.use_epoch_training:
            total_progress_steps = args.num_train_epochs * len(self.train_dataloader)
            progress_bar = tqdm(total=total_progress_steps, disable=not self.accelerator.is_main_process)
            if self.global_step > 0:
                progress_bar.update(min(self.global_step, total_progress_steps))

            start_epoch = int(self.current_epoch)
            if start_epoch == 0 and self.global_step > 0 and len(self.train_dataloader) > 0:
                start_epoch = min(self.global_step // len(self.train_dataloader), args.num_train_epochs)
                if start_epoch > 0:
                    print(
                        f"[Trainer] Derived start_epoch={start_epoch} from global_step={self.global_step} "
                        f"(resume checkpoint did not store current_epoch)"
                    )

            eval_epoch_freq = self._resolve_epoch_freq(args.eval_epoch_freq, args.eval_freq)
            save_epoch_freq = self._resolve_epoch_freq(args.save_epoch_freq, args.save_interval)
            print(
                f"[Trainer] Epoch schedule: total_epochs={args.num_train_epochs}, "
                f"steps_per_epoch={len(self.train_dataloader)}, "
                f"eval_every={eval_epoch_freq} epoch(s), save_every={save_epoch_freq} epoch(s)"
            )

            if start_epoch >= args.num_train_epochs:
                progress_bar.close()
                self.stage_profiler.close()
                print(
                    f"[Trainer] current_epoch={start_epoch} already reached "
                    f"num_train_epochs={args.num_train_epochs}; nothing to do."
                )
                return

            for epoch in range(start_epoch, args.num_train_epochs):
                self.current_epoch = epoch
                if self.vla is not None:
                    if args.freeze_vla:
                        self.vla.eval()
                    else:
                        self.vla.train()
                if self.action_head is not None:
                    self.action_head.train()
                if self.noisy_action_projector is not None:
                    if args.train_action_head_only:
                        self.noisy_action_projector.eval()
                    else:
                        self.noisy_action_projector.train()
                if self.proprio_projector is not None:
                    if args.train_action_head_only:
                        self.proprio_projector.eval()
                    else:
                        self.proprio_projector.train()

                _batch_fetch_t0 = time.perf_counter()
                for batch_idx, batch in enumerate(self.train_dataloader):
                    _batch_fetch_dur = time.perf_counter() - _batch_fetch_t0
                    self._run_training_step(
                        batch=batch,
                        progress_bar=progress_bar,
                        cleanup_interval=cleanup_interval,
                        enable_step_eval=False,
                        enable_step_save=False,
                        batch_fetch_time=_batch_fetch_dur,
                    )
                    _batch_fetch_t0 = time.perf_counter()

                self.current_epoch = epoch + 1

                if eval_epoch_freq > 0 and (epoch + 1) % eval_epoch_freq == 0 and not getattr(args, "no_eval", False):
                    self.evaluate()

                if save_epoch_freq > 0 and (
                        (epoch + 1) % save_epoch_freq == 0 or epoch == args.num_train_epochs - 1
                ):
                    self.save_checkpoint(f"epoch_{epoch + 1}")
        else:
            progress_bar = tqdm(total=args.total_iters, disable=not self.accelerator.is_main_process)
            if self.global_step > 0:
                progress_bar.update(min(self.global_step, args.total_iters))

            for epoch in range(args.num_train_epochs):
                self.current_epoch = epoch
                if self.vla is not None:
                    if args.freeze_vla:
                        self.vla.eval()
                    else:
                        self.vla.train()
                if self.action_head is not None:
                    self.action_head.train()
                if self.noisy_action_projector is not None:
                    if args.train_action_head_only:
                        self.noisy_action_projector.eval()
                    else:
                        self.noisy_action_projector.train()
                if self.proprio_projector is not None:
                    if args.train_action_head_only:
                        self.proprio_projector.eval()
                    else:
                        self.proprio_projector.train()

                _batch_fetch_t0 = time.perf_counter()
                for batch_idx, batch in enumerate(self.train_dataloader):
                    _batch_fetch_dur = time.perf_counter() - _batch_fetch_t0
                    self._run_training_step(
                        batch=batch,
                        progress_bar=progress_bar,
                        cleanup_interval=cleanup_interval,
                        enable_step_eval=True,
                        enable_step_save=True,
                        batch_fetch_time=_batch_fetch_dur,
                    )
                    _batch_fetch_t0 = time.perf_counter()

                    if self.global_step >= args.total_iters:
                        break

                self.current_epoch = epoch + 1
                if self.global_step >= args.total_iters:
                    break

        self.save_checkpoint("final")
        progress_bar.close()
        self.stage_profiler.close()
        print("[Trainer] Training complete!")

    def _forward_pass(self, batch):
        """Forward pass with memory optimizations.

        Two modes:
          - Hidden-state mode: use cached or frozen-VLA action_hidden_states,
            then apply output-side proprio_projector + action_head
          - Normal mode:   batch has pixel_values/input_ids/labels → run VLA forward
        """
        args = self.args
        use_cache = "vla_feat" in batch
        trace = get_active_step_trace()

        if trace is not None:
            with trace.stage("move_targets_to_device_s", sync_cuda=False):
                actions = batch["actions"].to(self.device, dtype=self.model_dtype, non_blocking=True)
                proprio = batch.get("proprio")
                if args.use_proprio:
                    if proprio is None:
                        raise ValueError("use_proprio is enabled but batch has no proprio")
                    proprio = proprio.to(self.device, dtype=self.model_dtype, non_blocking=True)
                else:
                    proprio = None
        else:
            actions = batch["actions"].to(self.device, dtype=self.model_dtype, non_blocking=True)
            proprio = batch.get("proprio")
            if args.use_proprio:
                if proprio is None:
                    raise ValueError("use_proprio is enabled but batch has no proprio")
                proprio = proprio.to(self.device, dtype=self.model_dtype, non_blocking=True)
            else:
                proprio = None

        metrics = {}

        def _forward_from_hidden_states(actions_hidden_states):
            head_compute_dtype = _module_param_dtype(self.action_head, self.model_dtype)
            if trace is not None:
                with trace.stage("hidden_state_cast_s"):
                    actions_hidden_states = actions_hidden_states.to(head_compute_dtype)
            else:
                actions_hidden_states = actions_hidden_states.to(head_compute_dtype)

            if proprio is not None and self.proprio_projector is not None:
                projector_dtype = _module_param_dtype(self.proprio_projector, actions_hidden_states.dtype)
                if trace is not None:
                    with trace.stage("output_side_proprio_proj_s"):
                        proprio_input = proprio.to(projector_dtype)
                        proprio_proj = self.proprio_projector.module.forward(proprio_input) if hasattr(
                            self.proprio_projector, "module"
                        ) else self.proprio_projector(proprio_input)
                        if proprio_proj.dtype != actions_hidden_states.dtype:
                            proprio_proj = proprio_proj.to(actions_hidden_states.dtype)
                    with trace.stage("output_side_proprio_fuse_s", sync_cuda=False):
                        actions_hidden_states = actions_hidden_states + proprio_proj.unsqueeze(1)
                else:
                    proprio_input = proprio.to(projector_dtype)
                    proprio_proj = self.proprio_projector.module.forward(proprio_input) if hasattr(
                        self.proprio_projector, "module"
                    ) else self.proprio_projector(proprio_input)
                    if proprio_proj.dtype != actions_hidden_states.dtype:
                        proprio_proj = proprio_proj.to(actions_hidden_states.dtype)
                    actions_hidden_states = actions_hidden_states + proprio_proj.unsqueeze(1)

            if trace is not None:
                with trace.stage("action_head_forward_s"):
                    predicted_actions = self.action_head.module.predict_action(actions_hidden_states) if hasattr(
                        self.action_head, "module"
                    ) else self.action_head.predict_action(actions_hidden_states)
                with trace.stage("loss_compute_s", sync_cuda=False):
                    target_actions = actions.to(predicted_actions.dtype)
                    loss = F.l1_loss(predicted_actions, target_actions)
            else:
                predicted_actions = self.action_head.module.predict_action(actions_hidden_states) if hasattr(
                    self.action_head, "module"
                ) else self.action_head.predict_action(actions_hidden_states)
                target_actions = actions.to(predicted_actions.dtype)
                loss = F.l1_loss(predicted_actions, target_actions)
            target_actions = actions.to(predicted_actions.dtype)
            metrics["curr_action_l1_loss"] = F.l1_loss(predicted_actions[:, 0], target_actions[:, 0]).item()
            metrics["next_actions_l1_loss"] = F.l1_loss(predicted_actions[:, 1:], target_actions[:, 1:]).item()
            metrics["loss"] = loss.item()
            return loss, metrics

        # ── Hidden-state route: cache or frozen-VLA online hidden states ─────
        if _uses_hidden_state_l1_route(args):
            if use_cache:
                if trace is not None:
                    with trace.stage("cache_hidden_state_to_device_s"):
                        actions_hidden_states = batch["vla_feat"].to(
                            self.device, dtype=self.model_dtype, non_blocking=True
                        )
                else:
                    actions_hidden_states = batch["vla_feat"].to(
                        self.device, dtype=self.model_dtype, non_blocking=True
                    )
            else:
                if trace is not None:
                    with trace.stage("move_model_inputs_to_device_s", sync_cuda=False):
                        pixel_values = batch["pixel_values"].to(self.device, dtype=self.model_dtype, non_blocking=True)
                        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                        labels = batch["labels"].to(self.device, non_blocking=True)
                else:
                    pixel_values = batch["pixel_values"].to(self.device, dtype=self.model_dtype, non_blocking=True)
                    input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                    labels = batch["labels"].to(self.device, non_blocking=True)

                use_autocast = torch.cuda.is_available() and self.model_dtype in (torch.float16, torch.bfloat16)
                if trace is not None:
                    with trace.stage("vla_forward_total_s"):
                        with torch.no_grad():
                            with torch.autocast("cuda", dtype=self.model_dtype, enabled=use_autocast):
                                output = self.vla(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    pixel_values=pixel_values,
                                    labels=labels,
                                    output_hidden_states=False,
                                    proprio=None,
                                    proprio_projector=None,
                                    noisy_actions=None,
                                    noisy_action_projector=None,
                                    diffusion_timestep_embeddings=None,
                                    use_film=args.use_film,
                                    continuous_actions_fast_path=True,
                                )
                else:
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=self.model_dtype, enabled=use_autocast):
                            output = self.vla(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=labels,
                                output_hidden_states=False,
                                proprio=None,
                                proprio_projector=None,
                                noisy_actions=None,
                                noisy_action_projector=None,
                                diffusion_timestep_embeddings=None,
                                use_film=args.use_film,
                                continuous_actions_fast_path=True,
                            )
                actions_hidden_states = output.action_hidden_states
                del output

            return _forward_from_hidden_states(actions_hidden_states)

        if use_cache:
            raise ValueError(
                "Cached VLA features are only supported with --action_mode l1_regression."
            )

        # ── Normal mode: run VLA forward ──────────────────────────────────────
        if trace is not None:
            with trace.stage("move_model_inputs_to_device_s", sync_cuda=False):
                pixel_values   = batch["pixel_values"].to(self.device, dtype=self.model_dtype, non_blocking=True)
                input_ids      = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels         = batch["labels"].to(self.device, non_blocking=True)
        else:
            pixel_values   = batch["pixel_values"].to(self.device, dtype=self.model_dtype, non_blocking=True)
            input_ids      = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels         = batch["labels"].to(self.device, non_blocking=True)

        num_patches = self.vla.module.vision_backbone.get_num_patches() if hasattr(self.vla,
                                                                                   'module') else self.vla.vision_backbone.get_num_patches()
        if args.use_proprio:
            num_patches += 1
        if args.action_mode == "diffusion":
            num_patches += 1

        noisy_actions = None
        diffusion_timestep_embeddings = None
        if args.action_mode == "diffusion":
            noisy_dict = self.action_head.module.sample_noisy_actions(actions) if hasattr(self.action_head,
                                                                                          'module') else self.action_head.sample_noisy_actions(
                actions)
            noise = noisy_dict["noise"]
            noisy_actions = noisy_dict["noisy_actions"].to(self.model_dtype)
            diffusion_timestep_embeddings = noisy_dict["diffusion_timestep_embeddings"]

        continuous_fast = args.action_mode in ("l1_regression", "diffusion")
        use_autocast = torch.cuda.is_available() and self.model_dtype in (torch.float16, torch.bfloat16)
        if trace is not None:
            with trace.stage("vla_forward_total_s"):
                with torch.autocast("cuda", dtype=self.model_dtype, enabled=use_autocast):
                    if args.train_action_head_only:
                        with torch.no_grad():
                            output = self.vla(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=labels,
                                output_hidden_states=not continuous_fast,
                                proprio=proprio,
                                proprio_projector=self.proprio_projector,
                                noisy_actions=noisy_actions,
                                noisy_action_projector=self.noisy_action_projector,
                                diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                                use_film=args.use_film,
                                continuous_actions_fast_path=continuous_fast,
                            )
                    else:
                        output = self.vla(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels,
                            output_hidden_states=not continuous_fast,
                            proprio=proprio,
                            proprio_projector=self.proprio_projector,
                            noisy_actions=noisy_actions,
                            noisy_action_projector=self.noisy_action_projector,
                            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                            use_film=args.use_film,
                            continuous_actions_fast_path=continuous_fast,
                        )
        else:
            with torch.autocast("cuda", dtype=self.model_dtype, enabled=use_autocast):
                if args.train_action_head_only:
                    with torch.no_grad():
                        output = self.vla(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels,
                            output_hidden_states=not continuous_fast,
                            proprio=proprio,
                            proprio_projector=self.proprio_projector,
                            noisy_actions=noisy_actions,
                            noisy_action_projector=self.noisy_action_projector,
                            diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                            use_film=args.use_film,
                            continuous_actions_fast_path=continuous_fast,
                        )
                else:
                    output = self.vla(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                        output_hidden_states=not continuous_fast,
                        proprio=proprio,
                        proprio_projector=self.proprio_projector,
                        noisy_actions=noisy_actions,
                        noisy_action_projector=self.noisy_action_projector,
                        diffusion_timestep_embeddings=diffusion_timestep_embeddings,
                        use_film=args.use_film,
                        continuous_actions_fast_path=continuous_fast,
                    )

        if args.action_mode == "discrete":
            loss = output.loss

            ground_truth_token_ids = labels[:, 1:]
            predicted_token_ids = output.logits[:, num_patches:-1].argmax(dim=2)

            current_action_mask = get_current_action_mask(ground_truth_token_ids)
            next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

            if current_action_mask.any():
                curr_action_accuracy = compute_token_accuracy(
                    predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
                )
                curr_action_l1_loss = compute_actions_l1_loss(
                    self.action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=current_action_mask
                )
                metrics["curr_action_accuracy"] = curr_action_accuracy.item()
                metrics["curr_action_l1_loss"] = curr_action_l1_loss.item()

            if next_actions_mask.any():
                next_actions_accuracy = compute_token_accuracy(
                    predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
                )
                next_actions_l1_loss = compute_actions_l1_loss(
                    self.action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask=next_actions_mask
                )
                metrics["next_actions_accuracy"] = next_actions_accuracy.item()
                metrics["next_actions_l1_loss"] = next_actions_l1_loss.item()

        else:
            head_compute_dtype = _module_param_dtype(self.action_head, self.model_dtype)
            actions_hidden_states = output.action_hidden_states.to(head_compute_dtype)

            if args.action_mode == "l1_regression":
                if trace is not None:
                    with trace.stage("action_head_forward_s"):
                        predicted_actions = self.action_head.module.predict_action(actions_hidden_states) if hasattr(
                            self.action_head, 'module') else self.action_head.predict_action(actions_hidden_states)
                    with trace.stage("loss_compute_s", sync_cuda=False):
                        target_actions = actions.to(predicted_actions.dtype)
                        loss = F.l1_loss(predicted_actions, target_actions)
                else:
                    predicted_actions = self.action_head.module.predict_action(actions_hidden_states) if hasattr(
                            self.action_head, 'module') else self.action_head.predict_action(actions_hidden_states)
                    target_actions = actions.to(predicted_actions.dtype)
                    loss = F.l1_loss(predicted_actions, target_actions)

                target_actions = actions.to(predicted_actions.dtype)
                curr_action_l1 = F.l1_loss(predicted_actions[:, 0], target_actions[:, 0])
                next_actions_l1 = F.l1_loss(predicted_actions[:, 1:], target_actions[:, 1:])
                metrics["curr_action_l1_loss"] = curr_action_l1.item()
                metrics["next_actions_l1_loss"] = next_actions_l1.item()

            elif args.action_mode == "diffusion":
                noise_pred = self.action_head.module.predict_noise(actions_hidden_states) if hasattr(self.action_head,
                                                                                                     'module') else self.action_head.predict_noise(
                    actions_hidden_states)
                noise_pred = noise_pred.reshape(noise.shape)
                loss = F.mse_loss(noise_pred, noise)

        metrics["loss"] = loss.item()

        del output

        return loss, metrics

    def _log_metrics(self):
        smoothed_metrics = {}
        for key, deque_vals in self.recent_metrics.items():
            if len(deque_vals) > 0:
                smoothed_metrics[key] = sum(deque_vals) / len(deque_vals)

        smoothed_metrics["learning_rate"] = self.scheduler.get_last_lr()[0]

        for key, value in smoothed_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)

        if self.args.use_wandb and self.accelerator.is_main_process:
            wandb.log({f"train/{k}": v for k, v in smoothed_metrics.items()}, step=self.global_step)

    def evaluate(self):
        print(f"\n[Trainer] Running evaluation at step {self.global_step}")

        # If --no-eval was set, skip evaluation entirely.
        if getattr(self.args, "no_eval", False):
            print('⏭️  Evaluation skipped (--no-eval)')
            return
        if self.eval_envs is None:
            print("[Trainer] Evaluation environments are unavailable; skipping evaluation.")
            return

        torch.cuda.empty_cache()
        gc.collect()

        loaded_eval_vla = None
        if self.vla is None:
            print("[Trainer] Cache mode: loading base VLA temporarily for online evaluation.")
            loaded_eval_vla = self._load_base_vla_for_eval()
            unwrapped_vla = loaded_eval_vla
        else:
            unwrapped_vla = self._safe_unwrap(self.vla)
        unwrapped_action_head = self._safe_unwrap(self.action_head)
        unwrapped_noisy_projector = self._safe_unwrap(self.noisy_action_projector)
        unwrapped_proprio_projector = self._safe_unwrap(self.proprio_projector)
        target_dtype = getattr(self, "model_dtype", None)
        if target_dtype is None:
            try:
                for param in unwrapped_vla.parameters():
                    if param.dtype in (torch.float16, torch.bfloat16, torch.float32):
                        target_dtype = param.dtype
                        break
            except Exception:
                target_dtype = None
        if target_dtype is None:
            target_dtype = torch.float32
        if unwrapped_action_head is not None:
            unwrapped_action_head = unwrapped_action_head.to(self.device, dtype=target_dtype)
        if unwrapped_noisy_projector is not None:
            unwrapped_noisy_projector = unwrapped_noisy_projector.to(self.device, dtype=target_dtype)
        if unwrapped_proprio_projector is not None:
            unwrapped_proprio_projector = unwrapped_proprio_projector.to(self.device, dtype=target_dtype)
        # ============ Unified interface: use evaluate_processor (consistent with Pi) ============
        eval_stats_path = None
        eval_dataset_name = None
        if self.args.use_lerobot:
            eval_stats_path = os.path.join(self.args.output_dir, "openvla_lerobot_stats.json")
            eval_dataset_name = self.args.env_id
        else:
            eval_stats_path = os.path.join(self.args.output_dir, "openvla_maniskill_pickcubeycb.json")
            eval_dataset_name = "maniskill_pickcubeycb"

        evaluate_processor = self.evaluate_processor
        sim_task_ids = self.eval_sim_task_ids
        if evaluate_processor is None or sim_task_ids is None:
            print("[Trainer] Evaluation helpers missing; rebuilding them lazily.")
            evaluate_processor = _build_evaluate_processor(self.args)
            sim_task_ids = _parse_sim_task_ids(
                self.args.eval_sim_task_ids,
                self.args.num_eval_envs,
                self.args.env_id,
            )
            self.evaluate_processor = evaluate_processor
            self.eval_sim_task_ids = sim_task_ids

        try:
            self.best_eval_metrics = evaluate_and_save_best(
                iteration=self.global_step,
                vla=unwrapped_vla,
                processor=self.processor,
                eval_envs=self.eval_envs,
                device=self.device,
                sim_backend=self.args.sim_backend,
                demo_path=self.args.demo_path,
                action_dim=self.args.action_dim,
                output_dir=self.args.output_dir,
                norm_stats_path=eval_stats_path,
                dataset_name=eval_dataset_name,
                action_head=unwrapped_action_head,
                noisy_action_projector=unwrapped_noisy_projector,
                proprio_projector=unwrapped_proprio_projector,
                action_tokenizer=self.action_tokenizer,
                use_film=self.args.use_film,
                action_mode=self.args.action_mode,
                image_size=self.args.image_size,
                num_diffusion_steps_inference=self.args.num_diffusion_steps_inference,
                attn_implementation=self.args.attn_implementation,
                num_eval_episodes=self.args.num_eval_episodes,
                eval_freq=self.args.eval_freq,
                best_eval_metrics=self.best_eval_metrics,
                save_checkpoint_fn=self.save_checkpoint,
                writer=self.writer,
                default_language=self.args.default_language,
                evaluate_processor=evaluate_processor,  # unified interface (required)
                sim_task_ids=sim_task_ids,  # required
                proprio_fusion_mode="output" if _uses_output_side_proprio_fusion(self.args) else "input",
            )
        finally:
            if loaded_eval_vla is not None:
                del loaded_eval_vla
            torch.cuda.empty_cache()
            gc.collect()

    def save_checkpoint(self, name: str):
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = os.path.join(self.args.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"[Trainer] Saving checkpoint to {checkpoint_dir}")

        unwrapped_vla = self._safe_unwrap(self.vla)

        # When VLA is frozen its weights are never updated; skip the expensive
        # save_pretrained to avoid writing 7B parameters every checkpoint.
        if not getattr(self.args, "freeze_vla", False):
            unwrapped_vla.save_pretrained(checkpoint_dir)
            self.processor.save_pretrained(checkpoint_dir)
        else:
            print(f"[Trainer] --freeze_vla: skipping VLA save_pretrained (weights unchanged).")

        stats_src = os.path.join(self.args.output_dir, "openvla_maniskill_pickcubeycb.json")
        stats_dst = os.path.join(checkpoint_dir, "openvla_maniskill_pickcubeycb.json")

        if os.path.exists(stats_src):
            import shutil
            shutil.copy2(stats_src, stats_dst)

        if self.action_head is not None:
            unwrapped_head = self._safe_unwrap(self.action_head)
            torch.save(unwrapped_head.state_dict(), os.path.join(checkpoint_dir, "action_head.pt"))

        if self.noisy_action_projector is not None:
            unwrapped_proj = self._safe_unwrap(self.noisy_action_projector)
            torch.save(unwrapped_proj.state_dict(), os.path.join(checkpoint_dir, "noisy_action_projector.pt"))

        if self.proprio_projector is not None:
            unwrapped_proprio = self._safe_unwrap(self.proprio_projector)
            torch.save(unwrapped_proprio.state_dict(), os.path.join(checkpoint_dir, "proprio_projector.pt"))

        # Save optimizer & scheduler for proper resume
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_eval_metrics": dict(self.best_eval_metrics),
        }
        training_state.update(self._capture_rng_state())
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))

        torch.cuda.empty_cache()


# ==========================================================================
# MAIN
# ==========================================================================

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="OpenVLA Training - Simplified (No Video Dependency)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Critical paths
    parser.add_argument(
        "--demo_path",
        default=None,
        help="Path to .h5 demos (required unless --use_lerobot is set).",
    )
    # ============ Simplified: remove lang_descriptions_path and video_path ============
    parser.add_argument("--output_dir", default="./runs/openvla")
    parser.add_argument("--resume_from", default=None)

    # ============ New: default language ============
    parser.add_argument("--default_language", type=str, default="pick red cube and place on plate.",
                        help="Default language instruction if not specified in dataset")

    # Experiment
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_deterministic", action="store_true", default=False)

    # Environment
    parser.add_argument("--env_id", default="PickCubeYCB-v1")
    parser.add_argument("--obs_mode", default="rgb")
    parser.add_argument("--control_mode", default="pd_joint_pos")
    parser.add_argument("--max_episode_steps", type=int, default=600)
    parser.add_argument("--sim_backend", default="physx_cpu")
    parser.add_argument("--shader", default="rt-fast")
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--capture_video", action="store_true", default=True)
    parser.add_argument("--eval_camera", type=str, default=None,
                        help="Optional camera name to use for evaluation observations.")

    # Model
    parser.add_argument("--model_name_or_path", default="openvla/openvla-7b")
    parser.add_argument("--from_scratch", action="store_true", default=False)

    # Memory optimization
    parser.add_argument("--freeze_vision_backbone", action="store_true", default=True)
    parser.add_argument("--freeze_vla", action="store_true", default=False,
                        help="Freeze the full VLA backbone. With --action_mode l1_regression, training uses "
                             "the same hidden-state route as --vla_cache_dir: VLA produces action_hidden_states "
                             "without proprio on the input side, then proprio/action heads run outside VLA.")
    parser.add_argument("--vla_cache_dir", type=str, default=None,
                        help="Path to precomputed VLA feature cache (from precompute_vla_features.py). "
                             "When set, VLA forward is skipped entirely during training; "
                             "the same output-side proprio/action-head route as frozen-VLA l1_regression is used.")
    parser.add_argument("--train_action_head_only", action="store_true", default=False,
                        help="Train only the external action head. Implies --freeze_vla and also freezes "
                             "proprio/noisy-action projectors, running VLA forward under no_grad.")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_8bit_optimizer", action="store_true", default=True)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--disable_tf32", action="store_true", default=False)
    parser.add_argument("--matmul_precision", type=str, default="high", choices=["highest", "high", "medium"])

    # Action
    parser.add_argument("--action_mode", default="discrete", choices=["discrete", "l1_regression", "diffusion"])
    parser.add_argument("--pred_horizon", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--n_action_bins", type=int, default=256)
    parser.add_argument("--action_min", type=float, default=-1.0)
    parser.add_argument("--action_max", type=float, default=1.0)

    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default=None)
    parser.add_argument("--lora_modules_to_save", type=str, default=None)

    # Training
    parser.add_argument("--total_iters", type=int, default=1000000)
    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument("--total_epochs", type=int, default=None,
                        help="Alias for --num_train_epochs; automatically enables --use_epoch_training.")
    parser.add_argument("--use_epoch_training", action="store_true",
                        help="Train for full dataloader epochs instead of stopping at --total_iters.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--effective_batch_size", type=int, default=16)

    # Optimizer
    parser.add_argument("--optimizer_type", type=str, default="adamw")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LR schedule
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--lr_min_ratio", type=float, default=0.1)

    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # Dataloader
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    parser.add_argument("--persistent_workers", action="store_true", default=True)
    parser.add_argument("--cleanup_interval", type=int, default=0,
                        help="Call torch.cuda.empty_cache/gc every N steps; 0 disables periodic cleanup.")
    parser.add_argument("--debug_stage_timing", action="store_true", default=False,
                        help="Write per-step stage timing/resource snapshots to output_dir for route comparison.")
    parser.add_argument("--debug_stage_timing_every", type=int, default=1,
                        help="Profile every Nth step when --debug_stage_timing is enabled.")
    parser.add_argument("--debug_stage_timing_max_steps", type=int, default=0,
                        help="Profile at most this many steps per run; 0 means no limit.")
    parser.add_argument("--debug_stage_timing_sync_cuda", dest="debug_stage_timing_sync_cuda",
                        action="store_true", help="Synchronize CUDA around timed stages for more accurate timings.")
    parser.add_argument("--no_debug_stage_timing_sync_cuda", dest="debug_stage_timing_sync_cuda",
                        action="store_false", help="Disable CUDA synchronization around timed stages.")
    parser.set_defaults(debug_stage_timing_sync_cuda=True)


    # Data
    parser.add_argument("--use_hand_camera", action="store_true", default=False)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])

    # Augmentation
    parser.add_argument("--use_augmentation", action="store_true", default=False)

    # Evaluation
    parser.add_argument("--eval_freq", type=int, default=2000)
    parser.add_argument("--eval_epoch_freq", "--epoch_eval_freq", dest="eval_epoch_freq", type=int, default=None,
                        help="Epoch-mode evaluation frequency. If omitted, derived from --eval_freq.")
    parser.add_argument("--num_eval_episodes", type=int, default=1)
    parser.add_argument("--num_eval_envs", type=int, default=1)
    parser.add_argument("--eval_at_start", action="store_true", default=False)
    parser.add_argument(
        "--no_eval", action="store_true", default=False,
        help="Skip ALL evaluation and avoid creating any simulation environment.",
    )
    parser.add_argument("--center_crop", action="store_true", default=False)
    parser.add_argument("--use_proprio", action="store_true", default=False)
    parser.add_argument("--use_film", action="store_true", default=False)
    parser.add_argument("--num_images_in_input", type=int, default=1)
    parser.add_argument("--use_lerobot", action="store_true", default=True)
    parser.add_argument("--lerobot_camera", type=str, default="zed2i")
    parser.add_argument("--lerobot_video_backend", type=str, default=None)
    parser.add_argument("--human_root", default="demos")
    parser.add_argument("--human_split", default="train")
    parser.add_argument("--human_dataset_file",
                        default="examples/baselines/lerobot_dataset/config/human_config.json")
    parser.add_argument("--human_task_description_file",
                        default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    parser.add_argument("--sim_task_description_file",
                        default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")
    parser.add_argument("--sim_root", default="lerobot_datasets")
    parser.add_argument("--sim_split", default="train")
    parser.add_argument("--sim_dataset_file", default=None)
    parser.add_argument("--task_mapping_file",
                        default="examples/baselines/lerobot_dataset/task_mapping.json")
    parser.add_argument("--eval_sim_task_ids", default=None,
                        help="Comma-separated sim_task_id list matching num_eval_envs.")
    parser.add_argument("--eval_lr_mirror", default="false", choices=["auto", "true", "false"],
                        help="Override tabletop left-right mirror during eval. Default 'false'.")
    parser.add_argument("--eval_lr_mirror_robot_pose", default="false", choices=["auto", "true", "false"],
                        help="Override robot pose swapping under mirror during eval. Default 'false' keeps agents unswapped.")

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="openvla")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--save_epoch_freq", "--epoch_save_freq", dest="save_epoch_freq", type=int, default=None,
                        help="Epoch-mode checkpoint frequency. If omitted, derived from --save_interval.")
    parser.add_argument("--save_best_only", action="store_true", default=False)

    # Diffusion
    parser.add_argument("--num_diffusion_steps_train", type=int, default=50)
    parser.add_argument("--num_diffusion_steps_inference", type=int, default=50)

    # Advanced
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--unnorm_key", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    args = finalize_args(args)

    print("\n" + "=" * 60)
    print("SIMPLIFIED TRAINING (NO VIDEO DEPENDENCY)")
    print("=" * 60)
    print(f"Demo Path: {args.demo_path}")
    print(f"Default Language: {args.default_language}")
    print(f"Action Mode: {args.action_mode}")
    print("=" * 60 + "\n")

    trainer = OpenVLATrainer(args)
    trainer.train()

    return 0


if __name__ == "__main__":
    exit(main())
