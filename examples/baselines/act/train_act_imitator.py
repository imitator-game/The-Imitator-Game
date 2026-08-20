"""
Training Script for ACT (Action Chunking with Transformers) + Task Encoder
===========================================================================
Architecture:
  task_encoder  →  z  (task latent, [B, task_latent_dim])
  DETRVAE:
    backbone    →  visual features from RGB obs
    transformer →  action sequence
    z           →  video_feature conditioning in transformer
  Output:         (B, pred_horizon, action_dim)
"""

ALGO_NAME = "ACT_TaskEncoder"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import tyro

from diffusers.optimization import get_scheduler

from examples.baselines.act.act.detr_video.backbone import build_backbone
from examples.baselines.act.act.detr_video.transformer import build_transformer
from examples.baselines.act.act.detr_video.detr_vae import build_encoder, DETRVAE

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanSimPairedDataset,
    PairedDatasetConfig,
    InputMode,
    get_collate_fn,
)


# ============================================================================
# Training arguments
# ============================================================================

@dataclass
class TrainingArgs:
    """Full training configuration for ACT + Task Encoder."""

    # ── Experiment ────────────────────────────────────────────────────────────
    exp_name:           Optional[str] = None
    seed:               int  = 1
    cuda:               bool = True
    track:              bool = False
    wandb_project_name: str  = "ACTTaskEncoder"
    wandb_entity:       Optional[str] = None

    # ── Data paths ────────────────────────────────────────────────────────────
    human_root:        str = "demos"
    sim_root:          str = "demos"
    task_mapping_file: str = "examples/baselines/lerobot_dataset/task_mapping.json"

    # ── Dataset configs ───────────────────────────────────────────────────────
    human_dataset_file: Optional[str] = "examples/baselines/lerobot_dataset/config/human_config.json"
    sim_dataset_file:   Optional[str] = "examples/baselines/lerobot_dataset/config/sim_config.json"
    human_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    sim_task_description_file:   Optional[str] = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"

    # ── Robot data (real-robot training) ─────────────────────────────────────
    data_source:                 str          = "sim"    # "sim" | "robot"
    robot_root:                  str          = "demos"
    robot_dataset_file:          Optional[str] = None
    robot_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/robot_desc.json"

    # ── Training mode ─────────────────────────────────────────────────────────
    use_epoch_training: bool  = True
    total_epochs:       int   = 100
    total_iters:        int   = 1_000_000
    batch_size:         int   = 256
    lr:                 float = 1e-4
    lr_backbone:        float = 1e-5
    kl_weight:          float = 10.0
    num_dataload_workers: int = 0
    warmup_epochs:      int   = 5

    # ── ACT model architecture ────────────────────────────────────────────────
    action_dim:         int  = 16
    state_dim:          int  = 18
    pred_horizon:       int  = 24         # num_queries for DETRVAE
    obs_horizon:        int  = 1

    # DETR / Transformer
    backbone:           str  = "resnet18"
    position_embedding: str  = "sine"
    dilation:           bool = False
    masks:              bool = False
    enc_layers:         int  = 2
    dec_layers:         int  = 4
    dim_feedforward:    int  = 512
    hidden_dim:         int  = 256
    dropout:            float = 0.1
    nheads:             int  = 8
    pre_norm:           bool = False

    # ── Observation settings ──────────────────────────────────────────────────
    include_depth: bool       = False
    cameras:       List[str]  = field(default_factory=lambda: ["zed2i"])
    image_size:    Tuple[int, int] = (224, 224)
    state_type:    str        = "qpos"
    single_arm:    bool       = False

    # ── Task encoder backend ──────────────────────────────────────────────────
    # "frozen_backbone" → FrozenVideoBackbone (off-the-shelf frozen backbone)
    #                     e.g. DINOv2, CLIP, SigLIP, VideoMAE  (Condition C / ablation)
    task_encoder_type: str = "frozen_backbone"

    # ── FrozenVideoBackbone settings (task_encoder_type == "frozen_backbone") ─
    # Backbone: "dinov2_vitl14" | "dinov2_vitb14" | "dinov2_vitl14_reg"
    #           "clip_vitl14"   | "clip_vitb16"
    #           "siglip2_so400m" | "videomae_large" | "videomae_base"
    frozen_backbone_type:           str          = "dinov2_vitl14"
    frozen_backbone_model:          Optional[str] = None
    frozen_backbone_adapter_layers: int          = 1
    frozen_backbone_seq_patches:    int          = 32
    frozen_backbone_num_frames:     int          = 4

    # ── VideoEncoder (scratch) settings ──────────────────────────────────────
    video_encoder_type:       str = "temporal_transformer"
    video_encoder_hidden_dim: int = 512
    video_encoder_num_layers: int = 4
    video_encoder_num_heads:  int = 8

    # ── LoRA fine-tuning of the video backbone ─────────────────────────────────
    # lora_rank == 0  → fully frozen backbone (cache enabled)
    # lora_rank > 0   → LoRA adapters injected into attention layers; cache disabled
    frozen_backbone_lora_rank:    int   = 0      # 0 = no LoRA; typical: 8 or 16
    frozen_backbone_lora_alpha:   float = 16.0   # LoRA scaling: alpha / rank
    frozen_backbone_lora_dropout: float = 0.0    # LoRA input dropout

    # ── Task latent settings ─────────────────────────────────────────────────
    task_latent_dim:           int   = 256
    task_num_frames:           int   = 10
    hf_cache_dir:              Optional[str] = None

    # ── VL cache ─────────────────────────────────────────────────────────────
    te_cache_root:               Optional[str] = "./te_cache"
    te_cache_preload_to_memory:  bool = True
    te_cache_recompute:          bool = False

    # ── Input mode ────────────────────────────────────────────────────────────
    input_mode:               str   = "video_only"
    random_modality_selection: bool = True
    use_video_prob:            float = 0.5
    include_first_frame:       bool  = True
    fps:                       int   = 30
    video_backend:             str   = "torchcodec"

    # ── Logging and evaluation ────────────────────────────────────────────────
    log_freq:           int  = 10
    log_epoch_freq:     int  = 1
    eval_freq:          int  = 5000
    eval_epoch_freq:    int  = 5
    save_freq:          Optional[int] = None
    save_epoch_freq:    int  = 10
    num_eval_episodes:  int  = 10
    num_eval_envs:      int  = 1
    eval_temporal_agg:  bool = True
    no_eval:            bool = True

    # ── Environment ───────────────────────────────────────────────────────────
    env_id:            str  = "TwoRobotStirSpoon-v1"
    max_episode_steps: int  = 500
    sim_backend:       str  = "physx_cpu"
    control_mode:      str  = "pd_joint_pos"
    obs_mode:          str  = "rgb"
    shader:            str  = "rt-fast"

    # ── Checkpointing ─────────────────────────────────────────────────────────
    resume_from:        Optional[str] = None
    reset_lr_scheduler: bool = False


# ============================================================================
# Frozen backbone encoder helper
# ============================================================================

def _build_frozen_backbone_encoder(args, device, dataloader=None):
    from examples.baselines.encoders.task_encoder.video_backbone import build_video_backbone
 
    lora_rank    = getattr(args, "frozen_backbone_lora_rank",    0)
    lora_alpha   = getattr(args, "frozen_backbone_lora_alpha",   16.0)
    lora_dropout = getattr(args, "frozen_backbone_lora_dropout", 0.0)
    mode_str = (f"LoRA (rank={lora_rank}, alpha={lora_alpha})"
                if lora_rank > 0 else "Frozen")
    print(f"\\n   Building FrozenVideoBackbone ({args.frozen_backbone_type}) [{mode_str}] ...")
 
    backbone = build_video_backbone(
        backbone_type=args.frozen_backbone_type,
        latent_dim=args.task_latent_dim,
        max_seq_patches=args.frozen_backbone_seq_patches,
        adapter_layers=args.frozen_backbone_adapter_layers,
        num_sampled_frames=args.frozen_backbone_num_frames,
        hf_cache_dir=getattr(args, "hf_cache_dir", None),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).to(device)
 
    if lora_rank > 0:
        print("   LoRA mode: raw-feature cache DISABLED.")
    elif dataloader is not None:
        from examples.baselines.encoders.task_encoder.frozen_backbone_cache import (
            setup_frozen_backbone_cache)
        backbone = setup_frozen_backbone_cache(
            backbone, dataloader=dataloader, device=device,
            backbone_type=args.frozen_backbone_type,
            te_cache_root=args.te_cache_root,
            preload_to_memory=args.te_cache_preload_to_memory,
            recompute=args.te_cache_recompute, verbose=True,
        )
        print("   FrozenVideoBackbone: backbone frozen, cache active, adapter trainable.")
    return backbone


def _load_ckpt(model: nn.Module, ckpt_path: str) -> None:
    """Generic checkpoint loader that handles various save formats.

    Supports:
      - Raw state dict
      - {'model': state_dict}
      - {'model_state_dict': state_dict}
      - {'agent_state_dict': state_dict}  (strips 'task_encoder.' prefix)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("model", "model_state_dict", "agent_state_dict"):
        if key in ckpt:
            sd = ckpt[key]
            # If loaded from a full agent checkpoint, strip 'task_encoder.' prefix
            if key == "agent_state_dict":
                prefix = "task_encoder."
                sd = {k[len(prefix):]: v for k, v in sd.items()
                      if k.startswith(prefix)}
            model.load_state_dict(sd, strict=False)
            return
    # Fallback: assume the checkpoint IS the state dict
    model.load_state_dict(ckpt, strict=False)


# ============================================================================
# ACTAgent
# ============================================================================

class ACTAgent(nn.Module):
    """ACT (Action Chunking Transformer) agent with Frozen Video Backbone.
    - prepare_for_eval(human_video, robot_obs, ...)  → caches task z
    - get_action(obs)                                → (B, pred_horizon, action_dim)
    - clear_cache()                                  → clears cached z
    - compute_loss(batch)                            → loss dict
    """

    def __init__(
        self,
        args: TrainingArgs,
        device: torch.device,
        dataloader: Optional[DataLoader] = None,
    ):
        super().__init__()
        self.args         = args
        self.device       = device
        self.pred_horizon = args.pred_horizon
        self.action_dim   = args.action_dim
        self.state_dim    = args.state_dim
        self.kl_weight    = args.kl_weight
        self.include_depth = args.include_depth

        self.input_mode                = InputMode(args.input_mode)
        self.random_modality_selection = args.random_modality_selection
        self.use_video_prob            = args.use_video_prob

        # ── Task encoder (FrozenVideoBackbone only) ───────────────────────────
        self.task_encoder = _build_frozen_backbone_encoder(args, device, dataloader)
        # Only skip video decoding when backbone is truly frozen (no LoRA)
        lora_rank = getattr(args, "frozen_backbone_lora_rank", 0)
        if dataloader is not None and lora_rank == 0:
            from examples.baselines.encoders.task_encoder.frozen_backbone_cache import (
                enable_skip_human_video)
            enable_skip_human_video(self.task_encoder, dataloader,
                                    recreate_dataloader=False, verbose=True)

        task_latent_dim = args.task_latent_dim

        # ── Task norm — registered as submodule so .to(device) works ─────────
        self.ftask_norm = nn.LayerNorm(task_latent_dim, elementwise_affine=True)

        # ── Image normalization ───────────────────────────────────────────────
        self.img_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # ── DETR-VAE model ────────────────────────────────────────────────────
        backbones = [build_backbone(args)]
        self.model = DETRVAE(
            backbones=backbones,
            transformer=build_transformer(args),
            encoder=build_encoder(args),
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            num_queries=args.pred_horizon,
            video_feature_dim=task_latent_dim,
        )

        # ── Eval cache ────────────────────────────────────────────────────────
        self._cached_task_z: Optional[torch.Tensor] = None

        # Move everything to device
        self.to(device)
        self._print_model_info()

    def _build_video_encoder(self, args, device):
        """Scratch VideoEncoder is not supported; use task_encoder_type='frozen_backbone'."""
        raise NotImplementedError(
            "Scratch VideoEncoder (task_encoder_type='video') is not supported. "
            "Use task_encoder_type='frozen_backbone' with --frozen_backbone_type "
            "dinov2_vitl14 (or clip_vitl14 / siglip2_so400m / videomae_large)."
        )


    def _print_model_info(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backend = f"frozen_backbone ({self.args.frozen_backbone_type})"
        print(f"\n   ACTAgent ready")
        print(f"   Task encoder:  {backend}")
        print(f"   Input mode:    {self.input_mode.value}")
        print(f"   Total params:  {total/1e6:.2f}M  |  Trainable: {trainable/1e6:.2f}M")

    # ── Task encoding ──────────────────────────────────────────────────────────

    def _encode_task(
        self,
        robot_first_frame: torch.Tensor,
        human_video:   Optional[torch.Tensor] = None,
        human_desc:    Optional[List[str]]    = None,
        human_vl_ids:  Optional[List[str]]    = None,
    ) -> torch.Tensor:
        """Encode task → z: (B, task_latent_dim)."""
        # FrozenVideoBackbone / CachedFrozenBackbone:
        # Pass human_vl_ids as cache keys — this is the hot path (dict lookup).
        enc_out = self.task_encoder.encode(
            human_video=human_video,
            human_vl_ids=human_vl_ids,
        )
        z = enc_out["z"]
        return self.ftask_norm(z)

    # ── Observation preprocessing ──────────────────────────────────────────────

    def _preprocess_first_frame(self, first_frame_obs: Dict) -> torch.Tensor:
        images = first_frame_obs.get("view_1", first_frame_obs.get("rgb"))
        if images is None:
            raise ValueError("first_frame_obs must contain 'view_1' or 'rgb'.")
        if images.dim() == 5:
            images = images[:, 0]
        if images.dim() == 4 and images.shape[-1] in (3, 4):
            images = images.permute(0, 3, 1, 2)
        return images

    def _preprocess_obs_for_model(self, obs: Dict) -> Dict:
        """Convert env obs dict to DETRVAE-compatible format.

        Input:
          state: (B, state_dim) or (B, T, state_dim)  → (B, state_dim)
          rgb:   (B, C, H, W) or (B, num_cams, C, H, W)  → kept as-is
        """
        out = {}

        state = obs.get("state", obs.get("states"))
        if state.dim() == 3:
            state = state[:, -1, :]  # (B, T, D) → (B, D)
        out["state"] = state.float()

        rgb = obs.get("rgb", obs.get("view_1"))
        if rgb is None:
            raise ValueError("obs must contain 'rgb' or 'view_1'.")
        # Ensure (B, num_cams, C, H, W)
        if rgb.dim() == 4:
            if rgb.shape[-1] in (3, 4):
                # (B, H, W, C) NHWC → (B, C, H, W) NCHW
                # This case arises during eval: item_to_obs() returns
                # "rgb" as (1, H, W, C_total) in NHWC layout.
                # The training path never hits this branch because
                # PairedDataset already permutes view tensors to
                # (T, H, W, C), so batched views arrive as 5D here.
                rgb = rgb.permute(0, 3, 1, 2)
            # (B, C, H, W) → (B, 1, C, H, W)
            rgb = rgb.unsqueeze(1)
        elif rgb.dim() == 5 and rgb.shape[-1] in (3, 4):
            # (B, T, H, W, C) → (B, 1, C, H, W)
            rgb = rgb[:, -1].permute(0, 3, 1, 2).unsqueeze(1)
        out["rgb"] = rgb.float()

        if self.include_depth and "depth" in obs:
            depth = obs["depth"]
            if depth.dim() == 5 and depth.shape[-1] == 1:
                depth = depth[:, -1].permute(0, 3, 1, 2).unsqueeze(1)
            elif depth.dim() == 4:
                depth = depth.unsqueeze(1)
            out["depth"] = (depth.float() / 1024.0)

        return out

    # ── Training loss ──────────────────────────────────────────────────────────

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        robot_obs  = {k: v.to(self.device, non_blocking=True)
                      for k, v in batch["robot_obs"].items()}
        action_seq = batch["robot_actions"].to(self.device, non_blocking=True)

        first_frame_obs = {k: v.to(self.device, non_blocking=True)
                           for k, v in batch["robot_first_frame_obs"].items()}
        robot_first_frame = self._preprocess_first_frame(first_frame_obs)

        # ── Resolve inputs by mode ────────────────────────────────────────────
        human_video  = None
        human_desc   = None
        human_vl_ids = None

        if self.input_mode == InputMode.VIDEO_ONLY:
            human_video = batch.get("human_video")
            if human_video is not None:
                human_video = human_video.to(self.device, non_blocking=True)
            human_vl_ids = (batch.get("human_repo_id")
                            or batch.get("human_video_path")
                            or batch.get("human_episode_id"))

        elif self.input_mode == InputMode.LANGUAGE_ONLY:
            human_desc = batch.get("human_desc") or batch.get("language")

        elif self.input_mode == InputMode.VIDEO_AND_LANGUAGE:
            if self.random_modality_selection and self.training:
                if random.random() < self.use_video_prob:
                    human_video = batch.get("human_video")
                    if human_video is not None:
                        human_video = human_video.to(self.device, non_blocking=True)
                    human_vl_ids = (batch.get("human_repo_id")
                                    or batch.get("human_video_path"))
                else:
                    human_desc = batch.get("human_desc") or batch.get("language")
            else:
                human_video = batch.get("human_video")
                if human_video is not None:
                    human_video = human_video.to(self.device, non_blocking=True)
                human_desc   = batch.get("human_desc") or batch.get("language")
                human_vl_ids = (batch.get("human_repo_id")
                                or batch.get("human_video_path"))

        # ── Encode task ───────────────────────────────────────────────────────
        # Frozen backbone: backbone weights frozen, but adapter MUST receive grad.
        task_z = self._encode_task(
            robot_first_frame, human_video, human_desc, human_vl_ids
        )  # (B, task_latent_dim)

        # ── Preprocess obs ────────────────────────────────────────────────────
        obs_for_model = self._preprocess_obs_for_model(robot_obs)

        # ── Forward pass (VAE training mode: actions provided) ────────────────
        a_hat, (mu, logvar) = self.model(
            obs_for_model, action_seq, video_feature=task_z
        )  # a_hat: (B, pred_horizon, action_dim)

        # ── L1 reconstruction loss ────────────────────────────────────────────
        l1_loss = F.l1_loss(action_seq.float(), a_hat.float())

        # ── KL divergence loss ────────────────────────────────────────────────
        mu_fp32     = mu.float()
        logvar_fp32 = logvar.float()
        klds        = -0.5 * (1 + logvar_fp32 - mu_fp32.pow(2) - logvar_fp32.exp())
        total_kld   = klds.sum(1).mean(0, True)[0]

        loss = l1_loss + total_kld * self.kl_weight

        return {"loss": loss, "l1": l1_loss, "kl": total_kld}

    # ── Evaluation ────────────────────────────────────────────────────────────

    def prepare_for_eval(
        self,
        human_video:   Optional[torch.Tensor],
        robot_obs:     Dict,
        human_tokens:  Optional[torch.Tensor] = None,   # unused, kept for API compat
        human_desc:    Optional[List[str]]    = None,
        human_vl_ids:  Optional[List[str]]   = None, 
    ):
        """Encode task ONCE per episode and cache."""
        self.eval()
        with torch.no_grad():
            if human_video is not None:
                human_video = human_video.to(self.device)
            robot_obs = {k: v.to(self.device) for k, v in robot_obs.items()}
            robot_first_frame = self._preprocess_first_frame(robot_obs)
            self._cached_task_z = self._encode_task(
                robot_first_frame, human_video, human_desc, human_vl_ids
            )

    def clear_cache(self):
        self._cached_task_z = None

    @torch.no_grad()
    def get_action(self, obs: Dict) -> torch.Tensor:
        """Single forward pass. Returns (B, pred_horizon, action_dim)."""
        assert self._cached_task_z is not None, (
            "Call prepare_for_eval() before get_action()."
        )
        obs_for_model = self._preprocess_obs_for_model(obs)
        B = obs_for_model["state"].shape[0]
        task_z = self._cached_task_z.expand(B, -1)

        # Inference mode: actions=None → DETRVAE uses zero latent
        a_hat, _ = self.model(obs_for_model, actions=None, video_feature=task_z)
        # a_hat: (B, pred_horizon, action_dim)  ✅
        return a_hat


# ============================================================================
# Evaluation loop  (for in-training eval on a single env)
# ============================================================================

def evaluate(n: int, agent, eval_envs, eval_kwargs, evaluate_processor, progress_bar=True):
    import gymnasium
    from mani_skill.utils import common

    env_id        = eval_kwargs.get("env_id")
    delta_control = eval_kwargs.get("delta_control")
    pred_horizon  = eval_kwargs.get("pred_horizon")
    temporal_agg  = eval_kwargs.get("temporal_agg")
    max_timesteps = eval_kwargs.get("max_timesteps")
    device        = eval_kwargs.get("device")
    sim_backend   = eval_kwargs.get("sim_backend")

    if isinstance(eval_envs.single_observation_space, gymnasium.spaces.Box):
        action_dim = eval_envs.action_space.shape[-1]
    else:
        action_dim = (eval_envs.action_space["panda_wristcam-0"].shape[-1] +
                      eval_envs.action_space["panda_wristcam-1"].shape[-1])

    num_envs = eval_envs.num_envs

    if temporal_agg:
        query_frequency  = 1
        all_time_actions = torch.zeros(
            [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim],
            device=device,
        )
    else:
        query_frequency  = pred_horizon
        actions_to_take  = torch.zeros([num_envs, pred_horizon, action_dim], device=device)

    agent.eval()
    val_videos = evaluate_processor.get_video(env_id, num_envs).to(device)

    with torch.no_grad():
        eval_metrics  = defaultdict(list)
        obs, info     = eval_envs.reset()
        ts, eps_count = 0, 0
        pbar = tqdm(total=n, desc="Evaluating", disable=not progress_bar, unit="episode")

        while eps_count < n:
            obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
            if not delta_control:
                obs["state"], obs["rgb"] = evaluate_processor.normalize_state_rgb(
                    obs["state"], obs["rgb"], env_id
                )

            if ts == 0:
                agent.prepare_for_eval(
                    human_video=val_videos,
                    robot_obs={
                        "states": obs.get("state",  obs.get("states")),
                        "view_1": obs.get("rgb",    obs.get("view_1")),
                    },
                )

            if ts % query_frequency == 0:
                action_seq = agent.get_action(obs)  # (num_envs, pred_horizon, action_dim)
                if action_seq.dim() == 2:
                    action_seq = action_seq.unsqueeze(0)

            if temporal_agg:
                all_time_actions[:, ts, ts:ts + pred_horizon] = action_seq
                actions_for_curr_step = all_time_actions[:, :, ts]
                populated = torch.zeros(max_timesteps, dtype=torch.bool, device=device)
                populated[max(0, ts + 1 - pred_horizon):ts + 1] = True
                actions_for_curr_step = actions_for_curr_step[:, populated]
                k = 0.01
                exp_w = torch.exp(
                    -k * torch.arange(actions_for_curr_step.shape[1], device=device)
                )
                exp_w = (exp_w / exp_w.sum()).unsqueeze(0).unsqueeze(-1).expand(num_envs, -1, -1)
                raw_action = (actions_for_curr_step * exp_w).sum(dim=1)
            else:
                if ts % query_frequency == 0:
                    actions_to_take = action_seq
                raw_action = actions_to_take[:, ts % query_frequency]  # (num_envs, action_dim)

            _action = (evaluate_processor.denormalize_action(raw_action, env_id)
                       if not delta_control else raw_action)
            if sim_backend == "physx_cpu":
                _action = _action.cpu().numpy()

            action = {
                "panda_wristcam-0": _action[:, :8],
                "panda_wristcam-1": _action[:, 8:16],
            }
            obs, rew, terminated, truncated, info = eval_envs.step(action)
            ts += 1

            if truncated.any():
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for fi in info["final_info"]:
                        for k, v in fi["episode"].items():
                            eval_metrics[k].append(v)
                pbar.update(num_envs)
                eps_count += num_envs
                ts = 0
                if temporal_agg:
                    all_time_actions = torch.zeros(
                        [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim],
                        device=device,
                    )
                agent.clear_cache()
                obs, info = eval_envs.reset()

        pbar.close()

    agent.train()
    for k in eval_metrics:
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics


# ============================================================================
# Checkpoint helpers
# ============================================================================

def save_checkpoint(path, iteration, agent, optimizer, lr_scheduler,
                    best_eval_metrics, args=None, epoch=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "iteration":               iteration,
        "agent_state_dict":        agent.state_dict(),
        "optimizer_state_dict":    optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "best_eval_metrics":       dict(best_eval_metrics),
    }
    if epoch is not None:
        ckpt["epoch"] = epoch
    if args is not None:
        ckpt["args"] = vars(args)
    torch.save(ckpt, path)
    print(f"  💾 Saved → {path}")


def load_checkpoint(path, agent, optimizer, lr_scheduler, scaler, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    agent_sd = ckpt.get("agent_state_dict") or ckpt.get("agent")
    if agent_sd is not None:
        # Strip CachedQwenVLEncoder 'original' prefix if present
        fixed = {}
        for k, v in agent_sd.items():
            new_k = k.replace(".video_encoder.original.", ".video_encoder.") \
                      .replace(".lang_encoder.original.", ".lang_encoder.")
            fixed[new_k] = v
        agent.load_state_dict(fixed, strict=False)

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device, non_blocking=True)

    if lr_scheduler is not None and "lr_scheduler_state_dict" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])

    if "scaler_state_dict" in ckpt and scaler is not None:
        saved = ckpt["scaler_state_dict"]
        saved["scale"] = min(float(saved.get("scale", 65536)), 1024.0)
        scaler.load_state_dict(saved)

    iteration         = ckpt.get("iteration", 0)
    epoch             = ckpt.get("epoch", 0)
    best_eval_metrics = defaultdict(float, ckpt.get("best_eval_metrics", {}))
    print(f"  📥 Resumed from {path}  (epoch={epoch}, iter={iteration})")
    return iteration, epoch, best_eval_metrics


def _run_evaluation(args, agent, eval_envs, evaluate_processor, device):
    return evaluate(
        n=args.num_eval_episodes,
        agent=agent,
        eval_envs=eval_envs,
        eval_kwargs=dict(
            env_id=args.env_id,
            delta_control=("delta" in args.control_mode),
            pred_horizon=args.pred_horizon,
            temporal_agg=args.eval_temporal_agg,
            max_timesteps=args.max_episode_steps,
            device=device,
            sim_backend=args.sim_backend,
        ),
        evaluate_processor=evaluate_processor,
        progress_bar=True,
    )


# ============================================================================
# Main training entry point
# ============================================================================

def train():
    args = tyro.cli(TrainingArgs)

    if args.exp_name is None:
        run_name = f"{args.env_id}__act__{args.seed}__{int(time.time())}"
    else:
        from datetime import datetime
        run_name = f"{args.exp_name}-{datetime.now().strftime('%Y%m%d')}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    _use_amp   = torch.cuda.is_available()
    _amp_dtype = torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(_amp_dtype == torch.float16 and _use_amp))

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            sync_tensorboard=True, name=run_name, config=vars(args), save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_config = PairedDatasetConfig(
        human_root=args.human_root,
        sim_root=args.sim_root,
        robot_root=args.robot_root,
        task_mapping_file=args.task_mapping_file,
        human_dataset_file=args.human_dataset_file,
        sim_dataset_file=args.sim_dataset_file,
        robot_dataset_file=args.robot_dataset_file,
        human_task_description_file=args.human_task_description_file,
        sim_task_description_file=args.sim_task_description_file,
        robot_task_description_file=args.robot_task_description_file,
        split="train",
        cameras=args.cameras,
        include_depth=args.include_depth,
        image_size=args.image_size,
        num_frames=args.task_num_frames,
        horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        state_type=args.state_type,
        single_arm=args.single_arm,
        fps=args.fps,
        video_backend=args.video_backend,
        input_mode=args.input_mode,
        include_first_frame=args.include_first_frame,
    )
    if args.data_source == "robot":
        from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
            HumanRobotPairedDataset,
        )
        print(f"\n[DataSource] Real-robot mode: robot_root={args.robot_root}")
        dataset = HumanRobotPairedDataset(dataset_config)
    else:
        dataset = HumanSimPairedDataset(dataset_config)
    print(f"\nDataset size: {len(dataset)}")

    # ── Evaluation processor ──────────────────────────────────────────────────
    if not args.no_eval:
        from examples.baselines.lerobot_dataset.evaluate_processor import (
            HumanVideoSimEvaluateProcessor,
            HumanVideoSimEvaluateProcessorConfig,
        )
        evaluate_processor = HumanVideoSimEvaluateProcessor(
            HumanVideoSimEvaluateProcessorConfig(
                human_root=args.human_root,
                human_split="train",
                human_dataset_file=args.human_dataset_file,
                human_task_description_file=args.human_task_description_file,
                human_cameras=args.cameras,
                human_include_depth=args.include_depth,
                human_num_frames=args.task_num_frames,
                human_image_size=args.image_size,
                human_video_backend=args.video_backend,
                human_fps=args.fps,
                sim_root=args.sim_root,
                sim_split="train",
                sim_dataset_file=args.sim_dataset_file,
                sim_task_description_file=args.sim_task_description_file,
                sim_state_type=args.state_type,
                sim_single_arm=args.single_arm,
                normalization_method="bounds_q99",
            )
        )
    else:
        evaluate_processor = None

    # ── DataLoader ────────────────────────────────────────────────────────────
    collate_fn = get_collate_fn(args.input_mode)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_dataload_workers,
        collate_fn=collate_fn,
        persistent_workers=(args.num_dataload_workers > 0),
        prefetch_factor=4 if args.num_dataload_workers > 0 else None,
        pin_memory=True,
        multiprocessing_context="forkserver" if args.num_dataload_workers > 0 else None,
    )
    num_batches_per_epoch = len(dataloader)
    print(f"Batches per epoch: {num_batches_per_epoch}")

    # ── Eval environments ─────────────────────────────────────────────────────
    if not args.no_eval:
        from examples.baselines.act.act.make_env import make_eval_envs
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
        eval_envs = make_eval_envs(
            args.env_id, args.num_eval_envs, args.sim_backend,
            dict(control_mode=args.control_mode, reward_mode="dense",
                obs_mode="rgbd" if args.include_depth else "rgb",
                render_mode="rgb_array", max_episode_steps=args.max_episode_steps, sensor_configs=dict(shader_pack="rt-fast"), human_render_camera_configs=dict(shader_pack="rt-fast"),
                ),
            dict(obs_horizon=args.obs_horizon),
            video_dir=f"runs/{run_name}/videos",
            wrappers=[FlattenRGBDObservationWrapper],
        )
    else:
        eval_envs = None

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = ACTAgent(args, device, dataloader=dataloader)

    # Recreate DataLoader if skip_human_video was activated
    if args.num_dataload_workers > 0 and dataset.skip_human_video:
        print("  Recreating DataLoader for skip_human_video ...")
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_dataload_workers, collate_fn=collate_fn,
            persistent_workers=True, prefetch_factor=4, pin_memory=True,
            multiprocessing_context="forkserver",
        )
        num_batches_per_epoch = len(dataloader)

    # ── Optimizer (separate lr for backbone) ─────────────────────────────────
    backbone_params = [p for n, p in agent.named_parameters()
                       if "backbone" in n and p.requires_grad]
    other_params    = [p for n, p in agent.named_parameters()
                       if "backbone" not in n and p.requires_grad]
    optimizer = optim.AdamW(
        [{"params": other_params}, {"params": backbone_params, "lr": args.lr_backbone}],
        lr=args.lr, weight_decay=1e-4,
    )

    total_steps  = (args.total_epochs * num_batches_per_epoch
                    if args.use_epoch_training else args.total_iters // 5)
    warmup_steps = (args.warmup_epochs * num_batches_per_epoch
                    if args.use_epoch_training else 1000)
    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    start_iteration   = 0
    start_epoch       = 0
    best_eval_metrics = defaultdict(float)

    if args.resume_from and os.path.exists(args.resume_from):
        start_iteration, start_epoch, best_eval_metrics = load_checkpoint(
            args.resume_from, agent, optimizer, lr_scheduler, scaler, device
        )

    print(f"\nStarting training  (skip_video: {dataset.skip_human_video})")
    agent.train()
    iteration = start_iteration

    # ── Training loop ─────────────────────────────────────────────────────────
    if args.use_epoch_training:
        global_step = start_epoch * num_batches_per_epoch

        for epoch in tqdm(range(start_epoch, args.total_epochs), desc="Epochs"):

            epoch_loss  = 0.0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
                with torch.autocast("cuda", dtype=_amp_dtype, enabled=_use_amp):
                    loss_dict = agent.compute_loss(batch)

                total_loss = loss_dict["loss"]
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)

                # Tighter clip for backbone
                backbone_ps = [p for n, p in agent.named_parameters()
                               if "backbone" in n and p.grad is not None]
                other_ps    = [p for n, p in agent.named_parameters()
                               if "backbone" not in n and p.grad is not None]
                torch.nn.utils.clip_grad_norm_(backbone_ps, max_norm=0.1)
                torch.nn.utils.clip_grad_norm_(other_ps,    max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                epoch_loss  += total_loss.item()
                num_batches += 1
                global_step += 1
                iteration   += 1

                if iteration % args.log_freq == 0:
                    cs = agent.task_encoder.cache_stats(batch.get("human_repo_id"))
                    writer.add_scalar("train/loss", total_loss.item(), iteration)
                    writer.add_scalar("train/l1",   loss_dict["l1"].item(), iteration)
                    writer.add_scalar("train/kl",   loss_dict["kl"].item(), iteration)
                    if cs != "(no cache)":
                        print(f"[iter {iteration}] loss={total_loss.item():.4f} | {cs}")
                    if args.track:
                        import wandb
                        wandb.log({"train/loss": total_loss.item(),
                                   "train/l1": loss_dict["l1"].item(),
                                   "train/kl": loss_dict["kl"].item()},
                                  step=global_step)

            avg_loss = epoch_loss / max(num_batches, 1)
            if epoch % args.log_epoch_freq == 0:
                writer.add_scalar("train_epoch/loss", avg_loss, epoch)
                writer.add_scalar("train_epoch/lr",   optimizer.param_groups[0]["lr"], epoch)
                print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

            if epoch % args.eval_epoch_freq == 0 and epoch > 0 and not args.no_eval:
                eval_metrics = _run_evaluation(args, agent, eval_envs, evaluate_processor, device)
                for k, v in eval_metrics.items():
                    writer.add_scalar(f"eval/{k}_mean", v.mean(), epoch)
                    print(f"  eval/{k}: {v.mean():.4f} +/- {v.std():.4f}")
                for k in ("success_once", "success_at_end"):
                    if k in eval_metrics and eval_metrics[k].mean() > best_eval_metrics[k]:
                        best_eval_metrics[k] = eval_metrics[k].mean()
                        save_checkpoint(
                            f"runs/{run_name}/checkpoints/best_model.pt",
                            iteration, agent, optimizer, lr_scheduler,
                            best_eval_metrics, args, epoch,
                        )
                agent.train()

            if epoch % args.save_epoch_freq == 0 and epoch > 0:
                save_checkpoint(
                    f"runs/{run_name}/checkpoints/epoch_{epoch}.pt",
                    iteration, agent, optimizer, lr_scheduler,
                    best_eval_metrics, args, epoch,
                )

    else:
        pbar = tqdm(total=args.total_iters - start_iteration)
        for epoch in range(10000):
            for batch in dataloader:
                if iteration >= args.total_iters:
                    break

                with torch.autocast("cuda", dtype=_amp_dtype, enabled=_use_amp):
                    loss_dict = agent.compute_loss(batch)

                total_loss = loss_dict["loss"]
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in agent.parameters() if p.grad is not None], max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                if iteration % args.log_freq == 0:
                    writer.add_scalar("train/loss", total_loss.item(), iteration)
                if iteration % args.eval_freq == 0 and iteration > 0 and not args.no_eval:
                    eval_metrics = _run_evaluation(args, agent, eval_envs, evaluate_processor, device)
                    for k, v in eval_metrics.items():
                        writer.add_scalar(f"eval/{k}_mean", v.mean(), iteration)
                    for k in ("success_once", "success_at_end"):
                        if k in eval_metrics and eval_metrics[k].mean() > best_eval_metrics[k]:
                            best_eval_metrics[k] = eval_metrics[k].mean()
                            save_checkpoint(
                                f"runs/{run_name}/checkpoints/best_model.pt",
                                iteration, agent, optimizer, lr_scheduler,
                                best_eval_metrics, args,
                            )
                    agent.train()

                if args.save_freq and iteration % args.save_freq == 0 and iteration > 0:
                    save_checkpoint(
                        f"runs/{run_name}/checkpoints/iter_{iteration}.pt",
                        iteration, agent, optimizer, lr_scheduler,
                        best_eval_metrics, args,
                    )

                iteration += 1
                pbar.update(1)
                pbar.set_description(f"Loss: {total_loss.item():.4f}")

            if iteration >= args.total_iters:
                break
        pbar.close()

    save_checkpoint(
        f"runs/{run_name}/checkpoints/final_model.pt",
        iteration, agent, optimizer, lr_scheduler, best_eval_metrics, args,
        args.total_epochs if args.use_epoch_training else None,
    )
    print(f"\nTraining complete.  Best metrics: {dict(best_eval_metrics)}")

    if args.track:
        import wandb
        wandb.finish()
    if eval_envs is not None:
        eval_envs.close()
    writer.close()


if __name__ == "__main__":
    train()