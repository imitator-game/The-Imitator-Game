"""
Training Script for VQ-BeT
"""

ALGO_NAME = "VQBeT_FoundationTaskEncoder"

import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import einops
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import tyro

from examples.baselines.vqbet.vqbet.gpt import GPT, GPTConfig
from examples.baselines.vqbet.vqbet.utils import MLP
from examples.baselines.vqbet.vqbet.vqvae import VqVae

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanSimPairedDataset,
    PairedDatasetConfig,
    InputMode,
    get_collate_fn,
)
from examples.baselines.encoders.obs_encoder import ObservationEncoder, ObsEncoderConfig
from examples.baselines.encoders.state_encoder import StateEncoder, StateEncoderConfig


# =============================================================================
# Training Arguments
# =============================================================================

@dataclass
class TrainingArgs:
    """Training configuration for VQ-BeT with a frozen-backbone task encoder."""

    # Experiment
    exp_name: Optional[str] = None
    seed: int = 1
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "VQBeT_FoundationTE"
    wandb_entity: Optional[str] = None

    # Data paths
    human_root: str = "demos"
    sim_root: str = "demos"
    task_mapping_file: str = "examples/baselines/lerobot_dataset/task_mapping.json"
    human_dataset_file: Optional[str] = "examples/baselines/lerobot_dataset/config/human_config.json"
    sim_dataset_file: Optional[str] = "examples/baselines/lerobot_dataset/config/sim_config.json"
    human_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    sim_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"
    # Robot data (real-robot training)
    data_source:                 str          = "sim"    # "sim" | "robot"
    robot_root:                  str          = "demos"
    robot_dataset_file:          Optional[str] = None
    robot_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/robot_desc.json"

    # Training
    use_epoch_training: bool = True
    total_epochs: int = 100
    total_iters: int = 500_000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    num_dataload_workers: int = 0
    warmup_epochs: int = 5
    max_grad_norm: float = 1.0

    # AMP / precision
    use_amp: bool = True
    """Enable bfloat16 autocast + GradScaler for ~1.5-2x throughput."""
    use_tf32: bool = True
    """Enable TF32 on Ampere+ GPUs (free ~10 %)."""

    # Model architecture
    action_dim: int = 16
    state_dim: int = 18
    obs_horizon: int = 1
    pred_horizon: int = 16
    obs_window_size: int = 10
    act_window_size: int = 10

    # VQ-VAE parameters
    vqvae_n_latent_dims: int = 512
    vqvae_n_embed: int = 32
    vqvae_groups: int = 2
    vqvae_ckpt: Optional[str] = None
    vqvae_pretrain_epochs: int = 50

    # GPT parameters
    gpt_n_layer: int = 24
    gpt_n_head: int = 16
    gpt_n_embd: int = 1280
    gpt_block_size: int = 100
    gpt_dropout: float = 0.1

    # Loss parameters
    offset_loss_multiplier: float = 1e3
    secondary_code_multiplier: float = 0.5
    focal_loss_gamma: float = 2.0

    # Observation settings
    include_depth: bool = False
    cameras: List[str] = field(default_factory=lambda: ["zed2i"])
    image_size: Tuple[int, int] = (224, 224)

    # Task Encoder parameters
    # "frozen_backbone" → FrozenVideoBackbone (off-the-shelf frozen backbone)
    #                     e.g. DINOv2, CLIP, SigLIP, VideoMAE
    task_encoder_type: str = "frozen_backbone"

    # ── FrozenVideoBackbone settings (task_encoder_type == "frozen_backbone") ─
    frozen_backbone_type:           str          = "dinov2_vitl14"
    frozen_backbone_model:          Optional[str] = None
    frozen_backbone_adapter_layers: int          = 1
    frozen_backbone_seq_patches:    int          = 32
    frozen_backbone_num_frames:     int          = 4

    # ── DEPRECATED: VideoEncoder (scratch) — kept for backward compat ─────────
    video_encoder_type:       str = "temporal_transformer"
    video_encoder_hidden_dim: int = 512
    video_encoder_num_layers: int = 4
    video_encoder_num_heads:  int = 8

    task_encoder_ckpt_path: Optional[str] = None
    task_latent_dim: int = 256
    task_semantic_dim: int = 128
    task_skill_dim: int = 128
    task_hidden_dim: int = 512
    task_proj_dim: int = 128
    task_seq_len: int = 10
    task_num_frames: int = 10
    task_num_encoder_layers: int = 6
    task_num_heads: int = 8
    task_use_vae: bool = False
    task_kl_weight: float = 1e-4
    task_use_hierarchical_repr: bool = True
    task_use_adaptive_fusion: bool = True
    task_use_level_encoding: bool = True
    task_lora_rank: int = 16
    task_lora_alpha: float = 32.0
    qwen_vl_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    qwen_vl_use_4bit: bool = True
    qwen2_decoder_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    qwen2_decoder_use_4bit: bool = True
    num_tasks: int = 200
    freeze_task_encoder: bool = True

    te_cache_root: Optional[str] = "./te_cache"
    """Root dir for per-checkpoint VL + task-embedding cache sub-dirs."""
    te_cache_preload_memory: bool = True
    """Bulk-load VL + task-embedding caches into RAM at startup."""
    enable_task_emb_cache: bool = True
    """Enable L3 TaskEmbeddingCache. Once all tasks are cached encode()~=0 ms."""
    te_cache_recompute: bool = False
    """Force re-computation of the VL disk cache even if it already exists."""
    te_cache_recompute_task_emb: bool = False
    """Force re-computation of the L3 task-embedding cache."""

    # Input mode
    input_mode: str = "video_only"
    random_modality_selection: bool = True
    use_video_prob: float = 0.5
    include_first_frame: bool = True

    # Obs encoder
    obs_encoder_type: str = "simple_cnn"
    obs_latent_dim: int = 256
    obs_freeze_backbone: bool = False
    obs_finetune_layers: int = 0

    # State encoder
    state_encoder_type: str = "mlp"
    state_latent_dim: int = 256
    state_hidden_dim: int = 256
    state_num_layers: int = 4
    state_num_heads: int = 4

    # Video conditioning
    video_conditioning_mode: str = "concat"

    # Evaluation / logging
    log_freq: int = 10
    log_epoch_freq: int = 1
    eval_epoch_freq: int = 5
    save_epoch_freq: int = 10
    num_eval_episodes: int = 10
    num_eval_envs: int = 1
    eval_temporal_agg: bool = True
    no_eval: bool = True

    # Environment
    env_id: str = "TwoRobotStirSpoon-v1"
    max_episode_steps: int = 500
    sim_backend: str = "physx_cpu"
    control_mode: str = "pd_joint_pos"
    obs_mode: str = "rgb"
    shader: str = "rt-fast"
    state_type: str = "qpos"
    single_arm: bool = False
    fps: int = 30
    video_backend: str = "torchcodec"

    # Checkpointing
    resume_from: Optional[str] = None
    resume_epoch: Optional[int] = None
    """Override start epoch when resuming (checkpoint value used by default)."""
    resume_iteration: Optional[int] = None
    """Override global iteration when resuming."""


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in VQ code prediction."""
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()
        loss = -1 * (1 - pt) ** self.gamma * logpt
        return loss.mean() if self.size_average else loss.sum()


# =============================================================================
# VQBeTAgent
# =============================================================================

class VQBeTAgent(nn.Module):
    """
    VQ-BeT Agent backed by a frozen backbone (FrozenVideoBackbone) with caching.

    Key changes vs the original script
    ------------------------------------
    1. _encode_task forwards human_vl_ids as the raw-feature cache key.
    2. compute_loss extracts batch["human_repo_id"] and passes it through.
    3. pretrain_vqvae uses non-blocking transfers for speed.
    4. AMP, TF32, zero_grad(set_to_none=True) in the training loop.
    """

    COND_MODE = Enum("COND_MODE", "concat stack unconditional")

    def __init__(self, args: TrainingArgs, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.obs_window_size = args.obs_window_size
        self.act_window_size = args.act_window_size
        self.action_dim = args.action_dim
        self.obs_horizon = args.obs_horizon
        self.input_mode = InputMode(args.input_mode)
        self.random_modality_selection = args.random_modality_selection
        self.use_video_prob = args.use_video_prob

        if args.video_conditioning_mode == "concat":
            self._cond_mode = self.COND_MODE.concat
        elif args.video_conditioning_mode == "stack":
            self._cond_mode = self.COND_MODE.stack
        else:
            self._cond_mode = self.COND_MODE.unconditional

        self._use_frozen_backbone = (args.task_encoder_type == "frozen_backbone")

        # ── Task Encoder (FrozenVideoBackbone only) ───────────────────────────
        if args.task_encoder_type == "video":
            import warnings
            warnings.warn(
                "task_encoder_type='video' is deprecated. "
                "Using frozen_backbone instead.", DeprecationWarning)
        from examples.baselines.encoders.task_encoder.video_backbone import (
            build_video_backbone,
        )
        print(f"\n🎬 Creating FrozenVideoBackbone ({args.frozen_backbone_type}) ...")
        self.task_encoder = build_video_backbone(
            backbone_type=args.frozen_backbone_type,
            latent_dim=args.task_latent_dim,
            max_seq_patches=args.frozen_backbone_seq_patches,
            adapter_layers=args.frozen_backbone_adapter_layers,
            num_sampled_frames=args.frozen_backbone_num_frames,
            hf_cache_dir=getattr(args, "hf_cache_dir", None),
        ).to(device)
        # Cache setup requires a DataLoader; call setup_frozen_backbone_cache()
        # from the training script after building the DataLoader.
        self._te_needs_cache_setup = True   # signal to training loop

        # ── Observation Encoder ───────────────────────────────────────────────
        print(f"\n🔍 Creating Observation Encoder ({args.obs_encoder_type})...")
        self.obs_encoder = ObservationEncoder(ObsEncoderConfig(
            encoder_type=args.obs_encoder_type,
            image_size=args.image_size[0],
            output_dim=args.obs_latent_dim,
            hidden_dim=args.obs_latent_dim,
            freeze_backbone=args.obs_freeze_backbone,
            finetune_layers=args.obs_finetune_layers,
        )).to(device)

        # ── State Encoder ─────────────────────────────────────────────────────
        print(f"\n📊 Creating State Encoder ({args.state_encoder_type})...")
        self.state_encoder = StateEncoder(StateEncoderConfig(
            state_type=args.state_type,
            state_dim=args.state_dim,
            num_frames=args.obs_horizon,
            encoder_type=args.state_encoder_type,
            hidden_dim=args.state_hidden_dim,
            output_dim=args.state_latent_dim,
            num_layers=args.state_num_layers,
            num_heads=args.state_num_heads,
        )).to(device)
        self._obs_cond_dim = args.obs_latent_dim + args.state_latent_dim

        # ── Task feature projector ────────────────────────────────────────────
        self.task_projector = nn.Linear(args.task_latent_dim, self._obs_cond_dim).to(device)

        # ── VQ-VAE ───────────────────────────────────────────────────────────
        print("\n🔢 Creating VQ-VAE...")
        self.vqvae = VqVae(
            obs_dim=args.state_dim,
            input_dim_h=args.act_window_size,
            input_dim_w=args.action_dim,
            n_latent_dims=args.vqvae_n_latent_dims,
            vqvae_n_embed=args.vqvae_n_embed,
            vqvae_groups=args.vqvae_groups,
            eval=False,
            device=device,
            load_dir=args.vqvae_ckpt,
        )
        self._G = self.vqvae.vqvae_groups
        self._C = self.vqvae.vqvae_n_embed
        self._D = self.vqvae.embedding_dim

        # ── GPT ──────────────────────────────────────────────────────────────
        gpt_input_dim = (
            self._obs_cond_dim if self._cond_mode != self.COND_MODE.stack
            else args.task_latent_dim + self._obs_cond_dim
        )
        print(f"\n🧠 Creating GPT (input_dim={gpt_input_dim})...")
        self.gpt = GPT(GPTConfig(
            block_size=args.gpt_block_size,
            input_dim=gpt_input_dim,
            output_dim=args.gpt_n_embd,
            n_layer=args.gpt_n_layer,
            n_head=args.gpt_n_head,
            n_embd=args.gpt_n_embd,
            dropout=args.gpt_dropout,
        )).to(device)

        # ── Prediction Heads ──────────────────────────────────────────────────
        self.code_predictor = MLP(
            in_channels=args.gpt_n_embd,
            hidden_channels=[1024, 1024, self._G * self._C],
        ).to(device)
        self.offset_predictor = MLP(
            in_channels=args.gpt_n_embd,
            hidden_channels=[
                1024, 1024,
                self._G * self._C * (args.action_dim * args.act_window_size),
            ],
        ).to(device)

        self._criterion = FocalLoss(gamma=args.focal_loss_gamma)
        self._offset_loss_multiplier = args.offset_loss_multiplier
        self._secondary_code_multiplier = args.secondary_code_multiplier
        self._cached_task_features = None
        self._print_model_info()

    def _build_video_encoder(self, args, device):
        """DEPRECATED: scratch VideoEncoder removed. Use task_encoder_type='frozen_backbone'."""
        raise NotImplementedError(
            "task_encoder_type='video' removed. Use 'frozen_backbone' instead."
        )

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _print_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        te = sum(p.numel() for p in self.task_encoder.parameters())
        print(f"\n✅ VQBeTAgent initialised")
        print(f"   Input mode   : {self.input_mode.value}")
        print(f"   Conditioning : {self._cond_mode.name}")
        print(f"   Task encoder : frozen_backbone ({self.args.frozen_backbone_type})  {te/1e6:.2f}M (FROZEN backbone + TRAINABLE adapter)")
        print(f"   Total params : {total/1e6:.2f}M  |  Trainable: {trainable/1e6:.2f}M")

    # ── Task encoding ─────────────────────────────────────────────────────────

    def _encode_task(
        self,
        robot_first_frame: torch.Tensor,
        human_video: Optional[torch.Tensor] = None,
        human_desc: Optional[List[str]] = None,
        human_vl_ids: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode task -> (z [B,D], z_seq [B,T,D]).

        frozen_backbone path: backbone is always frozen; adapter gets gradients.
        """
        # Backbone weights are frozen; the adapter must receive gradients → do
        # NOT wrap the call in no_grad.
        enc = self.task_encoder.encode(
            human_video=human_video,
            human_vl_ids=human_vl_ids,
        )
        return enc["z"], enc.get("z_seq", enc["z"].unsqueeze(1))

    # ── Observation preprocessing ─────────────────────────────────────────────

    def _preprocess_first_frame(self, first_frame_obs: Dict) -> torch.Tensor:
        images = first_frame_obs.get("view_1", first_frame_obs.get("rgb"))
        if images is None:
            raise ValueError("No image found in first_frame_obs")
        if images.dim() == 5:
            images = images[:, 0]
        if images.dim() == 4 and images.shape[-1] in (3, 4):
            images = images.permute(0, 3, 1, 2)
        return images

    def _get_obs_condition(self, obs_images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        T = min(states.shape[1], self.obs_horizon) if states.dim() == 3 else 1
        if obs_images.dim() == 5:
            obs_images = obs_images[:, :T].permute(0, 1, 4, 2, 3)
            if T == 1:
                obs_images = obs_images.squeeze(1)
        elif obs_images.dim() == 4:
            obs_images = obs_images.permute(0, 3, 1, 2)
        obs_feat, _ = self.obs_encoder(obs_images)
        states_in = states[:, :T] if states.dim() == 3 else states.unsqueeze(1)
        state_feat, _ = self.state_encoder(states_in)
        return torch.cat([obs_feat, state_feat], dim=-1)

    def _preprocess_obs(self, obs_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        states = obs_dict.get("state", obs_dict.get("states")).to(self.device)
        views = []
        for i in range(1, 10):
            if f"view_{i}" in obs_dict:
                views.append(obs_dict[f"view_{i}"].to(self.device))
        if not views and "rgb" in obs_dict:
            views.append(obs_dict["rgb"].to(self.device))
        images = views[0]
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if states.dim() == 2:
            states = states.unsqueeze(1)
        return images, states

    def _pad_obs_sequence(self, obs_seq: torch.Tensor) -> torch.Tensor:
        if obs_seq.shape[1] < self.obs_window_size:
            pad = obs_seq[:, 0:1].expand(-1, self.obs_window_size - obs_seq.shape[1], -1)
            obs_seq = torch.cat([pad, obs_seq], dim=1)
        return obs_seq

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        obs_seq,
        task_feat,
        action_seq=None,
    ):
        """
        Fixed VQBeTAgent.forward — drop-in replacement.
    
        Changes vs original
        -------------------
        1. probs.float() before torch.multinomial  [CRASH FIX — bfloat16 unsupported]
        2. predicted.float() / flat_acts.float()   [CRASH / stability FIX]
        3. cbet_loss kept in its natural dtype      [unchanged — classification loss
                                                    is already stable in bfloat16]
        """
        N = obs_seq.shape[0]
        obs_seq = self._pad_obs_sequence(obs_seq)
        T_obs = obs_seq.shape[1]
    
        if self._cond_mode == self.COND_MODE.unconditional:
            gpt_input, task_len = obs_seq, 0
        elif self._cond_mode == self.COND_MODE.concat:
            tp = self.task_projector(task_feat)
            if tp.dim() == 2:
                tp = tp.unsqueeze(1)
            gpt_input = torch.cat([tp, obs_seq], dim=1)
            task_len = tp.shape[1]
        else:  # stack
            te = task_feat.unsqueeze(1).expand(-1, T_obs, -1)
            gpt_input = torch.cat([te, obs_seq], dim=-1)
            task_len = 0
    
        gpt_out = self.gpt(gpt_input)
        if self._cond_mode != self.COND_MODE.unconditional and task_len > 0:
            gpt_out = gpt_out[:, task_len:]
    
        NT = N * T_obs
        flat = einops.rearrange(gpt_out, "N T D -> (N T) D")
        logits = einops.rearrange(self.code_predictor(flat),
                                "NT (G C) -> NT G C", G=self._G)
        offsets = einops.rearrange(
            self.offset_predictor(flat),
            "NT (G C WA) -> NT G C WA", G=self._G, C=self._C,
        )
    
        probs = torch.softmax(logits, dim=-1)
    
        # ── FIX A: torch.multinomial only supports float32 / float64. ────────────
        # Under bfloat16 AMP, probs is bfloat16 here → crash without the cast.
        probs_fp32 = probs.float()
        sampled = einops.rearrange(
            torch.multinomial(probs_fp32.view(-1, self._C), num_samples=1),
            "(NT G) 1 -> NT G", NT=NT,
        )
    
        idx = (
            torch.arange(NT, device=self.device).unsqueeze(1),
            torch.arange(self._G, device=self.device).unsqueeze(0),
            sampled,
        )
        sampled_offsets = offsets[idx].sum(dim=1)
        centers = self.vqvae.draw_code_forward(sampled).view(NT, -1, self._D)
        dec_in = einops.rearrange(centers.clone().detach(), "NT G D -> NT (G D)")
        decoded = self.vqvae.get_action_from_latent(dec_in).clone().detach()
        sampled_offsets = einops.rearrange(sampled_offsets,
                                            "NT (W A) -> NT W A",
                                            W=self.act_window_size)
        predicted = decoded + sampled_offsets
    
        if action_seq is not None:
            n, total_w, ad = action_seq.shape
            aw = self.act_window_size
            ow = total_w + 1 - aw
            windows = torch.empty((n, ow, aw, ad), device=action_seq.device)
            for i in range(ow):
                windows[:, i] = action_seq[:, i: i + aw]
            flat_acts = einops.rearrange(windows, "N T W A -> (N T) W A")
            _, bins = self.vqvae.get_code(flat_acts)
            if flat_acts.ndim == 2:
                flat_acts = flat_acts.unsqueeze(0)
    
            # ── FIX B: cast both sides to float32 before loss computation. ───────
            # Under AMP, predicted / decoded are bfloat16 while flat_acts (from the
            # DataLoader) is float32.  F.l1_loss / F.mse_loss require matching
            # dtypes; PyTorch 2.x auto-promotes but explicit casting is safer and
            # ensures the loss is always in full precision.
            predicted_fp32 = predicted.float()
            flat_acts_fp32 = flat_acts.float()
    
            offset_loss = F.l1_loss(flat_acts_fp32, predicted_fp32)
            action_diff = F.mse_loss(
                einops.rearrange(flat_acts_fp32,
                                "(N T) W A -> N T W A", T=ow)[:, -1, 0, :],
                einops.rearrange(predicted_fp32,
                                "(N T) W A -> N T W A", T=ow)[:, -1, 0, :],
            )
            cbet_l1 = self._criterion(logits[:, 0, :], bins[:, 0])
            cbet_l2 = (self._criterion(logits[:, 1, :], bins[:, 1])
                    if self._G > 1 else 0)
            cbet_loss = cbet_l1 * 5 + cbet_l2 * self._secondary_code_multiplier
            rate = (
                torch.sum(
                    (torch.sum((bins == sampled).int(), dim=1) == self._G).int()
                ) / NT
            )
            loss = cbet_loss + self._offset_loss_multiplier * offset_loss
            loss_dict = {
                "classification_loss": cbet_loss.detach().cpu().item()
                    if isinstance(cbet_loss, torch.Tensor) else cbet_loss,
                "offset_loss": offset_loss.detach().cpu().item(),
                "total_loss": loss.detach().cpu().item(),
                "equal_total_code_rate": rate.item(),
                "action_diff": action_diff.detach().cpu().item(),
            }
            return predicted, loss, loss_dict
    
        return predicted, None, {}

    # ── Training loss (CRITICAL FIX: extract + forward human_vl_ids) ─────────

    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Compute training loss.

        batch["human_repo_id"] is extracted and forwarded through _encode_task
        so FrozenEncoderWrapper can serve results from the L3 cache without
        ever invoking the expensive Qwen VLM encoder.

        Non-blocking=True requires pin_memory=True in the DataLoader (set in
        train() below).
        """
        robot_obs = {k: v.to(self.device, non_blocking=True)
                     for k, v in batch["robot_obs"].items()}
        action_seq = batch["robot_actions"].to(self.device, non_blocking=True)
        first_frame = {k: v.to(self.device, non_blocking=True)
                       for k, v in batch["robot_first_frame_obs"].items()}
        robot_first_frame = self._preprocess_first_frame(first_frame)

        human_video: Optional[torch.Tensor] = None
        human_desc: Optional[List[str]] = None
        # human_vl_ids is the stable per-task string list (= batch['human_repo_id']).
        # It is present even when skip_human_video=True (video=None, id is kept).
        human_vl_ids: Optional[List[str]] = None

        if self.input_mode == InputMode.VIDEO_ONLY:
            hv = batch.get("human_video")
            if hv is not None:
                human_video = hv.to(self.device, non_blocking=True)
            human_vl_ids = (
                batch.get("human_repo_id")
                or batch.get("human_video_path")
                or batch.get("human_episode_id")
            )

        elif self.input_mode == InputMode.LANGUAGE_ONLY:
            human_desc = batch.get("language")

        elif self.input_mode == InputMode.VIDEO_AND_LANGUAGE:
            if self.random_modality_selection and self.training:
                if random.random() < self.use_video_prob:
                    hv = batch.get("human_video")
                    if hv is not None:
                        human_video = hv.to(self.device, non_blocking=True)
                    human_vl_ids = batch.get("human_repo_id")
                else:
                    human_desc = batch.get("language")
            else:
                hv = batch.get("human_video")
                if hv is not None:
                    human_video = hv.to(self.device, non_blocking=True)
                human_desc = batch.get("language")
                human_vl_ids = batch.get("human_repo_id")

        task_feat, _ = self._encode_task(
            robot_first_frame, human_video, human_desc, human_vl_ids
        )

        obs_images, states = self._preprocess_obs(robot_obs)
        obs_cond = self._get_obs_condition(obs_images, states)
        obs_seq = obs_cond.unsqueeze(1).expand(-1, self.obs_window_size, -1) \
            if obs_cond.dim() == 2 else obs_cond

        _, loss, loss_dict = self.forward(obs_seq, task_feat, action_seq)
        return loss, loss_dict

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_action(self, obs_dict: Dict) -> torch.Tensor:
        if self._cached_task_features is None:
            raise RuntimeError("Call prepare_for_eval() first!")
        obs_images, states = self._preprocess_obs({
            "states": obs_dict.get("state", obs_dict.get("states")),
            "view_1": obs_dict.get("rgb", obs_dict.get("view_1")),
        })
        B = states.shape[0]
        obs_cond = self._get_obs_condition(obs_images, states)
        obs_seq = obs_cond.unsqueeze(1).expand(-1, self.obs_window_size, -1) \
            if obs_cond.dim() == 2 else obs_cond
        task_feat = self._cached_task_features
        if task_feat.shape[0] != B:
            task_feat = task_feat.expand(B, -1)
        predicted, _, _ = self.forward(obs_seq, task_feat)
        T_obs = obs_seq.shape[1]
        predicted = einops.rearrange(predicted, "(N T) W A -> N T W A", N=B, T=T_obs)
        return predicted[:, -1]

    def prepare_for_eval(
        self,
        human_video: Optional[torch.Tensor],
        robot_obs: Dict,
        human_desc: Optional[List[str]] = None,
        human_vl_ids: Optional[List[str]] = None,
    ):
        self.eval()
        with torch.no_grad():
            if human_video is not None:
                human_video = human_video.to(self.device)
            robot_obs = {k: v.to(self.device) for k, v in robot_obs.items()}
            robot_ff = self._preprocess_first_frame(robot_obs)
            task_feat, _ = self._encode_task(robot_ff, human_video, human_desc, human_vl_ids)
        self._cached_task_features = task_feat

    def clear_cache(self):
        self._cached_task_features = None

    # ── VQ-VAE pretraining (non-blocking, optimised) ──────────────────────────

    def pretrain_vqvae(self, dataloader: DataLoader, epochs: int = 50):
        """Pretrain the VQ-VAE component.

        vqvae_update() holds its own internal Adam optimizer, so we cannot
        apply AMP to the update call directly.  Non-blocking GPU transfers
        (enabled by pin_memory=True in the DataLoader) and the forkserver
        workers eliminate the dominant bottleneck that made each epoch slow.
        """
        print(f"\n📦 Pretraining VQ-VAE for {epochs} epochs...")
        for epoch in tqdm(range(epochs), desc="VQ-VAE Pretraining"):
            enc_sum = vq_sum = recon_sum = 0.0
            n = 0
            for batch in tqdm(dataloader, desc=f"VQ-VAE E{epoch}", leave=False):
                # Non-blocking: action tensor is pinned, GPU copy overlaps compute.
                acts = batch["robot_actions"].to(self.device, non_blocking=True)
                chunk = acts[:, : self.act_window_size]
                enc_loss, vq_loss, _codes, recon = self.vqvae.vqvae_update(chunk)
                enc_sum += enc_loss.item() if isinstance(enc_loss, torch.Tensor) else enc_loss
                vq_sum += vq_loss.item() if isinstance(vq_loss, torch.Tensor) else vq_loss
                recon_sum += recon
                n += 1
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                    f"  VQ-VAE E{epoch}: enc={enc_sum/max(n,1):.4f}  "
                    f"vq={vq_sum/max(n,1):.4f}  recon={recon_sum/max(n,1):.4f}"
                )
        self.vqvae.vq_layer.eval()
        print("✅ VQ-VAE pretraining complete")

    # ── Optimizer configuration ───────────────────────────────────────────────

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> Dict[str, optim.Optimizer]:
        opt1 = self.gpt.configure_optimizers(
            weight_decay=weight_decay, learning_rate=learning_rate, betas=betas
        )
        for pg in [self.code_predictor.parameters(),
                   self.obs_encoder.parameters(),
                   self.state_encoder.parameters(),
                   self.task_projector.parameters()]:
            opt1.add_param_group({"params": pg})
        if self._use_frozen_backbone or not self.args.freeze_task_encoder:
            opt1.add_param_group({
                "params": self.task_encoder.trainable_params(),
                "lr": learning_rate * 0.1,
                "weight_decay": weight_decay,
            })
        opt2 = torch.optim.AdamW(
            self.offset_predictor.parameters(),
            lr=learning_rate, weight_decay=weight_decay, betas=betas,
        )
        return {"optimizer1": opt1, "optimizer2": opt2}


# =============================================================================
# Evaluation helper
# =============================================================================

def evaluate_vqbet(n, agent, eval_envs, eval_kwargs, evaluate_processor, progress_bar=True):
    from mani_skill.utils import common
    env_id = eval_kwargs["env_id"]
    delta_control = eval_kwargs.get("delta_control", False)
    pred_horizon = eval_kwargs["pred_horizon"]
    temporal_agg = eval_kwargs.get("temporal_agg", True)
    max_timesteps = eval_kwargs["max_timesteps"]
    device = eval_kwargs["device"]
    sim_backend = eval_kwargs.get("sim_backend")

    if isinstance(eval_envs.single_observation_space, gymnasium.spaces.Box):
        action_dim = eval_envs.action_space.shape[-1]
    else:
        action_dim = (
            eval_envs.action_space["panda_wristcam-0"].shape[-1]
            + eval_envs.action_space["panda_wristcam-1"].shape[-1]
        )

    num_envs = eval_envs.num_envs
    if temporal_agg:
        query_freq = 1
        all_time_actions = torch.zeros(
            [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim], device=device
        )
    else:
        query_freq = pred_horizon
        actions_to_take = torch.zeros([num_envs, pred_horizon, action_dim], device=device)

    agent.eval()
    val_videos = evaluate_processor.get_video(env_id).to(device)
    input_mode = agent.input_mode
    val_desc = None
    if input_mode in [InputMode.LANGUAGE_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
        val_desc = [evaluate_processor.get_task_description(env_id)] * num_envs

    with torch.no_grad():
        metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        ts = eps = 0
        pbar = tqdm(total=n, desc="Eval", disable=not progress_bar, unit="ep")
        while eps < n:
            obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
            if not delta_control:
                obs["state"], obs["rgb"] = evaluate_processor.normalize_state_rgb(
                    obs["state"], obs["rgb"], env_id
                )
            if ts == 0:
                agent.prepare_for_eval(
                    val_videos if input_mode != InputMode.LANGUAGE_ONLY else None,
                    robot_obs={"states": obs.get("state"), "view_1": obs.get("rgb")},
                    human_desc=val_desc,
                )
            if ts % query_freq == 0:
                action_seq = agent.get_action(obs)
            if temporal_agg:
                all_time_actions[:, ts, ts: ts + pred_horizon] = action_seq
                acs = all_time_actions[:, :, ts]
                populated = torch.zeros(max_timesteps, dtype=torch.bool, device=device)
                populated[max(0, ts + 1 - pred_horizon): ts + 1] = True
                acs = acs[:, populated]
                w = torch.exp(-0.01 * torch.arange(acs.shape[1], device=device))
                w = (w / w.sum()).unsqueeze(0).unsqueeze(-1)
                raw = (acs * w).sum(dim=1)
            else:
                if ts % query_freq == 0:
                    actions_to_take = action_seq
                raw = actions_to_take[:, ts % query_freq]

            _a = evaluate_processor.denormalize_action(raw, env_id) if not delta_control else raw
            if sim_backend == "physx_cpu":
                _a = _a.cpu().numpy()
            action = {"panda_wristcam-0": _a[:, :8], "panda_wristcam-1": _a[:, 8:16]}
            obs, _, _, truncated, info = eval_envs.step(action)
            ts += 1
            if truncated.any():
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        metrics[k].append(v.float().cpu().numpy())
                else:
                    for fi in info["final_info"]:
                        for k, v in fi["episode"].items():
                            metrics[k].append(v)
                pbar.update(num_envs)
                eps += num_envs
                ts = 0
                if temporal_agg:
                    all_time_actions = torch.zeros_like(all_time_actions)
                agent.clear_cache()
                obs, info = eval_envs.reset()
        pbar.close()
    agent.train()
    for k in metrics:
        metrics[k] = np.stack(metrics[k])
    return metrics


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(path, iteration, agent, optimizers, best_metrics,
                    args=None, epoch=None, lr_scheduler=None, scaler=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration":               iteration,
        "agent_state_dict":        agent.state_dict(),
        "vqvae_state_dict":        agent.vqvae.state_dict(),
        "optimizer1_state_dict":   optimizers["optimizer1"].state_dict(),
        "optimizer2_state_dict":   optimizers["optimizer2"].state_dict(),
        "best_eval_metrics":       dict(best_metrics),
    }
    if epoch is not None:
        payload["epoch"] = epoch
    if args is not None:
        payload["args"] = vars(args)
    if lr_scheduler is not None:
        payload["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)
    print(f"  💾 Saved checkpoint → {path}")


def load_checkpoint(path, agent, optimizers, device,
                    lr_scheduler=None, scaler=None):
    """Load full training state. Returns (iteration, best_eval_metrics, epoch).

    Backwards-compatible: handles old checkpoints that used key ``"agent"``
    instead of ``"agent_state_dict"``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # agent weights — old key was "agent", new key is "agent_state_dict"
    agent_sd = ckpt.get("agent_state_dict") or ckpt.get("agent")
    if agent_sd is not None:
        agent.load_state_dict(agent_sd, strict=False)
    else:
        print(f"  ⚠️  load_checkpoint: no agent weights found in {path}")
    if "vqvae_state_dict" in ckpt:
        agent.vqvae.load_state_dict(ckpt["vqvae_state_dict"], strict=False)
    if "optimizer1_state_dict" in ckpt:
        optimizers["optimizer1"].load_state_dict(ckpt["optimizer1_state_dict"])
    if "optimizer2_state_dict" in ckpt:
        optimizers["optimizer2"].load_state_dict(ckpt["optimizer2_state_dict"])
    if lr_scheduler is not None and "lr_scheduler_state_dict" in ckpt:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    iteration    = ckpt.get("iteration", 0)
    epoch        = ckpt.get("epoch", 0)
    best_metrics = defaultdict(float, ckpt.get("best_eval_metrics", {}))
    print(f"  📥 Resumed from {path}  (epoch={epoch}, iter={iteration})")
    return iteration, best_metrics, epoch


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _run_evaluation(args, agent, evaluate_processor, writer, step, device) -> Dict:  # returns metrics
    from examples.baselines.vqbet.vqbet.make_env import make_eval_envs
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
    obs_mode = "rgbd" if args.include_depth else "rgb"
    env_kwargs = dict(
        control_mode=args.control_mode, reward_mode="dense", obs_mode=obs_mode,
        render_mode="rgb_array", max_episode_steps=args.max_episode_steps,
        sensor_configs=dict(shader_pack="rt-fast"), 
        human_render_camera_configs=dict(shader_pack="rt-fast"),
    )
    try:
        eval_envs = make_eval_envs(
            args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs,
            dict(obs_horizon=args.obs_horizon),
            video_dir=f"runs/videos/step_{step}", wrappers=[FlattenRGBDObservationWrapper],
        )
        eval_metrics = evaluate_vqbet(
            n=args.num_eval_episodes, agent=agent, eval_envs=eval_envs,
            eval_kwargs=dict(
                env_id=args.env_id, delta_control=("delta" in args.control_mode),
                pred_horizon=args.act_window_size, temporal_agg=args.eval_temporal_agg,
                max_timesteps=args.max_episode_steps, device=device,
                sim_backend=args.sim_backend,
            ),
            evaluate_processor=evaluate_processor,
        )
        _out: Dict = {}
        for k, v in eval_metrics.items():
            _out[k] = float(v.mean())
            writer.add_scalar(f"eval/{k}_mean", _out[k], step)
        print(f"   success_once: {_out.get('success_once', 0):.4f}  "
              f"success_at_end: {_out.get('success_at_end', 0):.4f}")
        eval_envs.close()
        return _out
    except Exception as e:
        import traceback
        print(f"   ⚠️  Evaluation failed: {e}")
        traceback.print_exc()
    return {}


# =============================================================================
# Main training loop
# =============================================================================

def train():
    args = tyro.cli(TrainingArgs)
    seed_everything(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # TF32: free ~10 % on Ampere+ with negligible precision loss.
    if args.use_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from datetime import datetime
    run_name = f"{datetime.now().strftime('%Y%m%d')}"
    if args.exp_name:
        run_name = f"{args.exp_name}_{run_name}"

    save_dir = Path(f"runs/{run_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "checkpoints").mkdir(exist_ok=True)
    writer = SummaryWriter(str(save_dir))

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   name=run_name, config=vars(args))

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\n📚 Setting up dataset...")
    total_action_horizon = args.obs_window_size + args.act_window_size - 1
    _vqbet_ds_config = PairedDatasetConfig(
        human_root=args.human_root, sim_root=args.sim_root,
        robot_root=args.robot_root,
        task_mapping_file=args.task_mapping_file,
        human_dataset_file=args.human_dataset_file,
        sim_dataset_file=args.sim_dataset_file,
        robot_dataset_file=args.robot_dataset_file,
        human_task_description_file=args.human_task_description_file,
        sim_task_description_file=args.sim_task_description_file,
        robot_task_description_file=args.robot_task_description_file,
        split="train", cameras=args.cameras, include_depth=args.include_depth,
        image_size=args.image_size, num_frames=args.task_num_frames,
        video_backend=args.video_backend, horizon=total_action_horizon,
        obs_horizon=args.obs_horizon, state_type=args.state_type,
        single_arm=args.single_arm, fps=args.fps, input_mode=args.input_mode,
        include_first_frame=args.include_first_frame,
    )
    if args.data_source == "robot":
        from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
            HumanRobotPairedDataset,
        )
        print(f"\n[DataSource] Real-robot mode: robot_root={args.robot_root}")
        dataset = HumanRobotPairedDataset(_vqbet_ds_config)
    else:
        dataset = HumanSimPairedDataset(_vqbet_ds_config)

    # pin_memory=True: enables non-blocking CPU->GPU transfers.
    # forkserver: avoids torchcodec fork-safety issues with multiple workers.
    # prefetch_factor only when num_workers>0: removes a spurious warning.
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_dataload_workers,
        collate_fn=get_collate_fn(args.input_mode),
        persistent_workers=(args.num_dataload_workers > 0),
        prefetch_factor=(4 if args.num_dataload_workers > 0 else None),
        pin_memory=True,
        drop_last=True,
        multiprocessing_context="forkserver" if args.num_dataload_workers > 0 else None,
    )
    num_batches_per_epoch = len(dataloader)
    print(f"📊 Batches/epoch: {num_batches_per_epoch}")

    # ── Evaluate processor ────────────────────────────────────────────────────
    if not args.no_eval:
        from examples.baselines.lerobot_dataset.evaluate_processor import (
            HumanVideoSimEvaluateProcessor, HumanVideoSimEvaluateProcessorConfig,
        )
        evaluate_processor = HumanVideoSimEvaluateProcessor(
            HumanVideoSimEvaluateProcessorConfig(
                human_root=args.human_root, human_split="train",
                human_dataset_file=args.human_dataset_file,
                human_task_description_file=args.human_task_description_file,
                human_cameras=args.cameras, human_include_depth=args.include_depth,
                human_num_frames=args.task_num_frames, human_image_size=args.image_size,
                human_video_backend=args.video_backend, human_fps=args.fps,
                sim_root=args.sim_root, sim_split="train",
                sim_dataset_file=args.sim_dataset_file,
                sim_task_description_file=args.sim_task_description_file,
                sim_state_type=args.state_type, sim_single_arm=args.single_arm,
                normalization_method="bounds_q99",
                task_mapping_file=args.task_mapping_file,
            )
        )
    else:
        evaluate_processor = None
        print("⏭️  Evaluate processor skipped (--no-eval)")

    # ── Agent ─────────────────────────────────────────────────────────────────
    print("\n🤖 Creating VQ-BeT agent...")
    agent = VQBeTAgent(args, device)

    if getattr(agent, "_te_needs_cache_setup", False):
        # Frozen backbone — build raw-feature cache
        from examples.baselines.encoders.task_encoder.frozen_backbone_cache import (
            enable_skip_human_video,
            setup_frozen_backbone_cache,
        )
        agent.task_encoder = setup_frozen_backbone_cache(
            agent.task_encoder,
            dataloader=dataloader,
            device=device,
            backbone_type=args.frozen_backbone_type,
            te_cache_root=args.te_cache_root,
            preload_to_memory=args.te_cache_preload_memory,
            recompute=getattr(args, 'te_cache_recompute', False),
            verbose=True,
        )
        agent._te_needs_cache_setup = False
        new_loader = enable_skip_human_video(
            agent.task_encoder, dataloader,
            recreate_dataloader=(args.num_dataload_workers > 0),
        )
        if new_loader is not None:
            dataloader = new_loader
            num_batches_per_epoch = len(dataloader)

    # ── Optimizers ────────────────────────────────────────────────────────────
    optimizers = agent.configure_optimizers(
        weight_decay=args.weight_decay, learning_rate=args.lr, betas=(0.95, 0.999)
    )

    # ── VQ-VAE pretraining ────────────────────────────────────────────────────
    if args.vqvae_ckpt is None and args.vqvae_pretrain_epochs > 0:
        agent.pretrain_vqvae(dataloader, epochs=args.vqvae_pretrain_epochs)

    # ── AMP setup ─────────────────────────────────────────────────────────────
    use_amp = args.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # GradScaler is needed only for float16; bfloat16 has sufficient dynamic range.
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    print(f"   AMP: {'bfloat16' if use_amp and amp_dtype == torch.bfloat16 else 'float16' if use_amp else 'disabled'}")

    from diffusers.optimization import get_scheduler as _get_lr_scheduler
    _warmup_steps = args.warmup_epochs * num_batches_per_epoch
    _total_steps  = args.total_epochs  * num_batches_per_epoch
    lr_scheduler = _get_lr_scheduler(
        name="cosine",
        optimizer=optimizers["optimizer2"],
        num_warmup_steps=_warmup_steps,
        num_training_steps=_total_steps,
    )

    agent.train()
    iteration = 0
    start_epoch = 0
    best_metrics: Dict = defaultdict(float)

    if args.resume_from and os.path.exists(args.resume_from):
        iteration, best_metrics, start_epoch = load_checkpoint(
            args.resume_from, agent, optimizers, device,
            lr_scheduler=lr_scheduler, scaler=scaler,
        )
        # CLI overrides (useful for curriculum restarts)
        if args.resume_epoch is not None:
            start_epoch = args.resume_epoch
        if args.resume_iteration is not None:
            iteration = args.resume_iteration
        print(f"Resumed from epoch {start_epoch}, iter {iteration}")
    elif args.resume_from:
        print(f"  ⚠️  resume_from={args.resume_from!r} not found — starting fresh.")

    # ── Epoch training loop ───────────────────────────────────────────────────
    if args.use_epoch_training:
        print(f"\n🚀 Training ({args.total_epochs} epochs)...")
        half_epochs = args.total_epochs // 2
        global_step = start_epoch * num_batches_per_epoch

        for epoch in tqdm(range(start_epoch, args.total_epochs), desc="Epochs"):
            epoch_loss = 0.0
            epoch_loss_dict: Dict[str, float] = defaultdict(float)
            n_batches = 0
            in_phase1 = epoch < half_epochs

            for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
                # zero_grad(set_to_none=True) avoids zeroing then re-allocating
                # gradient buffers; reduces memory-clear overhead.
                if in_phase1:
                    optimizers["optimizer1"].zero_grad(set_to_none=True)
                    optimizers["optimizer2"].zero_grad(set_to_none=True)
                else:
                    optimizers["optimizer2"].zero_grad(set_to_none=True)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    loss, loss_dict = agent.compute_loss(batch)

                scaler.scale(loss).backward()

                if scaler.is_enabled():
                    if in_phase1:
                        scaler.unscale_(optimizers["optimizer1"])
                    scaler.unscale_(optimizers["optimizer2"])

                torch.nn.utils.clip_grad_norm_(
                    [p for p in agent.parameters() if p.requires_grad], args.max_grad_norm
                )

                if in_phase1:
                    scaler.step(optimizers["optimizer1"])
                scaler.step(optimizers["optimizer2"])
                scaler.update()
                lr_scheduler.step()  # cosine LR

                epoch_loss += loss.item()
                for k, v in loss_dict.items():
                    epoch_loss_dict[k] += v
                n_batches += 1
                global_step += 1
                iteration += 1

                if global_step % args.log_freq == 0:
                    cs = agent.task_encoder.cache_stats(batch.get("human_repo_id"))
                    print(f"  [step {global_step}] loss={loss.item():.4f} | {cs}")
                    if args.track:
                        import wandb
                        wandb.log({"train/loss": loss.item(), "cache": cs,
                                   **{f"train/{k}": v for k, v in loss_dict.items()}},
                                  step=global_step)

            avg = epoch_loss / max(n_batches, 1)
            if epoch % args.log_epoch_freq == 0:
                writer.add_scalar("train_epoch/loss", avg, epoch)
                ls = "  ".join(f"{k}={v/max(n_batches,1):.4f}" for k, v in epoch_loss_dict.items())
                print(f"\nEpoch {epoch}: Loss={avg:.4f}  [{ls}]")
                if args.track:
                    import wandb
                    wandb.log({"train_epoch/loss": avg, "epoch": epoch,
                               **{f"train_epoch/{k}": v/max(n_batches,1)
                                  for k, v in epoch_loss_dict.items()}}, step=global_step)

            if epoch % args.eval_epoch_freq == 0 and epoch > 0 and not args.no_eval:
                eval_m = _run_evaluation(args, agent, evaluate_processor, writer, epoch, device)
                agent.train()
                for _k in ("success_once", "success_at_end"):
                    if _k in eval_m and eval_m[_k] > best_metrics.get(_k, 0.0):
                        best_metrics[_k] = eval_m[_k]
                        save_checkpoint(
                            f"{save_dir}/checkpoints/best_model_{_k}.pt",
                            iteration, agent, optimizers, best_metrics,
                            args, epoch, lr_scheduler=lr_scheduler, scaler=scaler,
                        )
                        print(f"  🏆 New best {_k}: {eval_m[_k]:.4f}")

            if epoch % args.save_epoch_freq == 0 and epoch > 0:
                save_checkpoint(
                    f"{save_dir}/checkpoints/epoch_{epoch}.pt",
                    iteration, agent, optimizers, best_metrics, args, epoch,
                    lr_scheduler=lr_scheduler, scaler=scaler,
                )

    save_checkpoint(
        f"{save_dir}/checkpoints/final_model.pt", iteration, agent, optimizers,
        best_metrics, args, args.total_epochs if args.use_epoch_training else None,
        lr_scheduler=lr_scheduler, scaler=scaler,
    )
    print("\n✅ Training completed!")
    if args.track:
        import wandb; wandb.finish()
    writer.close()


if __name__ == "__main__":
    train()