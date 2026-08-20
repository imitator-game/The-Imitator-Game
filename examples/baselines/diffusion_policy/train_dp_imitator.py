"""
Training Script for Diffusion Policy with Task Encoder (LeRobot)
================================================================
LoRA notes:
  - frozen_backbone_lora_rank / lora_alpha / lora_dropout control LoRA on the
    video backbone via TrainingArgs
  - _build_frozen_backbone_encoder() skips the feature cache when LoRA is enabled
    (cached features would be stale after each gradient step)
  - LoRA backbone params are included in the optimizer via agent.trainable_params()

Architecture:
  task_encoder  →  z  (task latent)
  obs_encoder   →  obs_feat
  state_encoder →  state_feat
  [obs_feat; state_feat; z]  →  global_cond  →  ConditionalUnet1D
"""

ALGO_NAME = "DiffusionPolicy_TaskEncoder"

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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import tyro

from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanSimPairedDataset,
    PairedDatasetConfig,
    InputMode,
    get_collate_fn,
)
from examples.baselines.encoders.obs_encoder import (
    ObservationEncoder,
    ObsEncoderConfig,
)
from examples.baselines.encoders.state_encoder import (
    StateEncoder,
    StateEncoderConfig,
)
from examples.baselines.diffusion_policy.diffusion_policy.conditional_unet1d import ConditionalUnet1D


# ============================================================================
# Training arguments
# ============================================================================

@dataclass
class TrainingArgs:
    """Full training configuration for Diffusion Policy + Task Encoder."""

    # ── Experiment ────────────────────────────────────────────────────────────
    exp_name:           Optional[str] = None
    seed:               int  = 1
    cuda:               bool = True
    track:              bool = False
    wandb_project_name: str  = "DPTaskEncoder"
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

    # ── Robot data ─────────────────────────────────────────────────────────────
    data_source:                 str           = "sim"
    robot_root:                  str           = "demos"
    robot_dataset_file:          Optional[str] = None
    robot_task_description_file: Optional[str] = "examples/baselines/lerobot_dataset/task_desc/robot_desc.json"

    # ── Training mode ─────────────────────────────────────────────────────────
    use_epoch_training: bool  = True
    total_epochs:       int   = 100
    total_iters:        int   = 1_000_000
    batch_size:         int   = 256
    lr:                 float = 1e-4
    num_dataload_workers: int = 0
    warmup_epochs:      int   = 5

    # ── Model architecture ────────────────────────────────────────────────────
    action_dim:               int        = 16
    state_dim:                int        = 18
    obs_horizon:              int        = 1
    act_horizon:              int        = 4
    pred_horizon:             int        = 16
    diffusion_step_embed_dim: int        = 32
    unet_dims:                List[int]  = field(default_factory=lambda: [256, 512, 1024])
    n_groups:                 int        = 8
    num_diffusion_iters:      int        = 100

    # ── Observation settings ──────────────────────────────────────────────────
    include_depth: bool       = False
    cameras:       List[str]  = field(default_factory=lambda: ["zed2i"])
    image_size:    Tuple[int, int] = (224, 224)

    # ── Task encoder backend ──────────────────────────────────────────────────
    task_encoder_type: str = "frozen_backbone"   # "frozen_backbone"

    # ── FrozenVideoBackbone settings ──────────────────────────────────────────
    frozen_backbone_type:           str          = "dinov2_vitl14"
    frozen_backbone_model:          Optional[str] = None
    frozen_backbone_adapter_layers: int          = 1
    frozen_backbone_seq_patches:    int          = 32
    frozen_backbone_num_frames:     int          = 4

    # ── LoRA fine-tuning of the video backbone ────────────────────────────
    # lora_rank == 0  → fully frozen backbone (cache enabled)
    # lora_rank > 0   → LoRA adapters injected into attention layers; cache disabled
    frozen_backbone_lora_rank:    int   = 0      # 0 = no LoRA; typical: 8 or 16
    frozen_backbone_lora_alpha:   float = 16.0   # LoRA scaling: alpha / rank
    frozen_backbone_lora_dropout: float = 0.0    # LoRA input dropout

    task_latent_dim:          int   = 256
    task_num_frames:          int   = 10
    hf_cache_dir:             Optional[str] = None

    # ── VL cache ─────────────────────────────────────────────────────────────
    te_cache_root:               Optional[str] = "./te_cache"
    te_cache_preload_to_memory:  bool = True
    te_cache_recompute:          bool = False

    # ── Input mode ────────────────────────────────────────────────────────────
    input_mode:               str   = "video_only"
    random_modality_selection: bool = True
    use_video_prob:            float = 0.5
    include_first_frame:       bool  = True

    # ── Observation encoder ───────────────────────────────────────────────────
    obs_encoder_type:    str  = "simple_cnn"
    obs_latent_dim:      int  = 256
    obs_freeze_backbone: bool = False
    obs_finetune_layers: int  = 0

    # ── State encoder ─────────────────────────────────────────────────────────
    state_encoder_type: str = "mlp"
    state_latent_dim:   int = 256
    state_hidden_dim:   int = 256
    state_num_layers:   int = 4
    state_num_heads:    int = 4

    # ── Logging / eval ────────────────────────────────────────────────────────
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
    state_type:        str  = "qpos"
    single_arm:        bool = False
    fps:               int  = 30
    video_backend:     str  = "torchcodec"

    # ── Checkpointing ─────────────────────────────────────────────────────────
    resume_from:        Optional[str] = None
    reset_lr_scheduler: bool = False


# ============================================================================
# Frozen backbone encoder helper
# ============================================================================

def _build_frozen_backbone_encoder(
    args: TrainingArgs,
    device: torch.device,
    dataloader=None,
) -> nn.Module:
    """Build FrozenVideoBackbone, with or without LoRA.

    LoRA mode (frozen_backbone_lora_rank > 0)
    ──────────────────────────────────────────
    The backbone attention layers receive LoRA adapters. Features change every
    gradient step, so the raw-feature cache is completely bypassed.

      ✓  LoRA params + adapter params are trainable
      ✗  setup_frozen_backbone_cache is NOT called
      ✗  enable_skip_human_video is NOT called

    Frozen mode (frozen_backbone_lora_rank == 0, default)
    ──────────────────────────────────────────────────────
    Behaviour: backbone frozen, cache enabled, adapter trainable.
    """
    from examples.baselines.encoders.task_encoder.video_backbone import (
        build_video_backbone,
    )

    lora_rank    = getattr(args, "frozen_backbone_lora_rank",    0)
    lora_alpha   = getattr(args, "frozen_backbone_lora_alpha",   16.0)
    lora_dropout = getattr(args, "frozen_backbone_lora_dropout", 0.0)
    latent_dim   = args.task_latent_dim

    mode_str = (f"LoRA (rank={lora_rank}, alpha={lora_alpha})"
                if lora_rank > 0 else "Frozen")
    print(f"\n   Building FrozenVideoBackbone ({args.frozen_backbone_type})"
          f" [{mode_str}] ...")

    backbone = build_video_backbone(
        backbone_type=args.frozen_backbone_type,
        latent_dim=latent_dim,
        max_seq_patches=args.frozen_backbone_seq_patches,
        adapter_layers=args.frozen_backbone_adapter_layers,
        num_sampled_frames=args.frozen_backbone_num_frames,
        hf_cache_dir=getattr(args, "hf_cache_dir", None),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).to(device)

    if lora_rank > 0:
        # ── LoRA mode: no cache (features change each step) ───────────────────
        print(
            "   LoRA mode: raw-feature cache DISABLED.\n"
            "   Backbone LoRA params + adapter params are trainable.\n"
            "   Note: training will be slower than cached frozen mode\n"
            "   (~20-50 ms/iter backbone forward vs ~0.15 ms/iter from cache)."
        )
    elif dataloader is not None:
        # ── Frozen mode: enable cache ─────────────────────────────────────────
        from examples.baselines.encoders.task_encoder.frozen_backbone_cache import (
            setup_frozen_backbone_cache,
        )
        backbone = setup_frozen_backbone_cache(
            backbone,
            dataloader=dataloader,
            device=device,
            backbone_type=args.frozen_backbone_type,
            te_cache_root=args.te_cache_root,
            preload_to_memory=args.te_cache_preload_to_memory,
            recompute=args.te_cache_recompute,
            verbose=True,
        )
        print("   FrozenVideoBackbone: backbone frozen, cache active, adapter trainable.")
    else:
        print("   FrozenVideoBackbone ready (no cache — pass dataloader to enable).")

    return backbone


# ============================================================================
# DPAgent
# ============================================================================

class DPAgent(nn.Module):
    """Diffusion Policy agent with pretrained task encoder."""

    def __init__(
        self,
        args: TrainingArgs,
        device: torch.device,
        dataloader: Optional[DataLoader] = None,
    ):
        super().__init__()
        self.args         = args
        self.device       = device
        self.obs_horizon  = args.obs_horizon
        self.act_horizon  = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.action_dim   = args.action_dim
        self.state_dim    = args.state_dim
        self.include_depth = args.include_depth

        self.input_mode                = InputMode(args.input_mode)
        self.random_modality_selection = args.random_modality_selection
        self.use_video_prob            = args.use_video_prob

        self._backbone_lora = (
            getattr(args, "frozen_backbone_lora_rank", 0) > 0
        )

        # ── Task encoder (FrozenVideoBackbone only) ───────────────────────────
        self.task_encoder = _build_frozen_backbone_encoder(args, device, dataloader)
        # enable_skip_human_video only when frozen (no LoRA)
        if dataloader is not None and not self._backbone_lora:
            from examples.baselines.encoders.task_encoder.frozen_backbone_cache import (
                enable_skip_human_video,
            )
            enable_skip_human_video(self.task_encoder, dataloader,
                                    recreate_dataloader=False, verbose=True)

        task_latent_dim = args.task_latent_dim

        self.ftask_norm = nn.LayerNorm(task_latent_dim, elementwise_affine=True)

        # ── Observation encoder ───────────────────────────────────────────────
        obs_config = ObsEncoderConfig(
            encoder_type=args.obs_encoder_type,
            image_size=args.image_size[0],
            output_dim=args.obs_latent_dim,
            hidden_dim=args.obs_latent_dim,
            freeze_backbone=args.obs_freeze_backbone,
            finetune_layers=args.obs_finetune_layers,
        )
        self.obs_encoder = ObservationEncoder(obs_config)

        # ── State encoder ─────────────────────────────────────────────────────
        state_config = StateEncoderConfig(
            state_type=args.state_type,
            state_dim=args.state_dim,
            num_frames=args.obs_horizon,
            encoder_type=args.state_encoder_type,
            hidden_dim=args.state_hidden_dim,
            output_dim=args.state_latent_dim,
            num_layers=args.state_num_layers,
            num_heads=args.state_num_heads,
        )
        self.state_encoder = StateEncoder(state_config)

        # ── Diffusion UNet ────────────────────────────────────────────────────
        obs_cond_dim = args.obs_latent_dim + args.state_latent_dim + task_latent_dim
        self.unet = ConditionalUnet1D(
            input_dim=args.action_dim,
            global_cond_dim=obs_cond_dim,
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )

        self.num_diffusion_iters = args.num_diffusion_iters
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        self._cached_task_z: Optional[torch.Tensor] = None

        self.to(device)

        # Move DDPMScheduler tensors to device (non-Module)
        _scheduler_tensor_names = (
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "log_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
        )
        for attr in _scheduler_tensor_names:
            val = getattr(self.noise_scheduler, attr, None)
            if isinstance(val, torch.Tensor):
                setattr(self.noise_scheduler, attr, val.to(device))

        _DDIM_STEPS = 10
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler as _DDIM
        self._inference_scheduler = _DDIM.from_config(self.noise_scheduler.config)
        self._inference_scheduler.set_timesteps(_DDIM_STEPS)
        self._inference_num_steps = _DDIM_STEPS

        self._print_model_info()

    def _print_model_info(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora_rank = getattr(self.args, "frozen_backbone_lora_rank", 0)
        backend = (f"frozen_backbone_lora (rank={lora_rank}, {self.args.frozen_backbone_type})"
                   if lora_rank > 0
                   else f"frozen_backbone ({self.args.frozen_backbone_type})")
        print(f"\n   DPAgent ready")
        print(f"   Task encoder:  {backend}")
        print(f"   Input mode:    {self.input_mode.value}")
        print(f"   Total params:  {total/1e6:.2f}M  |  Trainable: {trainable/1e6:.2f}M")

    # ── Task encoding ──────────────────────────────────────────────────────────

    def _encode_task(
        self,
        robot_first_frame: torch.Tensor,
        human_video:   Optional[torch.Tensor] = None,
        human_tokens:  Optional[torch.Tensor] = None,
        human_desc:    Optional[List[str]]    = None,
        human_vl_ids:  Optional[List[str]]    = None,
    ) -> torch.Tensor:
        enc_out = self.task_encoder.encode(
            human_video=human_video,
            human_vl_ids=human_vl_ids,
        )
        z = enc_out["z"]
        return self.ftask_norm(z)

    # ── Observation processing ─────────────────────────────────────────────────

    def _preprocess_first_frame(self, first_frame_obs: Dict) -> torch.Tensor:
        images = first_frame_obs.get("view_1", first_frame_obs.get("rgb"))
        if images is None:
            raise ValueError("first_frame_obs must contain 'view_1' or 'rgb'.")
        if images.dim() == 5:
            images = images[:, 0]
        if images.dim() == 4 and images.shape[-1] in (3, 4):
            images = images.permute(0, 3, 1, 2)
        return images

    def _preprocess_obs(self, obs_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        states = obs_dict.get("state", obs_dict.get("states"))
        images = obs_dict.get("rgb", obs_dict.get("view_1"))
        if images is None:
            raise ValueError("obs_dict must contain 'rgb' or 'view_1'.")
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if states.dim() == 2:
            states = states.unsqueeze(1)
        return images, states

    def _get_obs_condition(
        self, obs_images: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        T = min(states.shape[1], self.obs_horizon)
        if obs_images.dim() == 5:
            obs_images = obs_images[:, :T].permute(0, 1, 4, 2, 3)
            if T == 1:
                obs_images = obs_images.squeeze(1)
        elif obs_images.dim() == 4:
            obs_images = obs_images.permute(0, 3, 1, 2)
        obs_feat,   _ = self.obs_encoder(obs_images)
        state_feat, _ = self.state_encoder(states[:, :T])
        return torch.cat([obs_feat, state_feat], dim=-1)

    # ── Training loss ──────────────────────────────────────────────────────────

    def compute_loss(self, batch: Dict) -> torch.Tensor:
        robot_obs  = {k: v.to(self.device, non_blocking=True)
                      for k, v in batch["robot_obs"].items()}
        action_seq = batch["robot_actions"].to(self.device, non_blocking=True)
        B          = action_seq.shape[0]

        first_frame_obs   = {k: v.to(self.device, non_blocking=True)
                             for k, v in batch["robot_first_frame_obs"].items()}
        robot_first_frame = self._preprocess_first_frame(first_frame_obs)

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
                use_video = random.random() < self.use_video_prob
                if use_video:
                    human_video = batch.get("human_video")
                    if human_video is not None:
                        human_video = human_video.to(self.device, non_blocking=True)
                    human_vl_ids = (batch.get("human_repo_id")
                                    or batch.get("human_video_path")
                                    or batch.get("human_episode_id"))
                else:
                    human_desc = batch.get("human_desc") or batch.get("language")
            else:
                human_video = batch.get("human_video")
                if human_video is not None:
                    human_video = human_video.to(self.device, non_blocking=True)
                human_desc   = batch.get("human_desc") or batch.get("language")
                human_vl_ids = (batch.get("human_repo_id")
                                or batch.get("human_video_path")
                                or batch.get("human_episode_id"))

        # ── Encode task ───────────────────────────────────────────────────────
        # LoRA backbone needs gradients → do NOT suppress with no_grad here.
        # For the frozen backbone the task encoder itself handles grad context.
        task_z = self._encode_task(
            robot_first_frame, human_video, None, human_desc, human_vl_ids
        )

        # ── Encode observation ────────────────────────────────────────────────
        obs_images, states = self._preprocess_obs(robot_obs)
        obs_cond = self._get_obs_condition(obs_images, states)
        global_cond = torch.cat([obs_cond, task_z], dim=-1)

        # ── Diffusion loss ────────────────────────────────────────────────────
        noise = torch.randn(
            B, self.pred_horizon, self.action_dim,
            device=self.device, dtype=action_seq.dtype,
        )
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device, dtype=torch.long,
        )
        noisy_actions = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred    = self.unet(noisy_actions, timesteps, global_cond=global_cond)

        return F.mse_loss(noise_pred, noise)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def prepare_for_eval(
        self,
        human_video:   Optional[torch.Tensor],
        robot_obs:     Dict,
        human_tokens:  Optional[torch.Tensor] = None,
        human_desc:    Optional[List[str]]    = None,
        human_vl_ids:  Optional[List[str]]   = None,
    ):
        self.eval()
        with torch.no_grad():
            if human_video is not None:
                human_video = human_video.to(self.device)
            robot_obs = {k: v.to(self.device) for k, v in robot_obs.items()}
            robot_first_frame = self._preprocess_first_frame(robot_obs)
            self._cached_task_z = self._encode_task(
                robot_first_frame, human_video, human_tokens, human_desc, human_vl_ids,
            )

    def clear_cache(self):
        self._cached_task_z = None

    @torch.no_grad()
    def get_action(self, obs_dict: Dict) -> torch.Tensor:
        assert self._cached_task_z is not None
        obs_images, states = self._preprocess_obs({
            "states": obs_dict.get("state",  obs_dict.get("states")),
            "rgb":    obs_dict.get("rgb",    obs_dict.get("view_1")),
        })
        B        = states.shape[0]
        obs_cond = self._get_obs_condition(obs_images, states)
        task_z      = self._cached_task_z.expand(B, -1)
        global_cond = torch.cat([obs_cond, task_z], dim=-1)

        actions = torch.randn((B, self.pred_horizon, self.action_dim), device=self.device)
        self._inference_scheduler.set_timesteps(self._inference_num_steps)
        for k in self._inference_scheduler.timesteps:
            noise_pred = self.unet(actions, k.expand(B).to(self.device), global_cond=global_cond)
            actions = self._inference_scheduler.step(noise_pred, k, actions).prev_sample

        return actions


def evaluate(n, agent, eval_envs, eval_kwargs, evaluate_processor, progress_bar=True):
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
            [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim], device=device)
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
                    obs["state"], obs["rgb"], env_id)

            if ts == 0:
                agent.prepare_for_eval(
                    human_video=val_videos,
                    robot_obs={"states": obs.get("state", obs.get("states")),
                               "view_1": obs.get("rgb",   obs.get("view_1"))})

            if ts % query_frequency == 0:
                action_seq = agent.get_action(obs)

            if temporal_agg:
                all_time_actions[:, ts, ts:ts + pred_horizon] = action_seq
                actions_for_curr_step = all_time_actions[:, :, ts]
                populated = torch.zeros(max_timesteps, dtype=torch.bool, device=device)
                populated[max(0, ts + 1 - pred_horizon):ts + 1] = True
                actions_for_curr_step = actions_for_curr_step[:, populated]
                k = 0.01
                exp_w = torch.exp(-k * torch.arange(actions_for_curr_step.shape[1], device=device))
                exp_w = (exp_w / exp_w.sum()).unsqueeze(0).unsqueeze(-1).expand(num_envs, -1, -1)
                raw_action = (actions_for_curr_step * exp_w).sum(dim=1)
            else:
                if ts % query_frequency == 0:
                    actions_to_take = action_seq
                raw_action = actions_to_take[:, ts % query_frequency]

            _action = (evaluate_processor.denormalize_action(raw_action, env_id)
                       if not delta_control else raw_action)
            if sim_backend == "physx_cpu":
                _action = _action.cpu().numpy()

            action = {"panda_wristcam-0": _action[:, :8], "panda_wristcam-1": _action[:, 8:16]}
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
                        device=device)
                agent.clear_cache()
                obs, info = eval_envs.reset()

        pbar.close()

    agent.train()
    for k in eval_metrics:
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics


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


def load_checkpoint(path, agent, optimizer, lr_scheduler, scaler, device,
                    new_total_steps=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    agent_sd = ckpt.get("agent_state_dict") or ckpt.get("agent")
    if agent_sd is not None:
        agent.load_state_dict(agent_sd, strict=False)
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device, non_blocking=True)
    if lr_scheduler is not None and "lr_scheduler_state_dict" in ckpt:
        if new_total_steps is not None:
            restored_iteration = ckpt.get("iteration", 0)
            lr_scheduler.last_epoch = restored_iteration - 1
            lr_scheduler.step()
        else:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
    if "scaler_state_dict" in ckpt and scaler is not None:
        saved_scaler = ckpt["scaler_state_dict"]
        saved_scaler["scale"] = min(float(saved_scaler.get("scale", 65536)), 1024.0)
        scaler.load_state_dict(saved_scaler)
    iteration         = ckpt.get("iteration", 0)
    epoch             = ckpt.get("epoch", 0)
    best_eval_metrics = defaultdict(float, ckpt.get("best_eval_metrics", {}))
    print(f"  📥 Resumed from {path}  (epoch={epoch}, iter={iteration})")
    return iteration, epoch, best_eval_metrics


def _run_evaluation(args, agent, eval_envs, evaluate_processor, device):
    return evaluate(
        n=args.num_eval_episodes, agent=agent, eval_envs=eval_envs,
        eval_kwargs=dict(
            env_id=args.env_id, delta_control=("delta" in args.control_mode),
            pred_horizon=args.pred_horizon, temporal_agg=args.eval_temporal_agg,
            max_timesteps=args.max_episode_steps, device=device,
            sim_backend=args.sim_backend,
        ),
        evaluate_processor=evaluate_processor, progress_bar=True,
    )


def train():
    args = tyro.cli(TrainingArgs)

    if args.exp_name is None:
        lora_rank = getattr(args, "frozen_backbone_lora_rank", 0)
        lora_suffix = f"_lora{lora_rank}" if lora_rank > 0 else ""
        run_name = (f"{args.env_id}__dp_{args.task_encoder_type}{lora_suffix}"
                    f"__{args.seed}__{int(time.time())}")
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

    _num_frames = args.task_num_frames
    dataset_config = PairedDatasetConfig(
        human_root=args.human_root, sim_root=args.sim_root,
        robot_root=args.robot_root, task_mapping_file=args.task_mapping_file,
        human_dataset_file=args.human_dataset_file,
        sim_dataset_file=args.sim_dataset_file,
        robot_dataset_file=args.robot_dataset_file,
        human_task_description_file=args.human_task_description_file,
        sim_task_description_file=args.sim_task_description_file,
        robot_task_description_file=args.robot_task_description_file,
        split="train", cameras=args.cameras, include_depth=args.include_depth,
        image_size=args.image_size, num_frames=_num_frames,
        horizon=args.pred_horizon, obs_horizon=args.obs_horizon,
        state_type=args.state_type, single_arm=args.single_arm,
        fps=args.fps, video_backend=args.video_backend,
        input_mode=args.input_mode, include_first_frame=args.include_first_frame,
    )
    if args.data_source == "robot":
        from examples.baselines.lerobot_dataset.lerobot_paired_dataset import HumanRobotPairedDataset
        dataset = HumanRobotPairedDataset(dataset_config)
    else:
        dataset = HumanSimPairedDataset(dataset_config)

    if not args.no_eval:
        from examples.baselines.lerobot_dataset.evaluate_processor import (
            HumanVideoSimEvaluateProcessor, HumanVideoSimEvaluateProcessorConfig)
        nf = args.task_num_frames
        evaluate_processor = HumanVideoSimEvaluateProcessor(
            HumanVideoSimEvaluateProcessorConfig(
                human_root=args.human_root, human_split="train",
                human_dataset_file=args.human_dataset_file,
                human_task_description_file=args.human_task_description_file,
                human_cameras=args.cameras, human_include_depth=args.include_depth,
                human_num_frames=nf, human_image_size=args.image_size,
                human_video_backend=args.video_backend, human_fps=args.fps,
                sim_root=args.sim_root, sim_split="train",
                sim_dataset_file=args.sim_dataset_file,
                sim_task_description_file=args.sim_task_description_file,
                sim_state_type=args.state_type, sim_single_arm=args.single_arm,
                normalization_method="bounds_q99",
            ))
    else:
        evaluate_processor = None

    collate_fn = get_collate_fn(args.input_mode)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_dataload_workers, collate_fn=collate_fn,
        persistent_workers=(args.num_dataload_workers > 0),
        prefetch_factor=4 if args.num_dataload_workers > 0 else None,
        pin_memory=True,
        multiprocessing_context="forkserver" if args.num_dataload_workers > 0 else None,
    )
    num_batches_per_epoch = len(dataloader)
    print(f"Batches per epoch: {num_batches_per_epoch}")

    if not args.no_eval:
        from examples.baselines.diffusion_policy.diffusion_policy.make_env import make_eval_envs
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
        eval_envs = make_eval_envs(
            args.env_id, args.num_eval_envs, args.sim_backend,
            dict(control_mode=args.control_mode, reward_mode="dense",
                 obs_mode="rgbd" if args.include_depth else "rgb",
                 render_mode="rgb_array", max_episode_steps=args.max_episode_steps,
                 sensor_configs=dict(shader_pack="rt-fast"),
                 human_render_camera_configs=dict(shader_pack="rt-fast")),
            dict(obs_horizon=args.obs_horizon),
            video_dir=f"runs/{run_name}/videos",
            wrappers=[FlattenRGBDObservationWrapper],
        )
    else:
        eval_envs = None

    agent = DPAgent(args, device, dataloader=dataloader)

    # Recreate DataLoader if skip_human_video was set (only for frozen, non-LoRA)
    if args.num_dataload_workers > 0 and dataset.skip_human_video:
        print("  Recreating DataLoader so persistent workers pick up skip_human_video=True ...")
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_dataload_workers, collate_fn=collate_fn,
            persistent_workers=True, prefetch_factor=4, pin_memory=True,
            multiprocessing_context="forkserver",
        )
        num_batches_per_epoch = len(dataloader)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # When LoRA is enabled, agent.parameters() that require_grad includes:
    #   - LoRA params in the backbone
    #   - adapter params (cls_adapter, seq_adapter, _pe)
    #   - obs_encoder, state_encoder, ftask_norm, unet
    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, betas=(0.95, 0.999),
                            weight_decay=1e-6)

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
            args.resume_from, agent, optimizer, lr_scheduler, scaler, device,
            new_total_steps=(None if not args.reset_lr_scheduler
                             else args.total_epochs * num_batches_per_epoch),
        )

    lora_rank = getattr(args, "frozen_backbone_lora_rank", 0)
    print(f"\nStarting training  (backend: {args.task_encoder_type}"
          f"{f', LoRA rank={lora_rank}' if lora_rank > 0 else ''}"
          f", skip_video: {dataset.skip_human_video})")
    agent.train()
    iteration = start_iteration

    if args.use_epoch_training:
        global_step = start_epoch * num_batches_per_epoch

        for epoch in tqdm(range(start_epoch, args.total_epochs), desc="Epochs"):
            epoch_loss  = 0.0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
                with torch.autocast("cuda", dtype=_amp_dtype, enabled=_use_amp):
                    loss = agent.compute_loss(batch)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Clip gradients — separate budget for LoRA backbone vs rest
                if lora_rank > 0 and hasattr(agent, "task_encoder"):
                    backbone_params = [
                        p for p in agent.task_encoder.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=0.5)

                torch.nn.utils.clip_grad_norm_(
                    [p for p in agent.unet.parameters() if p.requires_grad], max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(
                    ([p for p in agent.obs_encoder.parameters() if p.requires_grad] +
                     [p for p in agent.state_encoder.parameters() if p.requires_grad] +
                     [p for p in agent.ftask_norm.parameters() if p.requires_grad]),
                    max_norm=1.0,
                )

                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

                epoch_loss  += loss.item()
                num_batches += 1
                global_step += 1
                iteration   += 1

                if iteration % args.log_freq == 0:
                    writer.add_scalar("train/loss", loss.item(), iteration)
                    if args.track:
                        import wandb
                        wandb.log({"train/loss": loss.item()}, step=global_step)

            avg_loss = epoch_loss / max(num_batches, 1)
            if epoch % args.log_epoch_freq == 0:
                writer.add_scalar("train_epoch/loss", avg_loss, epoch)
                writer.add_scalar("train_epoch/lr", optimizer.param_groups[0]["lr"], epoch)
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
                            best_eval_metrics, args, epoch)
                agent.train()

            if epoch % args.save_epoch_freq == 0 and epoch > 0:
                save_checkpoint(
                    f"runs/{run_name}/checkpoints/epoch_{epoch}.pt",
                    iteration, agent, optimizer, lr_scheduler,
                    best_eval_metrics, args, epoch)

    else:
        pbar = tqdm(total=args.total_iters - start_iteration)
        for epoch in range(10000):
            for batch in dataloader:
                if iteration >= args.total_iters:
                    break
                with torch.autocast("cuda", dtype=_amp_dtype, enabled=_use_amp):
                    loss = agent.compute_loss(batch)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in agent.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                if iteration % args.log_freq == 0:
                    writer.add_scalar("train/loss", loss.item(), iteration)
                iteration += 1
                pbar.update(1)
                pbar.set_description(f"Loss: {loss.item():.4f}")
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