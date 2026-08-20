"""
Single-Task Diffusion Policy Training (LeRobot)
================================================
No task encoder, no human dataset.  Trains a ConditionalUnet1D conditioned
only on obs_latent + state_latent — the fastest and simplest baseline.

Architecture:
  obs_encoder   →  obs_feat  (obs_latent_dim)
  state_encoder →  state_feat (state_latent_dim)
  [obs_feat ; state_feat]  →  global_cond  →  ConditionalUnet1D

Usage:
  python train_dp_single.py \
    --sub-config config/sub_configs/L0_TwoRobotCleanCup-v1.json \
    --sim-root   /data/sim \
    --output-dir runs/dp_single_task/L0_CleanCup
"""

ALGO_NAME = "DPSingleTask"

import os, random, time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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

from examples.baselines.lerobot_dataset.lerobot_dataloader import (
    LeRobotDataConfig,
    build_lerobot_dataset,
)
from examples.baselines.encoders.obs_encoder import ObservationEncoder, ObsEncoderConfig
from examples.baselines.encoders.state_encoder import StateEncoder, StateEncoderConfig
from examples.baselines.diffusion_policy.diffusion_policy.conditional_unet1d import ConditionalUnet1D


# =============================================================================
# Training arguments
# =============================================================================

@dataclass
class TrainingArgs:
    """Training configuration for single-task Diffusion Policy."""

    # ── Experiment ────────────────────────────────────────────────────────────
    exp_name:           Optional[str] = None
    seed:               int  = 1
    cuda:               bool = True
    track:              bool = False
    wandb_project_name: str  = "DPSingleTask"
    wandb_entity:       Optional[str] = None

    # ── Data ──────────────────────────────────────────────────────────────────
    sub_config:  str  = ""           # single-task sub_config JSON
    sim_root:    str  = "./data/sim"
    state_type:  str  = "qpos"       # qpos | eepos | mixpos
    cameras:     List[str] = field(default_factory=lambda: ["cam1", "cam2"])
    include_depth: bool = False
    single_arm:  bool = False
    image_size:  List[int] = field(default_factory=lambda: [224, 224])

    # ── Model dims ────────────────────────────────────────────────────────────
    state_dim:         int = 18
    action_dim:        int = 16
    obs_latent_dim:    int = 256
    state_latent_dim:  int = 64
    obs_encoder_type:  str = "resnet18"
    obs_freeze_backbone: bool = False
    obs_finetune_layers: int = 2
    state_type_enc:    str = "mlp"   # mlp | transformer
    state_hidden_dim:  int = 128
    state_num_layers:  int = 2
    state_num_heads:   int = 4

    # ── Diffusion UNet ────────────────────────────────────────────────────────
    pred_horizon:   int = 16
    obs_horizon:    int = 1
    unet_dims:      List[int] = field(default_factory=lambda: [256, 512, 1024])
    unet_kernel_size: int = 5
    n_groups:       int = 8
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    beta_schedule:  str = "squaredcos_cap_v2"
    clip_sample:    bool = True
    prediction_type: str = "epsilon"

    # ── Training ──────────────────────────────────────────────────────────────
    total_epochs:    int   = 30       # number of full passes over the dataset
    save_freq_epochs: int  = 5        # save a checkpoint every N epochs
    batch_size:      int   = 64
    lr:              float = 1e-4
    lr_warmup_steps: int   = 1_000
    weight_decay:    float = 1e-6
    grad_clip:       float = 1.0
    num_workers:     int   = 4
    log_freq:        int   = 100      # log every N gradient steps (within epoch)

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "runs/dp_single_task"


# =============================================================================
# Agent
# =============================================================================

class DPSingleTaskAgent(nn.Module):
    """Diffusion Policy — single-task, no task encoder."""

    def __init__(self, args: TrainingArgs, device: torch.device):
        super().__init__()
        self.args   = args
        self.device = device
        self.pred_horizon = args.pred_horizon
        self.action_dim   = args.action_dim
        self.obs_horizon  = args.obs_horizon

        # ── Obs encoder ───────────────────────────────────────────────────────
        self.obs_encoder = ObservationEncoder(ObsEncoderConfig(
            encoder_type=args.obs_encoder_type,
            image_size=args.image_size[0],
            output_dim=args.obs_latent_dim,
            hidden_dim=args.obs_latent_dim,
            freeze_backbone=args.obs_freeze_backbone,
            finetune_layers=args.obs_finetune_layers,
        )).to(device)

        # ── State encoder ─────────────────────────────────────────────────────
        self.state_encoder = StateEncoder(StateEncoderConfig(
            state_type=args.state_type,
            state_dim=args.state_dim,
            num_frames=args.obs_horizon,
            encoder_type=args.state_type_enc,
            hidden_dim=args.state_hidden_dim,
            output_dim=args.state_latent_dim,
            num_layers=args.state_num_layers,
            num_heads=args.state_num_heads,
        )).to(device)

        global_cond_dim = args.obs_latent_dim + args.state_latent_dim

        # ── ConditionalUnet1D ─────────────────────────────────────────────────
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=args.action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=global_cond_dim,
            down_dims=args.unet_dims,
            kernel_size=args.unet_kernel_size,
            n_groups=args.n_groups,
        ).to(device)

        # ── DDPM scheduler ────────────────────────────────────────────────────
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule=args.beta_schedule,
            clip_sample=args.clip_sample,
            prediction_type=args.prediction_type,
        )

        param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  DPSingleTaskAgent: {param_count/1e6:.1f}M params | "
              f"global_cond_dim={global_cond_dim}")

    # ------------------------------------------------------------------
    def _encode_obs(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        rgb  : (B, C, H, W) or (B, T, C, H, W)
        state: (B, D)        or (B, T, D)
        """
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(1)       # (B, C, H, W) → (B, 1, C, H, W)
        if state.dim() == 2:
            state = state.unsqueeze(1)   # (B, D) → (B, 1, D)

        T = min(state.shape[1], self.obs_horizon)

        obs_in = rgb[:, :T]
        if T == 1:
            obs_in = obs_in.squeeze(1)   # (B, 1, C, H, W) → (B, C, H, W)

        obs_feat,   _ = self.obs_encoder(obs_in)           # (B, obs_latent_dim)
        state_feat, _ = self.state_encoder(state[:, :T])   # (B, state_latent_dim)
        return torch.cat([obs_feat, state_feat], dim=-1)

    # ------------------------------------------------------------------
    def compute_loss(self, batch: Dict) -> torch.Tensor:
        robot_obs = batch.get("robot_obs", batch)
        rgb    = robot_obs.get("rgb",   robot_obs.get("view_1")).to(self.device)   # (B, C, H, W)
        state  = robot_obs.get("state", robot_obs.get("states")).to(self.device)   # (B, state_dim)
        action = batch.get("robot_actions", batch.get("actions", batch.get("action"))).to(self.device)   # (B, pred_horizon, action_dim)

        B = rgb.shape[0]

        global_cond = self._encode_obs(rgb, state)   # (B, obs_cond_dim)

        # DDPM forward: add noise and predict
        noise = torch.randn_like(action)
        t     = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                              (B,), device=self.device).long()
        noisy_action = self.noise_scheduler.add_noise(action, noise, t)

        noise_pred = self.noise_pred_net(noisy_action, t, global_cond=global_cond)
        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Inference: DDPM denoising.

        Parameters
        ----------
        rgb   : (B, C, H, W)  normalized robot observation
        state : (B, state_dim) normalized state

        Returns
        -------
        action : (B, pred_horizon, action_dim) in normalized space
        """
        B = rgb.shape[0]
        self.noise_scheduler.set_timesteps(self.args.num_inference_steps)

        global_cond  = self._encode_obs(rgb, state)
        action       = torch.randn(B, self.pred_horizon, self.action_dim, device=self.device)

        for t in self.noise_scheduler.timesteps:
            t_batch   = t.unsqueeze(0).expand(B).to(self.device)
            noise_pred = self.noise_pred_net(action, t_batch, global_cond=global_cond)
            action    = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        return action   # (B, pred_horizon, action_dim)


# =============================================================================
# Dataset / DataLoader helpers
# =============================================================================

def _auto_detect_cameras(sim_root: str, dataset_file: str,
                          fallback: List[str]) -> List[str]:
    """Peek at meta.features to find real camera names (works for video format)."""
    try:
        import json as _j, os as _o
        from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset as _D
        with open(dataset_file) as _f:
            _c = _j.load(_f)[0]
        _probe = _D(repo_id=_c["repo_id"],
                    root=_o.path.join(sim_root, _c.get("root", _c["repo_id"])),
                    delta_timestamps=None)
        _cams = [k.replace("observation.images.", "")
                 for k, v in _probe.meta.features.items()
                 if k.startswith("observation.images.") and not k.endswith("_depth")
                 and v.get("dtype") in ("image", "video")]
        if _cams:
            if set(_cams) != set(fallback):
                print(f"  📷 Auto-detected cameras: {_cams}  (config had: {fallback})")
            return _cams
    except Exception as _e:
        print(f"  ⚠️  Camera auto-detect failed ({_e}); using {fallback}")
    return fallback


def build_dataloader(args: TrainingArgs):
    # Mirrors the paired dataset approach: use LeRobotDataConfig +
    # build_lerobot_dataset so tolerance_s=0.05 is applied (not the
    # default 0.0001 in LeRobotSimDataConfig which causes FrameTimestampError).
    cameras = _auto_detect_cameras(args.sim_root, args.sub_config, args.cameras)
    cfg = LeRobotDataConfig(
        source_type="sim",
        root=args.sim_root,
        split="train",
        dataset_file=args.sub_config,
        cameras=cameras,
        include_depth=args.include_depth,
        image_size=tuple(args.image_size),
        state_type=args.state_type,
        single_arm=args.single_arm,
        horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        fps=30,
        video_backend="torchcodec",
        tolerance_s=0.05,
        enable_augmentation=True,
    )
    dataset = build_lerobot_dataset(cfg)
    print(f"  Dataset size: {len(dataset)} samples")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, dataset


# =============================================================================
# Training loop
# =============================================================================

def train(args: TrainingArgs):
    # ── Seed ──────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.exp_name or (
        f"{ALGO_NAME}_{Path(args.sub_config).stem}_{args.seed}"
    )
    writer = SummaryWriter(str(output_dir / "tb" / run_name))

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   name=run_name, config=vars(args), sync_tensorboard=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n📂 Loading dataset...")
    dataloader, dataset = build_dataloader(args)

    # ── Agent ─────────────────────────────────────────────────────────────────
    print("\n🤖 Building agent...")
    agent = DPSingleTaskAgent(args, device)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    steps_per_epoch = len(dataloader)
    total_steps     = args.total_epochs * steps_per_epoch

    optimizer = optim.AdamW(agent.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n🚀 Training {run_name} for {args.total_epochs} epochs "
          f"({steps_per_epoch} steps/epoch)\n")
    agent.train()
    losses    = defaultdict(list)
    global_step = 0

    for epoch in tqdm(range(1, args.total_epochs + 1), desc="Epochs"):
        epoch_losses: list = []

        for batch in dataloader:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = agent.compute_loss(batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            loss_val = loss.item()
            losses["loss"].append(loss_val)
            epoch_losses.append(loss_val)

            if global_step % args.log_freq == 0:
                avg_loss = np.mean(losses["loss"])
                losses["loss"].clear()
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                tqdm.write(f"[E{epoch:03d}/{args.total_epochs} "
                           f"S{global_step:>8}] loss={avg_loss:.5f}")

        # ── End-of-epoch bookkeeping ──────────────────────────────────────────
        epoch_avg = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        writer.add_scalar("train/epoch_loss", epoch_avg, epoch)
        tqdm.write(f"── Epoch {epoch:03d}/{args.total_epochs} avg_loss={epoch_avg:.5f}")

        if epoch % args.save_freq_epochs == 0 or epoch == args.total_epochs:
            ckpt_path = output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch":           epoch,
                "global_step":     global_step,
                "agent_state_dict": agent.state_dict(),
                "optimizer":       optimizer.state_dict(),
                "args":            vars(args),
                "dataset_stats":   [
                    {
                        "dataset_idx": idx,
                        **dataset.normalizer.dataset_info.get(idx, {}),
                        "stats": dataset.normalizer.dataset_stats.get(idx, {}),
                    }
                    for idx in sorted(dataset.normalizer.dataset_stats.keys())
                ],
                "dataset_info":    [{
                    "repo_id": ds.repo_id if hasattr(ds, "repo_id") else "",
                    "root":    str(ds.root),
                } for ds in dataset.sub_datasets] if hasattr(dataset, "sub_datasets") else [],
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

    writer.close()
    if args.track:
        import wandb; wandb.finish()
    print("\n✅ Training complete.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    train(tyro.cli(TrainingArgs))