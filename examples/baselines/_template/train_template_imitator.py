"""
Minimal baseline template: train an imitation policy.

This file shows the MINIMUM needed to train a new imitation-learning model
on The Imitator Game benchmark. Keep the TemplateAgent interface unchanged
and replace only the policy network to plug in your own model.

Key interfaces you must implement:
    TemplateAgent.compute_loss(batch)   -> dict with "loss"   (training)
    TemplateAgent.prepare_for_eval(...) -> cache task features (per episode)
    TemplateAgent.get_action(obs)       -> (B, pred_horizon, action_dim)
    TemplateAgent.clear_cache()         -> drop cached features

Dataset contract (see examples/baselines/lerobot_dataset/README.md):
    - Human demonstration videos + paired simulation robot trajectories.
    - Every sample provides: robot_obs, robot_actions, human_video, ...
    - Config JSON files under config/exp_configs/ select which tasks to use.
"""

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanSimPairedDataset,
    PairedDatasetConfig,
    InputMode,
    get_collate_fn,
)
from examples.baselines.encoders.obs_encoder import ObservationEncoder, ObsEncoderConfig
from examples.baselines.encoders.state_encoder import StateEncoder, StateEncoderConfig


# --------------------------------------------------------------------------- #
# Config (all CLI args are auto-generated from this dataclass by tyro)
# --------------------------------------------------------------------------- #
@dataclass
class TrainingArgs:
    # Experiment
    exp_name: Optional[str] = None
    seed: int = 1
    cuda: bool = True

    # --- Dataset (The Imitator Game data contract) ---
    # demos/demo_data      -> human demonstration videos (LeRobot)
    # demos/imitator_data  -> paired simulation robot demos (LeRobot)
    human_root: str = "demos/demo_data"
    sim_root: str = "demos/imitator_data"
    task_mapping_file: str = "examples/baselines/lerobot_dataset/task_mapping.json"
    human_dataset_file: str = "examples/baselines/lerobot_dataset/config/exp_configs/human_train_config_15.json"
    sim_dataset_file: str = "examples/baselines/lerobot_dataset/config/exp_configs/sim_train_config_15.json"
    human_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    sim_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"

    # --- Training ---
    total_epochs: int = 10
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    grad_clip: float = 1.0

    # --- Data shapes (dual-panda default) ---
    action_dim: int = 16
    state_dim: int = 18
    pred_horizon: int = 16      # action chunk length
    obs_horizon: int = 1

    # --- Dataset options ---
    input_mode: str = "video_only"   # video_only | language | video_language
    cameras: List[str] = field(default_factory=lambda: ["zed2i"])
    image_size: Tuple[int, int] = (224, 224)
    state_type: str = "qpos"
    include_depth: bool = False
    single_arm: bool = False
    fps: int = 30
    video_backend: str = "torchcodec"

    # --- Frozen video backbone (task encoder) ---
    frozen_backbone_type: str = "dinov2_vitl14"
    frozen_backbone_num_frames: int = 10
    frozen_backbone_seq_patches: int = 32
    frozen_backbone_adapter_layers: int = 1
    task_latent_dim: int = 256
    hf_cache_dir: Optional[str] = None

    # --- Observation / state encoders ---
    obs_encoder_type: str = "simple_cnn"
    obs_latent_dim: int = 256
    state_encoder_type: str = "mlp"
    state_latent_dim: int = 256

    # --- Policy (replace this) ---
    hidden_dim: int = 1024
    num_layers: int = 3

    # --- Logging ---
    log_freq: int = 20
    save_epoch_freq: int = 5


# --------------------------------------------------------------------------- #
# Policy network (REPLACE THIS with your own model)
# --------------------------------------------------------------------------- #
class TemplatePolicy(nn.Module):
    """Minimal MLP that maps a conditioning vector to an action chunk."""

    def __init__(self, cond_dim: int, hidden_dim: int, num_layers: int,
                 pred_horizon: int, action_dim: int):
        super().__init__()
        layers = []
        dim = cond_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            dim = hidden_dim
        layers.append(nn.Linear(dim, pred_horizon * action_dim))
        self.net = nn.Sequential(*layers)
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond).view(-1, self.pred_horizon, self.action_dim)


# --------------------------------------------------------------------------- #
# Agent: the interface the benchmark evaluator relies on
# --------------------------------------------------------------------------- #
class TemplateAgent(nn.Module):
    def __init__(self, args: TrainingArgs, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.pred_horizon = args.pred_horizon
        self.action_dim = args.action_dim
        self.obs_horizon = args.obs_horizon
        self.input_mode = InputMode(args.input_mode)

        # 1. Task encoder: human demonstration video -> task feature vector
        from examples.baselines.encoders.task_encoder.video_backbone import build_video_backbone
        self.task_encoder = build_video_backbone(
            backbone_type=args.frozen_backbone_type,
            latent_dim=args.task_latent_dim,
            max_seq_patches=args.frozen_backbone_seq_patches,
            adapter_layers=args.frozen_backbone_adapter_layers,
            num_sampled_frames=args.frozen_backbone_num_frames,
            hf_cache_dir=args.hf_cache_dir,
        ).to(device)
        self.task_norm = nn.LayerNorm(args.task_latent_dim)

        # 2. Robot RGB + state encoders
        self.obs_encoder = ObservationEncoder(ObsEncoderConfig(
            encoder_type=args.obs_encoder_type, image_size=args.image_size[0],
            output_dim=args.obs_latent_dim, hidden_dim=args.obs_latent_dim))
        self.state_encoder = StateEncoder(StateEncoderConfig(
            state_type=args.state_type, state_dim=args.state_dim,
            num_frames=args.obs_horizon, encoder_type=args.state_encoder_type,
            output_dim=args.state_latent_dim, hidden_dim=args.state_latent_dim))

        # 3. Policy (replace with your own model)
        self.policy = TemplatePolicy(
            cond_dim=args.task_latent_dim + args.obs_latent_dim + args.state_latent_dim,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            pred_horizon=args.pred_horizon, action_dim=args.action_dim)

        self.cached_task_z = None
        self.to(device)

    # ------------------------------------------------------------------ #
    # Tensor formatting helpers
    # ------------------------------------------------------------------ #
    def _format_image(self, image: torch.Tensor) -> torch.Tensor:
        """Accept (B,H,W,C) / (B,C,H,W) / (B,T,...) and output channel-first."""
        if image.dim() == 5:
            image = image[:, : self.obs_horizon]
            if image.shape[-1] in [3, 4]:
                image = image.permute(0, 1, 4, 2, 3)
            if image.shape[1] == 1:
                image = image[:, 0]
        elif image.dim() == 4 and image.shape[-1] in [3, 4]:
            image = image.permute(0, 3, 1, 2)
        return image.float()

    def _format_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 2:
            state = state.unsqueeze(1)
        return state[:, : self.obs_horizon].float()

    def _get_rgb_and_state(self, obs: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        image = obs.get("rgb", obs.get("view_1"))
        state = obs.get("state", obs.get("states"))
        assert image is not None, "obs must contain 'rgb' or 'view_1'"
        assert state is not None, "obs must contain 'state' or 'states'"
        return self._format_image(image), self._format_state(state)

    # ------------------------------------------------------------------ #
    # Encoders
    # ------------------------------------------------------------------ #
    def encode_robot_obs(self, robot_obs: Dict) -> torch.Tensor:
        image, state = self._get_rgb_and_state(robot_obs)
        obs_z, _ = self.obs_encoder(image.to(self.device))
        state_z, _ = self.state_encoder(state.to(self.device))
        return torch.cat([obs_z, state_z], dim=-1)

    def encode_task(self, human_video: Optional[torch.Tensor],
                    human_vl_ids: Optional[List[str]] = None) -> torch.Tensor:
        if human_video is not None:
            human_video = human_video.to(self.device)
        out = self.task_encoder.encode(human_video=human_video, human_vl_ids=human_vl_ids)
        return self.task_norm(out["z"])

    # ------------------------------------------------------------------ #
    # Training API
    # ------------------------------------------------------------------ #
    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Training entry: predict the action chunk from paired data."""
        robot_obs = {k: v.to(self.device, non_blocking=True)
                     for k, v in batch["robot_obs"].items()}
        target = batch["robot_actions"].to(self.device, non_blocking=True)

        robot_z = self.encode_robot_obs(robot_obs)
        task_z = self.encode_task(human_video=batch.get("human_video"),
                                  human_vl_ids=batch.get("human_repo_id"))
        pred = self.policy(torch.cat([robot_z, task_z], dim=-1))
        return {"loss": F.l1_loss(pred, target), "action_loss": pred.detach()}

    # ------------------------------------------------------------------ #
    # Evaluation API (used by the simulator)
    # ------------------------------------------------------------------ #
    def prepare_for_eval(self, human_video: Optional[torch.Tensor],
                         robot_obs: Dict, human_desc: Optional[List[str]] = None,
                         human_vl_ids: Optional[List[str]] = None):
        """Called once per episode: encode the task and cache its features."""
        del robot_obs, human_desc
        self.eval()
        with torch.no_grad():
            self.cached_task_z = self.encode_task(human_video, human_vl_ids)

    @torch.no_grad()
    def get_action(self, obs: Dict) -> torch.Tensor:
        """Called at every control step. Returns (B, pred_horizon, action_dim)."""
        assert self.cached_task_z is not None, "call prepare_for_eval() first"
        obs = {k: v.to(self.device) for k, v in obs.items()}
        robot_z = self.encode_robot_obs(obs)
        task_z = self.cached_task_z
        if task_z.shape[0] != robot_z.shape[0]:
            task_z = task_z.expand(robot_z.shape[0], -1)
        return self.policy(torch.cat([robot_z, task_z], dim=-1))

    def clear_cache(self):
        self.cached_task_z = None


# --------------------------------------------------------------------------- #
# Dataset: paired human-video + simulation-robot loader
# --------------------------------------------------------------------------- #
def build_dataloader(args: TrainingArgs) -> DataLoader:
    dataset = HumanSimPairedDataset(PairedDatasetConfig(
        human_root=args.human_root, sim_root=args.sim_root,
        task_mapping_file=args.task_mapping_file,
        human_dataset_file=args.human_dataset_file,
        sim_dataset_file=args.sim_dataset_file,
        human_task_description_file=args.human_task_description_file,
        sim_task_description_file=args.sim_task_description_file,
        split="train", cameras=args.cameras, include_depth=args.include_depth,
        image_size=args.image_size, num_frames=args.frozen_backbone_num_frames,
        horizon=args.pred_horizon, obs_horizon=args.obs_horizon,
        state_type=args.state_type, single_arm=args.single_arm, fps=args.fps,
        video_backend=args.video_backend, input_mode=args.input_mode,
        include_first_frame=False))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                      num_workers=args.num_workers,
                      collate_fn=get_collate_fn(args.input_mode),
                      pin_memory=True, drop_last=True)


def save_checkpoint(path: str, agent: TemplateAgent, optimizer: optim.Optimizer,
                    args: TrainingArgs, epoch: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "args": vars(args),
                "agent_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()}, path)
    print(f"Saved checkpoint: {path}")


def main():
    args = tyro.cli(TrainingArgs)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    run_name = args.exp_name or f"template__seed_{args.seed}__{int(time.time())}"
    log_dir = Path("runs") / run_name
    writer = SummaryWriter(str(log_dir))

    dataloader = build_dataloader(args)
    agent = TemplateAgent(args, device)
    optimizer = optim.AdamW([p for p in agent.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(args.total_epochs):
        agent.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss = agent.compute_loss(batch)["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in agent.parameters() if p.grad is not None], args.grad_clip)
            optimizer.step()
            if global_step % args.log_freq == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                pbar.set_postfix(loss=f"{loss.item():.5f}")
            global_step += 1
        if epoch > 0 and epoch % args.save_epoch_freq == 0:
            save_checkpoint(str(log_dir / "checkpoints" / f"epoch_{epoch}.pt"),
                            agent, optimizer, args, epoch)

    save_checkpoint(str(log_dir / "checkpoints" / "final_model.pt"),
                    agent, optimizer, args, args.total_epochs)
    writer.close()


if __name__ == "__main__":
    main()
