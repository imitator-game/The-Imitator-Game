#!/usr/bin/env python3
"""
eval_xskill_real_agent.py
=========================

XSkillRealAgent for real-robot deployment through real_scripts/eval_real_robot.py.

Why this file exists
--------------------
The normal XSkill eval loop maintains two pieces of state outside the base
XSkillAgent:

  1. human demo prototype cache:
       human_video -> stage1 XSkill encoder -> _cached_proto_snap

  2. robot RGB history buffer:
       current real-robot RGB frame -> rgb_history_buffer
       -> sample 4 frames from recent 30-frame window
       -> obs["rgb"] = (B, 4, C, H, W)

For real robot deployment, eval_real_robot.py expects every model agent to expose:

  - prepare_for_eval(...)
  - get_action(obs)
  - clear_cache()

So this file wraps XSkill as a real-agent-compatible class and keeps the RGB
history buffer inside the agent itself.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torchvision.transforms as Tr
from torchvision import transforms

import hydra
import re
from omegaconf import OmegaConf

from xskill.model.diffusion_model import get_resnet, replace_bn_with_gn
from xskill.model.encoder import ResnetConv

def _find_stage2_run_dir(checkpoint_path: str) -> Optional[Path]:
    """
    Given a checkpoint path, walk upward to find a directory named like:
      stage2_15_robot
      stage2_30_robot
      stage2_45_robot
      stage2_15_robot_finetune
      stage2_30_robot_finetune
      stage2_45_robot_finetune
      stage2_15_robot_scratch
      stage2_30_robot_scratch
      stage2_45_robot_scratch
    """
    p = Path(checkpoint_path).resolve()
    cur = p if p.is_dir() else p.parent

    pat = re.compile(r"^stage2_(\d+)_robot(?:_finetune|_scratch)?$")

    for candidate in [cur, *cur.parents]:
        if pat.match(candidate.name):
            return candidate

    return None


def _infer_stage1_dir_from_stage2_ckpt(checkpoint_path: str) -> Optional[Path]:
    """
    If stage2 checkpoint is under stage2_{N}_robot{optional_suffix},
    override cfg_dict['pretrain_path'] with sibling stage1_{N}_robot.

    Examples:
    stage2_15_robot          -> stage1_15_robot
    stage2_15_robot_finetune -> stage1_15_robot
    stage2_15_robot_scratch  -> stage1_15_robot
    """
    stage2_dir = _find_stage2_run_dir(checkpoint_path)
    if stage2_dir is None:
        return None

    m = re.match(r"^stage2_(\d+)_robot(?:_finetune|_scratch)?$", stage2_dir.name)
    if m is None:
        return None

    demos = m.group(1)
    stage1_name = f"stage1_{demos}_robot"
    stage1_dir = stage2_dir.parent / stage1_name

    if stage1_dir.exists():
        return stage1_dir

    print(
        "[XSkillRealAgent] inferred stage1 dir does not exist: "
        f"{stage1_dir}"
    )
    return None


def _patch_stage1_path_from_stage2_ckpt(
    cfg_dict: Dict[str, Any],
    checkpoint_path: str,
) -> Dict[str, Any]:
    """
    Minimal patch:
      If stage2 checkpoint is under stage2_xx_robot{suffix},
      override cfg_dict['pretrain_path'] with sibling stage1_30_robot{suffix}
      when that directory exists.

    This keeps old behavior as fallback.
    """
    inferred = _infer_stage1_dir_from_stage2_ckpt(checkpoint_path)

    if inferred is None:
        print(
            "[XSkillRealAgent] Could not infer sibling stage1 dir from stage2 ckpt; "
            f"using cfg pretrain_path={cfg_dict.get('pretrain_path')}"
        )
        return cfg_dict

    old = cfg_dict.get("pretrain_path", None)
    cfg_dict["pretrain_path"] = str(inferred)

    print(
        "[XSkillRealAgent] stage1 pretrain_path patched from stage2 ckpt:\n"
        f"  old: {old}\n"
        f"  new: {cfg_dict['pretrain_path']}"
    )

    return cfg_dict

# =============================================================================
# XSkillRealAgent
# =============================================================================

class XSkillRealAgent(nn.Module):
    """
    Real-robot compatible XSkill wrapper.

    Required by eval_real_robot.py:
      - prepare_for_eval(...)
      - get_action(obs)
      - clear_cache()

    Input expected from RealRobotController._build_obs():
      obs["state"]: (B, state_dim), already normalized by robot_normalizer
      obs["rgb"]  : (B, H, W, C_total), float in [0, 1]

    Internal XSkill input:
      states: (B, obs_horizon, obs_dim)
      rgb   : (B, real_num_frames, C_total, H, W)

    Output:
      actions: (B, pred_horizon, action_dim), normalized action space
    """

    def __init__(
        self,
        nets: nn.ModuleDict,
        noise_scheduler,
        xskill_model: nn.Module,
        proto_pipeline: nn.Module,
        cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        super().__init__()

        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.xskill_model = xskill_model
        self.proto_pipeline = proto_pipeline
        self.device = device

        # Core config
        self.obs_horizon = int(cfg["obs_horizon"])
        self.pred_horizon = int(cfg["pred_horizon"])
        self.action_dim = int(cfg["action_dim"])
        self.obs_dim = int(cfg["obs_dim"])
        self.vision_feature_dim = int(cfg["vision_feature_dim"])

        # Prototype config
        self.use_proto = bool(cfg.get("use_proto", True))
        self.proto_horizon = int(cfg.get("proto_horizon", self.obs_horizon))
        self.proto_dim = int(cfg.get("proto_dim", 256))
        self.upsample_proto = bool(cfg.get("upsample_proto", False))
        self.snap_frames = int(cfg.get("snap_frames", 30))
        self.slide = int(getattr(xskill_model, "slide", 2))

        # Diffusion config
        self.num_diffusion_iters = int(cfg.get("num_diffusion_iters", 60))

        # robot RGB history buffer:
        # current real-robot RGB frame -> rgb_history_buffer
        #   -> sample real_num_frames from recent real_rgb_window
        #   -> align to obs_horizon before feeding stage2 policy
        self.real_num_frames = int(cfg.get("real_num_frames", 4))
        self.real_rgb_window = int(cfg.get("real_rgb_window", 30))
        self._rgb_history_buffer = deque(maxlen=self.real_rgb_window)

        # Cached human-demo prototypes, set by prepare_for_eval()
        self._cached_proto_snap: Optional[torch.Tensor] = None
        self._cached_predict_proto: Optional[torch.Tensor] = None

        # Step counter used for sampling history; reset by prepare_for_eval/clear_cache.
        self._ts = 0

        self.to(device)
        self.eval()

    # -------------------------------------------------------------------------
    # Human demo prototype encoding
    # -------------------------------------------------------------------------

    def _extract_traj_representation(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract trajectory prototypes using pretrained XSkill stage1 model.

        Args:
            images: (B, T, C, H, W), float in [0,1] or [0,255]

        Returns:
            traj_rep: (B, T, proto_dim)
        """
        images = images.to(self.device).float()
        if images.numel() > 0 and images.max() > 1.0:
            images = images / 255.0

        b, t, c, h, w = images.shape

        # The original XSkill eval preprocesses human video with:
        # CenterCrop(112,112) + ImageNet Normalize.
        images = self.proto_pipeline(
            images.reshape(b * t, c, h, w)
        ).reshape(b, t, c, 112, 112)

        if t <= self.slide:
            raise ValueError(
                f"Need T > slide for proto extraction, got T={t}, slide={self.slide}"
            )

        windows = [images[:, j:j + self.slide + 1] for j in range(t - self.slide)]
        im_q_processed = torch.cat(windows, dim=0)

        state_rep = self.xskill_model.encoder_q.get_state_representation(
            im_q_processed, None
        )
        traj_rep = self.xskill_model.encoder_q.get_traj_representation(state_rep)
        traj_rep = traj_rep.reshape(t - self.slide, b, -1).permute(1, 0, 2).contiguous()

        # Pad to original T.
        if traj_rep.shape[1] < t:
            last = traj_rep[:, -1:, :].repeat(1, t - traj_rep.shape[1], 1)
            traj_rep = torch.cat([traj_rep, last], dim=1)

        return traj_rep

    def _sample_proto_snap(self, proto_seq: torch.Tensor) -> torch.Tensor:
        """
        Sample snap_frames evenly from human prototype sequence.

        Args:
            proto_seq: (B, T, proto_dim)

        Returns:
            proto_snap: (B, snap_frames, proto_dim)
        """
        b, t, d = proto_seq.shape

        if t >= self.snap_frames:
            idx = torch.linspace(
                0, t - 1, steps=self.snap_frames, device=proto_seq.device
            ).long()
            return proto_seq[:, idx, :]

        pad = proto_seq[:, -1:, :].repeat(1, self.snap_frames - t, 1)
        return torch.cat([proto_seq, pad], dim=1)

    @torch.no_grad()
    def prepare_for_eval(
        self,
        human_video: Optional[torch.Tensor] = None,
        robot_obs: Optional[Dict[str, torch.Tensor]] = None,
        human_tokens: Optional[torch.Tensor] = None,
        human_desc: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """
        Called once by eval_real_robot.py at task start.

        Compatible with current RealRobotController call:

            agent.prepare_for_eval(
                human_video=human_video,
                robot_obs=obs,
                human_desc=...,
                human_tokens=None,
            )

        XSkill ignores language/human_tokens/human_desc and only uses human_video.
        """
        self.eval()

        # New episode/task: clear robot-frame history.
        self._rgb_history_buffer.clear()
        self._ts = 0
        self._cached_predict_proto = None

        if not self.use_proto:
            self._cached_proto_snap = None
            return

        if human_video is None:
            # Keep this as a hard error because XSkill with use_proto=True expects
            # human demo conditioning. Otherwise it silently becomes a different policy.
            raise ValueError(
                "XSkillRealAgent.prepare_for_eval got human_video=None while "
                "use_proto=True. Check --input-mode video_only/video_and_language "
                "and evaluate_processor.get_video(real_env_id, 1)."
            )

        human_video = human_video.to(self.device)

        # evaluate_processor usually returns (B,T,H,W,C). Convert to (B,T,C,H,W).
        if human_video.ndim == 5 and human_video.shape[-1] in (1, 3, 4):
            human_video = human_video.permute(0, 1, 4, 2, 3).contiguous()

        if human_video.ndim != 5:
            raise ValueError(
                f"Expected human_video shape (B,T,H,W,C) or (B,T,C,H,W), "
                f"got {tuple(human_video.shape)}"
            )

        human_proto_seq = self._extract_traj_representation(human_video)
        self._cached_proto_snap = self._sample_proto_snap(human_proto_seq)

        print(
            "[XSkillRealAgent] prepare_for_eval done: "
            f"human_video={tuple(human_video.shape)}, "
            f"proto_snap={tuple(self._cached_proto_snap.shape)}"
        )

    # -------------------------------------------------------------------------
    # Real RGB history handling
    # -------------------------------------------------------------------------

    def _to_bchw(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert real-robot RGB tensor to BCHW.

        RealRobotController._build_obs() returns:
            (B, H, W, C_total)

        But this also accepts:
            (B, C, H, W)
            (B, T, H, W, C)
            (B, T, C, H, W)
        """
        rgb = rgb.to(self.device).float()

        if rgb.ndim == 4:
            # BHWC -> BCHW
            if rgb.shape[-1] in (1, 3, 4, 6, 8, 9, 12):
                return rgb.permute(0, 3, 1, 2).contiguous()
            # Already BCHW
            return rgb.contiguous()

        if rgb.ndim == 5:
            # BTHWC -> BTCHW, then use last frame.
            if rgb.shape[-1] in (1, 3, 4, 6, 8, 9, 12):
                rgb = rgb.permute(0, 1, 4, 2, 3).contiguous()
            # BTCHW -> last frame BCHW.
            return rgb[:, -1].contiguous()

        raise ValueError(f"Unexpected rgb shape for XSkillRealAgent: {tuple(rgb.shape)}")

    def _push_rgb_and_sample_clip(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Maintain real-robot RGB cache and sample temporal clip.

        This mirrors XSkill eval:
          - append current frame, shape (B,C,H,W)
          - sample real_num_frames evenly over available recent window
          - if not enough unique frames, repeat first available frame

        Returns:
            rgb_clip: (B, real_num_frames, C, H, W)
        """
        cur = self._to_bchw(rgb)
        self._rgb_history_buffer.append(cur.detach())

        hist = list(self._rgb_history_buffer)
        t = len(hist)

        # Match original XSkill eval semantics:
        # indices = linspace(start_idx, sample_idx, num_frames).round()
        # but here sample_idx is local to the deque, so it is t - 1.
        if t <= 0:
            raise RuntimeError("RGB history buffer is empty after append; impossible state.")

        raw_indices = torch.linspace(
            0, t - 1, steps=self.real_num_frames, device=self.device
        ).round().long().tolist()

        # Original eval uses unique indices; if fewer than num_frames, repeat first.
        unique_indices = []
        for idx in raw_indices:
            if idx not in unique_indices:
                unique_indices.append(idx)

        if len(unique_indices) < self.real_num_frames:
            first_idx = unique_indices[0]
            unique_indices = unique_indices + [first_idx] * (
                self.real_num_frames - len(unique_indices)
            )

        frames = [hist[int(i)] for i in unique_indices]
        rgb_clip = torch.stack(frames, dim=1)  # (B,T,C,H,W)
        return rgb_clip

    # -------------------------------------------------------------------------
    # Policy inference
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Called repeatedly by eval_real_robot.py control loop.

        Args:
            obs_dict:
              state/states: (B,D) or (B,T,D)
              rgb/view_1:   real robot uses (B,H,W,C)

        Returns:
            actions: (B, pred_horizon, action_dim), normalized action space
        """
        self.eval()

        states = obs_dict.get("state", obs_dict.get("states"))
        rgb = obs_dict.get("rgb", obs_dict.get("view_1"))

        if states is None:
            raise KeyError("XSkillRealAgent.get_action expected obs['state'] or obs['states']")
        if rgb is None:
            raise KeyError("XSkillRealAgent.get_action expected obs['rgb'] or obs['view_1']")

        states = states.to(self.device).float()
        if states.ndim == 2:
            states = states.unsqueeze(1)  # (B,1,D)
        elif states.ndim != 3:
            raise ValueError(f"Unexpected state shape: {tuple(states.shape)}")

        # Maintain RGB buffer internally.
        nimage = self._push_rgb_and_sample_clip(rgb)  # (B,T,C,H,W)
        b = states.shape[0]

        # Real robot state is usually current state only. XSkill network expects
        # obs_horizon state tokens. Repeat current state if needed.
        if states.shape[1] == 1 and self.obs_horizon > 1:
            states = states.repeat(1, self.obs_horizon, 1)
        elif states.shape[1] > self.obs_horizon:
            states = states[:, -self.obs_horizon:, :]

        # Keep RGB temporal dimension consistent with state obs_horizon if the
        # training cfg expects a specific horizon. In your existing XSkill eval,
        # real_num_frames=4 is used as the visual horizon, so we do not force it
        # to equal obs_horizon. The global_cond_dim must match training cfg.
        # Therefore make sure cfg["obs_horizon"] matches the number of visual
        # frames used during training/eval, or set real_num_frames accordingly.
        #
        # If your hydra cfg has obs_horizon=4, this is naturally aligned.
        if nimage.shape[1] != self.obs_horizon:
            if nimage.shape[1] > self.obs_horizon:
                nimage = nimage[:, -self.obs_horizon:, :, :, :]
            else:
                pad = nimage[:, :1].repeat(1, self.obs_horizon - nimage.shape[1], 1, 1, 1)
                nimage = torch.cat([nimage, pad], dim=1)

        # Encode vision: flatten B*T frames.
        image_features = self.nets["vision_encoder"](nimage.flatten(end_dim=1))
        image_features = image_features.reshape(
            *nimage.shape[:2], -1
        )  # (B, obs_horizon, vision_feature_dim)

        if states.shape[1] != image_features.shape[1]:
            if states.shape[1] == 1:
                states = states.repeat(1, image_features.shape[1], 1)
            else:
                states = states[:, -image_features.shape[1]:, :]

        obs_feature = torch.cat([image_features, states], dim=-1)

        if self.use_proto and self._cached_proto_snap is not None:
            proto_snap = self._cached_proto_snap
            if proto_snap.shape[0] != b:
                if proto_snap.shape[0] == 1:
                    proto_snap = proto_snap.repeat(b, 1, 1)
                else:
                    proto_snap = proto_snap[:b]

            if "proto_pred_net" in self.nets:
                predict_proto = self.nets["proto_pred_net"](
                    obs_feature.flatten(start_dim=1),
                    proto_snap,
                )
                nproto = predict_proto.unsqueeze(1)  # (B,1,proto_dim)
            else:
                nproto = proto_snap[:, -self.proto_horizon:, :]

            if self.upsample_proto and "upsample_proto_net" in self.nets:
                upsample_proto = self.nets["upsample_proto_net"](
                    nproto.flatten(start_dim=1)
                )
                upsample_proto = upsample_proto.reshape(b, self.proto_horizon, -1)
                obs_cond = torch.cat(
                    [
                        obs_feature.flatten(start_dim=1),
                        upsample_proto.flatten(start_dim=1),
                    ],
                    dim=1,
                )
            else:
                obs_cond = torch.cat(
                    [
                        obs_feature.flatten(start_dim=1),
                        nproto.flatten(start_dim=1),
                    ],
                    dim=1,
                )
        else:
            obs_cond = obs_feature.flatten(start_dim=1)

        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        actions = torch.randn(
            (b, self.pred_horizon, self.action_dim),
            device=self.device,
        )

        for k in self.noise_scheduler.timesteps:
            kk = k.expand(b).to(self.device) if hasattr(k, "expand") else torch.full(
                (b,), int(k), device=self.device, dtype=torch.long
            )
            noise_pred = self.nets["noise_pred_net"](
                actions,
                kk,
                global_cond=obs_cond,
            )
            actions = self.noise_scheduler.step(noise_pred, k, actions).prev_sample

        self._ts += 1
        return actions

    def clear_cache(self) -> None:
        """
        Clear all episode-level caches.

        For real robot:
          - called manually when starting a new task/episode
          - also safe to call on shutdown/restart
        """
        self._cached_proto_snap = None
        self._cached_predict_proto = None
        self._rgb_history_buffer.clear()
        self._ts = 0


# =============================================================================
# Checkpoint loading
# =============================================================================

def _load_xskill_stage1_model(cfg_dict: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    Load pretrained stage1 XSkill model for human prototype extraction.
    Mirrors the original XSkill eval loader.
    """
    config_path = Path(cfg_dict["pretrain_path"]) / ".hydra" / "config.yaml"
    print(f"[XSkillRealAgent] Loading stage1 config: {config_path}")

    exp_cfg = OmegaConf.load(str(config_path))
    model = hydra.utils.instantiate(exp_cfg.Model).to(device)

    ckpt_id = int(cfg_dict["pretrain_ckpt"])
    candidates = [
        Path(cfg_dict["pretrain_path"]) / f"{ckpt_id}.ckpt",
        Path(cfg_dict["pretrain_path"]) / f"{ckpt_id:02d}.ckpt",
        Path(cfg_dict["pretrain_path"]) / f"{ckpt_id:04d}.ckpt",
        Path(cfg_dict["pretrain_path"]) / f"epoch={ckpt_id}.ckpt",
        Path(cfg_dict["pretrain_path"]) / f"epoch={ckpt_id:02d}.ckpt",
        Path(cfg_dict["pretrain_path"]) / f"epoch={ckpt_id:04d}.ckpt",
    ]

    ckpt_path = None
    for c in candidates:
        if c.exists():
            ckpt_path = c
            break

    if ckpt_path is None:
        all_ckpts = sorted(Path(cfg_dict["pretrain_path"]).glob("*.ckpt"))
        if not all_ckpts:
            raise FileNotFoundError(f"No stage1 checkpoint under {cfg_dict['pretrain_path']}")
        ckpt_path = all_ckpts[-1]
        print(
            f"[XSkillRealAgent] pretrain_ckpt={ckpt_id} not found; "
            f"using latest: {ckpt_path}"
        )

    print(f"[XSkillRealAgent] Loading stage1 checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def _resolve_hydra_config_path(checkpoint_path: str, args=None) -> str:
    """
    Resolve hydra config from the same run folder as the stage2 checkpoint.

    Expected layouts:
      stage2_15_robot/checkpoints/epoch04.ckpt
        -> stage2_15_robot/hydra_config.yaml
        -> stage2_15_robot/config.yaml
        -> stage2_15_robot/.hydra/config.yaml

      stage2_15_robot/epoch04.ckpt
        -> stage2_15_robot/hydra_config.yaml
        -> stage2_15_robot/config.yaml
        -> stage2_15_robot/.hydra/config.yaml
    """
    ckpt = Path(checkpoint_path).resolve()
    ckpt_dir = ckpt if ckpt.is_dir() else ckpt.parent

    # If the checkpoint lives under checkpoints/, step back to the run directory.
    run_dir = ckpt_dir.parent if ckpt_dir.name == "checkpoints" else ckpt_dir

    candidates = [
        run_dir / "hydra_config.yaml",
        run_dir / "config.yaml",
        run_dir / ".hydra" / "config.yaml",
        ckpt_dir / "hydra_config.yaml",
        ckpt_dir / "config.yaml",
        ckpt_dir / ".hydra" / "config.yaml",
    ]

    for c in candidates:
        if c.exists():
            print(f"[XSkillRealAgent] hydra config: {c}")
            return str(c)

    raise FileNotFoundError(
        "[XSkillRealAgent] Cannot find hydra config near checkpoint.\n"
        f"  checkpoint: {checkpoint_path}\n"
        f"  searched:\n" +
        "\n".join(f"    - {p}" for p in candidates)
    )


def load_agent_from_checkpoint(
    checkpoint_path: str,
    args,
    device: torch.device,
):
    """
    Real-robot loader signature expected by eval_real_robot.py:

        load_agent_from_checkpoint(checkpoint_path, args, device)

    Returns:
        agent, model_config
    """
    hydra_config_path = _resolve_hydra_config_path(checkpoint_path, args)

    cfg = OmegaConf.load(hydra_config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    cfg_dict = _patch_stage1_path_from_stage2_ckpt(
        cfg_dict=cfg_dict,
        checkpoint_path=checkpoint_path,
    )

    obs_horizon = int(cfg_dict.get("obs_horizon", getattr(args, "obs_horizon", 1)))
    obs_dim = int(cfg_dict.get("obs_dim", getattr(args, "state_dim", 8)))
    action_dim = int(cfg_dict.get("action_dim", getattr(args, "action_dim", 16)))
    vision_feature_dim = int(cfg_dict.get("vision_feature_dim", 64))
    use_proto = bool(cfg_dict.get("use_proto", True))
    proto_horizon = int(cfg_dict.get("proto_horizon", obs_horizon))
    proto_dim = int(cfg_dict.get("proto_dim", 256))
    upsample_proto = bool(cfg_dict.get("upsample_proto", False))

    # Build vision encoder.
    if vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18")
    else:
        vision_encoder = ResnetConv(embedding_size=vision_feature_dim)
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # Compute global_cond_dim exactly as training/eval did.
    if use_proto and upsample_proto:
        upsample_out_size = cfg_dict.get("upsample_proto_net", {}).get("out_size", 256)
        global_cond_dim = (
            vision_feature_dim * obs_horizon
            + obs_dim * obs_horizon
            + proto_horizon * upsample_out_size
        )
    elif use_proto:
        global_cond_dim = (
            vision_feature_dim * obs_horizon
            + obs_dim * obs_horizon
            + proto_horizon * proto_dim
        )
    else:
        global_cond_dim = vision_feature_dim * obs_horizon + obs_dim * obs_horizon

    noise_pred_net = hydra.utils.instantiate(
        cfg.noise_pred_net,
        global_cond_dim=global_cond_dim,
    )

    nets = nn.ModuleDict({
        "vision_encoder": vision_encoder,
        "noise_pred_net": noise_pred_net,
    })

    if use_proto:
        proto_pred_net = hydra.utils.instantiate(
            cfg.proto_pred_net,
            input_dim=vision_feature_dim * obs_horizon + obs_dim * obs_horizon,
        )
        nets["proto_pred_net"] = proto_pred_net

        if upsample_proto:
            nets["upsample_proto_net"] = hydra.utils.instantiate(cfg.upsample_proto_net)

    # Load stage2 checkpoint.
    print(f"[XSkillRealAgent] Loading stage2 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "nets" in ckpt:
        state_dict = ckpt["nets"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = nets.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[XSkillRealAgent] Missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"[XSkillRealAgent] Unexpected keys ({len(unexpected)}): {unexpected[:10]}")

    nets.to(device)
    nets.eval()

    # Noise scheduler.
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)

    # Stage1 XSkill model.
    xskill_model = _load_xskill_stage1_model(cfg_dict, device)

    # Human prototype preprocessing.
    proto_pipeline = nn.Sequential(
        Tr.CenterCrop((112, 112)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ).to(device)

    # Real RGB cache settings.
    # If not provided, default to original XSkill eval logic:
    # sample 4 frames from recent 30 frames.
    # Important:
    #   snap_frames controls HUMAN prototype sampling.
    #   real_num_frames controls ROBOT RGB history fed into policy.
    #
    # They are not the same thing.
    #
    # If your training config obs_horizon is not 4 but human proto is sampled to 4,
    # that means:
    #   snap_frames = 4
    #   real_num_frames should still follow obs_horizon unless your stage2 policy
    #   was explicitly trained/evaluated with a different robot visual horizon.
    real_num_frames = int(cfg_dict.get("real_num_frames", obs_horizon))
    real_rgb_window = int(cfg_dict.get("real_rgb_window", 30))

    if real_num_frames != obs_horizon:
        print(
            "[XSkillRealAgent] WARNING: real_num_frames != obs_horizon\n"
            f"  real_num_frames={real_num_frames}\n"
            f"  obs_horizon={obs_horizon}\n"
            "This is only valid if your stage2 noise_pred_net/proto_pred_net was "
            "trained with this robot visual horizon. Human proto snap_frames is "
            "separate and does NOT require real_num_frames=4."
        )

    agent_cfg = {
        "obs_horizon": obs_horizon,
        "pred_horizon": int(cfg_dict.get("pred_horizon", getattr(args, "pred_horizon", 16))),
        "action_dim": action_dim,
        "obs_dim": obs_dim,
        "vision_feature_dim": vision_feature_dim,
        "use_proto": use_proto,
        "proto_horizon": proto_horizon,
        "proto_dim": proto_dim,
        "upsample_proto": upsample_proto,
        "snap_frames": int(
            cfg_dict.get(
                "snap_frames",
                cfg_dict.get("proto_snap_frames", 4)
            )
        ),
        "num_diffusion_iters": int(cfg_dict.get("num_diffusion_iters", 60)),
        "real_num_frames": real_num_frames,
        "real_rgb_window": real_rgb_window,
    }

    agent = XSkillRealAgent(
        nets=nets,
        noise_scheduler=noise_scheduler,
        xskill_model=xskill_model,
        proto_pipeline=proto_pipeline,
        cfg=agent_cfg,
        device=device,
    )
    agent.to(device)
    agent.eval()

    human_num_frames = int(
        cfg_dict.get(
            "human_num_frames",
            cfg_dict.get("task_num_frames", cfg_dict.get("num_video_frames", 30))
        )
    )
    
    model_config = {
        "pred_horizon": agent.pred_horizon,
        "obs_horizon": agent.obs_horizon,
        "cameras": list(cfg_dict.get("cameras", getattr(args, "cameras", ["zed2i"]))),
        "include_depth": bool(cfg_dict.get("include_depth", getattr(args, "include_depth", False))),
        "task_num_frames": human_num_frames,
        "num_video_frames": human_num_frames,
        "image_size": list(cfg_dict.get("image_size", getattr(args, "image_size", [224, 224]))),
        "state_type": str(cfg_dict.get("state_type", getattr(args, "state_type", "qpos"))),
        "single_arm": bool(cfg_dict.get("single_arm", getattr(args, "single_arm", False))),
    }

    print(
        "[XSkillRealAgent] loaded successfully: "
        f"obs_horizon={agent.obs_horizon}, "
        f"real_num_frames={agent.real_num_frames}, "
        f"rgb_window={agent.real_rgb_window}, "
        f"pred_horizon={agent.pred_horizon}, "
        f"action_dim={agent.action_dim}, "
        f"use_proto={agent.use_proto}"
    )

    return agent, model_config
