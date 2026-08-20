"""
Evaluation Script for VQ-BeT with LeRobot Paired Dataset  (+ DTW Trajectory Metrics)
"""

import gc
import json
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

from typing import Dict, Any, List, Optional
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from examples.baselines.lerobot_dataset.evaluate_processor import (
    HumanVideoSimEvaluateProcessor,
    HumanVideoSimEvaluateProcessorConfig,
)
from examples.baselines.vqbet.vqbet.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils

from examples.baselines.lerobot_dataset.trajectory_metrics import (
    TrajectoryMetrics,
    GTTrajectoryProvider,
    EpisodeActionBuffer,
)

L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)

_L_ENV_VARS = {
    "L1": "MANI_SKILL_L1",
    "L2": "MANI_SKILL_L2",
    "L3": "MANI_SKILL_L3",
}


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VQ-BeT Batch Evaluation (+ DTW)")

    # Evaluation configuration
    parser.add_argument("--eval-config", type=str, required=True,
                        help="Path to eval configuration file (TXT, one env ID per line)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")

    # Data configs
    parser.add_argument("--sim-config", type=str,
                        default="examples/baselines/lerobot_dataset/config/sim_config.json")
    parser.add_argument("--human-config", type=str,
                        default="examples/baselines/lerobot_dataset/config/human_config.json")
    parser.add_argument("--task-mapping", type=str,
                        default="examples/baselines/lerobot_dataset/task_mapping.json")
    parser.add_argument("--human-task-desc", type=str,
                        default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    parser.add_argument("--sim-task-desc", type=str,
                        default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")

    # Data paths
    parser.add_argument("--human-root", type=str, default="demos")
    parser.add_argument("--sim-root",   type=str, default="demos")

    # Evaluation settings
    parser.add_argument("--num-episodes",      type=int, default=10)
    parser.add_argument("--num-envs",          type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=500)

    # Environment settings
    parser.add_argument("--sim-backend",  type=str, default="physx_cpu")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--obs-mode",     type=str, default="rgb")
    parser.add_argument("--shader",       type=str, default="rt-fast")

    # Model architecture parameters (can override checkpoint values)
    parser.add_argument("--action-dim",      type=int, default=None)
    parser.add_argument("--state-dim",       type=int, default=None)
    parser.add_argument("--obs-horizon",     type=int, default=None)
    parser.add_argument("--obs-latent-dim",  type=int, default=None)
    parser.add_argument("--act-horizon",     type=int, default=None)
    parser.add_argument("--pred-horizon",    type=int, default=None)

    # Window sizes (KEY for VQ-BeT)
    parser.add_argument("--obs-window-size", type=int, default=None)
    parser.add_argument("--act-window-size", type=int, default=None)

    # VQ-VAE parameters
    parser.add_argument("--vqvae-n-latent-dims", type=int, default=None)
    parser.add_argument("--vqvae-n-embed",        type=int, default=None)
    parser.add_argument("--vqvae-groups",          type=int, default=None)

    # GPT parameters
    parser.add_argument("--gpt-n-layer", type=int, default=None)
    parser.add_argument("--gpt-n-head",  type=int, default=None)
    parser.add_argument("--gpt-n-embd",  type=int, default=None)

    # Video encoder settings
    parser.add_argument("--video-encoder-type",     type=str, default=None,
                        choices=["base", "seq", "task", "wm"])
    parser.add_argument("--video-latent-dim",        type=int,  default=None)
    parser.add_argument("--num-video-frames",         type=int,  default=None)
    parser.add_argument("--use-temporal-encoder",     action="store_true", default=None)
    parser.add_argument("--num-transformer-layers",   type=int,  default=None)
    parser.add_argument("--transformer-heads",        type=int,  default=None)
    parser.add_argument("--use-language-prompt",      action="store_true", default=None)

    # Other settings
    parser.add_argument("--state-type",    type=str,  default=None)
    parser.add_argument("--single-arm",    action="store_true", default=None)
    parser.add_argument("--include-depth", action="store_true", default=None)
    parser.add_argument("--cameras",       type=str, nargs="+", default=None)
    parser.add_argument("--image-size",    type=int, nargs=2,   default=None)

    # Evaluation settings
    parser.add_argument("--temporal-agg",       action="store_true", default=False)
    parser.add_argument("--light-temporal-agg", action="store_true", default=True,
                        help="Lightweight sliding-window temporal aggregation. "
                             "Queries every --tagg-window steps and averages the last "
                             "--tagg-window predictions. Faster than --temporal-agg "
                             "while still smoothing chunk-boundary jitter.")
    parser.add_argument("--tagg-window",         type=int, default=4,
                        help="Window size K for --light-temporal-agg. "
                             "Smaller = smoother but slower. "
                             "Typical: 4 (act_window_size=10 → 2.5× faster than temporal-agg).")

    # Input mode
    parser.add_argument("--input-mode", type=str, default="video_only",
                        choices=["video_only", "language_only", "video_and_language"],
                        help="Task conditioning mode. VQ-BeT currently supports video_only.")

    # Task encoder type
    parser.add_argument("--task-encoder-type", type=str, default="frozen_backbone",
                        choices=["frozen_backbone"],
                        help="'frozen_backbone' = FrozenVideoBackbone (DINOv2/CLIP/VideoMAE)")

    # FrozenVideoBackbone parameters
    parser.add_argument("--frozen-backbone-type",           type=str,  default="dinov2_vitl14",
                        help="Backbone: dinov2_vitl14 | dinov2_vitb14 | clip_vitl14 | "
                             "clip_vitb16 | siglip2_so400m | videomae_large | videomae_base")
    parser.add_argument("--frozen-backbone-adapter-layers", type=int,  default=1)
    parser.add_argument("--frozen-backbone-adapter-ln",     action="store_true", default=True)
    parser.add_argument("--frozen-backbone-seq-patches",    type=int,  default=32)
    parser.add_argument("--frozen-backbone-num-frames",     type=int,  default=4)

    # Output
    parser.add_argument("--output-dir",   type=str, required=True)
    parser.add_argument("--save-videos",  action="store_true")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    # ── NEW: DTW options ───────────────────────────────────────────────────
    parser.add_argument(
        "--compute-dtw",
        action="store_true",
        default=True,
        help="Compute DTW trajectory similarity metrics during evaluation.",
    )
    parser.add_argument(
        "--dtw-band-ratio",
        type=float,
        default=0.15,
        help=(
            "Sakoe-Chiba band radius as a fraction of max(T_pred, T_gt). "
            "Controls how much temporal warping is allowed (default 0.15 = 15%%)."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Config / level utilities  (unchanged)
# =============================================================================

def load_eval_config(config_path: str) -> Dict[str, Any]:
    """Load eval configuration from TXT file (one env ID per line)."""
    config_path = Path(config_path)
    if config_path.suffix != '.txt':
        raise ValueError(f"Only .txt config files are supported. Got: {config_path.suffix}")
    with open(config_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return {
        "eval_name": config_path.stem,
        "description": f"Evaluation from {config_path.name}",
        "environments": [{"env_id": env_id} for env_id in lines],
    }


def extract_base_env_name(env_id: str) -> str:
    if env_id.startswith("L") and "_" in env_id:
        parts = env_id.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            level, base = parts[0], parts[1]
            if level == "L3":
                if "-v" in base:
                    name_part, version_part = base.rsplit("-v", 1)
                    return f"{name_part}L3-v{version_part}"
                return f"{base}L3"
            return base
    return env_id


def extract_level(env_id: str) -> str:
    if env_id.startswith("L") and "_" in env_id:
        parts = env_id.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            return parts[0]
    return "L0"


def set_l_level(level: str):
    for env_var in _L_ENV_VARS.values():
        os.environ.pop(env_var, None)
    if level in _L_ENV_VARS:
        os.environ[_L_ENV_VARS[level]] = "1"
    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)
    if level == "L1":
        L0_L3_utils.set_l1_enabled(True)
    elif level == "L2":
        L0_L3_utils.set_l2_enabled(True)
    elif level == "L3":
        L0_L3_utils.set_l3_enabled(True)


def clear_l_level():
    for env_var in _L_ENV_VARS.values():
        os.environ.pop(env_var, None)
    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)


def make_eval_envs_with_level(
    base_env_name: str,
    level: str,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: str,
    wrappers: list,
):
    """Set L-level then create eval envs."""
    set_l_level(level)
    clean_env_kwargs = {k: v for k, v in env_kwargs.items() if k != "l_level"}
    envs = make_eval_envs(
        base_env_name, num_envs, sim_backend, clean_env_kwargs, other_kwargs,
        video_dir=video_dir,
        wrappers=wrappers,
        l_level=level,
    )
    return envs


# =============================================================================
# Checkpoint loading  (unchanged)
# =============================================================================

def load_agent_from_checkpoint(checkpoint_path: str, args, device):
    """Load VQ-BeT agent from checkpoint (frozen-backbone variant)."""
    from examples.baselines.vqbet.train_vqbet_imitator import VQBeTAgent, TrainingArgs

    print(f"\n📦 Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    saved = checkpoint.get('args', {})
    if saved:
        print("✅ Loaded model config from checkpoint['args']")
    else:
        print("⚠️  No args found in checkpoint, using defaults")

    def _get(key, default):
        return saved.get(key, default)

    training_args = TrainingArgs(
        action_dim            = _get('action_dim',          16),
        state_dim             = _get('state_dim',           18),
        obs_horizon           = _get('obs_horizon',          1),
        obs_window_size       = _get('obs_window_size',     10),
        act_window_size       = _get('act_window_size',     10),
        obs_latent_dim        = _get('obs_latent_dim',     256),
        vqvae_n_latent_dims   = _get('vqvae_n_latent_dims', 512),
        vqvae_n_embed         = _get('vqvae_n_embed',        32),
        vqvae_groups          = _get('vqvae_groups',          2),
        gpt_n_layer           = _get('gpt_n_layer',          24),
        gpt_n_head            = _get('gpt_n_head',           16),
        gpt_n_embd            = _get('gpt_n_embd',         1280),
        gpt_block_size        = _get('gpt_block_size',      100),
        gpt_dropout           = _get('gpt_dropout',         0.1),
        offset_loss_multiplier    = _get('offset_loss_multiplier',    1e3),
        secondary_code_multiplier = _get('secondary_code_multiplier', 0.5),
        focal_loss_gamma          = _get('focal_loss_gamma',          2.0),
        include_depth         = _get('include_depth',      False),
        cameras               = _get('cameras',       ['zed2i']),
        image_size            = tuple(_get('image_size', (224, 224))),
        task_encoder_type              = _get('task_encoder_type',              getattr(args, 'task_encoder_type', 'frozen_backbone')),
        frozen_backbone_type           = _get('frozen_backbone_type',           getattr(args, 'frozen_backbone_type', 'dinov2_vitl14')),
        frozen_backbone_adapter_layers = _get('frozen_backbone_adapter_layers', getattr(args, 'frozen_backbone_adapter_layers', 1)),
        frozen_backbone_seq_patches    = _get('frozen_backbone_seq_patches',    getattr(args, 'frozen_backbone_seq_patches', 32)),
        frozen_backbone_num_frames     = _get('frozen_backbone_num_frames',     getattr(args, 'frozen_backbone_num_frames', 4)),
        input_mode               = _get('input_mode',           'video_only'),
        video_conditioning_mode  = _get('video_conditioning_mode', 'concat'),
        state_type               = _get('state_type',               'qpos'),
    )

    print(f"\n📋 TrainingArgs (from checkpoint):")
    for key in ('gpt_n_layer', 'gpt_n_head', 'gpt_n_embd',
                'task_latent_dim', 'task_num_frames', 'num_tasks',
                'obs_window_size', 'act_window_size',
                'vqvae_n_latent_dims', 'vqvae_n_embed', 'vqvae_groups'):
        print(f"   {key}: {getattr(training_args, key)}")

    print(f"\n🤖 Creating VQBeTAgent (frozen_backbone)...")
    agent = VQBeTAgent(training_args, device)

    raw_sd   = checkpoint.get('agent_state_dict', checkpoint.get('agent', {}))
    fixed_sd = {}
    for k, v in raw_sd.items():
        new_k = k.replace(".video_encoder.original.", ".video_encoder.") \
          .replace(".lang_encoder.original.",  ".lang_encoder.") \
          .replace("task_encoder.backbone.",    "task_encoder.")
        fixed_sd[new_k] = v

    missing, unexpected = agent.load_state_dict(fixed_sd, strict=False)
    if missing:
        print(f"⚠️  Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    if 'vqvae_state_dict' in checkpoint:
        vqvae_sd = checkpoint['vqvae_state_dict']
        if 'encoder' in vqvae_sd:
            agent.vqvae.load_vqvae_state_dict(vqvae_sd)
        else:
            agent.vqvae.load_state_dict(vqvae_sd, strict=False)
            agent.vqvae.vq_layer.eval()
            print("✓ VQ-VAE loaded (standard state_dict format)")

    agent = agent.to(device)
    agent.eval()

    print(f"\n✅ Agent loaded successfully")
    print(f"   Epoch: {checkpoint.get('epoch', checkpoint.get('iteration', 'unknown'))}")
    print(f"   Total parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.2f}M")

    model_config = {
        'cameras':          training_args.cameras,
        'include_depth':    training_args.include_depth,
        'num_video_frames': training_args.frozen_backbone_num_frames,
        'image_size':       training_args.image_size,
        'state_type':       training_args.state_type,
        'single_arm':       saved.get('single_arm', False),
        'use_language_prompt': (training_args.input_mode == 'language_only'),
        'act_window_size':  training_args.act_window_size,
        'obs_horizon':      training_args.obs_horizon,
    }
    return agent, model_config


# =============================================================================
# Core evaluation function  — VQ-BeT specific + DTW
# =============================================================================

def evaluate_vqbet(
    n: int,
    agent,
    eval_envs,
    eval_kwargs,
    evaluate_processor,
    progress_bar: bool = True,
    # ── NEW: DTW arguments ────────────────────────────────────────────────
    dtw_provider: Optional[GTTrajectoryProvider] = None,
    traj_metrics: Optional[TrajectoryMetrics] = None,
    dtw_band_ratio: float = 0.15,
):
    """
    Evaluate VQ-BeT agent with temporal aggregation.

    DTW trajectory metrics
    ----------------------
    When *dtw_provider* is given, each episode's executed action trajectory
    is compared against a randomly sampled GT demo via normalized multi-
    dimensional DTW.  See trajectory_metrics.py for the full metric set.

    Why DTW instead of MSE?  VQ-BeT may execute the correct motion at a
    different speed or with a slight phase offset.  DTW finds the optimal
    non-linear temporal alignment and therefore measures *shape* similarity
    (are the joint trajectories qualitatively the same?) rather than
    point-to-point synchrony (are the joint angles identical at t=k?).
    """
    import gymnasium
    from mani_skill.utils import common

    env_id         = eval_kwargs.get('env_id')
    delta_control  = eval_kwargs.get('delta_control')
    pred_horizon   = eval_kwargs.get('pred_horizon')
    temporal_agg       = eval_kwargs.get('temporal_agg')
    light_temporal_agg = eval_kwargs.get('light_temporal_agg', True)
    tagg_window        = eval_kwargs.get('tagg_window', 4)
    max_timesteps      = eval_kwargs.get('max_timesteps')
    device             = eval_kwargs.get('device')
    sim_backend        = eval_kwargs.get('sim_backend')

    # light_temporal_agg takes priority over temporal_agg
    if light_temporal_agg:
        temporal_agg = False

    use_visual_obs = isinstance(eval_envs.single_observation_space.sample(), dict)

    if isinstance(eval_envs.single_observation_space, gymnasium.spaces.Box):
        action_dim = eval_envs.action_space.shape[-1]
    else:
        action_dim = (eval_envs.action_space["panda_wristcam-0"].shape[-1] +
                      eval_envs.action_space["panda_wristcam-1"].shape[-1])

    num_envs = eval_envs.num_envs

    if light_temporal_agg:
        # ── Lightweight sliding-window temporal aggregation ────────────────
        query_frequency = tagg_window
        action_window   = deque(maxlen=tagg_window)
    elif temporal_agg:
        query_frequency  = 1
        all_time_actions = torch.zeros(
            [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim],
            device=device,
        )
    else:
        query_frequency  = pred_horizon
        actions_to_take  = torch.zeros([num_envs, pred_horizon, action_dim], device=device)

    agent.eval()

    # Get prompt for conditioning
    if agent.args.input_mode == "language_only":
        val_prompts = evaluate_processor.get_language(env_id, num_envs)
    else:
        val_videos = evaluate_processor.get_video(env_id, num_envs).to(device)

    # ── NEW: initialise DTW helpers ───────────────────────────────────────
    compute_dtw = (dtw_provider is not None)
    if compute_dtw:
        if traj_metrics is None:
            traj_metrics = TrajectoryMetrics(band_ratio=dtw_band_ratio)
        # Pre-load a GT trajectory for this task
        gt_traj = dtw_provider.sample_gt_trajectory(env_id, seed=None, normalize=False)
        if gt_traj is None:
            print(f"⚠️  No GT trajectory found for {env_id}; TSS will be skipped.")
            compute_dtw = False

        action_buf = EpisodeActionBuffer(num_envs=num_envs)
        # Track first success step per env (-1 = not yet succeeded)
        success_steps = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info    = eval_envs.reset()
        ts, eps_count = 0, 0

        pbar = tqdm(
            total=n,
            desc=f"Evaluating{'  [+DTW]' if compute_dtw else ''}",
            disable=not progress_bar,
            unit="episode",
        )

        while eps_count < n:
            if use_visual_obs:
                obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
                if not delta_control:
                    obs['state'], obs['rgb'] = evaluate_processor.normalize_state_rgb(
                        obs['state'], obs['rgb'], env_id
                    )
            else:
                raise RuntimeError("Non-visual obs are not supported.")

            # Prepare for evaluation with prompt (ONCE per episode)
            if ts == 0:
                if agent.args.input_mode == "language_only":
                    agent.prepare_for_eval(val_prompts, robot_obs=obs)
                else:
                    agent.prepare_for_eval(val_videos, robot_obs=obs)

            # Query policy
            if ts % query_frequency == 0:
                action_seq = agent.get_action(obs)
                if light_temporal_agg:
                    action_window.append(action_seq)

            # Temporal ensembling or direct action selection
            if light_temporal_agg:
                step_in_chunk = ts % query_frequency
                acts_list, ages = [], []
                for age, past_pred in enumerate(reversed(action_window)):
                    idx = step_in_chunk + age * query_frequency
                    if idx < pred_horizon:
                        acts_list.append(past_pred[:, idx])
                        ages.append(age)
                if acts_list:
                    w = torch.exp(-0.1 * torch.tensor(ages, dtype=torch.float32, device=device))
                    w = w / w.sum()
                    stacked = torch.stack(acts_list, dim=1)  # (num_envs, valid, action_dim)
                    raw_action = (stacked * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
                else:
                    raw_action = action_window[-1][:, step_in_chunk]
            elif temporal_agg:
                all_time_actions[:, ts, ts:ts + pred_horizon] = action_seq
                actions_for_curr_step = all_time_actions[:, :, ts]
                actions_populated = torch.zeros(max_timesteps, dtype=torch.bool, device=device)
                actions_populated[max(0, ts + 1 - pred_horizon):ts + 1] = True
                actions_for_curr_step = actions_for_curr_step[:, actions_populated]
                k_exp       = 0.01
                exp_weights = torch.exp(
                    -k_exp * torch.arange(len(actions_for_curr_step[0]), device=device)
                )
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.tile(exp_weights, (num_envs, 1))
                exp_weights = torch.unsqueeze(exp_weights, -1)
                raw_action  = (actions_for_curr_step * exp_weights).sum(dim=1)
            else:
                if ts % query_frequency == 0:
                    actions_to_take = action_seq
                raw_action = actions_to_take[:, ts % query_frequency]

            # Denormalize action
            _action = (
                evaluate_processor.denormalize_action(raw_action, env_id)
                if not delta_control else raw_action
            )

            if sim_backend == "physx_cpu":
                _action = _action.cpu().numpy()

            # ── NEW: record executed action ────────────────────────────────
            if compute_dtw:
                action_buf.append(_action)

            # Handle dual-arm action format
            action = {
                "panda_wristcam-0": _action[:, :8],
                "panda_wristcam-1": _action[:, 8:16],
            }

            obs, rew, terminated, truncated, info = eval_envs.step(action)
            ts += 1

            # Track first success per env for TSS (valid while not truncated)
            if compute_dtw and not truncated.any():
                _cur_s = info.get("success")
                if _cur_s is not None:
                    _cur_s_t = torch.as_tensor(_cur_s, dtype=torch.bool, device=device)
                    _new_s = (success_steps == -1) & _cur_s_t
                    success_steps[_new_s] = ts

            if truncated.any():
                assert truncated.all() == truncated.any()

                # ── Standard episode metrics ───────────────────────────────
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)

                # ── TSS: split by success / fail ─────────────────────────────────────
                if compute_dtw:
                    _fi = info.get("final_info", {})
                    _ls = (_fi.get("success") if isinstance(_fi, dict) else None)
                    if _ls is not None:
                        _ls_t = torch.as_tensor(_ls, dtype=torch.bool, device=device)
                        success_steps[(success_steps == -1) & _ls_t] = ts
                    pred_trajs = action_buf.get_and_reset()
                    ss_np      = success_steps.cpu().numpy()
                    success_steps[:] = -1
                    T_gt = len(gt_traj)   # GT length for fair fail comparison
                    for env_i, pred_traj in enumerate(pred_trajs):
                        ss = int(ss_np[env_i])
                        if ss >= 2:
                            # Successful: trim to first-success step
                            try:
                                m = traj_metrics.compute(pred_traj[:ss], gt_traj)
                                eval_metrics["tss_success"].append(np.array(m["tss"],  dtype=np.float32))
                                eval_metrics["ndtw_success"].append(np.array(m["ndtw"], dtype=np.float32))
                            except Exception as exc:
                                print(f"⚠️  TSS(success) env {env_i}: {exc}")
                        else:
                            # Failed: trim to GT length for fair shape comparison
                            pred_trimmed = pred_traj[:T_gt]
                            if len(pred_trimmed) >= 2:
                                try:
                                    m = traj_metrics.compute(pred_trimmed, gt_traj)
                                    eval_metrics["tss_fail"].append(np.array(m["tss"],  dtype=np.float32))
                                    eval_metrics["ndtw_fail"].append(np.array(m["ndtw"], dtype=np.float32))
                                except Exception as exc:
                                    print(f"⚠️  TSS(fail) env {env_i}: {exc}")
                pbar.update(num_envs)
                eps_count += num_envs
                ts = 0

                # Reset temporal aggregation buffer
                if light_temporal_agg:
                    action_window.clear()
                elif temporal_agg:
                    all_time_actions = torch.zeros(
                        [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim],
                        device=device,
                    )

                obs, info = eval_envs.reset()

                # ── NEW: re-sample GT each episode for variety ─────────────
                if compute_dtw:
                    new_gt = dtw_provider.sample_gt_trajectory(env_id, seed=None)
                    if new_gt is not None:
                        gt_traj = new_gt

        pbar.close()

    agent.train()

    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])

    return eval_metrics


# =============================================================================
# Per-environment evaluation entry point
# =============================================================================

def evaluate_single_env(
    agent,
    env_id: str,
    args,
    evaluate_processor,
    output_dir: Path,
    model_config: Dict,
    # ── NEW ──
    dtw_provider: Optional[GTTrajectoryProvider] = None,
    traj_metrics: Optional[TrajectoryMetrics] = None,
):
    """Evaluate on a single environment."""
    base_env_name = extract_base_env_name(env_id)
    level         = extract_level(env_id)

    print(f"\n{'='*60}")
    print(f"🎯 Evaluating: {env_id}")
    print(f"   Base env: {base_env_name}, Level: {level}")
    print(f"{'='*60}")

    obs_mode = "rgbd" if model_config['include_depth'] else "rgb"

    env_kwargs   = dict(
        control_mode=args.control_mode,
        reward_mode="dense",
        obs_mode=obs_mode,
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
        sensor_configs=dict(shader_pack="rt-fast"), 
        human_render_camera_configs=dict(shader_pack="rt-fast"),
    )
    other_kwargs = dict(obs_horizon=model_config['obs_horizon'])

    # NOTE: eval_envs is initialized to None so that the finally block can
    # safely check whether env creation succeeded before calling .close().
    # Previously eval_envs was created OUTSIDE the try block, which meant
    # that if make_eval_envs_with_level raised (e.g. SAPIEN Vulkan resource
    # exhaustion on the 5th consecutive env), the finally cleanup was never
    # reached and the worker process died immediately.
    eval_envs = None
    try:
        eval_envs = make_eval_envs_with_level(
            base_env_name=base_env_name,
            level=level,
            num_envs=args.num_envs,
            sim_backend=args.sim_backend,
            env_kwargs=env_kwargs,
            other_kwargs=other_kwargs,
            video_dir=str(output_dir / "videos" / env_id),
            wrappers=[FlattenRGBDObservationWrapper],
        )

        eval_kwargs = dict(
            env_id=env_id,
            delta_control=("delta" in args.control_mode),
            pred_horizon=model_config['act_window_size'],
            temporal_agg=args.temporal_agg,
            light_temporal_agg=args.light_temporal_agg,
            tagg_window=args.tagg_window,
            max_timesteps=args.max_episode_steps,
            device=args.device,
            sim_backend=args.sim_backend,
        )

        eval_metrics = evaluate_vqbet(
            n=args.num_episodes,
            agent=agent,
            eval_envs=eval_envs,
            eval_kwargs=eval_kwargs,
            progress_bar=True,
            evaluate_processor=evaluate_processor,
            # ── NEW ──
            dtw_provider=dtw_provider,
            traj_metrics=traj_metrics,
            dtw_band_ratio=args.dtw_band_ratio,
        )

        results = {
            "env_id":        env_id,
            "base_env_name": base_env_name,
            "level":         level,
            "num_episodes":  args.num_episodes,
            "timestamp":     datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status":        "success",
        }

        for k, v in eval_metrics.items():
            results[f"{k}_mean"] = float(v.mean())
            results[f"{k}_std"]  = float(v.std())
            results[f"{k}_min"]  = float(v.min())
            results[f"{k}_max"]  = float(v.max())

        print(f"\n📊 Results:")
        for k in ["success_once", "success_at_end", "return"]:
            if f"{k}_mean" in results:
                print(f"   {k}: {results[f'{k}_mean']:.4f} ± {results[f'{k}_std']:.4f}")

        # ── NEW: DTW console summary ───────────────────────────────────────
        # Ensure TSS fields always present; compute wTSS
        if args.compute_dtw:
            for _k in ["tss_success", "ndtw_success", "tss_fail", "ndtw_fail"]:
                if f"{_k}_mean" not in results:
                    results[f"{_k}_mean"] = None
                    results[f"{_k}_std"]  = None
            _sr  = results.get("success_once_mean") or 0.0
            _tss = results.get("tss_success_mean")
            results["wtss_mean"] = round(_sr * _tss, 6) if _tss is not None else 0.0

        if args.compute_dtw:
            print("\n📏 Trajectory Similarity:")
            print(f"   wTSS        : {results['wtss_mean']:.4f}  (= SR × TSS_success)")
            if results.get("tss_success_mean") is not None:
                print(f"   TSS_success : {results['tss_success_mean']:.4f} ± {results['tss_success_std']:.4f}  (higher=better)")
                print(f"   nDTW_success: {results['ndtw_success_mean']:.4f} ± {results['ndtw_success_std']:.4f}")
            else:
                print("   TSS_success : N/A  (no successful episodes)")
            if results.get("tss_fail_mean") is not None:
                print(f"   TSS_fail    : {results['tss_fail_mean']:.4f} ± {results['tss_fail_std']:.4f}")
                print(f"   nDTW_fail   : {results['ndtw_fail_mean']:.4f} ± {results['ndtw_fail_std']:.4f}")
        return results

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "env_id":        env_id,
            "base_env_name": base_env_name,
            "level":         level,
            "timestamp":     datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status":        "error",
            "error":         str(e),
        }

    finally:
        # Close and explicitly delete the env to trigger SAPIEN resource
        # teardown.  Without del + gc.collect(), Vulkan/CUDA render contexts
        # accumulate across sequential env creations, exhausting GPU render
        # resources by the 5th env when a large model (VQBeT ~10 GB) is
        # resident.  The sleep gives the driver time to fully release handles
        # before the next RenderSystem is created.
        if eval_envs is not None:
            try:
                eval_envs.close()
            except Exception:
                pass
            del eval_envs
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)
        clear_l_level()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("🧪 VQ-BeT BATCH EVALUATION")
    if args.compute_dtw:
        print("   📏 TSS trajectory metrics ENABLED (band_ratio=%.2f)" % args.dtw_band_ratio)
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load eval configuration
    print(f"\n📋 Loading eval configuration: {args.eval_config}")
    eval_config  = load_eval_config(args.eval_config)
    eval_name    = eval_config.get("eval_name", "unnamed_eval")
    environments = eval_config.get("environments", [])

    print(f"\n✅ Eval Configuration:")
    print(f"   Name: {eval_name}")
    print(f"   Description: {eval_config.get('description', 'N/A')}")
    print(f"   Environments: {len(environments)}")

    # Load agent
    device = torch.device(args.device)
    agent, model_config = load_agent_from_checkpoint(args.checkpoint, args, device)

    # Setup evaluation processor
    print("\n📹 Setting up evaluation processor...")
    evaluate_processor_config = HumanVideoSimEvaluateProcessorConfig(
        human_root=args.human_root,
        human_split="train",
        human_dataset_file=args.human_config,
        human_task_description_file=args.human_task_desc,
        human_cameras=model_config['cameras'],
        human_include_depth=model_config['include_depth'],
        human_num_frames=model_config['num_video_frames'],
        human_image_size=model_config['image_size'],
        human_video_backend="torchcodec",
        human_fps=30,
        vla=model_config.get('use_language_prompt', False),
        sim_root=args.sim_root,
        sim_split="train",
        sim_dataset_file=args.sim_config,
        sim_task_description_file=args.sim_task_desc,
        sim_state_type=model_config['state_type'],
        sim_single_arm=model_config['single_arm'],
        normalization_method="bounds_q99",
        task_mapping_file=args.task_mapping,
    )
    evaluate_processor = HumanVideoSimEvaluateProcessor(evaluate_processor_config)
    print("✅ Evaluate processor ready")

    # ── NEW: build DTW provider ────────────────────────────────────────────
    dtw_provider = None
    traj_metrics = None
    if args.compute_dtw:
        print("\n📏 Building GT trajectory provider for TSS...")
        _, action_key = evaluate_processor._resolve_state_action_keys()
        try:
            dtw_provider = GTTrajectoryProvider(
                sim_dataset_file=args.sim_config,
                sim_root=args.sim_root,
                action_key=action_key,
                normalizer=evaluate_processor.normalizer,
                normalization_method="bounds_q99",
            )
            traj_metrics = TrajectoryMetrics(band_ratio=args.dtw_band_ratio)
            task_count = len(dtw_provider._episodes)
            print(f"✅ GT trajectories loaded for {task_count} task(s).")
        except Exception as exc:
            print(f"⚠️  Could not build GTTrajectoryProvider: {exc}")
            print("   DTW metrics will be skipped.")
            args.compute_dtw = False

    # Run evaluations
    print("\n" + "=" * 80)
    print("🚀 STARTING BATCH EVALUATIONS")
    print("=" * 80)

    all_results = []
    progress_bar = tqdm(environments, desc="Evaluating")
    for env_config in progress_bar:
        env_id = env_config["env_id"]
        progress_bar.set_description(f"Evaluating {env_id}")

        result = evaluate_single_env(
            agent=agent,
            env_id=env_id,
            args=args,
            evaluate_processor=evaluate_processor,
            output_dir=output_dir,
            model_config=model_config,
            # ── NEW ──
            dtw_provider=dtw_provider,
            traj_metrics=traj_metrics,
        )
        all_results.append(result)

    # Save results
    print("\n" + "=" * 80)
    print("💾 SAVING RESULTS")
    print("=" * 80)

    results_json = output_dir / f"{eval_name}_{args.input_mode}_{timestamp}.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Saved JSON: {results_json}")

    df = pd.DataFrame(all_results)
    results_csv = output_dir / f"{eval_name}_{timestamp}.csv"
    df.to_csv(results_csv, index=False)
    print(f"✅ Saved CSV : {results_csv}")

    # Print summary
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nEval: {eval_name}")
    print(f"Environments tested: {len(environments)}")
    print(f"Episodes per env: {args.num_episodes}\n")

    has_dtw = args.compute_dtw and "wtss_mean" in df.columns

    if "success_once_mean" in df.columns:
        # ── Per-environment table ──────────────────────────────────────────
        header = f"{'Environment':<50} {'Level':<8} {'Success Once':<15} {'Success At End':<15}"
        if has_dtw:
            header += f" {'wTSS':<10} {'TSS_fail':<10} {'nDTW_fail':<10}"
        print(header)
        print("-" * (90 + (32 if has_dtw else 0)))

        for _, row in df.iterrows():
            env_id = row['env_id']
            level  = row.get('level', 'L0')

            if row.get('status') == 'error':
                print(f"{env_id:<50} {level:<8} {'ERROR':<15} {'ERROR':<15}")
            else:
                s1  = row.get('success_once_mean', 0.0)
                s2  = row.get('success_at_end_mean', 0.0)
                line = f"{env_id:<50} {level:<8} {s1:<15.4f} {s2:<15.4f}"
                if has_dtw:
                    wtss = row.get('wtss_mean',      float('nan'))
                    tssf = row.get('tss_fail_mean',  float('nan'))
                    ndtw = row.get('ndtw_fail_mean', float('nan'))
                    line += f" {wtss:<10.4f} {tssf:<10.4f} {ndtw:<10.4f}"
                print(line)

        print("\n" + "-" * (90 + (32 if has_dtw else 0)))
        # Overall averages
        valid_df = df[df['status'] == 'success']
        if len(valid_df) > 0:
            avg_s1 = valid_df['success_once_mean'].mean()
            avg_s2 = valid_df['success_at_end_mean'].mean()
            line   = f"{'OVERALL AVERAGE':<50} {'ALL':<8} {avg_s1:<15.4f} {avg_s2:<15.4f}"
            if has_dtw and 'wtss_mean' in valid_df.columns:
                line += (
                    f" {valid_df['wtss_mean'].mean():<10.4f}"
                    f" {valid_df['tss_fail_mean'].dropna().mean() if valid_df['tss_fail_mean'].notna().any() else float('nan'):<10.4f}"
                    f" {valid_df['ndtw_fail_mean'].dropna().mean() if valid_df['ndtw_fail_mean'].notna().any() else float('nan'):<10.4f}"
                )
            print(line)

        # ── Per-level breakdown ────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("📊 RESULTS BY L-LEVEL")
        print("=" * 80)

        for level in ['L0', 'L1', 'L2', 'L3']:
            level_df = valid_df[valid_df['level'] == level] if 'level' in valid_df.columns else pd.DataFrame()
            if len(level_df) == 0:
                continue
            print(f"\n{level}:")
            print(f"  Environments : {len(level_df)}")
            print(f"  Success Once : "
                  f"{level_df['success_once_mean'].mean():.4f} ± "
                  f"{level_df['success_once_std'].mean():.4f}")
            print(f"  Success @End : "
                  f"{level_df['success_at_end_mean'].mean():.4f} ± "
                  f"{level_df['success_at_end_std'].mean():.4f}")
            if has_dtw and 'wtss_mean' in level_df.columns:
                print(f"  wTSS         : "
                      f"{level_df['wtss_mean'].mean():.4f}")
                if level_df['tss_fail_mean'].notna().any():
                    print(f"  TSS_fail     : "
                          f"{level_df['tss_fail_mean'].dropna().mean():.4f} ± "
                          f"{level_df['tss_fail_std'].dropna().mean():.4f}")
                    print(f"  nDTW_fail    : "
                          f"{level_df['ndtw_fail_mean'].dropna().mean():.4f}")

    print("\n" + "=" * 80)
    print("✅ VQ-BeT BATCH EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()