"""
Evaluation Script for Diffusion Policy with Task Encoder
=========================================================
- Supports three input modes: video_only, language_only, video_and_language
- Supports the "frozen_backbone" task encoder backend (FrozenVideoBackbone)
- Task encoder called ONCE at episode start
- Temporal aggregation support
- Incremental JSON logging: results saved after each env evaluation
- Auto-skip: environments with sufficient existing results are skipped
- JSON-based skip check (not video-file count)
- Subprocess-safe L-level differentiation
"""

import gc
import json
import time
import os
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import gymnasium
from tqdm import tqdm
import pandas as pd

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import InputMode
from examples.baselines.lerobot_dataset.evaluate_processor import (
    HumanVideoSimEvaluateProcessor,
    HumanVideoSimEvaluateProcessorConfig,
)
from examples.baselines.lerobot_dataset.trajectory_metrics import (
    TrajectoryMetrics,
    GTTrajectoryProvider,
    EpisodeActionBuffer,
)
from examples.baselines.diffusion_policy.diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
from mani_skill.utils import common

L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)


# =============================================================================
# JSON-based skip check
# =============================================================================

def load_existing_results(output_dir: Path, input_mode: str) -> Dict[str, Dict]:
    existing: Dict[str, Dict] = {}
    for json_file in output_dir.glob(f"*_{input_mode}_*.json"):
        try:
            with open(json_file) as f:
                results = json.load(f)
            for r in results:
                env_id = r.get("env_id")
                if env_id and r.get("status") == "success":
                    prev = existing.get(env_id)
                    if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
                        existing[env_id] = r
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return existing


def is_env_already_evaluated(
    env_id: str,
    existing_results: Dict[str, Dict],
    num_episodes: int,
) -> bool:
    prev = existing_results.get(env_id)
    if prev is None:
        return False
    if prev.get("num_episodes", 0) >= num_episodes:
        print(f"⏭️  Skipping {env_id}: already evaluated "
              f"({prev['num_episodes']} episodes, "
              f"success_once={prev.get('success_once_mean', 'N/A')})")
        return True
    print(f"⚠️  Partial eval for {env_id}: previous run had "
          f"{prev.get('num_episodes', '?')}/{num_episodes} episodes — re-evaluating")
    return False


# =============================================================================
# Subprocess-safe L-level differentiation
# =============================================================================

_L_ENV_VARS = {
    "L1": "MANI_SKILL_L1",
    "L2": "MANI_SKILL_L2",
    "L3": "MANI_SKILL_L3",
}


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
# Core evaluation function
# =============================================================================

def evaluate_with_task_encoder(
    n: int,
    agent,
    eval_envs,
    eval_kwargs: Dict,
    evaluate_processor,
    input_mode: InputMode,
    progress_bar: bool = True,
    # ── DTW arguments ────────────────────────────────────────────────────────
    dtw_provider=None,
    traj_metrics=None,
    dtw_band_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Evaluate Diffusion Policy agent with task encoder + optional DTW trajectory metrics.

    - action_seq shape: (num_envs, pred_horizon, action_dim)
    - Correct temporal slice: actions_to_take[:, ts % query_frequency]
    - Env step format: {"panda_wristcam-0": (num_envs, 8), "panda_wristcam-1": (num_envs, 8)}
    """
    env_id             = eval_kwargs.get("env_id")
    delta_control      = eval_kwargs.get("delta_control")
    pred_horizon       = eval_kwargs.get("pred_horizon")
    temporal_agg       = eval_kwargs.get("temporal_agg")
    light_temporal_agg = eval_kwargs.get("light_temporal_agg", False)
    tagg_window        = eval_kwargs.get("tagg_window", 4)
    max_timesteps      = eval_kwargs.get("max_timesteps")
    device             = eval_kwargs.get("device")
    sim_backend        = eval_kwargs.get("sim_backend")

    # light_temporal_agg takes priority over temporal_agg
    if light_temporal_agg:
        temporal_agg = False

    # ── Action dim from env (dual arm: 8 + 8 = 16) ───────────────────────────
    if isinstance(eval_envs.single_observation_space, gymnasium.spaces.Box):
        action_dim = eval_envs.action_space.shape[-1]
    else:
        action_dim = (eval_envs.action_space["panda_wristcam-0"].shape[-1] +
                      eval_envs.action_space["panda_wristcam-1"].shape[-1])

    num_envs = eval_envs.num_envs

    if light_temporal_agg:
        # ── Lightweight sliding-window temporal aggregation ────────────────
        # Query every tagg_window steps; keep last tagg_window predictions
        # and exponentially-weight them. Much faster than full temp agg (1
        # query per step) while still smoothing chunk-boundary transitions.
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

    # ── Prepare task inputs ───────────────────────────────────────────────────
    human_video  = None
    human_desc   = None

    if input_mode in [InputMode.VIDEO_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
        human_video = evaluate_processor.get_video(env_id, num_envs).to(device)

    if input_mode in [InputMode.LANGUAGE_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
        human_desc = [evaluate_processor.get_task_description(env_id)] * num_envs

    # ── DTW helpers ───────────────────────────────────────────────────────────
    compute_dtw = (dtw_provider is not None)
    if compute_dtw:
        if traj_metrics is None:
            traj_metrics = TrajectoryMetrics(
                band_ratio=dtw_band_ratio
            )
        gt_traj = dtw_provider.sample_gt_trajectory(env_id, seed=None, normalize=False)
        if gt_traj is None:
            print(f"⚠️  No GT trajectory for {env_id}; TSS skipped.")
            compute_dtw = False
        action_buf = EpisodeActionBuffer(num_envs=num_envs)
        # Track first success step per env (-1 = not yet succeeded)
        success_steps = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    with torch.no_grad():
        eval_metrics  = defaultdict(list)
        obs, info     = eval_envs.reset()
        ts, eps_count = 0, 0

        pbar = tqdm(
            total=n,
            desc=f"Eval [{input_mode.value}]{'[+DTW]' if compute_dtw else ''}",
            disable=not progress_bar,
            unit="ep",
        )

        while eps_count < n:
            obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
            if not delta_control:
                obs["state"], obs["rgb"] = evaluate_processor.normalize_state_rgb(
                    obs["state"], obs["rgb"], env_id
                )

            # ── Encode task ONCE at episode start ─────────────────────────────
            if ts == 0:
                robot_obs_for_task = {
                    "states": obs.get("state", obs.get("states")),
                    "view_1": obs.get("rgb",   obs.get("view_1")),
                }
                agent.prepare_for_eval(
                    human_video=human_video,
                    robot_obs=robot_obs_for_task,
                    human_desc=human_desc,
                )

            # ── Query policy every query_frequency steps ──────────────────────
            if ts % query_frequency == 0:
                action_seq = agent.get_action(obs)
                if action_seq.dim() == 2:
                    action_seq = action_seq.unsqueeze(0)
                if light_temporal_agg:
                    action_window.append(action_seq)

            # ── Temporal aggregation or direct slice ──────────────────────────
            if light_temporal_agg:
                # Sliding-window weighted average of the last tagg_window predictions.
                # For each past prediction, find which index in that chunk corresponds
                # to the current timestep:
                #   age=0 → newest (queried this step), index = step_in_chunk
                #   age=1 → queried tagg_window steps ago, index = step_in_chunk + tagg_window
                #   ...
                # Skip any prediction whose index falls outside pred_horizon.
                step_in_chunk = ts % query_frequency
                acts_list, ages = [], []
                for age, past_pred in enumerate(reversed(action_window)):
                    idx = step_in_chunk + age * query_frequency
                    if idx < pred_horizon:
                        acts_list.append(past_pred[:, idx])  # (num_envs, action_dim)
                        ages.append(age)
                if acts_list:
                    w = torch.exp(-0.1 * torch.tensor(ages, dtype=torch.float32, device=device))
                    w = w / w.sum()
                    stacked = torch.stack(acts_list, dim=1)  # (num_envs, valid, action_dim)
                    raw_action = (stacked * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
                else:
                    # Fallback (shouldn't happen in practice)
                    raw_action = action_window[-1][:, step_in_chunk]
            elif temporal_agg:
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
                raw_action = actions_to_take[:, ts % query_frequency]

            # ── Denormalize & step ────────────────────────────────────────────
            _action = (
                evaluate_processor.denormalize_action(raw_action, env_id)
                if not delta_control else raw_action
            )

            if sim_backend == "physx_cpu":
                _action = _action.cpu().numpy()

            # ── Collect action for DTW ────────────────────────────────────────
            if compute_dtw:
                action_buf.append(_action)

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

                if light_temporal_agg:
                    action_window.clear()
                elif temporal_agg:
                    all_time_actions = torch.zeros(
                        [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim],
                        device=device,
                    )

                agent.clear_cache()
                obs, info = eval_envs.reset()

                if compute_dtw:
                    new_gt = dtw_provider.sample_gt_trajectory(env_id, seed=None)
                    if new_gt is not None:
                        gt_traj = new_gt

        pbar.close()

    agent.train()

    for k in eval_metrics:
        eval_metrics[k] = np.stack(eval_metrics[k])

    return eval_metrics


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diffusion Policy Evaluation with Task Encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--eval-config",  type=str, required=True)
    parser.add_argument("--checkpoint",   type=str, required=True)

    # Input mode
    parser.add_argument("--input-mode", type=str, default="video_only",
                        choices=["video_only", "language_only", "video_and_language"])

    # Data configs
    parser.add_argument("--sim-config",     type=str,
                        default="examples/baselines/lerobot_dataset/config/sim_config.json")
    parser.add_argument("--human-config",   type=str,
                        default="examples/baselines/lerobot_dataset/config/human_config.json")
    parser.add_argument("--task-mapping",   type=str,
                        default="examples/baselines/lerobot_dataset/task_mapping.json")
    parser.add_argument("--human-task-desc", type=str,
                        default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    parser.add_argument("--sim-task-desc",  type=str,
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

    # Model architecture
    parser.add_argument("--action-dim",          type=int, default=16)
    parser.add_argument("--state-dim",           type=int, default=18)
    parser.add_argument("--obs-horizon",         type=int, default=1)
    parser.add_argument("--pred-horizon",        type=int, default=16)
    parser.add_argument("--obs-latent-dim",      type=int, default=256)
    parser.add_argument("--task-latent-dim",     type=int, default=256)
    parser.add_argument("--task-num-frames",     type=int, default=10)
    parser.add_argument("--image-size",          type=int, nargs=2, default=[224, 224])
    parser.add_argument("--cameras",             type=str, nargs="+", default=["zed2i"])
    parser.add_argument("--include-depth",       action="store_true", default=False)
    parser.add_argument("--state-type",          type=str, default="qpos")
    parser.add_argument("--single-arm",          action="store_true", default=False)

    # Scratch encoder parameters
    parser.add_argument("--vocab-size",   type=int, default=32000)
    parser.add_argument("--max-text-len", type=int, default=500)
    parser.add_argument("--num-tasks",    type=int, default=200)
    parser.add_argument("--task-seq-len", type=int, default=10)
    parser.add_argument("--mano-dim",     type=int, default=14)

    # Task encoder backend
    parser.add_argument("--task-encoder-type", type=str, default="frozen_backbone",
                        choices=["frozen_backbone"])

    # FrozenVideoBackbone parameters (task_encoder_type == "frozen_backbone")
    parser.add_argument("--frozen-backbone-type",           type=str,  default="dinov2_vitl14",
                        help="Backbone architecture: dinov2_vitl14 | dinov2_vitb14 | "
                             "clip_vitl14 | clip_vitb16 | siglip2_so400m | "
                             "videomae_large | videomae_base")
    parser.add_argument("--frozen-backbone-adapter-layers", type=int,  default=1,
                        help="Adapter depth: 1=Linear (default), 2=Linear→LN→GELU→Linear")
    parser.add_argument("--frozen-backbone-adapter-ln",     action="store_true", default=True,
                        help="Add LayerNorm after adapter projection")
    parser.add_argument("--frozen-backbone-seq-patches",    type=int,  default=32,
                        help="Max patch tokens kept for cross-attention")
    parser.add_argument("--frozen-backbone-num-frames",     type=int,  default=4,
                        help="Number of frames sampled from video for frame-level backbones")
    parser.add_argument("--frozen-backbone-lora-rank",  type=int,   default=0,
                        help="LoRA rank used during training (0=frozen)")
    parser.add_argument("--frozen-backbone-lora-alpha", type=float, default=16.0,
                        help="LoRA alpha used during training")

    # Diffusion policy specific
    parser.add_argument("--temporal-agg",        action="store_true", default=False)
    parser.add_argument("--light-temporal-agg",  action="store_true", default=False,
                        help="Sliding-window temp agg (query every --tagg-window steps, "
                             "exponential weighting). Takes priority over --temporal-agg.")
    parser.add_argument("--tagg-window",         type=int, default=4,
                        help="Window size for --light-temporal-agg (default 4).")
    parser.add_argument("--use-ddim",            action="store_true", default=True,
                        help="Replace DDPMScheduler with DDIMScheduler at eval (fast, ~lossless)")
    parser.add_argument("--num-ddim-steps",      type=int, default=10,
                        help="DDIM denoising steps (default 10; only used when --use-ddim)")
    parser.add_argument("--num-diffusion-iters", type=int, default=100)

    # DTW trajectory metrics
    parser.add_argument("--compute-dtw",   action="store_true", default=True)
    parser.add_argument("--dtw-band-ratio", type=float, default=0.15)

    # Output / device
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda")

    return parser.parse_args()


# =============================================================================
# Config helpers
# =============================================================================

def load_eval_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    if config_path.suffix != ".txt":
        raise ValueError(f"Only .txt config files are supported. Got: {config_path.suffix}")
    with open(config_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return {
        "eval_name":    config_path.stem,
        "description":  f"Evaluation from {config_path.name}",
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


def save_results_to_json(results: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# =============================================================================
# Agent loader
# =============================================================================

def load_agent_from_checkpoint(checkpoint_path: str, args, device: torch.device):
    """Load DPAgent from checkpoint, rebuilding from saved args."""
    from examples.baselines.diffusion_policy.train_dp_imitator import DPAgent, TrainingArgs

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ── Reconstruct TrainingArgs from saved args ──────────────────────────────
    saved = checkpoint.get("args", {})
    if not isinstance(saved, dict):
        saved = vars(saved)

    def _get(key, default):
        return saved.get(key, default)

    training_args = TrainingArgs(
        action_dim               = _get("action_dim",               args.action_dim),
        state_dim                = _get("state_dim",                args.state_dim),
        obs_horizon              = _get("obs_horizon",              args.obs_horizon),
        pred_horizon             = _get("pred_horizon",             args.pred_horizon),
        unet_dims                = _get("unet_dims",                [256, 512, 1024]),
        n_groups                 = _get("n_groups",                 8),
        num_diffusion_iters      = _get("num_diffusion_iters",      args.num_diffusion_iters),
        diffusion_step_embed_dim = _get("diffusion_step_embed_dim", 32),
        obs_latent_dim           = _get("obs_latent_dim",           args.obs_latent_dim),
        state_latent_dim         = _get("state_latent_dim",         256),
        cameras                  = _get("cameras",                  args.cameras),
        include_depth            = _get("include_depth",            args.include_depth),
        image_size               = tuple(_get("image_size",         args.image_size)),
        state_type               = _get("state_type",               args.state_type),
        single_arm               = _get("single_arm",               args.single_arm),
        task_encoder_type        = _get("task_encoder_type",        args.task_encoder_type),
        task_latent_dim          = _get("task_latent_dim",          args.task_latent_dim),
        task_num_frames          = _get("task_num_frames",          args.task_num_frames),
        task_seq_len             = _get("task_seq_len",             args.task_seq_len),
        vocab_size               = _get("vocab_size",               args.vocab_size),
        max_text_len             = _get("max_text_len",             args.max_text_len),
        num_tasks                = _get("num_tasks",                args.num_tasks),
        mano_dim                 = _get("mano_dim",                 args.mano_dim),
        # FrozenVideoBackbone fields — read from checkpoint, CLI overrides allowed
        frozen_backbone_type           = _get("frozen_backbone_type",           args.frozen_backbone_type),
        frozen_backbone_adapter_layers = _get("frozen_backbone_adapter_layers", args.frozen_backbone_adapter_layers),
        frozen_backbone_seq_patches    = _get("frozen_backbone_seq_patches",    args.frozen_backbone_seq_patches),
        frozen_backbone_num_frames     = _get("frozen_backbone_num_frames",     args.frozen_backbone_num_frames),
        frozen_backbone_lora_rank      = _get("frozen_backbone_lora_rank", getattr(args, "frozen_backbone_lora_rank", 0)),
        frozen_backbone_lora_alpha = _get("frozen_backbone_lora_alpha", getattr(args, "frozen_backbone_lora_alpha", 16.0)),
        input_mode                       = _get("input_mode", "video_only"),
        obs_encoder_type                 = _get("obs_encoder_type", "simple_cnn"),
        state_encoder_type               = _get("state_encoder_type", "mlp"),
        state_hidden_dim                 = _get("state_hidden_dim", 256),
        state_num_layers                 = _get("state_num_layers", 4),
        state_num_heads                  = _get("state_num_heads", 4),
        cuda                             = True,
    )

    print(f"\n🤖 Creating DPAgent ...")
    agent = DPAgent(training_args, device)

    # ── Load state dict, stripping CachedQwenVLEncoder 'original' prefix ─────
    raw_sd = checkpoint.get("agent_state_dict", checkpoint.get("agent", {}))
    fixed_sd = {}
    for k, v in raw_sd.items():
        new_k = k.replace(".video_encoder.original.", ".video_encoder.") \
            .replace(".lang_encoder.original.",  ".lang_encoder.") \
            .replace("task_encoder.backbone.",    "task_encoder.")
        fixed_sd[new_k] = v

    missing, unexpected = agent.load_state_dict(fixed_sd, strict=False)
    if missing:
        print(f"⚠️  Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"⚠️  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    agent = agent.to(device)
    agent.eval()

    print(f"\n✅ Agent loaded successfully")
    print(f"   Epoch: {checkpoint.get('epoch', checkpoint.get('iteration', 'unknown'))}")
    total_params = sum(p.numel() for p in agent.parameters()) / 1e6
    print(f"   Total parameters: {total_params:.2f}M")

    model_config = {
        "cameras":          training_args.cameras,
        "include_depth":    training_args.include_depth,
        "num_video_frames": training_args.frozen_backbone_num_frames,
        "image_size":       training_args.image_size,
        "state_type":       training_args.state_type,
        "single_arm":       training_args.single_arm,
        "pred_horizon":     training_args.pred_horizon,
        "obs_horizon":      training_args.obs_horizon,
    }
    return agent, model_config


# =============================================================================
# Evaluate a single environment
# =============================================================================

def evaluate_single_env(
    agent,
    env_id: str,
    args,
    evaluate_processor,
    output_dir: Path,
    model_config: Dict,
    input_mode: InputMode,
    dtw_provider=None,
    traj_metrics=None,
) -> Dict:
    base_env_name = extract_base_env_name(env_id)
    level         = extract_level(env_id)

    print(f"\n{'='*60}")
    print(f"🎯 Evaluating: {env_id}")
    print(f"   Base env: {base_env_name}, Level: {level}")
    print(f"{'='*60}")

    obs_mode = "rgbd" if model_config["include_depth"] else "rgb"
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="dense",
        obs_mode=obs_mode,
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps, 
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
    )
    other_kwargs = dict(obs_horizon=model_config["obs_horizon"])

    # NOTE: initialise to None so the finally block can safely guard
    # eval_envs.close() even if env creation itself raises (e.g. SAPIEN
    # Vulkan resource exhaustion after several sequential env creations).
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
            pred_horizon=model_config["pred_horizon"],
            temporal_agg=args.temporal_agg,
            light_temporal_agg=args.light_temporal_agg,
            tagg_window=args.tagg_window,
            max_timesteps=args.max_episode_steps,
            device=args.device,
            sim_backend=args.sim_backend,
        )

        eval_metrics = evaluate_with_task_encoder(
            n=args.num_episodes,
            agent=agent,
            eval_envs=eval_envs,
            eval_kwargs=eval_kwargs,
            evaluate_processor=evaluate_processor,
            input_mode=input_mode,
            progress_bar=True,
            dtw_provider=dtw_provider,
            traj_metrics=traj_metrics,
            dtw_band_ratio=args.dtw_band_ratio,
        )

        results = {
            "env_id":      env_id,
            "level":       level,
            "input_mode":  input_mode.value,
            "backend":     "frozen_backbone",
            "num_episodes": args.num_episodes,
            "timestamp":   datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status":      "success",
        }
        for k, v in eval_metrics.items():
            results[f"{k}_mean"] = float(v.mean())
            results[f"{k}_std"]  = float(v.std())

        print(f"\n📊 Results [{input_mode.value}]:")
        for k in ["success_once", "success_at_end", "return"]:
            if f"{k}_mean" in results:
                print(f"   {k}: {results[f'{k}_mean']:.4f} ± {results[f'{k}_std']:.4f}")
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
            "env_id":    env_id,
            "level":     level,
            "input_mode": input_mode.value,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status":    "error",
            "error":     str(e),
        }

    finally:
        # Close the env and explicitly delete the reference so Python's GC can
        # immediately trigger SAPIEN's destructor.  Without del + gc.collect(),
        # Vulkan/CUDA render contexts accumulate across sequential env creations
        # and exhaust GPU render resources (typically crashes on the 5th env
        # when a large model occupies ~10 GB of VRAM).
        # time.sleep(1) gives the GPU driver time to fully release Vulkan
        # handles before the next RenderSystem is initialised.
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
    print("🧪 DIFFUSION POLICY EVALUATION")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_mode = InputMode(args.input_mode)
    print(f"\n📝 Input Mode: {input_mode.value}")

    print(f"\n📋 Loading eval configuration: {args.eval_config}")
    eval_config  = load_eval_config(args.eval_config)
    eval_name    = eval_config.get("eval_name", "unnamed_eval")
    environments = eval_config.get("environments", [])
    print(f"   Total environments in config: {len(environments)}")

    existing_results = load_existing_results(output_dir, input_mode.value)
    print(f"   Found existing results for {len(existing_results)} env(s)")

    pending_environments = []
    skipped_count = 0
    for env_config in environments:
        env_id = env_config["env_id"]
        if is_env_already_evaluated(env_id, existing_results, args.num_episodes):
            skipped_count += 1
        else:
            pending_environments.append(env_config)

    print(f"\n   ⏭️  Already evaluated (skipping): {skipped_count}")
    print(f"   🔜 Pending evaluation:            {len(pending_environments)}")

    if not pending_environments:
        print("\n✅ All environments already evaluated. Nothing to do.")
        return

    # ── Load agent ────────────────────────────────────────────────────────────
    device = torch.device(args.device)
    agent, model_config = load_agent_from_checkpoint(args.checkpoint, args, device)

    # ── DDIM injection (optional eval-time acceleration) ──────────────────
    if args.use_ddim:
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        print(f"  [DDIM] Replacing DDPMScheduler → DDIMScheduler ({args.num_ddim_steps} steps)")
        ddim_sched = DDIMScheduler.from_config(agent.noise_scheduler.config)
        agent.noise_scheduler = ddim_sched
        agent.num_diffusion_iters = args.num_ddim_steps
        print(f"  [DDIM] ~{100 // args.num_ddim_steps}× fewer denoising steps vs DDPM-100")

    print(f"\n🔧 Task encoder backend: frozen_backbone ({getattr(agent.args, 'frozen_backbone_type', 'dinov2_vitl14')})")

    # ── Evaluation processor ──────────────────────────────────────────────────
    print("\n📹 Setting up evaluation processor...")
    evaluate_processor_config = HumanVideoSimEvaluateProcessorConfig(
        human_root=args.human_root,
        human_split="train",
        human_dataset_file=args.human_config,
        human_task_description_file=args.human_task_desc,
        human_cameras=model_config.get("cameras", args.cameras),
        human_include_depth=model_config.get("include_depth", args.include_depth),
        human_num_frames=model_config.get("num_video_frames", args.task_num_frames),
        human_image_size=model_config.get("image_size", tuple(args.image_size)),
        human_video_backend="torchcodec",
        human_fps=30,
        sim_root=args.sim_root,
        sim_split="train",
        sim_dataset_file=args.sim_config,
        sim_task_description_file=args.sim_task_desc,
        sim_state_type=model_config.get("state_type", args.state_type),
        sim_single_arm=model_config.get("single_arm", args.single_arm),
        normalization_method="bounds_q99",
        task_mapping_file=args.task_mapping,
    )
    evaluate_processor = HumanVideoSimEvaluateProcessor(evaluate_processor_config)
    print("✅ Evaluate processor ready")

    # ── DTW provider ──────────────────────────────────────────────────────────
    dtw_provider = None
    traj_metrics = None
    if args.compute_dtw:
        print("\n📏 Building GT trajectory provider for TSS...")
        try:
            action_key = (
                "action.qpos_gripper_actions"
                if args.state_type in ("qpos", "mixpos")
                else "action.eepos_gripper_actions"
            )
            dtw_provider = GTTrajectoryProvider(
                sim_dataset_file=args.sim_config,
                sim_root=args.sim_root,
                action_key=action_key,
                normalizer=evaluate_processor.normalizer,
                normalization_method="bounds_q99",
            )
            traj_metrics = TrajectoryMetrics(band_ratio=args.dtw_band_ratio)
            print(f"✅ GT trajectories loaded for {len(dtw_provider._episodes)} task(s).")
        except Exception as exc:
            print(f"⚠️  Could not build GTTrajectoryProvider: {exc}")
            args.compute_dtw = False

    results_json = output_dir / f"{eval_name}_{input_mode.value}_{timestamp}.json"
    print(f"\n📝 Incremental log: {results_json}")

    print("\n" + "=" * 80)
    print(f"🚀 STARTING EVALUATIONS [{input_mode.value}]")
    print("=" * 80)

    all_results = []
    for i, env_config in enumerate(tqdm(pending_environments, desc="Evaluating")):
        env_id = env_config["env_id"]
        result = evaluate_single_env(
            agent=agent,
            env_id=env_id,
            args=args,
            evaluate_processor=evaluate_processor,
            output_dir=output_dir,
            model_config=model_config,
            input_mode=input_mode,
            dtw_provider=dtw_provider,
            traj_metrics=traj_metrics,
        )
        all_results.append(result)
        save_results_to_json(all_results, results_json)

        status_icon = "✅" if result.get("status") == "success" else "❌"
        success_str = (
            f"{result['success_once_mean']:.4f}"
            if result.get("status") == "success" and "success_once_mean" in result
            else result.get("error", "N/A")
        )
        tss_str = (
            f" | wTSS={result['wtss_mean']:.4f}"
            if result.get("wtss_mean") is not None else ""
        )
        print(f"{status_icon} [{i+1}/{len(pending_environments)}] {env_id} "
              f"| success_once={success_str}{tss_str} | log updated")

    print("\n" + "=" * 80)
    print("💾 SAVING FINAL SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(all_results)
    results_csv = output_dir / f"{eval_name}_{input_mode.value}_{timestamp}.csv"
    df.to_csv(results_csv, index=False)
    print(f"✅ JSON log:      {results_json}")
    print(f"✅ CSV summary:   {results_csv}")

    print("\n" + "=" * 80)
    print(f"📊 RESULTS SUMMARY [{input_mode.value}]")
    print("=" * 80)

    if "success_once_mean" in df.columns:
        print(f"\n{'Environment':<45} {'Level':<6} {'Mode':<15} {'Backend':<12} {'Success':<10}")
        print("-" * 92)
        for _, row in df.iterrows():
            env_id  = row["env_id"]
            level   = row.get("level", "L0")
            mode    = row.get("input_mode", "unknown")
            backend = row.get("backend", "scratch")
            if row.get("status") == "error":
                print(f"{env_id:<45} {level:<6} {mode:<15} {backend:<12} {'ERROR':<10}")
            else:
                success = row.get("success_once_mean", 0.0)
                print(f"{env_id:<45} {level:<6} {mode:<15} {backend:<12} {success:<10.4f}")

        print("-" * 92)
        valid_df = df[df["success_once_mean"].notna()]
        if len(valid_df) > 0:
            print(f"{'OVERALL AVERAGE':<45} {'ALL':<6} {input_mode.value:<15} "
                  f"{'—':<12} {valid_df['success_once_mean'].mean():<10.4f}")

    print(f"\n✅ EVALUATION COMPLETE  (skipped {skipped_count}, evaluated {len(all_results)})")


if __name__ == "__main__":
    main()