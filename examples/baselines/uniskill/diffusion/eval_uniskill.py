"""
Evaluation Script for UniSkill (Conditional Diffusion Policy with IDM)
======================================================================
Wraps the UniSkill model (IDM + AlignmentTransformer + ConditionalUnet1D)

The UniSkillAgent exposes:
  - prepare_for_eval(human_video, robot_obs, human_tokens=None)
  - get_action(obs_dict) -> (B, pred_horizon, action_dim)
  - clear_cache()
  - eval() / train()
"""

import json
import os
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import gymnasium
import torchvision

from diffusers import DDPMScheduler
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# UniSkill model components
from examples.baselines.uniskill.diffusion.policy_model import (
    ConditionalUnet1D,
    AlignmentTransformer,
    get_resnet,
    replace_bn_with_gn,
)
from examples.baselines.uniskill.dynamics.idm import IDM
import gc
import time

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import InputMode
from examples.baselines.lerobot_dataset.evaluate_processor import (
    HumanVideoSimEvaluateProcessor,
    HumanVideoSimEvaluateProcessorConfig,
)
from examples.baselines.openvla_oft.utils.make_env import make_eval_envs, parse_optional_bool_flag
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
from mani_skill.utils import common

from examples.baselines.lerobot_dataset.trajectory_metrics import (
    TrajectoryMetrics,
    GTTrajectoryProvider,
    EpisodeActionBuffer,
)

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
# FIX 2 & 3: Subprocess-safe level differentiation
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
    lr_mirror_enabled=None,
    lr_mirror_robot_pose_enabled=None,
):
    set_l_level(level)
    clean_env_kwargs = {k: v for k, v in env_kwargs.items() if k != "l_level"}
    envs = make_eval_envs(
        base_env_name,
        num_envs,
        sim_backend,
        clean_env_kwargs,
        other_kwargs,
        video_dir=video_dir,
        wrappers=wrappers,
        l_level=level,
        lr_mirror_enabled=lr_mirror_enabled,
        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
    )
    return envs

def evaluate_with_task_encoder(
    n: int,
    agent,
    eval_envs,
    eval_kwargs,
    evaluate_processor,
    input_mode: InputMode,
    progress_bar: bool = True,
    # ── NEW: DTW arguments ────────────────────────────────────────────────
    dtw_provider: Optional[GTTrajectoryProvider] = None,
    traj_metrics: Optional[TrajectoryMetrics] = None,
    dtw_band_ratio: float = 0.15,
):
    """
    Evaluate agent with task encoder.

    DTW trajectory metrics
    ----------------------
    If *dtw_provider* is supplied, at the end of each episode the executed
    action trajectory is compared against a randomly sampled GT demo via
    multi-dimensional normalized DTW.  The following keys are added to
    eval_metrics:

        ndtw              – primary metric  (lower = better; 0 = perfect match)
        dtw_raw           – raw DTW accumulated cost
            dtw_length_ratio  – T_pred / T_gt (speed ratio; 1 = same speed)
        dtw_path_len      – length of optimal warping path

    Why not MSE?
    ndtw handles speed differences gracefully: a robot that executes the
    correct motion 1.5× faster gets ndtw ≈ 0 but MSE >> 0.  The Sakoe-Chiba
    band (default 15 % of trajectory length) prevents degenerate alignments.
    """
    env_id        = eval_kwargs.get('env_id')
    delta_control = eval_kwargs.get('delta_control')
    pred_horizon  = eval_kwargs.get('pred_horizon')
    temporal_agg  = eval_kwargs.get('temporal_agg')
    max_timesteps = eval_kwargs.get('max_timesteps')
    device        = eval_kwargs.get('device')
    sim_backend   = eval_kwargs.get('sim_backend')

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
            device=device
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
        human_desc  = [evaluate_processor.get_task_description(env_id)] * num_envs

    # ── NEW: initialise DTW helpers ───────────────────────────────────────────
    compute_dtw = (dtw_provider is not None)
    if compute_dtw:
        if traj_metrics is None:
            traj_metrics = TrajectoryMetrics(band_ratio=dtw_band_ratio)
        gt_traj = dtw_provider.sample_gt_trajectory(env_id, seed=None, normalize=False)
        if gt_traj is None:
            print(f"⚠️  No GT trajectory found for {env_id}; TSS will be skipped.")
            compute_dtw = False

        # EpisodeActionBuffer collects _action at every step
        action_buf = EpisodeActionBuffer(num_envs=num_envs)
        # Track first success step per env (-1 = not yet succeeded)
        success_steps = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info    = eval_envs.reset()
        ts, eps_count = 0, 0

        pbar = tqdm(
            total=n,
            desc=(f"Eval [{input_mode.value}]"
                  f"{'[+DTW]' if compute_dtw else ''}"),
            disable=not progress_bar,
            unit="ep",
        )

        while eps_count < n:
            obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
            if not delta_control:
                obs['state'], obs['rgb'] = evaluate_processor.normalize_state_rgb(
                    obs['state'], obs['rgb'], env_id
                )

            # Generate task features ONCE at episode start
            if ts == 0:
                robot_obs_for_task = {
                    'states': obs.get('state', obs.get('states')),
                    'view_1': obs.get('rgb', obs.get('view_1'))
                }
                agent.prepare_for_eval(
                    human_video=human_video,
                    robot_obs=robot_obs_for_task,
                    human_desc=human_desc,
                )

            if ts % query_frequency == 0:
                action_seq = agent.get_action(obs)

            # Temporal ensembling
            if temporal_agg:
                all_time_actions[:, ts, ts:ts + pred_horizon] = action_seq
                actions_for_curr_step = all_time_actions[:, :, ts]
                actions_populated     = torch.zeros(max_timesteps, dtype=torch.bool, device=device)
                actions_populated[max(0, ts + 1 - pred_horizon):ts + 1] = True
                actions_for_curr_step = actions_for_curr_step[:, actions_populated]
                k_exp     = 0.01
                exp_weights = torch.exp(-k_exp * torch.arange(
                    len(actions_for_curr_step[0]), device=device))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.tile(exp_weights, (num_envs, 1))
                exp_weights = torch.unsqueeze(exp_weights, -1)
                raw_action  = (actions_for_curr_step * exp_weights).sum(dim=1)
            else:
                if ts % query_frequency == 0:
                    actions_to_take = action_seq
                raw_action = actions_to_take[:, ts % query_frequency]

            _action = (
                evaluate_processor.denormalize_action(raw_action, env_id)
                if not delta_control else raw_action
            )

            if sim_backend == "physx_cpu":
                _action = _action.cpu().numpy()

            # ── NEW: record executed action ───────────────────────────────
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
                # ── Standard episode metrics ──────────────────────────────
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

                if temporal_agg:
                    all_time_actions = torch.zeros(
                        [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim],
                        device=device
                    )

                agent.clear_cache()
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
    input_mode: InputMode,
    # ── NEW ──
    dtw_provider: Optional[GTTrajectoryProvider] = None,
    traj_metrics: Optional[TrajectoryMetrics] = None,
) -> Dict:
    """Run evaluation for a single environment and return result dict."""
    base_env_name = extract_base_env_name(env_id)
    level         = extract_level(env_id)
    print(f"🎯 Evaluating: {env_id}")
    print(f"   Base env: {base_env_name}, Level: {level}")

    obs_mode = "rgbd" if args.include_depth else "rgb"
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="dense",
        obs_mode=obs_mode,
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps, 
        sensor_configs=dict(shader_pack="rt-fast"), 
        human_render_camera_configs=dict(shader_pack="rt-fast"),
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)

    # NOTE: initialise to None so the finally block can safely guard
    # eval_envs.close() even if env creation itself raises (e.g. SAPIEN
    # Vulkan resource exhaustion after several sequential env creations).
    eval_envs = None
    try:
        lr_mirror_enabled = parse_optional_bool_flag(args.eval_lr_mirror)
        lr_mirror_robot_pose_enabled = parse_optional_bool_flag(args.eval_lr_mirror_robot_pose)

        eval_envs = make_eval_envs_with_level(
            base_env_name=base_env_name,
            level=level,
            num_envs=args.num_envs,
            sim_backend=args.sim_backend,
            env_kwargs=env_kwargs,
            other_kwargs=other_kwargs,
            video_dir=str(output_dir / "videos" / env_id),
            wrappers=[FlattenRGBDObservationWrapper],
            lr_mirror_enabled=lr_mirror_enabled,
            lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
        )
        eval_kwargs = dict(
            env_id=env_id,
            delta_control=("delta" in args.control_mode),
            pred_horizon=model_config['pred_horizon'],
            temporal_agg=args.temporal_agg,
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
            # ── NEW ──
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

        # ── Console summary ────────────────────────────────────────────
        print(f"\n📊 Results [{input_mode.value}]:")
        for k in ["success_once", "success_at_end", "return"]:
            if f"{k}_mean" in results:
                print(f"   {k}: {results[f'{k}_mean']:.4f} ± {results[f'{k}_std']:.4f}")

        # NEW: print DTW summary if computed
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
# UniSkill Agent — wraps IDM + AlignmentTransformer + DiffusionPolicy
# =============================================================================

class UniSkillAgent(nn.Module):
    """
    Components (loaded from accelerator checkpoint):
      - vision_encoder:   ResNet18 (with GroupNorm) -> 512-dim
      - vision_projection: Linear(512, vision_feature_dim)
      - idm_projection:   Linear(768, idm_feature_dim)
      - noise_pred_net:   ConditionalUnet1D
      - alignment_net:    AlignmentTransformer

    Frozen components:
      - idm:              IDM (from Stage 1)
      - depth_estimator:  Depth-Anything-V2-Small
    """

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args
        self.pred_horizon = args.policy_pred_horizon
        self.obs_horizon = args.policy_obs_horizon
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.vision_feature_dim = args.vision_feature_dim
        self.idm_feature_dim = args.idm_feature_dim
        self.idm_resolution = args.idm_resolution
        self.resolution = args.resolution

        # --- Frozen IDM ---
        self.idm = IDM(
            num_layers=8,
            num_heads=4,
            hidden_dim=256,
            skill_dim=64,
            out_dim=768,
            idm_resolution=args.idm_resolution,
        )
        idm_state_dict = torch.load(args.idm_ckpt_path, map_location="cpu")
        self.idm.load_state_dict(idm_state_dict, strict=False)
        self.idm.requires_grad_(False)
        self.idm.eval()

        # --- Frozen depth estimator ---
        self.depth_estimator = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        self.depth_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        self.depth_estimator.requires_grad_(False)
        self.depth_estimator.eval()

        # --- Trainable components ---
        self.vision_encoder = get_resnet("resnet18")
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)

        self.vision_projection = (
            nn.Linear(512, args.vision_feature_dim)
            if args.vision_feature_dim != 512
            else nn.Identity()
        )

        self.idm_projection = nn.Linear(768, args.idm_feature_dim)

        global_cond_dim = (
            args.vision_feature_dim * args.policy_obs_horizon
            + args.obs_dim * args.policy_obs_horizon
            + args.idm_feature_dim
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=args.action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=256,
        )

        robot_dim = (
            args.vision_feature_dim * args.policy_obs_horizon
            + args.obs_dim * args.policy_obs_horizon
        )
        self.alignment_net = AlignmentTransformer(
            robot_dim=robot_dim,
            idm_dim=768,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
            
        # Eval-time caches
        self._cached_human_idm_latents = None  # (B, T, 768)
        self._cached_human_mask = None          # (B, T)

    # -----------------------------------------------------------------
    # Checkpoint loading from accelerator.save_state format
    # -----------------------------------------------------------------
    def load_from_accelerator_checkpoint(self, checkpoint_dir: str):
        """
        Load weights saved by accelerator.save_state().
        The order matches the order passed to accelerator.prepare() in
        train_cond_dp.py:
          0 -> vision_encoder
          1 -> vision_projection
          2 -> idm_projection
          3 -> noise_pred_net
          4 -> alignment_net
        """
        ckpt_dir = Path(checkpoint_dir)

        # accelerator saves each model as model_N/ or pytorch_model_N/
        # Try safetensors first, then bin
        def load_model_weights(model, idx):
            if idx == 0:
                prefix = "model"
            else:
                prefix = f"model_{idx}"

            from safetensors.torch import load_file

            # Try subfolder layout: <prefix>/model.safetensors
            st_path = ckpt_dir / prefix / "model.safetensors"
            if st_path.exists():
                state_dict = load_file(str(st_path))
                model.load_state_dict(state_dict, strict=False)
                return

            # Try flat layout: <prefix>.safetensors
            flat_st_path = ckpt_dir / f"{prefix}.safetensors"
            if flat_st_path.exists():
                state_dict = load_file(str(flat_st_path))
                model.load_state_dict(state_dict, strict=False)
                return

            # Try pytorch bin (subfolder)
            bin_path = ckpt_dir / prefix / "pytorch_model.bin"
            if bin_path.exists():
                state_dict = torch.load(bin_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                return

            raise FileNotFoundError(
                f"No checkpoint found for {prefix} in {ckpt_dir}"
            )

        load_model_weights(self.vision_encoder, 0)
        load_model_weights(self.vision_projection, 1)
        load_model_weights(self.idm_projection, 2)
        load_model_weights(self.noise_pred_net, 3)
        load_model_weights(self.alignment_net, 4)

        print(f"✅ Loaded UniSkill policy from {checkpoint_dir}")

    # -----------------------------------------------------------------
    # Human video -> IDM latent encoding
    # -----------------------------------------------------------------
    @torch.no_grad()
    def _encode_human_video_to_idm_latents(
        self, human_video: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode a human demonstration video into a sequence of IDM latents.

        Args:
            human_video: (B, T, H, W, C) float tensor in [0, 1]

        Returns:
            latents: (B, T-1, 768) — one latent per consecutive frame pair
        """
        B, T, H, W, C = human_video.shape

        # Build consecutive frame pairs: (B*(T-1), 2, 3, H, W)
        frames_chw = human_video.permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        curr_frames = frames_chw[:, :-1]  # (B, T-1, C, H, W)
        next_frames = frames_chw[:, 1:]   # (B, T-1, C, H, W)

        num_pairs = T - 1
        # Stack into pair format: (B*(T-1), 2, C, H, W)
        visual_pairs = torch.stack([curr_frames, next_frames], dim=2)
        visual_pairs = visual_pairs.reshape(B * num_pairs, 2, C, H, W)

        # Resize visual to IDM resolution
        visual_flat = visual_pairs.reshape(B * num_pairs * 2, C, H, W)
        visual_flat = F.interpolate(
            visual_flat,
            size=(self.idm_resolution, self.idm_resolution),
            mode="bilinear",
            align_corners=False,
        )
        visual_pairs = visual_flat.reshape(B * num_pairs, 2, C, self.idm_resolution, self.idm_resolution)

        # Depth estimation on all frames.
        # Match training: dataset applies HF depth_processor (ImageNet mean/std) before depth_estimator.
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=visual_flat.device).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=visual_flat.device).view(1, 3, 1, 1)
        depth_input = (visual_flat - imagenet_mean) / imagenet_std
        depth_out = self.depth_estimator(depth_input).predicted_depth  # (N, Hd, Wd)

        # Normalize depth per-sample
        d_min = depth_out.flatten(1).min(dim=1)[0]
        d_max = depth_out.flatten(1).max(dim=1)[0]
        d_denom = (d_max - d_min).clamp(min=1e-6)
        depth_out = (depth_out - d_min[..., None, None]) / d_denom[..., None, None]

        # Reshape to pair format: (B*(T-1), 2, Hd, Wd)
        depth_pairs = depth_out.reshape(B * num_pairs, 2, depth_out.shape[-2], depth_out.shape[-1])
        depth_pairs = F.interpolate(
            depth_pairs,
            size=(self.idm_resolution, self.idm_resolution),
            mode="bilinear",
            align_corners=False,
        )

        # Run IDM: (B*(T-1), 1, 768) -> (B*(T-1), 768)
        idm_latents = self.idm(depth_pairs, visual_pairs).squeeze(1)

        # Reshape back to (B, T-1, 768)
        idm_latents = idm_latents.reshape(B, num_pairs, 768)
        return idm_latents

    def prepare_for_eval(
        self,
        human_video: Optional[torch.Tensor] = None,
        robot_obs: Optional[Dict] = None,
        human_tokens: Optional[torch.Tensor] = None,
    ):
        """
        Called ONCE at episode start.
        Encodes human video through IDM and caches the latent sequence.

        Args:
            human_video: (B, T, H, W, C) float tensor — human demonstration
            robot_obs: dict with 'states' and 'view_1'/'rgb' — not used by UniSkill
            human_tokens: not used by UniSkill (video-only)
        """
        self.eval()
        with torch.no_grad():
            if human_video is not None:
                human_video = human_video.to(self.device)
                # Encode human demo -> IDM latent sequence
                latents = self._encode_human_video_to_idm_latents(human_video)
                self._cached_human_idm_latents = latents  # (B, T-1, 768)
                # No padding needed — all frames are valid
                self._cached_human_mask = torch.zeros(
                    latents.shape[0], latents.shape[1], dtype=torch.bool, device=self.device
                )
            else:
                # Fallback: zero latent with full mask
                B = 1
                if robot_obs is not None:
                    state = robot_obs.get("states", robot_obs.get("state"))
                    if state is not None:
                        B = state.shape[0]
                self._cached_human_idm_latents = torch.zeros(B, 1, 768, device=self.device)
                self._cached_human_mask = torch.ones(B, 1, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def get_action(self, obs_dict: Dict) -> torch.Tensor:
        """
        Get action sequence for current observation.

        Args:
            obs_dict: dict with 'state' (B, D) or (B, T, D) and
                      'rgb' (B, H, W, C) or (B, T, C, H, W)

        Returns:
            actions: (B, pred_horizon, action_dim)
        """
        states = obs_dict.get("state", obs_dict.get("states"))
        images = obs_dict.get("rgb", obs_dict.get("view_1"))

        # Ensure (B, T, ...) format
        if states.dim() == 2:
            states = states.unsqueeze(1)
        if images.dim() == 4:
            images = images.unsqueeze(1)

        B = states.shape[0]

        # Process images -> vision features
        # images may be (B, T, H, W, C) or (B, T, C, H, W)
        if images.shape[-1] in [3, 4]:  # (B, T, H, W, C)
            images = images.permute(0, 1, 4, 2, 3)  # -> (B, T, C, H, W)

        # Match training: state uses last obs_horizon frames, vision uses the latest single frame
        # and its feature is repeated obs_horizon times (dataset only emits curr_images, train loop
        # does visual_feat.repeat(1, policy_obs_horizon)).
        if states.shape[1] < self.obs_horizon:
            states = states.repeat(1, self.obs_horizon, 1)[:, :self.obs_horizon]
        else:
            states = states[:, -self.obs_horizon:]

        img_latest = images[:, -1]  # (B, C, H, W)
        img_latest = F.interpolate(
            img_latest,
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize to [-1, 1] to match training (dataset applies (x - 0.5) / 0.5)
        img_latest = (img_latest - 0.5) / 0.5

        vis_feat = self.vision_encoder(img_latest).flatten(1)  # (B, 512)
        vis_feat = self.vision_projection(vis_feat)  # (B, vision_feature_dim)
        vis_feat = vis_feat.repeat(1, self.obs_horizon)  # (B, obs_horizon * vision_feature_dim)

        state_flat = states.flatten(1)  # (B, obs_horizon * obs_dim)

        # Alignment: query = robot features, memory = cached human IDM latents
        robot_feat = torch.cat([vis_feat, state_flat], dim=-1)
        pred_idm_latent = self.alignment_net(
            robot_feat,
            self._cached_human_idm_latents,
            self._cached_human_mask,
        )  # (B, 768)

        # Project IDM latent
        cond_idm = self.idm_projection(pred_idm_latent)  # (B, idm_feature_dim)

        # Global conditioning
        global_cond = torch.cat([vis_feat, state_flat, cond_idm], dim=-1)

        # Diffusion denoising
        self.noise_scheduler.set_timesteps(self.args.num_diffusion_iters)
        actions = torch.randn(
            (B, self.pred_horizon, self.action_dim), device=self.device
        )

        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                actions,
                k.expand(B).to(self.device),
                global_cond=global_cond,
            )
            actions = self.noise_scheduler.step(noise_pred, k, actions).prev_sample

        return actions

    def clear_cache(self):
        """Clear cached human IDM latents between episodes."""
        self._cached_human_idm_latents = None
        self._cached_human_mask = None

# =============================================================================
# Checkpoint loading  (unchanged from original)
# =============================================================================

def load_eval_config(config_path: str) -> Dict[str, Any]:
    """Read a plain-text .txt eval config (one env-id per line, # comments ignored).
    Returns the same {"eval_name", "environments"} dict used by the other eval scripts.
    """
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
    """Strip optional level prefix (e.g. 'L2_TwoRobotPourCup-v1' → 'TwoRobotPourCup-v1').
    Mirrors the identical helper in eval_dp/act/vqbet_imitator.py.
    """
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
    """Extract level string from env_id (e.g. 'L2_TwoRobotPourCup-v1' → 'L2').
    Returns 'L0' for env_ids without a level prefix.
    """
    if env_id.startswith("L") and "_" in env_id:
        parts = env_id.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            return parts[0]
    return "L0"


def save_results_to_json(results: List[Dict], path: Path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

# =============================================================================
# Checkpoint loading
# =============================================================================

def load_agent_from_checkpoint(checkpoint_path: str, args, device):
    """
    Load UniSkillAgent from an accelerator checkpoint directory.

    Expected directory structure (from accelerator.save_state):
      checkpoint_path/
        model/model.safetensors        -> vision_encoder
        model_1/model.safetensors      -> vision_projection
        model_2/model.safetensors      -> idm_projection
        model_3/model.safetensors      -> noise_pred_net
        model_4/model.safetensors      -> alignment_net

    Also expects args.idm_ckpt_path to point to the frozen IDM weights.
    """
    agent = UniSkillAgent(args, device)
    agent.load_from_accelerator_checkpoint(checkpoint_path)
    agent.to(device)

    model_config = {
        "pred_horizon": args.policy_pred_horizon,
        "obs_horizon": args.policy_obs_horizon,
        "action_dim": args.action_dim,
        "cameras": args.cameras,
        "include_depth": args.include_depth,
        "image_size": tuple(args.image_size),
        "task_num_frames": args.num_video_frames,
        "state_type": args.state_type,
        "single_arm": args.single_arm,
    }

    return agent, model_config


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="UniSkill Evaluation")

    parser.add_argument("--eval-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to accelerator checkpoint dir (e.g. outputs_policy/checkpoint-50000)")

    # UniSkill-specific
    parser.add_argument("--idm-ckpt-path", type=str, required=True,
                        help="Path to frozen IDM checkpoint (idm.pth)")
    parser.add_argument("--input-mode", type=str, default="video_only",
                        choices=["video_only", "language_only", "video_and_language"])

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
    parser.add_argument("--sim-root", type=str, default="demos")

    # Evaluation settings
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=500)

    # Environment settings
    parser.add_argument("--sim-backend", type=str, default="physx_cpu")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--obs-mode", type=str, default="rgb")

    # Model parameters (must match training)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--obs-dim", type=int, default=18)
    parser.add_argument("--obs-horizon", type=int, default=2,
                        help="Alias for policy_obs_horizon, kept for CLI")
    parser.add_argument("--policy-obs-horizon", type=int, default=None,
                        help="If set, overrides --obs-horizon")
    parser.add_argument("--policy-pred-horizon", type=int, default=16)
    parser.add_argument("--vision-feature-dim", type=int, default=256)
    parser.add_argument("--idm-feature-dim", type=int, default=128)
    parser.add_argument("--num-diffusion-iters", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=112,
                        help="Image resolution for policy vision encoder")
    parser.add_argument("--idm-resolution", type=int, default=224,
                        help="Image resolution for IDM")

    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--cameras", type=str, nargs="+", default=["zed2i"])
    parser.add_argument("--include-depth", action="store_true", default=False)
    parser.add_argument("--state-type", type=str, default="qpos")
    parser.add_argument("--single-arm", action="store_true", default=False)
    parser.add_argument("--num-video-frames", type=int, default=10)

    # Eval settings
    parser.add_argument("--temporal-agg", action="store_true", default=False)

    # Output
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shader", type=str, default="rt-fast")

    # Kept for CLI compat (unused; UniSkill is video-only)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-text-len", type=int, default=500)
    parser.add_argument("--task-seq-len", type=int, default=10)
    parser.add_argument("--mano-dim", type=int, default=14)
    parser.add_argument("--pred-horizon", type=int, default=None,
                        help="defaults to policy-pred-horizon")

    parser.add_argument("--eval_lr_mirror", default="auto", choices=["auto", "true", "false"],
                        help="Override tabletop left-right mirror during eval.")
    parser.add_argument("--eval_lr_mirror_robot_pose", default="false", choices=["auto", "true", "false"],
                        help="Override robot pose swapping under mirror during eval. Default 'false' keeps agents unswapped.")
    dtw_group = parser.add_mutually_exclusive_group()
    dtw_group.add_argument(
        "--compute-dtw",
        dest="compute_dtw",
        action="store_true",
        default=False,
        help="Compute TSS/nDTW trajectory metrics when GT trajectories are available",
    )
    dtw_group.add_argument(
        "--no-compute-dtw",
        dest="compute_dtw",
        action="store_false",
        help="Disable TSS/nDTW trajectory metrics",
    )
    parser.add_argument("--dtw-band-ratio", type=float, default=0.15)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Resolve horizon aliases
    if args.policy_obs_horizon is None:
        args.policy_obs_horizon = args.obs_horizon
    if args.pred_horizon is None:
        args.pred_horizon = args.policy_pred_horizon

    print("\n" + "=" * 80)
    print("🧪 UNISKILL EVALUATION (IDM + Alignment + Diffusion Policy)")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_mode = InputMode(args.input_mode)
    print(f"\n📝 Input Mode: {input_mode.value}")

    # Load eval configuration
    print(f"\n📋 Loading eval configuration: {args.eval_config}")
    eval_config = load_eval_config(args.eval_config)
    eval_name = eval_config.get("eval_name", "unnamed_eval")
    environments = eval_config.get("environments", [])
    print(f"   Total environments in config: {len(environments)}")

    # Auto-skip based on JSON eval results
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
        human_cameras=model_config.get("cameras", args.cameras),
        human_include_depth=model_config.get("include_depth", args.include_depth),
        human_num_frames=model_config.get("task_num_frames", args.num_video_frames),
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

    # Incremental log
    results_json = output_dir / f"{eval_name}_{input_mode.value}_{timestamp}.json"
    print(f"\n📝 Incremental log: {results_json}")

    # Run evaluations
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
        )
        all_results.append(result)

        save_results_to_json(all_results, results_json)
        status_icon = "✅" if result.get("status") == "success" else "❌"
        success_str = (
            f"{result['success_once_mean']:.4f}"
            if result.get("status") == "success" and "success_once_mean" in result
            else result.get("error", "N/A")
        )
        print(
            f"{status_icon} [{i+1}/{len(pending_environments)}] {env_id} "
            f"| success_once={success_str} | log updated"
        )

    # Save final CSV summary
    print("\n" + "=" * 80)
    print("💾 SAVING FINAL SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(all_results)
    results_csv = output_dir / f"{eval_name}_{input_mode.value}_{timestamp}.csv"
    df.to_csv(results_csv, index=False)
    print(f"✅ JSON log : {results_json}")
    print(f"✅ CSV summary: {results_csv}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"📊 RESULTS SUMMARY [{input_mode.value}]")
    print("=" * 80)

    if "success_once_mean" in df.columns:
        print(f"\n{'Environment':<45} {'Level':<6} {'Mode':<15} {'Success':<10}")
        print("-" * 80)
        for _, row in df.iterrows():
            env_id = row["env_id"]
            level = row.get("level", "L0")
            mode = row.get("input_mode", "unknown")
            if row.get("status") == "error":
                print(f"{env_id:<45} {level:<6} {mode:<15} {'ERROR':<10}")
            else:
                success = row.get("success_once_mean", 0.0)
                print(f"{env_id:<45} {level:<6} {mode:<15} {success:<10.4f}")

        print("-" * 80)
        valid_df = df[df["success_once_mean"].notna()]
        if len(valid_df) > 0:
            print(
                f"{'OVERALL AVERAGE':<45} {'ALL':<6} {input_mode.value:<15} "
                f"{valid_df['success_once_mean'].mean():<10.4f}"
            )

    print(
        f"\n✅ EVALUATION COMPLETE  (skipped {skipped_count}, evaluated {len(all_results)})"
    )


if __name__ == "__main__":
    main()
