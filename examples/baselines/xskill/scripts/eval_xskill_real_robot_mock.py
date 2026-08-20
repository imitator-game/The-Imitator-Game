"""
Evaluation Script for XSkill
"""

import json
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
XSKILL_PROJECT_ROOT = PROJECT_ROOT / "examples" / "baselines" / "xskill"

for p in [PROJECT_ROOT, XSKILL_PROJECT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pickle
import argparse
import gc
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import gymnasium
import hydra
from omegaconf import OmegaConf
import torchvision.transforms as Tr
from torchvision import transforms

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import InputMode
from examples.baselines.lerobot_dataset.evaluate_processor import (
    HumanVideoSimEvaluateProcessor,
    HumanVideoSimEvaluateProcessorConfig,
)
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer
from examples.baselines.lerobot_dataset.trajectory_metrics import (
    TrajectoryMetrics,
    GTTrajectoryProvider,
    EpisodeActionBuffer,
)
from examples.baselines.openvla_oft.utils.make_env import make_eval_envs, parse_optional_bool_flag
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
from mani_skill.utils import common

from xskill.model.diffusion_model import get_resnet, replace_bn_with_gn
from xskill.model.encoder import ResnetConv


class XSkillAgent(nn.Module):

    def __init__(
        self,
        nets: nn.ModuleDict,
        noise_scheduler,
        xskill_model,
        proto_pipeline: nn.Module,
        cfg: dict,
        device: torch.device,
    ):
        super().__init__()
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.xskill_model = xskill_model
        self.proto_pipeline = proto_pipeline
        self.device = device

        # Config values
        self.obs_horizon = cfg["obs_horizon"]
        self.pred_horizon = cfg["pred_horizon"]
        self.action_dim = cfg["action_dim"]
        self.obs_dim = cfg["obs_dim"]
        self.vision_feature_dim = cfg["vision_feature_dim"]
        self.use_proto = cfg.get("use_proto", True)
        self.proto_horizon = cfg.get("proto_horizon", self.obs_horizon)
        self.proto_dim = cfg.get("proto_dim", 256)
        self.upsample_proto = cfg.get("upsample_proto", False)
        self.snap_frames = cfg.get("snap_frames", 10)
        self.num_diffusion_iters = cfg.get("num_diffusion_iters", 60)
        self.slide = getattr(xskill_model, "slide", 2)

        # Cached prototype snaps (set during prepare_for_eval)
        self._cached_proto_snap = None
        self._cached_predict_proto = None

    def _extract_traj_representation(self, images: torch.Tensor) -> torch.Tensor:
        """Extract trajectory prototypes using the pretrained stage1 model."""
        images = images.float()
        if images.max() > 1.0:
            images = images / 255.0
        b, t, c, h, w = images.shape
        images = self.proto_pipeline(images.reshape(b * t, c, h, w)).reshape(b, t, c, 112, 112)
        if t <= self.slide:
            raise ValueError(f"Need T > slide for proto extraction, got T={t}, slide={self.slide}")
        windows = [images[:, j: j + self.slide + 1] for j in range(t - self.slide)]
        im_q_processed = torch.cat(windows, dim=0)
        state_rep = self.xskill_model.encoder_q.get_state_representation(im_q_processed, None)
        traj_rep = self.xskill_model.encoder_q.get_traj_representation(state_rep)
        traj_rep = traj_rep.reshape(t - self.slide, b, -1).permute(1, 0, 2).contiguous()
        # Pad to match original time dimension
        if traj_rep.shape[1] < t:
            last = traj_rep[:, -1:, :].repeat(1, t - traj_rep.shape[1], 1)
            traj_rep = torch.cat([traj_rep, last], dim=1)
        return traj_rep

    def _sample_proto_snap(self, proto_seq: torch.Tensor) -> torch.Tensor:
        """Sample snap_frames evenly spaced prototypes from the sequence."""
        b, t, d = proto_seq.shape
        if t >= self.snap_frames:
            idx = torch.linspace(0, t - 1, steps=self.snap_frames, device=proto_seq.device).long()
            return proto_seq[:, idx, :]
        else:
            pad = proto_seq[:, -1:, :].repeat(1, self.snap_frames - t, 1)
            return torch.cat([proto_seq, pad], dim=1)

    @torch.no_grad()
    def prepare_for_eval(
        self,
        human_video: torch.Tensor,
        robot_obs: Dict,
        human_tokens: Optional[torch.Tensor] = None,
    ):
        """Prepare for evaluation by extracting and caching human prototypes.

        Args:
            human_video: (B, T, H, W, C) or (B, T, C, H, W) human demonstration video
            robot_obs: dict with 'states' and 'view_1' (not used for proto extraction)
            human_tokens: ignored (XSkill doesn't use language)
        """
        self.eval()
        human_video = human_video.to(self.device)

        if not self.use_proto:
            return

        # Ensure TCHW format: human_video from evaluate_processor is (B, T, H, W, C)
        if human_video.ndim == 5 and human_video.shape[-1] in (1, 3, 4):
            human_video = human_video.permute(0, 1, 4, 2, 3)  # -> (B, T, C, H, W)

        # Extract trajectory prototypes from human demo
        human_proto_seq = self._extract_traj_representation(human_video)
        # Sample snap frames
        self._cached_proto_snap = self._sample_proto_snap(human_proto_seq)

    @torch.no_grad()
    def get_action(self, obs_dict: Dict) -> torch.Tensor:
        """Get action sequence for the current observation.

        Args:
            obs_dict: dict with 'state' (B, obs_horizon, state_dim) and
                      'rgb' (B, obs_horizon, C, H, W) — already normalized.

        Returns:
            actions: (B, pred_horizon, action_dim)
        """
        states = obs_dict.get("state", obs_dict.get("states"))
        rgb = obs_dict.get("rgb", obs_dict.get("view_1"))

        B = states.shape[0]

        # Encode vision features
        # rgb may be (B, obs_horizon, C, H, W) or (B, C, H, W)
        if rgb.ndim == 5:
            nimage = rgb
        else:
            nimage = rgb.unsqueeze(1)

        image_features = self.nets["vision_encoder"](
            nimage.flatten(end_dim=1)
        )
        image_features = image_features.reshape(
            *nimage.shape[:2], -1
        )  # (B, obs_horizon, vision_feature_dim)

        # States: ensure (B, obs_horizon, obs_dim)
        if states.ndim == 2:
            states = states.unsqueeze(1)

        # Concatenate obs features
        obs_feature = torch.cat([image_features, states], dim=-1)

        # Build conditioning
        if self.use_proto and self._cached_proto_snap is not None:
            proto_snap = self._cached_proto_snap
            if proto_snap.shape[0] != B:
                proto_snap = proto_snap[:B]

            if "proto_pred_net" in self.nets:
                # Use proto_pred_net to predict the current robot prototype
                predict_proto = self.nets["proto_pred_net"](
                    obs_feature.flatten(start_dim=1),
                    proto_snap,
                )
                # predict_proto shape: (B, proto_dim)
                nproto = predict_proto.unsqueeze(1)  # (B, 1, proto_dim)
            else:
                # Fallback: use mean of proto_snap as conditioning
                nproto = proto_snap[:, -self.proto_horizon:, :]

            if self.upsample_proto and "upsample_proto_net" in self.nets:
                upsample_proto = self.nets["upsample_proto_net"](
                    nproto.flatten(start_dim=1)
                )
                upsample_proto = upsample_proto.reshape(B, self.proto_horizon, -1)
                obs_cond = torch.cat([
                    obs_feature.flatten(start_dim=1),
                    upsample_proto.flatten(start_dim=1),
                ], dim=1)
            else:
                obs_cond = torch.cat([
                    obs_feature.flatten(start_dim=1),
                    nproto.flatten(start_dim=1),
                ], dim=1)
        else:
            obs_cond = obs_feature.flatten(start_dim=1)

        # DDPM denoising loop
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        actions = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=self.device,
        )

        for k in self.noise_scheduler.timesteps:
            noise_pred = self.nets["noise_pred_net"](
                actions,
                k.expand(B).to(self.device),
                global_cond=obs_cond,
            )
            actions = self.noise_scheduler.step(noise_pred, k, actions).prev_sample

        return actions

    def clear_cache(self):
        """Clear cached prototypes."""
        self._cached_proto_snap = None
        self._cached_predict_proto = None



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


def is_env_already_evaluated(env_id, existing_results, num_episodes):
    prev = existing_results.get(env_id)
    if prev is None:
        return False
    if prev.get("num_episodes", 0) >= num_episodes:
        print(f"  Skipping {env_id}: already evaluated "
              f"({prev['num_episodes']} episodes, "
              f"success_once={prev.get('success_once_mean', 'N/A')})")
        return True
    print(f"  Partial eval for {env_id}: previous run had "
          f"{prev.get('num_episodes', '?')}/{num_episodes} episodes -- re-evaluating")
    return False


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


def make_eval_envs_with_level(base_env_name, level, num_envs, sim_backend,
                              env_kwargs, other_kwargs, video_dir, wrappers,
                              lr_mirror_enabled, lr_mirror_robot_pose_enabled):
    set_l_level(level)
    clean_env_kwargs = {k: v for k, v in env_kwargs.items() if k != "l_level"}
    envs = make_eval_envs(
        base_env_name, num_envs, sim_backend, clean_env_kwargs, other_kwargs,
        video_dir=video_dir, wrappers=wrappers, l_level=level, lr_mirror_enabled=lr_mirror_enabled,
        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
    )
    return envs


def build_xskill_eval_rgb_from_history(
    rgb_history_buffer,
    sample_idx: int,
    num_frames: int = 4,
    window: int = 30,
):
    """
    Follow the training-time XSkill logic: look back up to `window` frames
    within the current episode, sample `num_frames` frames uniformly, and
    repeat the first frame if there are not enough unique indices.

    rgb_history_buffer: list of Tensor, each with shape (B, C, H, W)
    sample_idx: current evaluation timestep, i.e. ts
    return: Tensor, shape = (B, num_frames, C, H, W)
    """
    ep_start_idx = 0

    start_idx = max(ep_start_idx, sample_idx - window)

    indices = np.linspace(start_idx, sample_idx, num_frames).round().astype(int)

    unique_indices = np.unique(indices)
    if len(unique_indices) < num_frames:
        first_idx = unique_indices[0]
        indices = np.concatenate(
            [unique_indices, [first_idx] * (num_frames - len(unique_indices))]
        )
    else:
        indices = unique_indices

    frames = [rgb_history_buffer[int(idx)] for idx in indices]
    return torch.stack(frames, dim=1)

def evaluate_with_task_encoder(
    n,
    agent,
    eval_envs,
    eval_kwargs,
    evaluate_processor,
    input_mode,
    progress_bar=True,
    # DTW / TSS trajectory metrics. Kept optional so normal success-rate eval still works.
    dtw_provider=None,
    traj_metrics=None,
    dtw_band_ratio: float = 0.15,
):
    env_id = eval_kwargs.get("env_id")
    delta_control = eval_kwargs.get("delta_control")
    pred_horizon = eval_kwargs.get("pred_horizon")
    temporal_agg = eval_kwargs.get("temporal_agg")
    max_timesteps = eval_kwargs.get("max_timesteps")
    device = eval_kwargs.get("device")
    sim_backend = eval_kwargs.get("sim_backend")

    if isinstance(eval_envs.single_observation_space, gymnasium.spaces.Box):
        action_dim = eval_envs.action_space.shape[-1]
    else:
        action_dim = (eval_envs.action_space["panda_wristcam-0"].shape[-1] +
                      eval_envs.action_space["panda_wristcam-1"].shape[-1])

    num_envs = eval_envs.num_envs

    if temporal_agg:
        query_frequency = 1
        all_time_actions = torch.zeros(
            [num_envs, max_timesteps, max_timesteps + pred_horizon, action_dim],
            device=device,
        )
    else:
        query_frequency = pred_horizon
        actions_to_take = torch.zeros([num_envs, pred_horizon, action_dim], device=device)

    agent.eval()

    # Prepare task inputs based on input_mode.
    human_video = None

    if input_mode in [InputMode.VIDEO_ONLY, InputMode.VIDEO_AND_LANGUAGE]:
        human_video = evaluate_processor.get_video(env_id, num_envs).to(device)

    # XSkill does not use language tokens — human_video is the only task input.

    # DTW / TSS helpers.
    # We sample one GT trajectory for this env. After each vectorized episode batch, we resample
    # so repeated episodes can compare against different demos when the provider supports it.
    compute_dtw = dtw_provider is not None
    if compute_dtw:
        if traj_metrics is None:
            traj_metrics = TrajectoryMetrics(band_ratio=dtw_band_ratio)

        gt_traj = dtw_provider.sample_gt_trajectory(env_id, seed=None, normalize=False)
        if gt_traj is None:
            print(f"  No GT trajectory for {env_id}; TSS/DTW skipped.")
            compute_dtw = False
        else:
            action_buf = EpisodeActionBuffer(num_envs=num_envs)
            success_steps = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        ts, eps_count = 0, 0

        rgb_history_buffer = []

        pbar_desc = f"Eval [{input_mode.value}]" + ("[+DTW]" if compute_dtw else "")
        pbar = tqdm(total=n, desc=pbar_desc, disable=not progress_bar, unit="ep")

        while eps_count < n:
            obs = {k: common.to_tensor(v, device) for k, v in obs.items()}
            if not delta_control:
                obs["state"], obs["rgb"] = evaluate_processor.normalize_state_rgb(
                    obs["state"], obs["rgb"], env_id,
                )

            # HWC -> CHW. XSkill vision encoder expects CHW.
            if obs["rgb"].ndim == 5 and obs["rgb"].shape[-1] in (3, 4):
                obs["rgb"] = obs["rgb"].permute(0, 1, 4, 2, 3)
            elif obs["rgb"].ndim == 4 and obs["rgb"].shape[-1] in (3, 4):
                obs["rgb"] = obs["rgb"].permute(0, 3, 1, 2)

            if obs["rgb"].ndim == 5:
                cur_rgb_frame = obs["rgb"][:, -1]      # (B, C, H, W)
            elif obs["rgb"].ndim == 4:
                cur_rgb_frame = obs["rgb"]             # (B, C, H, W)
            else:
                raise ValueError(f"Unexpected rgb shape: {obs['rgb'].shape}")

            rgb_history_buffer.append(cur_rgb_frame)

            obs["rgb"] = build_xskill_eval_rgb_from_history(
                rgb_history_buffer=rgb_history_buffer,
                sample_idx=ts,
                num_frames=4,
                window=30,
            )

            if ts == 0:
                robot_obs_for_task = {
                    "states": obs.get("state", obs.get("states")),
                    "view_1": obs.get("rgb", obs.get("view_1")),
                }
                agent.prepare_for_eval(
                    human_video=human_video,
                    robot_obs=robot_obs_for_task,
                )

            if ts % query_frequency == 0:
                action_seq = agent.get_action(obs)
                if action_seq.dim() == 2:
                    action_seq = action_seq.unsqueeze(0)

            if temporal_agg:
                all_time_actions[:, ts, ts:ts + pred_horizon] = action_seq
                actions_for_curr_step = all_time_actions[:, :, ts]
                actions_populated = torch.zeros(max_timesteps, dtype=torch.bool, device=device)
                actions_populated[max(0, ts + 1 - pred_horizon):ts + 1] = True
                actions_for_curr_step = actions_for_curr_step[:, actions_populated]
                k = 0.01
                exp_weights = torch.exp(-k * torch.arange(actions_for_curr_step.shape[1], device=device))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = exp_weights.unsqueeze(0).unsqueeze(-1).expand(num_envs, -1, -1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=1)
            else:
                if ts % query_frequency == 0:
                    actions_to_take = action_seq
                raw_action = actions_to_take[:, ts % query_frequency]

            _action = (evaluate_processor.denormalize_action(raw_action, env_id)
                       if not delta_control else raw_action)

            if sim_backend == "physx_cpu":
                _action = _action.cpu().numpy()

            if compute_dtw:
                action_buf.append(_action)

            action = {
                "panda_wristcam-0": _action[:, :8],
                "panda_wristcam-1": _action[:, 8:16],
            }

            obs, rew, terminated, truncated, info = eval_envs.step(action)
            ts += 1

            # Track first success step before episode truncates.
            # For vectorized envs this records per-env first success within the current episode batch.
            if compute_dtw and not truncated.any():
                cur_success = info.get("success")
                if cur_success is not None:
                    cur_success = torch.as_tensor(cur_success, dtype=torch.bool, device=device)
                    newly_success = (success_steps == -1) & cur_success
                    success_steps[newly_success] = ts

            if truncated.any():
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)

                # TSS / nDTW split by success and failure.
                if compute_dtw:
                    final_info = info.get("final_info", {})
                    last_success = final_info.get("success") if isinstance(final_info, dict) else None
                    if last_success is not None:
                        last_success = torch.as_tensor(last_success, dtype=torch.bool, device=device)
                        success_steps[(success_steps == -1) & last_success] = ts

                    pred_trajs = action_buf.get_and_reset()
                    success_steps_np = success_steps.cpu().numpy()
                    success_steps[:] = -1

                    T_gt = len(gt_traj)
                    for env_i, pred_traj in enumerate(pred_trajs):
                        ss = int(success_steps_np[env_i])
                        if ss >= 2:
                            # Successful episode: compare until first success step.
                            try:
                                m = traj_metrics.compute(pred_traj[:ss], gt_traj)
                                eval_metrics["tss_success"].append(np.array(m["tss"], dtype=np.float32))
                                eval_metrics["ndtw_success"].append(np.array(m["ndtw"], dtype=np.float32))
                            except Exception as exc:
                                print(f"  TSS(success) env {env_i}: {exc}")
                        else:
                            # Failed episode: trim predicted trajectory to GT length for fair comparison.
                            pred_trimmed = pred_traj[:T_gt]
                            if len(pred_trimmed) >= 2:
                                try:
                                    m = traj_metrics.compute(pred_trimmed, gt_traj)
                                    eval_metrics["tss_fail"].append(np.array(m["tss"], dtype=np.float32))
                                    eval_metrics["ndtw_fail"].append(np.array(m["ndtw"], dtype=np.float32))
                                except Exception as exc:
                                    print(f"  TSS(fail) env {env_i}: {exc}")

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

                rgb_history_buffer = []
                
                if compute_dtw:
                    new_gt = dtw_provider.sample_gt_trajectory(env_id, seed=None, normalize=False)
                    if new_gt is not None:
                        gt_traj = new_gt

        pbar.close()

    agent.train()

    # Stack non-empty metric lists. This avoids crashing when e.g. all episodes fail and
    # tss_success is naturally empty.
    for k in list(eval_metrics.keys()):
        if len(eval_metrics[k]) > 0:
            eval_metrics[k] = np.stack(eval_metrics[k])
        else:
            del eval_metrics[k]

    return eval_metrics



# =============================================================================
# Real-robot mock evaluation for XSkill
# =============================================================================

def _state_action_keys(state_type: str):
    if state_type == "eepos":
        return "observation.eepos_gripper_states", "action.eepos_gripper_actions"
    if state_type == "qpos":
        return "observation.qpos_gripper_states", "action.qpos_gripper_actions"
    if state_type == "mixpos":
        return "observation.eepos_gripper_states", "action.qpos_gripper_actions"
    raise ValueError(f"Unknown state_type: {state_type}")


def build_robot_normalizer(args) -> ActionNormalizer:
    """Build real-robot dataset normalizer for mock evaluation."""
    from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDatasetMetadata

    state_key, action_key = _state_action_keys(args.state_type)
    normalizer = ActionNormalizer()

    with open(args.robot_config, "r") as f:
        dataset_configs = json.load(f)

    for idx, ds_cfg in enumerate(dataset_configs):
        repo_id = ds_cfg["repo_id"]
        ds_root = os.path.join(args.robot_root, ds_cfg["root"])
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
        normalizer.add_dataset_stats(
            dataset_idx=idx,
            repo_id=repo_id,
            stats=meta.stats,
            state_key=state_key,
            action_key=action_key,
            single_arm=args.single_arm,
        )

    return normalizer


def _extract_dataset_idx(item: Dict[str, Any]) -> int:
    value = item.get("dataset_idx", 0)
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def _xskill_item_to_obs(
    item: Dict[str, Any],
    args,
    device: torch.device,
    rgb_history_buffer: List[torch.Tensor],
    ts: int,
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert one LeRobotRobotDataset item into XSkill agent obs.

    XSkill needs:
      state: (B, obs_horizon, state_dim)
      rgb:   (B, num_frames, C, H, W), built using the same history sampling
             logic as evaluate_with_task_encoder().
    """
    state = item["states"].to(device).float()

    expected_obs_horizon = int(
        model_config.get("obs_horizon", args.obs_horizon)
        if model_config is not None else args.obs_horizon
    )
    expected_obs_dim = int(
        model_config.get("obs_dim", args.state_dim)
        if model_config is not None else args.state_dim
    )

    if state.dim() == 1:
        # Prefer reshaping according to the training config.
        expected_total = expected_obs_horizon * expected_obs_dim

        if state.numel() == expected_total:
            state = state.reshape(expected_obs_horizon, expected_obs_dim)
        elif state.numel() % expected_obs_horizon == 0:
            # If the real data layout differs from the training config, reshape first,
            # then pad or truncate to the training dimension.
            real_dim = state.numel() // expected_obs_horizon
            state = state.reshape(expected_obs_horizon, real_dim)

            if real_dim < expected_obs_dim:
                pad = torch.zeros(
                    expected_obs_horizon,
                    expected_obs_dim - real_dim,
                    device=device,
                    dtype=state.dtype,
                )
                state = torch.cat([state, pad], dim=-1)
            elif real_dim > expected_obs_dim:
                state = state[:, :expected_obs_dim]
        else:
            raise ValueError(
                f"Cannot reshape state with numel={state.numel()} into "
                f"obs_horizon={expected_obs_horizon}, obs_dim={expected_obs_dim}"
            )

    elif state.dim() == 2:
        # Ensure both T and D align with the training config.
        if state.shape[0] < expected_obs_horizon:
            pad_t = state[:1].repeat(expected_obs_horizon - state.shape[0], 1)
            state = torch.cat([pad_t, state], dim=0)
        elif state.shape[0] > expected_obs_horizon:
            state = state[-expected_obs_horizon:]

        if state.shape[1] < expected_obs_dim:
            pad_d = torch.zeros(
                state.shape[0],
                expected_obs_dim - state.shape[1],
                device=device,
                dtype=state.dtype,
            )
            state = torch.cat([state, pad_d], dim=-1)
        elif state.shape[1] > expected_obs_dim:
            state = state[:, :expected_obs_dim]
    else:
        raise ValueError(f"Unexpected state shape: {tuple(state.shape)}")

    state = state.unsqueeze(0)  # (1, expected_obs_horizon, expected_obs_dim)

    rgb_views: List[torch.Tensor] = []
    for i in range(1, len(args.cameras) + 1):
        key = f"view_{i}"
        if key not in item:
            continue
        v = item[key].to(device).float()  # (T,C,H,W) or (C,H,W)
        if v.dim() == 4:
            v = v[-1]
        elif v.dim() != 3:
            raise ValueError(f"Unexpected image shape for {key}: {tuple(v.shape)}")
        if v.max() > 1.0:
            v = v / 255.0
        rgb_views.append(v)

    if rgb_views:
        cur_rgb = torch.cat(rgb_views, dim=0).unsqueeze(0)  # (1,C_total,H,W)
    else:
        H, W = tuple(args.image_size)
        cur_rgb = torch.zeros(1, 3, H, W, device=device)

    rgb_history_buffer.append(cur_rgb)
    rgb = build_xskill_eval_rgb_from_history(
        rgb_history_buffer=rgb_history_buffer,
        sample_idx=ts,
        num_frames=args.xskill_eval_num_frames,
        window=args.xskill_eval_window,
    )

    return {"state": state, "rgb": rgb}


def _write_xskill_mock_report(
    pred_actions: List[np.ndarray],
    gt_actions: List[np.ndarray],
    per_step_mse: List[float],
    per_dim_sq: List[np.ndarray],
    dataset_indices: List[int],
    label: str,
    debug_dir: Path,
    checkpoint_path: str,
):
    preds = np.stack(pred_actions)
    gts = np.stack(gt_actions)
    mses = np.asarray(per_step_mse)
    dim_sq = np.stack(per_dim_sq)
    n, action_dim = preds.shape

    debug_dir.mkdir(parents=True, exist_ok=True)

    csv_path = debug_dir / "per_step_mse.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "dataset_idx", "mse"] + [f"dim_{i}_sq_err" for i in range(action_dim)])
        for i, (dataset_idx, mse, sq) in enumerate(zip(dataset_indices, per_step_mse, per_dim_sq)):
            writer.writerow([i, dataset_idx, round(float(mse), 8)] + [round(float(x), 8) for x in sq])

    summary = {
        "label": label,
        "checkpoint_path": checkpoint_path,
        "n_frames": n,
        "action_dim": action_dim,
        "overall_mse_mean": float(mses.mean()),
        "overall_mse_std": float(mses.std()),
        "overall_mse_max": float(mses.max()),
        "per_dim_mse_mean": dim_sq.mean(axis=0).tolist(),
        "per_dim_mse_std": dim_sq.std(axis=0).tolist(),
        "per_dim_mse_max": dim_sq.max(axis=0).tolist(),
        "timestamp": datetime.now().isoformat(),
    }
    json_path = debug_dir / "mock_debug_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    frames = np.arange(n)
    cols = 2
    rows = (action_dim + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 2.6 * rows), squeeze=False)
    for dim in range(action_dim):
        r, c = divmod(dim, cols)
        ax = axes[r][c]
        ax.plot(frames, gts[:, dim], linewidth=1.3, label="GT")
        ax.plot(frames, preds[:, dim], linewidth=1.0, linestyle="--", label="Pred")
        ax.set_title(f"dim {dim}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for dim in range(action_dim, rows * cols):
        r, c = divmod(dim, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    action_plot = debug_dir / "action_comparison.png"
    fig.savefig(action_plot, dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(frames, mses, linewidth=1.2)
    ax2.axhline(mses.mean(), linestyle="--", label=f"mean={mses.mean():.4f}")
    ax2.set_title(f"XSkill real-robot mock action MSE - {label}")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("MSE")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    mse_plot = debug_dir / "mse_over_time.png"
    fig2.tight_layout()
    fig2.savefig(mse_plot, dpi=120)
    plt.close(fig2)

    print("[XSKILL-ROBOT-MOCK] report")
    print(f"  frames: {n}")
    print(f"  mse mean/std/max: {summary['overall_mse_mean']:.6f} / {summary['overall_mse_std']:.6f} / {summary['overall_mse_max']:.6f}")
    print(f"  csv: {csv_path}")
    print(f"  json: {json_path}")
    print(f"  action plot: {action_plot}")
    print(f"  mse plot: {mse_plot}")


def run_xskill_robot_mock(agent, model_config, evaluate_processor, args, device: torch.device) -> None:
    """
    Replay real-robot LeRobot frames and compare XSkill predicted first action
    against GT action after denormalising with real-robot stats.
    """
    from examples.baselines.lerobot_dataset.lerobot_robot_dataset import (
        LeRobotRobotDataset,
        LeRobotRobotDataConfig,
    )

    repo_id = args.mock_repo_id
    if not repo_id:
        raise ValueError("--mock-repo-id is required for --mock-source robot")

    debug_dir = Path(args.mock_debug_dir)
    robot_normalizer = build_robot_normalizer(args)

    print(f"\n{'=' * 70}")
    print(f"  XSkill ROBOT MOCK")
    print(f"{'=' * 70}")
    print(f"  repo_id    : {repo_id}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  debug_dir  : {debug_dir.resolve()}")
    print()

    robot_ds = LeRobotRobotDataset(
        LeRobotRobotDataConfig(
            root=args.robot_root,
            split=args.robot_split,
            image_size=tuple(model_config.get("image_size", args.image_size)),
            state_type=model_config.get("state_type", args.state_type),
            include_depth=model_config.get("include_depth", args.include_depth),
            cameras=model_config.get("cameras", args.cameras),
            single_arm=model_config.get("single_arm", args.single_arm),
            horizon=model_config.get("pred_horizon", args.pred_horizon),
            obs_horizon=model_config.get("obs_horizon", args.obs_horizon),
            dataset_file=args.robot_config,
            task_description_file=args.robot_task_desc,
            normalization_method=args.normalization_method,
            enable_augmentation=False,
        )
    )

    episode_indices: List[int] = []
    for i in range(len(robot_ds)):
        item = robot_ds[i]
        if str(item.get("repo_id", "")) == repo_id:
            episode_indices.append(i)
        if len(episode_indices) >= args.mock_num_frames:
            break

    if not episode_indices:
        print(f"[XSKILL-ROBOT-MOCK] no frames matched repo_id={repo_id}; fallback to first N frames")
        episode_indices = list(range(min(args.mock_num_frames, len(robot_ds))))

    print(f"[XSKILL-ROBOT-MOCK] dataset frames={len(robot_ds)}, replay frames={len(episode_indices)}")

    input_mode = InputMode(args.input_mode)
    task_id = args.mock_task_id or repo_id

    human_video = None
    if input_mode in (InputMode.VIDEO_ONLY, InputMode.VIDEO_AND_LANGUAGE):
        human_video = evaluate_processor.get_video(task_id, 1).to(device)
        print(f"[XSKILL-ROBOT-MOCK] human_video={tuple(human_video.shape)} from task_id={task_id}")

    rgb_history_buffer: List[torch.Tensor] = []
    first_item = robot_ds[episode_indices[0]]
    first_obs = _xskill_item_to_obs(first_item, args, device, rgb_history_buffer, ts=0, model_config=model_config)

    agent.prepare_for_eval(
        human_video=human_video,
        robot_obs=first_obs,
    )
    print("[XSKILL-ROBOT-MOCK] task encoded. Running inference...")

    pred_actions: List[np.ndarray] = []
    gt_actions: List[np.ndarray] = []
    per_step_mse: List[float] = []
    per_dim_sq: List[np.ndarray] = []
    dataset_indices: List[int] = []

    rgb_history_buffer = []

    with torch.no_grad():
        for step, frame_idx in enumerate(episode_indices):
            item = robot_ds[frame_idx]
            dataset_idx = _extract_dataset_idx(item)
            obs = _xskill_item_to_obs(item, args, device, rgb_history_buffer, ts=step, model_config=model_config)

            action_seq = agent.get_action(obs)
            if action_seq.dim() == 2:
                action_seq = action_seq.unsqueeze(0)

            pred_raw = action_seq[:, 0]
            gt_raw = item["actions"][0:1].to(device)

            pred_den = robot_normalizer.denormalize_action(
                pred_raw,
                dataset_idx=dataset_idx,
                method=args.normalization_method,
            )
            gt_den = robot_normalizer.denormalize_action(
                gt_raw,
                dataset_idx=dataset_idx,
                method=args.normalization_method,
            )

            p = pred_den[0].detach().cpu().numpy()
            g = gt_den[0].detach().cpu().numpy()
            sq_err = (p - g) ** 2
            mse = float(sq_err.mean())

            pred_actions.append(p)
            gt_actions.append(g)
            per_step_mse.append(mse)
            per_dim_sq.append(sq_err)
            dataset_indices.append(dataset_idx)

            if step < 3 or step % args.mock_log_interval == 0:
                print(
                    f"[XSKILL-ROBOT-MOCK] step={step:04d} dataset_idx={dataset_idx} "
                    f"mse={mse:.6f} pred[:4]={p[:4].round(4).tolist()} gt[:4]={g[:4].round(4).tolist()}"
                )

    agent.clear_cache()
    _write_xskill_mock_report(
        pred_actions=pred_actions,
        gt_actions=gt_actions,
        per_step_mse=per_step_mse,
        per_dim_sq=per_dim_sq,
        dataset_indices=dataset_indices,
        label=repo_id,
        debug_dir=debug_dir,
        checkpoint_path=args.checkpoint,
    )


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="XSkill Evaluation")

    parser.add_argument("--eval-config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to stage2 checkpoint (.pt)")
    parser.add_argument("--hydra-config", type=str, required=True,
                        help="Path to hydra_config.yaml saved during stage2 training")

    # Input mode
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

    # Model parameters (defaults from xskill config, overridden by hydra config)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=18)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--pred-horizon", type=int, default=16)
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--cameras", type=str, nargs="+", default=["zed2i"])
    parser.add_argument("--include-depth", action="store_true", default=False)
    parser.add_argument("--state-type", type=str, default="qpos")
    parser.add_argument("--single-arm", action="store_true", default=False)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-text-len", type=int, default=500)
    parser.add_argument("--num-tasks", type=int, default=200)
    parser.add_argument("--eval_lr_mirror", default="auto", choices=["auto", "true", "false"],
                        help="Override tabletop left-right mirror during eval.")
    parser.add_argument("--eval_lr_mirror_robot_pose", default="false", choices=["auto", "true", "false"],
                        help="Override robot pose swapping under mirror during eval. Default 'false' keeps agents unswapped.")
    # XSkill-specific
    parser.add_argument("--temporal-agg", action="store_true", default=False)
    parser.add_argument("--task-num-frames", type=int, default=10)

    # Real-robot mock evaluation
    parser.add_argument("--mock", action="store_true",
                        help="Replay real-robot dataset frames and compare predicted action with GT.")
    parser.add_argument("--mock-source", type=str, default="robot", choices=["robot"],
                        help="Currently only robot mock is implemented for XSkill.")
    parser.add_argument("--robot-root", type=str, default=None)
    parser.add_argument("--robot-config", type=str,
                        default="examples/baselines/lerobot_dataset/config/robot_config.json")
    parser.add_argument("--robot-task-desc", type=str,
                        default="examples/baselines/lerobot_dataset/task_desc/robot_desc.json")
    parser.add_argument("--robot-split", type=str, default="train")
    parser.add_argument("--mock-repo-id", type=str, default=None)
    parser.add_argument("--mock-task-id", type=str, default=None,
                    help="[robot mock] task id used for human-video prototype lookup. Defaults to --mock-repo-id.")
    parser.add_argument("--mock-num-frames", type=int, default=200)
    parser.add_argument("--mock-debug-dir", type=str, default="mock_debug/xskill")
    parser.add_argument("--mock-log-interval", type=int, default=20)
    parser.add_argument("--normalization-method", type=str, default="bounds_q99")
    parser.add_argument("--xskill-eval-num-frames", type=int, default=4,
                        help="Number of RGB history frames fed to XSkill, matching training/eval logic.")
    parser.add_argument("--xskill-eval-window", type=int, default=30,
                        help="History window used by build_xskill_eval_rgb_from_history.")


    # DTW trajectory metrics
    dtw_group = parser.add_mutually_exclusive_group()
    dtw_group.add_argument("--compute-dtw", dest="compute_dtw", action="store_true", default=True,
                           help="Compute TSS/nDTW trajectory metrics when GT trajectories are available")
    dtw_group.add_argument("--no-compute-dtw", dest="compute_dtw", action="store_false",
                           help="Disable TSS/nDTW trajectory metrics")
    parser.add_argument("--dtw-band-ratio", type=float, default=0.15)

    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/xskill_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shader", type=str, default="rt-fast")

    return parser.parse_args()


# =============================================================================
# Config / checkpoint loading
# =============================================================================

def load_eval_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    if config_path.suffix != ".txt":
        raise ValueError(f"Only .txt config files are supported. Got: {config_path.suffix}")
    with open(config_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
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


def _load_xskill_stage1_model(cfg, device):
    """Load the pretrained stage1 XSkill model for prototype extraction."""
    from pathlib import Path as _Path

    config_path = _Path(cfg["pretrain_path"]) / ".hydra" / "config.yaml"
    print(f"  Loading stage1 config from: {config_path}")
    exp_cfg = OmegaConf.load(str(config_path))
    model = hydra.utils.instantiate(exp_cfg.Model).to(device)

    ckpt_id = int(cfg["pretrain_ckpt"])
    candidates = [
        _Path(cfg["pretrain_path"]) / f"{ckpt_id}.ckpt",
        _Path(cfg["pretrain_path"]) / f"{ckpt_id:02d}.ckpt",
        _Path(cfg["pretrain_path"]) / f"{ckpt_id:04d}.ckpt",
        _Path(cfg["pretrain_path"]) / f"epoch={ckpt_id}.ckpt",
        _Path(cfg["pretrain_path"]) / f"epoch={ckpt_id:02d}.ckpt",
        _Path(cfg["pretrain_path"]) / f"epoch={ckpt_id:04d}.ckpt",
    ]
    ckpt_path = None
    for c in candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        all_ckpts = sorted(_Path(cfg["pretrain_path"]).glob("*.ckpt"))
        if not all_ckpts:
            raise FileNotFoundError(f"No checkpoint found under {cfg['pretrain_path']}")
        ckpt_path = all_ckpts[-1]
        print(f"  Checkpoint {cfg['pretrain_ckpt']} not found, using latest: {ckpt_path}")

    print(f"  Loading stage1 checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  Stage1 model loaded and frozen")
    return model


def load_agent_from_checkpoint(checkpoint_path: str, hydra_config_path: str, args, device):
    """Load XSkill agent from a stage2 checkpoint.

    Returns:
        agent: XSkillAgent instance
        model_config: dict of model config values for the evaluate processor
    """
    # Load hydra config from training
    cfg = OmegaConf.load(hydra_config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Build network components (same as training)
    obs_horizon = int(cfg_dict.get("obs_horizon", args.obs_horizon))
    obs_dim = int(cfg_dict.get("obs_dim", args.state_dim))
    action_dim = int(cfg_dict.get("action_dim", args.action_dim))
    vision_feature_dim = int(cfg_dict.get("vision_feature_dim", 64))
    use_proto = bool(cfg_dict.get("use_proto", True))
    proto_horizon = int(cfg_dict.get("proto_horizon", obs_horizon))
    proto_dim = int(cfg_dict.get("proto_dim", 256))
    upsample_proto = bool(cfg_dict.get("upsample_proto", False))

    # Create vision encoder
    if vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18")
    else:
        vision_encoder = ResnetConv(embedding_size=vision_feature_dim)
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # Compute global_cond_dim (same logic as training)
    if use_proto and upsample_proto:
        upsample_out_size = cfg_dict.get("upsample_proto_net", {}).get("out_size", 256)
        global_cond_dim = (vision_feature_dim * obs_horizon +
                           obs_dim * obs_horizon +
                           proto_horizon * upsample_out_size)
    elif use_proto:
        global_cond_dim = (vision_feature_dim * obs_horizon +
                           obs_dim * obs_horizon +
                           proto_horizon * proto_dim)
    else:
        global_cond_dim = vision_feature_dim * obs_horizon + obs_dim * obs_horizon

    # Create noise prediction network
    noise_pred_net = hydra.utils.instantiate(
        cfg.noise_pred_net, global_cond_dim=global_cond_dim,
    )

    nets = nn.ModuleDict({
        "vision_encoder": vision_encoder,
        "noise_pred_net": noise_pred_net,
    })

    # Add prototype networks if enabled
    if use_proto:
        proto_pred_net = hydra.utils.instantiate(
            cfg.proto_pred_net,
            input_dim=vision_feature_dim * obs_horizon + obs_dim * obs_horizon,
        )
        nets["proto_pred_net"] = proto_pred_net

        if upsample_proto:
            upsample_proto_net = hydra.utils.instantiate(cfg.upsample_proto_net)
            nets["upsample_proto_net"] = upsample_proto_net

    # Load checkpoint weights
    print(f"  Loading stage2 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "nets" in ckpt:
        state_dict = ckpt["nets"]
    else:
        state_dict = ckpt
    nets.load_state_dict(state_dict)
    nets.to(device)
    nets.eval()
    print("  Stage2 networks loaded")

    # Load noise scheduler
    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)

    # Load stage1 model for prototype extraction
    xskill_model = _load_xskill_stage1_model(cfg_dict, device)

    # Build proto preprocessing pipeline
    proto_pipeline = nn.Sequential(
        Tr.CenterCrop((112, 112)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ).to(device)

    # Build config dict for agent
    agent_cfg = {
        "obs_horizon": obs_horizon,
        "pred_horizon": int(cfg_dict.get("pred_horizon", args.pred_horizon)),
        "action_dim": action_dim,
        "obs_dim": obs_dim,
        "vision_feature_dim": vision_feature_dim,
        "use_proto": use_proto,
        "proto_horizon": proto_horizon,
        "proto_dim": proto_dim,
        "upsample_proto": upsample_proto,
        "snap_frames": int(cfg_dict.get("snap_frames", 10)),
        "num_diffusion_iters": int(cfg_dict.get("num_diffusion_iters", 60)),
    }

    agent = XSkillAgent(
        nets=nets,
        noise_scheduler=noise_scheduler,
        xskill_model=xskill_model,
        proto_pipeline=proto_pipeline,
        cfg=agent_cfg,
        device=device,
    )
    agent.to(device)

    model_config = {
        "pred_horizon": agent_cfg["pred_horizon"],
        "obs_horizon": agent_cfg["obs_horizon"],
        "obs_dim": obs_dim,
        "vision_feature_dim": vision_feature_dim,
        "cameras": list(cfg_dict.get("cameras", args.cameras)),
        "include_depth": bool(cfg_dict.get("include_depth", args.include_depth)),
        "task_num_frames": int(cfg_dict.get("snap_frames", args.task_num_frames)),
        "image_size": list(cfg_dict.get("image_size", args.image_size)),
        "state_type": str(cfg_dict.get("state_type", args.state_type)),
        "single_arm": bool(cfg_dict.get("single_arm", args.single_arm)),
        "vocab_size": args.vocab_size,
        "max_text_len": args.max_text_len,
    }

    print(f"  Loaded XSkill agent (use_proto={use_proto}, "
          f"obs_horizon={obs_horizon}, pred_horizon={agent_cfg['pred_horizon']})")

    return agent, model_config


def save_results_to_json(results: List[Dict], json_path: Path):
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)


def evaluate_single_env(
    agent,
    env_id,
    args,
    evaluate_processor,
    output_dir,
    model_config,
    input_mode,
    dtw_provider=None,
    traj_metrics=None,
):
    base_env_name = extract_base_env_name(env_id)
    level = extract_level(env_id)

    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {env_id}")
    print(f"   Base name: {base_env_name}")
    print(f"   Level: {level}")
    print(f"   Input Mode: {input_mode.value}")
    print(f"{'=' * 60}")

    eval_envs = None
    try:
        # Use model_config here, not only args, so eval follows the checkpoint/training config.
        obs_mode = "rgbd" if model_config.get("include_depth", args.include_depth) else "rgb"
        shader = getattr(args, "shader", "rt-fast")
        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode="dense",
            obs_mode=obs_mode,
            render_mode="rgb_array",
            max_episode_steps=args.max_episode_steps,
            sensor_configs=dict(shader_pack=shader),
            human_render_camera_configs=dict(shader_pack=shader),
        )
        other_kwargs = dict(obs_horizon=model_config["obs_horizon"])

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
            pred_horizon=model_config["pred_horizon"],
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
            dtw_provider=dtw_provider,
            traj_metrics=traj_metrics,
            dtw_band_ratio=args.dtw_band_ratio,
        )

        results = {
            "env_id": env_id,
            "level": level,
            "input_mode": input_mode.value,
            "num_episodes": args.num_episodes,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status": "success",
        }
        for k, v in eval_metrics.items():
            results[f"{k}_mean"] = float(v.mean())
            results[f"{k}_std"] = float(v.std())

        # Weighted TSS: success rate * successful-trajectory similarity.
        # This follows the DP eval convention and keeps failures from inflating the score.
        if "success_once_mean" in results and "tss_success_mean" in results:
            results["wtss_mean"] = float(results["success_once_mean"] * results["tss_success_mean"])

        print(f"\n  Results [{input_mode.value}]:")
        for k in ["success_once", "success_at_end", "return", "tss_success", "ndtw_success", "tss_fail", "ndtw_fail"]:
            if f"{k}_mean" in results:
                print(f"   {k}: {results[f'{k}_mean']:.4f} +/- {results[f'{k}_std']:.4f}")
        if "wtss_mean" in results:
            print(f"   wtss: {results['wtss_mean']:.4f}")

        return results

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "env_id": env_id,
            "level": level,
            "input_mode": input_mode.value,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status": "error",
            "error": str(e),
        }

    finally:
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
    print("  XSKILL EVALUATION")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    input_mode = InputMode(args.input_mode)
    print(f"\n  Input Mode: {input_mode.value}")

    # Load eval configuration only for normal sim eval.
    if args.mock:
        eval_name = "xskill_robot_mock"
        pending_environments = []
        skipped_count = 0
    else:
        if args.eval_config is None:
            raise ValueError("--eval-config is required for normal sim evaluation. Use --mock for real-robot mock.")
        print(f"\n  Loading eval configuration: {args.eval_config}")
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

        print(f"\n   Already evaluated (skipping): {skipped_count}")
        print(f"   Pending evaluation:            {len(pending_environments)}")

        if not pending_environments:
            print("\n  All environments already evaluated. Nothing to do.")
            return

    # Load agent
    device = torch.device(args.device)
    agent, model_config = load_agent_from_checkpoint(
        args.checkpoint, args.hydra_config, args, device
    )

    # Setup evaluation processor
    print("\n  Setting up evaluation processor...")
    evaluate_processor_config = HumanVideoSimEvaluateProcessorConfig(
        human_root=args.human_root,
        human_split="train",
        human_dataset_file=args.human_config,
        human_task_description_file=args.human_task_desc,
        human_cameras=model_config.get("cameras", args.cameras),
        human_include_depth=model_config.get("include_depth", args.include_depth),
        human_num_frames=model_config.get("task_num_frames", args.task_num_frames),
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
    print("  Evaluate processor ready")

    if args.mock:
        if args.robot_root is None:
            raise ValueError("--robot-root is required when using --mock")
        run_xskill_robot_mock(agent, model_config, evaluate_processor, args, device)
        return

    # Build GT trajectory provider for TSS/nDTW.
    dtw_provider = None
    traj_metrics = None
    if args.compute_dtw:
        print("\n  Building GT trajectory provider for TSS/nDTW...")
        try:
            action_key = (
                "action.qpos_gripper_actions"
                if model_config.get("state_type", args.state_type) in ("qpos", "mixpos")
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
            print(f"  GT trajectories loaded for {len(dtw_provider._episodes)} task(s).")
        except Exception as exc:
            print(f"  Could not build GTTrajectoryProvider: {exc}")
            print("  Continue without TSS/nDTW.")
            dtw_provider = None
            traj_metrics = None

    # Prepare incremental log file
    results_json = output_dir / f"{eval_name}_{input_mode.value}_{timestamp}.json"
    print(f"\n  Incremental log: {results_json}")

    # Run evaluations
    print("\n" + "=" * 80)
    print(f"  STARTING EVALUATIONS [{input_mode.value}]")
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
        status_icon = "OK" if result.get("status") == "success" else "FAIL"
        success_str = (
            f"{result['success_once_mean']:.4f}"
            if result.get("status") == "success" and "success_once_mean" in result
            else result.get("error", "N/A")
        )
        tss_str = f" | wTSS={result['wtss_mean']:.4f}" if result.get("wtss_mean") is not None else ""
        print(f"{status_icon} [{i + 1}/{len(pending_environments)}] {env_id} | "
              f"success_once={success_str}{tss_str} | log updated")

    # Save final CSV summary
    print("\n" + "=" * 80)
    print("  SAVING FINAL SUMMARY")
    print("=" * 80)

    df = pd.DataFrame(all_results)
    results_csv = output_dir / f"{eval_name}_{input_mode.value}_{timestamp}.csv"
    df.to_csv(results_csv, index=False)
    print(f"  JSON log : {results_json}")
    print(f"  CSV summary: {results_csv}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"  RESULTS SUMMARY [{input_mode.value}]")
    print("=" * 80)

    if "success_once_mean" in df.columns:
        print(f"\n{'Environment':<45} {'Level':<6} {'Mode':<15} {'Success':<10} {'wTSS':<10}")
        print("-" * 96)
        for _, row in df.iterrows():
            env_id = row["env_id"]
            level = row.get("level", "L0")
            mode = row.get("input_mode", "unknown")
            if row.get("status") == "error":
                print(f"{env_id:<45} {level:<6} {mode:<15} {'ERROR':<10} {'-':<10}")
            else:
                success = row.get("success_once_mean", 0.0)
                wtss = row.get("wtss_mean", np.nan)
                wtss_str = f"{wtss:.4f}" if pd.notna(wtss) else "-"
                print(f"{env_id:<45} {level:<6} {mode:<15} {success:<10.4f} {wtss_str:<10}")

        print("-" * 96)
        valid_df = df[df["success_once_mean"].notna()]
        if len(valid_df) > 0:
            avg_success = valid_df["success_once_mean"].mean()
            avg_wtss = valid_df["wtss_mean"].dropna().mean() if "wtss_mean" in valid_df.columns else np.nan
            avg_wtss_str = f"{avg_wtss:.4f}" if pd.notna(avg_wtss) else "-"
            print(f"{'OVERALL AVERAGE':<45} {'ALL':<6} {input_mode.value:<15} "
                  f"{avg_success:<10.4f} {avg_wtss_str:<10}")

    print(f"\n  EVALUATION COMPLETE  (skipped {skipped_count}, evaluated {len(all_results)})")


if __name__ == "__main__":
    main()
