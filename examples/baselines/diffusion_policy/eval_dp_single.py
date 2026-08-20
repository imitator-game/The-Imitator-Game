"""
Single-Task Diffusion Policy Evaluation (LeRobot)
==================================================
Evaluates a checkpoint trained by train_dp_single.py.
No task encoder — obs + state only.

Aligned with eval_dp_imitator.py / eval_act_single.py:
  - Subprocess-safe L-level handling (set_l_level / clear_l_level)
  - Correct L3 base-env-name extraction
  - JSON-based skip / resume
  - Incremental result saving

Usage:
  python eval_dp_single.py \\
    --checkpoint runs/single_task/dp/L0_TwoRobotCleanCup-v1/checkpoint_0200000.pt \\
    --sub-config  config/sub_configs/L0_TwoRobotCleanCup-v1.json \\
    --sim-root   /data/sim \\
    --output-dir eval_results/single_task/dp/L0_TwoRobotCleanCup-v1
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from examples.baselines.diffusion_policy.diffusion_policy.single_task_eval_utils import (
    extract_level,
    extract_base_env_name,
    set_l_level,
    clear_l_level,
    load_existing_results,
    is_env_already_evaluated,
    save_results_incremental,
    print_summary,
)

from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer
from examples.baselines.lerobot_dataset.trajectory_metrics import (
    TrajectoryMetrics,
    GTTrajectoryProvider,
    EpisodeActionBuffer,
)
from examples.baselines.diffusion_policy.diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
from mani_skill.utils import common

from train_dp_single import DPSingleTaskAgent, TrainingArgs

L0_L3_utils.set_lr_mirror_enabled(False)
L0_L3_utils.set_lr_mirror_robot_pose_enabled(False)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

_EVAL_TAG = "dp_single"


# =============================================================================
# Normalizer
# =============================================================================

class SingleTaskNormalizer:
    """
    Loads normalization statistics from disk via LeRobotDatasetMetadata —
    identical to evaluate_processor._load_sim_normalizer().

    Args:
        sub_config:  path to the sub-config JSON  [{repo_id, root, ...}, ...]
        sim_root:    root directory for all sim datasets
        state_type:  "qpos" | "eepos" | "mixpos"  (read from train_args)
        single_arm:  bool (read from train_args)
    """

    _STATE_ACTION_KEYS = {
        "qpos":   ("observation.qpos_gripper_states",  "action.qpos_gripper_actions"),
        "eepos":  ("observation.eepos_gripper_states", "action.eepos_gripper_actions"),
        "mixpos": ("observation.eepos_gripper_states", "action.qpos_gripper_actions"),
    }

    def __init__(
        self,
        sub_config: str,
        sim_root: str,
        state_type: str = "qpos",
        single_arm: bool = False,
        normalization_method: str = "bounds_q99",
    ):
        import json, os
        from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDatasetMetadata

        self.normalizer = ActionNormalizer()
        self.method = normalization_method

        state_key, action_key = self._STATE_ACTION_KEYS.get(
            state_type,
            ("observation.qpos_gripper_states", "action.qpos_gripper_actions"),
        )

        with open(sub_config) as f:
            dataset_configs = json.load(f)

        for idx, ds_cfg in enumerate(dataset_configs):
            repo_id = ds_cfg.get("repo_id")
            ds_root = os.path.join(sim_root, ds_cfg.get("root"))
            meta    = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
            self.normalizer.add_dataset_stats(
                dataset_idx=idx,
                repo_id=repo_id,
                stats=meta.stats,
                state_key=state_key,
                action_key=action_key,
                single_arm=single_arm,
            )

        # repo_id → dataset_idx (mirrors evaluate_processor.repo_id_to_dataset_idx)
        self._repo_to_idx: dict = {
            ds_cfg.get("repo_id"): idx
            for idx, ds_cfg in enumerate(dataset_configs)
        }

    def resolve_idx(self, env_id: str) -> int:
        """Return dataset_idx for *env_id*, stripping Lx_ prefix as fallback."""
        if env_id in self._repo_to_idx:
            return self._repo_to_idx[env_id]
        if "_" in env_id:
            base = env_id.split("_", 1)[1]
            for repo_id, idx in self._repo_to_idx.items():
                if repo_id == base or repo_id.endswith(base):
                    return idx
        return 0  # single-dataset fallback

    def normalize_state(self, state: torch.Tensor, dataset_idx: int = 0) -> torch.Tensor:
        return self.normalizer.normalize_state(state, dataset_idx, method=self.method)

    def normalize_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        mean = IMAGENET_MEAN.to(rgb.device)[None, :, None, None]
        std  = IMAGENET_STD.to(rgb.device)[None, :, None, None]
        return (rgb - mean) / std

    def denormalize_action(self, action: torch.Tensor, dataset_idx: int = 0) -> torch.Tensor:
        return self.normalizer.denormalize_action(action, dataset_idx, method=self.method)


# =============================================================================
# Checkpoint loading
# =============================================================================

def load_checkpoint(ckpt_path: str, args_override, device: torch.device):
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stored = ckpt.get("args", {})

    ta = TrainingArgs()
    for k, v in stored.items():
        if hasattr(ta, k):
            setattr(ta, k, v)
    for k in ["state_type", "pred_horizon", "obs_horizon", "num_inference_steps"]:
        cli_val = getattr(args_override, k, None)
        if cli_val is not None:
            setattr(ta, k, cli_val)

    agent = DPSingleTaskAgent(ta, device)
    agent.load_state_dict(ckpt["agent_state_dict"], strict=True)
    agent.eval()
    return agent, ta


# =============================================================================
# Core evaluation loop
# =============================================================================

def evaluate_episode(
    n: int,
    agent,
    eval_envs,
    normalizer: SingleTaskNormalizer,
    args,
    device: torch.device,
    train_args: TrainingArgs,
    dtw_provider: Optional[GTTrajectoryProvider] = None,
    traj_metrics: Optional[TrajectoryMetrics]    = None,
    env_id: str = "",
) -> Dict[str, np.ndarray]:
    query_frequency = train_args.pred_horizon
    num_envs        = eval_envs.num_envs

    compute_dtw = dtw_provider is not None
    gt_traj     = None
    if compute_dtw:
        gt_traj = dtw_provider.sample_gt_trajectory(env_id, seed=None)
        if gt_traj is None:
            print(f"⚠️  No GT trajectory for {env_id}; DTW skipped.")
            compute_dtw = False
        else:
            action_buf = EpisodeActionBuffer(num_envs=num_envs)
        # Track first success step per env (-1 = not yet succeeded)
        success_steps = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    agent.eval()
    eval_metrics    = defaultdict(list)
    obs, info       = eval_envs.reset()
    ts, eps_count   = 0, 0
    actions_to_take = None

    pbar = tqdm(
        total=n,
        desc=f"Eval {env_id}{'[+DTW]' if compute_dtw else ''}",
        unit="ep",
    )

    with torch.no_grad():
        while eps_count < n:
            obs = {k: common.to_tensor(v, device) for k, v in obs.items()}

            rgb_obs = obs.get("rgb", obs.get("view_1"))
            if rgb_obs is None:
                raise KeyError(f"obs has neither 'rgb' nor 'view_1'. Keys: {list(obs)}")
            state_obs = obs.get("state", obs.get("states"))
            if state_obs is None:
                raise KeyError(f"obs has neither 'state' nor 'states'. Keys: {list(obs)}")

            rgb   = normalizer.normalize_rgb(rgb_obs)
            state = normalizer.normalize_state(state_obs)

            if ts % query_frequency == 0:
                actions_to_take = agent.get_action(rgb, state)

            raw_action = actions_to_take[:, ts % query_frequency]
            _action    = normalizer.denormalize_action(raw_action)

            if args.sim_backend == "physx_cpu":
                _action = _action.cpu().numpy()

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
                    for fi in info["final_info"]:
                        for k, v in fi["episode"].items():
                            eval_metrics[k].append(v)

                if compute_dtw:
                    _fi = info.get("final_info", {})
                    _ls = (_fi.get("success") if isinstance(_fi, dict) else None)
                    if _ls is not None:
                        _ls_t = torch.as_tensor(_ls, dtype=torch.bool, device=device)
                        success_steps[(success_steps == -1) & _ls_t] = ts
                    pred_trajs = action_buf.get_and_reset()
                    ss_np      = success_steps.cpu().numpy()
                    success_steps[:] = -1
                    T_gt = len(gt_traj)
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
                    new_gt = dtw_provider.sample_gt_trajectory(env_id, seed=None)
                    if new_gt is not None:
                        gt_traj = new_gt

                pbar.update(num_envs)
                eps_count += num_envs
                ts = 0
                obs, info = eval_envs.reset()

    pbar.close()
    agent.train()
    return {
        k: np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v)
        for k, v in eval_metrics.items()
    }


# =============================================================================
# Single-env orchestration
# =============================================================================

def evaluate_single_env(
    agent,
    env_id: str,
    args,
    normalizer: SingleTaskNormalizer,
    train_args: TrainingArgs,
    output_dir: Path,
    timestamp: str,
    dtw_provider=None,
    traj_metrics=None,
    device: torch.device = torch.device("cuda"),
) -> Dict:
    level         = extract_level(env_id)
    base_env_name = extract_base_env_name(env_id)

    print(f"\n{'='*60}\n🎯 {env_id}  (base={base_env_name}, level={level})\n{'='*60}")

    obs_mode = "rgbd" if args.include_depth else "rgb"
    set_l_level(level)
    try:
        eval_envs = make_eval_envs_with_level(
            base_env_name=base_env_name,
            level=level,
            num_envs=args.num_envs,
            sim_backend=args.sim_backend,
            env_kwargs=dict(
                control_mode=args.control_mode,
                reward_mode="dense",
                obs_mode=obs_mode,
                render_mode="rgb_array",
                max_episode_steps=args.max_episode_steps, 
                sensor_configs=dict(shader_pack="rt-fast"), 
                human_render_camera_configs=dict(shader_pack="rt-fast"),
            ),
            other_kwargs=dict(obs_horizon=train_args.obs_horizon),
            video_dir=str(output_dir / "videos" / env_id),
            wrappers=[FlattenRGBDObservationWrapper],
        )
    except Exception as exc:
        clear_l_level()
        import traceback; traceback.print_exc()
        return {
            "env_id": env_id, "level": level,
            "timestamp": timestamp, "status": "error", "error": str(exc),
        }

    try:
        metrics = evaluate_episode(
            n=args.num_episodes,
            agent=agent,
            eval_envs=eval_envs,
            normalizer=normalizer,
            args=args,
            device=device,
            train_args=train_args,
            dtw_provider=dtw_provider,
            traj_metrics=traj_metrics,
            env_id=env_id,
        )
        result = {
            "env_id":       env_id,
            "level":        level,
            "num_episodes": args.num_episodes,
            "timestamp":    timestamp,
            "status":       "success",
        }
        for k, v in metrics.items():
            result[f"{k}_mean"] = float(v.mean())
            result[f"{k}_std"]  = float(v.std())
            result[f"{k}_min"]  = float(v.min())
            result[f"{k}_max"]  = float(v.max())
        # Ensure TSS fields always present; compute wTSS
        for _k in ["tss_success", "ndtw_success", "tss_fail", "ndtw_fail"]:
            if f"{_k}_mean" not in result:
                result[f"{_k}_mean"] = None
                result[f"{_k}_std"]  = None
        _sr  = result.get("success_once_mean") or 0.0
        _tss = result.get("tss_success_mean")
        result["wtss_mean"] = round(_sr * _tss, 6) if _tss is not None else 0.0
        succ = result.get("success_once_mean", float("nan"))
        ndtw = result.get("ndtw_fail_mean", None)
        print(f"✅ success={succ:.4f}" + (f" | nDTW={ndtw:.4f}" if ndtw else ""))

    except Exception as exc:
        import traceback; traceback.print_exc()
        result = {
            "env_id": env_id, "level": level,
            "timestamp": timestamp, "status": "error", "error": str(exc),
        }
    finally:
        eval_envs.close()
        clear_l_level()

    return result


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("eval_dp_single")
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--sub-config",   required=True)
    p.add_argument("--sim-root",     default="./data/sim")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--num-envs",     type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=800)
    p.add_argument("--control-mode", default="pd_joint_pos")
    p.add_argument("--sim-backend",  default="physx_cuda",
                   choices=["physx_cuda", "physx_cpu"])
    p.add_argument("--state-type",   default=None)
    p.add_argument("--pred-horizon", type=int, default=None)
    p.add_argument("--obs-horizon",  type=int, default=None)
    # NOTE: --num-inference-steps is the only DP-specific override.
    # Do NOT also put it in eval_extra_args in run_single_va.yaml.
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--include-depth", action="store_true", default=False)
    p.add_argument("--compute-dtw",   action="store_true", default=True)
    p.add_argument("--dtw-band-ratio", type=float, default=0.15)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args      = parse_args()
    device    = torch.device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("🧪 SINGLE-TASK DIFFUSION POLICY EVALUATION")
    print("=" * 70)

    print(f"\n📦 Loading checkpoint: {args.checkpoint}")
    agent, train_args = load_checkpoint(args.checkpoint, args, device)
    agent.to(device)
    normalizer = SingleTaskNormalizer(
        sub_config=args.sub_config,
        sim_root=args.sim_root,
        state_type=train_args.state_type,
        single_arm=getattr(train_args, "single_arm", False),
    )
    print(f"   pred_horizon={train_args.pred_horizon}  "
          f"num_inference_steps={train_args.num_inference_steps}  "
          f"state_type={train_args.state_type}")

    with open(args.sub_config) as f:
        ds_cfgs = json.load(f)
    env_ids = [c["repo_id"] for c in ds_cfgs]
    print(f"   Environments: {env_ids}")

    existing_results = load_existing_results(output_dir, _EVAL_TAG)
    print(f"   Existing results: {len(existing_results)} env(s)")
    pending_env_ids = [
        eid for eid in env_ids
        if not is_env_already_evaluated(eid, existing_results, args.num_episodes)
    ]
    skipped = len(env_ids) - len(pending_env_ids)
    print(f"   ⏭️  Skipping: {skipped}  |  🔜 Pending: {len(pending_env_ids)}")

    if not pending_env_ids:
        print("\n✅ All environments already evaluated.")
        return

    dtw_provider = None
    traj_metrics = None
    if args.compute_dtw:
        action_key = (
            "action.qpos_gripper_actions"
            if train_args.state_type in ("qpos", "mixpos")
            else "action.eepos_gripper_actions"
        )
        try:
            dtw_provider = GTTrajectoryProvider(
                sim_dataset_file=args.sub_config,
                sim_root=args.sim_root,
                action_key=action_key,
            )
            traj_metrics = TrajectoryMetrics(band_ratio=args.dtw_band_ratio)
            print(f"   ✅ DTW provider ready ({len(dtw_provider._episodes)} tasks).")
        except Exception as exc:
            print(f"   ⚠️  DTW provider failed: {exc}")

    results_json = output_dir / f"{_EVAL_TAG}_{timestamp}.json"
    print(f"\n📝 Incremental log: {results_json}")
    all_results: List[Dict] = list(existing_results.values())

    for env_id in pending_env_ids:
        result = evaluate_single_env(
            agent=agent,
            env_id=env_id,
            args=args,
            normalizer=normalizer,
            train_args=train_args,
            output_dir=output_dir,
            timestamp=timestamp,
            dtw_provider=dtw_provider,
            traj_metrics=traj_metrics,
            device=device,
        )
        all_results.append(result)
        save_results_incremental(all_results, results_json)

    out_csv = output_dir / f"{_EVAL_TAG}_{timestamp}.csv"
    pd.DataFrame(all_results).to_csv(out_csv, index=False)
    print(f"\n💾 CSV → {out_csv}")
    print_summary(all_results, _EVAL_TAG)


if __name__ == "__main__":
    main()