"""IG rollout adapter copied from eval_dp_imitator.py with Zero-WAM history hooks.

The function is unchanged from the implementation used by the final runs.
Keep the adaptation here so other baselines' evaluation files stay untouched.
"""
from collections import defaultdict, deque
from typing import Dict
import gymnasium
import numpy as np
import torch
from tqdm import tqdm
from mani_skill.utils import common
from examples.baselines.lerobot_dataset.lerobot_paired_dataset import InputMode
from examples.baselines.lerobot_dataset.trajectory_metrics import TrajectoryMetrics, EpisodeActionBuffer

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
    dynamic_chunks = eval_kwargs.get("dynamic_action_chunks", False)
    if dynamic_chunks and (light_temporal_agg or temporal_agg or num_envs != 1):
        raise ValueError("Dynamic chunks require one environment and no temporal aggregation")
    next_query_step, chunk_start_step = 0, 0

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

            if dynamic_chunks and ts > 0:
                agent.observe_step(obs)

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
            if (ts == next_query_step if dynamic_chunks else ts % query_frequency == 0):
                action_seq = agent.get_action(obs)
                if action_seq.dim() == 2:
                    action_seq = action_seq.unsqueeze(0)
                if dynamic_chunks:
                    if not 0 < action_seq.shape[1] <= pred_horizon:
                        raise ValueError(f"Invalid dynamic action length: {action_seq.shape}")
                    chunk_start_step = ts
                    next_query_step = ts + action_seq.shape[1]
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
                if dynamic_chunks:
                    raw_action = action_seq[:, ts - chunk_start_step]
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
            if dynamic_chunks:
                agent.record_executed_action(raw_action)
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
                next_query_step, chunk_start_step = 0, 0

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
