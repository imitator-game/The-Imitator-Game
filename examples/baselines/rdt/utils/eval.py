"""
Simplified evaluation script - Language-only, no video
Directly uses language prompts from environment
"""

from collections import defaultdict
import logging
import numpy as np
import torch
from tqdm import tqdm
from mani_skill.utils import common
from typing import Optional

from examples.baselines.lerobot_dataset.trajectory_metrics import EpisodeActionBuffer


def _any(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    return bool(np.asarray(value).any())


def _all(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.all().item())
    return bool(np.asarray(value).all())


def _bool_array(value, size: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=bool)
    if arr.ndim == 0:
        return np.full((size,), bool(arr), dtype=bool)
    return arr.reshape(-1)[:size]


def _final_success_array(final_info, size: int) -> np.ndarray | None:
    def _extract_success(item):
        if not isinstance(item, dict):
            return None
        for key in ("success", "success_once", "success_at_end"):
            if key in item:
                return item[key]
        episode = item.get("episode")
        if isinstance(episode, dict):
            for key in ("success", "success_once", "success_at_end"):
                if key in episode:
                    return episode[key]
        return None

    if isinstance(final_info, dict):
        return _bool_array(_extract_success(final_info), size)
    if isinstance(final_info, (list, tuple)):
        values = []
        found = False
        for item in final_info[:size]:
            success = _extract_success(item)
            if success is None:
                values.append(False)
                continue
            arr = _bool_array(success, 1)
            values.append(bool(arr[0]) if arr is not None and len(arr) else False)
            found = True
        if found:
            values.extend([False] * (size - len(values)))
            return np.asarray(values[:size], dtype=bool)
    return None


def evaluate(
    n: int,
    agent,
    eval_envs,
    device,
    sim_backend: str,
    progress_bar: bool = True,
    precomputed_lang=None,
    language_prompts: Optional[list[str]] = None,
    gripper_binary_action: bool = True,
    gripper_threshold: float = 0.4,
    gripper_indices: Optional[list[int]] = None,
    env_id: Optional[str] = None,
    dtw_provider=None,
    traj_metrics=None,
    dtw_dataset_idx: int = 0,
    dtw_normalize_gt: bool = False,
):
    """Evaluate agent with simplified language-only conditioning
    
    Args:
        n: Number of episodes to evaluate
        agent: Agent to evaluate
        eval_envs: Evaluation environments
        device: Device for computation
        sim_backend: Simulation backend
        progress_bar: Whether to show progress bar
        precomputed_lang: Precomputed language embedding loader (optional)
    """
    agent.eval()

    num_envs = eval_envs.num_envs
    compute_dtw = dtw_provider is not None and traj_metrics is not None and env_id is not None
    gt_traj = None
    action_buf = None
    success_steps = None
    if compute_dtw:
        gt_traj = dtw_provider.sample_gt_trajectory(
            env_id,
            seed=None,
            normalize=dtw_normalize_gt,
            dataset_idx=dtw_dataset_idx,
        )
        if gt_traj is None:
            logging.warning("No GT trajectory for %s; TSS metrics will be skipped.", env_id)
            compute_dtw = False
        else:
            action_buf = EpisodeActionBuffer(num_envs=num_envs)
            success_steps = np.full((num_envs,), -1, dtype=np.int64)

    if progress_bar:
        pbar = tqdm(total=n, desc=f"Evaluating RDT{' [+TSS]' if compute_dtw else ''}")

    prompt_idx = 0
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        ts = 0

        while eps_count < n:
            obs = common.to_tensor(obs, device)

            # Get language instructions from dataset or environment
            if language_prompts:
                batch_prompts = []
                for _ in range(num_envs):
                    batch_prompts.append(language_prompts[prompt_idx % len(language_prompts)])
                    prompt_idx += 1
                prompts = batch_prompts
            else:
                prompts = info.get('prompt', [''] * num_envs)
                if isinstance(prompts, str):
                    prompts = [prompts] * num_envs
            
            # Handle empty prompts with default
            prompts = [p if p else "pick red cube and place on plate." for p in prompts]

            # Get action sequence from agent with language prompt
            action_seq = agent.get_action(obs, language_prompt=prompts)

            # Binarize gripper action dimensions before stepping the env.
            if gripper_binary_action:
                if gripper_indices is None:
                    gripper_indices = [7, 15]
                if isinstance(action_seq, torch.Tensor):
                    for idx in gripper_indices:
                        if idx < action_seq.shape[-1]:
                            g = action_seq[..., idx]
                            action_seq[..., idx] = torch.where(
                                g > gripper_threshold,
                                torch.ones_like(g),
                                -torch.ones_like(g),
                            )
                else:
                    for idx in gripper_indices:
                        if idx < action_seq.shape[-1]:
                            g = action_seq[..., idx]
                            action_seq[..., idx] = np.where(
                                g > gripper_threshold,
                                1.0,
                                -1.0,
                            )

            # Convert to appropriate format for the simulation backend
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()

            # Execute action sequence
            for i in range(action_seq.shape[1]):
                step_action = action_seq[:, i]
                if compute_dtw and action_buf is not None:
                    action_buf.append(step_action)
                obs, rew, terminated, truncated, info = eval_envs.step(step_action)
                ts += 1
                if compute_dtw and success_steps is not None:
                    success = _bool_array(info.get("success"), num_envs)
                    if success is not None:
                        new_success = (success_steps == -1) & success
                        success_steps[new_success] = ts
                if _any(truncated):
                    break

            # Process episode completion
            if _any(truncated):
                assert _all(truncated), \
                    "all episodes should truncate at the same time for fair evaluation"

                # Collect metrics
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)

                if compute_dtw and action_buf is not None and success_steps is not None and gt_traj is not None:
                    final_success = _final_success_array(info.get("final_info"), num_envs)
                    if final_success is not None:
                        success_steps[(success_steps == -1) & final_success] = ts
                    pred_trajs = action_buf.get_and_reset()
                    gt_len = len(gt_traj)
                    for env_i, pred_traj in enumerate(pred_trajs):
                        success_step = int(success_steps[env_i])
                        if success_step >= 2:
                            try:
                                metric = traj_metrics.compute(pred_traj[:success_step], gt_traj)
                                eval_metrics["tss_success"].append(np.array(metric["tss"], dtype=np.float32))
                                eval_metrics["ndtw_success"].append(np.array(metric["ndtw"], dtype=np.float32))
                            except Exception as exc:
                                logging.warning("TSS(success) failed for %s env %s: %s", env_id, env_i, exc)
                        else:
                            pred_trimmed = pred_traj[:gt_len]
                            if len(pred_trimmed) >= 2:
                                try:
                                    metric = traj_metrics.compute(pred_trimmed, gt_traj)
                                    eval_metrics["tss_fail"].append(np.array(metric["tss"], dtype=np.float32))
                                    eval_metrics["ndtw_fail"].append(np.array(metric["ndtw"], dtype=np.float32))
                                except Exception as exc:
                                    logging.warning("TSS(fail) failed for %s env %s: %s", env_id, env_i, exc)
                    success_steps[:] = -1
                    new_gt = dtw_provider.sample_gt_trajectory(
                        env_id,
                        seed=None,
                        normalize=dtw_normalize_gt,
                        dataset_idx=dtw_dataset_idx,
                    )
                    if new_gt is not None:
                        gt_traj = new_gt

                eps_count += num_envs
                if progress_bar:
                    pbar.update(num_envs)

                # Reset environment for next episode
                obs, info = eval_envs.reset()
                ts = 0

                # Update prompts for the new episodes
                if language_prompts:
                    batch_prompts = []
                    for _ in range(num_envs):
                        batch_prompts.append(language_prompts[prompt_idx % len(language_prompts)])
                        prompt_idx += 1
                    prompts = batch_prompts
                else:
                    prompts = info.get('prompt', [''] * num_envs)
                    if isinstance(prompts, str):
                        prompts = [prompts] * num_envs

    agent.train()
    if progress_bar:
        pbar.close()

    # Convert metrics to numpy arrays
    for k in list(eval_metrics.keys()):
        if len(eval_metrics[k]) > 0:
            eval_metrics[k] = np.stack(eval_metrics[k])

    return eval_metrics


def evaluate_and_save_best(
    iteration: int,
    agent,
    eval_envs,
    device,
    sim_backend: str,
    precomputed_lang=None,
    language_prompts: Optional[list[str]] = None,
    gripper_binary_action: bool = True,
    gripper_threshold: float = 0.4,
    gripper_indices: Optional[list[int]] = None,
    num_eval_episodes: int = 50,
    eval_freq: int = 5000,
    best_eval_metrics=None,
    save_checkpoint_fn=None,
    run_name: str = "",
    writer=None,
):
    """Evaluate agent and save best checkpoints during training
    
    Args:
        iteration: Current training iteration
        agent: Agent to evaluate
        eval_envs: Evaluation environments
        device: Device for computation
        sim_backend: Simulation backend
        precomputed_lang: Precomputed language embedding loader (optional)
        language_prompts: List of prompts to use instead of env prompts (optional)
        num_eval_episodes: Number of episodes to evaluate
        eval_freq: Evaluation frequency
        best_eval_metrics: Dict to track best metrics
        save_checkpoint_fn: Function to save checkpoints
        run_name: Experiment run name
        writer: Tensorboard writer
        
    Returns:
        Updated best_eval_metrics dict
    """
    if best_eval_metrics is None:
        best_eval_metrics = defaultdict(float)

    if iteration % eval_freq == 0 and iteration > 0:
        print(f"\n{'=' * 60}")
        print(f"Evaluating at iteration {iteration}")
        print(f"{'=' * 60}")

        # Quick evaluation first to check if worth doing full eval
        quick_eval_metrics = evaluate(
            n=min(10, num_eval_episodes),
            agent=agent,
            eval_envs=eval_envs,
            device=device,
            sim_backend=sim_backend,
            progress_bar=True,
            precomputed_lang=precomputed_lang,
            language_prompts=language_prompts,
            gripper_binary_action=gripper_binary_action,
            gripper_threshold=gripper_threshold,
            gripper_indices=gripper_indices,
        )

        # If quick eval shows promise (>30% success), do full evaluation
        quick_success = np.mean(quick_eval_metrics.get('success_at_end', [0]))
        if quick_success >= 0.3:
            eval_metrics = evaluate(
                n=num_eval_episodes,
                agent=agent,
                eval_envs=eval_envs,
                device=device,
                sim_backend=sim_backend,
                progress_bar=True,
                precomputed_lang=precomputed_lang,
                language_prompts=language_prompts,
                gripper_binary_action=gripper_binary_action,
                gripper_threshold=gripper_threshold,
                gripper_indices=gripper_indices,
            )
        else:
            eval_metrics = quick_eval_metrics

        # Process and log metrics
        print(f"\nEvaluated {len(eval_metrics['success_at_end'])} episodes")

        metric_summary = {}
        for k in eval_metrics.keys():
            metric_value = np.mean(eval_metrics[k])
            metric_summary[k] = metric_value

            if writer:
                writer.add_scalar(f"eval/{k}", metric_value, iteration)

            print(f"  {k}: {metric_value:.4f}")

        # Save checkpoints for best metrics
        save_on_best_metrics = ["success_once", "success_at_end"]
        for metric_name in save_on_best_metrics:
            if metric_name in metric_summary:
                current_value = metric_summary[metric_name]

                if current_value > best_eval_metrics[metric_name]:
                    best_eval_metrics[metric_name] = current_value

                    if save_checkpoint_fn:
                        save_checkpoint_fn(run_name, f"best_eval_{metric_name}", iteration, best_eval_metrics)

                    print(f"  ✓ New best {metric_name}: {current_value:.4f} (saved checkpoint)")

        print(f"{'=' * 60}\n")

    return best_eval_metrics
