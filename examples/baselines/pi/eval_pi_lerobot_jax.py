from __future__ import annotations

import argparse
import json
import logging
import os
from functools import partial
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import flax.nnx as nnx
import jax
import numpy as np
from transformers import AutoProcessor

from examples.baselines.lerobot_dataset.evaluate_processor import (
    HumanVideoSimEvaluateProcessor,
    HumanVideoSimEvaluateProcessorConfig,
)
from examples.baselines.lerobot_dataset.trajectory_metrics import (
    GTTrajectoryProvider,
    TrajectoryMetrics,
)
from examples.baselines.pi.train_pi_lerobot_jax import (
    FlattenRGBDObservationWrapper,
    JaxPiEvaluator,
    build_model_config,
    detect_pi05_from_params,
    evaluate,
    extract_base_env_name,
    extract_level,
    init_logging,
    load_partial_params,
    make_eval_envs,
    resolve_params_path,
    set_l_level,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a JAX Pi/Pi0.5 checkpoint in ManiSkill.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--checkpoint_step", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./runs/pi_lerobot_jax_eval")

    parser.add_argument("--env_id", type=str, default="L0_TwoRobotStirSpoon-v1")
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--num_eval_envs", type=int, default=1)
    parser.add_argument("--num_diffusion_steps", type=int, default=10)
    parser.add_argument("--capture_video", action="store_true")

    parser.add_argument("--obs_mode", type=str, default="rgb")
    parser.add_argument("--control_mode", type=str, default="pd_joint_pos")
    parser.add_argument("--max_episode_steps", type=int, default=400)
    parser.add_argument("--sim_backend", type=str, default="physx_cpu")
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="dense",
        choices=["sparse", "dense", "normalized_dense", "none"],
    )
    parser.add_argument("--shader", default="rt-fast")

    parser.add_argument("--precision", type=str, default="bfloat16")
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--pred_horizon", type=int, default=50)
    parser.add_argument("--max_token_len", type=int, default=200)
    parser.add_argument("--processor_name_or_path", type=str, default="google/paligemma-3b-pt-224")
    parser.add_argument("--processor_local_files_only", action="store_true")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--pi05", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sim_root", type=str, default="/path/to/sim_root")
    parser.add_argument("--sim_split", type=str, default="train")
    parser.add_argument("--sim_dataset_file", type=str, default="examples/baselines/lerobot_dataset/config/sim_config.json")
    parser.add_argument("--sim_task_desc_file", type=str, default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")
    parser.add_argument("--sim_state_type", type=str, default="qpos")
    parser.add_argument("--cameras", type=str, nargs="+", default=["zed2i"])

    parser.add_argument("--human_root", type=str, default="/path/to/human_root")
    parser.add_argument("--human_split", type=str, default="train")
    parser.add_argument("--human_dataset_file", type=str, default="examples/baselines/lerobot_dataset/config/human_config.json")
    parser.add_argument("--human_task_desc_file", type=str, default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    parser.add_argument("--human_cameras", type=list, default=["zed2i"])
    parser.add_argument("--human_num_frames", type=int, default=10)
    parser.add_argument("--human_sampling_strategy", type=str, default="uniform_jitter")
    parser.add_argument("--human_max_jitter", type=int, default=3)
    parser.add_argument("--human_fps", type=int, default=30)
    parser.add_argument("--human_image_size", type=int, nargs=2, default=[224, 224])

    parser.add_argument("--task_mapping_file", type=str, default="examples/baselines/lerobot_dataset/task_mapping.json")
    parser.add_argument("--include_depth", action="store_true", default=False)
    parser.add_argument("--vla", action="store_true", default=False)
    parser.add_argument(
        "--skip_masked_cameras",
        action="store_true",
        help="Only feed real camera views to Pi instead of padding missing canonical cameras with masked zero images.",
    )
    parser.add_argument(
        "--use_prefix_kv_cache",
        action="store_true",
        help="Accepted for train/eval config parity. Pi JAX sampling already uses prefix KV cache.",
    )
    parser.add_argument(
        "--l3_eval_l_level",
        type=str,
        default="L0",
        choices=["L0", "L1", "L2", "L3"],
        help="L-level flags to apply for L3 env_ids. The L3 gym env is still used; only the level switches default to L0.",
    )
    parser.add_argument(
        "--eval_lr_mirror",
        default="auto",
        choices=["auto", "true", "false"],
        help="Override tabletop left-right mirror during eval. 'auto' follows the active L-level flags.",
    )
    parser.add_argument(
        "--eval_lr_mirror_robot_pose",
        default="false",
        choices=["auto", "true", "false"],
        help="Override robot pose swapping under mirror. 'false' keeps robot articulations unswapped.",
    )
    parser.add_argument("--compute_dtw", action="store_true", help="Compute TSS/nDTW trajectory metrics against sim GT actions.")
    parser.add_argument("--dtw_band_ratio", type=float, default=0.15, help="Sakoe-Chiba band ratio for TSS/nDTW.")
    return parser.parse_args()


def parse_optional_bool_flag(raw: str | bool | None) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"auto", "default", "none"}:
        return None
    if value in {"1", "true", "on", "yes", "enable", "enabled"}:
        return True
    if value in {"0", "false", "off", "no", "disable", "disabled"}:
        return False
    raise ValueError(f"Invalid boolean flag value: {raw}")


def _resolve_checkpoint_params_path(checkpoint_path: Path, checkpoint_step: int | None) -> tuple[Path, int | None]:
    step_dirs = sorted(
        (int(p.name), p)
        for p in checkpoint_path.iterdir()
        if p.is_dir() and p.name.isdigit()
    ) if checkpoint_path.is_dir() else []
    if not step_dirs:
        return Path(resolve_params_path(checkpoint_path)), None

    available_steps = [step for step, _ in step_dirs]
    restore_step = checkpoint_step if checkpoint_step is not None else available_steps[-1]
    if restore_step not in available_steps:
        raise ValueError(
            f"Checkpoint step {restore_step} not found in {checkpoint_path}. Available steps: {available_steps}"
        )
    params_path = checkpoint_path / str(restore_step) / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Expected params directory at {params_path}")
    return params_path, restore_step


def _load_model(args):
    checkpoint_path = Path(args.checkpoint_path)
    params_path, restored_step = _resolve_checkpoint_params_path(checkpoint_path, args.checkpoint_step)
    detected_pi05 = detect_pi05_from_params(params_path, args.pi05)
    if detected_pi05 != args.pi05:
        logging.info("Overriding --pi05=%s based on checkpoint contents: %s", args.pi05, detected_pi05)
        args.pi05 = detected_pi05

    model_config = build_model_config(args, args.pi05)
    model = model_config.create(jax.random.key(args.seed))
    params = nnx.state(model)
    partial_params, _ = load_partial_params(params.to_pure_dict(), params_path)
    if partial_params is None:
        raise FileNotFoundError(f"Unable to load JAX params from {params_path}")
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(partial_params)
    model = nnx.merge(graphdef, state)
    if hasattr(model, "eval"):
        model.eval()
    return model, restored_step


def _build_evaluator(args, processor: AutoProcessor):
    evaluate_processor_config = HumanVideoSimEvaluateProcessorConfig(
        human_root=args.human_root,
        human_split=args.human_split,
        human_dataset_file=args.human_dataset_file,
        human_task_description_file=args.human_task_desc_file,
        human_cameras=args.human_cameras,
        human_fps=args.human_fps,
        human_image_size=args.human_image_size,
        human_num_frames=args.human_num_frames,
        human_sampling_strategy=args.human_sampling_strategy,
        human_max_jitter=args.human_max_jitter,
        sim_root=args.sim_root,
        sim_split=args.sim_split,
        sim_dataset_file=args.sim_dataset_file,
        sim_task_description_file=args.sim_task_desc_file,
        sim_state_type=args.sim_state_type,
        task_mapping_file=args.task_mapping_file,
        vla=args.vla,
    )
    return JaxPiEvaluator(
        args,
        HumanVideoSimEvaluateProcessor(evaluate_processor_config),
        processor,
    )


def _resolve_eval_l_level(args) -> str:
    task_level = extract_level(args.env_id)
    if task_level == "L3":
        return args.l3_eval_l_level
    return task_level


def _build_eval_envs(args):
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        max_episode_steps=args.max_episode_steps,
    )
    base_env_name = extract_base_env_name(args.env_id)
    task_level = extract_level(args.env_id)
    eval_l_level = _resolve_eval_l_level(args)
    lr_mirror_enabled = parse_optional_bool_flag(args.eval_lr_mirror)
    lr_mirror_robot_pose_enabled = parse_optional_bool_flag(args.eval_lr_mirror_robot_pose)
    set_l_level(eval_l_level)
    from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils

    L0_L3_utils.set_lr_mirror_enabled(lr_mirror_enabled)
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(lr_mirror_robot_pose_enabled)
    logging.info(
        "Eval env mapping: env_id=%s -> gym_env_id=%s task_level=%s eval_l_level=%s lr_mirror=%s lr_mirror_robot_pose=%s",
        args.env_id,
        base_env_name,
        task_level,
        eval_l_level,
        lr_mirror_enabled,
        lr_mirror_robot_pose_enabled,
    )
    return make_eval_envs(
        base_env_name,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs=dict(obs_horizon=1),
        video_dir=os.path.join(args.output_dir, "videos") if args.capture_video else None,
        wrappers=[partial(FlattenRGBDObservationWrapper, depth=args.include_depth)],
        l_level=eval_l_level,
        lr_mirror_enabled=lr_mirror_enabled,
        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
    )


def _build_trajectory_metrics(args):
    if not args.compute_dtw:
        return None, None

    if args.sim_state_type in ("qpos", "mixpos"):
        action_key = "action.qpos_gripper_actions"
    elif args.sim_state_type == "eepos":
        action_key = "action.eepos_gripper_actions"
    else:
        raise ValueError(f"Unknown sim_state_type for TSS metrics: {args.sim_state_type}")

    try:
        dtw_provider = GTTrajectoryProvider(
            sim_dataset_file=args.sim_dataset_file,
            sim_root=args.sim_root,
            action_key=action_key,
        )
        traj_metrics = TrajectoryMetrics(band_ratio=args.dtw_band_ratio)
        logging.info(
            "Loaded GT trajectories for TSS: tasks=%s action_key=%s band_ratio=%s",
            len(dtw_provider._episodes),
            action_key,
            args.dtw_band_ratio,
        )
        return dtw_provider, traj_metrics
    except Exception as exc:
        logging.warning("Failed to initialize TSS metrics; continuing without TSS: %s", exc)
        return None, None


def main():
    init_logging()
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        args.processor_name_or_path,
        trust_remote_code=True,
        use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        local_files_only=args.processor_local_files_only,
    )
    model, restored_step = _load_model(args)
    evaluator = _build_evaluator(args, processor)
    eval_envs = _build_eval_envs(args)
    dtw_provider, traj_metrics = _build_trajectory_metrics(args)

    try:
        metrics = evaluate(
            env_id=args.env_id,
            n=args.num_eval_episodes,
            model=model,
            evaluator=evaluator,
            eval_envs=eval_envs,
            sim_backend=args.sim_backend,
            num_diffusion_steps=args.num_diffusion_steps,
            dtw_provider=dtw_provider,
            traj_metrics=traj_metrics,
        )
    finally:
        eval_envs.close()

    reduced = {k: float(np.mean(v)) for k, v in metrics.items()}
    if args.compute_dtw:
        for key in ("tss_success", "tss_fail", "ndtw_success", "ndtw_fail"):
            reduced.setdefault(key, None)
    logging.info("Eval metrics: %s", reduced)

    results = {
        "env_id": args.env_id,
        "task_level": extract_level(args.env_id),
        "eval_l_level": _resolve_eval_l_level(args),
        "eval_lr_mirror": args.eval_lr_mirror,
        "eval_lr_mirror_robot_pose": args.eval_lr_mirror_robot_pose,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_step": restored_step,
        "num_eval_episodes": args.num_eval_episodes,
        "num_eval_envs": args.num_eval_envs,
        "num_diffusion_steps": args.num_diffusion_steps,
        "compute_dtw": args.compute_dtw,
        "dtw_band_ratio": args.dtw_band_ratio if args.compute_dtw else None,
        "tss_success_mean": reduced.get("tss_success"),
        "tss_fail_mean": reduced.get("tss_fail"),
        "metrics_mean": reduced,
        "metrics_raw_shape": {k: list(v.shape) for k, v in metrics.items()},
    }
    output_path = Path(args.output_dir) / "eval_metrics.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    logging.info("Saved eval results to %s", output_path)


if __name__ == "__main__":
    main()
