#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch


def _parse_checkpoint_step(checkpoint_path: Path) -> int | None:
    name = checkpoint_path.name
    if name.startswith("checkpoint-"):
        step_part = name.removeprefix("checkpoint-").split("-", maxsplit=1)[0]
        if step_part.isdigit():
            return int(step_part)
    return None


def _has_processor_config(path: Path) -> bool:
    return (path / "processor_config.json").is_file()


def _resolve_processor_path(checkpoint_path: Path, explicit_processor_path: str | None) -> Path:
    candidates: list[Path] = []
    if explicit_processor_path:
        candidates.append(Path(explicit_processor_path).expanduser().resolve())

    candidates.append(checkpoint_path)
    candidates.append(checkpoint_path / "processor")

    run_dir = checkpoint_path.parent if checkpoint_path.name.startswith("checkpoint-") else checkpoint_path
    candidates.append(run_dir / "processor")
    candidates.append(run_dir)

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)

    for candidate in unique_candidates:
        if _has_processor_config(candidate):
            return candidate

    checked = ", ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(
        "Could not find GR00T processor_config.json. "
        f"Checked: {checked}. Pass --processor_path explicitly if it is stored elsewhere."
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _parse_optional_bool_flag(raw: str | bool | None) -> bool | None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GR00T policy-generation checkpoint on one ManiSkill task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--processor_path", default=None)
    parser.add_argument("--env_id", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--embodiment_tag", default="NEW_EMBODIMENT")
    parser.add_argument("--language_source", default="human_desc", choices=["task", "human_desc", "sim_desc"])
    parser.add_argument("--task_mapping_path", default=None)
    parser.add_argument("--human_desc_path", default=None)
    parser.add_argument("--sim_desc_path", default=None)
    parser.add_argument("--sim_backend", default="physx_cpu")
    parser.add_argument("--control_mode", default="pd_joint_pos")
    parser.add_argument("--obs_mode", default="rgb")
    parser.add_argument("--reward_mode", default="dense", choices=["sparse", "dense", "normalized_dense", "none"])
    parser.add_argument("--shader", default="rt-fast")
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--num_eval_envs", type=int, default=5)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compute_dtw", action="store_true", help="Compute TSS/nDTW trajectory metrics.")
    parser.add_argument("--dtw_band_ratio", type=float, default=0.15)
    parser.add_argument("--sim_dataset_file", default=None)
    parser.add_argument("--sim_root", default="demos/imitator_data")
    parser.add_argument("--dtw_action_key", default="action.qpos_gripper_actions")
    parser.add_argument(
        "--eval_lr_mirror",
        default="auto",
        choices=["auto", "true", "false"],
        help="Override tabletop left-right mirror during eval. 'auto' follows L2/L3 task flags.",
    )
    parser.add_argument(
        "--eval_lr_mirror_robot_pose",
        default="false",
        choices=["auto", "true", "false"],
        help="Override robot pose swapping under mirror. 'false' keeps robot root poses unswapped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModel, AutoProcessor

    import gr00t.model  # noqa: F401 - registers GR00T AutoModel/AutoProcessor classes
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.online_eval import Gr00tOnlineEvaluator, OnlineEvalConfig

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    lr_mirror_enabled = _parse_optional_bool_flag(args.eval_lr_mirror)
    lr_mirror_robot_pose_enabled = _parse_optional_bool_flag(args.eval_lr_mirror_robot_pose)

    model = AutoModel.from_pretrained(checkpoint_path)
    model.eval()
    model.to(device=device, dtype=dtype)

    processor_path = _resolve_processor_path(checkpoint_path, args.processor_path)
    print(f"Loading GR00T processor from {processor_path}", flush=True)
    processor = AutoProcessor.from_pretrained(processor_path)
    processor.eval()

    evaluator = Gr00tOnlineEvaluator(
        OnlineEvalConfig(
            env_id=args.env_id,
            embodiment_tag=EmbodimentTag[args.embodiment_tag],
            language_source=args.language_source,
            task_mapping_path=args.task_mapping_path,
            human_desc_path=args.human_desc_path,
            sim_desc_path=args.sim_desc_path,
            num_episodes=args.num_eval_episodes,
            num_envs=args.num_eval_envs,
            max_episode_steps=args.max_episode_steps,
            sim_backend=args.sim_backend,
            control_mode=args.control_mode,
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
            shader=args.shader,
            capture_video=args.capture_video,
            output_dir=str(output_dir),
            compute_dtw=args.compute_dtw,
            dtw_band_ratio=args.dtw_band_ratio,
            sim_dataset_file=args.sim_dataset_file,
            sim_root=args.sim_root,
            dtw_action_key=args.dtw_action_key,
            lr_mirror_enabled=lr_mirror_enabled,
            lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
        ),
        processor=processor,
    )
    try:
        checkpoint_step = _parse_checkpoint_step(checkpoint_path)
        metrics_mean = evaluator.evaluate(model, global_step=checkpoint_step or 0)
    finally:
        evaluator.close()

    payload = {
        "env_id": args.env_id,
        "checkpoint_path": str(checkpoint_path),
        "processor_path": str(processor_path),
        "checkpoint_step": checkpoint_step,
        "num_eval_episodes": args.num_eval_episodes,
        "num_eval_envs": args.num_eval_envs,
        "compute_dtw": args.compute_dtw,
        "dtw_band_ratio": args.dtw_band_ratio if args.compute_dtw else None,
        "dtw_action_key": args.dtw_action_key if args.compute_dtw else None,
        "eval_lr_mirror": args.eval_lr_mirror,
        "eval_lr_mirror_robot_pose": args.eval_lr_mirror_robot_pose,
        "tss_success_mean": metrics_mean.get("tss_success"),
        "tss_fail_mean": metrics_mean.get("tss_fail"),
        "metrics_mean": _to_jsonable(metrics_mean),
    }
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
