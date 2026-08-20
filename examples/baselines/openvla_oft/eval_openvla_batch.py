import argparse
import copy
import json
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from examples.baselines.openvla_oft.eval_openvla import (
    OpenVLAEvaluator,
    _build_evaluate_processor,
    _load_aux_modules,
    _load_model_and_processor,
    _parse_sim_task_ids,
    _resolve_dataset_name,
    _resolve_stats_path,
    evaluate,
)
from examples.baselines.openvla_oft.utils.make_env import (
    SelectCameraObservationWrapper,
    clear_tabletop_task_flags,
    configure_tabletop_task_flags,
    extract_base_env_name,
    extract_level,
    make_eval_envs,
    parse_optional_bool_flag,
)
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluation for OpenVLA")

    parser.add_argument("--eval-config", required=True,
                        help="TXT file with one env_id per line (supports L0_/L1_/L2_/L3_ prefixes)")
    parser.add_argument("--output-dir", required=True, help="Directory to save JSON/CSV results")

    parser.add_argument("--checkpoint_dir", required=True, help="Path to saved checkpoint directory")
    parser.add_argument("--model_name_or_path", required=True, help="Base OpenVLA model path")
    parser.add_argument("--stats_path", default=None, help="Path to stats json for action unnormalization")
    parser.add_argument("--dataset_name", default=None, help="Dataset name key inside stats json")
    parser.add_argument("--use_lerobot", action="store_true", default=False)

    parser.add_argument("--env_id", default="PickCubeYCB-v1")
    parser.add_argument("--obs_mode", default="rgb")
    parser.add_argument("--control_mode", default="pd_joint_pos")
    parser.add_argument("--max_episode_steps", type=int, default=600)
    parser.add_argument("--sim_backend", default="physx_cpu")
    parser.add_argument("--shader", default="rt-fast")
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--num_eval_envs", type=int, default=1)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--capture_video", action="store_true", default=False)
    parser.add_argument("--eval_camera", type=str, default=None,
                        help="Optional camera name to use for evaluation observations.")
    parser.add_argument("--lerobot_camera", type=str, default="zed2i")

    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--default_language", default="pick red cube and place on plate.")
    parser.add_argument("--human_root", default="data")
    parser.add_argument("--human_dataset_file",
                        default="examples/baselines/lerobot_dataset/config/human_config.json")
    parser.add_argument("--human_split", default="train")
    parser.add_argument("--human_task_description_file",
                        default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    parser.add_argument("--sim_task_description_file",
                        default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")
    parser.add_argument("--task_mapping_file", default="examples/baselines/lerobot_dataset/task_mapping.json")
    parser.add_argument("--sim_root", default="data")
    parser.add_argument("--sim_dataset_file", default=None)
    parser.add_argument("--eval_sim_task_ids", default=None,
                        help="Comma-separated sim_task_id list matching num_eval_envs.")
    parser.add_argument("--eval_lr_mirror", default="false", choices=["auto", "true", "false"],
                        help="Override tabletop left-right mirror during eval. Default 'false'.")
    parser.add_argument("--eval_lr_mirror_robot_pose", default="false", choices=["auto", "true", "false"],
                        help="Override robot pose swapping under mirror during eval. Default 'false' keeps agents unswapped.")

    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--action_mode", default="discrete", choices=["discrete", "l1_regression", "diffusion"])
    parser.add_argument("--num_diffusion_steps_train", type=int, default=50)
    parser.add_argument("--use_proprio", action="store_true", default=False)
    parser.add_argument("--use_film", action="store_true", default=False)
    parser.add_argument("--proprio_fusion_mode", type=str, default="input", choices=["input", "output"],
                        help="Where to fuse proprio for continuous-action eval. "
                             "'input' keeps legacy VLA-input fusion; "
                             "'output' matches cached-VL-feature training and fuses proprio after VLA.")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--debug_obs", action="store_true", default=False)

    return parser.parse_args()


def load_eval_config(path: str) -> List[str]:
    config_path = Path(path)
    if config_path.suffix != ".txt":
        raise ValueError(f"Expected .txt eval config, got: {config_path.suffix}")
    with open(config_path, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return lines

def _resolve_dataset_name_for_env(args, base_env: str) -> str:
    tmp = copy.copy(args)
    tmp.env_id = base_env
    return _resolve_dataset_name(tmp)


def _build_eval_envs(
    args,
    base_env: str,
    video_dir: Optional[str],
    l_level: str,
    lr_mirror_enabled: Optional[bool],
    lr_mirror_robot_pose_enabled: Optional[bool],
):
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="dense",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)

    wrappers = []
    eval_camera = args.eval_camera
    if eval_camera is None and args.use_lerobot:
        eval_camera = args.lerobot_camera
    if eval_camera:
        wrappers.append(partial(SelectCameraObservationWrapper, camera_names=[eval_camera]))
    wrappers.append(FlattenRGBDObservationWrapper)

    return make_eval_envs(
        base_env,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=video_dir,
        wrappers=wrappers,
        l_level=l_level,
        lr_mirror_enabled=lr_mirror_enabled,
        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
    )


def evaluate_single_env(
    env_id: str,
    args,
    vla,
    processor,
    action_head,
    noisy_action_projector,
    proprio_projector,
    evaluate_processor,
    stats_path: str,
    output_dir: Path,
) -> Dict[str, Any]:
    base_env_name = extract_base_env_name(env_id)
    level = extract_level(env_id)

    print("\n" + "=" * 80)
    print(f"Evaluating: {env_id}")
    print(f"  Base env: {base_env_name}")
    print(f"  Level: {level}")
    print("=" * 80)

    lr_mirror_enabled = parse_optional_bool_flag(args.eval_lr_mirror)
    lr_mirror_robot_pose_enabled = parse_optional_bool_flag(args.eval_lr_mirror_robot_pose)
    configure_tabletop_task_flags(
        level=level,
        lr_mirror_enabled=lr_mirror_enabled,
        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
    )

    video_dir = None
    if args.capture_video:
        video_dir = str(output_dir / "videos" / env_id)

    eval_envs = _build_eval_envs(
        args,
        base_env_name,
        video_dir,
        l_level=level,
        lr_mirror_enabled=lr_mirror_enabled,
        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
    )

    try:
        dataset_name = _resolve_dataset_name_for_env(args, base_env_name)
        eval_args = type("EvalArgs", (), {})()
        eval_args.action_mode = args.action_mode
        eval_args.use_proprio = args.use_proprio
        eval_args.proprio_fusion_mode = args.proprio_fusion_mode
        eval_args.image_size = args.image_size
        eval_args.demo_path = ""
        eval_args.action_dim = args.action_dim
        eval_args.output_dir = args.checkpoint_dir
        eval_args.default_language = args.default_language
        eval_args.stats_path = stats_path
        eval_args.dataset_name = dataset_name
        eval_args.debug_obs = args.debug_obs

        evaluator = OpenVLAEvaluator(eval_args, evaluate_processor)
        evaluator.processor = processor

        sim_task_raw = ",".join([env_id] * args.num_eval_envs)
        sim_task_ids = _parse_sim_task_ids(sim_task_raw, args.num_eval_envs, env_id)

        metrics = evaluate(
            n=args.num_eval_episodes,
            vla=vla,
            processor=processor,
            eval_envs=eval_envs,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            sim_backend=args.sim_backend,
            progress_bar=True,
            action_head=action_head,
            noisy_action_projector=noisy_action_projector,
            proprio_projector=proprio_projector,
            use_film=args.use_film,
            image_size=args.image_size,
            evaluator=evaluator,
            sim_task_ids=sim_task_ids,
        )

        results = {
            "env_id": env_id,
            "base_env_name": base_env_name,
            "level": level,
            "prompt_mode": "language",
            "num_episodes": args.num_eval_episodes,
        }

        for k, v in metrics.items():
            if len(v) == 0:
                continue
            arr = np.asarray(v)
            results[f"{k}_mean"] = float(arr.mean())
            results[f"{k}_std"] = float(arr.std())
            results[f"{k}_min"] = float(arr.min())
            results[f"{k}_max"] = float(arr.max())

        return results

    except Exception as exc:
        return {
            "env_id": env_id,
            "base_env_name": base_env_name,
            "level": level,
            "prompt_mode": "language",
            "num_episodes": args.num_eval_episodes,
            "error": str(exc),
        }
    finally:
        eval_envs.close()
        if video_dir is not None:
            moved = _flatten_task_videos(Path(video_dir))
            if moved:
                print(f"  Flattened {moved} video(s) into {video_dir}")
        clear_tabletop_task_flags()


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _flatten_task_videos(task_video_dir: Path) -> int:
    """Move mp4s from per-env subdirectories back into the task root directory."""
    if not task_video_dir.exists():
        return 0

    moved = 0
    for mp4 in sorted(task_video_dir.rglob("*.mp4")):
        if mp4.parent == task_video_dir:
            continue
        target = task_video_dir / mp4.name
        suffix = 0
        while target.exists():
            suffix += 1
            target = task_video_dir / f"{mp4.stem}_{suffix}{mp4.suffix}"
        mp4.replace(target)
        moved += 1

    for child in sorted(task_video_dir.rglob("*"), reverse=True):
        if child.is_dir():
            try:
                child.rmdir()
            except OSError:
                pass

    return moved


def main():
    args = parse_args()
    args.image_size = tuple(args.image_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_ids = load_eval_config(args.eval_config)
    if not env_ids:
        raise ValueError("No env_ids found in eval-config.")

    stats_path = _resolve_stats_path(args)

    vla, processor = _load_model_and_processor(args)
    action_head, noisy_action_projector, proprio_projector = _load_aux_modules(args, vla)

    evaluate_processor = _build_evaluate_processor(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_name = Path(args.eval_config).stem

    results = []
    for env_id in env_ids:
        result = evaluate_single_env(
            env_id=env_id,
            args=args,
            vla=vla,
            processor=processor,
            action_head=action_head,
            noisy_action_projector=noisy_action_projector,
            proprio_projector=proprio_projector,
            evaluate_processor=evaluate_processor,
            stats_path=stats_path,
            output_dir=output_dir,
        )
        results.append(result)

    results_json = output_dir / f"{eval_name}_{timestamp}.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)

    results_csv = output_dir / f"{eval_name}_{timestamp}.csv"
    _write_csv(results_csv, results)

    print("\nBatch evaluation complete.")
    print(f"Saved JSON: {results_json}")
    print(f"Saved CSV: {results_csv}")


if __name__ == "__main__":
    main()
