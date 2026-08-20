#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path

import gymnasium as gym
import h5py
import imageio

import mani_skill.envs  # noqa: F401  Ensures env registration.
from mani_skill.envs.tasks.tabletop.utils.dual_task_camera_utils import (
    iter_dual_task_cameras,
    patch_dual_task_camera_defaults,
    configure_dual_task_level,
)
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a dual-task trajectory h5 and export one mp4 per camera view."
    )
    parser.add_argument("--traj-path", required=True, help="Path to the source trajectory .h5 file.")
    parser.add_argument(
        "--level",
        required=True,
        choices=["L0", "L1", "L2", "L3"],
        help="Task level used when the source trajectory was generated.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for rendered videos. Output layout is <root>/<camera>/<task>/video.mp4.",
    )
    parser.add_argument("--env-id", help="Optional env id override. Defaults to env_info.env_id from the json metadata.")
    parser.add_argument(
        "--task-name",
        help="Optional logical task name used for output directories. Defaults to env_id.",
    )
    parser.add_argument("--count", type=int, default=None, help="Replay only the first N episodes.")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument("--sim-backend", default="physx_cpu", help="Simulation backend for replay.")
    parser.add_argument("--shader", default="default", help="Shader pack for sensor cameras.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["cam1", "cam2", "cam3", "zed2i"],
        help="Camera names to export.",
    )
    parser.add_argument("--cam-width", type=int, default=640, help="Width for cam1/cam2/cam3.")
    parser.add_argument("--cam-height", type=int, default=480, help="Height for cam1/cam2/cam3.")
    parser.add_argument("--zed-width", type=int, default=1280, help="Width for zed2i.")
    parser.add_argument("--zed-height", type=int, default=720, help="Height for zed2i.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a video if the target mp4 already exists.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-episode progress.")
    return parser.parse_args()


def normalize_episode_reset(episode: dict) -> tuple[int | None, dict]:
    reset_kwargs = copy.deepcopy(episode.get("reset_kwargs", {}))
    seed = reset_kwargs.pop("seed", episode.get("episode_seed"))
    if isinstance(seed, list):
        if len(seed) != 1:
            raise ValueError(f"Ambiguous reset seed for episode {episode.get('episode_id')}: {seed}")
        seed = seed[0]
    return seed, reset_kwargs


def extract_rgb_image(sensor_images_for_camera: dict):
    rgb_key = None
    for key in sensor_images_for_camera.keys():
        lowered = key.lower()
        if "rgb" in lowered or "color" in lowered:
            rgb_key = key
            break
    if rgb_key is None:
        raise KeyError(f"Could not find an RGB image in sensor outputs: {list(sensor_images_for_camera.keys())}")
    image = common.to_numpy(sensor_images_for_camera[rgb_key])
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    return image


def infer_env_id(metadata: dict, args_env_id: str | None) -> str:
    if args_env_id:
        return args_env_id
    env_info = metadata.get("env_info", {})
    env_id = env_info.get("env_id")
    if not env_id:
        raise ValueError("Could not infer env_id from trajectory metadata; please pass --env-id explicitly.")
    return env_id


def normalize_control_mode(control_mode) -> str | None:
    if not control_mode:
        return None
    if isinstance(control_mode, str):
        return control_mode
    if isinstance(control_mode, dict):
        control_modes = list(control_mode.values())
        if not control_modes:
            return None
        if not all(mode == control_modes[0] for mode in control_modes):
            raise ValueError(
                "Replay currently expects all robots to share the same control mode, "
                f"but got {control_mode}."
            )
        return control_modes[0]
    raise TypeError(f"Unsupported control_mode type: {type(control_mode).__name__}")


def infer_control_mode(metadata: dict) -> str:
    episodes = metadata.get("episodes", [])
    if episodes:
        control_mode = normalize_control_mode(episodes[0].get("control_mode"))
        if control_mode:
            return control_mode
    env_info = metadata.get("env_info", {})
    env_kwargs = env_info.get("env_kwargs", {})
    return normalize_control_mode(env_kwargs.get("control_mode")) or "pd_joint_pos"


def build_video_output_path(
    output_root: Path,
    camera_name: str,
    task_name: str,
    video_name: str,
) -> Path:
    return (
        output_root.resolve()
        / camera_name
        / task_name
        / (video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
    )


def is_valid_video_output(path: Path, min_size_bytes: int = 1024) -> bool:
    return path.exists() and path.stat().st_size >= min_size_bytes


def stream_single_camera_video(
    env,
    env_states,
    camera_name: str,
    output_path: Path,
    fps: int,
    verbose: bool = False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=fps, quality=5)
    frame_count = 0
    try:
        for frame_idx, env_state in enumerate(env_states):
            env.unwrapped.set_state_dict(env_state)
            sensor_images = env.unwrapped.get_sensor_images()
            writer.append_data(extract_rgb_image(sensor_images[camera_name]))
            frame_count += 1
            if verbose and (frame_idx == 0 or (frame_idx + 1) % 100 == 0):
                print(
                    f"  [{camera_name}] wrote frame {frame_idx + 1}/{len(env_states)}",
                    flush=True,
                )
    finally:
        writer.close()
    if frame_count == 0:
        raise RuntimeError(f"No frames were written for {camera_name} -> {output_path}")
    return frame_count


def main():
    args = parse_args()
    traj_path = Path(args.traj_path).resolve()
    json_path = traj_path.with_suffix(".json")
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Trajectory metadata json not found: {json_path}")

    cameras = iter_dual_task_cameras(args.cameras)

    with json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    env_id = infer_env_id(metadata, args.env_id)
    task_name = args.task_name or env_id
    control_mode = infer_control_mode(metadata)
    episodes = metadata.get("episodes", [])
    if args.count is not None:
        episodes = episodes[: args.count]
    if not episodes:
        raise ValueError(f"No episodes found in {json_path}")
    total_video_jobs = len(episodes) * len(cameras)
    completed_video_jobs = 0

    with h5py.File(traj_path, "r") as trajectory_h5:
        for episode in episodes:
            episode_id = episode["episode_id"]
            traj_id = f"traj_{episode_id}"
            if traj_id not in trajectory_h5:
                raise KeyError(f"{traj_id} not found in trajectory file {traj_path}")
            if "env_states" not in trajectory_h5[traj_id]:
                raise KeyError(f"{traj_id} does not contain env_states; cannot replay by state.")

            seed, reset_kwargs = normalize_episode_reset(episode)
            env_states = trajectory_utils.dict_to_list_of_dicts(
                trajectory_h5[traj_id]["env_states"]
            )

            for camera_name in cameras:
                completed_video_jobs += 1
                patched_classes = patch_dual_task_camera_defaults(
                    cam_width=args.cam_width,
                    cam_height=args.cam_height,
                    zed_width=args.zed_width,
                    zed_height=args.zed_height,
                    camera_names=[camera_name],
                )
                configure_dual_task_level(args.level)

                video_name = f"{args.level}_{env_id}_{camera_name}_ep{episode_id}"
                video_output_path = build_video_output_path(
                    output_root=Path(args.output_dir),
                    camera_name=camera_name,
                    task_name=task_name,
                    video_name=video_name,
                )
                if args.skip_existing and is_valid_video_output(video_output_path):
                    print(
                        f"[video {completed_video_jobs}/{total_video_jobs}] "
                        f"skip existing {video_output_path}",
                        flush=True,
                    )
                    continue
                if video_output_path.exists() and not is_valid_video_output(video_output_path):
                    if args.verbose:
                        print(
                            f"[video {completed_video_jobs}/{total_video_jobs}] "
                            f"remove incomplete output {video_output_path}",
                            flush=True,
                        )
                    video_output_path.unlink()

                if args.verbose:
                    print(
                        f"[video {completed_video_jobs}/{total_video_jobs}] "
                        f"replay episode {episode_id} for {camera_name} "
                        f"with env_id={env_id} ({len(patched_classes)} patched env classes)",
                        flush=True,
                    )

                env = gym.make(
                    env_id,
                    obs_mode="rgb",
                    control_mode=control_mode,
                    reward_mode="none",
                    render_mode="sensors",
                    sim_backend=args.sim_backend,
                    num_envs=1,
                    sensor_configs=dict(shader_pack=args.shader),
                    human_render_camera_configs=dict(shader_pack=args.shader),
                    viewer_camera_configs=dict(shader_pack=args.shader),
                )
                try:
                    env.reset(seed=seed, **reset_kwargs)
                    frame_count = stream_single_camera_video(
                        env=env,
                        env_states=env_states,
                        camera_name=camera_name,
                        output_path=video_output_path,
                        fps=args.fps,
                        verbose=args.verbose,
                    )
                except Exception:
                    if video_output_path.exists():
                        video_output_path.unlink()
                    raise
                finally:
                    env.close()
                if args.verbose:
                    print(
                        f"[video {completed_video_jobs}/{total_video_jobs}] "
                        f"finished {video_output_path} ({frame_count} frames)",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
