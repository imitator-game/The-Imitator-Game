#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


LEVELS = ("L0", "L1", "L2", "L3")
CAMERAS = ("cam1", "cam2", "cam3", "zed2i")
SCRIPT_DIR = Path(__file__).resolve().parent
REPLAY_SCRIPT = SCRIPT_DIR / "replay_dual_task_multiview.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_task_name(task_name: str) -> str:
    task = task_name.strip()
    if not task:
        raise ValueError("Empty task name is not allowed.")
    if task.endswith("L3-v1"):
        return task.replace("L3-v1", "-v1")
    if task.endswith("-v1"):
        return task
    if task.endswith("L3"):
        task = task[:-2]
    return task + "-v1"


def load_task_list(tasks: list[str], tasks_file: str | None) -> list[str]:
    raw_tasks = list(tasks)
    if tasks_file:
        path = Path(tasks_file).resolve()
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = data.get("tasks", [])
            if not isinstance(data, list):
                raise ValueError(f"Unsupported json task file format: {path}")
            raw_tasks.extend(data)
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                raw_tasks.append(stripped)
    deduped = []
    seen = set()
    for task in raw_tasks:
        normalized = normalize_task_name(task)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    if not deduped:
        raise ValueError("No tasks provided. Use --tasks and/or --tasks-file.")
    return deduped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sequentially generate dual-task trajectories and replay them into per-camera videos."
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=[],
        help="Task env ids or task base names. Example: TwoRobotStirSpoon-v1 TwoRobotPlaceBookBookcase-v1",
    )
    parser.add_argument(
        "--tasks-file",
        help="Path to a text file (one task per line) or a json file containing a task list.",
    )
    parser.add_argument("--out-root", default="demos/dual_task_multiview_batch", help="Root directory for outputs.")
    parser.add_argument("--count", type=int, default=1, help="Number of successful trajectories per task-level.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to invoke ManiSkill generation and replay scripts.",
    )
    parser.add_argument("--sim-backend", default="physx_cpu", help="Replay backend.")
    parser.add_argument("--shader", default="rt", help="Replay shader pack for sensor cameras.")
    parser.add_argument("--cam-width", type=int, default=640, help="Width for cam1/cam2/cam3.")
    parser.add_argument("--cam-height", type=int, default=480, help="Height for cam1/cam2/cam3.")
    parser.add_argument("--zed-width", type=int, default=1280, help="Width for zed2i.")
    parser.add_argument("--zed-height", type=int, default=720, help="Height for zed2i.")
    parser.add_argument(
        "--levels",
        nargs="*",
        default=list(LEVELS),
        choices=list(LEVELS),
        help="Subset of levels to process.",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=list(CAMERAS),
        choices=list(CAMERAS),
        help="Subset of cameras to replay.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip trajectory/video jobs whose outputs already exist.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra subprocess progress.")
    return parser.parse_args()


def derive_env_id(base_env: str, level: str) -> str:
    if level == "L3":
        return base_env.replace("-v1", "L3-v1")
    return base_env


def derive_level_args(level: str) -> list[str]:
    if level == "L1":
        return ["--l1"]
    if level == "L2":
        return ["--l2"]
    return []


def build_traj_paths(out_root: Path, base_env: str, level: str, env_id: str) -> tuple[Path, Path]:
    record_dir = out_root / "trajs" / base_env / level
    traj_dir = record_dir / env_id / "motionplanning"
    return traj_dir / "trajectory.h5", traj_dir / "trajectory.json"


def build_video_output_path(
    video_root: Path,
    base_env: str,
    level: str,
    env_id: str,
    camera_name: str,
    episode_id: int,
) -> Path:
    file_name = f"{level}_{env_id}_{camera_name}_ep{episode_id}.mp4"
    return video_root / camera_name / base_env / file_name


def is_valid_video_output(path: Path, min_size_bytes: int = 1024) -> bool:
    return path.exists() and path.stat().st_size >= min_size_bytes


def load_episode_ids(traj_json_path: Path, count: int) -> list[int]:
    metadata = json.loads(traj_json_path.read_text(encoding="utf-8"))
    episodes = metadata.get("episodes", [])
    if count is not None:
        episodes = episodes[:count]
    return [int(ep["episode_id"]) for ep in episodes]


def init_state(
    tasks: list[str],
    levels: list[str],
    cameras: list[str],
    count: int,
    out_root: Path,
    progress_path: Path,
) -> dict:
    state = {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "out_root": str(out_root),
        "progress_path": str(progress_path),
        "config": {
            "tasks": tasks,
            "levels": levels,
            "cameras": cameras,
            "count": count,
        },
        "summary": {
            "trajectory_total": len(tasks) * len(levels),
            "trajectory_done": 0,
            "trajectory_skipped": 0,
            "video_total": len(tasks) * len(levels) * len(cameras) * count,
            "video_done": 0,
            "video_skipped": 0,
        },
        "current": None,
        "tasks": {},
    }
    for task in tasks:
        state["tasks"][task] = {}
        for level in levels:
            env_id = derive_env_id(task, level)
            state["tasks"][task][level] = {
                "env_id": env_id,
                "trajectory": {"status": "pending", "path": None, "updated_at": None},
                "videos": {},
                "updated_at": None,
            }
    return state


def recalc_summary(state: dict) -> None:
    trajectory_done = 0
    trajectory_skipped = 0
    video_done = 0
    video_skipped = 0
    for _, levels in state["tasks"].items():
        for _, level_entry in levels.items():
            traj_status = level_entry["trajectory"]["status"]
            if traj_status == "done":
                trajectory_done += 1
            elif traj_status == "skipped":
                trajectory_skipped += 1
            for _, video_entry in level_entry["videos"].items():
                if video_entry["status"] == "done":
                    video_done += 1
                elif video_entry["status"] == "skipped":
                    video_skipped += 1
    state["summary"]["trajectory_done"] = trajectory_done
    state["summary"]["trajectory_skipped"] = trajectory_skipped
    state["summary"]["video_done"] = video_done
    state["summary"]["video_skipped"] = video_skipped


def save_state(state: dict, progress_path: Path) -> None:
    state["updated_at"] = now_iso()
    recalc_summary(state)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def append_event(progress_log_path: Path, event: dict) -> None:
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def ensure_video_entries(
    state: dict,
    video_root: Path,
    task: str,
    level: str,
    episode_ids: list[int],
    cameras: list[str],
) -> None:
    level_entry = state["tasks"][task][level]
    env_id = level_entry["env_id"]
    for episode_id in episode_ids:
        for camera_name in cameras:
            key = f"{camera_name}_ep{episode_id}"
            if key in level_entry["videos"]:
                continue
            path = build_video_output_path(
                video_root=video_root,
                base_env=task,
                level=level,
                env_id=env_id,
                camera_name=camera_name,
                episode_id=episode_id,
            )
            level_entry["videos"][key] = {
                "status": "pending",
                "path": str(path),
                "updated_at": None,
            }


def run_command(cmd: list[str], verbose: bool) -> None:
    if verbose:
        print(" ".join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    subprocess.run(cmd, check=True, env=env)


def mark_trajectory(state: dict, task: str, level: str, status: str, traj_path: Path) -> None:
    entry = state["tasks"][task][level]["trajectory"]
    entry["status"] = status
    entry["path"] = str(traj_path)
    entry["updated_at"] = now_iso()
    state["tasks"][task][level]["updated_at"] = now_iso()


def mark_video(state: dict, task: str, level: str, key: str, status: str) -> None:
    entry = state["tasks"][task][level]["videos"][key]
    entry["status"] = status
    entry["updated_at"] = now_iso()
    state["tasks"][task][level]["updated_at"] = now_iso()


def main():
    args = parse_args()
    tasks = load_task_list(args.tasks, args.tasks_file)
    levels = list(args.levels)
    cameras = list(args.cameras)

    out_root = Path(args.out_root).resolve()
    traj_root = out_root / "trajs"
    video_root = out_root / "videos"
    progress_path = out_root / "batch_progress.json"
    progress_log_path = out_root / "batch_progress.jsonl"

    state = init_state(
        tasks=tasks,
        levels=levels,
        cameras=cameras,
        count=args.count,
        out_root=out_root,
        progress_path=progress_path,
    )
    save_state(state, progress_path)

    trajectory_job_total = len(tasks) * len(levels)
    trajectory_job_index = 0

    for task in tasks:
        for level in levels:
            trajectory_job_index += 1
            env_id = derive_env_id(task, level)
            traj_path, traj_json_path = build_traj_paths(
                out_root=out_root,
                base_env=task,
                level=level,
                env_id=env_id,
            )

            state["current"] = {
                "phase": "trajectory",
                "task": task,
                "level": level,
                "env_id": env_id,
                "job_index": trajectory_job_index,
                "job_total": trajectory_job_total,
                "updated_at": now_iso(),
            }
            save_state(state, progress_path)

            if args.skip_existing and traj_path.exists() and traj_json_path.exists():
                print(f"[trajectory {trajectory_job_index}/{trajectory_job_total}] skip existing {traj_path}")
                mark_trajectory(state, task, level, "skipped", traj_path)
                append_event(
                    progress_log_path,
                    {
                        "time": now_iso(),
                        "phase": "trajectory",
                        "task": task,
                        "level": level,
                        "env_id": env_id,
                        "status": "skipped",
                        "path": str(traj_path),
                    },
                )
            else:
                print(f"[trajectory {trajectory_job_index}/{trajectory_job_total}] generate {env_id} {level}")
                cmd = [
                    args.python_bin,
                    "-m",
                    "mani_skill.examples.motionplanning.dual.two_robot_run",
                    "-e",
                    env_id,
                    "-n",
                    str(args.count),
                    "-o",
                    "none",
                    "--only-count-success",
                    "--record-dir",
                    str(traj_root / task / level),
                    "--traj-name",
                    "trajectory",
                ]
                cmd.extend(derive_level_args(level))
                run_command(cmd, verbose=args.verbose)
                mark_trajectory(state, task, level, "done", traj_path)
                append_event(
                    progress_log_path,
                    {
                        "time": now_iso(),
                        "phase": "trajectory",
                        "task": task,
                        "level": level,
                        "env_id": env_id,
                        "status": "done",
                        "path": str(traj_path),
                    },
                )
            save_state(state, progress_path)

            episode_ids = load_episode_ids(traj_json_path, args.count)
            ensure_video_entries(
                state=state,
                video_root=video_root,
                task=task,
                level=level,
                episode_ids=episode_ids,
                cameras=cameras,
            )
            save_state(state, progress_path)

            for episode_id in episode_ids:
                for camera_name in cameras:
                    key = f"{camera_name}_ep{episode_id}"
                    path = Path(state["tasks"][task][level]["videos"][key]["path"])
                    if args.skip_existing and is_valid_video_output(path):
                        mark_video(state, task, level, key, "skipped")
            save_state(state, progress_path)

            state["current"] = {
                "phase": "replay",
                "task": task,
                "level": level,
                "env_id": env_id,
                "updated_at": now_iso(),
            }
            save_state(state, progress_path)

            all_exist = all(
                is_valid_video_output(Path(video_entry["path"]))
                for video_entry in state["tasks"][task][level]["videos"].values()
            )
            if args.skip_existing and all_exist:
                print(f"[replay] skip existing {task} {level}")
                append_event(
                    progress_log_path,
                    {
                        "time": now_iso(),
                        "phase": "replay",
                        "task": task,
                        "level": level,
                        "env_id": env_id,
                        "status": "skipped",
                    },
                )
                continue

            print(f"[replay] render {task} {level}")
            replay_cmd = [
                args.python_bin,
                str(REPLAY_SCRIPT),
                "--traj-path",
                str(traj_path),
                "--level",
                level,
                "--task-name",
                task,
                "--output-dir",
                str(video_root),
                "--count",
                str(args.count),
                "--sim-backend",
                args.sim_backend,
                "--shader",
                args.shader,
                "--cam-width",
                str(args.cam_width),
                "--cam-height",
                str(args.cam_height),
                "--zed-width",
                str(args.zed_width),
                "--zed-height",
                str(args.zed_height),
            ]
            if args.skip_existing:
                replay_cmd.append("--skip-existing")
            if cameras:
                replay_cmd.extend(["--cameras", *cameras])
            if args.verbose:
                replay_cmd.append("--verbose")
            run_command(replay_cmd, verbose=args.verbose)

            for episode_id in episode_ids:
                for camera_name in cameras:
                    key = f"{camera_name}_ep{episode_id}"
                    path = Path(state["tasks"][task][level]["videos"][key]["path"])
                    if not is_valid_video_output(path):
                        raise FileNotFoundError(f"Expected replay output not found: {path}")
                    old_status = state["tasks"][task][level]["videos"][key]["status"]
                    new_status = "skipped" if old_status == "skipped" else "done"
                    mark_video(state, task, level, key, new_status)
            append_event(
                progress_log_path,
                {
                    "time": now_iso(),
                    "phase": "replay",
                    "task": task,
                    "level": level,
                    "env_id": env_id,
                    "status": "done",
                },
            )
            save_state(state, progress_path)

    state["current"] = None
    save_state(state, progress_path)
    print(f"Done. Progress saved to {progress_path}")


if __name__ == "__main__":
    main()
