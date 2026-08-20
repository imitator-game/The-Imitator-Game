#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_dump_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _load_tasks_from_sim_config(sim_cfg_path: Path) -> list[str]:
    with sim_cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"sim config must be a list: {sim_cfg_path}")

    tasks: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        task_id = item.get("repo_id") or item.get("env_id") or item.get("root")
        if isinstance(task_id, str) and task_id:
            task_name = Path(task_id).name
            tasks.append(task_name if re.match(r"^L[0-3]_", task_name) else task_id)

    uniq: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        if task not in seen:
            uniq.append(task)
            seen.add(task)
    return uniq


def _extract_level(task_name: str) -> str | None:
    match = re.match(r"^(L[0-3])_", Path(task_name).name)
    return match.group(1) if match else None


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _relocate_videos(task_dir: Path) -> list[str]:
    moved: list[str] = []
    for mp4 in sorted(task_dir.rglob("*.mp4")):
        if mp4.parent == task_dir:
            continue
        target = task_dir / mp4.name
        suffix = 0
        while target.exists():
            suffix += 1
            target = task_dir / f"{mp4.stem}_{suffix}{mp4.suffix}"
        shutil.move(str(mp4), str(target))
        moved.append(target.name)
    for child in ("videos", "online_eval_videos"):
        shutil.rmtree(task_dir / child, ignore_errors=True)
    return moved


def _mirror_videos_to_shared_dir(task_dir: Path, shared_video_dir: Path) -> list[str]:
    shared_video_dir.mkdir(parents=True, exist_ok=True)
    mirrored: list[str] = []
    for mp4 in sorted(task_dir.glob("*.mp4")):
        target = shared_video_dir / mp4.name
        suffix = 0
        while target.exists():
            suffix += 1
            target = shared_video_dir / f"{mp4.stem}_{suffix}{mp4.suffix}"
        try:
            os.link(mp4, target)
        except OSError:
            shutil.copy2(mp4, target)
        mirrored.append(target.name)
    return mirrored


def _task_is_done(
    task_name: str,
    result_root: Path,
    num_eval_episodes: int,
    summary: dict[str, Any],
    eval_lr_mirror: str,
    eval_lr_mirror_robot_pose: str,
) -> bool:
    task_dir = result_root / task_name
    task_entry = summary.get("tasks", {}).get(task_name)
    if not task_dir.exists() or task_entry is None:
        return False
    if task_entry.get("status") != "ok":
        return False
    if task_entry.get("num_eval_episodes") != num_eval_episodes:
        return False
    if task_entry.get("eval_lr_mirror") != eval_lr_mirror:
        return False
    if task_entry.get("eval_lr_mirror_robot_pose") != eval_lr_mirror_robot_pose:
        return False
    eval_metrics_file = Path(task_entry.get("eval_metrics_file", task_dir / "eval_metrics.json"))
    return eval_metrics_file.exists()


def _get_gpu_free_mem_gb(gpu_id: int) -> float:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--id",
                str(gpu_id),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return float(out.splitlines()[0]) / 1024.0
    except Exception:
        return 0.0


@dataclass
class TaskSpec:
    task_name: str
    retry_count: int = 0


class ParallelGr00tEvalScheduler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ckpt_path = Path(args.checkpoint_path).expanduser().resolve()
        self.sim_cfg = Path(args.sim_dataset_file).expanduser().resolve()
        self.result_root = Path(args.result_root).expanduser().resolve()
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.result_root / "batch_eval_summary.json"
        self.shared_videos_root = self.result_root / "videos"
        self.shared_videos_root.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.result_root / f"scheduler_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.scheduler_log_path = self.log_dir / "scheduler.log"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_name = self.sim_cfg.stem
        self.incremental_json_path = self.result_root / f"{self.eval_name}_{self.timestamp}.json"
        self.final_csv_path = self.result_root / f"{self.eval_name}_{self.timestamp}.csv"
        self.log_lock = threading.Lock()
        self.summary_lock = threading.Lock()
        self.queue: queue.Queue[TaskSpec] = queue.Queue()
        self.summary = self._load_or_init_summary()

    def log(self, msg: str, level: str = "INFO") -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}"
        with self.log_lock:
            print(line, flush=True)
            with self.scheduler_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _load_or_init_summary(self) -> dict[str, Any]:
        if self.summary_path.exists():
            with self.summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            summary.setdefault("tasks", {})
            return summary
        summary: dict[str, Any] = {
            "checkpoint_path": str(self.ckpt_path),
            "sim_dataset_file": str(self.sim_cfg),
            "created_at": _now(),
            "tasks_total": 0,
            "tasks_done": 0,
            "tasks": {},
        }
        _atomic_dump_json(self.summary_path, summary)
        return summary

    def _build_results_locked(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for task_name, task_result in sorted(self.summary.get("tasks", {}).items()):
            metrics_mean = task_result.get("metrics_mean", {})
            record = {
                "env_id": task_name,
                "level": task_result.get("level"),
                "status": "success" if task_result.get("status") == "ok" else "error",
                "num_eval_episodes": task_result.get("num_eval_episodes"),
                "reward_mode": task_result.get("reward_mode"),
                "eval_lr_mirror": task_result.get("eval_lr_mirror"),
                "eval_lr_mirror_robot_pose": task_result.get("eval_lr_mirror_robot_pose"),
                "timestamp": task_result.get("updated_at"),
                "success_once_mean": metrics_mean.get("success_once"),
                "success_at_end_mean": metrics_mean.get("success_at_end"),
                "return_mean": metrics_mean.get("return"),
                "reward_mean": metrics_mean.get("reward"),
                "episode_len_mean": metrics_mean.get("episode_len"),
                "tss_success_mean": metrics_mean.get("tss_success"),
                "tss_fail_mean": metrics_mean.get("tss_fail"),
                "ndtw_success_mean": metrics_mean.get("ndtw_success"),
                "ndtw_fail_mean": metrics_mean.get("ndtw_fail"),
                "log_file": task_result.get("log_file"),
                "video_files": task_result.get("shared_video_files", task_result.get("video_files", [])),
            }
            if task_result.get("status") != "ok":
                record["error"] = task_result.get("error", f"return_code={task_result.get('return_code')}")
            records.append(record)
        return records

    def _save_summary_locked(self) -> None:
        self.summary["tasks_done"] = len(self.summary.get("tasks", {}))
        self.summary["incremental_json"] = str(self.incremental_json_path)
        self.summary["final_csv"] = str(self.final_csv_path)
        _atomic_dump_json(self.summary_path, self.summary)
        records = self._build_results_locked()
        _atomic_dump_json(self.incremental_json_path, records)
        pd.DataFrame(records).to_csv(self.final_csv_path, index=False)

    def _resolve_gpu_ids(self) -> list[int]:
        if self.args.no_cuda:
            return [-1]
        gpu_ids = self.args.gpu_ids if self.args.gpu_ids else list(range(self.args.num_gpus))
        usable: list[int] = []
        for gpu_id in gpu_ids:
            free_gb = _get_gpu_free_mem_gb(gpu_id)
            if free_gb >= self.args.min_free_mem_gb:
                usable.append(gpu_id)
            else:
                self.log(
                    f"Skip GPU {gpu_id}: free memory {free_gb:.1f} GB < required {self.args.min_free_mem_gb:.1f} GB",
                    level="WARNING",
                )
        if not usable:
            raise RuntimeError("No GPUs satisfy min_free_mem_gb; use --gpu_ids/--min_free_mem_gb or --no_cuda")
        return usable

    def _build_eval_cmd(self, task_name: str, task_dir: Path) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "gr00t.eval.eval_policygen",
            "--checkpoint_path",
            str(self.ckpt_path),
            "--env_id",
            task_name,
            "--output_dir",
            str(task_dir),
            "--embodiment_tag",
            self.args.embodiment_tag,
            "--language_source",
            self.args.language_source,
            "--sim_backend",
            self.args.sim_backend,
            "--control_mode",
            self.args.control_mode,
            "--obs_mode",
            self.args.obs_mode,
            "--reward_mode",
            self.args.reward_mode,
            "--max_episode_steps",
            str(self.args.max_episode_steps),
            "--num_eval_envs",
            str(self.args.num_eval_envs),
            "--num_eval_episodes",
            str(self.args.num_eval_episodes),
            "--shader",
            self.args.shader,
            "--dtype",
            self.args.dtype,
            "--eval_lr_mirror",
            self.args.eval_lr_mirror,
            "--eval_lr_mirror_robot_pose",
            self.args.eval_lr_mirror_robot_pose,
        ]
        if self.args.processor_path:
            cmd.extend(["--processor_path", self.args.processor_path])
        if self.args.task_mapping_path:
            cmd.extend(["--task_mapping_path", self.args.task_mapping_path])
        if self.args.human_desc_path:
            cmd.extend(["--human_desc_path", self.args.human_desc_path])
        if self.args.sim_desc_path:
            cmd.extend(["--sim_desc_path", self.args.sim_desc_path])
        if self.args.capture_video:
            cmd.append("--capture_video")
        if self.args.compute_dtw:
            cmd.extend(
                [
                    "--compute_dtw",
                    "--dtw_band_ratio",
                    str(self.args.dtw_band_ratio),
                    "--sim_dataset_file",
                    str(self.sim_cfg),
                    "--sim_root",
                    self.args.sim_root,
                    "--dtw_action_key",
                    self.args.dtw_action_key,
                ]
            )
        for extra in self.args.extra_eval_arg:
            cmd.append(extra)
        return cmd

    def _write_task_result(self, task_name: str, task_result: dict[str, Any]) -> None:
        with self.summary_lock:
            self.summary.setdefault("tasks", {})[task_name] = task_result
            self._save_summary_locked()

    def _run_one_task(self, task_name: str, worker_name: str, gpu_id: int) -> bool:
        task_dir = self.result_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        log_path = task_dir / "eval.log"
        cmd = self._build_eval_cmd(task_name, task_dir)
        env = os.environ.copy()
        if gpu_id >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if self.args.cpu_threads_per_worker is not None:
            threads = str(self.args.cpu_threads_per_worker)
            env["OMP_NUM_THREADS"] = threads
            env["MKL_NUM_THREADS"] = threads
            env["OPENBLAS_NUM_THREADS"] = threads
            env["NUMEXPR_NUM_THREADS"] = threads
            env["VECLIB_MAXIMUM_THREADS"] = threads

        self.log(f"{worker_name} start task={task_name} gpu={gpu_id}")
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                prefixed = f"[{worker_name}|{task_name}] {line}"
                sys.stdout.write(prefixed)
                log_f.write(line)
            return_code = proc.wait()

        video_files = _relocate_videos(task_dir)
        shared_video_files = _mirror_videos_to_shared_dir(task_dir, self.shared_videos_root / task_name)
        eval_result = _load_json_if_exists(task_dir / "eval_metrics.json")
        task_result = {
            "task_name": task_name,
            "env_id": task_name,
            "level": _extract_level(task_name),
            "return_code": return_code,
            "status": "ok" if return_code == 0 else "failed",
            "reward_mode": self.args.reward_mode,
            "num_eval_episodes": self.args.num_eval_episodes,
            "num_eval_envs": self.args.num_eval_envs,
            "capture_video": self.args.capture_video,
            "compute_dtw": self.args.compute_dtw,
            "dtw_band_ratio": self.args.dtw_band_ratio if self.args.compute_dtw else None,
            "dtw_action_key": self.args.dtw_action_key if self.args.compute_dtw else None,
            "eval_lr_mirror": self.args.eval_lr_mirror,
            "eval_lr_mirror_robot_pose": self.args.eval_lr_mirror_robot_pose,
            "metrics_mean": (eval_result or {}).get("metrics_mean", {}),
            "checkpoint_step": (eval_result or {}).get("checkpoint_step"),
            "video_files": video_files,
            "shared_video_files": shared_video_files,
            "log_file": str(log_path),
            "eval_metrics_file": str(task_dir / "eval_metrics.json"),
            "updated_at": _now(),
        }
        if return_code != 0:
            task_result["error"] = f"return_code={return_code}"

        self._write_task_result(task_name, task_result)
        if return_code == 0:
            self.log(
                f"{worker_name} finished task={task_name} success_at_end={task_result['metrics_mean'].get('success_at_end')} videos={len(video_files)}"
            )
            return True
        self.log(f"{worker_name} failed task={task_name} return_code={return_code}", level="ERROR")
        return False

    def _worker_loop(self, worker_id: int, gpu_id: int) -> None:
        worker_name = f"W{worker_id}"
        while True:
            try:
                spec = self.queue.get_nowait()
            except queue.Empty:
                return

            task_name = spec.task_name
            if _task_is_done(
                task_name,
                self.result_root,
                self.args.num_eval_episodes,
                self.summary,
                self.args.eval_lr_mirror,
                self.args.eval_lr_mirror_robot_pose,
            ):
                self.log(f"{worker_name} skip completed task={task_name}")
                self.queue.task_done()
                continue

            ok = self._run_one_task(task_name, worker_name, gpu_id)
            if not ok and spec.retry_count < self.args.max_retries:
                self.log(
                    f"{worker_name} requeue task={task_name} retry={spec.retry_count + 1}/{self.args.max_retries}",
                    level="WARNING",
                )
                self.queue.put(TaskSpec(task_name=task_name, retry_count=spec.retry_count + 1))
            self.queue.task_done()

    def run(self) -> None:
        tasks = _load_tasks_from_sim_config(self.sim_cfg)
        if self.args.only_tasks:
            selected = set(self.args.only_tasks)
            tasks = [task for task in tasks if task in selected]
        if self.args.limit_tasks is not None:
            tasks = tasks[: self.args.limit_tasks]
        if not tasks:
            raise ValueError(f"No tasks found from config: {self.sim_cfg}")

        previous_tasks_total = int(self.summary.get("tasks_total") or 0)
        if self.args.only_tasks and previous_tasks_total > len(tasks):
            self.summary["tasks_total"] = previous_tasks_total
        else:
            self.summary["tasks_total"] = len(tasks)
        self._save_summary_locked()

        pending_tasks: list[str] = []
        skipped = 0
        for task_name in tasks:
            if _task_is_done(
                task_name,
                self.result_root,
                self.args.num_eval_episodes,
                self.summary,
                self.args.eval_lr_mirror,
                self.args.eval_lr_mirror_robot_pose,
            ):
                skipped += 1
            else:
                pending_tasks.append(task_name)

        self.log(f"Total tasks: {len(tasks)} | skipped: {skipped} | pending: {len(pending_tasks)}")
        if not pending_tasks:
            self.log(f"All tasks already evaluated. Summary: {self.summary_path}")
            return

        for task_name in pending_tasks:
            self.queue.put(TaskSpec(task_name=task_name))

        gpu_ids = self._resolve_gpu_ids()
        if self.args.no_cuda:
            worker_specs = [(0, -1)]
        else:
            worker_specs: list[tuple[int, int]] = []
            next_worker_id = 0
            for gpu_id in gpu_ids:
                for _ in range(self.args.max_procs_per_gpu):
                    worker_specs.append((next_worker_id, gpu_id))
                    next_worker_id += 1
            worker_specs = worker_specs[: min(len(worker_specs), len(pending_tasks))]

        self.log(f"Using workers: {worker_specs}")
        threads: list[threading.Thread] = []
        for worker_id, gpu_id in worker_specs:
            thread = threading.Thread(target=self._worker_loop, args=(worker_id, gpu_id), daemon=True)
            thread.start()
            threads.append(thread)
            time.sleep(self.args.launch_stagger_sec)

        for thread in threads:
            thread.join()

        self.summary["finished_at"] = _now()
        self._save_summary_locked()
        failed_tasks = [
            task_name
            for task_name, task_result in self.summary.get("tasks", {}).items()
            if task_result.get("status") != "ok"
        ]
        self.log(f"Parallel GR00T batch eval finished. Summary: {self.summary_path}")
        if failed_tasks:
            failed_preview = ", ".join(sorted(failed_tasks)[:10])
            if len(failed_tasks) > 10:
                failed_preview += f", ... (+{len(failed_tasks) - 10} more)"
            raise RuntimeError(f"{len(failed_tasks)} GR00T eval task(s) failed: {failed_preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel batch evaluate GR00T policy-generation checkpoints by iterating tasks from sim config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--processor_path", default=None)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--sim_dataset_file", required=True)
    parser.add_argument("--human_dataset_file", default=None)
    parser.add_argument("--embodiment_tag", default="NEW_EMBODIMENT")
    parser.add_argument("--language_source", default="human_desc", choices=["task", "human_desc", "sim_desc"])
    parser.add_argument("--task_mapping_path", default=None)
    parser.add_argument("--human_desc_path", default=None)
    parser.add_argument("--sim_desc_path", default=None)
    parser.add_argument("--control_mode", default="pd_joint_pos")
    parser.add_argument("--obs_mode", default="rgb")
    parser.add_argument("--reward_mode", default="dense", choices=["sparse", "dense", "normalized_dense", "none"])
    parser.add_argument("--sim_backend", default="physx_cpu")
    parser.add_argument("--shader", default="rt-fast")
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--num_eval_envs", type=int, default=5)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument(
        "--eval_lr_mirror",
        default="auto",
        choices=["auto", "true", "false"],
        help="Forward --eval_lr_mirror to eval_policygen.py.",
    )
    parser.add_argument(
        "--eval_lr_mirror_robot_pose",
        default="false",
        choices=["auto", "true", "false"],
        help="Forward --eval_lr_mirror_robot_pose to eval_policygen.py.",
    )
    parser.add_argument("--limit_tasks", type=int, default=None)
    parser.add_argument("--only_tasks", nargs="*", default=None)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--compute_dtw", action="store_true", help="Forward --compute_dtw to eval_policygen.py.")
    parser.add_argument("--dtw_band_ratio", type=float, default=0.15)
    parser.add_argument("--sim_root", default="demos/imitator_data")
    parser.add_argument("--dtw_action_key", default="action.qpos_gripper_actions")
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_procs_per_gpu", type=int, default=1)
    parser.add_argument("--cpu_threads_per_worker", type=int, default=None)
    parser.add_argument("--min_free_mem_gb", type=float, default=20.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--launch_stagger_sec", type=float, default=1.0)
    parser.add_argument(
        "--extra_eval_arg",
        action="append",
        default=[],
        help="Additional raw arg forwarded to eval_policygen.py. Repeat this flag for multiple arguments.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ParallelGr00tEvalScheduler(parse_args()).run()
