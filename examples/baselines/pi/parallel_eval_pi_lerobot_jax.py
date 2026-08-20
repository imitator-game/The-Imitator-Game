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

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_dump_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def _parse_ckpt_path(ckpt_path: Path) -> tuple[str, str]:
    parts = list(ckpt_path.resolve().parts)
    if "runs" not in parts:
        raise ValueError(f"checkpoint path must contain 'runs': {ckpt_path}")
    runs_idx = parts.index("runs")
    if runs_idx + 1 >= len(parts):
        raise ValueError(f"checkpoint path format is invalid: {ckpt_path}")
    exp_name = parts[runs_idx + 1]
    ckpt_tag = ckpt_path.name
    return exp_name, ckpt_tag


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
    if match is None:
        return None
    return match.group(1)


def _resolve_eval_l_level_for_task(task_name: str, l3_eval_l_level: str) -> str | None:
    task_level = _extract_level(task_name)
    if task_level == "L3":
        return l3_eval_l_level
    return task_level


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return None


def _relocate_videos(task_dir: Path) -> list[str]:
    videos_root = task_dir / "videos"
    if not videos_root.exists():
        return []
    moved: list[str] = []
    for mp4 in sorted(videos_root.rglob("*.mp4")):
        target = task_dir / mp4.name
        suffix = 0
        while target.exists():
            suffix += 1
            target = task_dir / f"{mp4.stem}_{suffix}{mp4.suffix}"
        shutil.move(str(mp4), str(target))
        moved.append(target.name)
    shutil.rmtree(videos_root, ignore_errors=True)
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


def _count_task_videos(task_dir: Path) -> int:
    return len(list(task_dir.glob("*.mp4")))


def _task_is_done(task_name: str, result_root: Path, num_eval_episodes: int, summary: dict[str, Any]) -> bool:
    task_dir = result_root / task_name
    task_entry = summary.get("tasks", {}).get(task_name)
    if not task_dir.exists() or task_entry is None:
        return False
    if task_entry.get("status") != "ok":
        return False
    if task_entry.get("num_eval_episodes") != num_eval_episodes:
        return False
    raw_shapes = task_entry.get("metrics_raw_shape") or {}
    if raw_shapes:
        completed_episode_counts = [
            shape[0]
            for shape in raw_shapes.values()
            if isinstance(shape, list) and shape and isinstance(shape[0], int)
        ]
        if completed_episode_counts and max(completed_episode_counts) < num_eval_episodes:
            return False
    eval_metrics_file = Path(task_entry.get("eval_metrics_file", task_dir / "eval_metrics.json"))
    if not eval_metrics_file.exists():
        return False
    if task_entry.get("capture_video"):
        video_files = task_entry.get("video_files") or []
        shared_video_files = task_entry.get("shared_video_files") or []
        if not video_files and not shared_video_files and _count_task_videos(task_dir) == 0:
            return False
    return True


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


class ParallelPiJaxEvalScheduler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ckpt_path = Path(args.checkpoint_path).resolve()
        self.exp_name, self.ckpt_tag = _parse_ckpt_path(self.ckpt_path)
        self.sim_cfg = Path(args.sim_dataset_file).resolve()
        self.output_name = args.output_name or f"{self.ckpt_tag}_parallel_eval_result"
        if args.result_root:
            self.result_root = Path(args.result_root).expanduser().resolve()
        else:
            self.result_root = Path("runs") / self.exp_name / self.output_name
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
            "ckpt_exp_name": self.exp_name,
            "ckpt_tag": self.ckpt_tag,
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
                "eval_l_level": task_result.get("eval_l_level"),
                "eval_lr_mirror": task_result.get("eval_lr_mirror"),
                "eval_lr_mirror_robot_pose": task_result.get("eval_lr_mirror_robot_pose"),
                "status": "success" if task_result.get("status") == "ok" else "error",
                "num_eval_episodes": task_result.get("num_eval_episodes"),
                "reward_mode": task_result.get("reward_mode"),
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
        if self.args.gpu_ids:
            gpu_ids = self.args.gpu_ids
        else:
            gpu_ids = list(range(self.args.num_gpus))
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
            "examples.baselines.pi.eval_pi_lerobot_jax",
            "--checkpoint_path",
            str(self.ckpt_path),
            "--env_id",
            task_name,
            "--output_dir",
            str(task_dir),
            "--human_root",
            self.args.human_root,
            "--sim_root",
            self.args.sim_root,
            "--human_dataset_file",
            self.args.human_dataset_file,
            "--sim_dataset_file",
            self.args.sim_dataset_file,
            "--sim_state_type",
            self.args.sim_state_type,
            "--human_task_desc_file",
            self.args.human_task_desc_file,
            "--sim_task_desc_file",
            self.args.sim_task_desc_file,
            "--task_mapping_file",
            self.args.task_mapping_file,
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
            "--num_diffusion_steps",
            str(self.args.num_diffusion_steps),
            "--shader",
            self.args.shader,
            "--action_dim",
            str(self.args.action_dim),
            "--pred_horizon",
            str(self.args.pred_horizon),
            "--max_token_len",
            str(self.args.max_token_len),
            "--processor_name_or_path",
            self.args.processor_name_or_path,
            "--l3_eval_l_level",
            self.args.l3_eval_l_level,
            "--eval_lr_mirror",
            self.args.eval_lr_mirror,
            "--eval_lr_mirror_robot_pose",
            self.args.eval_lr_mirror_robot_pose,
            "--dtw_band_ratio",
            str(self.args.dtw_band_ratio),
            "--cameras",
            *self.args.cameras,
        ]
        if self.args.processor_local_files_only:
            cmd.append("--processor_local_files_only")
        if self.args.checkpoint_step is not None:
            cmd.extend(["--checkpoint_step", str(self.args.checkpoint_step)])
        if self.args.capture_video:
            cmd.append("--capture_video")
        if self.args.include_depth:
            cmd.append("--include_depth")
        if self.args.vla:
            cmd.append("--vla")
        if self.args.pi05:
            cmd.append("--pi05")
        if self.args.skip_masked_cameras:
            cmd.append("--skip_masked_cameras")
        if self.args.use_prefix_kv_cache:
            cmd.append("--use_prefix_kv_cache")
        if self.args.compute_dtw:
            cmd.append("--compute_dtw")
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
            "eval_l_level": (eval_result or {}).get("eval_l_level")
            or _resolve_eval_l_level_for_task(task_name, self.args.l3_eval_l_level),
            "eval_lr_mirror": (eval_result or {}).get("eval_lr_mirror", self.args.eval_lr_mirror),
            "eval_lr_mirror_robot_pose": (eval_result or {}).get(
                "eval_lr_mirror_robot_pose",
                self.args.eval_lr_mirror_robot_pose,
            ),
            "return_code": return_code,
            "status": "ok" if return_code == 0 else "failed",
            "reward_mode": self.args.reward_mode,
            "num_eval_episodes": self.args.num_eval_episodes,
            "num_eval_envs": self.args.num_eval_envs,
            "capture_video": self.args.capture_video,
            "metrics_mean": (eval_result or {}).get("metrics_mean", {}),
            "metrics_raw_shape": (eval_result or {}).get("metrics_raw_shape", {}),
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
            if not self.args.force_rerun and _task_is_done(
                task_name, self.result_root, self.args.num_eval_episodes, self.summary
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

        previous_total = self.summary.get("tasks_total")
        if isinstance(previous_total, int) and previous_total > len(tasks):
            self.summary["tasks_total"] = previous_total
        else:
            self.summary["tasks_total"] = len(tasks)
        self._save_summary_locked()

        pending_tasks: list[str] = []
        skipped = 0
        for task_name in tasks:
            if not self.args.force_rerun and _task_is_done(
                task_name, self.result_root, self.args.num_eval_episodes, self.summary
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
        self.log(f"Parallel batch eval finished. Summary: {self.summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel batch evaluate JAX Pi checkpoints by iterating tasks from sim config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--checkpoint_step", type=int, default=None)
    parser.add_argument("--output_name", default=None)
    parser.add_argument(
        "--result_root",
        default=None,
        help="Explicit directory for all batch eval outputs. Defaults to runs/<checkpoint_exp>/<output_name>.",
    )
    parser.add_argument("--sim_dataset_file", required=True)
    parser.add_argument("--sim_state_type", default="qpos", choices=["qpos", "eepos", "mixpos"])
    parser.add_argument("--human_dataset_file", required=True)
    parser.add_argument("--human_root", required=True)
    parser.add_argument("--sim_root", required=True)
    parser.add_argument("--human_task_desc_file", default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    parser.add_argument("--sim_task_desc_file", default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")
    parser.add_argument("--task_mapping_file", default="examples/baselines/lerobot_dataset/task_mapping.json")
    parser.add_argument("--control_mode", default="pd_joint_pos")
    parser.add_argument("--obs_mode", default="rgb")
    parser.add_argument("--reward_mode", default="dense", choices=["sparse", "dense", "normalized_dense", "none"])
    parser.add_argument("--sim_backend", default="physx_cpu")
    parser.add_argument("--shader", default="rt-fast")
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--num_eval_episodes", type=int, default=5)
    parser.add_argument("--num_eval_envs", type=int, default=1)
    parser.add_argument("--num_diffusion_steps", type=int, default=10)
    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--pred_horizon", type=int, default=50)
    parser.add_argument("--max_token_len", type=int, default=200)
    parser.add_argument("--processor_name_or_path", default="google/paligemma-3b-pt-224")
    parser.add_argument("--processor_local_files_only", action="store_true")
    parser.add_argument(
        "--l3_eval_l_level",
        default="L0",
        choices=["L0", "L1", "L2", "L3"],
        help="Forwarded L-level flags for L3 env_ids. The child still launches the L3 gym env.",
    )
    parser.add_argument(
        "--eval_lr_mirror",
        default="auto",
        choices=["auto", "true", "false"],
        help="Forwarded tabletop left-right mirror override.",
    )
    parser.add_argument(
        "--eval_lr_mirror_robot_pose",
        default="false",
        choices=["auto", "true", "false"],
        help="Forwarded robot pose mirror override.",
    )
    parser.add_argument("--cameras", type=str, nargs="+", default=["zed2i"])
    parser.add_argument("--limit_tasks", type=int, default=None)
    parser.add_argument("--only_tasks", nargs="*", default=None)
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Re-run selected tasks even if batch_eval_summary marks them complete.",
    )
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--include_depth", action="store_true", default=False)
    parser.add_argument("--vla", action="store_true", default=False)
    parser.add_argument("--pi05", action="store_true")
    parser.add_argument(
        "--skip_masked_cameras",
        action="store_true",
        help="Forward --skip_masked_cameras to eval_pi_lerobot_jax.py.",
    )
    parser.add_argument(
        "--use_prefix_kv_cache",
        action="store_true",
        help="Forward --use_prefix_kv_cache to eval_pi_lerobot_jax.py for config parity.",
    )
    parser.add_argument("--compute_dtw", action="store_true", help="Forward --compute_dtw to eval_pi_lerobot_jax.py.")
    parser.add_argument("--dtw_band_ratio", type=float, default=0.15, help="Forwarded TSS/nDTW band ratio.")
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_procs_per_gpu", type=int, default=1)
    parser.add_argument(
        "--cpu_threads_per_worker",
        type=int,
        default=None,
        help="Set OMP/MKL/OpenBLAS/NumExpr thread limits for each eval subprocess.",
    )
    parser.add_argument("--min_free_mem_gb", type=float, default=8.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--launch_stagger_sec", type=float, default=1.0)
    parser.add_argument(
        "--extra_eval_arg",
        action="append",
        default=[],
        help="Additional raw arg forwarded to eval_pi_lerobot_jax.py. Repeat this flag for multiple arguments.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ParallelPiJaxEvalScheduler(parse_args()).run()
