#!/usr/bin/env python3
"""
Parallel evaluation scheduler for RDT paired multi-task online eval.

- Reads tasks from a sim dataset config JSON
- Launches multiple eval_rdt_* subprocesses across available GPUs
- Passes the full env_id/task id to each subprocess
- Skips tasks that already have enough videos and a successful summary
- Writes incremental JSON/CSV summaries under runs/<exp>/<output_name>/
"""

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_dump_json(path: Path, payload: Any) -> None:
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
    if runs_idx + 3 >= len(parts):
        raise ValueError(f"checkpoint path format is invalid: {ckpt_path}")
    exp_name = parts[runs_idx + 1]
    return exp_name, ckpt_path.stem


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
            tasks.append(task_id)
    uniq: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        if task not in seen:
            uniq.append(task)
            seen.add(task)
    return uniq


def _extract_level(task_name: str) -> str | None:
    match = re.match(r"^(L[0-3])_", task_name)
    return match.group(1) if match else None


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
    if not task_dir.exists():
        return 0
    return len(list(task_dir.glob("*.mp4")))


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


def _sanitize_eval_extra_args(args: list[str]) -> list[str]:
    """Drop argparse REMAINDER separators before forwarding to tyro child CLIs."""
    sanitized = list(args)
    while sanitized and sanitized[0] == "--":
        sanitized.pop(0)
    return sanitized


def _capture_video_enabled(eval_extra_args: list[str]) -> bool:
    return "--no-capture-video" not in _sanitize_eval_extra_args(eval_extra_args)


def _extract_reward_mode(eval_extra_args: list[str]) -> str | None:
    sanitized = _sanitize_eval_extra_args(eval_extra_args)
    for idx, arg in enumerate(sanitized):
        if arg == "--reward-mode" and idx + 1 < len(sanitized):
            return sanitized[idx + 1]
    return None


@dataclass
class TaskSpec:
    task_name: str
    retry_count: int = 0


class ParallelRDTEvalScheduler:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ckpt_path = Path(args.checkpoint_path).resolve()
        self.exp_name, self.ckpt_tag = _parse_ckpt_path(self.ckpt_path)
        self.sim_cfg = Path(args.lerobot_sim_dataset_file).resolve()
        self.result_root = Path("runs") / self.exp_name / (args.output_name or f"{self.ckpt_tag}_parallel_eval")
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
        self.print_lock = threading.Lock()
        self.summary_lock = threading.Lock()
        self.task_queue: queue.Queue[TaskSpec | None] = queue.Queue()
        self.summary = self._load_or_init_summary()

    def log(self, msg: str, level: str = "INFO") -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}"
        with self.print_lock:
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
            "eval_module": self.args.eval_module,
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
                "timestamp": task_result.get("updated_at"),
                "success_once_mean": metrics_mean.get("success_once"),
                "success_at_end_mean": metrics_mean.get("success_at_end"),
                "return_mean": metrics_mean.get("return"),
                "reward_mean": metrics_mean.get("reward"),
                "episode_len_mean": metrics_mean.get("length"),
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

    def _task_is_done(self, task_name: str) -> bool:
        task_dir = self.result_root / task_name
        task_entry = self.summary.get("tasks", {}).get(task_name)
        if not task_dir.exists() or task_entry is None:
            return False
        if task_entry.get("status") != "ok":
            return False
        if task_entry.get("num_eval_episodes") != self.args.num_eval_episodes:
            return False
        raw_shapes = task_entry.get("metrics_raw_shape") or {}
        if raw_shapes:
            completed_episode_counts = [
                shape[0]
                for shape in raw_shapes.values()
                if isinstance(shape, list) and shape and isinstance(shape[0], int)
            ]
            if completed_episode_counts and max(completed_episode_counts) < self.args.num_eval_episodes:
                return False
        eval_metrics_file = Path(task_entry.get("eval_metrics_file", task_dir / "eval_metrics.json"))
        if not eval_metrics_file.exists():
            return False
        if not task_entry.get("capture_video", True):
            return True
        video_files = task_entry.get("video_files") or []
        shared_video_files = task_entry.get("shared_video_files") or []
        if video_files or shared_video_files:
            return True
        return _count_task_videos(task_dir) > 0

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
            raise RuntimeError("No GPUs satisfy min_free_mem_gb; adjust args or use --no_cuda")
        return usable

    def _build_cmd(self, task_name: str) -> tuple[list[str], Path]:
        task_dir = self.result_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        task_video_dir = task_dir / "videos"
        task_video_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            self.args.eval_module,
            "--checkpoint_path",
            str(self.ckpt_path),
            "--output_dir",
            str(task_dir),
            "--env_id",
            task_name,
            "--control_mode",
            self.args.control_mode,
            "--num_eval_episodes",
            str(self.args.num_eval_episodes),
            "--num_eval_envs",
            str(self.args.num_eval_envs),
            "--video_dir",
            str(task_video_dir),
            "--use_lerobot",
            "--lerobot_use_paired_dataset",
            "--lerobot_eval_online",
            "--lerobot_human_root",
            self.args.lerobot_human_root,
            "--lerobot_sim_root",
            self.args.lerobot_sim_root,
            "--lerobot_human_dataset_file",
            self.args.lerobot_human_dataset_file,
            "--lerobot_sim_dataset_file",
            self.args.lerobot_sim_dataset_file,
            "--lerobot_task_mapping_file",
            self.args.lerobot_task_mapping_file,
            "--lerobot_human_task_description_file",
            self.args.lerobot_human_task_description_file,
            "--lerobot_sim_task_description_file",
            self.args.lerobot_sim_task_description_file,
            "--lerobot_state_type",
            self.args.lerobot_state_type,
            "--lerobot_video_backend",
            self.args.lerobot_video_backend,
            "--lerobot_image_size",
            str(self.args.lerobot_image_size[0]),
            str(self.args.lerobot_image_size[1]),
            "--vision_encoder",
            self.args.vision_encoder,
            "--text_encoder",
            self.args.text_encoder,
            "--t5_version",
            self.args.t5_version,
            "--max_lang_len",
            str(self.args.max_lang_len),
        ]
        if self.args.use_precomputed_lang and self.args.precomputed_lang_dir:
            cmd += ["--use_precomputed_lang", "--precomputed_lang_dir", self.args.precomputed_lang_dir]
        if self.args.use_dummy_language:
            cmd.append("--use_dummy_language")
        if self.args.sim_backend:
            cmd += ["--sim_backend", self.args.sim_backend]
        if self.args.shader:
            cmd += ["--shader", self.args.shader]
        if self.args.eval_module.endswith("eval_rdt_lora"):
            if self.args.pretrained_path:
                cmd += ["--pretrained_path", self.args.pretrained_path]
            cmd += [
                "--lora_r",
                str(self.args.lora_r),
                "--lora_alpha",
                str(self.args.lora_alpha),
                "--lora_dropout",
                str(self.args.lora_dropout),
            ]
            if self.args.precomputed_vl_dir:
                cmd += [
                    "--precomputed_vl_dir",
                    self.args.precomputed_vl_dir,
                    "--expected_precomputed_vl_mode",
                    self.args.expected_precomputed_vl_mode,
                ]
                if self.args.precomputed_vl_preload:
                    cmd.append("--precomputed_vl_preload")
            if self.args.allow_online_text_encoder:
                cmd.append("--allow_online_text_encoder")
        if self.args.compute_dtw:
            cmd += ["--compute_dtw", "--dtw_band_ratio", str(self.args.dtw_band_ratio)]
        if self.args.eval_extra_args:
            cmd.extend(_sanitize_eval_extra_args(self.args.eval_extra_args))
        return cmd, task_dir

    def _run_task(self, task: TaskSpec, gpu_id: int, worker_name: str) -> None:
        cmd, task_dir = self._build_cmd(task.task_name)
        env = os.environ.copy()
        if gpu_id >= 0:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_file = task_dir / "eval.log"
        self.log(f"{worker_name} evaluating {task.task_name} on {'CPU' if gpu_id < 0 else f'GPU {gpu_id}'}")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with log_file.open("w", encoding="utf-8") as f:
            assert proc.stdout is not None
            for line in proc.stdout:
                tagged = f"[{worker_name}|{task.task_name}] {line.rstrip()}"
                with self.print_lock:
                    print(tagged, flush=True)
                f.write(line)
        ret = proc.wait()

        video_files = _relocate_videos(task_dir)
        shared_video_files = _mirror_videos_to_shared_dir(task_dir, self.shared_videos_root / task.task_name)
        eval_result = _load_json_if_exists(task_dir / "eval_metrics.json")

        task_result = {
            "task_name": task.task_name,
            "env_id": task.task_name,
            "level": _extract_level(task.task_name),
            "return_code": ret,
            "status": "ok" if ret == 0 else "failed",
            "reward_mode": _extract_reward_mode(self.args.eval_extra_args),
            "num_eval_episodes": self.args.num_eval_episodes,
            "num_eval_envs": self.args.num_eval_envs,
            "capture_video": _capture_video_enabled(self.args.eval_extra_args),
            "metrics_mean": (eval_result or {}).get("metrics_mean", {}),
            "metrics_raw_shape": (eval_result or {}).get("metrics_raw_shape", {}),
            "checkpoint_step": (eval_result or {}).get("checkpoint_step"),
            "video_files": video_files,
            "shared_video_files": shared_video_files,
            "log_file": str(log_file),
            "eval_metrics_file": str(task_dir / "eval_metrics.json"),
            "updated_at": _now(),
        }
        if ret != 0:
            task_result["error"] = f"return_code={ret}"

        with self.summary_lock:
            self.summary.setdefault("tasks", {})[task.task_name] = task_result
            self._save_summary_locked()

        if ret == 0:
            self.log(
                f"{worker_name} finished {task.task_name} success_at_end={task_result['metrics_mean'].get('success_at_end')} videos={len(video_files)}"
            )
            return

        error = f"return_code={ret}"
        self.log(f"{worker_name} failed {task.task_name} ({error})", level="WARNING")
        if task.retry_count < self.args.max_retries:
            self.task_queue.put(TaskSpec(task.task_name, retry_count=task.retry_count + 1))

    def _worker_loop(self, gpu_id: int, slot_id: int) -> None:
        worker_name = f"W{slot_id}"
        while True:
            task = self.task_queue.get()
            if task is None:
                self.task_queue.task_done()
                return
            try:
                self._run_task(task, gpu_id, worker_name)
            finally:
                self.task_queue.task_done()

    def run(self) -> None:
        tasks = _load_tasks_from_sim_config(self.sim_cfg)
        if self.args.limit_tasks is not None:
            tasks = tasks[: self.args.limit_tasks]
        pending_tasks = [task for task in tasks if not self._task_is_done(task)]
        self.summary["tasks_total"] = len(tasks)
        with self.summary_lock:
            self._save_summary_locked()
        self.log(f"Loaded {len(tasks)} tasks; pending {len(pending_tasks)}")
        if not pending_tasks:
            self.log("All tasks already completed. Nothing to do.")
            return

        gpu_ids = self._resolve_gpu_ids()
        slots: list[int] = []
        for gpu_id in gpu_ids:
            for _ in range(self.args.max_procs_per_gpu):
                slots.append(gpu_id)
        if self.args.no_cuda:
            slots = [-1] * max(1, self.args.max_cpu_workers)
        slots = slots[: max(1, min(len(slots), len(pending_tasks)))]

        for task_name in pending_tasks:
            self.task_queue.put(TaskSpec(task_name))
        for _ in slots:
            self.task_queue.put(None)

        threads = []
        for idx, gpu_id in enumerate(slots):
            t = threading.Thread(target=self._worker_loop, args=(gpu_id, idx), daemon=True)
            t.start()
            threads.append(t)

        self.task_queue.join()
        for t in threads:
            t.join()
        self.log(f"Done. Summary: {self.summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel eval scheduler for RDT paired online eval")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--eval_module",
        default="examples.baselines.rdt.eval_rdt_scratch",
        choices=[
            "examples.baselines.rdt.eval_rdt_scratch",
            "examples.baselines.rdt.eval_rdt_lora",
        ],
    )
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--limit_tasks", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=1)

    parser.add_argument("--num_eval_episodes", type=int, default=5)
    parser.add_argument("--num_eval_envs", type=int, default=1)
    parser.add_argument("--sim_backend", default="physx_cpu")
    parser.add_argument("--control_mode", default="pd_joint_pos")
    parser.add_argument("--shader", default="rt-fast")

    parser.add_argument("--vision_encoder", required=True)
    parser.add_argument("--text_encoder", required=True)
    parser.add_argument("--t5_version", default="t5-v1_1-xxl")
    parser.add_argument("--max_lang_len", type=int, default=1024)
    parser.add_argument("--use_precomputed_lang", action="store_true")
    parser.add_argument("--precomputed_lang_dir", default=None)
    parser.add_argument("--precomputed_vl_dir", default=None)
    parser.add_argument("--precomputed_vl_preload", action="store_true")
    parser.add_argument("--expected_precomputed_vl_mode", default="language_only")
    parser.add_argument("--allow_online_text_encoder", action="store_true")
    parser.add_argument("--use_dummy_language", action="store_true")

    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--lerobot_human_root", required=True)
    parser.add_argument("--lerobot_sim_root", required=True)
    parser.add_argument("--lerobot_human_dataset_file", required=True)
    parser.add_argument("--lerobot_sim_dataset_file", required=True)
    parser.add_argument("--lerobot_task_mapping_file", required=True)
    parser.add_argument("--lerobot_human_task_description_file", required=True)
    parser.add_argument("--lerobot_sim_task_description_file", required=True)
    parser.add_argument("--lerobot_state_type", default="qpos")
    parser.add_argument("--lerobot_video_backend", default="torchcodec")
    parser.add_argument("--lerobot_image_size", nargs=2, type=int, default=[224, 224])

    parser.add_argument("--gpu_ids", nargs="*", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--max_procs_per_gpu", type=int, default=1)
    parser.add_argument("--min_free_mem_gb", type=float, default=8.0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--max_cpu_workers", type=int, default=1)
    parser.add_argument("--compute_dtw", action="store_true", help="Forward TSS/nDTW trajectory metric computation to RDT eval.")
    parser.add_argument("--dtw_band_ratio", type=float, default=0.15, help="Forwarded Sakoe-Chiba band ratio for TSS/nDTW.")

    parser.add_argument("eval_extra_args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scheduler = ParallelRDTEvalScheduler(args)
    scheduler.run()


if __name__ == "__main__":
    main()
