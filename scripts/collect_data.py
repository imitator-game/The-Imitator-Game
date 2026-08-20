    #!/usr/bin/env python3
"""
Smart simulation data generation scheduler v2.1

New GPU smart-scheduling features:
  - Real-time VRAM detection; a GPU is only allocated when its free memory reaches the threshold
  - Can specify which GPUs to use (--gpu-ids 0 1 3)
  - Per-GPU process count + memory threshold config (--gpu-config)
  - L3 uses its own env name ({task}L3-v1), no --l3 flag needed
  - Stats are persisted to generation_stats.json, enabling resume from breakpoints

Directory layout:
  {demos_dir}/
  ├── {task}-v1/motionplanning/
  │   ├── L0_{task}-v1.h5/.json
  │   ├── L1_{task}-v1.h5/.json
  │   └── L2_{task}-v1.h5/.json
  ├── {task}L3-v1/motionplanning/
  │   └── L3_{task}-v1.h5/.json
  └── generation_stats.json

Usage examples:
  # Simplest: all GPUs, default config
  python3 collect_data_v2.py --demos-dir demos

  # Only use GPU 0 and 2
  python3 collect_data_v2.py --gpu-ids 0 2

  # Fine-grained config: GPU0 runs 3 procs needs 6GB free, GPU1 runs 1 proc needs 10GB free, GPU2 unused
  python3 collect_data_v2.py \\
    --gpu-config "0:max_procs=3,mem_threshold=6144" \\
                 "1:max_procs=1,mem_threshold=10240"

  # Only collect a subset of tasks at L3
  python3 collect_data_v2.py --levels L3 --tasks TwoRobotStirSpoon TwoRobotPourKettle

  # View progress (without running)
  python3 collect_data_v2.py --dry-run
"""

import json
import os
import re
import subprocess
import sys
import signal
import time
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Tuple
import argparse

import psutil

# Import the GPU management module (same directory)
from gpu_manager import GPUManager, add_gpu_args, build_gpu_manager


# ═══════════════════════════════════════════════════════
#  Task list & level configs
# ═══════════════════════════════════════════════════════

DEFAULT_TASKS = [
    "TwoRobotStirSpoon",
    "TwoRobotPlaceBookBookcase",
    "TwoRobotPlaceClothBasket",
    "TwoRobotPickRemoteControl",
    "TwoRobotPlaceMagazineFolder",
    "TwoRobotPickWash",
    "TwoRobotPickAppleBasket",
    "TwoRobotPickAppleBananaToBaskets",
    "TwoRobotPickAppleToScale",
    "TwoRobotPlaceChipsRack",
    "TwoRobotPlaceCommodityRack",
    "TwoRobotPlaceFruitBox",
    "TwoRobotScanMilkBox",
    "TwoRobotPlacePlateRack",
    "TwoRobotCutFruit",
    "TwoRobotPickFruitsToPlate",
    "TwoRobotPourKetchupFries",
    "TwoRobotWipePot",
    "TwoRobotCleanDesk",
    "TwoRobotPourKettle",
    "TwoRobotPickFood",
    "TwoRobotTransFood",
    "TwoRobotPlaceFoodScale",
    "TwoRobotPressStapler",
    "TwoRobotPlaceBurgerTray",
    "TwoRobotPickTennisBallGolfBall",
    "TwoRobotScanPillBottle",
    "TwoRobotPlaceShoeBox",
    "TwoRobotOpenBox",
    "TwoRobotFoldBox",
    "TwoRobotPickPillToRegions",
    "TwoRobotPlaceFileFolder",
    "TwoRobotPourLiquidCup",
    "TwoRobotPlacePillBox",
    "TwoRobotPlaceBrushRest",
    "TwoRobotPlaceCupPlate",
    "TwoRobotPlaceScrewdriver",
    "TwoRobotPlaceMugRack",
    "TwoRobotPourLiquidMug",
    "TwoRobotPutCubeOnScale",
    "TwoRobotPourCup",
    "TwoRobotCleanCup",
    "TwoRobotOpenLiquidCap",
    "TwoRobotGrindFood",
    "TwoRobotPourLiquidFilter",
    "TwoRobotPutBox",
    "TwoRobotKnifeBowlFork",
    "TwoRobotLiftLidFromSkillet",
    "TwoRobotFoldTowel",
    "TwoRobotPressJuicer",
]

# name, env suffix (L3 special), level flag (None for L3)
ALL_LEVEL_CONFIGS = [
    {"name": "L0", "env_suffix": "",   "flag": "--l0"},
    {"name": "L1", "env_suffix": "",   "flag": "--l1"},
    {"name": "L2", "env_suffix": "",   "flag": "--l2"},
    {"name": "L3", "env_suffix": "L3", "flag": None},
]


# ═══════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════

@dataclass
class RunStats:
    left_success_rate: float = 0.0
    right_success_rate: float = 0.0
    failed_motion_plan_rate: float = 0.0
    failed_motion_plans_count: int = 0
    left_avg_episode_length: float = 0.0
    right_avg_episode_length: float = 0.0
    left_max_episode_length: float = 0.0
    right_max_episode_length: float = 0.0
    total_seeds_tried: int = 0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TaskInfo:
    task_base: str
    level: str
    env_id: str
    level_flag: Optional[str]
    traj_name: str
    h5_path: str
    json_path: str
    target_episodes: int = 50
    current_episodes: int = 0
    status: str = "pending"
    retries: int = 0

    # Runtime (not persisted)
    gpu_id: int = -1
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    start_time: float = 0.0
    last_output_time: float = 0.0
    output_buffer: List[str] = field(default_factory=list, repr=False)
    error_message: str = ""

    @property
    def key(self):
        return (self.task_base, self.level)

    @property
    def display_name(self):
        return f"{self.level}_{self.task_base}-v1"

    @property
    def remaining_episodes(self):
        return max(0, self.target_episodes - self.current_episodes)

    @property
    def is_done(self):
        return self.current_episodes >= self.target_episodes


# ═══════════════════════════════════════════════════════
#  Stats file management
# ═══════════════════════════════════════════════════════

class StatsManager:
    def __init__(self, stats_path: str):
        self.path = stats_path
        self._lock = Lock()
        self._data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_updated": "", "summary": {}, "tasks": {}}

    def _save(self):
        self._data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def update_task_level(self, task: TaskInfo, run_stats: Optional[RunStats],
                          duration: float, episodes_this_run: int):
        with self._lock:
            td = self._data.setdefault("tasks", {})
            ld = td.setdefault(task.task_base, {}).setdefault(task.level, {})
            ld.update({
                "status": task.status,
                "episodes_collected": task.current_episodes,
                "target_episodes": task.target_episodes,
                "retries": task.retries,
                "last_run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration_seconds": round(
                    ld.get("total_duration_seconds", 0) + duration, 1),
            })
            rec = {
                "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(duration, 1),
                "episodes_in_this_run": episodes_this_run,
            }
            if run_stats:
                rec.update(run_stats.to_dict())
            ld.setdefault("run_history", []).append(rec)
            self._save()

    def update_summary(self, tasks: dict):
        with self._lock:
            statuses = [t.status for t in tasks.values()]
            self._data["summary"] = {
                "total": len(statuses),
                "completed": statuses.count("completed"),
                "failed": statuses.count("failed"),
                "running": statuses.count("running"),
                "pending": statuses.count("pending"),
            }
            self._save()


# ═══════════════════════════════════════════════════════
#  Output parsing
# ═══════════════════════════════════════════════════════

class OutputParser:
    _PATTERNS = {
        "left_success_rate":        r"left_success_rate=([0-9.]+)",
        "right_success_rate":       r"right_success_rate=([0-9.]+)",
        "failed_motion_plan_rate":  r"failed_motion_plan_rate=([0-9.]+)",
        "left_avg_episode_length":  r"left_avg_episode_length=([0-9.]+)",
        "right_avg_episode_length": r"right_avg_episode_length=([0-9.]+)",
        "left_max_episode_length":  r"left_max_episode_length=([0-9.]+)",
        "right_max_episode_length": r"right_max_episode_length=([0-9.]+)",
    }

    @classmethod
    def parse_stats(cls, lines: List[str], target_n: int) -> RunStats:
        stats = RunStats()
        for line in reversed(lines):
            if "left_success_rate" not in line:
                continue
            for fname, pat in cls._PATTERNS.items():
                m = re.search(pat, line)
                if m:
                    setattr(stats, fname, float(m.group(1)))
            m = re.search(r"(\d+)/\d+\s+\[", line)
            if m:
                stats.total_seeds_tried = int(m.group(1))
            break
        if stats.total_seeds_tried > 0 and stats.failed_motion_plan_rate > 0:
            stats.failed_motion_plans_count = round(
                stats.failed_motion_plan_rate * stats.total_seeds_tried)
        return stats

    @classmethod
    def extract_errors(cls, lines: List[str]) -> List[str]:
        kws = ["Error", "Exception", "Traceback", "FAILED",
               "RuntimeError", "OOM", "out of memory", "Segmentation fault"]
        result, in_tb = [], False
        for line in lines:
            if "Traceback" in line:
                in_tb = True
            if in_tb:
                result.append(line)
                if line.strip() and not line.startswith(" ") and ":" in line \
                        and "Traceback" not in line:
                    in_tb = False
            elif any(k in line for k in kws):
                result.append(line)
        return result or (["[last 20 lines:]"] + lines[-20:])


# ═══════════════════════════════════════════════════════
#  Main scheduler
# ═══════════════════════════════════════════════════════

class SmartScheduler:

    def __init__(self, args, gpu_mgr: GPUManager):
        self.demos_dir = args.demos_dir
        self.target_episodes = args.target_episodes
        self.task_timeout = args.task_timeout
        self.stall_timeout = args.stall_timeout
        self.max_retries = args.max_retries
        self.task_filter = set(args.tasks) if args.tasks else None
        self.level_filter = set(args.levels)

        self.gpu_mgr = gpu_mgr
        self.stats_mgr = StatsManager(
            os.path.join(args.demos_dir, "generation_stats.json"))

        self.tasks: Dict[tuple, TaskInfo] = {}
        self.running: List[TaskInfo] = []
        self._lock = Lock()
        self._stop = False

        log_dir = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        self._log_fh = open(os.path.join(log_dir, "scheduler.log"), "w")
        self._err_dir = os.path.join(log_dir, "errors")
        os.makedirs(self._err_dir, exist_ok=True)
        self.log_dir = log_dir

    # ── Logging ──────────────────────────────────────────

    def log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        print(line)
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    # ── Paths ────────────────────────────────────────────

    def _paths(self, task_base: str, lvl_cfg: dict):
        suffix = lvl_cfg["env_suffix"]
        level = lvl_cfg["name"]
        env_id = f"{task_base}{suffix}-v1"
        traj_name = f"{level}_{task_base}-v1"
        mp_dir = os.path.join(self.demos_dir, env_id, "motionplanning")
        return (env_id, traj_name,
                os.path.join(mp_dir, traj_name + ".h5"),
                os.path.join(mp_dir, traj_name + ".json"))

    def _count_episodes(self, json_path: str) -> int:
        try:
            if not os.path.exists(json_path):
                return 0
            with open(json_path) as f:
                data = json.load(f)
            return sum(1 for ep in data.get("episodes", [])
                       if ep.get("success", False))
        except Exception:
            return 0

    # ── Initialization ───────────────────────────────────

    def init_tasks(self):
        self.log("Scanning existing data...")
        task_list = DEFAULT_TASKS
        if self.task_filter:
            task_list = [t for t in task_list if t in self.task_filter]
        level_cfgs = [lc for lc in ALL_LEVEL_CONFIGS
                      if lc["name"] in self.level_filter]

        for task_base in task_list:
            for lvl_cfg in level_cfgs:
                env_id, traj_name, h5_path, json_path = self._paths(task_base, lvl_cfg)
                current = self._count_episodes(json_path)
                task = TaskInfo(
                    task_base=task_base, level=lvl_cfg["name"],
                    env_id=env_id, level_flag=lvl_cfg["flag"],
                    traj_name=traj_name, h5_path=h5_path, json_path=json_path,
                    target_episodes=self.target_episodes,
                    current_episodes=current,
                )
                if task.is_done:
                    task.status = "completed"
                self.tasks[task.key] = task

        total = len(self.tasks)
        done = sum(1 for t in self.tasks.values() if t.status == "completed")
        self.log(f"Init: {total} tasks total, {done} done, {total - done} pending")

    # ── Starting tasks ───────────────────────────────────

    def _build_cmd(self, task: TaskInfo) -> List[str]:
        cmd = [
            "python3", "-m",
            "mani_skill.examples.motionplanning.dual.two_robot_run",
            "--shader", "rt-fast",
            "-e", task.env_id,
            "-n", str(task.remaining_episodes),
            "--only-count-success",
            "--traj-name", task.traj_name,
            "--record-dir", self.demos_dir,
            "--num-procs", "1",
        ]
        if task.level_flag:
            cmd.append(task.level_flag)
        return cmd

    def _start_task(self, task: TaskInfo) -> bool:
        gpu_id = self.gpu_mgr.acquire()
        if gpu_id is None:
            return False

        # Re-check (may have just been completed by another thread)
        task.current_episodes = self._count_episodes(task.json_path)
        if task.is_done:
            task.status = "completed"
            self.gpu_mgr.release(gpu_id)
            return True

        task.gpu_id = gpu_id
        task.status = "running"
        task.start_time = time.time()
        task.last_output_time = time.time()
        task.output_buffer = []
        task.error_message = ""

        cmd = self._build_cmd(task)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.log(
            f">> [{task.display_name}] -> GPU{gpu_id} | "
            f"needs {task.remaining_episodes} eps "
            f"(already {task.current_episodes}/{task.target_episodes})"
        )

        try:
            task.process = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, universal_newlines=True,
            )
            with self._lock:
                self.running.append(task)
            threading.Thread(
                target=self._read_output, args=(task,), daemon=True).start()
            return True
        except Exception as e:
            self.log(f"Failed to start [{task.display_name}]: {e}", "ERROR")
            task.status = "pending"
            task.retries += 1
            self.gpu_mgr.release(gpu_id)
            return False

    def _read_output(self, task: TaskInfo):
        try:
            while task.process and task.process.poll() is None:
                if task.process.stdout:
                    line = task.process.stdout.readline()
                    if line:
                        task.output_buffer.append(line.rstrip())
                        task.last_output_time = time.time()
                        if len(task.output_buffer) > 1000:
                            task.output_buffer = task.output_buffer[-800:]
            if task.process and task.process.stdout:
                rest = task.process.stdout.read()
                if rest:
                    task.output_buffer.extend(rest.strip().split("\n"))
        except Exception as e:
            task.output_buffer.append(f"[read error: {e}]")

    # ── Health checks ────────────────────────────────────

    def _health(self, task: TaskInfo) -> str:
        if not task.process:
            return "failed"
        ret = task.process.poll()
        if ret is not None:
            return "completed" if ret == 0 else "failed"
        if time.time() - task.start_time > self.task_timeout:
            return "timeout"
        try:
            proc = psutil.Process(task.process.pid)
            if proc.cpu_percent(interval=0.1) > 1.0:
                task.last_output_time = time.time()
            elif time.time() - task.last_output_time > self.stall_timeout:
                return "stalled"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return "running"

    def _kill(self, task: TaskInfo, reason: str):
        self.log(f"! Terminating [{task.display_name}] reason: {reason}", "WARNING")
        if task.process:
            try:
                task.process.terminate()
                try:
                    task.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    task.process.kill()
                    task.process.wait()
                time.sleep(0.5)
            except Exception as e:
                self.log(f"  Terminate failed: {e}", "ERROR")

    # ── Completion handling ──────────────────────────────

    def _on_finish(self, task: TaskInfo, status: str):
        with self._lock:
            if task in self.running:
                self.running.remove(task)
        self.gpu_mgr.release(task.gpu_id)

        duration = time.time() - task.start_time
        prev = task.current_episodes
        task.current_episodes = self._count_episodes(task.json_path)
        eps_this_run = task.current_episodes - prev
        run_stats = OutputParser.parse_stats(task.output_buffer, task.remaining_episodes)

        if status == "completed" or task.is_done:
            task.status = "completed"
            self.log(
                f"OK [{task.display_name}] {task.current_episodes}/{task.target_episodes} eps "
                f"took {duration/60:.1f}min | "
                f"left:{run_stats.left_success_rate:.1%} right:{run_stats.right_success_rate:.1%} "
                f"plan fails:{run_stats.failed_motion_plans_count} "
                f"({run_stats.failed_motion_plan_rate:.1%})"
            )
        else:
            task.retries += 1
            errors = OutputParser.extract_errors(task.output_buffer)
            task.error_message = "\n".join(errors)
            self._save_error_log(task, status, errors)
            if task.retries >= self.max_retries:
                task.status = "failed"
                self.log(
                    f"XX completely failed [{task.display_name}] "
                    f"after {task.retries} retries progress {task.current_episodes}/{task.target_episodes}",
                    "ERROR"
                )
                for line in errors[:6]:
                    self.log(f"    {line}", "ERROR")
            else:
                task.status = "pending"
                self.log(
                    f"!! retry [{task.display_name}] "
                    f"({task.retries}/{self.max_retries}) "
                    f"progress {task.current_episodes}/{task.target_episodes}",
                    "WARNING"
                )

        self.stats_mgr.update_task_level(task, run_stats, duration, eps_this_run)
        self.stats_mgr.update_summary(self.tasks)

    def _save_error_log(self, task: TaskInfo, status: str, errors: List[str]):
        path = os.path.join(self._err_dir,
                            f"{task.display_name}_retry{task.retries}.log")
        try:
            with open(path, "w") as f:
                f.write(f"{'='*60}\n")
                f.write(f"Task:    {task.display_name}\n")
                f.write(f"Status:  {status}\n")
                f.write(f"GPU:     {task.gpu_id}\n")
                f.write(f"Retry:   {task.retries}\n")
                f.write(f"Elapsed: {time.time()-task.start_time:.1f}s\n")
                f.write(f"Progress:{task.current_episodes}/{task.target_episodes}\n")
                f.write("="*60 + "\n\n=== Error Extract ===\n")
                for line in errors:
                    f.write(line + "\n")
                f.write("\n=== Full Output ===\n")
                for line in task.output_buffer:
                    f.write(line + "\n")
            self.log(f"  Error log: {path}", "WARNING")
        except Exception as e:
            self.log(f"  Failed to save error log: {e}", "ERROR")

    # ── Monitor thread ───────────────────────────────────

    def _monitor(self):
        while not self._stop:
            to_handle = []
            with self._lock:
                for task in list(self.running):
                    h = self._health(task)
                    if h != "running":
                        to_handle.append((task, h))
                        if h in ("stalled", "timeout"):
                            self._kill(task, h)
            for task, status in to_handle:
                self._on_finish(task, status)
            time.sleep(30)

    # ── Main loop ────────────────────────────────────────

    def run(self):
        self.init_tasks()

        def _sig(sig, frame):
            self.log("Interrupt received, cleaning up...", "WARNING")
            self._stop = True
            for t in list(self.running):
                self._kill(t, "user interrupt")
            self._log_fh.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, _sig)
        signal.signal(signal.SIGTERM, _sig)
        threading.Thread(target=self._monitor, daemon=True).start()

        last_status_ts = 0
        self.log("Starting scheduling...")

        while not self._stop:
            pending = [t for t in self.tasks.values()
                       if t.status == "pending" and t.retries < self.max_retries]

            if not pending and not self.running:
                self.log("All tasks completed!")
                break

            pending.sort(key=lambda t: t.remaining_episodes)

            for task in pending:
                if self._stop:
                    break
                if len(self.running) >= self.gpu_mgr.total_capacity:
                    break
                if task.status != "pending":
                    continue
                self._start_task(task)
                time.sleep(1)

            if time.time() - last_status_ts > 60:
                self._print_status()
                last_status_ts = time.time()

            time.sleep(5)

        self.log("Waiting for running tasks to finish...")
        while self.running and not self._stop:
            time.sleep(5)

        self._print_final_report()
        self.stats_mgr.update_summary(self.tasks)
        self._log_fh.close()

    # ── Status printing ──────────────────────────────────

    def _print_status(self):
        s = {st: sum(1 for t in self.tasks.values() if t.status == st)
             for st in ("pending", "running", "completed", "failed")}
        self.log(
            f"[Status] pending:{s['pending']} running:{s['running']} "
            f"completed:{s['completed']} failed:{s['failed']}"
        )
        self.log(f"[GPU]  {self.gpu_mgr.status_str()}")
        for t in self.running:
            elapsed = (time.time() - t.start_time) / 60
            self.log(f"  >> [{t.display_name}] GPU{t.gpu_id} {elapsed:.1f}min")

    def _print_final_report(self):
        done = [t for t in self.tasks.values() if t.status == "completed"]
        fail = [t for t in self.tasks.values() if t.status == "failed"]
        left = [t for t in self.tasks.values()
                if t.status not in ("completed", "failed")]

        self.log("\n" + "="*60)
        self.log(f"Completed:{len(done)}  Failed:{len(fail)}  Incomplete:{len(left)}")
        self.log(f"Stats file: {self.stats_mgr.path}")
        self.log(f"Log dir: {self.log_dir}")

        if fail:
            self.log("\nFailed tasks:")
            for t in fail:
                self.log(f"  X [{t.display_name}] {t.current_episodes}/{t.target_episodes}")
                for line in t.error_message.split("\n")[:4]:
                    if line.strip():
                        self.log(f"      {line}")

        if left:
            self.log("\nIncomplete (can resume from breakpoint):")
            for t in left:
                self.log(f"  o [{t.display_name}] {t.current_episodes}/{t.target_episodes}")


# ═══════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Smart simulation data generation scheduler v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data directory
    parser.add_argument("--demos-dir", type=str, default="demos",
                        help="Root directory for data")

    # GPU args (injected by the gpu_manager module)
    add_gpu_args(parser)

    # Task args
    parser.add_argument("--target-episodes", type=int, default=50)
    parser.add_argument("--task-timeout", type=int, default=7200)
    parser.add_argument("--stall-timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Only process the given tasks (base names)")
    parser.add_argument("--levels", nargs="+", default=["L0","L1","L2","L3"],
                        choices=["L0","L1","L2","L3"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print status, do not run")

    args = parser.parse_args()
    os.makedirs(args.demos_dir, exist_ok=True)

    # Build the GPU Manager (prints the config table)
    gpu_mgr = build_gpu_manager(args)

    scheduler = SmartScheduler(args, gpu_mgr)
    scheduler.init_tasks()

    if args.dry_run:
        print(f"\n{'task':45s} {'level':4s} {'progress':12s} {'status'}")
        print("-" * 72)
        for task in sorted(scheduler.tasks.values(),
                           key=lambda t: (t.task_base, t.level)):
            icon = {"completed":"✓","pending":"○","failed":"✗","running":"▶"
                    }.get(task.status, "?")
            print(f"  {icon} {task.task_base:43s} {task.level:4s} "
                  f"{task.current_episodes:3d}/{task.target_episodes:<6d} {task.status}")
        return

    scheduler.run()


if __name__ == "__main__":
    main()