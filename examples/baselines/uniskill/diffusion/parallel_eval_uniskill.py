#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# GPU helpers
# =============================================================================

def get_gpu_free_mem_gb(gpu_id: int) -> float:
    """Return free memory in GB for one GPU using nvidia-smi."""
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
        return float(out.split("\n")[0]) / 1024.0
    except Exception:
        return 0.0


def pick_gpus(
    num_gpus: int,
    min_free_mem_gb: float,
    explicit_ids: Optional[List[int]] = None,
) -> List[int]:
    candidate_ids = explicit_ids if explicit_ids is not None else list(range(num_gpus))
    picked: List[int] = []

    for gpu_id in candidate_ids:
        free = get_gpu_free_mem_gb(gpu_id)
        if free >= min_free_mem_gb:
            picked.append(gpu_id)
        else:
            print(
                f"  GPU{gpu_id}: only {free:.1f} GB free "
                f"(need >= {min_free_mem_gb:.1f} GB), skip"
            )

    return picked


# =============================================================================
# Skip logic
# =============================================================================

def _load_existing_results_json(output_dir: Path, input_mode: str) -> Dict[str, Dict]:
    existing: Dict[str, Dict] = {}

    for json_file in output_dir.glob(f"*_{input_mode}_*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                results = json.load(f)

            for r in results:
                env_id = r.get("env_id")
                if env_id and r.get("status") == "success":
                    prev = existing.get(env_id)
                    if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
                        existing[env_id] = r

        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            continue

    return existing


def is_env_done(env_id: str, output_dir: Path, num_episodes: int, input_mode: str) -> bool:
    existing = _load_existing_results_json(output_dir, input_mode)
    prev = existing.get(env_id)

    if prev is not None and prev.get("num_episodes", 0) >= num_episodes:
        return True

    # Backward-compatible fallback: if enough videos exist, treat it as done.
    video_dir = output_dir / "videos" / env_id
    if video_dir.exists():
        mp4s = list(video_dir.glob("*.mp4"))
        if len(mp4s) >= num_episodes:
            return True

    return False


# =============================================================================
# Worker state
# =============================================================================

@dataclass
class WorkerTask:
    worker_id: int
    gpu_id: int
    env_ids: List[str]
    config_file: str = ""
    process: Optional[subprocess.Popen] = None
    status: str = "pending"  # pending | running | done | failed
    retries: int = 0
    start_time: float = 0.0
    output_lines: List[str] = field(default_factory=list)


# =============================================================================
# Scheduler
# =============================================================================

class ParallelEvalUniSkillScheduler:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = self.output_dir / f"scheduler_logs_uniskill_{ts}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._log_fh = open(self.log_dir / "scheduler.log", "w", encoding="utf-8")
        self.lock = threading.Lock()
        self.stop_flag = False
        self.workers: List[WorkerTask] = []

    def log(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        print(line)
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    def load_env_ids(self) -> List[str]:
        with open(self.args.eval_config, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]

    def build_workers(self, pending_envs: List[str], gpu_ids: List[int]) -> List[WorkerTask]:
        max_workers = len(gpu_ids) * self.args.max_procs_per_gpu
        num_workers = min(max_workers, len(pending_envs))

        if num_workers == 0:
            return []

        chunks: Dict[int, List[str]] = defaultdict(list)
        for i, env_id in enumerate(pending_envs):
            chunks[i % num_workers].append(env_id)

        workers: List[WorkerTask] = []
        for worker_idx in range(num_workers):
            env_list = chunks[worker_idx]
            if not env_list:
                continue

            gpu_id = gpu_ids[worker_idx % len(gpu_ids)]

            tmp = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                prefix=f"eval_uniskill_worker{worker_idx}_",
                dir=self.log_dir,
                delete=False,
                encoding="utf-8",
            )
            tmp.write("\n".join(env_list) + "\n")
            tmp.close()

            workers.append(
                WorkerTask(
                    worker_id=worker_idx,
                    gpu_id=gpu_id,
                    env_ids=env_list,
                    config_file=tmp.name,
                )
            )

        return workers

    def _eval_entry_cmd_prefix(self) -> List[str]:
        if self.args.eval_script:
            return [sys.executable, self.args.eval_script]
        return [sys.executable, "-m", self.args.eval_module]

    def build_cmd(self, worker: WorkerTask) -> List[str]:
        a = self.args

        cmd = self._eval_entry_cmd_prefix() + [
            "--eval-config", worker.config_file,
            "--checkpoint", a.checkpoint,
            "--idm-ckpt-path", a.idm_ckpt_path,
            "--output-dir", a.output_dir,

            "--input-mode", a.input_mode,
            "--human-root", a.human_root,
            "--sim-root", a.sim_root,
            "--sim-config", a.sim_config,
            "--human-config", a.human_config,
            "--task-mapping", a.task_mapping,
            "--human-task-desc", a.human_task_desc,
            "--sim-task-desc", a.sim_task_desc,

            "--num-episodes", str(a.num_episodes),
            "--num-envs", str(a.num_envs),
            "--max-episode-steps", str(a.max_episode_steps),

            "--sim-backend", a.sim_backend,
            "--control-mode", a.control_mode,
            "--obs-mode", a.obs_mode,

            "--action-dim", str(a.action_dim),
            "--obs-dim", str(a.obs_dim),
            "--obs-horizon", str(a.obs_horizon),
            "--policy-pred-horizon", str(a.policy_pred_horizon),
            "--vision-feature-dim", str(a.vision_feature_dim),
            "--idm-feature-dim", str(a.idm_feature_dim),
            "--num-diffusion-iters", str(a.num_diffusion_iters),
            "--resolution", str(a.resolution),
            "--idm-resolution", str(a.idm_resolution),

            "--image-size", str(a.image_size[0]), str(a.image_size[1]),
            "--num-video-frames", str(a.num_video_frames),
            "--state-type", a.state_type,

            "--device", "cuda",
            "--shader", a.shader,

            "--vocab-size", str(a.vocab_size),
            "--max-text-len", str(a.max_text_len),
            "--task-seq-len", str(a.task_seq_len),
            "--mano-dim", str(a.mano_dim),

            "--eval_lr_mirror", a.eval_lr_mirror,
            "--eval_lr_mirror_robot_pose", a.eval_lr_mirror_robot_pose,

            "--dtw-band-ratio", str(a.dtw_band_ratio),
        ]

        if a.policy_obs_horizon is not None:
            cmd += ["--policy-obs-horizon", str(a.policy_obs_horizon)]

        if a.pred_horizon is not None:
            cmd += ["--pred-horizon", str(a.pred_horizon)]

        if a.cameras:
            cmd += ["--cameras"] + list(a.cameras)

        if a.include_depth:
            cmd.append("--include-depth")

        if a.single_arm:
            cmd.append("--single-arm")

        if a.temporal_agg:
            cmd.append("--temporal-agg")

        cmd.append("--compute-dtw" if a.compute_dtw else "--no-compute-dtw")

        return cmd

    def _stream_output(self, worker: WorkerTask):
        prefix = f"[W{worker.worker_id}|GPU{worker.gpu_id}]"
        worker_log = self.log_dir / f"worker_{worker.worker_id}.log"

        with open(worker_log, "w", encoding="utf-8") as fh:
            assert worker.process is not None
            assert worker.process.stdout is not None

            for line in worker.process.stdout:
                line = line.rstrip()
                worker.output_lines.append(line)
                if len(worker.output_lines) > 200:
                    worker.output_lines = worker.output_lines[-200:]

                tagged = f"{prefix} {line}"
                print(tagged)
                fh.write(tagged + "\n")
                fh.flush()

    def launch_worker(self, worker: WorkerTask):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker.gpu_id)
        env.setdefault("PYTHONUNBUFFERED", "1")

        cmd = self.build_cmd(worker)

        self.log(
            f"Launching Worker {worker.worker_id} on GPU{worker.gpu_id} | "
            f"{len(worker.env_ids)} envs | num_envs={self.args.num_envs} | "
            f"DTW={self.args.compute_dtw}"
        )
        self.log("Command: " + " ".join(cmd))

        worker.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        worker.status = "running"
        worker.start_time = time.time()

        t = threading.Thread(target=self._stream_output, args=(worker,), daemon=True)
        t.start()

    def _monitor(self):
        while not self.stop_flag:
            time.sleep(10)

            with self.lock:
                for worker in self.workers:
                    if worker.status != "running":
                        continue

                    assert worker.process is not None
                    ret = worker.process.poll()

                    if ret is None:
                        continue

                    elapsed = (time.time() - worker.start_time) / 60.0

                    if ret == 0:
                        worker.status = "done"
                        self.log(f"Worker {worker.worker_id} finished ({elapsed:.1f} min)")
                    else:
                        worker.retries += 1
                        if worker.retries <= self.args.max_retries:
                            worker.status = "pending"
                            self.log(
                                f"Worker {worker.worker_id} failed (ret={ret}), "
                                f"retry {worker.retries}/{self.args.max_retries}",
                                "WARNING",
                            )
                        else:
                            worker.status = "failed"
                            self.log(
                                f"Worker {worker.worker_id} failed after {worker.retries} retries",
                                "ERROR",
                            )

    def _status(self):
        counts = defaultdict(int)
        for worker in self.workers:
            counts[worker.status] += 1

        self.log(
            f"Workers -- pending:{counts['pending']} running:{counts['running']} "
            f"done:{counts['done']} failed:{counts['failed']}"
        )

        for gpu_id in sorted(set(worker.gpu_id for worker in self.workers)):
            self.log(f"   GPU{gpu_id}: {get_gpu_free_mem_gb(gpu_id):.1f} GB free")

    def run(self):
        self.log("=" * 80)
        self.log("PARALLEL EVAL SCHEDULER -- UniSkill")
        self.log("=" * 80)
        self.log(f"Input mode: {self.args.input_mode}")
        self.log(f"Eval entry: {self.args.eval_script or ('-m ' + self.args.eval_module)}")
        self.log(f"Checkpoint: {self.args.checkpoint}")
        self.log(f"IDM checkpoint: {self.args.idm_ckpt_path}")

        all_envs = self.load_env_ids()
        self.log(f"Total envs in config: {len(all_envs)}")

        pending, skipped = [], 0
        for env_id in all_envs:
            if is_env_done(env_id, self.output_dir, self.args.num_episodes, self.args.input_mode):
                skipped += 1
            else:
                pending.append(env_id)

        self.log(f"Already done (JSON or video): {skipped} | Pending: {len(pending)}")

        if not pending:
            self.log("All environments already evaluated. Nothing to do.")
            return

        self.log(f"Checking GPU memory (need >= {self.args.min_free_mem_gb:.1f} GB each)...")
        gpu_ids = pick_gpus(
            num_gpus=self.args.num_gpus,
            min_free_mem_gb=self.args.min_free_mem_gb,
            explicit_ids=self.args.gpu_ids,
        )

        if not gpu_ids:
            self.log("No GPUs with sufficient free memory.", "ERROR")
            sys.exit(1)

        self.log(f"Using GPUs: {gpu_ids}")

        self.workers = self.build_workers(pending, gpu_ids)
        self.log(f"Launching {len(self.workers)} workers ({len(pending)} envs total)")

        def _sighandler(sig, frame):
            self.log("Interrupted -- terminating workers...", "WARNING")
            self.stop_flag = True
            for worker in self.workers:
                if worker.process and worker.process.poll() is None:
                    worker.process.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, _sighandler)
        signal.signal(signal.SIGTERM, _sighandler)

        monitor_t = threading.Thread(target=self._monitor, daemon=True)
        monitor_t.start()

        for worker in self.workers:
            self.launch_worker(worker)
            time.sleep(self.args.launch_interval_sec)

        last_status = 0.0

        while not self.stop_flag:
            all_done = all(worker.status in ("done", "failed") for worker in self.workers)

            for worker in self.workers:
                if worker.status == "pending" and worker.retries > 0:
                    free = get_gpu_free_mem_gb(worker.gpu_id)
                    if free >= self.args.min_free_mem_gb:
                        self.log(f"Retrying Worker {worker.worker_id} on GPU{worker.gpu_id}")
                        self.launch_worker(worker)
                    else:
                        self.log(
                            f"GPU{worker.gpu_id} only {free:.1f} GB free; wait before retry.",
                            "WARNING",
                        )

            if time.time() - last_status > self.args.status_interval_sec:
                self._status()
                last_status = time.time()

            if all_done:
                break

            time.sleep(5)

        self.log("\n" + "=" * 80)
        self.log("FINAL SUMMARY")
        self.log("=" * 80)
        self._status()

        done_count = sum(1 for worker in self.workers if worker.status == "done")
        failed_count = sum(1 for worker in self.workers if worker.status == "failed")
        self.log(f"Workers done: {done_count} | Workers failed: {failed_count}")

        if failed_count > 0:
            self.log("Failed workers:", "ERROR")
            for worker in self.workers:
                if worker.status == "failed":
                    self.log(f"  Worker {worker.worker_id} GPU{worker.gpu_id}: {worker.env_ids}", "ERROR")
                    if worker.output_lines:
                        last_lines = "\n".join(f"    {line}" for line in worker.output_lines[-10:])
                        self.log("  Last output:\n" + last_lines, "ERROR")

        self.log(f"Scheduler finished. Logs: {self.log_dir}")
        self._log_fh.close()


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel evaluation scheduler for UniSkill",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    scheduler = parser.add_argument_group("Scheduler")
    scheduler.add_argument("--num-gpus", type=int, default=4)
    scheduler.add_argument("--gpu-ids", type=int, nargs="+", default=None)
    scheduler.add_argument("--max-procs-per-gpu", type=int, default=1)
    scheduler.add_argument("--min-free-mem-gb", type=float, default=16.0)
    scheduler.add_argument("--max-retries", type=int, default=1)
    scheduler.add_argument("--status-interval-sec", type=float, default=60.0)
    scheduler.add_argument("--launch-interval-sec", type=float, default=2.0)

    entry = parser.add_argument_group("Eval entry")
    entry.add_argument(
        "--eval-module",
        type=str,
        default="examples.baselines.uniskill.diffusion.eval_uniskill",
        help="Python module for eval, used as: python -m MODULE",
    )
    entry.add_argument(
        "--eval-script",
        type=str,
        default=None,
        help="Optional direct file path for eval script. If set, overrides --eval-module.",
    )

    eval_args = parser.add_argument_group("Forwarded to UniSkill eval")
    eval_args.add_argument("--eval-config", type=str, required=True)
    eval_args.add_argument("--checkpoint", type=str, required=True)
    eval_args.add_argument("--idm-ckpt-path", type=str, required=True)
    eval_args.add_argument("--output-dir", type=str, required=True)

    eval_args.add_argument(
        "--input-mode",
        type=str,
        default="video_only",
        choices=["video_only", "language_only", "video_and_language"],
    )

    eval_args.add_argument("--human-root", type=str, default="demos")
    eval_args.add_argument("--sim-root", type=str, default="demos")
    eval_args.add_argument("--sim-config", type=str, default="examples/baselines/lerobot_dataset/config/sim_config.json")
    eval_args.add_argument("--human-config", type=str, default="examples/baselines/lerobot_dataset/config/human_config.json")
    eval_args.add_argument("--task-mapping", type=str, default="examples/baselines/lerobot_dataset/task_mapping.json")
    eval_args.add_argument("--human-task-desc", type=str, default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    eval_args.add_argument("--sim-task-desc", type=str, default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")

    eval_args.add_argument("--num-episodes", type=int, default=10)
    eval_args.add_argument("--num-envs", type=int, default=1)
    eval_args.add_argument("--max-episode-steps", type=int, default=500)

    eval_args.add_argument("--sim-backend", type=str, default="physx_cpu")
    eval_args.add_argument("--control-mode", type=str, default="pd_joint_pos")
    eval_args.add_argument("--obs-mode", type=str, default="rgb")
    eval_args.add_argument("--shader", type=str, default="rt-fast")

    eval_args.add_argument("--action-dim", type=int, default=16)
    eval_args.add_argument("--obs-dim", type=int, default=18)
    eval_args.add_argument("--obs-horizon", type=int, default=2)
    eval_args.add_argument("--policy-obs-horizon", type=int, default=None)
    eval_args.add_argument("--policy-pred-horizon", type=int, default=16)
    eval_args.add_argument("--pred-horizon", type=int, default=None)

    eval_args.add_argument("--vision-feature-dim", type=int, default=256)
    eval_args.add_argument("--idm-feature-dim", type=int, default=128)
    eval_args.add_argument("--num-diffusion-iters", type=int, default=100)
    eval_args.add_argument("--resolution", type=int, default=112)
    eval_args.add_argument("--idm-resolution", type=int, default=224)

    eval_args.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    eval_args.add_argument("--cameras", type=str, nargs="+", default=["zed2i"])
    eval_args.add_argument("--include-depth", action="store_true", default=False)
    eval_args.add_argument("--state-type", type=str, default="qpos")
    eval_args.add_argument("--single-arm", action="store_true", default=False)
    eval_args.add_argument("--num-video-frames", type=int, default=10)

    eval_args.add_argument("--temporal-agg", action="store_true", default=False)

    eval_args.add_argument("--vocab-size", type=int, default=32000)
    eval_args.add_argument("--max-text-len", type=int, default=500)
    eval_args.add_argument("--task-seq-len", type=int, default=10)
    eval_args.add_argument("--mano-dim", type=int, default=14)

    eval_args.add_argument(
        "--eval_lr_mirror",
        default="auto",
        choices=["auto", "true", "false"],
    )
    eval_args.add_argument(
        "--eval_lr_mirror_robot_pose",
        default="false",
        choices=["auto", "true", "false"],
    )

    dtw = eval_args.add_mutually_exclusive_group()
    dtw.add_argument("--compute-dtw", dest="compute_dtw", action="store_true", default=False)
    dtw.add_argument("--no-compute-dtw", dest="compute_dtw", action="store_false")
    eval_args.add_argument("--dtw-band-ratio", type=float, default=0.15)

    return parser.parse_args()


if __name__ == "__main__":
    ParallelEvalUniSkillScheduler(parse_args()).run()
