#!/usr/bin/env python3
"""
Parallel Evaluation Scheduler for Diffusion Policy
"""

import os
import sys
import json
import time
import signal
import argparse
import tempfile
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# GPU helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_gpu_free_mem_gb(gpu_id: int) -> float:
    """Return free memory (GB) for a single GPU via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id", str(gpu_id),
             "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return float(out.split("\n")[0]) / 1024.0
    except Exception:
        return 0.0


def parse_gpu_procs(values):
    """Parse ['2:4', '3:3'] → {2: 4, 3: 3}."""
    result = {}
    for v in (values or []):
        gpu_str, n_str = v.split(":")
        result[int(gpu_str)] = int(n_str)
    return result

def pick_gpus(num_gpus: int, min_free_mem_gb: float,
              explicit_ids: Optional[List[int]] = None) -> List[int]:
    candidate_ids = explicit_ids if explicit_ids is not None else list(range(num_gpus))
    result = []
    for gpu_id in candidate_ids:
        free = get_gpu_free_mem_gb(gpu_id)
        if free >= min_free_mem_gb:
            result.append(gpu_id)
        else:
            print(f"  ⚠️  GPU{gpu_id}: only {free:.1f} GB free "
                  f"(need {min_free_mem_gb} GB) — skipping")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# JSON-based skip logic (mirrors eval_dp_imitator.load_existing_results)
# ──────────────────────────────────────────────────────────────────────────────

def _load_existing_results_json(
    output_dir: Path, input_mode: str
) -> "tuple[Dict[str, Dict], set]":
    """
    Scan all JSON result files in output_dir.

    Returns
    -------
    existing : Dict[str, Dict]
        env_id -> best success record (latest timestamp).
    errored  : set[str]
        env_ids that have at least one record with status == "error".

    Scans both *_{input_mode}_*.json and *.json so that files written
    with a different naming convention (no input_mode infix) are also found.
    """
    existing: Dict[str, Dict] = {}
    errored: set = set()

    seen_files: set = set()
    for pattern in (f"*_{input_mode}_*.json", "*.json"):
        for json_file in output_dir.glob(pattern):
            if json_file in seen_files:
                continue
            seen_files.add(json_file)
            try:
                with open(json_file) as f:
                    results = json.load(f)
                if not isinstance(results, list):
                    continue
                for r in results:
                    env_id = r.get("env_id")
                    if not env_id:
                        continue
                    status = r.get("status", "")
                    if status == "success":
                        prev = existing.get(env_id)
                        if prev is None or r.get("timestamp", "") >= prev.get("timestamp", ""):
                            existing[env_id] = r
                    elif status == "error":
                        errored.add(env_id)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    return existing, errored


def is_env_done(env_id: str, output_dir: Path, num_episodes: int,
                input_mode: str) -> bool:
    """
    Return True only when this env_id has a confirmed successful result with
    at least num_episodes recorded in a JSON file.

    Video-file fallback: accepted ONLY when the output directory contains no
    JSON result files at all (true cold-start scenario where the eval script
    predates JSON output).  Once any JSON file has been written, we rely
    entirely on JSON records — old video files from a previous partial run
    must NOT mask missing or failed envs.
    """
    existing, errored = _load_existing_results_json(output_dir, input_mode)

    # Primary: success record with enough episodes.
    prev = existing.get(env_id)
    if prev is not None and prev.get("num_episodes", 0) >= num_episodes:
        return True

    # Video fallback: only when NO JSON files exist in the output dir at all.
    has_any_json = bool(existing) or bool(errored)
    if not has_any_json:
        video_dir = output_dir / "videos" / env_id
        if video_dir.exists() and len(list(video_dir.glob("*.mp4"))) >= num_episodes:
            return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Task state
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkerTask:
    worker_id:    int
    gpu_id:       int
    env_ids:      List[str]
    config_file:  str = ""
    process:      Optional[subprocess.Popen] = None
    status:       str = "pending"   # pending | running | done | failed
    retries:      int = 0
    start_time:   float = 0.0
    output_lines: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────────────────────────────────────

class ParallelEvalScheduler:

    def __init__(self, args):
        self.args       = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = self.output_dir / f"scheduler_logs_{ts}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_dir / "scheduler.log", "w")

        self.lock      = threading.Lock()
        self.stop_flag = False
        self.workers: List[WorkerTask] = []
        # Seconds to wait after a launch before considering that GPU available
        # again. Prevents concurrent workers from all seeing the same "free" VRAM
        # before the first process has finished allocating its model weights.
        self.VRAM_SETTLE_SECS: int = 60

    # ── logging ───────────────────────────────────────────────────────────────

    def _get_limit(self, gpu_id: int) -> int:
        """Per-GPU process limit.  Falls back to max_procs_per_gpu."""
        gpu_procs = getattr(self.args, "gpu_procs", None) or {}
        return gpu_procs.get(gpu_id, self.args.max_procs_per_gpu)

    def log(self, msg: str, level: str = "INFO"):
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        print(line)
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    # ── read env list ─────────────────────────────────────────────────────────

    def load_env_ids(self) -> List[str]:
        with open(self.args.eval_config) as f:
            return [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # ── assign envs to workers ────────────────────────────────────────────────

    def build_workers(self, pending_envs: List[str],
                      gpu_ids: List[int]) -> List[WorkerTask]:
        # Build flat GPU assignment list respecting per-GPU limits:
        #   gpu_ids=[2,3], gpu_procs={2:4, 3:3}  →  [2,2,2,2,3,3,3]
        gpu_slots: List[int] = []
        for g in gpu_ids:
            gpu_slots.extend([g] * self._get_limit(g))
        num_workers = min(len(gpu_slots), len(pending_envs))
        if num_workers == 0:
            return []
        gpu_slots = gpu_slots[:num_workers]

        chunks: Dict[int, List[str]] = defaultdict(list)
        for i, env_id in enumerate(pending_envs):
            chunks[i % num_workers].append(env_id)

        workers = []
        for worker_idx, gpu_id in enumerate(gpu_slots):
            env_list = chunks[worker_idx]
            if not env_list:
                continue
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt",
                prefix=f"eval_dp_worker{worker_idx}_",
                dir=self.log_dir, delete=False,
            )
            tmp.write("\n".join(env_list) + "\n")
            tmp.close()
            workers.append(WorkerTask(
                worker_id=worker_idx,
                gpu_id=gpu_id,
                env_ids=env_list,
                config_file=tmp.name,
            ))
        return workers

    # ── build subprocess command ──────────────────────────────────────────────

    def build_cmd(self, worker: "WorkerTask") -> list:  # parallel_eval_dp.py version
        a = self.args
        cmd = [
            sys.executable, "-m",
            "examples.baselines.diffusion_policy.eval_dp_imitator",   # ← different module
            "--eval-config",         worker.config_file,
            "--checkpoint",          a.checkpoint,
            "--output-dir",          a.output_dir,
            "--num-episodes",        str(a.num_episodes),
            "--num-envs",            str(a.num_envs),
            "--input-mode",          a.input_mode,
            "--sim-backend",         a.sim_backend,
            "--control-mode",        a.control_mode,
            "--obs-mode",            a.obs_mode,
            "--max-episode-steps",   str(a.max_episode_steps),
            "--action-dim",          str(a.action_dim),
            "--state-dim",           str(a.state_dim),
            "--obs-horizon",         str(a.obs_horizon),
            "--pred-horizon",        str(a.pred_horizon),
            "--obs-latent-dim",      str(a.obs_latent_dim),
            "--task-latent-dim",     str(a.task_latent_dim),
            "--task-num-frames",     str(a.task_num_frames),
            "--image-size",          str(a.image_size[0]), str(a.image_size[1]),
            "--cameras",             *a.cameras,
            "--state-type",          a.state_type,
            "--vocab-size",          str(a.vocab_size),
            "--max-text-len",        str(a.max_text_len),
            "--num-tasks",           str(a.num_tasks),
            "--human-root",          a.human_root,
            "--sim-root",            a.sim_root,
            "--sim-config",          a.sim_config,
            "--human-config",        a.human_config,
            "--task-mapping",        a.task_mapping,
            "--human-task-desc",     a.human_task_desc,
            "--sim-task-desc",       a.sim_task_desc,
            "--device",              "cuda",
            "--shader",              a.shader,
            "--task-seq-len",        str(a.task_seq_len),
            "--mano-dim",            str(a.mano_dim),
            "--task-encoder-type",   a.task_encoder_type,
            "--num-diffusion-iters", str(a.num_diffusion_iters),
        ]
 
        if a.include_depth:  cmd.append("--include-depth")
        if a.single_arm:     cmd.append("--single-arm")
        if a.temporal_agg:   cmd.append("--temporal-agg")
        if a.use_ddim:
            cmd += ["--use-ddim", "--num-ddim-steps", str(a.num_ddim_steps)]
 
        # ── Frozen-backbone-specific args ────────────────────────────────────
        if a.task_encoder_type == "frozen_backbone":
            cmd += [
                "--frozen-backbone-type",           a.frozen_backbone_type,
                "--frozen-backbone-adapter-layers", str(a.frozen_backbone_adapter_layers),
                "--frozen-backbone-seq-patches",    str(a.frozen_backbone_seq_patches),
                "--frozen-backbone-num-frames",     str(a.frozen_backbone_num_frames),
                "--frozen-backbone-lora-rank",      str(a.frozen_backbone_lora_rank),
                "--frozen-backbone-lora-alpha",     str(a.frozen_backbone_lora_alpha),
            ]
            if a.frozen_backbone_model:
                cmd += ["--frozen-backbone-model", a.frozen_backbone_model]
            if a.frozen_backbone_adapter_ln:
                cmd.append("--frozen-backbone-adapter-ln")
 
        return cmd

    # ── stream output ─────────────────────────────────────────────────────────

    def _stream_output(self, worker: WorkerTask):
        prefix      = f"[W{worker.worker_id}|GPU{worker.gpu_id}]"
        worker_log  = self.log_dir / f"worker_{worker.worker_id}.log"
        with open(worker_log, "w") as fh:
            for line in worker.process.stdout:
                line = line.rstrip()
                worker.output_lines.append(line)
                if len(worker.output_lines) > 200:
                    worker.output_lines = worker.output_lines[-200:]
                tagged = f"{prefix} {line}"
                print(tagged)
                fh.write(tagged + "\n")
                fh.flush()

    # ── launch a worker ───────────────────────────────────────────────────────

    def launch_worker(self, worker: WorkerTask):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker.gpu_id)

        cmd = self.build_cmd(worker)
        self.log(f"🚀 Launching Worker {worker.worker_id} on GPU {worker.gpu_id} "
                 f"| {len(worker.env_ids)} envs "
                 f"| backend={self.args.task_encoder_type}")

        worker.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        worker.status     = "running"
        worker.start_time = time.time()

        t = threading.Thread(target=self._stream_output, args=(worker,), daemon=True)
        t.start()

    # ── monitor workers ───────────────────────────────────────────────────────

    def _monitor(self):
        while not self.stop_flag:
            time.sleep(10)
            with self.lock:
                for w in self.workers:
                    if w.status != "running":
                        continue
                    ret = w.process.poll()
                    if ret is None:
                        continue
                    elapsed = (time.time() - w.start_time) / 60
                    if ret == 0:
                        w.status = "done"
                        self.log(f"✅ Worker {w.worker_id} finished ({elapsed:.1f} min)")
                    else:
                        w.retries += 1
                        if w.retries <= self.args.max_retries:
                            self.log(f"⚠️  Worker {w.worker_id} failed (ret={ret}), "
                                     f"retry {w.retries}/{self.args.max_retries}", "WARNING")
                            w.status = "pending"
                        else:
                            w.status = "failed"
                            self.log(f"❌ Worker {w.worker_id} failed after "
                                     f"{w.retries} retries", "ERROR")

    # ── status print ──────────────────────────────────────────────────────────

    def _status(self):
        counts = defaultdict(int)
        for w in self.workers:
            counts[w.status] += 1
        self.log(
            f"📊 Workers — pending:{counts['pending']} "
            f"running:{counts['running']} "
            f"done:{counts['done']} "
            f"failed:{counts['failed']}"
        )
        for gpu_id in sorted(set(w.gpu_id for w in self.workers)):
            free = get_gpu_free_mem_gb(gpu_id)
            self.log(f"   GPU{gpu_id}: {free:.1f} GB free")

    # ── main run ──────────────────────────────────────────────────────────────

    def run(self):
        self.log("=" * 70)
        self.log("🧪 PARALLEL EVAL SCHEDULER — Diffusion Policy (frozen-backbone)")
        self.log("=" * 70)
        self.log(f"   Task encoder backend : {self.args.task_encoder_type}")
        self.log(f"   Input mode           : {self.args.input_mode}")

        # ── 1. Discover envs ──────────────────────────────────────────────────
        all_envs = self.load_env_ids()
        self.log(f"Total envs in config: {len(all_envs)}")

        # Load the result cache once to avoid re-reading JSON files per env.
        existing, errored = _load_existing_results_json(
            self.output_dir, self.args.input_mode
        )
        pending, skipped = [], 0
        for env_id in all_envs:
            prev = existing.get(env_id)
            if prev is not None and prev.get("num_episodes", 0) >= self.args.num_episodes:
                skipped += 1
                continue
            # Video fallback only on true cold-start (no JSON files written yet).
            has_any_json = bool(existing) or bool(errored)
            if not has_any_json:
                video_dir = self.output_dir / "videos" / env_id
                if (video_dir.exists() and
                        len(list(video_dir.glob("*.mp4"))) >= self.args.num_episodes):
                    skipped += 1
                    continue
            pending.append(env_id)
        self.log(f"Already done (JSON or video): {skipped}  |  Pending: {len(pending)}")
        if errored:
            self.log(f"  ({len(errored)} env(s) have explicit error records "
                     f"and will be re-run even if videos exist)")

        if not pending:
            self.log("✅ All environments already evaluated. Nothing to do.")
            return

        # ── 2. Pick GPUs ──────────────────────────────────────────────────────
        self.log(f"\nChecking GPU memory (need ≥ {self.args.min_free_mem_gb} GB each)...")
        WAIT_POLL_INTERVAL = 30  # seconds

        while True:
            gpu_ids = pick_gpus(
                self.args.num_gpus,
                self.args.min_free_mem_gb,
                explicit_ids=self.args.gpu_ids,
            )
            if gpu_ids:
                break
            self.log(
                f"⏳ No GPU with ≥{self.args.min_free_mem_gb} GB free, "
                f"retrying in {WAIT_POLL_INTERVAL}s...", "WARNING"
            )
            time.sleep(WAIT_POLL_INTERVAL)

        self.log(f"Using GPUs: {gpu_ids}")

        # ── 3. Build workers ─────────────────────────────────────────────────
        self.workers = self.build_workers(pending, gpu_ids)
        self.log(f"Built {len(self.workers)} workers ({len(pending)} envs total).")
        self.log(f"VRAM-aware launch: max_procs_per_gpu={self.args.max_procs_per_gpu}, "
                 f"min_free_mem_gb={self.args.min_free_mem_gb}, "
                 f"settle={self.VRAM_SETTLE_SECS}s\n")

        def _sighandler(sig, frame):
            self.log("⚠️  Interrupted — killing workers...", "WARNING")
            self.stop_flag = True
            for w in self.workers:
                if w.process and w.process.poll() is None:
                    w.process.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT,  _sighandler)
        signal.signal(signal.SIGTERM, _sighandler)

        monitor_t = threading.Thread(target=self._monitor, daemon=True)
        monitor_t.start()

        # ── 4. VRAM-aware unified scheduling loop ─────────────────────────────
        # Workers are NOT all launched at once.  In each iteration we check
        # every unstarted (or retry-pending) worker and launch it only when:
        #   (a) running workers on its GPU < max_procs_per_gpu
        #   (b) the GPU has been idle for at least VRAM_SETTLE_SECS since the
        #       last launch (gives the previous process time to allocate memory
        #       so the next VRAM check reflects reality)
        #   (c) free VRAM on its GPU >= min_free_mem_gb
        # ─────────────────────────────────────────────────────────────────────
        # gpu_last_launch[gpu_id] = monotonic time of the most recent launch on
        # that GPU.  Shared between initial launches and retries.
        gpu_last_launch: Dict[int, float] = {g: 0.0 for g in gpu_ids}

        # Workers waiting for their first launch (not yet started at all).
        pending_launch: List[WorkerTask] = list(self.workers)

        last_status = 0.0
        while not self.stop_flag:

            # ── Try to schedule pending (first-launch) workers ────────────────
            still_pending: List[WorkerTask] = []
            for w in pending_launch:
                gpu_id = w.gpu_id

                # (a) Slot check: how many workers are running on this GPU?
                running_on_gpu = sum(
                    1 for x in self.workers
                    if x.gpu_id == gpu_id and x.status == "running"
                )
                if running_on_gpu >= self._get_limit(gpu_id):
                    still_pending.append(w)
                    continue

                # (b) Settle window: don't probe VRAM right after a launch
                elapsed = time.monotonic() - gpu_last_launch.get(gpu_id, 0.0)
                if elapsed < self.VRAM_SETTLE_SECS:
                    remaining = self.VRAM_SETTLE_SECS - elapsed
                    self.log(
                        f"⏳ GPU{gpu_id} settle window ({remaining:.0f}s left) — "
                        f"Worker {w.worker_id} queued"
                    )
                    still_pending.append(w)
                    continue

                # (c) VRAM check
                free = get_gpu_free_mem_gb(gpu_id)
                if free < self.args.min_free_mem_gb:
                    self.log(
                        f"⏳ GPU{gpu_id}: {free:.1f} GB free < "
                        f"{self.args.min_free_mem_gb} GB needed — "
                        f"Worker {w.worker_id} queued",
                        "WARNING",
                    )
                    still_pending.append(w)
                    continue

                # ✓ All checks passed — launch
                self.launch_worker(w)
                gpu_last_launch[gpu_id] = time.monotonic()

            pending_launch = still_pending

            # ── Retry failed workers through the same VRAM gate ───────────────
            for w in self.workers:
                if w.status != "pending" or w.retries == 0:
                    continue
                gpu_id = w.gpu_id

                running_on_gpu = sum(
                    1 for x in self.workers
                    if x.gpu_id == gpu_id and x.status == "running"
                )
                if running_on_gpu >= self._get_limit(gpu_id):
                    continue

                elapsed = time.monotonic() - gpu_last_launch.get(gpu_id, 0.0)
                if elapsed < self.VRAM_SETTLE_SECS:
                    continue

                free = get_gpu_free_mem_gb(gpu_id)
                if free >= self.args.min_free_mem_gb:
                    self.log(f"🔄 Retrying Worker {w.worker_id} on GPU {w.gpu_id} "                             f"({free:.1f} GB free)")
                    self.launch_worker(w)
                    gpu_last_launch[gpu_id] = time.monotonic()
                else:
                    self.log(
                        f"  GPU{w.gpu_id}: {free:.1f} GB free — retry for "                        f"Worker {w.worker_id} deferred",
                        "WARNING",
                    )

            # ── Termination check ─────────────────────────────────────────────
            all_done = (
                not pending_launch
                and all(w.status in ("done", "failed") for w in self.workers)
            )
            if all_done:
                break

            if time.time() - last_status > 60:
                self._status()
                last_status = time.time()

            time.sleep(10)

        # ── 5. Final summary ──────────────────────────────────────────────────
        self.log("\n" + "=" * 70)
        self.log("📊 FINAL SUMMARY")
        self.log("=" * 70)
        self._status()

        done_count   = sum(1 for w in self.workers if w.status == "done")
        failed_count = sum(1 for w in self.workers if w.status == "failed")
        self.log(f"Workers done: {done_count}  |  Workers failed: {failed_count}")

        if failed_count > 0:
            self.log("\n❌ Failed workers:", "ERROR")
            for w in self.workers:
                if w.status == "failed":
                    self.log(f"   Worker {w.worker_id} (GPU{w.gpu_id}): "
                             f"{w.env_ids}", "ERROR")
                    if w.output_lines:
                        self.log(f"   Last output:\n"
                                 + "\n".join(f"     {l}" for l in w.output_lines[-5:]),
                                 "ERROR")

        self.log(f"\n✅ Scheduler finished.  Logs: {self.log_dir}")
        self._log_fh.close()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel evaluation scheduler for Diffusion Policy (frozen_backbone)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Scheduler-specific ────────────────────────────────────────────────────
    g = p.add_argument_group("Scheduler")
    g.add_argument("--num-gpus",          type=int,   default=4,
                   help="Auto-scan GPU 0..N-1 (overridden by --gpu-ids)")
    g.add_argument("--gpu-ids",           type=int,   nargs="+", default=None,
                   metavar="ID",
                   help="Explicit GPU IDs to use, e.g. --gpu-ids 0 2 3")
    g.add_argument("--max-procs-per-gpu", type=int,   default=1)
    g.add_argument("--gpu-procs",         type=str,   nargs="+", default=None,
                   metavar="GPU:N",
                   help="Per-GPU process limit, e.g. --gpu-procs 2:4 3:3. "
                        "Overrides --max-procs-per-gpu for the specified GPUs.")
    g.add_argument("--min-free-mem-gb",   type=float, default=16.0)
    g.add_argument("--max-retries",       type=int,   default=2)

    # ── Forwarded to eval_dp_imitator ────────────────────────────────────────────────
    g2 = p.add_argument_group("Eval (forwarded to eval_dp_imitator)")
    g2.add_argument("--eval-config",          type=str, required=True)
    g2.add_argument("--checkpoint",           type=str, required=True)
    g2.add_argument("--output-dir",           type=str, required=True)
    g2.add_argument("--input-mode",           type=str, default="video_only",
                    choices=["video_only", "language_only", "video_and_language"])
    g2.add_argument("--num-episodes",         type=int,   default=10)
    g2.add_argument("--num-envs",             type=int,   default=1)
    g2.add_argument("--max-episode-steps",    type=int,   default=500)
    g2.add_argument("--sim-backend",          type=str,   default="physx_cpu")
    g2.add_argument("--control-mode",         type=str,   default="pd_joint_pos")
    g2.add_argument("--obs-mode",             type=str,   default="rgb")
    g2.add_argument("--action-dim",           type=int,   default=16)
    g2.add_argument("--state-dim",            type=int,   default=18)
    g2.add_argument("--obs-horizon",          type=int,   default=1)
    g2.add_argument("--pred-horizon",         type=int,   default=16)
    g2.add_argument("--obs-latent-dim",       type=int,   default=256)
    g2.add_argument("--task-latent-dim",      type=int,   default=256)
    g2.add_argument("--task-num-frames",      type=int,   default=10)
    g2.add_argument("--image-size",           type=int,   nargs=2, default=[224, 224])
    g2.add_argument("--cameras",              type=str,   nargs="+", default=["zed2i"])
    g2.add_argument("--include-depth",        action="store_true", default=False)
    g2.add_argument("--state-type",           type=str,   default="qpos")
    g2.add_argument("--single-arm",           action="store_true", default=False)
    g2.add_argument("--vocab-size",           type=int,   default=32000)
    g2.add_argument("--max-text-len",         type=int,   default=500)
    g2.add_argument("--num-tasks",            type=int,   default=200)
    g2.add_argument("--temporal-agg",         action="store_true", default=False)
    g2.add_argument("--use-ddim",              action="store_true", default=True,
                    help="Use DDIMScheduler at eval (10 steps, ~lossless speedup)")
    g2.add_argument("--num-ddim-steps",        type=int, default=10,
                    help="DDIM steps (only used when --use-ddim)")
    g2.add_argument("--shader",               type=str,   default="rt-fast")
    g2.add_argument("--human-root",           type=str,   default="demos")
    g2.add_argument("--sim-root",             type=str,   default="demos")
    g2.add_argument("--sim-config",           type=str,
                    default="examples/baselines/lerobot_dataset/config/sim_config.json")
    g2.add_argument("--human-config",         type=str,
                    default="examples/baselines/lerobot_dataset/config/human_config.json")
    g2.add_argument("--task-mapping",         type=str,
                    default="examples/baselines/lerobot_dataset/task_mapping.json")
    g2.add_argument("--human-task-desc",      type=str,
                    default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    g2.add_argument("--sim-task-desc",        type=str,
                    default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")
    g2.add_argument("--task-seq-len",         type=int,   default=10)
    g2.add_argument("--mano-dim",             type=int,   default=14)
    g2.add_argument("--num-diffusion-iters",  type=int,   default=100)

# ── Task encoder backend ──────────────────────────────────────────────────
    g2.add_argument("--task-encoder-type", type=str, default="frozen_backbone",
                    choices=["frozen_backbone"])
 
    # ── Frozen backbone group ─────────────────────────────────────────────────
    g4 = p.add_argument_group("Frozen backbone")
    g4.add_argument("--frozen-backbone-type", type=str, default="dinov2_vitl14",
                    choices=["dinov2_vitl14", "dinov2_vitb14", "dinov2_vitl14_reg",
                             "clip_vitl14", "clip_vitb16", "siglip2_so400m",
                             "videomae_large", "videomae_base"])
    g4.add_argument("--frozen-backbone-model",          type=str,  default=None)
    g4.add_argument("--frozen-backbone-adapter-layers", type=int,  default=1)
    g4.add_argument("--frozen-backbone-adapter-ln",     action="store_true", default=True)
    g4.add_argument("--frozen-backbone-seq-patches",    type=int,  default=32)
    g4.add_argument("--frozen-backbone-num-frames",     type=int,  default=4)
    g4.add_argument("--frozen-backbone-lora-rank",  type=int,   default=0)
    g4.add_argument("--frozen-backbone-lora-alpha", type=float, default=16.0)

    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    _args.gpu_procs = parse_gpu_procs(_args.gpu_procs)
    ParallelEvalScheduler(_args).run()