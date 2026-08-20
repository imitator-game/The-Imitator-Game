#!/usr/bin/env python3
"""
Precompute and cache VLA action_hidden_states for all training samples.

V4 Changes vs V3  ── RAM-Buffer + Async Disk Writer
-----------------------------------------------------
ROOT CAUSE of the 20-minute flush stall:
  np.memmap.flush() calls msync(MS_SYNC) on Linux — NOT MS_ASYNC.
  Even in a background thread, commit() blocks on join() for the full
  20 minutes. The HPC parallel filesystem (~8 MB/s effective write
  throughput) makes a 10 GB shard take 20 min no matter how the
  syscall is scheduled.

V4 FIX — eliminate msync(MS_SYNC) entirely:

  1. Replace np.memmap (write mode) with plain np.empty() RAM buffers.
     Writes go directly to RAM — no dirty pages, no flush, no stall.

  2. When a shard buffer is full, hand it to _AsyncDiskWriter via a
     thread-safe queue. The background thread writes feat/actions/states
     using POSIX write() (tofile), renames tmp→final atomically, then
     writes shard_meta.json with completed=True.

  3. The GPU forward loop immediately starts filling the NEXT shard's
     RAM buffer — GPU compute and disk I/O are fully overlapped.

  4. _AsyncDiskWriter.submit() provides back-pressure: if the previous
     shard's disk write is still running when the next one is ready,
     submit() blocks until it finishes. This is the only stall point,
     and it only fires once per shard boundary.

Timeline (GPU=2 min/shard, disk=20 min/shard, N shards):
  V2/V3:  (2+20) × N = 22N min   [main thread stalls on msync every shard]
  V4:      20 × N + 2 ≈ 20N min  [disk is the bottleneck; GPU runs free]

V3 features retained:
  • Single DataLoader across all pending shards  (no per-shard re-fork)
  • _CudaPrefetcher  (CUDA-stream H2D / forward overlap)
  • madvise(MADV_DONTNEED) after RAM buffer is handed off to disk
  • Shard-level resume, --pool_features, monolithic fallback, timing debug

Reading side (lerobot_dataset.py) is UNCHANGED — tofile() produces
byte-identical files to np.memmap, so the sharded reader works as-is.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import threading
import time
from collections import deque
from functools import partial
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from examples.baselines.openvla_oft.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from examples.baselines.openvla_oft.prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from examples.baselines.openvla_oft.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from examples.baselines.openvla_oft.prismatic.vla.action_tokenizer import ActionTokenizer
from examples.baselines.openvla_oft.prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from examples.baselines.openvla_oft.timing_debug import print_timing, should_emit
from examples.baselines.openvla_oft.train_openvla import openvla_collate_fn
from examples.baselines.openvla_oft.utils.utils import worker_init_fn


# ============================================================================
# Dataset builder
# ============================================================================

def _build_dataset(args, processor, action_tokenizer):
    if not args.use_lerobot:
        raise ValueError("Only --use_lerobot is supported by this precompute script.")

    from examples.baselines.openvla_oft.lerobot_dataset import OpenVLAPairedDataset
    from examples.baselines.lerobot_dataset.lerobot_paired_dataset import PairedDatasetConfig

    if args.sim_dataset_file is None:
        raise ValueError("--sim_dataset_file is required when --use_lerobot is set.")
    if args.human_dataset_file is None:
        raise ValueError("--human_dataset_file is required when --use_lerobot is set.")

    paired_config = PairedDatasetConfig(
        human_root=args.human_root,
        sim_root=args.sim_root,
        task_mapping_file=args.task_mapping_file,
        human_dataset_file=args.human_dataset_file,
        sim_dataset_file=args.sim_dataset_file,
        human_task_description_file=args.human_task_description_file,
        sim_task_description_file=args.sim_task_description_file,
        split=args.sim_split,
        cameras=[args.lerobot_camera],
        include_depth=False,
        image_size=tuple(args.image_size),
        horizon=args.pred_horizon,
        obs_horizon=args.obs_horizon,
        video_backend=args.lerobot_video_backend or "torchcodec",
        input_mode="language_only",
    )
    return OpenVLAPairedDataset(
        paired_config=paired_config,
        processor=processor,
        action_tokenizer=action_tokenizer,
        default_language=args.default_language,
        action_dim=args.action_dim,
        pred_horizon=args.pred_horizon,
        image_size=tuple(args.image_size),
        vla_cache_dir=None,
    )


# ============================================================================
# Shard path / meta helpers
# ============================================================================

def _shard_dir(cache_dir: str, shard_idx: int) -> str:
    return os.path.join(cache_dir, "shards", f"{shard_idx:05d}")


def _shard_meta_path(cache_dir: str, shard_idx: int) -> str:
    return os.path.join(_shard_dir(cache_dir, shard_idx), "shard_meta.json")


def _is_shard_complete(cache_dir: str, shard_idx: int) -> bool:
    p = _shard_meta_path(cache_dir, shard_idx)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            return bool(json.load(f).get("completed", False))
    except Exception:
        return False


def _write_shard_meta(
    cache_dir: str, shard_idx: int,
    start: int, end: int, shard_n: int,
    feat_shape: tuple, completed: bool, pool_features: bool,
) -> None:
    sd = _shard_dir(cache_dir, shard_idx)
    os.makedirs(sd, exist_ok=True)
    meta = dict(shard_idx=shard_idx, start=start, end=end, shard_n=shard_n,
                feat_shape=list(feat_shape), pool_features=pool_features, completed=completed)
    tmp = _shard_meta_path(cache_dir, shard_idx) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, _shard_meta_path(cache_dir, shard_idx))


def _write_top_meta(
    cache_dir: str, total: int, feat_shape: tuple,
    pred_horizon: int, action_dim: int, has_proprio: bool,
    pool_features: bool, shard_size: int, num_shards: int,
    completed: bool, fmt: str = "sharded",
) -> None:
    meta = dict(
        format=fmt, total_samples=total, feat_shape=list(feat_shape),
        pool_features=pool_features, pred_horizon=pred_horizon, action_dim=action_dim,
        has_proprio=has_proprio, dtype="float16",
        shard_size=shard_size, num_shards=num_shards, completed=completed,
        num_tokens=feat_shape[0] if not pool_features else 1,
        llm_dim=feat_shape[-1],
    )
    tmp = os.path.join(cache_dir, "vla_cache_meta.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, os.path.join(cache_dir, "vla_cache_meta.json"))


def _read_meta(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _any_shard_has_proprio(cache_dir: str, num_shards: int) -> bool:
    for si in range(num_shards):
        if os.path.exists(os.path.join(_shard_dir(cache_dir, si), "states.bin")):
            return True
    return False


# ============================================================================
# madvise(MADV_DONTNEED)  —  release RAM pages after hand-off to disk writer
# ============================================================================

_libc: Optional[ctypes.CDLL] = None


def _get_libc() -> Optional[ctypes.CDLL]:
    global _libc
    if _libc is None:
        try:
            _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError:
            _libc = None
    return _libc


def _madvise_dontneed(arr: np.ndarray) -> None:
    """Hint the OS to reclaim `arr`'s physical pages. Linux-only, non-fatal."""
    libc = _get_libc()
    if libc is None:
        return
    libc.madvise(
        ctypes.c_void_p(arr.ctypes.data),
        ctypes.c_size_t(arr.nbytes),
        ctypes.c_int(4),   # MADV_DONTNEED
    )


# ============================================================================
# _write_array_slice  —  write first n rows of arr to path via POSIX write
# ============================================================================

def _write_array_slice(arr: np.ndarray, n: int, path: str) -> None:
    """
    Write arr[:n] to `path` using buffered POSIX write.

    Uses memoryview to avoid a Python-level copy when n < arr.shape[0].
    No msync / fsync: the kernel handles page-cache writeback asynchronously.
    Atomicity is provided by the caller's tmp→final rename.
    """
    view = memoryview(arr[:n])     # zero-copy slice (arr is C-contiguous)
    with open(path, "wb") as f:
        f.write(view)


# ============================================================================
# _RamShardBuffer  —  one shard fully in RAM, zero disk I/O on write_batch()
# ============================================================================

class _RamShardBuffer:
    """
    Accumulates all feature/action/state data for one shard in RAM.

    write_batch() is a plain numpy slice assignment — effectively free
    (~200 GB/s RAM bandwidth) compared to GPU forward or disk write.

    When is_full, hand this object to _AsyncDiskWriter.submit().
    """

    def __init__(
        self,
        shard_idx: int,
        shard_n: int,
        feat_shape: tuple,
        pred_horizon: int,
        action_dim: int,
    ) -> None:
        self.shard_idx   = shard_idx
        self.shard_n     = shard_n
        self.feat_shape  = feat_shape
        self._offset     = 0
        self.has_proprio = False

        self.feat_buf = np.empty((shard_n, *feat_shape), dtype=np.float16)
        self.act_buf  = np.empty((shard_n, pred_horizon, action_dim), dtype=np.float32)
        self.sta_buf  = np.empty((shard_n, action_dim), dtype=np.float32)

    def write_batch(
        self,
        feat: np.ndarray,
        actions: np.ndarray,
        proprio: Optional[np.ndarray],
    ) -> int:
        """Copy batch data into RAM buffers. Returns rows actually written."""
        cap = self.shard_n - self._offset
        bs  = min(feat.shape[0], cap)
        end = self._offset + bs

        self.feat_buf[self._offset:end] = feat[:bs]
        self.act_buf[self._offset:end]  = actions[:bs]
        if proprio is not None:
            self.sta_buf[self._offset:end] = proprio[:bs]
            self.has_proprio = True

        self._offset += bs
        return bs

    @property
    def is_full(self) -> bool:
        return self._offset >= self.shard_n

    @property
    def n_written(self) -> int:
        return self._offset


# ============================================================================
# _DiskWriteJob  —  immutable job descriptor for the background writer thread
# ============================================================================

class _DiskWriteJob(NamedTuple):
    buf:           _RamShardBuffer
    cache_dir:     str
    global_start:  int
    pool_features: bool
    feat_shape:    tuple


# ============================================================================
# _AsyncDiskWriter  —  single background thread writing shards to disk
# ============================================================================

class _AsyncDiskWriter:
    """
    Accepts completed _RamShardBuffer objects and writes them to disk in a
    single daemon thread, completely off the GPU-forward critical path.

    submit(job):
        Blocks until the PREVIOUS disk write finishes, then starts the new
        write in the background and returns immediately.
        The GPU can now start filling the next shard's RAM buffer.

    join() / shutdown():
        Wait for the currently running write to complete.
        Re-raise any exception that occurred in the background thread.

    Why one thread (not a pool)?
        A single sequential writer avoids interleaving writes from multiple
        shards on the same filesystem stripe, which would cause random I/O
        and reduce throughput on Lustre/GPFS.
    """

    _SENTINEL = object()

    def __init__(self) -> None:
        self._q: "queue.Queue" = queue.Queue(maxsize=1)
        self._error: Optional[BaseException] = None
        self._done  = threading.Event()
        self._done.set()   # initially idle
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------

    def submit(self, job: _DiskWriteJob) -> None:
        """Wait for previous write, then start new write in background."""
        self._done.wait()                 # back-pressure: wait for previous shard
        self._reraise_if_error()
        self._done.clear()               # mark in-progress before enqueue
        self._q.put(job)

    def join(self) -> None:
        """Block until current write is complete. Re-raises exceptions."""
        self._done.wait()
        self._reraise_if_error()

    def shutdown(self) -> None:
        """Wait for current write, then stop the worker thread."""
        self.join()
        self._q.put(self._SENTINEL)
        self._thread.join()

    # ------------------------------------------------------------------

    def _reraise_if_error(self) -> None:
        if self._error is not None:
            err, self._error = self._error, None
            raise RuntimeError(f"[AsyncDiskWriter] disk write failed") from err

    def _worker(self) -> None:
        while True:
            job = self._q.get()
            if job is self._SENTINEL:
                return
            try:
                self._write(job)
            except BaseException as exc:
                self._error = exc
            finally:
                self._done.set()

    # ------------------------------------------------------------------

    @staticmethod
    def _write(job: _DiskWriteJob) -> None:
        """
        Write a completed shard buffer to disk:
        1. tofile() each array to *.bin.tmp  (no msync/fsync)
        2. madvise(MADV_DONTNEED) to release RAM pages
        3. Atomic rename tmp → final
        4. Write shard_meta.json with completed=True

        The rename (step 3) happens AFTER tofile() returns (data is in page
        cache). shard_meta (step 4) is written AFTER the rename, so the reader
        only ever sees a complete shard.
        """
        buf = job.buf
        sd  = _shard_dir(job.cache_dir, buf.shard_idx)
        os.makedirs(sd, exist_ok=True)
        n   = buf.n_written

        feat_tmp      = os.path.join(sd, "feat.bin.tmp")
        actions_tmp   = os.path.join(sd, "actions.bin.tmp")
        states_tmp    = os.path.join(sd, "states.bin.tmp")
        feat_final    = os.path.join(sd, "feat.bin")
        actions_final = os.path.join(sd, "actions.bin")
        states_final  = os.path.join(sd, "states.bin")

        t_write = time.time()

        _write_array_slice(buf.feat_buf, n, feat_tmp)
        _write_array_slice(buf.act_buf,  n, actions_tmp)
        if buf.has_proprio:
            _write_array_slice(buf.sta_buf, n, states_tmp)

        elapsed_write = time.time() - t_write

        # Release physical RAM pages — buffers are no longer needed
        _madvise_dontneed(buf.feat_buf)
        _madvise_dontneed(buf.act_buf)
        _madvise_dontneed(buf.sta_buf)

        # Atomic rename
        os.replace(feat_tmp,    feat_final)
        os.replace(actions_tmp, actions_final)
        if buf.has_proprio and os.path.exists(states_tmp):
            os.replace(states_tmp, states_final)
        elif os.path.exists(states_tmp):
            os.remove(states_tmp)

        # Mark complete (reader only trusts shard after this)
        _write_shard_meta(
            job.cache_dir, buf.shard_idx,
            start=job.global_start,
            end=job.global_start + n,
            shard_n=n,
            feat_shape=job.feat_shape,
            completed=True,
            pool_features=job.pool_features,
        )

        gb = buf.feat_buf[:n].nbytes / 1e9
        print(
            f"\n[DiskWriter] Shard {buf.shard_idx:05d}: "
            f"{n:,} samples, {gb:.2f} GB feat written in {elapsed_write:.1f}s"
        )


# ============================================================================
# _ShardPipeline  —  routes GPU batches → RAM buffers → disk writer
# ============================================================================

class _ShardPipeline:
    """
    Manages a queue of _RamShardBuffers, one per pending shard.

    write_batch():
        Fills the current buffer from the GPU batch (nanosecond-level RAM copy).
        When a buffer is full, submits it to _AsyncDiskWriter and opens a new one.
        Handles mid-batch shard transitions (when batch_size ∤ shard_size).

    The GPU loop calls write_batch() and commit_last() only. It never touches
    disk I/O directly.
    """

    def __init__(
        self,
        cache_dir: str,
        feat_shape: tuple,
        pred_horizon: int,
        action_dim: int,
        pool_features: bool,
        disk_writer: _AsyncDiskWriter,
        pending_specs: List[Tuple[int, int, int]],  # [(shard_idx, global_start, shard_n)]
    ) -> None:
        self._cache_dir     = cache_dir
        self._feat_shape    = feat_shape
        self._pred_horizon  = pred_horizon
        self._action_dim    = action_dim
        self._pool_features = pool_features
        self._disk_writer   = disk_writer

        self._pending: Deque[Tuple[int, int, int]] = deque(pending_specs)
        self._current: Optional[_RamShardBuffer]   = None
        self._current_gs: int                       = 0
        self.has_proprio_global                     = False

        self._open_next()

    # ------------------------------------------------------------------

    def _open_next(self) -> None:
        if not self._pending:
            self._current = None
            return
        si, gs, sn   = self._pending.popleft()
        self._current = _RamShardBuffer(si, sn, self._feat_shape,
                                         self._pred_horizon, self._action_dim)
        self._current_gs = gs
        print(f"\n[Precompute] Shard {si:05d}: RAM buffer allocated "
              f"(global [{gs:,}–{gs+sn:,}), n={sn:,})")

    def _submit_current(self) -> None:
        buf = self._current
        if buf is None:
            return
        self.has_proprio_global = self.has_proprio_global or buf.has_proprio
        job = _DiskWriteJob(
            buf=buf,
            cache_dir=self._cache_dir,
            global_start=self._current_gs,
            pool_features=self._pool_features,
            feat_shape=self._feat_shape,
        )
        # Back-pressure: waits if previous shard disk write is still running.
        # Only fires once per shard boundary. GPU is free during the wait.
        self._disk_writer.submit(job)
        self._open_next()

    # ------------------------------------------------------------------

    def write_batch(
        self,
        feat: np.ndarray,
        actions: np.ndarray,
        proprio: Optional[np.ndarray],
    ) -> None:
        """Distribute batch across one or more shard buffers (handles splits)."""
        rem_f = feat
        rem_a = actions
        rem_p = proprio
        while rem_f.shape[0] > 0 and self._current is not None:
            written = self._current.write_batch(rem_f, rem_a, rem_p)
            rem_f   = rem_f[written:]
            rem_a   = rem_a[written:]
            if rem_p is not None:
                rem_p = rem_p[written:]
            if self._current.is_full:
                self._submit_current()

    def commit_last(self) -> None:
        """Submit the final (possibly partial) buffer after the loader is exhausted."""
        if self._current is not None and self._current.n_written > 0:
            self._submit_current()

    def abort(self) -> None:
        self._current = None
        self._pending.clear()


# ============================================================================
# _CudaPrefetcher  —  H2D transfer overlapped with GPU forward
# ============================================================================

class _CudaPrefetcher:
    """
    Pre-transfers the next batch's tensors to GPU on a dedicated CUDA stream
    while the current batch's forward pass runs on the default stream.
    Returns (gpu_dict, cpu_dict) pairs.
    """

    def __init__(self, loader: DataLoader, device: torch.device, dtype: torch.dtype) -> None:
        self._iter       = iter(loader)
        self._device     = device
        self._dtype      = dtype
        self._h2d_stream = torch.cuda.Stream(device=device)
        self._next_gpu:  Optional[Dict[str, torch.Tensor]] = None
        self._next_cpu:  Optional[Dict]                    = None
        self._stopped    = False
        self._prefetch()

    def _prefetch(self) -> None:
        try:
            cpu = next(self._iter)
        except StopIteration:
            self._stopped  = True
            self._next_gpu = None
            self._next_cpu = None
            return
        with torch.cuda.stream(self._h2d_stream):
            self._next_gpu = {
                "pixel_values":   cpu["pixel_values"].to(self._device, dtype=self._dtype, non_blocking=True),
                "input_ids":      cpu["input_ids"].to(self._device, non_blocking=True),
                "attention_mask": cpu["attention_mask"].to(self._device, non_blocking=True),
                "labels":         cpu["labels"].to(self._device, non_blocking=True),
            }
        self._next_cpu = cpu

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Dict[str, torch.Tensor], Dict]:
        if self._stopped and self._next_gpu is None:
            raise StopIteration
        torch.cuda.current_stream(self._device).wait_stream(self._h2d_stream)
        gpu_batch, cpu_batch = self._next_gpu, self._next_cpu
        self._prefetch()
        return gpu_batch, cpu_batch   # type: ignore[return-value]


# ============================================================================
# Timing helpers
# ============================================================================

def _configure_timing_debug(args) -> None:
    if not args.debug_timing:
        return
    os.environ["IMITATOR_TIMING_DEBUG"]        = "1"
    os.environ["IMITATOR_TIMING_BATCH_LIMIT"]  = str(args.debug_timing_batch_limit)
    os.environ["IMITATOR_TIMING_SAMPLE_LIMIT"] = str(args.debug_timing_sample_limit)
    os.environ["IMITATOR_TIMING_VIDEO_LIMIT"]  = str(args.debug_timing_sample_limit)
    os.environ["IMITATOR_TIMING_EVERY"]        = str(args.debug_timing_every)
    os.environ["IMITATOR_TIMING_THRESHOLD_MS"] = str(args.debug_timing_threshold_ms)


def _sync_if_needed(enabled: bool, device: torch.device) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


# ============================================================================
# _run_sharded  —  V4 core loop
# ============================================================================

def _run_sharded(
    args,
    dataset,
    vla,
    processor,
    device: torch.device,
    model_dtype: torch.dtype,
    N: int,
    feat_shape: tuple,
    llm_dim: int,
    num_tokens: int,
    shard_size: int,
    num_shards: int,
    pending_shards: List[int],
) -> None:
    t0 = time.time()

    # ── 1. Flatten all pending shard indices → single DataLoader ──────────
    pending_specs:       List[Tuple[int, int, int]] = []
    all_pending_indices: List[int]                   = []
    for si in pending_shards:
        gs  = si * shard_size
        sn  = min(gs + shard_size, N) - gs
        pending_specs.append((si, gs, sn))
        all_pending_indices.extend(range(gs, gs + sn))

    total_pending  = len(all_pending_indices)
    shard_ram_gb   = shard_size * int(np.prod(feat_shape)) * 2 / 1e9
    feat_gb_total  = total_pending * int(np.prod(feat_shape)) * 2 / 1e9
    print(
        f"[Precompute] {len(pending_shards)} pending shard(s), "
        f"{total_pending:,} samples, {feat_gb_total:.1f} GB feat total.\n"
        f"[Precompute] RAM buffer per shard: ~{shard_ram_gb:.1f} GB "
        f"(peak ~{2*shard_ram_gb:.1f} GB with double-buffering)."
    )

    # ── 2. Single DataLoader — workers spawned ONCE for entire job ─────────
    loader = DataLoader(
        Subset(dataset, all_pending_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(openvla_collate_fn, pad_token_id=processor.tokenizer.pad_token_id),
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        multiprocessing_context="forkserver" if args.num_workers > 0 else None,
    )

    # ── 3. Background disk writer + shard pipeline ─────────────────────────
    disk_writer = _AsyncDiskWriter()
    pipeline    = _ShardPipeline(
        cache_dir=args.cache_dir,
        feat_shape=feat_shape,
        pred_horizon=args.pred_horizon,
        action_dim=args.action_dim,
        pool_features=args.pool_features,
        disk_writer=disk_writer,
        pending_specs=pending_specs,
    )

    # ── 4. CUDA prefetcher ─────────────────────────────────────────────────
    use_prefetch = torch.cuda.is_available()
    if use_prefetch:
        iterator = _CudaPrefetcher(loader, device, model_dtype)
    else:
        def _cpu_iter():
            for batch in loader:
                yield None, batch
        iterator = _cpu_iter()

    # ── 5. Main GPU forward loop ───────────────────────────────────────────
    pbar      = tqdm(total=total_pending, desc="Precomputing V4", unit="sample")
    batch_idx = 0

    try:
        with torch.no_grad():
            for gpu_batch, cpu_batch in iterator:
                _sync_if_needed(args.debug_timing, device)
                fwd_t0 = time.perf_counter()

                if use_prefetch:
                    px = gpu_batch["pixel_values"]
                    ii = gpu_batch["input_ids"]
                    am = gpu_batch["attention_mask"]
                    lb = gpu_batch["labels"]
                else:
                    px = cpu_batch["pixel_values"].to(device, dtype=model_dtype, non_blocking=True)
                    ii = cpu_batch["input_ids"].to(device, non_blocking=True)
                    am = cpu_batch["attention_mask"].to(device, non_blocking=True)
                    lb = cpu_batch["labels"].to(device, non_blocking=True)

                bs = px.shape[0]

                with torch.autocast("cuda", dtype=model_dtype, enabled=torch.cuda.is_available()):
                    output = vla(
                        input_ids=ii, attention_mask=am,
                        pixel_values=px, labels=lb,
                        output_hidden_states=False,
                        proprio=None, proprio_projector=None,
                        noisy_actions=None, noisy_action_projector=None,
                        use_film=False, continuous_actions_fast_path=True,
                    )

                _sync_if_needed(args.debug_timing, device)
                fwd_s = time.perf_counter() - fwd_t0

                # D2H + numpy
                extract_t0 = time.perf_counter()
                hidden  = output.action_hidden_states
                feat_gpu = hidden.mean(dim=1).to(torch.float16) if args.pool_features \
                           else hidden.to(torch.float16)
                feat_np    = feat_gpu.cpu().numpy()
                actions_np = cpu_batch["actions"].numpy()
                proprio_np = cpu_batch["proprio"].numpy() if "proprio" in cpu_batch else None
                del output, hidden, feat_gpu
                extract_s = time.perf_counter() - extract_t0

                # RAM write (μs-level) + possible back-pressure at shard boundary
                write_t0 = time.perf_counter()
                pipeline.write_batch(feat_np, actions_np, proprio_np)
                write_s = time.perf_counter() - write_t0

                pbar.update(bs)
                total_s = fwd_s + extract_s + write_s
                if should_emit("batch", batch_idx, total_s):
                    print_timing(
                        "precompute-v4-batch", batch_idx,
                        dict(fwd_s=fwd_s, extract_s=extract_s,
                             write_s=write_s, total_s=total_s),
                        extra=f"bs={bs}",
                    )
                batch_idx += 1

    except Exception as exc:
        pbar.close()
        pipeline.abort()
        disk_writer.shutdown()
        raise RuntimeError(f"[Precompute] Failed at batch {batch_idx}: {exc}") from exc

    pbar.close()
    pipeline.commit_last()
    disk_writer.shutdown()   # wait for the final shard write before writing top meta

    # ── 6. Update top-level meta ──────────────────────────────────────────
    _write_top_meta(
        args.cache_dir, N, feat_shape, args.pred_horizon, args.action_dim,
        has_proprio=(pipeline.has_proprio_global or _any_shard_has_proprio(args.cache_dir, num_shards)),
        pool_features=args.pool_features,
        shard_size=shard_size, num_shards=num_shards,
        completed=True, fmt="sharded",
    )
    elapsed = time.time() - t0
    print(f"\n[Precompute] All shards complete. {N:,} samples in {elapsed/3600:.2f}h → {args.cache_dir}")


# ============================================================================
# Monolithic (legacy) precompute  —  unchanged interface, updated to use RAM buf
# ============================================================================

def _run_monolithic(
    args, dataset, vla, processor,
    device: torch.device, model_dtype: torch.dtype,
    N: int, feat_shape: tuple, llm_dim: int, num_tokens: int,
) -> None:
    print("[Precompute] Monolithic (legacy) mode. No shard resume support.")

    final_feat    = os.path.join(args.cache_dir, "vla_features.bin")
    final_actions = os.path.join(args.cache_dir, "actions.bin")
    final_states  = os.path.join(args.cache_dir, "states.bin")
    meta_path     = os.path.join(args.cache_dir, "vla_cache_meta.json")
    feat_tmp    = final_feat    + ".tmp"
    actions_tmp = final_actions + ".tmp"
    states_tmp  = final_states  + ".tmp"

    for p in (feat_tmp, actions_tmp, states_tmp, meta_path + ".tmp"):
        if os.path.exists(p):
            os.remove(p)

    feat_buf = np.empty((N, *feat_shape),               dtype=np.float16)
    act_buf  = np.empty((N, args.pred_horizon, args.action_dim), dtype=np.float32)
    sta_buf  = np.empty((N, args.action_dim),            dtype=np.float32)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(openvla_collate_fn, pad_token_id=processor.tokenizer.pad_token_id),
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        multiprocessing_context="forkserver" if args.num_workers > 0 else None,
    )

    processed = 0; has_proprio = False
    t0 = time.time()
    pbar = tqdm(total=N, desc="Precomputing (monolithic)", unit="sample")

    with torch.no_grad():
        for batch in loader:
            bs = batch["pixel_values"].shape[0]
            px = batch["pixel_values"].to(device, dtype=model_dtype, non_blocking=True)
            ii = batch["input_ids"].to(device, non_blocking=True)
            am = batch["attention_mask"].to(device, non_blocking=True)
            lb = batch["labels"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=model_dtype, enabled=torch.cuda.is_available()):
                output = vla(
                    input_ids=ii, attention_mask=am, pixel_values=px, labels=lb,
                    output_hidden_states=False, proprio=None, proprio_projector=None,
                    noisy_actions=None, noisy_action_projector=None,
                    use_film=False, continuous_actions_fast_path=True,
                )

            hidden   = output.action_hidden_states
            feat_np  = (hidden.mean(dim=1) if args.pool_features else hidden).to(torch.float16).cpu().numpy()
            act_np   = batch["actions"].numpy()
            prop_np  = batch["proprio"].numpy() if "proprio" in batch else None
            del output, hidden

            feat_buf[processed:processed+bs] = feat_np
            act_buf[processed:processed+bs]  = act_np
            if prop_np is not None:
                sta_buf[processed:processed+bs] = prop_np; has_proprio = True

            processed += bs; pbar.update(bs)

    pbar.close()
    print(f"[Precompute] GPU forward done. Writing {N:,} samples to disk ...")
    t_write = time.time()
    _write_array_slice(feat_buf, N, feat_tmp)
    _write_array_slice(act_buf,  N, actions_tmp)
    if has_proprio:
        _write_array_slice(sta_buf, N, states_tmp)
    print(f"[Precompute] Disk write done in {time.time()-t_write:.1f}s")

    os.replace(feat_tmp, final_feat); os.replace(actions_tmp, final_actions)
    if has_proprio:
        os.replace(states_tmp, final_states)
    elif os.path.exists(states_tmp):
        os.remove(states_tmp)

    meta = dict(
        format="monolithic", total_samples=N, feat_shape=list(feat_shape),
        pool_features=args.pool_features,
        num_tokens=feat_shape[0] if not args.pool_features else 1,
        llm_dim=feat_shape[-1], pred_horizon=args.pred_horizon,
        action_dim=args.action_dim, has_proprio=has_proprio,
        dtype="float16", shard_size=N, num_shards=1, completed=True,
    )
    tmp_meta = os.path.join(args.cache_dir, "vla_cache_meta.json.tmp")
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_meta, meta_path)
    print(f"[Precompute] Done (monolithic). {N:,} samples in {(time.time()-t0)/3600:.2f}h")


# ============================================================================
# Main
# ============================================================================

def main():
    args = _parse_args()
    _configure_timing_debug(args)
    os.makedirs(args.cache_dir, exist_ok=True)

    use_shards  = not args.no_shard
    meta_path   = os.path.join(args.cache_dir, "vla_cache_meta.json")
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[Precompute] Loading VLA from {args.model_name_or_path} ...")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=False, local_files_only=True,
    )
    action_tokenizer = ActionTokenizer(
        processor.tokenizer, bins=args.n_action_bins,
        min_action=args.action_min, max_action=args.action_max,
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path, torch_dtype=model_dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=False, local_files_only=True,
    ).to(device)
    vla.eval()
    for p in vla.parameters():
        p.requires_grad_(False)

    llm_dim    = vla.llm_dim
    num_tokens = ACTION_DIM * NUM_ACTIONS_CHUNK
    feat_shape = (llm_dim,) if args.pool_features else (num_tokens, llm_dim)

    if args.pool_features:
        print(f"[Precompute] Pool mode: ({llm_dim},) instead of ({num_tokens},{llm_dim}). "
              f"Cache ~{num_tokens}× smaller.")
    print(f"[Precompute] VLA ready. llm_dim={llm_dim}, feat_shape={feat_shape}")

    print("[Precompute] Building dataset ...")
    dataset = _build_dataset(args, processor, action_tokenizer)
    N       = len(dataset)
    estimated_gb = N * int(np.prod(feat_shape)) * 2 / 1e9
    print(f"[Precompute] Dataset: {N:,} samples, estimated feat cache: {estimated_gb:.1f} GB")

    if use_shards:
        shard_size = args.shard_size
        num_shards = (N + shard_size - 1) // shard_size

        if os.path.exists(meta_path) and not args.force:
            existing = _read_meta(meta_path)
            if existing.get("format", "monolithic") == "monolithic":
                print("[Precompute] WARNING: existing cache is monolithic. Pass --force to overwrite.")
                return
            if existing.get("completed", False):
                print(f"[Precompute] Cache already complete ({existing['total_samples']:,} samples). "
                      "Use --force to recompute.")
                return

        _write_top_meta(
            args.cache_dir, N, feat_shape, args.pred_horizon, args.action_dim,
            has_proprio=False, pool_features=args.pool_features,
            shard_size=shard_size, num_shards=num_shards,
            completed=False, fmt="sharded",
        )

        pending_shards = []
        for si in range(num_shards):
            if args.force or not _is_shard_complete(args.cache_dir, si):
                pending_shards.append(si)
            else:
                gs = si * shard_size
                sn = min(gs + shard_size, N) - gs
                print(f"[Precompute] Shard {si:05d} [{gs:,}–{gs+sn:,}) already complete — skipping.")

        if not pending_shards:
            print("[Precompute] All shards complete.")
            _write_top_meta(
                args.cache_dir, N, feat_shape, args.pred_horizon, args.action_dim,
                has_proprio=_any_shard_has_proprio(args.cache_dir, num_shards),
                pool_features=args.pool_features,
                shard_size=shard_size, num_shards=num_shards,
                completed=True, fmt="sharded",
            )
            return

        print(f"[Precompute] {len(pending_shards)}/{num_shards} shard(s) to compute.")
        _run_sharded(
            args, dataset, vla, processor, device, model_dtype,
            N, feat_shape, llm_dim, num_tokens,
            shard_size, num_shards, pending_shards,
        )
    else:
        if os.path.exists(meta_path) and not args.force:
            if _read_meta(meta_path).get("completed", False):
                print("[Precompute] Monolithic cache already complete. Use --force.")
                return
        _run_monolithic(
            args, dataset, vla, processor, device, model_dtype,
            N, feat_shape, llm_dim, num_tokens,
        )


# ============================================================================
# Argument parser
# ============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Precompute VLA action_hidden_states cache "
                    "(V4: RAM buffer + async disk writer + single DataLoader + CUDA prefetch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--shard_size", type=int, default=10_000,
                   help="Samples per shard. Each shard held in RAM during forward, "
                        "written to disk asynchronously.")
    p.add_argument("--no_shard", action="store_true", default=False)
    p.add_argument("--pool_features", action="store_true", default=False,
                   help="Mean-pool token dim: (N,llm_dim) vs (N,num_tokens,llm_dim). "
                        "~128x smaller. STRONGLY RECOMMENDED on slow storage.")
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--n_action_bins", type=int, default=256)
    p.add_argument("--action_min", type=float, default=-1.0)
    p.add_argument("--action_max", type=float, default=1.0)
    p.add_argument("--use_lerobot", action="store_true", default=True)
    p.add_argument("--human_root", type=str, default=None)
    p.add_argument("--sim_root", type=str, default=None)
    p.add_argument("--task_mapping_file", type=str, default=None)
    p.add_argument("--human_dataset_file", type=str, default=None)
    p.add_argument("--sim_dataset_file", type=str, default=None)
    p.add_argument("--human_task_description_file", type=str, default=None)
    p.add_argument("--sim_task_description_file", type=str, default=None)
    p.add_argument("--sim_split", type=str, default="train")
    p.add_argument("--human_split", type=str, default=None)
    p.add_argument("--lerobot_camera", type=str, default="zed2i")
    p.add_argument("--lerobot_video_backend", type=str, default="torchcodec")
    p.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--pred_horizon", type=int, default=16)
    p.add_argument("--obs_horizon", type=int, default=1)
    p.add_argument("--action_dim", type=int, default=16)
    p.add_argument("--default_language", type=str, default="pick red cube and place on plate.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--debug_timing", action="store_true", default=False)
    p.add_argument("--debug_timing_batch_limit", type=int, default=20)
    p.add_argument("--debug_timing_sample_limit", type=int, default=20)
    p.add_argument("--debug_timing_every", type=int, default=0)
    p.add_argument("--debug_timing_threshold_ms", type=float, default=0.0)
    # Deprecated — kept so old shell scripts don't break
    p.add_argument("--flush_every", type=int, default=100,
                   help="[Deprecated in V4] No-op. V4 writes to RAM; no periodic flush.")
    p.add_argument("--checkpoint_interval", type=int, default=200_000,
                   help="[Deprecated] Ignored.")
    p.add_argument("--flush_every_v1", dest="_flush_every_v1", type=int, default=5,
                   help="[Deprecated] Ignored.")
    return p.parse_args()


if __name__ == "__main__":
    main()