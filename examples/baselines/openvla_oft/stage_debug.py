from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import torch

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

_trace_state = threading.local()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_cuda_sync(device: torch.device | None, enabled: bool) -> None:
    if enabled and device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def get_active_step_trace() -> "StageTimingTrace | None":
    return getattr(_trace_state, "trace", None)


@contextlib.contextmanager
def activate_step_trace(trace: "StageTimingTrace | None"):
    previous = getattr(_trace_state, "trace", None)
    _trace_state.trace = trace
    try:
        yield trace
    finally:
        _trace_state.trace = previous


def _init_nvml():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            import pynvml

            pynvml.nvmlInit()
            return pynvml
        except Exception:
            return None


@dataclass
class StageTimingTrace:
    logger: "TrainingStageProfiler"
    route: str
    global_step: int
    current_epoch: int
    batch_size: int
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    stage_resources: dict[str, dict[str, Any]] = field(default_factory=dict)
    step_start_wall: float = field(default_factory=time.perf_counter)
    step_start_snapshot: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.step_start_snapshot = self.logger.snapshot_resources()

    @property
    def device(self) -> torch.device | None:
        return self.logger.device

    @property
    def sync_cuda(self) -> bool:
        return self.logger.sync_cuda

    def set_metadata(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self.metadata[key] = value

    def add_timing(self, name: str, duration_s: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + float(duration_s)

    @contextlib.contextmanager
    def stage(self, name: str, sync_cuda: Optional[bool] = None):
        sync = self.sync_cuda if sync_cuda is None else sync_cuda
        _safe_cuda_sync(self.device, sync)
        start_wall = time.perf_counter()
        start_cpu = self.logger.read_process_cpu_time()
        start_snapshot = self.logger.snapshot_resources()
        try:
            yield
        finally:
            _safe_cuda_sync(self.device, sync)
            end_wall = time.perf_counter()
            end_cpu = self.logger.read_process_cpu_time()
            end_snapshot = self.logger.snapshot_resources()
            duration_s = end_wall - start_wall
            self.add_timing(name, duration_s)

            cpu_process_pct_est = None
            if start_cpu is not None and end_cpu is not None and duration_s > 0:
                cpu_process_pct_est = (end_cpu - start_cpu) / duration_s * 100.0

            self.stage_resources[name] = {
                "duration_s": duration_s,
                "cpu_process_pct_est": cpu_process_pct_est,
                "start": start_snapshot,
                "end": end_snapshot,
            }

    def finalize(self, status: str = "ok") -> None:
        step_end_snapshot = self.logger.snapshot_resources()
        payload = {
            "timestamp_utc": _now_iso(),
            "process_index": self.logger.process_index,
            "pid": os.getpid(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "route": self.route,
            "batch_size": self.batch_size,
            "status": status,
            "total_step_s": time.perf_counter() - self.step_start_wall,
            "timings_s": self.timings,
            "stage_resources": self.stage_resources,
            "step_start": self.step_start_snapshot,
            "step_end": step_end_snapshot,
            "metadata": self.metadata,
        }
        self.logger.write(payload)


class TrainingStageProfiler:
    def __init__(self, output_dir: str, enabled: bool, process_index: int, device: torch.device | None, args) -> None:
        self.enabled = bool(enabled)
        self.output_dir = output_dir
        self.process_index = int(process_index)
        self.device = device
        self.sync_cuda = bool(getattr(args, "debug_stage_timing_sync_cuda", True))
        self.every = max(1, int(getattr(args, "debug_stage_timing_every", 1)))
        self.max_steps = int(getattr(args, "debug_stage_timing_max_steps", 0))
        self.process = psutil.Process(os.getpid()) if psutil is not None else None
        self.nvml = _init_nvml() if self.enabled else None
        self._nvml_handles: dict[int, Any] = {}
        self.file_path: str | None = None
        self._fh = None

        if self.process is not None:
            try:
                self.process.cpu_percent(interval=None)
            except Exception:
                pass
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

        if not self.enabled:
            return

        os.makedirs(output_dir, exist_ok=True)
        self.file_path = os.path.join(output_dir, f"train_stage_timing_rank{self.process_index}.jsonl")
        self._fh = open(self.file_path, "a", buffering=1, encoding="utf-8")

        meta_path = os.path.join(output_dir, f"train_stage_timing_rank{self.process_index}_meta.json")
        if not os.path.exists(meta_path):
            meta = {
                "created_at_utc": _now_iso(),
                "pid": os.getpid(),
                "process_index": self.process_index,
                "device": str(device),
                "sync_cuda": self.sync_cuda,
                "every": self.every,
                "max_steps": self.max_steps,
                "format": "jsonl-v1",
                "args": vars(args),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def should_profile(self, step: int) -> bool:
        if not self.enabled:
            return False
        if self.max_steps > 0 and step >= self.max_steps:
            return False
        return step % self.every == 0

    def create_trace(self, route: str, global_step: int, current_epoch: int, batch_size: int) -> StageTimingTrace | None:
        if not self.should_profile(global_step):
            return None
        return StageTimingTrace(
            logger=self,
            route=route,
            global_step=global_step,
            current_epoch=current_epoch,
            batch_size=batch_size,
        )

    def write(self, payload: dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def read_process_cpu_time(self) -> float | None:
        if self.process is None:
            return None
        try:
            cpu_times = self.process.cpu_times()
            return float(cpu_times.user + cpu_times.system)
        except Exception:
            return None

    def _snapshot_gpu(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        if self.device is None or self.device.type != "cuda" or not torch.cuda.is_available():
            return snapshot

        device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        snapshot["cuda_device_index"] = device_index
        snapshot["torch_cuda_allocated_mb"] = round(torch.cuda.memory_allocated(device_index) / (1024 ** 2), 2)
        snapshot["torch_cuda_reserved_mb"] = round(torch.cuda.memory_reserved(device_index) / (1024 ** 2), 2)
        snapshot["torch_cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated(device_index) / (1024 ** 2), 2)

        if self.nvml is None:
            return snapshot

        try:
            handle = self._nvml_handles.get(device_index)
            if handle is None:
                handle = self.nvml.nvmlDeviceGetHandleByIndex(device_index)
                self._nvml_handles[device_index] = handle

            util = self.nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = self.nvml.nvmlDeviceGetMemoryInfo(handle)
            snapshot["gpu_util_pct"] = util.gpu
            snapshot["gpu_mem_util_pct"] = util.memory
            snapshot["gpu_mem_used_mb"] = round(mem.used / (1024 ** 2), 2)
            snapshot["gpu_mem_total_mb"] = round(mem.total / (1024 ** 2), 2)

            try:
                snapshot["gpu_power_w"] = round(self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 2)
            except Exception:
                pass
            try:
                snapshot["gpu_temp_c"] = self.nvml.nvmlDeviceGetTemperature(
                    handle, self.nvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                pass
        except Exception as exc:
            snapshot["gpu_snapshot_error"] = f"{type(exc).__name__}: {exc}"

        return snapshot

    def snapshot_resources(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"monotonic_s": time.perf_counter()}

        if psutil is not None:
            try:
                snapshot["system_cpu_percent"] = psutil.cpu_percent(interval=None)
            except Exception:
                pass
            try:
                snapshot["system_mem_percent"] = psutil.virtual_memory().percent
            except Exception:
                pass

        if self.process is not None:
            try:
                snapshot["process_cpu_percent"] = self.process.cpu_percent(interval=None)
            except Exception:
                pass
            try:
                snapshot["process_rss_mb"] = round(self.process.memory_info().rss / (1024 ** 2), 2)
            except Exception:
                pass

        snapshot.update(self._snapshot_gpu())
        return snapshot
