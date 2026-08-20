"""
gpu_manager.py - smart GPU resource manager

Features:
  - Queries per-GPU VRAM usage in real time via nvidia-smi
  - Only allocates a GPU when its free memory exceeds the threshold
  - Supports independent max-process count per GPU
  - Supports a whitelist (only use the specified GPUs)
  - Supports auto-discovery of all GPUs

Configuration examples (JSON / argparse):
  Global defaults:
    --gpu-mem-threshold 4096   # at least 4 GB free before a GPU is usable
    --max-procs-per-gpu 2      # default max 2 processes per GPU

  Fine-grained control (--gpu-config):
    "0:max_procs=3,mem_threshold=6144"   GPU0 max 3 procs, threshold 6 GB
    "1:max_procs=1,mem_threshold=8192"   GPU1 max 1 proc, threshold 8 GB
    "2:max_procs=2"                      GPU2 uses the global threshold
    # GPUs not listed -> not used

  Only use specified GPUs:
    --gpu-ids 0 1 3             only use GPU 0, 1, 3 (2 is not used)
"""

from __future__ import annotations

import subprocess
import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════
#  VRAM query (via nvidia-smi, no extra dependencies)
# ═══════════════════════════════════════════════════════

def _query_nvidia_smi() -> Dict[int, Tuple[int, int]]:
    """
    Returns { gpu_id: (used_MiB, total_MiB) }
    Returns {} if nvidia-smi is unavailable
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode()
        result = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                idx, used, total = int(parts[0]), int(parts[1]), int(parts[2])
                result[idx] = (used, total)
        return result
    except Exception:
        return {}


def get_gpu_free_memory_mib(gpu_id: int) -> Optional[int]:
    """Return the free VRAM of the given GPU in MiB, or None on query failure"""
    info = _query_nvidia_smi()
    if gpu_id not in info:
        return None
    used, total = info[gpu_id]
    return total - used


def get_all_gpus() -> List[int]:
    """Return the ids of all GPUs on the system"""
    info = _query_nvidia_smi()
    return sorted(info.keys())


# ═══════════════════════════════════════════════════════
#  Per-GPU config
# ═══════════════════════════════════════════════════════

@dataclass
class GPUConfig:
    gpu_id: int
    max_procs: int          # max concurrent processes on this GPU
    mem_threshold_mib: int  # minimum free VRAM required before this GPU can be allocated


def parse_gpu_configs(
    gpu_ids: Optional[List[int]],
    gpu_config_strs: Optional[List[str]],
    default_max_procs: int,
    default_mem_threshold_mib: int,
) -> Dict[int, GPUConfig]:
    """
    Parse command-line args into {gpu_id: GPUConfig}.

    gpu_config_strs format (one entry per GPU):
        "0:max_procs=3,mem_threshold=6144"
        "1:max_procs=1"
        "2:mem_threshold=8192"
        "3"                        <- just declares this GPU is used, all defaults
    """
    # If no explicit config is given, auto-discover all GPUs
    available = get_all_gpus()
    if not available:
        # No nvidia-smi: use fake data (convenient for debugging in non-GPU environments)
        available = list(range(4))

    # Determine the set of GPUs to use
    if gpu_ids:
        selected = [g for g in gpu_ids if g in available]
    elif gpu_config_strs:
        selected = []   # decided by gpu_config_strs
    else:
        selected = available  # default to all

    configs: Dict[int, GPUConfig] = {}

    # Fill selected with defaults first
    for gid in selected:
        configs[gid] = GPUConfig(
            gpu_id=gid,
            max_procs=default_max_procs,
            mem_threshold_mib=default_mem_threshold_mib,
        )

    # Then override / append with gpu_config_strs
    if gpu_config_strs:
        for spec in gpu_config_strs:
            spec = spec.strip()
            # Parse "ID:key=val,key=val"
            m = re.match(r"^(\d+)(?::(.+))?$", spec)
            if not m:
                raise ValueError(f"Cannot parse --gpu-config entry: '{spec}'  "
                                 f"expected format 'ID[:max_procs=N][,mem_threshold=M]'")
            gid = int(m.group(1))
            if gid not in available:
                raise ValueError(f"GPU {gid} does not exist (system detected: {available})")

            # Base values
            cfg = configs.get(gid, GPUConfig(
                gpu_id=gid,
                max_procs=default_max_procs,
                mem_threshold_mib=default_mem_threshold_mib,
            ))

            if m.group(2):
                for kv in m.group(2).split(","):
                    kv = kv.strip()
                    if "=" not in kv:
                        continue
                    k, v = kv.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k == "max_procs":
                        cfg.max_procs = int(v)
                    elif k in ("mem_threshold", "mem_threshold_mib"):
                        cfg.mem_threshold_mib = int(v)
                    else:
                        raise ValueError(f"Unknown parameter '{k}', valid values: max_procs, mem_threshold")

            configs[gid] = cfg

    if not configs:
        raise RuntimeError("No usable GPU config; check --gpu-ids or --gpu-config args")

    return configs


# ═══════════════════════════════════════════════════════
#  GPU resource manager
# ═══════════════════════════════════════════════════════

@dataclass
class _GPUSlot:
    config: GPUConfig
    active_procs: int = 0


class GPUManager:
    """
    Thread-safe GPU resource manager.

    acquire() logic:
      1. Iterate over all configured GPUs
      2. Skip: active_procs >= max_procs
      3. Skip: real-time free VRAM < mem_threshold_mib
      4. Pick the GPU with the lowest active_procs / max_procs ratio (least loaded)
      5. Atomically increment and return the gpu_id
    """

    # VRAM query cache (avoid calling nvidia-smi on every acquire)
    _CACHE_TTL = 5.0   # seconds

    def __init__(self, configs: Dict[int, GPUConfig]):
        self._slots: Dict[int, _GPUSlot] = {
            gid: _GPUSlot(config=cfg) for gid, cfg in configs.items()
        }
        self._lock = Lock()
        self._mem_cache: Dict[int, Tuple[int, float]] = {}  # {gpu_id: (free_mib, timestamp)}

    # ── VRAM query (cached) ──────────────────────────────

    def _free_mem(self, gpu_id: int) -> int:
        """Return the GPU's free VRAM in MiB. Not re-queried within the cache TTL."""
        now = time.time()
        if gpu_id in self._mem_cache:
            cached_mib, ts = self._mem_cache[gpu_id]
            if now - ts < self._CACHE_TTL:
                return cached_mib

        free = get_gpu_free_memory_mib(gpu_id)
        if free is None:
            # nvidia-smi unavailable: allow (do not restrict VRAM)
            free = 999_999
        self._mem_cache[gpu_id] = (free, now)
        return free

    def _invalidate_cache(self, gpu_id: int):
        self._mem_cache.pop(gpu_id, None)

    # ── Public interface ─────────────────────────────────

    def acquire(self) -> Optional[int]:
        """
        Acquire a GPU slot.
        Returns the gpu_id, or None (no GPU available right now).
        """
        with self._lock:
            candidates = []
            for gid, slot in self._slots.items():
                if slot.active_procs >= slot.config.max_procs:
                    continue
                free = self._free_mem(gid)
                if free < slot.config.mem_threshold_mib:
                    continue
                load_ratio = slot.active_procs / slot.config.max_procs
                candidates.append((load_ratio, free, gid))

            if not candidates:
                return None

            # Prefer the lowest load ratio; on ties pick the one with the most free VRAM
            candidates.sort(key=lambda x: (x[0], -x[1]))
            best_gid = candidates[0][2]
            self._slots[best_gid].active_procs += 1
            self._invalidate_cache(best_gid)
            return best_gid

    def release(self, gpu_id: int):
        with self._lock:
            if gpu_id in self._slots:
                self._slots[gpu_id].active_procs = max(
                    0, self._slots[gpu_id].active_procs - 1
                )
            self._invalidate_cache(gpu_id)

    def status(self) -> List[dict]:
        """Return a real-time status list for every GPU (for printing/logging)"""
        rows = []
        with self._lock:
            for gid, slot in sorted(self._slots.items()):
                free = self._free_mem(gid)
                rows.append({
                    "gpu_id": gid,
                    "active_procs": slot.active_procs,
                    "max_procs": slot.config.max_procs,
                    "free_mem_mib": free,
                    "threshold_mib": slot.config.mem_threshold_mib,
                    "mem_ok": free >= slot.config.mem_threshold_mib,
                    "slots_ok": slot.active_procs < slot.config.max_procs,
                    "available": (free >= slot.config.mem_threshold_mib
                                  and slot.active_procs < slot.config.max_procs),
                })
        return rows

    def status_str(self) -> str:
        rows = self.status()
        parts = []
        for r in rows:
            mem_gb = r["free_mem_mib"] / 1024
            flag = "✓" if r["available"] else ("M" if not r["mem_ok"] else "P")
            parts.append(
                f"GPU{r['gpu_id']}[{flag}]"
                f"{r['active_procs']}/{r['max_procs']}proc "
                f"{mem_gb:.1f}GB↑{r['threshold_mib']//1024}GB"
            )
        return "  ".join(parts)

    @property
    def total_capacity(self) -> int:
        """Total max concurrent processes across all GPUs"""
        return sum(s.config.max_procs for s in self._slots.values())


# ═══════════════════════════════════════════════════════
#  argparse helpers (for use by the main script)
# ═══════════════════════════════════════════════════════

def add_gpu_args(parser):
    """Inject GPU-related arguments into the ArgumentParser"""
    g = parser.add_argument_group("GPU configuration")
    g.add_argument(
        "--gpu-ids", type=int, nargs="+", default=None,
        metavar="ID",
        help="Specify which GPUs to use (e.g. --gpu-ids 0 1 3). "
             "If unspecified, all detected GPUs are used. "
             "When used together with --gpu-config, --gpu-config takes precedence.",
    )
    g.add_argument(
        "--gpu-config", type=str, nargs="+", default=None,
        metavar="SPEC",
        help=(
            "Fine-grained per-GPU config, format: 'ID[:max_procs=N][,mem_threshold=M]'.\n"
            "Examples:\n"
            "  --gpu-config 0:max_procs=3,mem_threshold=6144 1:max_procs=1 2\n"
            "  => GPU0 max 3 procs/needs 6GB free, GPU1 max 1 proc, GPU2 uses defaults\n"
            "  Only the GPUs listed are used (anything not listed = not used)."
        ),
    )
    g.add_argument(
        "--max-procs-per-gpu", type=int, default=2,
        metavar="N",
        help="Default max concurrent processes per GPU (overridable via --gpu-config).",
    )
    g.add_argument(
        "--gpu-mem-threshold", type=int, default=4096,
        metavar="MiB",
        help="Default memory threshold (MiB): a GPU is only allocated when free VRAM >= this value. "
             "Default 4096 (4 GB). Overridable via --gpu-config.",
    )


def build_gpu_manager(args) -> GPUManager:
    """Build a GPUManager from an argparse Namespace"""
    configs = parse_gpu_configs(
        gpu_ids=args.gpu_ids,
        gpu_config_strs=args.gpu_config,
        default_max_procs=args.max_procs_per_gpu,
        default_mem_threshold_mib=args.gpu_mem_threshold,
    )
    mgr = GPUManager(configs)

    # Print config on startup
    print("\n[GPU Manager] config loaded:")
    print(f"  {'GPU':>4}  {'Max procs':>6}  {'Mem threshold':>8}  {'Free now':>8}  {'Status'}")
    print("  " + "-" * 48)
    for row in mgr.status():
        gid = row["gpu_id"]
        cfg = configs[gid]
        avail_str = "OK available" if row["available"] else (
            "X insufficient VRAM" if not row["mem_ok"] else "X all procs used"
        )
        print(f"  GPU{gid:>1}  {cfg.max_procs:>6}  "
              f"{cfg.mem_threshold_mib:>6} MiB  "
              f"{row['free_mem_mib']:>6} MiB  {avail_str}")
    print()

    return mgr