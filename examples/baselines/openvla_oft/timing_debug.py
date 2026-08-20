from __future__ import annotations

import os
import threading

_thread_state = threading.local()


def is_enabled() -> bool:
    value = os.getenv("IMITATOR_TIMING_DEBUG", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _next_counter(kind: str) -> int:
    counters = getattr(_thread_state, "counters", None)
    if counters is None:
        counters = {}
        _thread_state.counters = counters
    value = counters.get(kind, 0)
    counters[kind] = value + 1
    return value


def _should_log(kind: str, index: int | None, limit_env: str, default_limit: int) -> bool:
    if not is_enabled():
        return False

    limit = _int_env(limit_env, default_limit)
    every = _int_env("IMITATOR_TIMING_EVERY", 0)
    current_index = index if index is not None else _next_counter(kind)

    if current_index < limit:
        return True
    if every > 0 and (current_index + 1) % every == 0:
        return True
    return False


def should_log_batch(index: int | None = None) -> bool:
    return _should_log("batch", index, "IMITATOR_TIMING_BATCH_LIMIT", 20)


def should_log_sample(index: int | None = None) -> bool:
    return _should_log("sample", index, "IMITATOR_TIMING_SAMPLE_LIMIT", 20)


def should_log_video(index: int | None = None) -> bool:
    return _should_log("video", index, "IMITATOR_TIMING_VIDEO_LIMIT", 20)


def over_threshold(total_s: float) -> bool:
    threshold_ms = _float_env("IMITATOR_TIMING_THRESHOLD_MS", 0.0)
    return threshold_ms > 0 and total_s * 1000.0 >= threshold_ms


def should_emit(kind: str, index: int | None, total_s: float) -> bool:
    if not is_enabled():
        return False
    if kind == "batch":
        return should_log_batch(index) or over_threshold(total_s)
    if kind == "video":
        return should_log_video(index) or over_threshold(total_s)
    return should_log_sample(index) or over_threshold(total_s)


def print_timing(component: str, index: int | None, timings: dict[str, float], extra: str = "") -> None:
    if not is_enabled():
        return

    prefix = f"[timing][{component}]"
    if index is not None:
        prefix += f"[{index}]"
    prefix += f"[pid={os.getpid()}]"

    timing_parts = [f"{name}={value * 1000.0:.1f}ms" for name, value in timings.items()]
    if extra:
        timing_parts.append(extra)
    print(prefix, " ".join(timing_parts))
