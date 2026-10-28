"""Runtime paths and atomic output; no server-specific paths or source writes."""

from datetime import datetime
from datetime import timezone
import json
import os
from pathlib import Path
import socket
import sys


class Tee:
    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle

    def write(self, value):
        self.stream.write(value)
        self.handle.write(value)
        self.handle.flush()

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        return False


def start_log(root, stage):
    root = Path(root) / "logs"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = root / f"{stage}_{stamp}.log"
    handle = path.open("a", buffering=1)
    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)
    return path


def setup_runtime(cache_dir):
    root = Path(os.environ.get("IG_RUNTIME_CACHE_ROOT", str(Path(cache_dir).resolve() / "runtime")))
    rank = os.environ.get("RANK", os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")).replace(",", "_")
    node = socket.gethostname() + "_" + socket.gethostbyname(socket.gethostname())
    folder = root / f"{node}_rank{rank}"
    folder.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(folder / "hf_datasets")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(folder / "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(folder / "triton"))


def atomic_json_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_torch_save(value, path):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def training_rank_lock(run_dir, rank):
    """Keep mutual exclusion enabled; optionally use a launcher-fenced local lock.

    With a local lock root, the launcher must pin each global rank to one node
    and reject duplicate job launches. Checkpoints and logs stay in run_dir.
    """
    import hashlib
    from filelock import FileLock

    run_dir = Path(run_dir).resolve()
    override = os.environ.get("IG_TRAIN_LOCK_ROOT")
    if override:
        identity = hashlib.sha256(str(run_dir).encode()).hexdigest()[:16]
        folder = Path(override).resolve() / identity
    else:
        folder = run_dir
    folder.mkdir(parents=True, exist_ok=True)
    return FileLock(str(folder / f".rank{rank}.lock"))
