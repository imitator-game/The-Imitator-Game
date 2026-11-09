"""Shared credential environment and native-success evaluation entrypoint."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
PACKAGE = WORKSPACE / "The-Imitator-Game/astra_benchmark/paired80_20260922"
PYTHON = "/mnt/sdf/dataset/zhouxunzhe/zhiqi/PolicyGen/.venv/bin/python"
CREDENTIALS = "/mnt/sdf/dataset/zhouxunzhe/hongyun/.config/gpt-baseline/xfast.json"


def child_environment():
    cfg = json.loads(Path(CREDENTIALS).read_text())
    if cfg["base_url"].rstrip("/") != "https://xfast.transcengram.com/v1":
        raise RuntimeError("The selected platform is not the approved xfast endpoint")
    env = os.environ.copy()
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY", "ROLLOUT_AUTH_PROFILE"):
        env.pop(key, None)
    isolated_codex_home = ROOT / "runtime/codex_home"
    # Native Codex may append project trust metadata. Freeze the provider template,
    # then verify effective routing instead of hashing the mutable session config.
    checker = (
        "import pathlib,sys,tomllib; "
        "a=tomllib.loads(pathlib.Path(sys.argv[1]).read_text()); "
        "b=tomllib.loads(pathlib.Path(sys.argv[2]).read_text()); "
        "assert a['model_provider']==b['model_provider']=='xfast'; "
        "assert a['model_providers']['xfast']==b['model_providers']['xfast']"
    )
    subprocess.run([PYTHON, "-c", checker, str(ROOT/'xfast_provider.toml'),
                    str(isolated_codex_home/'config.toml')], check=True)
    env.update(
        XFAST_API_KEY=cfg["api_key"],
        CODEX_HOME=str(isolated_codex_home),
        PYTHONPATH=str(WORKSPACE / "The-Imitator-Game"),
        PYTHONDONTWRITEBYTECODE="1",
        XDG_CACHE_HOME=str(ROOT / "runtime/cache"),
        TMPDIR=str(ROOT / "runtime/tmp"),
        MPLCONFIGDIR=str(ROOT / "runtime/matplotlib"),
        CUDA_CACHE_PATH=str(ROOT / "runtime/cuda_cache"),
        TORCH_HOME=str(ROOT / "runtime/torch_cache"),
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
    )
    for name in ("codex_home", "cache", "tmp", "matplotlib", "cuda_cache", "torch_cache"):
        (ROOT / "runtime" / name).mkdir(parents=True, exist_ok=True)
    return env


def main():
    package=ROOT/'workspace/The-Imitator-Game/astra_benchmark/paired200_20261031_native_success_v4'
    return subprocess.run([sys.executable,'-B','-u',str(package/'budget_runner.py'),*sys.argv[1:]],cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
