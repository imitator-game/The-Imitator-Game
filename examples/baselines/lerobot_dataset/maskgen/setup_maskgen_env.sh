#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS2_ROOT="${GS2_ROOT:-${SCRIPT_DIR}/../../../../../Grounded-SAM-2}"

cd "$SCRIPT_DIR"

if [[ -z "${MASKGEN_CUDA_HOME:-}" && -x /usr/local/cuda-13.0/bin/nvcc ]]; then
  MASKGEN_CUDA_HOME=/usr/local/cuda-13.0
elif [[ -z "${MASKGEN_CUDA_HOME:-}" && -x /usr/local/cuda-12.1/bin/nvcc ]]; then
  MASKGEN_CUDA_HOME=/usr/local/cuda-12.1
fi

if [[ -n "${MASKGEN_CUDA_HOME:-}" ]]; then
  export CUDA_HOME="$MASKGEN_CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

if [[ -n "${MASKGEN_FFMPEG_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$MASKGEN_FFMPEG_LIB_DIR:${LD_LIBRARY_PATH:-}"
elif [[ -n "${CONDA_PREFIX:-}" && -e "${CONDA_PREFIX}/lib/libavutil.so" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

uv sync --extra viz
uv pip install -e "$GS2_ROOT"

uv run python - <<'PY'
import re
import shutil
import subprocess
import sys

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda)
nvcc = shutil.which("nvcc")
print("nvcc", nvcc or "not found")
if nvcc:
    out = subprocess.check_output([nvcc, "--version"], text=True)
    print(out.strip().splitlines()[-1])
    m = re.search(r"release\s+(\d+\.\d+)", out)
    nvcc_cuda = m.group(1) if m else None
    torch_cuda = torch.version.cuda
    if nvcc_cuda and torch_cuda and nvcc_cuda != torch_cuda:
        print(
            "ERROR: nvcc CUDA version does not match torch.version.cuda. "
            "Set MASKGEN_CUDA_HOME to a matching CUDA toolkit, e.g. "
            "MASKGEN_CUDA_HOME=/usr/local/cuda-12.1 bash setup_maskgen_env.sh",
            file=sys.stderr,
        )
        sys.exit(2)
PY

uv run python - "$GS2_ROOT/grounding_dino/setup.py" <<'PY'
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

setup_path = Path(sys.argv[1])
nvcc = shutil.which("nvcc")
if nvcc is None:
    raise SystemExit("nvcc not found; cannot build GroundingDINO CUDA extension")

out = subprocess.check_output([nvcc, "--version"], text=True)
match = re.search(r"release\s+(\d+\.\d+)", out)
cuda_version = float(match.group(1)) if match else 0.0

if cuda_version < 13.0:
    raise SystemExit(0)

text = setup_path.read_text(encoding="utf-8")
marker = "# maskgen CUDA 13 arch patch"
if marker in text:
    raise SystemExit(0)

old = '''        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_86,code=sm_86",
        ]
'''
new = '''        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_120,code=sm_120",  # maskgen CUDA 13 arch patch
        ]
'''
if old not in text:
    raise SystemExit(f"Could not find expected nvcc arch block in {setup_path}")

setup_path.write_text(text.replace(old, new), encoding="utf-8")
print(f"Patched GroundingDINO CUDA arch list for CUDA {cuda_version}: {setup_path}")
PY

uv pip install --no-build-isolation -e "$GS2_ROOT/grounding_dino"

uv run python - "$GS2_ROOT" <<'PY'
import sys
from pathlib import Path

import torch
import supervision

gs2_root = Path(sys.argv[1])
if str(gs2_root) not in sys.path:
    sys.path.insert(0, str(gs2_root))

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict

try:
    import torchcodec
except Exception as exc:
    raise RuntimeError(
        "torchcodec import failed. Install FFmpeg shared libraries and expose "
        "their lib directory, for example: "
        "conda create -y -p /home/ffmpeg-lib -c conda-forge ffmpeg=6; "
        "MASKGEN_FFMPEG_LIB_DIR=/home/ffmpeg-lib/lib "
        "MASKGEN_CUDA_HOME=/usr/local/cuda-12.1 bash setup_maskgen_env.sh"
    ) from exc

print("maskgen deps ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
