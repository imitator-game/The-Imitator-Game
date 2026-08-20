#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-}"

case "$TARGET" in
  4090|cuda121|cu121)
    cp "$SCRIPT_DIR/pyproject.cuda121.toml" "$SCRIPT_DIR/pyproject.toml"
    echo "Using pyproject.cuda121.toml for 4090 / CUDA 12.1"
    ;;
  5090|cuda130|cu130)
    cp "$SCRIPT_DIR/pyproject.cuda130.toml" "$SCRIPT_DIR/pyproject.toml"
    echo "Using pyproject.cuda130.toml for 5090 / CUDA 13.0"
    ;;
  *)
    echo "Usage: $0 {4090|5090}" >&2
    exit 2
    ;;
esac
