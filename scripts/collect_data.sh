SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/collect_data.py" \
    --demos-dir "demos" \
    --target-episodes 50 \
    --task-timeout 7200 \
    --stall-timeout 300 \
    --max-retries 3 \
    --gpu-ids 0 1 2 3 \
    --max-procs-per-gpu 8 \
    --gpu-mem-threshold 4096
