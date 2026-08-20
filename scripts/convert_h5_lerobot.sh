python -m examples.baselines.lerobot_dataset.h5_to_lerobot \
  --input demos \
  --output-dir imitator_data \
  --recursive \
  --no-gpu \
  --n-jobs 128 \
  --mem-per-proc 4.0 \
  --fps 30