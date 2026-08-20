"""
Convert Pi05-DROID JAX checkpoint to PyTorch format for fine-tuning.

Usage:
    python convert_pi05_droid_to_pytorch.py --output_path checkpoints/pi05_droid_pytorch
"""

import argparse
import logging
import os

import examples.baselines.pi.src.openpi.shared.download as download
import examples.baselines.pi.src.openpi.training.config as _config
from examples.baselines.pi.examples.convert_jax_model_to_pytorch import convert_pi0_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Convert Pi05-DROID checkpoint to PyTorch")
    parser.add_argument(
        "--output_path",
        type=str,
        default="checkpoints/pi05_droid_pytorch",
        help="Output path for converted PyTorch checkpoint"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Model precision"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    # Download Pi05-DROID checkpoint
    logging.info("Downloading Pi05-DROID checkpoint from GCS...")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
    logging.info(f"Checkpoint downloaded to: {checkpoint_dir}")
    
    # Get model config
    config = _config.get_config("pi05_droid")
    model_config = config.model
    
    logging.info(f"Model config: {model_config}")
    logging.info(f"Converting checkpoint with precision: {args.precision}")
    
    # Convert checkpoint
    convert_pi0_checkpoint(
        checkpoint_dir=str(checkpoint_dir),
        precision=args.precision,
        output_path=args.output_path,
        model_config=model_config
    )
    
    logging.info(f"✅ Conversion complete! PyTorch checkpoint saved to: {args.output_path}")
    logging.info("\nYou can now use this checkpoint for fine-tuning with:")
    logging.info(f"python train_pi_lora_finetune.py --pretrained_model_path {args.output_path}")


if __name__ == "__main__":
    main()