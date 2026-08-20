"""Convert Pi0 Base checkpoint to PyTorch."""

import argparse
import logging
import examples.baselines.pi.src.openpi.shared.download as download
import examples.baselines.pi.src.openpi.training.config as _config
from examples.baselines.pi.examples.convert_jax_model_to_pytorch import convert_pi0_checkpoint
from examples.baselines.pi.src.openpi.models.pi0_config import Pi0Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="checkpoints/pi0_base_pytorch")
    parser.add_argument("--precision", type=str, default="bfloat16")
    parser.add_argument("--action_dim", type=int, default=8, help="Your robot's action dimension")
    parser.add_argument("--action_horizon", type=int, default=50, help="Action horizon")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Download Pi0 Base checkpoint
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
    logging.info(f"Checkpoint: {checkpoint_dir}")

    # Create custom config with YOUR action_dim
    model_config = Pi0Config(
        dtype=args.precision,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=args.action_dim,  # 🔥 Your dimension!
        action_horizon=args.action_horizon,
        max_token_len=48,
        pi05=False,  # Pi0, not Pi05
    )

    logging.info(f"Converting with action_dim={args.action_dim}")

    convert_pi0_checkpoint(
        checkpoint_dir=str(checkpoint_dir),
        precision=args.precision,
        output_path=args.output_path,
        model_config=model_config
    )

    logging.info(f"✅ Converted to {args.output_path}")


if __name__ == "__main__":
    main()