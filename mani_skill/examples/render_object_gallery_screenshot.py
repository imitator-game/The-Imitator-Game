import argparse
from pathlib import Path

import gymnasium as gym
import mani_skill
import numpy as np
from PIL import Image

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a high-quality screenshot of the ObjectGallery scene."
    )
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="ObjectGalleryTwoPanda-v1",
        help="Environment ID to render.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/object_gallery_two_panda_high.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--shader",
        type=str,
        default="rt-fast",
        choices=["minimal", "default", "rt-fast", "rt-med", "rt"],
        help="Shader pack used by the render camera.",
    )
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "gpu"],
        help="Simulation backend. 'cpu' is the safest choice for single-image rendering.",
    )
    parser.add_argument(
        "--render-backend",
        type=str,
        default="gpu",
        help="Renderer device. Use 'cpu' if GPU ray tracing is unstable on your machine.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2560,
        help="Output image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1440,
        help="Output image height in pixels.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=44.0,
        help="Vertical field of view in degrees.",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.01,
        help="Near plane.",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=100.0,
        help="Far plane.",
    )
    parser.add_argument(
        "--eye",
        type=float,
        nargs=3,
        default=[0.0, 2.85, 1.8],
        metavar=("X", "Y", "Z"),
        help="Camera eye position. Default is a centered high angle behind the two robots.",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[0.0, 0.7, 0.38],
        metavar=("X", "Y", "Z"),
        help="Camera look-at target on the table, slightly in front of the robots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment seed.",
    )
    parser.add_argument(
        "--hdri",
        type=str,
        default="autumn",
        help="HDRI preset or file path. ObjectGallery presets include default/autumn/misty/overcast.",
    )
    parser.add_argument(
        "--backdrop",
        action="store_true",
        help="Enable the simple studio wall backdrop behind the stage.",
    )
    parser.add_argument(
        "--backdrop-color",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        help="Backdrop RGB color in 0-1 range. If set, the backdrop is enabled automatically.",
    )
    parser.add_argument(
        "--warmup-renders",
        type=int,
        default=1,
        help="Number of render passes before saving the final frame.",
    )
    return parser.parse_args()


def to_uint8_rgb(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(
                f"Expected a single-environment render, got shape {image.shape}"
            )
        image = image[0]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            scale = 255.0 if image.max() <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def main():
    args = parse_args()
    camera_pose = sapien_utils.look_at(eye=args.eye, target=args.target)
    if args.backdrop_color is not None:
        args.backdrop = True
    if args.shader == "rt" and args.width * args.height >= 3840 * 2160:
        print(
            "Warning: ObjectGalleryTwoPanda at 4K with full ray tracing is very heavy "
            "and may freeze the GPU driver or the IDE. If it is unstable, try "
            "--shader rt-fast or reduce the resolution to 2560x1440."
        )

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="none",
        reward_mode="none",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        enable_shadow=True,
        hdri_background=args.hdri,
        backdrop_enabled=args.backdrop,
        backdrop_color=args.backdrop_color,
        human_render_camera_configs=dict(
            shader_pack=args.shader,
            render_camera=dict(
                pose=camera_pose,
                width=args.width,
                height=args.height,
                fov=np.deg2rad(args.fov_deg),
                near=args.near,
                far=args.far,
            ),
        ),
        viewer_camera_configs=dict(
            shader_pack=args.shader,
            viewer=dict(
                pose=camera_pose,
                width=args.width,
                height=args.height,
                fov=np.deg2rad(args.fov_deg),
                near=args.near,
                far=args.far,
            ),
        ),
    )

    try:
        env.reset(seed=args.seed, options=dict(reconfigure=True))
        image = None
        for _ in range(max(1, args.warmup_renders)):
            image = env.render()
        image = to_uint8_rgb(image)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(args.output)

        print(f"Saved screenshot to: {args.output.resolve()}")
        print(f"env_id={args.env_id}")
        print(f"shader={args.shader}")
        print(f"resolution={args.width}x{args.height}")
        print(f"eye={args.eye}")
        print(f"target={args.target}")
        print(f"fov_deg={args.fov_deg}")
        print(f"hdri={args.hdri}")
        if args.backdrop_color is not None:
            print(f"backdrop_color={args.backdrop_color}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
