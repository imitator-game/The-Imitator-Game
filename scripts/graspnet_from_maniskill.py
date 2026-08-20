#!/usr/bin/env python3
"""
Run GraspNet inference directly on ManiSkill point clouds without writing RGBD files.

Requirements:
- This script assumes the repository contains `graspnet-baseline/` with a valid checkpoint,
  e.g., `graspnet-baseline/checkpoint-rs.tar`.

Usage example:
  python scripts/graspnet_from_maniskill.py \
    --env-id StackCube-v1 \
    --num-point 20000 \
    --checkpoint graspnet-baseline/checkpoint-rs.tar \
    --render-mode rgb_array --shader default \
    --only-cubeA --collision --top-k 20 --o3d-vis
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "graspnet-baseline").exists():
            return parent
    return Path.cwd()


def _load_graspnet(checkpoint_path: Path):
    root = _repo_root() / "graspnet-baseline"
    sys.path.append(str(root / "models"))
    sys.path.append(str(root / "utils"))
    from graspnet import GraspNet, pred_decode  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    ).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, pred_decode, device


def _get_pointcloud(
    env,
    only_cubeA: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (xyz, rgb, seg) in world coordinates (meters)."""
    obs, _ = env.reset()
    pcd = obs["pointcloud"]
    xyzw = pcd["xyzw"][0]  # (M,4)
    xyz = xyzw[:, :3].cpu().numpy()
    valid = (xyzw[:, 3] > 0).cpu().numpy()

    rgb = pcd.get("rgb")
    rgb = rgb[0].cpu().numpy() if rgb is not None else None
    seg = pcd.get("segmentation")
    seg = seg[0].cpu().numpy().reshape(-1) if seg is not None else None

    if only_cubeA:
        # find per-scene seg id for cubeA
        cubeA = env.unwrapped.cubeA
        seg_id = None
        for k, obj in env.unwrapped.segmentation_id_map.items():
            if obj is cubeA:
                seg_id = k
                break
        if seg_id is None or seg is None:
            raise RuntimeError("Cannot resolve segmentation id for cubeA or segmentation missing in pointcloud.")
        mask = valid & (seg == seg_id)
    else:
        mask = valid

    xyz = xyz[mask]
    if rgb is not None:
        rgb = rgb[mask]
    if seg is not None:
        seg = seg[mask]
    return xyz, rgb, seg


def _sample_points(xyz: np.ndarray, rgb: Optional[np.ndarray], num_point: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n = xyz.shape[0]
    if n == 0:
        raise RuntimeError("Point cloud is empty after filtering.")
    if n >= num_point:
        idx = np.random.choice(n, num_point, replace=False)
    else:
        idx1 = np.arange(n)
        idx2 = np.random.choice(n, num_point - n, replace=True)
        idx = np.concatenate([idx1, idx2], axis=0)
    xyz_sampled = xyz[idx].astype(np.float32)
    rgb_sampled = rgb[idx] if rgb is not None else None
    return xyz_sampled, rgb_sampled


def _predict_grasps(net, pred_decode, device, xyz: np.ndarray):
    end_points = {"point_clouds": torch.from_numpy(xyz[None, ...]).to(device)}
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    return grasp_preds[0].detach().cpu().numpy()


def _maybe_collision_filter(gg_array: np.ndarray, xyz: np.ndarray, enable: bool):
    if not enable:
        return gg_array
    try:
        root = _repo_root() / "graspnet-baseline"
        sys.path.append(str(root / "utils"))
        from collision_detector import ModelFreeCollisionDetector  # type: ignore
        from graspnetAPI import GraspGroup  # type: ignore

        gg = GraspGroup(gg_array)
        mfcdetector = ModelFreeCollisionDetector(xyz, voxel_size=0.01)
        mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
        return gg[~mask].grasp_group_array
    except Exception as e:
        print(f"[warn] Collision filtering skipped: {e}")
        return gg_array


def _print_topk(gg_array: np.ndarray, k: int):
    k = min(k, gg_array.shape[0])
    print(f"Total grasps: {gg_array.shape[0]}, showing top-{k}")
    # Sort by score descending (col 0)
    order = np.argsort(gg_array[:, 0])[::-1][:k]
    for i, idx in enumerate(order):
        g = gg_array[idx]
        score, width, height, depth = g[0], g[1], g[2], g[3]
        R = g[4:13].reshape(3, 3)
        t = g[13:16]
        print(f"#{i+1}: score={score:.3f}, width={width:.3f}m, depth={depth:.3f}m, center={t}, R=\n{R}")


def _maybe_o3d_vis(gg_array: np.ndarray, xyz: np.ndarray, k: int):
    try:
        import open3d as o3d  # type: ignore
        from graspnetAPI import GraspGroup  # type: ignore
    except Exception as e:
        print(f"[warn] Open3D/graspnetAPI not available for visualization: {e}")
        return
    gg = GraspGroup(gg_array)
    gg.sort_by_score()
    gg = gg[:k]
    geoms = gg.to_open3d_geometry_list()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    try:
        o3d.visualization.draw_geometries([cloud, *geoms])
    except Exception as e:
        print(f"[warn] Open3D visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="StackCube-v1")
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--checkpoint", type=str, default=str(_repo_root() / "graspnet-baseline" / "checkpoint-rs.tar"))
    parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["rgb_array", "human"]) 
    parser.add_argument("--shader", type=str, default="default", choices=["default", "rt", "rt-fast"]) 
    parser.add_argument("--vis", action="store_true", help="Open viewer (when render-mode=human)")
    parser.add_argument("--only-cubeA", action="store_true", help="Use only cubeA points via segmentation")
    parser.add_argument("--collision", action="store_true", help="Enable model-free collision filtering")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--o3d-vis", action="store_true", help="Visualize grasps with Open3D")
    args = parser.parse_args()

    # Create ManiSkill env with pointcloud obs
    env: gym.Env = gym.make(
        args.env_id,
        obs_mode="pointcloud",
        reward_mode="none",
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
    )

    try:
        xyz, rgb, seg = _get_pointcloud(env, only_cubeA=args.only_cubeA)
        xyz_s, _ = _sample_points(xyz, rgb, args.num_point)

        net, pred_decode, device = _load_graspnet(Path(args.checkpoint))
        gg_array = _predict_grasps(net, pred_decode, device, xyz_s)
        gg_array = _maybe_collision_filter(gg_array, xyz_s, enable=args.collision)

        _print_topk(gg_array, args.top_k)
        if args.o3d_vis:
            _maybe_o3d_vis(gg_array, xyz_s, args.top_k)
    finally:
        env.close()


if __name__ == "__main__":
    main()

