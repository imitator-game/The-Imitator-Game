#!/usr/bin/env python3
"""
Single-Camera RGBD → Point Cloud (Open3D) for StackCube

- Uses the default single camera from StackCube-v1, which is defined as:
  CameraConfig(
      uid="base_camera",
      pose=look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]),
      width=128, height=128, fov=pi/2, near=0.01, far=100
  )
  (See mani_skill/envs/tasks/tabletop/stack_cube.py)

- Reads RGBD + intrinsics for that camera, reconstructs point cloud in the
  OpenCV camera frame using Open3D, and visualizes it.

- Minimal: no multi-camera fusion, no GraspNet.
"""

import argparse
import numpy as np
import gymnasium as gym
import mani_skill  # ensure envs are registered
import open3d as o3d
from typing import Tuple, Optional
from pathlib import Path
import torch
import torch


def _choose_camera(obs, preferred="base_camera") -> str:
    cams = list(obs.get("sensor_param", {}).keys())
    if not cams:
        raise RuntimeError("No cameras in observation (sensor_param empty)")
    return preferred if preferred in cams else cams[0]


def _rgbd_to_pointcloud(rgb, depth_m, intrinsic_cv):
    # Open3D expects uint8 color and float32 depth (meters)
    if rgb.dtype != np.uint8:
        color_img = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        color_img = rgb
    rgb_o3d = o3d.geometry.Image(color_img)
    depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1.0,  # depth is already meters
        depth_trunc=10.0,
        convert_rgb_to_intensity=False,
    )

    H, W = depth_m.shape
    fx, fy = float(intrinsic_cv[0, 0]), float(intrinsic_cv[1, 1])
    cx, cy = float(intrinsic_cv[0, 2]), float(intrinsic_cv[1, 2])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return pcd


def _find_valid_pixel(depth_m: np.ndarray, u0: int, v0: int, max_radius: int = 5) -> Optional[Tuple[int, int, float]]:
    H, W = depth_m.shape
    for r in range(max_radius + 1):
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                u = int(np.clip(u0 + du, 0, W - 1))
                v = int(np.clip(v0 + dv, 0, H - 1))
                z = float(depth_m[v, u])
                if z > 0.01:
                    return u, v, z
    return None


def _backproject(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    return np.array([X, Y, z], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="StackCube-v1")
    parser.add_argument("--camera", type=str, default="base_camera")
    parser.add_argument("--shader", type=str, default="default", choices=["default", "rt", "rt-fast"])
    parser.add_argument("--cam-width", type=int, default=512, help="Override camera width (default 256; StackCube default is 128)")
    parser.add_argument("--cam-height", type=int, default=512, help="Override camera height (default 256; StackCube default is 128)")
    parser.add_argument("--verify", action="store_true", help="Run coordinate-system checks to verify camera-frame consistency")
    parser.add_argument("--id-seg", action="store_true", help="Keep only cubeA + table + robot via segmentation id filtering")
    parser.add_argument("--print-seg-ids", action="store_true", help="Print segmentation id -> object name mapping and exit")
    parser.add_argument("--keep-seg-ids", type=str, help="Comma-separated seg ids to keep (e.g., '17,19'); requires sensor_data mode")
    # ROI around cubeA (pixel-plane disk + depth window)
    parser.add_argument("--roi-cubea", action="store_true", help="Keep only a pixel/distance ROI around projected cubeA center")
    parser.add_argument("--roi-radius-px", type=int, default=50, help="Pixel radius around cubeA projection (default 50)")
    parser.add_argument("--roi-depth", type=float, default=1, help="Depth window (meters) around cubeA depth (default 0.15)")
    # GraspNet options
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to GraspNet checkpoint .tar; if omitted, GraspNet is disabled")
    parser.add_argument("--topk", type=int, default=20, help="Number of top grasps to visualize")
    parser.add_argument("--collision", action="store_true", help="Enable model-free collision filtering")
    args = parser.parse_args()

    # Create env with RGBD observations; render_mode not required for Open3D
    sensor_cfg = dict(shader_pack=args.shader)
    if args.cam_width:
        sensor_cfg["width"] = int(args.cam_width)
    if args.cam_height:
        sensor_cfg["height"] = int(args.cam_height)

    obs_mode = "sensor_data" if (args.id_seg or args.print_seg_ids or args.keep_seg_ids) else "rgbd"
    env = gym.make(
        args.env_id,
        obs_mode=obs_mode,
        render_mode="rgb_array",
        sensor_configs=sensor_cfg,
        human_render_camera_configs=dict(shader_pack=args.shader),
    )

    try:
        obs, _ = env.reset(seed=42)

        cam = _choose_camera(obs, preferred=args.camera)
        params = obs["sensor_param"][cam]
        intrinsic_cv = params["intrinsic_cv"][0].cpu().numpy()

        # Fetch RGBD for the chosen camera (robust to key names)
        sensor = obs["sensor_data"][cam]
        # rgb (robust key search)
        rgb_key = None
        for k in sensor.keys():
            kl = k.lower()
            if "rgb" in kl or "color" in kl:
                rgb_key = k
                break
        if rgb_key is None:
            raise KeyError("No RGB key found in sensor_data for camera")
        rgb_t = sensor[rgb_key][0]
        rgb = (rgb_t[..., :3].cpu().numpy() if isinstance(rgb_t, torch.Tensor) else rgb_t[..., :3])
        # depth (robust key search, fallback to position)
        depth_key = None
        for k in sensor.keys():
            if "depth" in k.lower():
                depth_key = k
                break
        if depth_key is not None:
            depth_t = sensor[depth_key][0]
            depth = (depth_t.cpu().numpy() if isinstance(depth_t, torch.Tensor) else depth_t).squeeze(-1)
        else:
            pos_key = None
            for k in sensor.keys():
                if "position" in k.lower():
                    pos_key = k
                    break
            if pos_key is None:
                raise KeyError("No depth or position key found in sensor_data for camera")
            pos_t = sensor[pos_key][0]
            pos = pos_t.cpu().numpy() if isinstance(pos_t, torch.Tensor) else pos_t
            depth = (-pos[..., 2]).astype(np.float32)

        # Depth unit heuristic: if values are large, assume millimeters
        depth_m = depth.astype(np.float32)
        if float(depth_m.max()) > 50.0:
            depth_m *= (1.0 / 1000.0)
        # Keep reasonable range
        valid_depth = (depth_m > 0.01) & (depth_m < 10.0)
        depth_m = np.where(valid_depth, depth_m, 0.0).astype(np.float32)

        # Build a unified HxW mask (boolean) to select pixels for point cloud generation
        H, W = depth_m.shape
        mask = np.ones((H, W), dtype=bool)

        # Handle segmentation-driven behaviors (print map / id-based filtering)
        # decode segmentation when needed
        seg = None
        if args.id_seg or args.print_seg_ids or args.keep_seg_ids:
            for k in sensor.keys():
                if "segmentation" in k.lower():
                    seg_t = sensor[k][0]
                    seg_np = seg_t.cpu().numpy() if isinstance(seg_t, torch.Tensor) else seg_t
                    if seg_np.ndim == 3 and seg_np.shape[-1] == 4:
                        if seg_np.dtype == np.uint8:
                            seg = (
                                seg_np[..., 0].astype(np.uint32)
                                | (seg_np[..., 1].astype(np.uint32) << 8)
                                | (seg_np[..., 2].astype(np.uint32) << 16)
                                | (seg_np[..., 3].astype(np.uint32) << 24)
                            )
                        else:
                            seg = seg_np[..., 0].astype(np.uint32)
                    else:
                        seg = seg_np.astype(np.uint32).squeeze()
                    break
            # Print id map and exit if requested
            if args.print_seg_ids:
                from mani_skill.utils.structs import Actor, Link
                print("ID to Actor/Link name mappings")
                print("0: Background")
                for obj_id, obj in sorted(env.unwrapped.segmentation_id_map.items()):
                    if isinstance(obj, Actor):
                        print(f"{obj_id}: Actor, name - {getattr(obj, 'name', '')}")
                    elif isinstance(obj, Link):
                        print(f"{obj_id}: Link, name - {getattr(obj, 'name', '')}")
                    else:
                        print(f"{obj_id}: {type(obj)}")
                return

        # Segmentation id filtering (cubeA + table + robot), if requested
        if args.id_seg:
            if seg is not None:
                if seg.shape != (H, W):
                    if seg.ndim == 1 and seg.size == H * W:
                        seg = seg.reshape(H, W)
                    else:
                        seg = seg[:H, :W]
                keep_ids = set()
                exclude_ids = set()
                seg_map = getattr(env.unwrapped, "segmentation_id_map", {})
                try:
                    robot_links = set(env.unwrapped.agent.robot.get_links())
                except Exception:
                    robot_links = set()
                table_obj = getattr(getattr(env.unwrapped, "table_scene", object()), "table", None)
                ground_obj = getattr(getattr(env.unwrapped, "table_scene", object()), "ground", None)
                cubeA_obj = getattr(env.unwrapped, "cubeA", None)
                for sid, obj in seg_map.items():
                    ik = int(sid)
                    if obj in robot_links:
                        keep_ids.add(ik)
                    if (table_obj is not None) and (obj is table_obj):
                        keep_ids.add(ik)
                    if (cubeA_obj is not None) and (obj is cubeA_obj):
                        keep_ids.add(ik)
                    if (ground_obj is not None) and (obj is ground_obj):
                        exclude_ids.add(ik)
                if keep_ids:
                    mask_keep = np.isin(seg, list(keep_ids))
                    if exclude_ids:
                        mask_excl = np.isin(seg, list(exclude_ids))
                        mask &= (mask_keep & (~mask_excl))
                    else:
                        mask &= mask_keep

        # Arbitrary id keep-list (comma-separated), if provided
        if args.keep_seg_ids and seg is not None:
            ids = [int(s.strip()) for s in args.keep_seg_ids.split(',') if s.strip().isdigit()]
            if ids:
                if seg.shape != (H, W):
                    if seg.ndim == 1 and seg.size == H * W:
                        seg = seg.reshape(H, W)
                    else:
                        seg = seg[:H, :W]
                mask &= np.isin(seg, ids)

        # Pixel-plane ROI around projected cubeA center (does not change coordinates)
        if args.roi_cubea:
            extrinsic_cv = params.get("extrinsic_cv")
            if extrinsic_cv is not None:
                world2cam = np.eye(4, dtype=np.float64)
                world2cam[:3, :4] = extrinsic_cv[0].cpu().numpy()
                cubeA = getattr(env.unwrapped, "cubeA", None)
                if cubeA is not None:
                    Pw = np.append(cubeA.pose.p.cpu().numpy()[0], 1.0)
                    Pc = (world2cam @ Pw)[:3]
                    if Pc[2] > 0:
                        fx, fy = float(intrinsic_cv[0, 0]), float(intrinsic_cv[1, 1])
                        cx, cy = float(intrinsic_cv[0, 2]), float(intrinsic_cv[1, 2])
                        u0 = int(round(fx * Pc[0] / Pc[2] + cx))
                        v0 = int(round(fy * Pc[1] / Pc[2] + cy))
                        rr = max(1, int(args.roi_radius_px))
                        xs = np.arange(W)[None, :]
                        ys = np.arange(H)[:, None]
                        disk = (xs - u0) ** 2 + (ys - v0) ** 2 <= rr * rr
                        z0 = float(Pc[2])
                        depth_band = (depth_m > 0.0) & (np.abs(depth_m - z0) <= float(args.roi_depth))
                        roi_mask = disk & depth_band
                        mask &= roi_mask

        # Apply the unified mask once (only generate point cloud for masked region)
        depth_m = np.where(mask, depth_m, 0.0)

        # No segmentation filtering: keep all pixels

        # Reconstruct camera-frame point cloud
        pcd = _rgbd_to_pointcloud(rgb, depth_m, intrinsic_cv)

        # No world-Z filtering; show full camera-frame cloud

        print(f"Camera: {cam}")
        print(f"RGB shape: {rgb.shape}, depth shape: {depth.shape}")
        print(f"Points: {np.asarray(pcd.points).shape[0]}")

        if args.verify:
            # 1) Compare manual back-projection vs Open3D point cloud (nearest neighbor)
            pts = np.asarray(pcd.points)
            if pts.shape[0] == 0:
                raise RuntimeError("Empty point cloud; cannot verify.")
            kdtree = o3d.geometry.KDTreeFlann(pcd)

            H, W = depth_m.shape
            cx, cy = float(intrinsic_cv[0, 2]), float(intrinsic_cv[1, 2])
            samples = []
            for du, dv in [(0, 0), (10, 0), (0, 10)]:
                res = _find_valid_pixel(depth_m, int(round(cx)) + du, int(round(cy)) + dv, max_radius=5)
                if res is None:
                    continue
                u, v, z = res
                Pm = _backproject(u, v, z, intrinsic_cv)
                _, idx, dists = kdtree.search_knn_vector_3d(Pm, 1)
                nn = pts[idx[0]]
                err = float(np.linalg.norm(nn - Pm))
                samples.append((u, v, z, err, Pm, nn))
            if samples:
                print("[verify] manual back-projection vs Open3D NN errors (m):")
                for (u, v, z, err, Pm, nn) in samples:
                    print(f"  pixel=({u},{v}) z={z:.3f} err={err:.6f}")
            else:
                print("[verify] could not find valid sample pixels near center")

            # 2) Axis sanity: X increases with u, Y increases with v
            base = _find_valid_pixel(depth_m, int(round(cx)), int(round(cy)))
            right = _find_valid_pixel(depth_m, int(round(cx)) + 5, int(round(cy)))
            down = _find_valid_pixel(depth_m, int(round(cx)), int(round(cy)) + 5)
            if base and right:
                Pu = _backproject(base[0], base[1], base[2], intrinsic_cv)
                Pr = _backproject(right[0], right[1], right[2], intrinsic_cv)
                print(f"[verify] ΔX (u+): {(Pr[0] - Pu[0]):.6f} (should be > 0 if depths similar)")
            if base and down:
                Pu = _backproject(base[0], base[1], base[2], intrinsic_cv)
                Pd = _backproject(down[0], down[1], down[2], intrinsic_cv)
                print(f"[verify] ΔY (v+): {(Pd[1] - Pu[1]):.6f} (should be > 0 if depths similar)")

        # Run GraspNet on camera-frame point cloud (optional)
        grasp_geoms = []
        best_axis = None
        if args.checkpoint and Path(args.checkpoint).exists():
            try:
                from mani_skill.examples.motionplanning.maniskill_graspnet_pose import (
                    ManiSkillGraspNetPredictor,
                )
                predictor = ManiSkillGraspNetPredictor(args.checkpoint)
                xyz_cam = np.asarray(pcd.points).astype(np.float32)
                colors = None
                gg = predictor.predict_grasps(xyz_cam, colors, use_collision_detection=args.collision)
                if len(gg) > 0:
                    gg.sort_by_score()
                    # Open3D geometries of top-K grasps (in camera coordinates)
                    try:
                        grasp_geoms = gg[: max(1, args.topk)].to_open3d_geometry_list()
                    except Exception:
                        grasp_geoms = []
                    # Best grasp axis frame
                    best = gg[0]
                    R = best.rotation_matrix
                    t = best.translation
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t
                    best_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
                    best_axis.transform(T)
                else:
                    print("[warn] GraspNet predicted 0 grasps; showing point cloud only")
            except Exception as e:
                print(f"[warn] GraspNet inference failed: {e}")
        else:
            print("[info] GraspNet disabled: provide --checkpoint to enable grasp prediction")

        # Use a viewer aligned with the camera coordinate frame:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"{args.env_id} / {cam} (camera-frame pcd + grasps)", width=960, height=720)
        vis.add_geometry(pcd)
        for g in grasp_geoms:
            vis.add_geometry(g)
        if best_axis is not None:
            vis.add_geometry(best_axis)
        vis.poll_events()
        vis.update_renderer()
        ctr = vis.get_view_control()
        bbox = pcd.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        ctr.set_lookat(center)
        ctr.set_front([0.0, 0.0, 1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.7)
        vis.run()
        vis.destroy_window()
    finally:
        env.close()


if __name__ == "__main__":
    main()
