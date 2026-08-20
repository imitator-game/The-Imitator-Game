#!/usr/bin/env python3
"""
Single-Camera RGBD → Point Cloud (Open3D) for StackCube (AnyGrasp SDK)

- Keeps the same logic as single_camera_cloudpoint.py:
  - Reads monocular RGBD and camera intrinsics from the ManiSkill environment
  - Builds a camera-frame point cloud (Open3D)
  - Supports segmentation filtering, ROI constraints, and optional coordinate-system verification
  - Open3D visualization
- Difference: grasp prediction uses the AnyGrasp SDK (anygrasp_sdk/grasp_detection)
  instead of the GraspNet baseline.
"""

import argparse
import numpy as np
import gymnasium as gym
import mani_skill  # ensure envs are registered
import open3d as o3d
from typing import Tuple, Optional
from pathlib import Path
import torch
from types import SimpleNamespace
import sys
import os


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
    parser.add_argument("--cam-width", type=int, default=256, help="Override camera width (default 256; StackCube default is 128)")
    parser.add_argument("--cam-height", type=int, default=256, help="Override camera height (default 256; StackCube default is 128)")
    parser.add_argument("--verify", action="store_true", help="Run coordinate-system checks to verify camera-frame consistency")
    parser.add_argument("--id-seg", action="store_true", help="Keep only cubeA + table + robot via segmentation id filtering")
    parser.add_argument("--print-seg-ids", action="store_true", help="Print segmentation id -> object name mapping and exit")
    parser.add_argument("--keep-seg-ids", type=str, help="Comma-separated seg ids to keep (e.g., '17,19'); requires sensor_data mode")
    parser.add_argument("--keep-seg-names", type=str, help="Comma-separated Actor/Link names to keep (e.g., 'cubeA,table'); matched against env.unwrapped.segmentation_id_map names")
    parser.add_argument("--debug-seg", action="store_true", help="Debug segmentation: print unique IDs and counts in current frame")
    # ROI around cubeA (pixel-plane disk + depth window)
    parser.add_argument("--roi-cubea", action="store_true", help="Keep only a pixel/distance ROI around projected cubeA center")
    parser.add_argument("--roi-radius-px", type=int, default=80, help="Pixel radius around cubeA projection (default 50)")
    parser.add_argument("--roi-depth", type=float, default=0.2, help="Depth window (meters) around cubeA depth (default 0.15)")
    # AnyGrasp options (--checkpoint is reused to point to the AnyGrasp weights path)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to AnyGrasp checkpoint (.tar). If omitted, prediction is disabled")
    parser.add_argument("--topk", type=int, default=20, help="Number of top grasps to visualize")
    parser.add_argument("--collision", action="store_true", help="Enable model-free collision filtering in AnyGrasp")
    parser.add_argument("--no-viewer", action="store_true", help="Disable Open3D viewer (useful to avoid GL segfaults or for batch runs)")
    args = parser.parse_args()

    # Create env with RGBD observations; render_mode not required for Open3D
    sensor_cfg = dict(shader_pack=args.shader)
    if args.cam_width:
        sensor_cfg["width"] = int(args.cam_width)
    if args.cam_height:
        sensor_cfg["height"] = int(args.cam_height)

    # Need sensor_data when using any segmentation-dependent feature
    obs_mode = (
        "sensor_data"
        if (args.id_seg or args.print_seg_ids or args.keep_seg_ids or args.keep_seg_names)
        else "rgbd"
    )
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
        if args.id_seg or args.print_seg_ids or args.keep_seg_ids or args.keep_seg_names:
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

        # Debug: print visible seg IDs in current frame (before any filtering)
        if args.debug_seg and seg is not None:
            seg_view = seg
            if seg_view.shape != (H, W):
                if seg_view.ndim == 1 and seg_view.size == H * W:
                    seg_view = seg_view.reshape(H, W)
                else:
                    seg_view = seg_view[:H, :W]
            uniq, cnt = np.unique(seg_view, return_counts=True)
            tops = min(50, uniq.size)
            print(f"[debug] seg unique IDs (top {tops}/{uniq.size}):", uniq[:tops].tolist())
            print(f"[debug] seg counts (aligned):", cnt[:tops].tolist())
            # Also try to print visible ID -> name mapping for this run
            try:
                seg_map = getattr(env.unwrapped, "segmentation_id_map", {})
                name_map = {int(k): getattr(v, 'name', str(type(v))) for k, v in seg_map.items()}
                labeled = [(int(u), int(c), name_map.get(int(u), None)) for u, c in zip(uniq, cnt)]
                labeled = [x for x in labeled if x[2] is not None]
                tops = min(50, len(labeled))
                if tops > 0:
                    print("[debug] visible ID->name (top {}/{})".format(tops, len(labeled)))
                    for u, c, n in labeled[:tops]:
                        print(f"  id={u:>4} count={c:>6} name={n}")
            except Exception:
                pass

        # Precedence: if keep_seg_ids is set, use ONLY those IDs (override id_seg); else optionally apply id_seg
        applied_seg_filter = False
        # Arbitrary id keep-list (comma-separated), if provided
        if (args.keep_seg_ids or args.keep_seg_names) and seg is not None:
            ids = []
            if args.keep_seg_ids:
                ids.extend([int(s.strip()) for s in args.keep_seg_ids.split(',') if s.strip().isdigit()])
            if args.keep_seg_names:
                # Map names to IDs for this run
                seg_map = getattr(env.unwrapped, "segmentation_id_map", {})
                reverse_map = {getattr(obj, 'name', None): int(sid) for sid, obj in seg_map.items() if getattr(obj, 'name', None) is not None}
                # Helper: resolve names via exact (case-sensitive), exact (case-insensitive), then unique substring (case-insensitive)
                all_names = list(reverse_map.keys())
                for raw in [s.strip() for s in args.keep_seg_names.split(',') if s.strip()]:
                    name = raw
                    sid_val = None
                    # 1) exact, case-sensitive
                    if name in reverse_map:
                        sid_val = reverse_map[name]
                    else:
                        # 2) exact, case-insensitive
                        lower_map = {n.lower(): reverse_map[n] for n in all_names}
                        if name.lower() in lower_map:
                            sid_val = lower_map[name.lower()]
                        else:
                            # 3) unique substring, case-insensitive
                            cands = [(n, reverse_map[n]) for n in all_names if name.lower() in n.lower()]
                            if len(cands) == 1:
                                sid_val = cands[0][1]
                                print(f"[info] keep-seg-name '{raw}' matched by substring to '{cands[0][0]}' (id={sid_val})")
                            elif len(cands) > 1:
                                print(f"[warn] keep-seg-name '{raw}' ambiguous; matches: {[n for n,_ in cands]}")
                            else:
                                print(f"[warn] keep-seg-name '{raw}' not found; available (sample): {sorted(all_names)[:20]}")
                    if sid_val is not None:
                        ids.append(int(sid_val))
            if ids:
                if seg.shape != (H, W):
                    if seg.ndim == 1 and seg.size == H * W:
                        seg = seg.reshape(H, W)
                    else:
                        seg = seg[:H, :W]
                mask = np.isin(seg, ids)
                applied_seg_filter = True
                # Warn if none of the requested IDs are visible
                if args.debug_seg:
                    print(f"[debug] keep-seg requested (IDs after name->id mapping): {ids}; kept pixels: {int(mask.sum())}")
                if mask.sum() == 0:
                    print("[warn] None of the requested keep-seg-ids appear in the current camera view; point cloud may be empty.")

        # Segmentation id filtering (cubeA + table + robot), only if not overridden by keep-seg-ids
        if (not applied_seg_filter) and args.id_seg:
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
                if args.debug_seg:
                    print(f"[debug] id-seg kept pixels: {int(mask.sum())}")
        # Note: ROI around cubeA can conflict with keep-seg-ids when targeting other objects.
        # If keep-seg-ids is set, skip ROI to avoid inadvertently zeroing the target object pixels.
        if args.roi_cubea and applied_seg_filter:
            print("[info] ROI around cubeA is disabled because --keep-seg-ids is set (to avoid empty masks).")
            roi_allowed = False
        else:
            roi_allowed = True

        # Pixel-plane ROI around projected cubeA center (does not change coordinates)
        if args.roi_cubea and roi_allowed:
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
                        if args.debug_seg:
                            print(f"[debug] roi_cubea kept pixels: {int(mask.sum())}")

        # Apply the unified mask once (only generate point cloud for masked region)
        depth_m = np.where(mask, depth_m, 0.0)

        # Reconstruct camera-frame point cloud
        pcd = _rgbd_to_pointcloud(rgb, depth_m, intrinsic_cv)

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

        # ------------------------------------------------------------------
        # AnyGrasp SDK prediction (replacing GraspNet baseline)
        # ------------------------------------------------------------------
        grasp_geoms = []
        best_axis = None
        if args.checkpoint and Path(args.checkpoint).exists():
            try:
                # Ensure AnyGrasp SDK and graspnetAPI are importable
                sdk_dir = "anygrasp_sdk/grasp_detection"
                if sdk_dir not in sys.path:
                    sys.path.insert(0, sdk_dir)
                baseline_root = "graspnet-baseline"
                if baseline_root not in sys.path:
                    sys.path.insert(0, baseline_root)

                from gsnet import AnyGrasp  # provided by AnyGrasp SDK (gsnet.so)

                # Build minimal cfgs compatible with AnyGrasp constructor
                cfgs = SimpleNamespace(
                    checkpoint_path=args.checkpoint,
                    max_gripper_width=0.1,  # meters, AnyGrasp expects <= 0.1
                    gripper_height=0.03,
                    top_down_grasp=False,
                    debug=False,
                )

                ag = AnyGrasp(cfgs)
                ag.load_net()

                xyz_cam = np.asarray(pcd.points).astype(np.float32)
                # Open3D stores colors in [0,1]
                colors = np.asarray(pcd.colors).astype(np.float32) if len(pcd.colors) else None

                gg, cloud = ag.get_grasp(
                    xyz_cam,
                    colors,
                    lims=None,  # logic kept unchanged: already filtered via seg/ROI, no extra cropping
                    apply_object_mask=True,
                    dense_grasp=False,
                    collision_detection=bool(args.collision),
                )

                if len(gg) == 0:
                    print("[warn] AnyGrasp predicted 0 grasps; showing point cloud only")
                else:
                    try:
                        gg = gg.nms().sort_by_score()
                    except Exception:
                        # fall back to sorting only
                        try:
                            gg.sort_by_score()
                        except Exception:
                            pass
                    try:
                        grasp_geoms = gg[: max(1, args.topk)].to_open3d_geometry_list()
                    except Exception:
                        grasp_geoms = []

                    # Visualize the best grasp's coordinate frame
                    try:
                        best = gg[0]
                        R = best.rotation_matrix
                        t = best.translation
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = t
                        best_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
                        best_axis.transform(T)
                    except Exception:
                        best_axis = None
            except Exception as e:
                print(f"[warn] AnyGrasp inference failed: {e}")
        else:
            print("[info] AnyGrasp disabled: provide --checkpoint to enable grasp prediction")

        if not args.no_viewer:
            # Viewer aligned with camera coordinate frame
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
