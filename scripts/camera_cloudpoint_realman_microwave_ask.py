#!/usr/bin/env python3
"""
Camera point cloud → AnyGrasp (RealmanMicrowave)

What this script does
- Creates the ManiSkill env `RealmanMicrowave-v1`.
- Reads a single camera's RGB + Depth + intrinsics from sensor_data.
- Builds a camera-frame point cloud (Open3D) without any segmentation filtering.
- Runs AnyGrasp inference on the resulting point cloud and visualizes predicted grasps.

Notes
- This mirrors scripts/single_camera_cloudpoint_sdk.py, but with no ID segmentation features.
- Depth unit is auto-detected (millimeters vs meters) the same way as in the original.
"""

import argparse
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
import mani_skill  # noqa: F401 (register envs)
import open3d as o3d
import torch
from mani_skill.examples.motionplanning.realman.utils import (
    find_hinge_and_handle_position_for_microwave,
    get_actor_obb,
)


def _choose_camera(obs, preferred: Optional[str] = None) -> str:
    params = obs.get("sensor_param", {})
    if not params:
        raise RuntimeError("No cameras in observation (sensor_param empty)")
    if preferred and preferred in params:
        return preferred
    # fallback to first available
    return list(params.keys())[0]


def _rgbd_to_pointcloud(rgb: np.ndarray, depth_m: np.ndarray, intrinsic_cv: np.ndarray) -> o3d.geometry.PointCloud:
    # Open3D expects uint8 color and float32 depth (meters)
    if rgb is None:
        color_img = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    else:
        if rgb.dtype != np.uint8:
            color_img = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            color_img = rgb
    rgb_o3d = o3d.geometry.Image(color_img)
    depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))

    H, W = depth_m.shape
    fx, fy = float(intrinsic_cv[0, 0]), float(intrinsic_cv[1, 1])
    cx, cy = float(intrinsic_cv[0, 2]), float(intrinsic_cv[1, 2])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1.0,  # meters already
        depth_trunc=10.0,
        convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)


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


def _parse_half_sizes(s: str) -> Tuple[float, float, float]:
    s = s.strip()
    if "," in s:
        parts = [p for p in s.split(",") if p.strip() != ""]
        if len(parts) != 3:
            raise ValueError(f"--handle-lim-xyz expects 3 comma-separated values, got {len(parts)}: '{s}'")
        rx, ry, rz = [float(p) for p in parts]
        return rx, ry, rz
    # single value -> broadcast to all axes
    v = float(s)
    return v, v, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="RealmanMicrowave-v1")
    parser.add_argument("--camera", type=str, default="base_camera_front", help="Camera uid to use; falls back to first if not found")
    parser.add_argument("--shader", type=str, default="default", choices=["default", "rt", "rt-fast"])
    parser.add_argument("--cam-width", type=int, default=512, help="Override camera width")
    parser.add_argument("--cam-height", type=int, default=512, help="Override camera height")
    parser.add_argument("--verify", action="store_true", help="Print simple camera-frame checks (optional)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to AnyGrasp checkpoint (.tar). If omitted, prediction is skipped")
    parser.add_argument("--topk", type=int, default=20, help="Top-K grasps to visualize")
    parser.add_argument("--print-candidates", action="store_true", help="Print numeric pose info for top-K grasps")
    parser.add_argument("--print-topk", type=int, default=10, help="How many candidates to print when --print-candidates is on")
    parser.add_argument("--collision", action="store_true", help="Enable collision filtering in AnyGrasp")
    parser.add_argument("--no-object-mask", action="store_true", help="Disable AnyGrasp internal object masking (apply_object_mask=False)")
    parser.add_argument("--no-viewer", action="store_true", help="Disable Open3D viewer")
    parser.add_argument("--mark-handle", action="store_true", help="Mark the estimated handle position in the point cloud viewer (camera frame)")
    parser.add_argument("--verify-xforms", action="store_true", help="Verify camera/world transform consistency using point cloud roundtrip and matrix comparisons")
    # Limit AnyGrasp workspace around the estimated handle position
    parser.add_argument("--handle-lim", action="store_true", help="Restrict AnyGrasp search space to a 3D box around a chosen anchor (handle or object)")
    parser.add_argument("--lim-anchor", type=str, default="object", choices=["object", "handle"], help="Choose lim anchor: object (movable target) or handle (microwave door handle)")
    parser.add_argument(
        "--handle-lim-xyz",
        type=str,
        default="0.08,0.08,0.10",
        help="Half-sizes (meters) of the lim box in camera frame as 'rx,ry,rz'. Default 8/8/10 cm",
    )
    parser.add_argument(
        "--offset-scale",
        type=float,
        default=1.0,
        help="Scale for hinge->handle offset along door width (1.0=full extent, 0.5=half extent)",
    )
    args = parser.parse_args()

    # Build sensor overrides
    sensor_cfg = dict(shader_pack=args.shader)
    if args.cam_width is not None:
        sensor_cfg["width"] = int(args.cam_width)
    if args.cam_height is not None:
        sensor_cfg["height"] = int(args.cam_height)

    # Create env (sensor_data provides rgb/depth/intrinsics)
    env: gym.Env = gym.make(
        args.env_id,
        obs_mode="sensor_data",
        reward_mode="none",
        render_mode="rgb_array",
        sensor_configs=sensor_cfg,
        human_render_camera_configs=dict(shader_pack=args.shader),
    )

    try:
        obs, _ = env.reset()
        cam = _choose_camera(obs, args.camera)
        # Extract intrinsics and depth/rgb from sensor_data
        params = obs.get("sensor_param", {}).get(cam, {})
        intrinsic_cv_t = params.get("intrinsic_cv")
        if intrinsic_cv_t is None:
            raise RuntimeError("intrinsic_cv missing in sensor_param for selected camera")
        intrinsic_cv = intrinsic_cv_t[0].cpu().numpy() if isinstance(intrinsic_cv_t, torch.Tensor) else np.array(intrinsic_cv_t)
        # extrinsic: prefer world->cam from extrinsic_cv; fallback to inverse of cam2world_gl
        world2cam = None
        extrinsic_cv_t = params.get("extrinsic_cv")
        if extrinsic_cv_t is not None:
            E = extrinsic_cv_t[0].cpu().numpy() if isinstance(extrinsic_cv_t, torch.Tensor) else np.array(extrinsic_cv_t)
            world2cam = np.eye(4, dtype=np.float64)
            world2cam[:3, :4] = E  # [R|t]
        else:
            cam2world_t = params.get("cam2world_gl")
            if cam2world_t is not None:
                C = cam2world_t[0].cpu().numpy() if isinstance(cam2world_t, torch.Tensor) else np.array(cam2world_t)
                try:
                    world2cam_gl = np.linalg.inv(C)
                    # Convert GL camera frame (x right, y up, z backward) to OpenCV camera frame (x right, y down, z forward)
                    S_gl2cv = np.diag([1.0, -1.0, -1.0, 1.0])
                    world2cam = S_gl2cv @ world2cam_gl
                    print("[info] Using cam2world_gl fallback with GL->CV conversion for world2cam")
                except Exception:
                    world2cam = None

        sensor = obs.get("sensor_data", {}).get(cam, {})
        if not sensor:
            raise RuntimeError(f"No sensor_data for camera '{cam}'")

        # Depth: attempt robust key search (any texture that contains 'depth')
        depth = None
        for k in sensor.keys():
            if "depth" in k.lower():
                depth_t = sensor[k][0]
                depth = depth_t.cpu().numpy() if isinstance(depth_t, torch.Tensor) else depth_t
                break
        if depth is None:
            # fallback via position texture (use -Z)
            pos = None
            for k in sensor.keys():
                if "position" in k.lower():
                    pos_t = sensor[k][0]
                    pos = pos_t.cpu().numpy() if isinstance(pos_t, torch.Tensor) else pos_t
                    break
            if pos is not None:
                depth = (-np.array(pos)[..., 2]).astype(np.float32)
            else:
                raise RuntimeError("No depth or position texture found in sensor_data")

        depth = np.array(depth).squeeze()
        depth_m = depth.astype(np.float32)
        # Heuristic: if large values, assume millimeters
        if float(depth_m.max()) > 50.0:
            depth_m *= (1.0 / 1000.0)
        # Valid range
        valid = (depth_m > 0.01) & (depth_m < 10.0)
        depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)

        # RGB (optional)
        # In obs_mode="sensor_data" we receive RAW texture names when apply_texture_transforms=False.
        # That means keys are usually "Color", "Position"/"PositionSegmentation", "Segmentation", etc.
        # Fall back to those if no "rgb"/"albedo" modality keys exist.
        rgb = None
        color_key = None
        # 1) Look for already-transformed modalities (unlikely in sensor_data)
        for k in sensor.keys():
            if ("rgb" in k.lower()) or ("albedo" in k.lower()):
                color_key = k
                break
        # 2) Fallback to raw texture names
        if color_key is None:
            for cand in ["Color", "Albedo"]:
                if cand in sensor:
                    color_key = cand
                    break
        if color_key is not None:
            rgb_t = sensor[color_key][0]
            rgb = rgb_t.cpu().numpy() if isinstance(rgb_t, torch.Tensor) else rgb_t
            # Some shaders return RGBA; use first 3 channels
            if rgb.ndim == 3 and rgb.shape[-1] >= 3:
                rgb = rgb[..., :3]
        # Normalize to uint8 if needed
        if rgb is not None and rgb.dtype != np.uint8:
            # Normalize common float ranges (0..1) or clamp arbitrary floats
            vmax = float(np.max(rgb)) if rgb.size else 1.0
            if vmax <= 1.0 + 1e-6:
                rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # Build camera-frame point cloud (no segmentation, no ROI)
        pcd = _rgbd_to_pointcloud(rgb, depth_m, intrinsic_cv)
        # Basic stats for debugging
        xyz_cam_all = np.asarray(pcd.points).astype(np.float32)
        if xyz_cam_all.size:
            mins = xyz_cam_all.min(axis=0)
            maxs = xyz_cam_all.max(axis=0)
            print(f"[debug] pcd bounds (camera frame): xmin={mins[0]:.3f}, xmax={maxs[0]:.3f}, ymin={mins[1]:.3f}, ymax={maxs[1]:.3f}, zmin={mins[2]:.3f}, zmax={maxs[2]:.3f}")

        # Optional: verify transforms consistency
        if args.verify_xforms:
            def build_world2cam_from_extr(extr):
                if extr is None:
                    return None, None
                E = extr[0].cpu().numpy() if isinstance(extr, torch.Tensor) else np.array(extr)
                W2C = np.eye(4, dtype=np.float64); W2C[:3,:4] = E
                # invert to get C2W (OpenCV)
                R = W2C[:3,:3]; t = W2C[:3,3]
                C2W = np.eye(4, dtype=np.float64)
                C2W[:3,:3] = R.T
                C2W[:3,3]  = -R.T @ t
                return W2C, C2W

            def build_world2cam_from_gl(cam2world_gl):
                if cam2world_gl is None:
                    return None
                C = cam2world_gl[0].cpu().numpy() if isinstance(cam2world_gl, torch.Tensor) else np.array(cam2world_gl)
                W2C_gl = np.linalg.inv(C)
                S_gl2cv = np.diag([1.0, -1.0, -1.0, 1.0])
                return S_gl2cv @ W2C_gl

            W2C_extr, C2W_extr = build_world2cam_from_extr(params.get("extrinsic_cv"))
            W2C_fallback = build_world2cam_from_gl(params.get("cam2world_gl"))

            if W2C_extr is not None and W2C_fallback is not None:
                diff = np.linalg.norm(W2C_extr - W2C_fallback, ord='fro')
                print(f"[verify] ||W2C_extr - W2C_fallback||_F = {diff:.6e}")

            # Print cam2world matrix
            if world2cam is not None:
                cam2world_computed = np.linalg.inv(world2cam)
                print(f"[verify] cam2world matrix:")
                print(cam2world_computed)
                print(f"[verify] Camera X-axis (world): {cam2world_computed[:3, 0]}")
                print(f"[verify] Camera Y-axis (world): {cam2world_computed[:3, 1]}")
                print(f"[verify] Camera Z-axis (world): {cam2world_computed[:3, 2]}")
                print(f"[verify] Camera position (world): {cam2world_computed[:3, 3]}")

            # Roundtrip test using extrinsic when possible
            if (C2W_extr is not None) and (W2C_extr is not None) and xyz_cam_all.size:
                samp = xyz_cam_all[:: max(1, len(xyz_cam_all)//2000)]  # up to ~2000 pts
                ones = np.ones((len(samp), 1), dtype=np.float64)
                cam_h = np.concatenate([samp.astype(np.float64), ones], axis=1).T  # 4xN
                world_h = C2W_extr @ cam_h
                cam_rt = (W2C_extr @ world_h)
                cam_rt = (cam_rt[:3,:] / np.maximum(cam_rt[3:4,:], 1e-12)).T
                err = np.linalg.norm(cam_rt - samp.astype(np.float64), axis=1)
                print(f"[verify] roundtrip (extrinsic): mean={err.mean():.6e}, max={err.max():.6e}, N={len(err)}")
            elif xyz_cam_all.size:
                print("[verify] Skipping roundtrip: missing extrinsic_cv")

        # Optional checks to ensure conventions are consistent
        if args.verify:
            H, W = depth_m.shape
            cx, cy = intrinsic_cv[0, 2], intrinsic_cv[1, 2]
            base = _find_valid_pixel(depth_m, int(round(cx)), int(round(cy)))
            right = _find_valid_pixel(depth_m, int(round(cx)) + 5, int(round(cy)))
            down = _find_valid_pixel(depth_m, int(round(cx)), int(round(cy)) + 5)
            if base and right:
                Pu = _backproject(base[0], base[1], base[2], intrinsic_cv)
                Pr = _backproject(right[0], right[1], right[2], intrinsic_cv)
                print(f"[verify] ΔX (u+): {(Pr[0] - Pu[0]):.6f} (expect > 0 if depths similar)")
            if base and down:
                Pu = _backproject(base[0], base[1], base[2], intrinsic_cv)
                Pd = _backproject(down[0], down[1], down[2], intrinsic_cv)
                print(f"[verify] ΔY (v+): {(Pd[1] - Pu[1]):.6f} (expect > 0 if depths similar)")

        # AnyGrasp inference
        grasp_geoms = []
        best_axis = None
        lims = None
        # Optional: compute a workspace lim box centered at the handle position (in camera frame)
        if args.handle_lim:
            try:
                if world2cam is None:
                    raise RuntimeError("extrinsic_cv missing; cannot compute camera-frame anchor position for lim")
                rx, ry, rz = _parse_half_sizes(args.handle_lim_xyz)

                if args.lim_anchor == "object":
                    # Use movable object's OBB center as lim center (world -> camera)
                    objs = getattr(env.unwrapped, "_objs", None)
                    if not objs:
                        raise RuntimeError("env.unwrapped._objs missing; cannot locate target object for lim")
                    target_obj = objs[0]
                    obb = get_actor_obb(target_obj)
                    center_w = np.append(obb.centroid.astype(np.float64), 1.0)
                    center_c = (world2cam @ center_w)[:3]
                    anchor_name = "object"
                else:
                    # Fallback to handle-centered lim (previous default)
                    info = find_hinge_and_handle_position_for_microwave(env, offset_scale=float(args.offset_scale), vis=False)
                    handle_w = np.append(info["handle_pos"].astype(np.float64), 1.0)
                    center_c = (world2cam @ handle_w)[:3]
                    anchor_name = "handle"

                # Ensure positive depth bounds and a small floor > 0
                zmin = max(0.01, center_c[2] - rz)
                zmax = max(zmin + 1e-3, center_c[2] + rz)
                lims = [
                    float(center_c[0] - rx), float(center_c[0] + rx),
                    float(center_c[1] - ry), float(center_c[1] + ry),
                    float(zmin), float(zmax),
                ]
                print(f"[info] Using {anchor_name}-centered lim (camera frame): {lims}")
                # Count how many points fall inside lim (sanity check)
                if xyz_cam_all.size:
                    xmn,xmx,ymn,ymx,zmn,zmx = lims
                    inside = (xyz_cam_all[:,0] >= xmn) & (xyz_cam_all[:,0] <= xmx) & \
                             (xyz_cam_all[:,1] >= ymn) & (xyz_cam_all[:,1] <= ymx) & \
                             (xyz_cam_all[:,2] >= zmn) & (xyz_cam_all[:,2] <= zmx)
                    n_in = int(inside.sum())
                    print(f"[debug] points inside lim: {n_in}/{len(xyz_cam_all)}")
            except Exception as e:
                print(f"[warn] Failed to compute {args.lim_anchor}-centered lim: {e}")

        # Optional: prepare a visual marker for the handle position in camera frame
        handle_marker = None
        if args.mark_handle:
            try:
                if world2cam is None:
                    raise RuntimeError("extrinsic missing; cannot compute camera-frame handle position for marking")
                info_h = find_hinge_and_handle_position_for_microwave(env, offset_scale=float(args.offset_scale), vis=False)
                handle_w = np.append(info_h["handle_pos"].astype(np.float64), 1.0)
                handle_c = (world2cam @ handle_w)[:3]
                print(f"[info] Handle (camera frame): x={handle_c[0]:.3f}, y={handle_c[1]:.3f}, z={handle_c[2]:.3f}")
                handle_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                handle_marker.paint_uniform_color([0.1, 0.9, 0.1])
                handle_marker.translate(handle_c.astype(float))
            except Exception as e:
                print(f"[warn] Failed to mark handle: {e}")
        if args.checkpoint and Path(args.checkpoint).exists():
            try:
                sdk_dir = "anygrasp_sdk/grasp_detection"
                if sdk_dir not in sys.path:
                    sys.path.insert(0, sdk_dir)
                # graspnet-baseline types used by AnyGrasp wrappers
                baseline_root = "graspnet-baseline"
                if baseline_root not in sys.path:
                    sys.path.insert(0, baseline_root)

                from gsnet import AnyGrasp  # provided by AnyGrasp SDK

                cfgs = SimpleNamespace(
                    checkpoint_path=args.checkpoint,
                    max_gripper_width=0.1,
                    gripper_height=0.03,
                    top_down_grasp=False,
                    debug=False,
                )
                ag = AnyGrasp(cfgs)
                ag.load_net()

                xyz_cam = np.asarray(pcd.points).astype(np.float32)
                colors = np.asarray(pcd.colors).astype(np.float32) if len(pcd.colors) else None

                gg, _cloud = ag.get_grasp(
                    xyz_cam,
                    colors,
                    lims=lims,
                    apply_object_mask=not args.no_object_mask,
                    dense_grasp=False,
                    collision_detection=bool(args.collision),
                )
                # Handle None gracefully
                if gg is None:
                    print("[warn] AnyGrasp returned None (no grasps). Consider relaxing --handle-lim-xyz or using --no-object-mask.")
                elif len(gg) > 0:
                    try:
                        gg = gg.nms().sort_by_score()
                    except Exception:
                        try:
                            gg.sort_by_score()
                        except Exception:
                            pass
                    # Optional: print numeric candidates
                    if args.print_candidates:
                        try:
                            # build cam2world if we have world2cam
                            cam2world_print = None
                            if world2cam is not None:
                                cam2world_print = np.linalg.inv(world2cam)
                            n_print = min(int(args.print_topk), len(gg))
                            for i in range(n_print):
                                c = gg[i]
                                R_cam = c.rotation_matrix
                                t_cam = c.translation
                                s = getattr(c, 'score', None)
                                print(f"[cand {i}] score={s} cam_t={t_cam}")
                                print(f"[cand {i}] cam_R[:,2](approach_cam)={R_cam[:,2]}")
                                if cam2world_print is not None:
                                    # transform position
                                    p_world = cam2world_print[:3,:3] @ t_cam + cam2world_print[:3,3]
                                    # transform rotation
                                    R_world = cam2world_print[:3,:3] @ R_cam
                                    approach_world = R_world[:,2]
                                    print(f"[cand {i}] world_p={p_world}")
                                    print(f"[cand {i}] world_R[:,2](approach_world)={approach_world}")
                        except Exception as e:
                            print(f"[warn] printing candidates failed: {e}")
                    try:
                        grasp_geoms = gg[: max(1, args.topk)].to_open3d_geometry_list()
                    except Exception:
                        grasp_geoms = []
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
                else:
                    print("[warn] AnyGrasp predicted 0 grasps; consider relaxing --handle-lim-xyz or using --no-object-mask")
            except Exception as e:
                print(f"[warn] AnyGrasp inference failed: {e}")
        else:
            print("[info] AnyGrasp disabled: provide --checkpoint to enable grasp prediction")

        if not args.no_viewer:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"{args.env_id} / {cam} (camera-frame pcd + grasps)", width=960, height=720)
            vis.add_geometry(pcd)
            for g in grasp_geoms:
                vis.add_geometry(g)
            if handle_marker is not None:
                vis.add_geometry(handle_marker)
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
