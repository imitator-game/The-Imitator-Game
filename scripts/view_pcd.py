#!/usr/bin/env python3
"""
Simple point cloud viewer for PLY/NPZ point clouds (world frame).

Examples
  python scripts/view_pcd.py /tmp/anygrasp_world_pcd_1760111124.ply
  python scripts/view_pcd.py /tmp/anygrasp_world_pcd_1760111124.npz --lims -0.63 -0.33 -0.45 -0.15 0.01 0.17

NPZ format expects keys: 'xyz' (Nx3 float), optional 'colors' (Nx3 float in [0,1]).
"""
import argparse
import os
import sys
from typing import Tuple, Optional

import numpy as np


def parse_lims(vals: Optional[list]) -> Optional[Tuple[float, float, float, float, float, float]]:
    if vals is None:
        return None
    if len(vals) == 1 and "," in vals[0]:
        parts = [p for p in vals[0].split(",") if p.strip()]
        if len(parts) != 6:
            raise ValueError("--lims expects 6 comma-separated values: xmn,xmx,ymn,ymx,zmn,zmx")
        xs = [float(p) for p in parts]
        return tuple(xs)  # type: ignore[return-value]
    if len(vals) != 6:
        raise ValueError("--lims expects 6 values: xmn xmx ymn ymx zmn zmx")
    return tuple(float(x) for x in vals)  # type: ignore[return-value]


def load_pcd(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        cols = None
        if "colors" in data:
            cols = np.asarray(data["colors"], dtype=np.float64)
            if cols.max() > 1.0:
                cols = np.clip(cols / 255.0, 0.0, 1.0)
        return xyz, cols
    try:
        import open3d as o3d
    except Exception as e:
        print(f"[error] open3d is required to read PLY: {e}")
        sys.exit(2)
    pcd = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    cols = None
    if pcd.has_colors():
        cols = np.asarray(pcd.colors, dtype=np.float64)
    return xyz, cols


def build_o3d_cloud(xyz: np.ndarray, cols: Optional[np.ndarray]):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if cols is not None and cols.shape == xyz.shape:
        c = cols.copy()
        if c.max() > 1.0:
            c = np.clip(c / 255.0, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
    return pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Path to PLY or NPZ point cloud")
    ap.add_argument("--lims", nargs="*", help="Optional AABB: xmn xmx ymn ymx zmn zmx or 'xmn,xmx,ymn,ymx,zmn,zmx'")
    ap.add_argument("--voxel-size", type=float, default=0.0, help="Optional voxel size for downsampling (meters)")
    ap.add_argument("--frame", action="store_true", help="Add a world coordinate frame")
    ap.add_argument("--pose", action="append", help="Overlay pose(s) as 'x y z qw qx qy qz' or 'x,y,z,qw,qx,qy,qz'. Can be repeated.")
    ap.add_argument("--frame-size", type=float, default=0.06, help="Size of pose coordinate frames (meters)")
    ap.add_argument("--marker-radius", type=float, default=0.008, help="Radius of spherical pose markers (meters)")
    # AnyGrasp inference on loaded point cloud (world frame)
    ap.add_argument("--anygrasp", action="store_true", help="Run AnyGrasp on this point cloud with optional lims and overlay predictions")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to AnyGrasp checkpoint (.tar). If omitted, use env ANYGRASP_CKPT")
    ap.add_argument("--sdk-dir", type=str, default=None, help="Path to anygrasp_sdk/grasp_detection (optional)")
    ap.add_argument("--baseline-dir", type=str, default=None, help="Path to graspnet-baseline (optional)")
    ap.add_argument("--topk", type=int, default=20, help="Top-K grasps to overlay")
    ap.add_argument("--apply-mask", action="store_true", help="Enable AnyGrasp internal object mask")
    ap.add_argument("--collision", action="store_true", help="Enable AnyGrasp collision filtering")
    args = ap.parse_args()

    xyz, cols = load_pcd(args.path)

    try:
        import open3d as o3d
    except Exception as e:
        print(f"[error] open3d is required for visualization: {e}")
        sys.exit(2)

    pcd = build_o3d_cloud(xyz, cols)
    if args.voxel_size and args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel_size))

    geoms = [pcd]
    lims = parse_lims(args.lims)
    if lims is not None:
        xmn, xmx, ymn, ymx, zmn, zmx = lims
        aabb = o3d.geometry.AxisAlignedBoundingBox([xmn, ymn, zmn], [xmx, ymx, zmx])
        aabb.color = (1.0, 0.0, 0.0)
        geoms.append(aabb)

    if args.frame:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

    # Optional: overlay predicted grasp poses
    if args.pose:
        def parse_pose(s: str):
            s = s.strip()
            parts = [p for p in s.replace(',', ' ').split() if p]
            vals = [float(p) for p in parts]
            if len(vals) == 3:
                x, y, z = vals
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            elif len(vals) == 7:
                x, y, z, qw, qx, qy, qz = vals
            else:
                raise ValueError("--pose expects 3 or 7 values: 'x y z [qw qx qy qz]'")
            return np.array([x, y, z], dtype=np.float64), np.array([qw, qx, qy, qz], dtype=np.float64)

        def quat_to_mat(qw, qx, qy, qz):
            # Normalize
            n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) + 1e-12
            w, x, y, z = qw/n, qx/n, qy/n, qz/n
            # Rotation matrix
            return np.array([
                [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
            ], dtype=np.float64)

        for i, s in enumerate(args.pose):
            try:
                p, q = parse_pose(s)
                R = quat_to_mat(q[0], q[1], q[2], q[3])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = p
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.frame_size))
                frame.transform(T)
                geoms.append(frame)
                # Small sphere marker at the position
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(args.marker_radius))
                sph.paint_uniform_color([0.1, 0.9, 0.1])
                sph.translate(p)
                geoms.append(sph)
            except Exception as e:
                print(f"[warn] failed to parse --pose #{i}: {e}")

    # Optional: run AnyGrasp on this point cloud (assumed world frame)
    if args.anygrasp:
        # Prepare inputs
        xyz = np.asarray(pcd.points, dtype=np.float32)
        cols_np = None
        if pcd.has_colors():
            cols_np = np.asarray(pcd.colors, dtype=np.float32)
        # Setup SDK
        ckpt = args.checkpoint or os.environ.get("ANYGRASP_CKPT", None)
        if ckpt is None or not os.path.exists(ckpt):
            print("[error] AnyGrasp checkpoint missing. Provide --checkpoint or set ANYGRASP_CKPT")
        else:
            if args.sdk_dir is not None or os.environ.get("ANYGRASP_SDK_DIR"):
                sdk_dir = args.sdk_dir or os.environ.get("ANYGRASP_SDK_DIR")
                if sdk_dir and sdk_dir not in sys.path:
                    sys.path.insert(0, sdk_dir)
            if args.baseline_dir is not None or os.environ.get("GRASPNET_BASELINE_DIR"):
                baseline_dir = args.baseline_dir or os.environ.get("GRASPNET_BASELINE_DIR")
                if baseline_dir and baseline_dir not in sys.path:
                    sys.path.insert(0, baseline_dir)
            try:
                from gsnet import AnyGrasp  # type: ignore
                from types import SimpleNamespace
                cfgs = SimpleNamespace(
                    checkpoint_path=ckpt,
                    max_gripper_width=0.1,
                    gripper_height=0.03,
                    top_down_grasp=True,
                    debug=False,
                )
                ag = AnyGrasp(cfgs)
                ag.load_net()
                lims_list = list(lims) if lims is not None else None
                print(f"[AnyGrasp] Running on {len(xyz)} pts; lims={lims_list}; mask={args.apply_mask}; collision={args.collision}")
                gg, _cloud = ag.get_grasp(
                    xyz,
                    cols_np,
                    lims=lims_list,
                    apply_object_mask=bool(args.apply_mask),
                    dense_grasp=False,
                    collision_detection=bool(args.collision),
                )
                if gg is None or len(gg) == 0:
                    print("[AnyGrasp] No grasps returned")
                else:
                    try:
                        gg = gg.nms().sort_by_score()
                    except Exception:
                        try:
                            gg.sort_by_score()
                        except Exception:
                            pass
                    n_use = min(int(args.topk), len(gg))
                    print(f"[AnyGrasp] Visualizing top-{n_use} grasps")
                    for i in range(n_use):
                        g = gg[i]
                        R = g.rotation_matrix
                        t = g.translation
                        # Overlay a coordinate frame at grasp pose
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = t
                        frame_g = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.frame_size))
                        frame_g.transform(T)
                        geoms.append(frame_g)
                        # Spherical marker
                        sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(args.marker_radius))
                        sph.paint_uniform_color([0.1, 0.9, 0.1])
                        sph.translate(t.astype(np.float64))
                        geoms.append(sph)
                        approach = R[:, 2]
                        print(
                            f"  [cand {i}] score={getattr(g,'score',None)} t={t} approach={approach} az={approach[2]:.3f}"
                        )
            except Exception as e:
                print(f"[error] AnyGrasp failed: {e}")

    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
