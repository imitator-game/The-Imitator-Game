#!/usr/bin/env python3
"""
GraspNet on ManiSkill pointcloud (camera frame) — minimal modes

Two modes only:
- --mode vis : visualize the input point cloud sent to GraspNet (only CubeA points),
               displayed in world frame (Open3D) with Cube markers.
- --mode gui : show the predicted grasp pose in ManiSkill GUI.

Everything else has been removed for simplicity.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import mani_skill  # ensure environments are registered
import numpy as np


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / "graspnet-baseline").exists():
            return parent
    return Path.cwd()


def _load_graspnet_predictor(checkpoint_path: Path):
    # Use the existing predictor wrapper in the repo
    sys.path.append(str(_repo_root()))
    from mani_skill.examples.motionplanning.maniskill_graspnet_pose import (  # type: ignore
        ManiSkillGraspNetPredictor,
    )
    return ManiSkillGraspNetPredictor(str(checkpoint_path))


def _choose_camera(obs) -> str:
    params = obs.get("sensor_param", {})
    cam_names = list(params.keys())
    if not cam_names:
        raise RuntimeError("No sensor_param cameras found in observation")
    return "base_camera" if "base_camera" in cam_names else cam_names[0]


def _get_world_pointcloud(
    env,
    obs,
    only_cubeA: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    pcd = obs["pointcloud"]
    xyzw = pcd["xyzw"][0]  # (M,4)
    xyz = xyzw[:, :3].cpu().numpy()
    valid = (xyzw[:, 3] > 0).cpu().numpy()

    rgb = pcd.get("rgb")
    rgb = rgb[0].cpu().numpy() if rgb is not None else None
    seg = pcd.get("segmentation")
    seg = seg[0].cpu().numpy().reshape(-1) if seg is not None else None

    if only_cubeA:
        # Prefer segmentation if available
        if seg is not None:
            cubeA = env.unwrapped.cubeA
            seg_id = None
            for k, obj in env.unwrapped.segmentation_id_map.items():
                if obj is cubeA:
                    seg_id = k
                    break
            if seg_id is None:
                raise RuntimeError("Cannot resolve segmentation id for cubeA.")
            mask = valid & (seg == seg_id)
        else:
            # Fallback: geometric ROI around cubeA in world frame
            cubeA_w = env.unwrapped.cubeA.pose.p.cpu().numpy()[0]
            dx = 0.20
            dy = 0.20
            z_min = cubeA_w[2] - 0.05
            z_max = cubeA_w[2] + 0.15
            mask = (
                valid
                & (np.abs(xyz[:, 0] - cubeA_w[0]) <= dx)
                & (np.abs(xyz[:, 1] - cubeA_w[1]) <= dy)
                & (xyz[:, 2] >= z_min)
                & (xyz[:, 2] <= z_max)
            )
    else:
        mask = valid

    xyz = xyz[mask]
    if rgb is not None:
        rgb = rgb[mask]
    if seg is not None:
        seg = seg[mask]
    return xyz, rgb, seg


def _cubeA_seg_id(env) -> Optional[int]:
    seg_map = getattr(env.unwrapped, "segmentation_id_map", {})
    try:
        cubeA = env.unwrapped.cubeA
        for k, obj in seg_map.items():
            if obj is cubeA:
                return int(k)
    except Exception:
        pass
    return None


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


def _world2cam_from_extrinsic(intrinsic_cv: np.ndarray, extrinsic_cv_3x4: np.ndarray) -> np.ndarray:
    world2cam = np.eye(4, dtype=np.float64)
    world2cam[:3, :4] = extrinsic_cv_3x4
    return world2cam


def _transform_points(T_4x4: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    ones = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    xyz1 = np.concatenate([xyz, ones], axis=1)
    out = (T_4x4 @ xyz1.T).T
    return out[:, :3]


def _maybe_o3d_vis(
    xyz_world_all: np.ndarray,
    cubes_world: list,
    T_g_world: Optional[np.ndarray],
    xyz_world_input: Optional[np.ndarray] = None,
):
    try:
        import open3d as o3d
    except Exception as e:
        print(f"[warn] Open3D not available for visualization: {e}")
        return
    geoms = []
    # Full world cloud (faint gray)
    cloud_all = o3d.geometry.PointCloud()
    cloud_all.points = o3d.utility.Vector3dVector(xyz_world_all.astype(np.float64))
    if len(xyz_world_all) > 0:
        cloud_all.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.7, 0.7, 0.7]]), (len(xyz_world_all), 1))
        )
    geoms.append(cloud_all)

    # Input subset sent to GraspNet (blue)
    if xyz_world_input is not None and len(xyz_world_input) > 0:
        cloud_in = o3d.geometry.PointCloud()
        cloud_in.points = o3d.utility.Vector3dVector(xyz_world_input.astype(np.float64))
        cloud_in.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([[0.2, 0.4, 1.0]]), (len(xyz_world_input), 1))
        )
        geoms.append(cloud_in)
    for p in cubes_world:
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([1.0, 0.2, 0.2])
        mesh.translate(p.astype(float))
        geoms.append(mesh)
    if T_g_world is not None:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        axes.transform(T_g_world)
        geoms.append(axes)
    try:
        o3d.visualization.draw_geometries(geoms, window_name="PointCloud (World) + Grasp")
    except Exception as e:
        print(f"[warn] Open3D visualization failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="StackCube-v1")
    parser.add_argument("--num-point", type=int, default=20000)
    parser.add_argument("--checkpoint", type=str, default=str(_repo_root() / "graspnet-baseline" / "checkpoint-rs.tar"))
    parser.add_argument("--mode", type=str, choices=["vis", "gui"], default="vis", help="vis: show input cloud; gui: show grasp pose in GUI")
    parser.add_argument("--shader", type=str, default="default", choices=["default", "rt", "rt-fast"]) 
    args = parser.parse_args()

    # Create ManiSkill env with pointcloud obs
    effective_render_mode = ("human" if args.mode == "gui" else "rgb_array")
    env: gym.Env = gym.make(
        args.env_id,
        obs_mode="pointcloud",
        reward_mode="none",
        render_mode=effective_render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
    )

    try:
        obs, _ = env.reset()

        # Choose camera and get parameters
        cam = _choose_camera(obs)
        params = obs["sensor_param"][cam]
        intrinsic_cv = params["intrinsic_cv"][0].cpu().numpy()
        extrinsic_cv = params.get("extrinsic_cv")
        if extrinsic_cv is None:
            raise RuntimeError("extrinsic_cv missing for selected camera; please use a camera that provides extrinsic_cv.")
        world2cam = _world2cam_from_extrinsic(intrinsic_cv, extrinsic_cv[0].cpu().numpy())
        cam2world = np.linalg.inv(world2cam)

        # Minimal logging: selected camera
        # print(f"[info] Selected camera: {cam}")

        # Fetch world-frame point cloud (optionally segmented)
        xyz_world_all, rgb_all, seg_all = _get_world_pointcloud(env, obs, only_cubeA=False)
        # Only CubeA points
        cube_id = _cubeA_seg_id(env)
        if seg_all is not None and cube_id is not None:
            keep_mask = (seg_all == cube_id)
            xyz_world = xyz_world_all[keep_mask]
        else:
            # Fallback: small geometric ROI around CubeA
            cubeA_w = env.unwrapped.cubeA.pose.p.cpu().numpy()[0]
            dx = dy = 0.20
            z_min = cubeA_w[2] - 0.05
            z_max = cubeA_w[2] + 0.15
            keep_mask = (
                (np.abs(xyz_world_all[:, 0] - cubeA_w[0]) <= dx)
                & (np.abs(xyz_world_all[:, 1] - cubeA_w[1]) <= dy)
                & (xyz_world_all[:, 2] >= z_min)
                & (xyz_world_all[:, 2] <= z_max)
            )
            xyz_world = xyz_world_all[keep_mask]

        # Transform world points to camera frame (OpenCV)
        xyz_cam = _transform_points(world2cam, xyz_world).astype(np.float32)
        if xyz_world.shape[0] == 0:
            raise RuntimeError("No CubeA points found in world point cloud.")

        # Basic frustum filter: keep points within (0.01, max_depth]
        # Keep in front of camera
        mask_z = (xyz_cam[:, 2] > 0.01)
        xyz_cam = xyz_cam[mask_z]

        # No ROI or extra sampling logic — minimal

        # Optional: sample to fixed count
        xyz_s, _ = _sample_points(xyz_cam, None, args.num_point)

        # Load predictor and run
        predictor = _load_graspnet_predictor(Path(args.checkpoint))
        gg = predictor.predict_grasps(xyz_s, colors=None, num_points=args.num_point, use_collision_detection=False)
        if len(gg) == 0:
            raise RuntimeError("No valid grasps found")
        gg.sort_by_score()
        best = gg[0]

        # Convert best grasp back to world frame
        R_g_cam = best.rotation_matrix.astype(np.float64)
        t_g_cam = best.translation.astype(np.float64)
        T_g_cam = np.eye(4, dtype=np.float64)
        T_g_cam[:3, :3] = R_g_cam
        T_g_cam[:3, 3] = t_g_cam
        T_g_world = cam2world @ T_g_cam

        # Optionally compute distances in camera for debugging if needed

        # Compute input cloud (in world) for vis mode
        xyz_world_input = _transform_points(cam2world, xyz_s.astype(np.float64))

        if args.mode == "vis":
            cubes_world = []
            try:
                cubes_world = [
                    env.unwrapped.cubeA.pose.p.cpu().numpy()[0],
                    env.unwrapped.cubeB.pose.p.cpu().numpy()[0],
                ]
            except Exception:
                pass
            # Show full world cloud (gray), the actual input cloud (blue), and grasp axes
            _maybe_o3d_vis(xyz_world_all, cubes_world, None, xyz_world_input)

        # No file saving in minimal version

        if args.mode == "gui":
            # Visualize grasp pose using ManiSkill planner GUI
            from mani_skill.examples.motionplanning.panda.motionplanner import (
                PandaArmMotionPlanningSolver,
            )
            import sapien
            from mani_skill.utils.structs.pose import to_sapien_pose

            pose = sapien.Pose(p=T_g_world[:3, 3], q=to_sapien_pose(env.unwrapped.agent.tcp.pose).q)
            # Use predicted rotation instead of TCP's rotation
            from transforms3d.quaternions import mat2quat
            q_wxyz = mat2quat(T_g_world[:3, :3])
            pose = sapien.Pose(p=T_g_world[:3, 3], q=[q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]])

            planner = PandaArmMotionPlanningSolver(
                env,
                debug=True,
                vis=True,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=True,
                print_env_info=False,
            )
            print("[gui] Visualizing grasp pose in ManiSkill GUI (press 'c' to continue)")
            res = planner.move_to_pose_with_screw(pose, dry_run=True)
            if res == -1:
                print("[gui] Planning/visualization failed")
            planner.close()

    finally:
        env.close()


if __name__ == "__main__":
    main()
