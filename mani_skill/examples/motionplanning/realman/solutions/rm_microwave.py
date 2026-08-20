import os
import sys
from types import SimpleNamespace
import numpy as np
import sapien
from transforms3d.euler import euler2quat
import trimesh
import torch
try:
    import open3d as o3d
except Exception:
    o3d = None
import time

from mani_skill.envs.tasks.tabletop.rm_tasks.rm_microwave import RealmanMicrowaveEnv
from mani_skill.examples.motionplanning.realman.motionplanner import \
    RealmanArmMotionPlanningSolver
from mani_skill.examples.motionplanning.realman.utils import (
    compute_grasp_info_by_obb, get_actor_obb, find_hinge_and_handle_position_for_microwave, get_link_obb)
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
import sapien.physx as physx


def _to_np(x):
    try:
        # torch.Tensor or sapien tensor-like
        return x.cpu().numpy()
    except Exception:
        return np.array(x, dtype=float)


def _make_reset_pose_near_base(env, planner: RealmanArmMotionPlanningSolver, arm_side: str = "right",
                               distance: float = 0.30, z_clearance: float = 0.18) -> sapien.Pose:
    """Generate a safe reset pose directly in front of the arm base.

    - Direction: horizontal direction from the base to the current TCP
    - Distance: distance (meters)
    - Height: no lower than the current TCP height, and at least z_clearance
    """
    base_link_name = "r_base_link" if arm_side == "right" else "l_base_link"
    base_link = env.agent.robot.links_map[base_link_name]
    base_pos = _to_np(base_link.pose.p[0])

    tcp_pose = planner.get_current_tcp_pose()
    # Position (remove batch dimension, convert to np.float32)
    try:
        tcp_pos = _to_np(tcp_pose.p[0])
    except Exception:
        tcp_pos = _to_np(tcp_pose.p)

    dir_xy = tcp_pos - base_pos
    dir_xy[2] = 0.0
    dir_norm = np.linalg.norm(dir_xy) + 1e-12
    dir_xy = dir_xy / dir_norm

    target_p = base_pos + dir_xy * float(distance)
    target_p[2] = max(tcp_pos[2], float(z_clearance))

    # Orientation (remove batch dimension, convert to np.float32, shape=(4,))
    q_raw = tcp_pose.q
    try:
        q_np = q_raw.cpu().numpy()
    except Exception:
        q_np = np.array(q_raw)
    if q_np.ndim == 2 and q_np.shape[0] == 1:
        q_np = q_np[0]
    q_np = q_np.astype(np.float32).reshape(4,)

    p_np = np.asarray(target_p, dtype=np.float32)

    return sapien.Pose(p=p_np, q=q_np)


def _rodrigues(u: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation formula: axis-angle (u, theta) -> rotation matrix (3x3)."""
    u = np.asarray(u, dtype=float)
    un = u / (np.linalg.norm(u) + 1e-12)
    ux, uy, uz = un
    K = np.array([[0.0, -uz,  uy],
                  [ uz,  0.0, -ux],
                  [-uy,  ux,  0.0]], dtype=float)
    I = np.eye(3, dtype=float)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _choose_camera_from_obs(obs: dict, preferred: str = None) -> str:
    params = obs.get("sensor_param", {}) if isinstance(obs, dict) else {}
    if not params:
        return preferred or ""
    if preferred and preferred in params:
        return preferred
    return list(params.keys())[0]


def _rgbd_to_pointcloud_o3d(rgb: np.ndarray, depth_m: np.ndarray, intrinsic_cv: np.ndarray):
    if o3d is None:
        raise RuntimeError("open3d not available")
    if rgb is None:
        color_img = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    else:
        color_img = rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    rgb_o3d = o3d.geometry.Image(color_img)
    depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
    H, W = depth_m.shape
    fx, fy = float(intrinsic_cv[0, 0]), float(intrinsic_cv[1, 1])
    cx, cy = float(intrinsic_cv[0, 2]), float(intrinsic_cv[1, 2])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)


def _parse_half_sizes_csv(s: str, default=(0.08, 0.08, 0.10)):
    try:
        if not s:
            return default
        parts = [p for p in str(s).split(',') if p.strip()]
        if len(parts) == 1:
            v = float(parts[0])
            return (v, v, v)
        if len(parts) == 3:
            return tuple(float(p) for p in parts)
    except Exception:
        pass
    return default


def _fetch_camera_frame(env, preferred_cam: str = "base_camera_back"):
    """Try to fetch one camera frame (sensor_data) and calibration matrices.

    Returns dict or None on failure:
      - cam_id, K (3x3), world2cam (4x4), cam2world (4x4), depth_m (HxW), rgb (HxWx3, uint8), pcd (o3d PointCloud)
    """
    # Try common APIs to fetch observation
    obs = None
    try:
        if hasattr(env, 'get_obs'):
            obs = env.get_obs()
        elif hasattr(env.unwrapped, 'get_obs'):
            obs = env.unwrapped.get_obs()
    except Exception:
        obs = None
    if not isinstance(obs, dict):
        return None

    cam = _choose_camera_from_obs(obs, preferred_cam)
    params = obs.get("sensor_param", {}).get(cam, {})
    sensor = obs.get("sensor_data", {}).get(cam, {})
    if not params or not sensor:
        return None

    # Intrinsics
    intrinsic_cv_t = params.get("intrinsic_cv")
    if intrinsic_cv_t is None:
        return None
    K = intrinsic_cv_t[0].cpu().numpy() if isinstance(intrinsic_cv_t, torch.Tensor) else np.array(intrinsic_cv_t)

    # Extrinsics: prefer extrinsic_cv, fallback cam2world_gl
    world2cam = None
    cam2world = None
    extrinsic_cv_t = params.get("extrinsic_cv")
    if extrinsic_cv_t is not None:
        E = extrinsic_cv_t[0].cpu().numpy() if isinstance(extrinsic_cv_t, torch.Tensor) else np.array(extrinsic_cv_t)
        world2cam = np.eye(4, dtype=np.float64)
        world2cam[:3, :4] = E
        # invert to get C2W (OpenCV)
        R = world2cam[:3, :3]
        t = world2cam[:3, 3]
        cam2world = np.eye(4, dtype=np.float64)
        cam2world[:3, :3] = R.T
        cam2world[:3, 3] = -R.T @ t
    else:
        cam2world_t = params.get("cam2world_gl")
        if cam2world_t is not None:
            C = cam2world_t[0].cpu().numpy() if isinstance(cam2world_t, torch.Tensor) else np.array(cam2world_t)
            try:
                W2C_gl = np.linalg.inv(C)
                S_gl2cv = np.diag([1.0, -1.0, -1.0, 1.0])
                world2cam = S_gl2cv @ W2C_gl
                cam2world = np.linalg.inv(world2cam)
            except Exception:
                world2cam = None
                cam2world = None
    if world2cam is None or cam2world is None:
        return None

    # Depth (try any key containing 'depth'; else use -Z from position)
    depth = None
    for k in list(sensor.keys()):
        if 'depth' in k.lower():
            depth_t = sensor[k][0]
            depth = depth_t.cpu().numpy() if isinstance(depth_t, torch.Tensor) else depth_t
            break
    if depth is None:
        pos = None
        for k in list(sensor.keys()):
            if 'position' in k.lower():
                pos_t = sensor[k][0]
                pos = pos_t.cpu().numpy() if isinstance(pos_t, torch.Tensor) else pos_t
                break
        if pos is not None:
            depth = (-np.array(pos)[..., 2]).astype(np.float32)
    if depth is None:
        return None
    depth_m = np.array(depth).squeeze().astype(np.float32)
    if float(depth_m.max()) > 50.0:
        depth_m *= (1.0 / 1000.0)
    valid = (depth_m > 0.01) & (depth_m < 10.0)
    depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)

    # RGB (optional)
    rgb = None
    color_key = None
    for k in list(sensor.keys()):
        kl = k.lower()
        if ('rgb' in kl) or ('albedo' in kl):
            color_key = k
            break
    if color_key is None:
        for cand in ["Color", "Albedo"]:
            if cand in sensor:
                color_key = cand
                break
    if color_key is not None:
        rgb_t = sensor[color_key][0]
        rgb = rgb_t.cpu().numpy() if isinstance(rgb_t, torch.Tensor) else rgb_t
        if rgb.ndim == 3 and rgb.shape[-1] >= 3:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            vmax = float(np.max(rgb)) if rgb.size else 1.0
            if vmax <= 1.0 + 1e-6:
                rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    try:
        pcd = _rgbd_to_pointcloud_o3d(rgb, depth_m, K)
    except Exception:
        return None

    return dict(cam_id=cam, K=K, world2cam=world2cam, cam2world=cam2world, depth_m=depth_m, rgb=rgb, pcd=pcd)


def solveRealmanMicrowave(env: RealmanMicrowaveEnv, seed=None, debug=False, vis=False):
    """
    Complete the microwave task using the right arm:
    1. The right arm opens the door
    2. After releasing the door, the right arm grasps the object and places it in the microwave
    """
    env.reset(seed=seed)

    # Get the microwave and door links
    microwave = env.unwrapped._microwaves[0]
    door_link = microwave.links[2]  # The 2nd link is the door

    # Create motion planners for both arms
    left_planner = RealmanArmMotionPlanningSolver(
        env,
        arm_side="left",
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    right_planner = RealmanArmMotionPlanningSolver(
        env,
        arm_side="right",
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # Get base positions of both arms (used to compute the grasp orientation)
    left_base_link = env.agent.robot.links_map["l_base_link"]
    right_base_link = env.agent.robot.links_map["r_base_link"]
    left_base_pos = _to_np(left_base_link.pose.p[0])
    right_base_pos = _to_np(right_base_link.pose.p[0])

    # -------------------------------------------------------------------------- #
    # Precompute hinge info (compute only once, reuse later)
    # -------------------------------------------------------------------------- #
    hinge_info = None
    try:
        hinge_info = find_hinge_and_handle_position_for_microwave(env, offset_scale=1.0, vis=False)
    except (KeyError, IndexError, RuntimeError, AttributeError) as e:
        if debug:
            print(f"Warning: Failed to compute hinge info: {e}")

    # -------------------------------------------------------------------------- #
    # Compute the door handle grasp pose (directly use the utils function results, similar to the Panda approach)
    # -------------------------------------------------------------------------- #
    if hinge_info is None:
        # Fallback: use a default pose
        door_link_pos = env.door_link_positions()[0].cpu().numpy()
        door_grasp_pose = sapien.Pose(p=door_link_pos + np.array([0.1, 0, 0]), q=euler2quat(0, np.pi / 2, 0))
        if debug:
            print("Warning: Using fallback door grasp pose")
    else:
        # Get the precomputed info from the utils functions
        handle_pos = hinge_info["handle_pos"]
        hinge_axis = hinge_info["hinge_axis"]
        perpendicular_direction = hinge_info["perpendicular_direction"]

        # Approach direction: perpendicular direction from the hinge towards the handle (already computed by utils)
        approaching = -perpendicular_direction  # Negative direction, approaching from outside in

        # Closing direction: orthogonal to both the hinge axis and the approach direction
        closing = np.cross(hinge_axis, approaching)
        closing = closing / (np.linalg.norm(closing) + 1e-12)

        # Build the grasp pose (similar to Panda's build_grasp_pose)
        ortho = np.cross(closing, approaching)
        ortho = ortho / (np.linalg.norm(ortho) + 1e-12)

        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = handle_pos
        door_grasp_pose = sapien.Pose(T)

    # Print the door handle grasp position
    door_grasp_pos = door_grasp_pose.p
    if debug:
        print(f"Door handle grasp position: [{door_grasp_pos[0]:.4f}, {door_grasp_pos[1]:.4f}, {door_grasp_pos[2]:.4f}]")

    # Add the microwave collision point cloud for collision detection
    microwave_mesh = microwave.get_first_collision_mesh(to_world_frame=True)
    if microwave_mesh is not None:
        # Sample a point cloud from the microwave mesh for collision detection (used by the arm transporting the object)
        pts, _ = trimesh.sample.sample_surface(microwave_mesh, 512)
        right_planner.add_collision_pts(pts)

    # -------------------------------------------------------------------------- #
    # Get information about the object to grasp
    # -------------------------------------------------------------------------- #
    target_obj = env._objs[0]
    obb = get_actor_obb(target_obj)

    # Compute the pose for the right arm to grasp the object (facing the right arm base)
    obj_center = obb.centroid
    direction_to_right_base = right_base_pos - obj_center
    direction_to_right_base[2] = 0  # Ignore the vertical component
    direction_to_right_base = direction_to_right_base / np.linalg.norm(direction_to_right_base)

    # Tilted approach: blend the original vertical downward (0,0,-1) with the horizontal direction from base to object at a given tilt angle
    # The tilt angle can be controlled by the environment variable OBB_TILT_DEG (default 30 degrees)
    tilt_deg = float(os.environ.get("OBB_TILT_DEG", 30.0))
    tilt_rad = float(tilt_deg) * np.pi / 180.0
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    horiz_towards_obj = -direction_to_right_base  # From the base towards the object
    # Combine and normalize into the approach direction (used as the TCP +Z axis)
    approaching = np.cos(tilt_rad) * down + np.sin(tilt_rad) * horiz_towards_obj
    approaching = approaching / (np.linalg.norm(approaching) + 1e-12)
    # Compute a suitable closing direction (perpendicular to approaching and the direction towards the base)
    target_closing = np.cross(approaching, direction_to_right_base)
    target_closing = target_closing / (np.linalg.norm(target_closing) + 1e-12)

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )

    # Fix: get closing and center separately
    closing = grasp_info["closing"]
    center = grasp_info["center"]
    approaching = grasp_info["approaching"]

    # Build the object grasp pose
    ortho = np.cross(closing, approaching)
    ortho = ortho / np.linalg.norm(ortho)

    T = np.eye(4)
    T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
    T[:3, 3] = center
    obj_grasp_pose = sapien.Pose(T)

    # Print the object grasp position
    obj_grasp_pos = obj_grasp_pose.p
    if debug:
        print(f"Object grasp position: [{obj_grasp_pos[0]:.4f}, {obj_grasp_pos[1]:.4f}, {obj_grasp_pos[2]:.4f}]")

    # -------------------------------------------------------------------------- #
    # Phase 1: Right arm opens the door
    # -------------------------------------------------------------------------- #
    print("Phase 1: Right arm opening microwave door")

    # Reuse the previously computed hinge info (avoid recomputation)
    try:
        if hinge_info is None:
            # If the previous computation failed, try again here
            hinge_info = find_hinge_and_handle_position_for_microwave(env, offset_scale=1.0, vis=False)

        handle_pos = hinge_info["handle_pos"]
        hinge_axis = hinge_info["hinge_axis"]
        hinge_pos = hinge_info["hinge_pos"]

        # Compute the radial direction (from the handle pointing to the hinge, used for approach)
        u = hinge_axis / (np.linalg.norm(hinge_axis) + 1e-12)
        r = handle_pos - hinge_pos
        d = r - np.dot(r, u) * u  # Component perpendicular to the hinge axis
        d_norm = np.linalg.norm(d)

        if d_norm > 1e-9:
            # Direction from the hinge pointing to the handle
            radial_dir = d / d_norm
        else:
            # Fallback: use the direction towards the robot arm base
            direction_to_base = right_base_pos - handle_pos
            direction_to_base[2] = 0
            radial_dir = direction_to_base / (np.linalg.norm(direction_to_base) + 1e-12)

        # Use the grasp pose rotation
        R = door_grasp_pose.to_transformation_matrix()[:3, :3]

        # Segment A: outer pre-grasp pose (0.18m from the handle)
        d_outer = 0.1
        outer_pre_p = handle_pos + radial_dir * d_outer
        Tpre = np.eye(4)
        Tpre[:3, :3] = R
        Tpre[:3, 3] = outer_pre_p
        outer_pre_pose = sapien.Pose(Tpre)
        if debug:
            print(f"Moving to outer pre-grasp pose (distance: {d_outer}m)")
        res = right_planner.move_to_pose_with_screw(outer_pre_pose)
        if res == -1:
            if debug:
                print("❌ Failed to reach outer pre-grasp pose")
            raise RuntimeError("Failed to reach outer pre-grasp pose")
        if debug:
            print("✅ Reached outer pre-grasp pose")

        # Segment B: approach radially in steps (fewer waypoints for speed)
        # Originally 5 steps: [0.12, 0.08, 0.05, 0.02, 0.0] - too slow
        # Optimized to 2 steps: keep only the intermediate point and the final pose
        offsets = [0.05, 0.0]  # 5cm intermediate point + final position
        for i, off in enumerate(offsets):
            if debug:
                print(f"Moving to offset {off}m (step {i+1}/{len(offsets)})")
            Tstep = np.eye(4)
            Tstep[:3, :3] = R
            Tstep[:3, 3] = handle_pos + radial_dir * off
            pose_i = sapien.Pose(Tstep)
            res = right_planner.move_to_pose_with_screw(pose_i)
            if res == -1:
                if debug:
                    print(f"❌ Failed to reach offset {off}m")
                raise RuntimeError(f"Failed to reach offset {off}")
            if debug:
                print(f"✅ Reached offset {off}m")

        # Grasp the handle
        right_planner.close_gripper()

    except (RuntimeError, KeyError, IndexError, AttributeError) as e:
        # Specific exception handling: fall back to the old two-step approach strategy
        if debug:
            print(f"Radial approach failed ({e}), using fallback approach")
        door_pre_grasp_pose = door_grasp_pose * sapien.Pose([0, 0, -0.15])
        right_planner.move_to_pose_with_screw(door_pre_grasp_pose)
        door_approach_pose = door_grasp_pose * sapien.Pose([0, 0, -0.05])
        right_planner.move_to_pose_with_screw(door_approach_pose)
        right_planner.move_to_pose_with_screw(door_grasp_pose)
        right_planner.close_gripper()

    # Simplified door opening: pull linearly in the horizontal direction towards the right arm base
    if debug:
        print("Opening door by pulling towards robot base")

    T0 = door_grasp_pose.to_transformation_matrix()
    current_handle_pos = T0[:3, 3].copy()
    base_np = _to_np(right_base_pos)
    pull_direction = base_np - current_handle_pos
    pull_direction[2] = 0.0  # Pull only horizontally
    nrm = np.linalg.norm(pull_direction) + 1e-12
    pull_direction = pull_direction / nrm

    # Gradually increase the pull distance while keeping the pose unchanged
    pull_distances = [0.10, 0.20, 0.30, 0.40]
    for i, dist in enumerate(pull_distances):
        if debug:
            print(f"  Pulling {dist}m (step {i+1}/{len(pull_distances)})")
        new_pos = current_handle_pos + pull_direction * dist
        pull_pose = sapien.Pose(p=new_pos, q=door_grasp_pose.q)
        res = right_planner.move_to_pose_with_screw(pull_pose)
        if res == -1:
            if debug:
                print(f"  Failed to pull {dist}m, stopping")
            break
        if debug:
            print(f"  ✅ Pulled {dist}m")

    # # Right arm releases the door handle
    # if debug:
    #     print("Right arm releasing door handle...")
    # right_planner.open_gripper()

    # Right arm retreats a short distance from the door handle
    # door_retreat_pose = door_grasp_pose * sapien.Pose([-0.15, 0, 0])
    # right_planner.move_to_pose_with_screw(door_retreat_pose)

    # After opening the door, first return to the reset waypoint in front of the base, then grasp the object
    # if debug:
    #     print("Resetting to front-of-base waypoint before grasping object")
    # try:
    #     z_safe = float(_to_np(door_grasp_pose.p)[2]) + 0.05
    # except Exception:
    #     z_safe = 0.18
    # reset_pose = _make_reset_pose_near_base(env, right_planner, arm_side="right", distance=0.30, z_clearance=z_safe)
    # right_planner.move_to_pose_with_screw(reset_pose)

    # -------------------------------------------------------------------------- #
    # Phase 2: Right arm grasps the object (prefers AnyGrasp + lim on base_camera_back; falls back to OBB on failure)
    # -------------------------------------------------------------------------- #
    print("Phase 2: Right arm grasping object (AnyGrasp first, fallback to OBB)")

    # Add table collision detection for the right arm
    table_height = 0.0  # Assume the table height is 0
    table_extents = np.array([2.0, 2.0, 0.02])
    table_pose = sapien.Pose(p=[0, 0, table_height - table_extents[2] / 2])
    right_planner.add_box_collision(table_extents, table_pose)

    used_anygrasp = False

    # -------- AnyGrasp integration (camera: base_camera_back, lim only) --------
    try:
        # Lazy import AnyGrasp
        ag_ckpt = os.environ.get("ANYGRASP_CKPT", None)
        if ag_ckpt and os.path.exists(ag_ckpt):
            # Inject SDK paths if provided
            sdk_dir = os.environ.get("ANYGRASP_SDK_DIR", "anygrasp_sdk/grasp_detection")
            baseline_dir = os.environ.get("GRASPNET_BASELINE_DIR", "graspnet-baseline")
            if sdk_dir and sdk_dir not in sys.path:
                sys.path.insert(0, sdk_dir)
            if baseline_dir and baseline_dir not in sys.path:
                sys.path.insert(0, baseline_dir)
            try:
                from gsnet import AnyGrasp  # type: ignore
            except Exception as e:
                AnyGrasp = None
                if debug:
                    print(f"[AnyGrasp] SDK import failed: {e}")

            if 'AnyGrasp' in globals() or 'AnyGrasp' in locals():
                # Cache instance globally to avoid repeated load
                global _AG_INSTANCE
                try:
                    _AG_INSTANCE
                except NameError:
                    _AG_INSTANCE = None

                if _AG_INSTANCE is None:
                    cfgs = SimpleNamespace(
                        checkpoint_path=ag_ckpt,
                        max_gripper_width=float(os.environ.get("ANYGRASP_MAX_WIDTH", 0.10)),
                        gripper_height=float(os.environ.get("ANYGRASP_GRIPPER_HEIGHT", 0.03)),
                        top_down_grasp=True,
                        debug=False,
                    )
                    _AG_INSTANCE = AnyGrasp(cfgs)
                    _AG_INSTANCE.load_net()

                # fetch camera frame from base_camera_back
                cam_frame = _fetch_camera_frame(env, preferred_cam="base_camera_back")
                if cam_frame is None:
                    if debug:
                        print("[AnyGrasp] camera frame missing; fallback to OBB grasp")
                    raise RuntimeError("camera frame missing")

                # Build camera-frame point cloud and lim around object center (camera frame)
                world2cam = cam_frame['world2cam']
                cam2world = cam_frame['cam2world']
                pcd = cam_frame['pcd']
                xyz_cam = np.asarray(pcd.points).astype(np.float32)
                colors = np.asarray(pcd.colors).astype(np.float32) if len(pcd.colors) else None

                # Camera <-> World transforms
                Rcw = cam2world[:3, :3].astype(np.float32)  # X_w = Rcw * X_c + tcw
                tcw = cam2world[:3, 3].astype(np.float32)
                Rwc = Rcw.T  # X_c = Rwc * (X_w - tcw)

                # Lims in camera frame centered at object OBB center (converted to camera)
                obj_center_w = obb.centroid.astype(np.float64)
                obj_center_c = (Rwc @ (obj_center_w.astype(np.float32) - tcw)).astype(np.float64)
                rx, ry, rz = _parse_half_sizes_csv(os.environ.get("ANYGRASP_LIM_HALFSIZES", "0.08,0.08,0.10"))
                zmin = max(0.01, float(obj_center_c[2] - rz))
                zmax = max(zmin + 1e-3, float(obj_center_c[2] + rz))
                lims_cam = [
                    float(obj_center_c[0] - rx), float(obj_center_c[0] + rx),
                    float(obj_center_c[1] - ry), float(obj_center_c[1] + ry),
                    float(zmin), float(zmax),
                ]
                if debug:
                    print(f"[AnyGrasp] Using lim (camera frame): {lims_cam}")
                    # Print how many points fall inside lim (camera frame)
                    try:
                        if xyz_cam.size:
                            xmn, xmx, ymn, ymx, zmn, zmx = lims_cam
                            inside = (
                                (xyz_cam[:, 0] >= xmn) & (xyz_cam[:, 0] <= xmx) &
                                (xyz_cam[:, 1] >= ymn) & (xyz_cam[:, 1] <= ymx) &
                                (xyz_cam[:, 2] >= zmn) & (xyz_cam[:, 2] <= zmx)
                            )
                            n_in = int(inside.sum())
                            print(f"[AnyGrasp] points inside lim (camera): {n_in}/{len(xyz_cam)}")
                    except Exception as _e:
                        if debug:
                            print(f"[AnyGrasp] lim inside-count failed: {_e}")

                # Optional: dump camera-frame point cloud to disk for inspection
                try:
                    dump_flag = os.environ.get("ANYGRASP_DUMP_PLY", "0") not in ("0", "false", "False")
                    if dump_flag:
                        out_dir = os.environ.get("ANYGRASP_DUMP_DIR", "/tmp")
                        os.makedirs(out_dir, exist_ok=True)
                        ts = int(time.time())
                        base = f"anygrasp_cam_pcd_{ts}"
                        npz_path = os.path.join(out_dir, base + ".npz")
                        np.savez(npz_path, xyz=xyz_cam, colors=colors)
                        if o3d is not None:
                            pcd_o3d = o3d.geometry.PointCloud()
                            pcd_o3d.points = o3d.utility.Vector3dVector(xyz_cam.astype(float))
                            if colors is not None and colors.shape == xyz_cam.shape:
                                cols = np.clip(colors, 0.0, 1.0)
                                pcd_o3d.colors = o3d.utility.Vector3dVector(cols.astype(float))
                            ply_path = os.path.join(out_dir, base + ".ply")
                            o3d.io.write_point_cloud(ply_path, pcd_o3d)
                            if debug:
                                print(f"[AnyGrasp] Dumped camera point cloud: {ply_path} (and NPZ: {npz_path})")
                        else:
                            if debug:
                                print(f"[AnyGrasp] Dumped camera point cloud NPZ only: {npz_path} (open3d not available)")
                except Exception as _e:
                    if debug:
                        print(f"[AnyGrasp] Dump point cloud failed: {_e}")

                apply_mask = os.environ.get("ANYGRASP_APPLY_MASK", "1") not in ("0", "false", "False")
                use_collision = os.environ.get("ANYGRASP_COLLISION", "0") in ("1", "true", "True")
                gg, _cloud = _AG_INSTANCE.get_grasp(
                    xyz_cam,
                    colors,
                    lims=lims_cam,
                    apply_object_mask=apply_mask,
                    dense_grasp=False,
                    collision_detection=use_collision,
                )

                if gg is not None and len(gg) > 0:
                    try:
                        gg = gg.nms().sort_by_score()
                    except Exception:
                        try:
                            gg.sort_by_score()
                        except Exception:
                            pass

                    # iterate candidates: dry-run IK/planning first
                    topk = int(os.environ.get("ANYGRASP_TOPK", 20))
                    pre_dist = float(os.environ.get("ANYGRASP_PRE_DIST", 0.10))
                    lift_dist = float(os.environ.get("ANYGRASP_LIFT_DIST", 0.15))

                    def pose_from_RT(R, t):
                        T = np.eye(4)
                        T[:3, :3] = R
                        T[:3, 3] = t
                        return sapien.Pose(T)

                    # optional fixed alignment (currently identity)
                    R_align = np.eye(3)
                    t_align = np.zeros(3)

                    selected = None
                    for i in range(min(topk, len(gg))):
                        try:
                            cand = gg[i]
                            # Candidate is in camera frame; convert to world frame
                            R_cam = cand.rotation_matrix
                            p_cam = cand.translation
                            R_world = Rcw @ R_cam
                            p_world = (Rcw @ p_cam) + tcw
                            if debug:
                                try:
                                    score = getattr(cand, 'score', None)
                                    print(f"[AnyGrasp] cand {i}: score={score} world_p={p_world}")
                                except Exception:
                                    print(f"[AnyGrasp] cand {i}: world_p={p_world}")
                            # align to TCP
                            R_tcp = R_world @ R_align
                            p_tcp = p_world + (R_world @ t_align)
                            final_pose = pose_from_RT(R_tcp, p_tcp)
                            if debug:
                                print(f"[AnyGrasp] cand {i}: tcp_p={final_pose.p}, tcp_q={final_pose.q}")
                            # approach axis: z-axis of TCP
                            approach = R_tcp[:, 2]
                            if debug:
                                print(f"[AnyGrasp] cand {i}: tcp_approach_axis(z)={approach}")
                            pre_p = p_tcp - approach * pre_dist
                            pre_pose = pose_from_RT(R_tcp, pre_p)

                            # dry-run planning to pre and final
                            res1 = right_planner.move_to_pose_with_screw(pre_pose, dry_run=True)
                            if res1 == -1:
                                continue
                            res2 = right_planner.move_to_pose_with_screw(final_pose, dry_run=True)
                            if res2 == -1:
                                continue
                            selected = (pre_pose, final_pose, approach)
                            break
                        except Exception as e:
                            if debug:
                                print(f"[AnyGrasp] candidate {i} failed in dry-run: {e}")
                            continue

                    if selected is not None:
                        if debug:
                            print("[AnyGrasp] Found feasible grasp candidate; executing")
                        pre_pose, final_pose, approach = selected
                        right_planner.open_gripper()
                        r1 = right_planner.move_to_pose_with_screw(pre_pose)
                        if r1 == -1:
                            raise RuntimeError("pre-grasp failed")
                        r2 = right_planner.move_to_pose_with_screw(final_pose)
                        if r2 == -1:
                            raise RuntimeError("final grasp failed")
                        right_planner.close_gripper()
                        # lift
                        lift_p = final_pose.p + (-approach) * lift_dist
                        lift_pose = sapien.Pose(p=lift_p, q=final_pose.q)
                        right_planner.move_to_pose_with_screw(lift_pose)
                        used_anygrasp = True
                    else:
                        if debug:
                            print("[AnyGrasp] All candidates failed dry-run planning; using OBB fallback")
                else:
                    if debug:
                        print("[AnyGrasp] No grasps returned; fallback to OBB")
        else:
            if debug:
                print("[AnyGrasp] Checkpoint missing; set ANYGRASP_CKPT to enable")
    except Exception as e:
        if debug:
            print(f"[AnyGrasp] Failed; reason: {e}. Using OBB fallback.")

    obb_success = False
    if not used_anygrasp:
        # ---------- Fallback: OBB-based grasp (now with basic result checks) ----------
        if debug:
            print("[OBB] Fallback grasp: approach -> pre -> final -> close -> lift")
        # Move the right arm above the object
        obj_approach_pose = obj_grasp_pose * sapien.Pose([0, 0, -0.2])
        r1 = right_planner.move_to_pose_with_screw(obj_approach_pose)
        if r1 == -1:
            if debug:
                print("[OBB] approach move failed; aborting OBB fallback")
        else:
            # Precise approach
            obj_pre_grasp_pose = obj_grasp_pose * sapien.Pose([0, 0, -0.05])
            r2 = right_planner.move_to_pose_with_screw(obj_pre_grasp_pose)
            if r2 == -1:
                if debug:
                    print("[OBB] pre-grasp move failed; aborting OBB fallback")
            else:
                # The right arm descends and grasps
                r3 = right_planner.move_to_pose_with_screw(obj_grasp_pose)
                if r3 == -1:
                    if debug:
                        print("[OBB] final grasp move failed; aborting OBB fallback")
                else:
                    right_planner.close_gripper()
                    # Lift the object
                    obj_lift_pose = obj_grasp_pose * sapien.Pose([0, 0, -0.15])
                    r4 = right_planner.move_to_pose_with_screw(obj_lift_pose)
                    if r4 == -1:
                        if debug:
                            print("[OBB] lift move failed; aborting OBB fallback")
                    else:
                        obb_success = True

    # -------------------------------------------------------------------------- #
    # Phase 3: Right arm places the object in the microwave
    # -------------------------------------------------------------------------- #
    print("Phase 3: Right arm placing object in microwave")

    # Compute the target position inside the microwave
    microwave_pos = env.microwave.pose.p[0].cpu().numpy()

    # Get the center of the microwave interior - simplified handling
    place_target_pos = microwave_pos + np.array([0, 0, 0.15])

    # Only attempt to place the object after a successful grasp (AnyGrasp or OBB)
    if used_anygrasp or obb_success:
        # Move to the front of the microwave
        pre_place_pose = sapien.Pose(
            p=place_target_pos + np.array([0.3, 0, 0]),
            q=obj_grasp_pose.q
        )
        right_planner.move_to_pose_with_screw(pre_place_pose)

        # Slowly place the object into the microwave
        mid_place_pose = sapien.Pose(
            p=place_target_pos + np.array([0.1, 0, 0]),
            q=obj_grasp_pose.q
        )
        right_planner.move_to_pose_with_screw(mid_place_pose)

        # Final placement position
        place_pose = sapien.Pose(p=place_target_pos, q=obj_grasp_pose.q)
        right_planner.move_to_pose_with_screw(place_pose)

        # Release the object
        right_planner.open_gripper()

        # The right arm retreats carefully
        right_retreat_pose1 = place_pose * sapien.Pose([-0.1, 0, 0])
        right_planner.move_to_pose_with_screw(right_retreat_pose1)

        right_retreat_pose2 = place_pose * sapien.Pose([-0.3, 0, 0.1])
        res = right_planner.move_to_pose_with_screw(right_retreat_pose2)
    else:
        if debug:
            print("[Place] Skipped placing because grasp did not succeed")
        res = -1

    # # -------------------------------------------------------------------------- #
    # # Phase 4: Right arm releases the door (optional: close the door)
    # # -------------------------------------------------------------------------- #
    # # Note: the right arm already released the door after Phase 1, so this phase is no longer needed
    # print("Phase 4: Right arm releasing door")

    # # Optionally keep the door open or closed
    # # Here we choose to keep it open
    # right_planner.open_gripper()

    # # Right arm retreats
    # right_retreat_pose = door_grasp_pose * sapien.Pose([-0.2, 0, 0])
    # right_planner.move_to_pose_with_screw(right_retreat_pose)

    # Cleanup
    left_planner.close()
    right_planner.close()

    return res
