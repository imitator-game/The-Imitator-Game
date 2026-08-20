import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import trimesh
from transforms3d import quaternions
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import common
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh


def get_link_obb(link, to_world_frame=True):
    """Get the OBB of a link in an articulated object (borrowed from the solutions files)"""
    # Get the link's collision mesh
    meshes = []
    for comp in link._objs[0].entity.components:
        if isinstance(comp, physx.PhysxRigidBodyComponent):
            mesh = get_component_mesh(comp, to_world_frame=to_world_frame)
            if mesh is not None:
                meshes.append(mesh)

    if not meshes:
        return None

    # Merge all meshes
    combined_mesh = trimesh.util.concatenate(meshes)
    return combined_mesh.bounding_box_oriented


def get_actor_obb(actor: Actor, to_world_frame=True, vis=False):
    mesh = get_component_mesh(
        actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
        to_world_frame=to_world_frame,
    )
    assert mesh is not None, "can not get actor mesh for {}".format(actor)

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented

    if vis:
        obb.visual.vertex_colors = (255, 0, 0, 10)
        trimesh.Scene([mesh, obb]).show()

    return obb


def compute_grasp_info_by_obb(
    obb: trimesh.primitives.Box,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.0,
    ortho=True,
):
    """Compute grasp info given an oriented bounding box.
    The grasp info includes axes to define grasp frame, namely approaching, closing, orthogonal directions and center.

    Args:
        obb: oriented bounding box to grasp
        approaching: direction to approach the object
        target_closing: target closing direction, used to select one of multiple solutions
        depth: displacement from hand to tcp along the approaching vector. Usually finger length.
        ortho: whether to orthogonalize closing  w.r.t. approaching.
    """
    # NOTE(jigu): DO NOT USE `x.extents`, which is inconsistent with `x.primitive.transform`!
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)

    # Assume normalized
    approaching = np.array(approaching)

    # Find the axis closest to approaching vector
    angles = approaching @ T[:3, :3]  # [3]
    inds0 = np.argsort(np.abs(angles))
    ind0 = inds0[-1]

    # Find the shorter axis as closing vector
    inds1 = np.argsort(extents[inds0[0:-1]])
    ind1 = inds0[0:-1][inds1[0]]
    ind2 = inds0[0:-1][inds1[1]]

    # If sizes are close, choose the one closest to the target closing
    if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
        vec1 = T[:3, ind1]
        vec2 = T[:3, ind2]
        if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
            ind1 = inds0[0:-1][inds1[1]]
            ind2 = inds0[0:-1][inds1[0]]
    closing = T[:3, ind1]

    # Flip if far from target
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing

    # Reorder extents
    extents = extents[[ind0, ind1, ind2]]

    # Find the origin on the surface
    center = T[:3, 3].copy()
    half_size = extents[0] * 0.5
    center = center + approaching * (-half_size + min(depth, half_size))

    if ortho:
        closing = closing - (approaching @ closing) * approaching
        closing = common.np_normalize_vector(closing)

    grasp_info = dict(
        approaching=approaching, closing=closing, center=center, extents=extents
    )
    return grasp_info


def find_hinge_and_handle_position_for_microwave(env, env_idx: int = 0, offset_scale: float = 1.0, vis: bool = False):
    """
    Estimate the microwave door handle position using a simple geometric approach.

    New Logic:
    1) Find the door hinge (revolute/revolute_unwrapped joint) and compute hinge world pose + rotation axis.
    2) Use hinge position as the axis center point.
    3) Find the door's geometric center (OBB center).
    4) Calculate direction from axis center to door center.
    5) Move along this direction by one object length to find handle position.

    Args:
        env: Gym environment with RealmanMicrowave-v1 loaded (after reset()).
        env_idx: Parallel environment index if running batched envs.
        offset_scale: Multiplier for the object length distance.
                      Default is 1.0 (move by one object length).
        vis: If True, open a trimesh viewer to visualize hinge/handle and the door OBB.

    Returns:
        dict with keys:
            - hinge_pos (3,): np.ndarray, world position of hinge (axis center)
            - hinge_axis (3,): np.ndarray, world unit vector of hinge axis
            - handle_pos (3,): np.ndarray, estimated world position of handle
            - dir_world (3,): np.ndarray, world unit vector from axis center towards door center
            - extent_used (float): the object length used for the offset (in meters)
            - link_name (str): the chosen door link name
    """
    # 1) Access the (unmerged) microwave articulation
    microw_list = getattr(env.unwrapped, "_microwaves", None)
    if not microw_list:
        raise RuntimeError("Microwave articulation not found on env.unwrapped._microwaves")
    microw = microw_list[0]

    # Find the door link directly (link_1), without using a scoring mechanism
    door_link = None

    for link in microw.links:
        jt = link.joint
        if jt is None:
            continue

        # Method 1: find the door directly by name
        if hasattr(link, 'name') and 'link_1' in str(link.name):
            door_link = link
            break

        # Method 2: find the door by joint features (revolute + with angle limits)
        jtype = getattr(jt, "type", [""])
        jtype_s = jtype[0] if isinstance(jtype, (list, tuple, np.ndarray)) else str(jtype)
        if ("revolute" in jtype_s and
            hasattr(jt, 'limit') and
            hasattr(jt.limit, 'upper') and
            jt.limit.upper > 1.0 and jt.limit.upper < 2.0):  # 60-120 degree range, most likely the door
            door_link = link
            break

    if door_link is None:
        raise RuntimeError("Could not find door link in microwave")

    # Handle the found door link directly
    link = door_link
    jt = link.joint

    # Hinge world pose = parent_link.world * joint.pose_in_parent
    parent_link = jt.get_parent_link()
    parent_pose = parent_link.pose[env_idx].sp  # sapien.Pose
    pose_in_parent = jt.get_pose_in_parent()
    try:
        pin_sp = Pose.create(pose_in_parent).sp  # type: ignore[name-defined]
    except Exception:
        # already a sapien.Pose
        pin_sp = pose_in_parent
    hinge_sp = parent_pose * pin_sp
    T_hinge = hinge_sp.to_transformation_matrix()
    hinge_pos = T_hinge[:3, 3]

    # Get the actual joint axis from the joint definition and transform to world coordinates
    # The axis is defined in the joint's local coordinate frame
    # From URDF: <axis xyz="0 -1 0"/> for microwave door (joint_1)
    joint_axis_local = np.array([0.0, -1.0, 0.0])  # Use correct axis from URDF

    # Try different ways to get the joint axis (but fall back to URDF value)
    if vis:
        print(f"DEBUG: Initial joint_axis_local (from URDF): {joint_axis_local}")

    if hasattr(jt, 'get_axis'):
        try:
            axis_candidate = jt.get_axis()
            if vis:
                print(f"DEBUG: jt.get_axis() returned: {axis_candidate}")
            if axis_candidate is not None and np.linalg.norm(axis_candidate) > 0:
                if vis:
                    print(f"DEBUG: Using axis from get_axis(): {axis_candidate}")
                joint_axis_local = axis_candidate
        except Exception as e:
            if vis:
                print(f"DEBUG: jt.get_axis() failed: {e}")
    elif hasattr(jt, 'axis'):
        try:
            axis_attr = getattr(jt, 'axis')
            if vis:
                print(f"DEBUG: jt.axis returned: {axis_attr}")
            if axis_attr is not None and np.linalg.norm(axis_attr) > 0:
                if vis:
                    print(f"DEBUG: Using axis from .axis: {axis_attr}")
                joint_axis_local = np.array(axis_attr)
        except Exception as e:
            if vis:
                print(f"DEBUG: jt.axis failed: {e}")
    elif hasattr(jt, '_axis'):
        try:
            axis_candidate = jt._axis
            if vis:
                print(f"DEBUG: jt._axis returned: {axis_candidate}")
            if axis_candidate is not None and np.linalg.norm(axis_candidate) > 0:
                if vis:
                    print(f"DEBUG: Using axis from ._axis: {axis_candidate}")
                joint_axis_local = axis_candidate
        except Exception as e:
            if vis:
                print(f"DEBUG: jt._axis failed: {e}")
    else:
        if vis:
            print(f"DEBUG: No axis methods found, using URDF default: {joint_axis_local}")

    if vis:
        print(f"DEBUG: Final joint_axis_local (before normalization): {joint_axis_local}")

    # Ensure it's a unit vector
    joint_axis_local = joint_axis_local / (np.linalg.norm(joint_axis_local) + 1e-12)

    if vis:
        print(f"DEBUG: joint_axis_local (after normalization): {joint_axis_local}")
        print(f"DEBUG: T_hinge rotation matrix:\n{T_hinge[:3, :3]}")

    # For microwave door, the hinge axis should always be vertical in world coordinates
    # regardless of how the microwave is oriented in the scene
    # Use world vertical axis (negative Z direction for typical door hinge)
    hinge_axis = np.array([0.0, 0.0, -1.0])  # Vertical downward (typical door hinge direction)

    if vis:
        print(f"DEBUG: Using fixed world vertical axis: {hinge_axis}")
        print(f"DEBUG: (Ignoring transformed local axis which would be: {T_hinge[:3, :3] @ joint_axis_local})")

    # Door mesh for this link (world frame) - use a more accurate method
    obb = get_link_obb(link, to_world_frame=True)
    if obb is None:
        if vis:
            print(f"❌ Error: Cannot get OBB for door link {link.name}")
        return None

    extents = np.array(obb.primitive.extents)

    # Get the door's center position (OBB centroid)
    door_center_pos = obb.centroid  # Use the OBB centroid as the door center

    # Compute the perpendicular direction from the door center to the hinge axis line
    # 1. Compute the projection of the door center onto the hinge axis line
    to_door = door_center_pos - hinge_pos
    projection_length = np.dot(to_door, hinge_axis)
    projection_point = hinge_pos + projection_length * hinge_axis

    # 2. Compute the direction from the projection point to the door center (perpendicular to the hinge axis)
    perpendicular_component = door_center_pos - projection_point
    perpendicular_distance = np.linalg.norm(perpendicular_component)

    if perpendicular_distance > 1e-9:
        # Perpendicular direction pointing from the hinge axis line to the door center
        perpendicular_direction = perpendicular_component / perpendicular_distance
    else:
        # If the door center lies on the hinge axis (should not happen in theory), use a default direction
        perpendicular_direction = np.array([1.0, 0.0, 0.0])

    # Door handle computation: move half the object length from the door center along the direction perpendicular to the hinge axis
    # Find the door's longest axis (object length)
    max_extent_idx = np.argmax(extents)
    max_extent = extents[max_extent_idx]

    # Move half the object length from the door center along the direction perpendicular to the hinge axis
    handle_distance = (max_extent / 2.0) * float(offset_scale)
    handle_pos = door_center_pos + perpendicular_direction * handle_distance

    # Use the computed handle position and direction
    direction = perpendicular_direction

    # Debug output (can be enabled for verification)
    if vis:
        print(f"Link: {getattr(link, 'name', '')}")
        print(f"Joint axis (world): {hinge_axis}")
        print(f"Joint origin (world): {hinge_pos}")
        print(f"Door center (OBB centroid): {door_center_pos}")
        print(f"Projection point on axis line: {projection_point}")
        print(f"Perpendicular component: {perpendicular_component}")
        print(f"Perpendicular direction: {perpendicular_direction}")
        print(f"Perpendicular distance: {perpendicular_distance:.3f}")
        print(f"Max extent (object length): {max_extent:.3f}")
        print(f"Handle distance (half length): {handle_distance:.3f}")
        print(f"Handle position: {handle_pos}")
        print(f"Direction (perpendicular): {direction}")
        print("---")

    res = dict(
        hinge_pos=hinge_pos,
        hinge_axis=hinge_axis,
        handle_pos=handle_pos,
        dir_world=direction,  # Direction perpendicular to the hinge axis
        extent_used=max_extent * float(offset_scale),  # Use the maximum extent
        link_name=getattr(link, "name", ""),
        door_center=door_center_pos,  # Door center position
        projection_point=projection_point,  # Projection of the door center on the hinge axis line
        perpendicular_direction=perpendicular_direction,  # Direction perpendicular to the hinge axis
        perpendicular_distance=perpendicular_distance,    # Perpendicular distance
        handle_distance=handle_distance,  # Distance from the handle to the door center
        door_obb_info=dict(           # Detailed door OBB info
            center=obb.centroid,
            transform=obb.primitive.transform,
            extents=extents,
            axes=[obb.primitive.transform[:3, 0], obb.primitive.transform[:3, 1], obb.primitive.transform[:3, 2]]
        )
    )

    # Return the result directly, no scoring needed
    out = res

    if vis:
        # Visualize hinge (red), handle (green), and link OBB
        try:
            import open3d as o3d
            geoms = []
            p_h = out["hinge_pos"].astype(float)
            p_g = out["handle_pos"].astype(float)
            s1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            s1.paint_uniform_color([1.0, 0.2, 0.2])
            s1.translate(p_h)
            s2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            s2.paint_uniform_color([0.2, 1.0, 0.2])
            s2.translate(p_g)
            geoms.extend([s1, s2])
            o3d.visualization.draw_geometries(geoms, window_name="Hinge (red) / Handle (green)")
        except Exception:
            pass

    return out
