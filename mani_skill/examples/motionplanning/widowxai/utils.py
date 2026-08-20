import numpy as np
import sapien
import sapien.physx as physx
import trimesh
from transforms3d import quaternions
from mani_skill.utils.structs import Actor
from mani_skill.utils import common
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

def get_actor_obb(actor: Actor, to_world_frame=True, vis=False):
    """
    Compute the oriented bounding box (OBB) of an actor in the ManiSkill environment.

    Args:
        actor: The actor object to compute the OBB for.
        to_world_frame: If True, transform the mesh to world frame coordinates.
        vis: If True, visualize the mesh and OBB using trimesh's Scene viewer.

    Returns:
        trimesh.primitives.Box: The oriented bounding box of the actor.
    """
    mesh = get_component_mesh(
        actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
        to_world_frame=to_world_frame,
    )
    assert mesh is not None, f"Cannot get actor mesh for {actor}"

    obb: trimesh.primitives.Box = mesh.bounding_box_oriented

    if vis:
        obb.visual.vertex_colors = (255, 0, 0, 10)
        trimesh.Scene([mesh, obb]).show()

    return obb

def compute_grasp_info_by_obb(
    obb: trimesh.primitives.Box,
    approaching=(0, 0, -1),
    target_closing=None,
    depth=0.025,
    ortho=True,
):
    """
    Compute grasp information given an oriented bounding box for WidowXAI.

    The grasp info includes axes to define grasp frame (approaching, closing, orthogonal directions)
    and the center of the grasp.

    Args:
        obb: Oriented bounding box to grasp.
        approaching: Direction to approach the object (default: from above, along negative z-axis).
        target_closing: Target closing direction, used to select one of multiple solutions.
        depth: Displacement from hand to TCP along the approaching vector (default: 0.025m for WidowXAI gripper).
        ortho: Whether to orthogonalize closing direction w.r.t. approaching.

    Returns:
        dict: Dictionary containing approaching, closing, center, and extents of the grasp.
    """
    extents = np.array(obb.primitive.extents)
    T = np.array(obb.primitive.transform)
    approaching = np.array(approaching)
    angles = approaching @ T[:3, :3]
    inds0 = np.argsort(np.abs(angles))
    ind0 = inds0[-1]
    inds1 = np.argsort(extents[inds0[0:-1]])
    ind1 = inds0[0:-1][inds1[0]]
    ind2 = inds0[0:-1][inds1[1]]
    if target_closing is not None and 0.99 < (extents[ind1] / extents[ind2]) < 1.01:
        vec1 = T[:3, ind1]
        vec2 = T[:3, ind2]
        if np.abs(target_closing @ vec1) < np.abs(target_closing @ vec2):
            ind1 = inds0[0:-1][inds1[1]]
            ind2 = inds0[0:-1][inds1[0]]
    closing = T[:3, ind1]
    if target_closing is not None and target_closing @ closing < 0:
        closing = -closing
    extents = extents[[ind0, ind1, ind2]]
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