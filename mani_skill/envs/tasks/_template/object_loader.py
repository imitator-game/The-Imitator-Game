"""
Minimal object loader template for building env scenes.

Reference: the unified object-loading pattern of the benchmark's asset system,
simplified to the four asset namespaces used by this benchmark:

    - YCB            -> mani_skill.utils.building.actors.get_actor_builder("ycb:...")
    - RoboTwin       -> mani_skill.utils.scene_builder.table.utils.create_actor(...)
    - PartNet-Mobility -> mani_skill.utils.building.articulations.get_articulation_builder(
                           "partnet-mobility:<id>", mode=..., scale=...)
    - sketchfab GLB  -> mani_skill.utils.building.actors.create_sketchfab_actor(...)

Every function returns the built actor (or articulation) and records its
initial pose, so your env can store them and reuse them in _initialize_episode.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.building import actors, articulations
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.building.actors.sketchfab import create_sketchfab_actor

# PartNet collision group bits (matching obj_loader.py).
PARTNET_COLLISION_BIT = 29


# --------------------------------------------------------------------------- #
# YCB (rigid, single-part objects)
# --------------------------------------------------------------------------- #
def load_ycb_object(scene, model_id: str, position, rotation=(0, 0, 0),
                    scale: float = 1.0, mass: Optional[float] = None,
                    name_suffix: str = ""):
    """
    Load one YCB object.

    Args:
        model_id: e.g. "013_apple", "002_master_chef_can"
        position: [x, y, z]
        rotation: [roll, pitch, yaw] in radians
        scale:    uniform scale factor
    """
    pose = sapien.Pose(p=np.asarray(position), q=euler2quat(*rotation))
    builder = actors.get_actor_builder(scene, id=f"ycb:{model_id}", scales=[scale])
    if mass is not None:
        builder._mass = mass
    builder.initial_pose = pose
    obj = builder.build(name=f"{model_id}{name_suffix}")
    return obj, pose


# --------------------------------------------------------------------------- #
# RoboTwin (glb-based single-part objects)
# --------------------------------------------------------------------------- #
def load_robotwin_object(scene, modelname: str, position, rotation=(0, 0, 0),
                         model_id: Optional[int] = None, scale=None,
                         replace_scale=False, convex=True, is_static=False,
                         name_suffix: str = ""):
    """
    Load one RoboTwin object.

    Args:
        modelname: e.g. "035_apple", "062_plasticbox", "076_breadbasket"
        model_id:  model variant id (model_data{id}.json); None = model_data.json
        scale:     tuple or scalar; leave None to use the json-defined scale
    """
    if model_id is None:
        model_id = get_model_id(modelname, random_id=False)
    pose = sapien.Pose(p=np.asarray(position), q=euler2quat(*rotation))
    actor_obj = create_actor(scene=scene, pose=pose, modelname=modelname,
                             scale=scale, replace_scale=replace_scale,
                             convex=convex, is_static=is_static, model_id=model_id)
    return actor_obj.actor, pose


# --------------------------------------------------------------------------- #
# PartNet-Mobility (articulated, multi-link)
# --------------------------------------------------------------------------- #
def load_partnet_object(scene, category: str, model_id: str, position,
                        rotation=(0, 0, 0), scale: float = 0.3,
                        face_robot_base: bool = True,
                        robot_base_position: Optional[List[float]] = None,
                        name_suffix: str = ""):
    """
    Load one PartNet-Mobility articulated object (microwave, cabinet, ...).

    Args:
        category: lower-case category used as the load mode, e.g. "microwave"
        model_id: e.g. "7119" (see PARTNET_ID_MAPPING in obj_cls.py)
        face_robot_base: if True, yaw the object so it faces the robot base
    """
    rotation = list(rotation)
    if face_robot_base and robot_base_position is not None:
        dx = position[0] - robot_base_position[0]
        dy = position[1] - robot_base_position[1]
        rotation[2] = math.atan2(dy, dx)
    pose = sapien.Pose(p=np.asarray(position), q=euler2quat(*rotation))

    builder = articulations.get_articulation_builder(
        scene, f"partnet-mobility:{model_id}", mode=category.lower(), scale=scale)
    builder.initial_pose = pose
    obj = builder.build(name=f"{category}-{model_id}{name_suffix}")

    # Standard collision-group setup (keeps the object interactable).
    for link in obj.links:
        link.set_collision_group_bit(group=2, bit_idx=PARTNET_COLLISION_BIT, bit=1)
    return obj, pose


# --------------------------------------------------------------------------- #
# sketchfab / external GLB (rigid, single-part)
# --------------------------------------------------------------------------- #
def load_sketchfab_object(scene, object_key: str, position, rotation=(0, 0, 0),
                          scale: Optional[List[float]] = None, name_suffix: str = ""):
    """
    Load an external GLB asset registered in assets/sketchfab_registry.json.

    Args:
        object_key: stable registry key, e.g. "balance_scale", "weight_on_scale"
        scale:      [sx, sy, sz]; None uses the registry scale
    """
    pose = sapien.Pose(p=np.asarray(position), q=euler2quat(*rotation))
    obj = create_sketchfab_actor(scene, object_key=object_key, initial_pose=pose,
                                 scales=scale)
    return obj, pose


# --------------------------------------------------------------------------- #
# Helper: compute the z-offset to place an actor's bottom on the table (z=0)
# --------------------------------------------------------------------------- #
def z_offset_to_table(obj) -> float:
    """Returns the z needed so obj's collision-mesh bottom rests at z=0."""
    collision_mesh = obj.get_first_collision_mesh()
    if collision_mesh is None:
        return 0.0
    return float(-collision_mesh.bounding_box.bounds[0, 2])


def batched_z_offsets(objs) -> torch.Tensor:
    """Like z_offset_to_table but for a list of actors (returns a tensor)."""
    return torch.tensor([z_offset_to_table(o) for o in objs], dtype=torch.float32)
