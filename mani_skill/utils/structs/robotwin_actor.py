import numpy as np
import sapien
import sapien.physx
import transforms3d as t3d
from typing import Union, List, Dict, Optional

# Try to import ManiSkill3 components - they may not always be available
try:
    from mani_skill.utils.structs import Actor as ManiSkillActor
    from mani_skill.utils.structs import Articulation as ManiSkillArticulation

    MANISKILL_AVAILABLE = True
except ImportError:
    ManiSkillActor = None
    ManiSkillArticulation = None
    MANISKILL_AVAILABLE = False


class Actor:

    POINTS = {
        "contact": "contact_points_pose",
        "target": "target_pose",
        "functional": "functional_matrix",
        "orientation": "orientation_point",
    }

    def __init__(self, actor: Union[sapien.Entity, 'ManiSkillActor'], actor_data: Dict, mass: float = 0.01):
        """
        Args:
            actor: sapien.Entity or ManiSkill3 Actor instance
            actor_data: Dictionary containing point configurations
            mass: Mass to set for the actor
        """
        # Store the underlying actor for compatibility
        self.actor = actor
        self.config = actor_data

        # Check if it's a ManiSkillActor or sapien.Entity
        if MANISKILL_AVAILABLE and isinstance(actor, ManiSkillActor):
            # Copy relevant attributes from ManiSkillActor
            self._is_maniskill_actor = True
            self._copy_maniskill_attributes(actor)
        else:
            # It's a sapien.Entity (original usage)
            self._is_maniskill_actor = False

        self.set_mass(mass)

    def _copy_maniskill_attributes(self, ms_actor):
        """Copy relevant attributes from ManiSkillActor"""
        # Only copy essential attributes to avoid conflicts
        self.scene = ms_actor.scene if hasattr(ms_actor, 'scene') else None
        self.device = ms_actor.device if hasattr(ms_actor, 'device') else None
        self._objs = ms_actor._objs if hasattr(ms_actor, '_objs') else [self.actor]
        self._bodies = ms_actor._bodies if hasattr(ms_actor, '_bodies') else None
        self._scene_idxs = ms_actor._scene_idxs if hasattr(ms_actor, '_scene_idxs') else None

    def get_point(self, type: str, idx: int, ret: str = "list") -> Union[np.ndarray, List, sapien.Pose]:
        """Get the point of the entity actor."""
        type_key = self.POINTS[type]

        # Get actor pose
        actor_pose = self.actor.get_pose()
        actor_matrix = actor_pose.to_transformation_matrix()

        local_matrix = np.array(self.config[type_key][idx])
        local_matrix[:3, 3] *= np.array(self.config["scale"])

        world_matrix = actor_matrix @ local_matrix

        if ret == "matrix":
            return world_matrix
        elif ret == "list":
            return (world_matrix[:3, 3].tolist() +
                    t3d.quaternions.mat2quat(world_matrix[:3, :3]).tolist())
        else:
            return sapien.Pose(world_matrix[:3, 3],
                               t3d.quaternions.mat2quat(world_matrix[:3, :3]))

    def get_pose(self) -> sapien.Pose:
        """Get the sapien.Pose of the actor."""
        if self._is_maniskill_actor and hasattr(self, 'scene'):
            # Use ManiSkill's pose property
            pose = self.pose
            if hasattr(pose, 'sp'):
                return pose.sp
            return pose
        else:
            # Use original sapien.Entity method
            return self.actor.get_pose()

    @property
    def pose(self):
        """Get the pose - compatible with ManiSkillActor interface"""
        if self._is_maniskill_actor and hasattr(self, '_objs'):
            # Delegate to ManiSkillActor's pose implementation
            if hasattr(self.actor, 'pose'):
                return self.actor.pose
        return self.actor.get_pose()

    def get_contact_point(self, idx: int, ret: str = "list"):
        """Get the transformation matrix of given contact point of the actor."""
        return self.get_point("contact", idx, ret)

    def iter_contact_points(self, ret: str = "list"):
        """Iterate over all contact points of the actor."""
        for i in range(len(self.config[self.POINTS["contact"]])):
            yield i, self.get_point("contact", i, ret)

    def get_functional_point(self, idx: int, ret: str = "list"):
        """Get the transformation matrix of given functional point of the actor."""
        return self.get_point("functional", idx, ret)

    def get_target_point(self, idx: int, ret: str = "list"):
        """Get the transformation matrix of given target point of the actor."""
        return self.get_point("target", idx, ret)

    def get_orientation_point(self, ret: str = "list"):
        """Get the transformation matrix of given orientation point of the actor."""
        return self.get_point("orientation", 0, ret)

    def get_name(self) -> str:
        """Get the name of the actor."""
        if hasattr(self, 'name'):
            return self.name
        return self.actor.get_name()

    def set_name(self, name: str):
        """Set the name of the actor."""
        if hasattr(self, 'name'):
            self.name = name
        if hasattr(self.actor, 'set_name'):
            self.actor.set_name(name)

    def set_mass(self, mass: float):
        """Set the mass of the actor."""
        if self._is_maniskill_actor and hasattr(self, '_bodies') and self._bodies:
            # ManiSkill3 actor with bodies
            for body in self._bodies:
                if hasattr(body, 'mass'):
                    body.mass = mass
        else:
            # Original sapien.Entity implementation
            for component in self.actor.get_components():
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.mass = mass


class ArticulationActor:

    POINTS = {
        "contact": "contact_points",
        "target": "target_points",
        "functional": "functional_points",
        "orientation": "orientation_point",
    }

    def __init__(self, actor: Union[sapien.physx.PhysxArticulation, 'ManiSkillArticulation'],
                 actor_data: Dict, mass: float = 0.01):
        """
        Args:
            actor: sapien.physx.PhysxArticulation or ManiSkill3 Articulation instance
            actor_data: Dictionary containing point configurations
            mass: Mass to set for the articulation links
        """
        # Store the underlying articulation for compatibility
        self.actor = actor
        self.config = actor_data

        # Check if it's a ManiSkillArticulation
        if MANISKILL_AVAILABLE and isinstance(actor, ManiSkillArticulation):
            self._is_maniskill_articulation = True
            self._copy_maniskill_attributes(actor)
        else:
            # It's a sapien.physx.PhysxArticulation (original usage)
            # Validate it has the expected methods
            if not (hasattr(actor, 'get_links') and hasattr(actor, 'get_joints')):
                raise TypeError(f"ArticulationActor requires a PhysxArticulation, got {type(actor)}")
            self._is_maniskill_articulation = False

        # Build link dictionary
        self.link_dict = self.get_link_dict()
        self.set_mass(mass)

    def _copy_maniskill_attributes(self, ms_articulation):
        """Copy relevant attributes from ManiSkillArticulation"""
        self.scene = ms_articulation.scene if hasattr(ms_articulation, 'scene') else None
        self.device = ms_articulation.device if hasattr(ms_articulation, 'device') else None
        self._objs = ms_articulation._objs if hasattr(ms_articulation, '_objs') else [self.actor]
        self.links = ms_articulation.links if hasattr(ms_articulation, 'links') else None
        self.links_map = ms_articulation.links_map if hasattr(ms_articulation, 'links_map') else None
        self.joints = ms_articulation.joints if hasattr(ms_articulation, 'joints') else None

    def get_link_dict(self) -> Dict[str, Union[sapien.physx.PhysxArticulationLinkComponent, 'Link']]:
        """Build a dictionary mapping link names to link objects."""
        link_dict = {}

        if self._is_maniskill_articulation:
            # Use ManiSkill3 style links_map if available
            if hasattr(self, 'links_map') and self.links_map:
                return self.links_map

            # Otherwise build from links
            if hasattr(self, 'links') and self.links:
                for link in self.links:
                    if hasattr(link, 'name'):
                        link_dict[link.name] = link
                    else:
                        link_dict[link.get_name()] = link
        else:
            # Original sapien.physx.PhysxArticulation
            for link in self.actor.get_links():
                link_dict[link.get_name()] = link

        return link_dict

    def get_point(self, type: str, idx: int, ret: str = "list") -> Union[np.ndarray, List, sapien.Pose]:
        """Get the point of the articulation actor."""
        type_key = self.POINTS[type]
        local_matrix = np.array(self.config[type_key][idx]["matrix"])
        local_matrix[:3, 3] *= self.config["scale"]

        link_name = self.config[type_key][idx]["base"]
        link = self.link_dict.get(link_name)

        if link is None:
            raise ValueError(f"Link '{link_name}' not found in articulation")

        # Get link pose - handle both ManiSkill Link and sapien link
        if hasattr(link, 'pose'):
            link_pose = link.pose
            if hasattr(link_pose, 'sp'):
                link_matrix = link_pose.sp.to_transformation_matrix()
            else:
                link_matrix = link_pose.to_transformation_matrix()
        else:
            link_matrix = link.get_pose().to_transformation_matrix()

        world_matrix = link_matrix @ local_matrix

        if ret == "matrix":
            return world_matrix
        elif ret == "list":
            return (world_matrix[:3, 3].tolist() +
                    t3d.quaternions.mat2quat(world_matrix[:3, :3]).tolist())
        else:
            return sapien.Pose(world_matrix[:3, 3],
                               t3d.quaternions.mat2quat(world_matrix[:3, :3]))

    def set_mass(self, mass: float, links_name: Optional[List[str]] = None):
        """Set mass for articulation links."""
        if self._is_maniskill_articulation and hasattr(self, 'links') and self.links:
            # ManiSkill3 style
            for link in self.links:
                link_name = link.name if hasattr(link, 'name') else link.get_name()
                if links_name is None or link_name in links_name:
                    # Try to set mass through bodies
                    if hasattr(link, '_bodies'):
                        for body in link._bodies:
                            if hasattr(body, 'mass'):
                                body.mass = mass
                    elif hasattr(link, 'set_mass'):
                        link.set_mass(mass)
        else:
            # Original sapien.physx.PhysxArticulation
            for link in self.actor.get_links():
                if links_name is None or link.get_name() in links_name:
                    link.set_mass(mass)

    def set_properties(self, damping: float, stiffness: float,
                       friction: Optional[float] = None, force_limit: Optional[float] = None):
        """Set joint properties for the articulation."""
        if self._is_maniskill_articulation and hasattr(self, 'joints') and self.joints:
            # ManiSkill3 style joints
            for joint in self.joints:
                if hasattr(joint, 'set_drive_properties'):
                    if force_limit is not None:
                        joint.set_drive_properties(damping=damping, stiffness=stiffness,
                                                   force_limit=force_limit)
                    else:
                        joint.set_drive_properties(damping=damping, stiffness=stiffness)
                if friction is not None and hasattr(joint, 'set_friction'):
                    joint.set_friction(friction)
        else:
            # Original sapien.physx.PhysxArticulation
            for joint in self.actor.get_joints():
                if force_limit is not None:
                    joint.set_drive_properties(damping=damping, stiffness=stiffness,
                                               force_limit=force_limit)
                else:
                    joint.set_drive_properties(damping=damping, stiffness=stiffness)
                if friction is not None:
                    joint.set_friction(friction)

    def set_qpos(self, qpos: np.ndarray):
        """Set joint positions."""
        if self._is_maniskill_articulation and hasattr(self.actor, 'qpos'):
            # Use ManiSkill3 property
            self.actor.qpos = qpos
        else:
            # Use original method
            self.actor.set_qpos(qpos)

    def set_qvel(self, qvel: np.ndarray):
        """Set joint velocities."""
        if self._is_maniskill_articulation and hasattr(self.actor, 'qvel'):
            # Use ManiSkill3 property
            self.actor.qvel = qvel
        else:
            # Use original method
            self.actor.set_qvel(qvel)

    def get_qlimits(self) -> np.ndarray:
        """Get joint limits."""
        if self._is_maniskill_articulation and hasattr(self.actor, 'qlimits'):
            return self.actor.qlimits
        return self.actor.get_qlimits()

    def get_qpos(self) -> np.ndarray:
        """Get joint positions."""
        if self._is_maniskill_articulation and hasattr(self.actor, 'qpos'):
            return self.actor.qpos
        return self.actor.get_qpos()

    def get_qvel(self) -> np.ndarray:
        """Get joint velocities."""
        if self._is_maniskill_articulation and hasattr(self.actor, 'qvel'):
            return self.actor.qvel
        return self.actor.get_qvel()

    def get_pose(self) -> sapien.Pose:
        """Get the pose of the articulation (root link pose)."""
        if self._is_maniskill_articulation and hasattr(self.actor, 'pose'):
            # Use ManiSkill3 pose property
            pose = self.actor.pose
            if hasattr(pose, 'sp'):
                return pose.sp
            return pose
        # Use original method
        return self.actor.get_pose()

    def get_name(self) -> str:
        """Get the name of the articulation."""
        if self._is_maniskill_articulation and hasattr(self.actor, 'name'):
            return self.actor.name
        return self.actor.get_name()

    def set_name(self, name: str):
        """Set the name of the articulation."""
        if self._is_maniskill_articulation and hasattr(self.actor, 'name'):
            self.actor.name = name
        if hasattr(self.actor, 'set_name'):
            self.actor.set_name(name)