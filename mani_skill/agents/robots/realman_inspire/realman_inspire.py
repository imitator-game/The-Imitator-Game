import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link


@register_agent()
class RealmanDexterous(BaseAgent):
    """Realman robot with dexterous hands"""
    uid = "realman_inspire"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/realman/inspire_urdf/realman.urdf"

    urdf_config = dict(
        _materials=dict(
            hand=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            # Left hand links - set materials for all finger links
            l_hand_base_link=dict(material="hand", patch_radius=0.1, min_patch_radius=0.1),
            left_thumb_1=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_thumb_2=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_thumb_3=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_thumb_4=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_index_1=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_index_2=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_middle_1=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_middle_2=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_ring_1=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_ring_2=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_little_1=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            left_little_2=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            # Right hand links
            r_hand_base_link=dict(material="hand", patch_radius=0.1, min_patch_radius=0.1),
            right_thumb_1=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_thumb_2=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_thumb_3=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_thumb_4=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_index_1=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_index_2=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_middle_1=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_middle_2=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_ring_1=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_ring_2=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_little_1=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
            right_little_2=dict(material="hand", patch_radius=0.05, min_patch_radius=0.05),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            qpos=np.array([
                # Head fixed to 0
                0.0, 0.0,  # head_joint1-2
                # Left arm: l_joint1-7
                -0.81842, -0.176488, 0, -1.37455, -0.875422, -0.997037, -1.54696,
                # Right arm: r_joint1-7
                0.82383, -0.17647, 0, -1.44114, -2.09, 0.926211, 1.55079,
                # Left dexterous hand (6 main control joints: thumb_1, thumb_2, index_1, middle_1, ring_1, little_1)
                0, 0, 0, 0, 0, 0,
                # Right dexterous hand (6 main control joints)
                0, 0, 0, 0, 0, 0,
            ])
        ),
        home=Keyframe(
            pose=sapien.Pose(),
            qpos=np.zeros(28)  # 2(head) + 7(left_arm) + 7(right_arm) + 6(left_hand) + 6(right_hand)
        )
    )

    def __init__(self, *args, **kwargs):
        # Left arm joints (7-DOF)
        self.left_arm_joint_names = [
            "l_joint1", "l_joint2", "l_joint3", "l_joint4",
            "l_joint5", "l_joint6", "l_joint7"
        ]
        self.left_arm_stiffness = 1e3
        self.left_arm_damping = 1e2
        self.left_arm_force_limit = 100

        # Right arm joints (7-DOF)
        self.right_arm_joint_names = [
            "r_joint1", "r_joint2", "r_joint3", "r_joint4",
            "r_joint5", "r_joint6", "r_joint7"
        ]
        self.right_arm_stiffness = 1e3
        self.right_arm_damping = 1e2
        self.right_arm_force_limit = 100

        # Head joints
        self.head_joint_names = ["head_joint1", "head_joint2"]
        self.head_stiffness = 1e3
        self.head_damping = 1e2
        self.head_force_limit = 50

        # Left dexterous hand joints (only control the main joints, mimic joints follow automatically)
        self.left_hand_joint_names = [
            "left_thumb_1_joint",  # Thumb first joint
            "left_thumb_2_joint",  # Thumb second joint (other joints follow via mimic)
            "left_index_1_joint",  # Index finger (second joint follows via mimic)
            "left_middle_1_joint",  # Middle finger
            "left_ring_1_joint",  # Ring finger
            "left_little_1_joint",  # Little finger
        ]

        # Right dexterous hand joints
        self.right_hand_joint_names = [
            "right_thumb_1_joint",
            "right_thumb_2_joint",
            "right_index_1_joint",
            "right_middle_1_joint",
            "right_ring_1_joint",
            "right_little_1_joint",
        ]

        self.hand_stiffness = 1e3
        self.hand_damping = 1e2
        self.hand_force_limit = 20

        # End effector links
        self.left_ee_link_name = "l_link7"
        self.right_ee_link_name = "r_link7"

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        """Configure the controller - head + two arms + two hands"""

        # Build the complete joint list (keep the interleaved order)
        all_joints = []
        all_stiffness = []
        all_damping = []
        all_force_limit = []

        # 1. Head joints
        all_joints.extend(self.head_joint_names)
        all_stiffness.extend([self.head_stiffness] * 2)
        all_damping.extend([self.head_damping] * 2)
        all_force_limit.extend([self.head_force_limit] * 2)

        # 2. Interleaved arm joints
        for i in range(7):
            all_joints.append(self.right_arm_joint_names[i])
            all_joints.append(self.left_arm_joint_names[i])
            all_stiffness.extend([self.right_arm_stiffness, self.left_arm_stiffness])
            all_damping.extend([self.right_arm_damping, self.left_arm_damping])
            all_force_limit.extend([self.right_arm_force_limit, self.left_arm_force_limit])

        # Body controller (head + two arms)
        body_controller = PDJointPosControllerConfig(
            all_joints,
            None,
            None,
            all_stiffness,
            all_damping,
            all_force_limit,
            normalize_action=False,
        )

        # Left hand controller
        left_hand_controller = PDJointPosControllerConfig(
            self.left_hand_joint_names,
            lower=[0.0] * 6,  # 6 main control joints
            upper=[1.15, 0.55, 1.6, 1.6, 1.6, 1.6],  # Based on URDF limits
            stiffness=[self.hand_stiffness] * 6,
            damping=[self.hand_damping] * 6,
            force_limit=[self.hand_force_limit] * 6,
            normalize_action=False,
        )

        # Right hand controller
        right_hand_controller = PDJointPosControllerConfig(
            self.right_hand_joint_names,
            lower=[0.0] * 6,
            upper=[1.15, 0.55, 1.6, 1.6, 1.6, 1.6],
            stiffness=[self.hand_stiffness] * 6,
            damping=[self.hand_damping] * 6,
            force_limit=[self.hand_force_limit] * 6,
            normalize_action=False,
        )

        controller_configs = dict(
            pd_joint_pos=dict(
                body=body_controller,
                left_hand=left_hand_controller,
                right_hand=right_hand_controller,
            ),
            pd_joint_pos_vel=dict(
                body=PDJointPosVelControllerConfig(
                    all_joints,
                    None,
                    None,
                    all_stiffness,
                    all_damping,
                    all_force_limit,
                    normalize_action=False,
                ),
                left_hand=left_hand_controller,
                right_hand=right_hand_controller,
            ),
        )

        return controller_configs

    def _after_init(self):
        """Fetch important links after initialization"""
        # Base links
        self.base_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "body_base_link"
        )

        # Left arm end effector
        self.left_tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.left_ee_link_name
        )

        # Right arm end effector
        self.right_tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.right_ee_link_name
        )

        # Left hand links
        self.left_hand_base = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_hand_base_link"
        )
        self.left_hand_links = []
        for finger in ["thumb", "index", "middle", "ring", "little"]:
            for i in [1, 2] if finger != "thumb" else [1, 2, 3, 4]:
                link_name = f"left_{finger}_{i}"
                link = sapien_utils.get_obj_by_name(self.robot.get_links(), link_name)
                if link is not None:
                    self.left_hand_links.append(link)

        # Right hand links
        self.right_hand_base = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_hand_base_link"
        )
        self.right_hand_links = []
        for finger in ["thumb", "index", "middle", "ring", "little"]:
            for i in [1, 2] if finger != "thumb" else [1, 2, 3, 4]:
                link_name = f"right_{finger}_{i}"
                link = sapien_utils.get_obj_by_name(self.robot.get_links(), link_name)
                if link is not None:
                    self.right_hand_links.append(link)

        # Head links
        self.head_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "head_link2"
        )
        self.camera_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "camera_link"
        )

    def is_left_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the left hand is grasping an object"""
        contact_forces = []
        for link in self.left_hand_links:
            forces = self.scene.get_pairwise_contact_forces(link, object)
            contact_forces.append(forces)

        # Check if at least 3 finger links have sufficient contact force
        num_contacts = 0
        for forces in contact_forces:
            force_magnitude = torch.linalg.norm(forces, axis=1)
            if torch.any(force_magnitude >= min_force):
                num_contacts += 1

        return torch.tensor([num_contacts >= 3])

    def is_right_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the right hand is grasping an object"""
        contact_forces = []
        for link in self.right_hand_links:
            forces = self.scene.get_pairwise_contact_forces(link, object)
            contact_forces.append(forces)

        num_contacts = 0
        for forces in contact_forces:
            force_magnitude = torch.linalg.norm(forces, axis=1)
            if torch.any(force_magnitude >= min_force):
                num_contacts += 1

        return torch.tensor([num_contacts >= 3])

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if either hand is grasping an object"""
        return self.is_left_grasping(object, min_force, max_angle) | \
            self.is_right_grasping(object, min_force, max_angle)

    def is_static(self, threshold: float = 0.2):
        """Check if the robot arms are static"""
        left_arm_qvel = self.robot.get_qvel()[..., 2:9]  # Skip the head
        right_arm_qvel = self.robot.get_qvel()[..., 9:16]
        return torch.all(torch.abs(left_arm_qvel) <= threshold, dim=1) & \
            torch.all(torch.abs(right_arm_qvel) <= threshold, dim=1)

    @property
    def left_tcp_pose(self) -> Pose:
        """Get the left arm TCP pose"""
        return self.left_tcp.pose

    @property
    def right_tcp_pose(self) -> Pose:
        """Get the right arm TCP pose"""
        return self.right_tcp.pose

    def get_left_hand_qpos(self):
        """Get the left hand joint positions (6 main control joints)"""
        qpos = self.robot.get_qpos()
        # Extract according to the actual joint order
        hand_indices = [16, 17, 18, 19, 20, 21]  # head(2) + two arms(14) = starts at 16
        return qpos[..., hand_indices]

    def get_right_hand_qpos(self):
        """Get the right hand joint positions (6 main control joints)"""
        qpos = self.robot.get_qpos()
        hand_indices = [22, 23, 24, 25, 26, 27]  # After the left hand
        return qpos[..., hand_indices]


@register_agent()
class RealmanDexterousMobileBase(RealmanDexterous):
    """Realman dexterous robot with mobile base"""
    uid = "realman_inspire_mobile_base"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/realman/inspire_urdf/realman_mobile_base.urdf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Mobile base joints
        self.base_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint"
        ]
        self.base_stiffness = 1e3
        self.base_damping = 1e2
        self.base_force_limit = 500

    @property
    def _controller_configs(self):
        """Configure the controller to include the mobile base"""
        parent_configs = super()._controller_configs

        # Create a new config that includes the base
        pd_joint_pos_with_base = dict()

        # Base velocity controller
        base_pd_joint_vel = PDBaseForwardVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -3.14],  # [linear_vel, angular_vel]
            upper=[1, 3.14],
            damping=1000,
            force_limit=500,
        )

        pd_joint_pos_with_base["body"] = parent_configs["pd_joint_pos"]["body"]
        pd_joint_pos_with_base["left_hand"] = parent_configs["pd_joint_pos"]["left_hand"]
        pd_joint_pos_with_base["right_hand"] = parent_configs["pd_joint_pos"]["right_hand"]
        pd_joint_pos_with_base["base"] = base_pd_joint_vel

        # Add to the config
        controller_configs = parent_configs.copy()
        controller_configs["pd_joint_pos_with_base"] = pd_joint_pos_with_base

        return controller_configs