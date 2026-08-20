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
class Realman(BaseAgent):
    uid = "realman"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/realman/gripper_urdf/realman.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            # Left gripper links
            l_gripper_link1=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            l_gripper_link2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            l_gripper_link3=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            l_gripper_link4=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            # Right gripper links
            r_gripper_link1=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            r_gripper_link2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            r_gripper_link3=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            r_gripper_link4=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
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
                # Grippers
                0, 0
            ])
        ),
        home=Keyframe(
            pose=sapien.Pose(),
            qpos=np.zeros(18)  # All joints at zero position
        )
    )

    def __init__(self, *args, **kwargs):
        # Left arm joints
        self.left_arm_joint_names = [
            "l_joint1", "l_joint2", "l_joint3", "l_joint4",
            "l_joint5", "l_joint6", "l_joint7"
        ]
        self.left_arm_stiffness = 1e3
        self.left_arm_damping = 1e2
        self.left_arm_force_limit = 100

        # Right arm joints
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

        # Gripper joints (only the main control joints, not mimic joints)
        self.left_gripper_joint_names = ["l_gripper_joint1"]
        self.right_gripper_joint_names = ["r_gripper_joint1"]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 50

        # End effector links
        self.left_ee_link_name = "l_link7"
        self.right_ee_link_name = "r_link7"

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # Create independent stiffness and damping parameters for each part
        arm_stiffness = self.left_arm_stiffness
        arm_damping = self.left_arm_damping
        arm_force_limit = self.left_arm_force_limit

        # Create grouped joint lists
        left_arm_joints = self.left_arm_joint_names
        right_arm_joints = self.right_arm_joint_names

        # Get all active joints (including head)
        all_joints = []
        all_stiffness = []
        all_damping = []
        all_force_limit = []

        # Head joints
        all_joints.extend(self.head_joint_names)
        all_stiffness.extend([self.head_stiffness] * 2)
        all_damping.extend([self.head_damping] * 2)
        all_force_limit.extend([self.head_force_limit] * 2)

        # Interleaved arm joints (following the actual order in the URDF)
        for i in range(7):
            all_joints.append(self.right_arm_joint_names[i])
            all_joints.append(self.left_arm_joint_names[i])
            all_stiffness.extend([self.right_arm_stiffness, self.left_arm_stiffness])
            all_damping.extend([self.right_arm_damping, self.left_arm_damping])
            all_force_limit.extend([self.right_arm_force_limit, self.left_arm_force_limit])

        pd_joint_pos = PDJointPosControllerConfig(
            all_joints,
            None,
            None,
            all_stiffness,
            all_damping,
            all_force_limit,
            normalize_action=False,
        )

        # Left gripper
        left_gripper = PDJointPosMimicControllerConfig(
            self.left_gripper_joint_names,
            -0.01,  # Allow a small negative value to ensure full closure
            1.0,  # Changed to 1.0 to match the URDF
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )

        # Right gripper
        right_gripper = PDJointPosMimicControllerConfig(
            self.right_gripper_joint_names,
            -0.01,
            1.0,  # Changed to 1.0 to match the URDF
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )

        controller_configs = dict(
            pd_joint_pos=dict(
                body=pd_joint_pos,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
            ),
            pd_joint_pos_vel=dict(
                body=PDJointPosVelControllerConfig(
                    [*left_arm_joints, *right_arm_joints],
                    None,
                    None,
                    [arm_stiffness] * 14,
                    [arm_damping] * 14,
                    [arm_force_limit] * 14,
                    normalize_action=False,
                ),
                left_gripper=left_gripper,
                right_gripper=right_gripper,
            ),
        )

        return controller_configs

    # @property
    # def _sensor_configs(self):
    #     return [
    #         # Head camera (main camera on the head)
    #         CameraConfig(
    #             uid="head_camera",
    #             pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
    #             width=128,
    #             height=128,
    #             fov=np.pi / 2,
    #             near=0.01,
    #             far=100,
    #             entity_uid="camera_link",
    #         ),
    #         # Left arm camera
    #         CameraConfig(
    #             uid="left_camera",
    #             pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
    #             width=128,
    #             height=128,
    #             fov=np.pi / 2,
    #             near=0.01,
    #             far=100,
    #             entity_uid="left_camera_optical",
    #         ),
    #         # Right arm camera
    #         CameraConfig(
    #             uid="right_camera",
    #             pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),
    #             width=128,
    #             height=128,
    #             fov=np.pi / 2,
    #             near=0.01,
    #             far=100,
    #             entity_uid="right_camera_optical",
    #         ),
    #     ]

    def _after_init(self):
        # Get important links
        self.base_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link"
        )
        self.body_base_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "body_base_link"
        )

        # Left arm end effector and gripper links
        self.left_tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.left_ee_link_name
        )
        self.left_gripper_links = [
            sapien_utils.get_obj_by_name(self.robot.get_links(), f"l_gripper_link{i}")
            for i in range(1, 5)
        ]

        # Right arm end effector and gripper links
        self.right_tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.right_ee_link_name
        )
        self.right_gripper_links = [
            sapien_utils.get_obj_by_name(self.robot.get_links(), f"r_gripper_link{i}")
            for i in range(1, 5)
        ]

        # Head links
        self.head_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "head_link2"
        )
        self.camera_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "camera_link"
        )

    def is_left_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the left gripper is grasping an object"""
        contact_forces = []
        for link in self.left_gripper_links:
            forces = self.scene.get_pairwise_contact_forces(link, object)
            contact_forces.append(forces)

        # Check if at least two gripper links have sufficient contact
        num_contacts = 0
        for forces in contact_forces:
            force_magnitude = torch.linalg.norm(forces, axis=1)
            if torch.any(force_magnitude >= min_force):
                num_contacts += 1

        return torch.tensor([num_contacts >= 2])

    def is_right_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the right gripper is grasping an object"""
        contact_forces = []
        for link in self.right_gripper_links:
            forces = self.scene.get_pairwise_contact_forces(link, object)
            contact_forces.append(forces)

        # Check if at least two gripper links have sufficient contact
        num_contacts = 0
        for forces in contact_forces:
            force_magnitude = torch.linalg.norm(forces, axis=1)
            if torch.any(force_magnitude >= min_force):
                num_contacts += 1

        return torch.tensor([num_contacts >= 2])

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if either gripper is grasping an object"""
        return self.is_left_grasping(object, min_force, max_angle) | \
            self.is_right_grasping(object, min_force, max_angle)

    def is_static(self, threshold: float = 0.2):
        """Check if the robot arms are static (not moving significantly)"""
        left_arm_qvel = self.robot.get_qvel()[..., :7]
        right_arm_qvel = self.robot.get_qvel()[..., 7:14]
        return torch.all(torch.abs(left_arm_qvel) <= threshold, dim=1) & \
            torch.all(torch.abs(right_arm_qvel) <= threshold, dim=1)

    @property
    def left_tcp_pose(self) -> Pose:
        """Get the pose of the left arm TCP (Tool Center Point)"""
        return self.left_tcp.pose

    @property
    def right_tcp_pose(self) -> Pose:
        """Get the pose of the right arm TCP (Tool Center Point)"""
        return self.right_tcp.pose

    def get_left_gripper_openness(self):
        """Get the openness of the left gripper (0=closed, 1=open)"""
        qpos = self.robot.get_qpos()
        left_gripper_idx = self.robot.active_joints_map["l_gripper_joint1"].active_index[0]
        return qpos[..., left_gripper_idx]

    def get_right_gripper_openness(self):
        """Get the openness of the right gripper (0=closed, 1=open)"""
        qpos = self.robot.get_qpos()
        right_gripper_idx = self.robot.active_joints_map["r_gripper_joint1"].active_index[0]
        return qpos[..., right_gripper_idx]


@register_agent()
class RealmanMobileBase(Realman):
    """Realman robot with mobile base control"""
    uid = "realman_mobile_base"

    # Use the new URDF that includes mobile base joints
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/realman/gripper_urdf/realman_mobile_base.urdf"

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
        # Get the controller configs from the parent class
        parent_configs = super()._controller_configs

        # Create a new config that includes the base
        pd_joint_pos_with_base = dict()

        # Merge all arm joints (keep the interleaved order)
        all_arm_joints = []
        for i in range(7):
            all_arm_joints.append(self.right_arm_joint_names[i])  # r_joint[i+1]
            all_arm_joints.append(self.left_arm_joint_names[i])  # l_joint[i+1]

        # Body controller (includes head and arms)
        body_joints = self.head_joint_names + all_arm_joints
        body_pd_joint_pos = PDJointPosControllerConfig(
            body_joints,
            None,
            None,
            [self.head_stiffness] * 2 + [self.left_arm_stiffness] * 14,
            [self.head_damping] * 2 + [self.left_arm_damping] * 14,
            [self.head_force_limit] * 2 + [self.left_arm_force_limit] * 14,
            normalize_action=False,
        )

        # Base velocity controller
        self.base_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint"
        ]
        base_pd_joint_vel = PDBaseForwardVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -3.14],  # [linear_vel, angular_vel]
            upper=[1, 3.14],
            damping=1000,
            force_limit=500,
        )

        pd_joint_pos_with_base["body"] = body_pd_joint_pos
        pd_joint_pos_with_base["left_gripper"] = parent_configs["pd_joint_pos"]["left_gripper"]
        pd_joint_pos_with_base["right_gripper"] = parent_configs["pd_joint_pos"]["right_gripper"]
        pd_joint_pos_with_base["base"] = base_pd_joint_vel

        # Add to the config
        controller_configs = parent_configs.copy()
        controller_configs["pd_joint_pos_with_base"] = pd_joint_pos_with_base

        return controller_configs