"""
Minimal robot agent template: import a custom robot from a URDF.

To use your own robot:
  1. Put your URDF + meshes under:  mani_skill/assets/robots/your_robot/
  2. Copy this file to:             mani_skill/agents/robots/your_robot/your_robot.py
  3. Edit ONLY the fields below (uid, urdf_path, keyframes, joint/link names,
     gripper range). The controller setup and helper methods can stay as-is.
"""

from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    PDJointPosMimicControllerConfig,
    PDEEPoseControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


@register_agent()
class YourRobot(BaseAgent):
    # ------------------------------------------------------------------ #
    # Robot identity and asset path (EDIT ME)
    # ------------------------------------------------------------------ #
    uid = "your_robot_uid"  # name used in gym.make(..., robot_uids="your_robot_uid")

    # Put your URDF under mani_skill/assets/robots/your_robot/your_robot.urdf
    urdf_path = f"{ASSET_DIR}/robots/your_robot/your_robot.urdf"

    # Default pose and joint configuration (EDIT ME)
    # qpos order must match the robot's active-joint order (arm then gripper).
    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0]),
            qpos=np.array(
                [
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # arm joints
                    0.04, 0.04,                         # gripper joints
                ]
            ),
        )
    )

    # Joint / link names — must exactly match the URDF (EDIT ME)
    arm_joint_names = [
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    ]
    gripper_joint_names = [
        "left_finger_joint", "right_finger_joint",
    ]
    ee_link_name = "tcp_link"                    # end-effector / TCP link
    finger_link_names = (                        # used by is_grasping()
        "left_finger_link",
        "right_finger_link",
    )
    # No gripper? use gripper_joint_names=[] and finger_link_names=(None, None)

    # ------------------------------------------------------------------ #
    # Controller parameters (EDIT ME: match your URDF limits / stiffness)
    # ------------------------------------------------------------------ #
    arm_stiffness = 1000
    arm_damping = 100
    arm_force_limit = 100
    gripper_stiffness = 1000
    gripper_damping = 100
    gripper_force_limit = 100
    gripper_lower = 0.0       # gripper action range (URDF joint limits)
    gripper_upper = 0.04
    use_mimic_gripper = True  # False if each finger joint is driven independently

    # ------------------------------------------------------------------ #
    # Controller set: exposes pd_joint_pos / pd_joint_delta_pos / pd_ee_delta_pose
    # ------------------------------------------------------------------ #
    @property
    def _controller_configs(self):
        arm_pos = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names, lower=None, upper=None,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, normalize_action=False)
        arm_delta = PDJointPosControllerConfig(
            joint_names=self.arm_joint_names, lower=-0.1, upper=0.1,
            stiffness=self.arm_stiffness, damping=self.arm_damping,
            force_limit=self.arm_force_limit, use_delta=True)
        arm_ee = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1,
            rot_lower=-0.1, rot_upper=0.1, stiffness=self.arm_stiffness,
            damping=self.arm_damping, force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name, urdf_path=self.urdf_path)

        if len(self.gripper_joint_names) > 0:
            if self.use_mimic_gripper:
                gripper = PDJointPosMimicControllerConfig(
                    joint_names=self.gripper_joint_names,
                    lower=self.gripper_lower, upper=self.gripper_upper,
                    stiffness=self.gripper_stiffness, damping=self.gripper_damping,
                    force_limit=self.gripper_force_limit)
            else:
                gripper = PDJointPosControllerConfig(
                    joint_names=self.gripper_joint_names,
                    lower=self.gripper_lower, upper=self.gripper_upper,
                    stiffness=self.gripper_stiffness, damping=self.gripper_damping,
                    force_limit=self.gripper_force_limit)

            def with_gripper(cfg):
                return dict(arm=cfg, gripper=gripper)
        else:

            def with_gripper(cfg):
                return dict(arm=cfg)

        return deepcopy(dict(pd_joint_pos=with_gripper(arm_pos),
                             pd_joint_delta_pos=with_gripper(arm_delta),
                             pd_ee_delta_pose=with_gripper(arm_ee)))

    def _after_init(self):
        """Cache the TCP and finger links after the articulation is loaded."""
        self.tcp = self.robot.links_map[self.ee_link_name]
        left, right = self.finger_link_names
        self.finger1_link = self.robot.links_map[left] if left else None
        self.finger2_link = self.robot.links_map[right] if right else None

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    def is_grasping(self, obj: Actor, min_force=0.5, max_angle=85):
        """Two-finger grasp check using contact forces (needs both finger links)."""
        if self.finger1_link is None or self.finger2_link is None:
            raise RuntimeError("is_grasping() requires two finger links.")
        left_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, obj)
        right_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, obj)
        left_norm = torch.linalg.norm(left_forces, dim=1)
        right_norm = torch.linalg.norm(right_forces, dim=1)
        left_mat = self.finger1_link.pose.to_transformation_matrix()
        right_mat = self.finger2_link.pose.to_transformation_matrix()
        # Assumes fingers close along local +Y / -Y. Adjust axis if needed.
        left_dir = left_mat[..., :3, 1]
        right_dir = -right_mat[..., :3, 1]
        left_ok = torch.logical_and(
            left_norm >= min_force,
            torch.rad2deg(common.compute_angle_between(left_dir, left_forces)) <= max_angle)
        right_ok = torch.logical_and(
            right_norm >= min_force,
            torch.rad2deg(common.compute_angle_between(right_dir, right_forces)) <= max_angle)
        return torch.logical_and(left_ok, right_ok)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., : len(self.arm_joint_names)]
        return torch.max(torch.abs(qvel), dim=1)[0] <= threshold
