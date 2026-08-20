from copy import deepcopy
from typing import List
import numpy as np
import torch
import sapien
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.structs.pose import Pose, vectorize_pose
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.actor import Actor
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.ur10e_shadow import components_name


def deepcopy_dict(configs):
    """Deep copy dictionary for controller configs"""
    return {k: deepcopy(v) for k, v in configs.items()}


@register_agent()
class UR10eShadowHand(BaseAgent):
    uid = "ur10e_shadow"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/ur10e_shadow/ur10e_shadow.urdf"
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            tip_link: dict(material="tip", patch_radius=0.1, min_patch_radius=0.1)
            for tip_link in components_name.dmp_links_names + ['rh_palm', "rh_lfmetacarpal"]
        },
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([
                # UR10e joints (6)
                0.0, -1.5708, 2.3562, 2.3562, -1.5708, -3.1416,
                # ShadowHand joints (24)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ]),
            pose=sapien.Pose(p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])
        ),
        open=Keyframe(
            qpos=np.array([
                # UR10e joints (6)
                0.0, -1.5708, 2.3562, 2.3562, -1.5708, -3.1416,
                # ShadowHand joints - open position
                0.0, 0.0, 0.0, 0.0,  # FFJ
                0.0, 0.0, 0.0, 0.0,  # MFJ
                0.0, 0.0, 0.0, 0.0,  # RFJ
                0.0, 0.0, 0.0, 0.0, 0.0,  # LFJ
                -0.5, 0.8, -0.2, 0.0, 0.0, 0.0, 0.0  # THJ - thumb open position
            ]),
            pose=sapien.Pose(p=[0.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])
        )
    )

    fix_root_link = True
    active_joints_name = components_name.hand_joints_name
    ee_link_name = "tool0"  # End effector is tool0 for UR10e

    def __init__(self, *args, **kwargs):
        # UR10e arm joints
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # UR10e control parameters
        self.arm_stiffness = 1e3
        self.arm_damping = 2e2
        self.arm_force_limit = 330  # Based on UR10e specifications

        # ShadowHand related configurations
        self.tip_link_names = components_name.useful_hand_link_names.get("tip", [])
        self.palm_link_name = "rh_palm"
        self.forearm_link_name = "rh_forearm"

        # Links that need force computation
        self.need_force_links_name = [
            "rh_palm",
            *components_name.useful_hand_link_names.get("th", []),
            *components_name.useful_hand_link_names.get("ff", []),
            *components_name.useful_hand_link_names.get("mf", []),
            *components_name.useful_hand_link_names.get("rf", []),
            *components_name.useful_hand_link_names.get("lf", []),
        ]

        super().__init__(*args, **kwargs)

    def _after_init(self):
        """Initialize after robot is loaded"""
        # Get joint limits
        joint_limits = self.robot.get_qlimits()[0]
        self.joint_limits_low = joint_limits[..., 0]
        self.joint_limits_high = joint_limits[..., 1]

        # Get important links
        self.tip_links: List[Link] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.palm_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name
        )

        # Get TCP link (tool0 for UR10e)
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        # Setup collision pairs for UR10e and ShadowHand
        pair_links_name = []
        if hasattr(components_name, 'collision_pairs_ur10e'):
            pair_links_name = [tuple(pair) for pair in components_name.collision_pairs_ur10e]

        # Add collision pairs between UR10e and ShadowHand
        ur10e_shadow_pairs = [
            ('tool0', 'rh_forearm'),
            ('tool0', 'rh_wrist'),
            ('tool0', 'rh_palm'),
            ('wrist_3_link', 'rh_forearm'),
            ('wrist_3_link', 'rh_wrist'),
            ('wrist_3_link', 'rh_palm'),
            ('wrist_2_link', 'rh_forearm'),
        ]
        pair_links_name.extend(ur10e_shadow_pairs)
        self.pair_links_name = list(set(pair_links_name))

        # Get all finger links for grasping check
        self.finger_links = []
        for finger in ["ff", "mf", "rf", "lf", "th"]:
            if finger in components_name.useful_hand_link_names:
                finger_link_names = components_name.useful_hand_link_names[finger]
                self.finger_links.extend(
                    sapien_utils.get_objs_by_names(self.robot.get_links(), finger_link_names)
                )

    @property
    def _controller_configs(self):
        """Configure controllers for UR10e arm and ShadowHand"""
        # UR10e arm controller
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # ShadowHand controller configurations
        self.stiffness = 1e3
        self.damping = 1e1
        joints_part_with_force_3e1 = ["rh_THJ4", "rh_THJ5", "rh_WRJ1", "rh_WRJ2"]
        joints_part_with_force_1e1 = list(set(self.active_joints_name) - set(joints_part_with_force_3e1))

        pd_joint_pos_3e1 = PDJointPosControllerConfig(
            joints_part_with_force_3e1,
            lower=None,
            upper=None,
            stiffness=self.stiffness,
            damping=self.damping,
            force_limit=3e1,
            normalize_action=False,
        )
        pd_joint_pos_1e1 = PDJointPosControllerConfig(
            joints_part_with_force_1e1,
            lower=None,
            upper=None,
            stiffness=self.stiffness,
            damping=self.damping,
            force_limit=1e1,
            normalize_action=False,
        )

        pd_joint_delta_pos_3e1 = PDJointPosControllerConfig(
            joints_part_with_force_3e1,
            -0.1,
            0.1,
            stiffness=self.stiffness,
            damping=self.damping,
            force_limit=3e1,
            use_delta=True,
        )
        pd_joint_delta_pos_1e1 = PDJointPosControllerConfig(
            joints_part_with_force_1e1,
            -0.1,
            0.1,
            stiffness=self.stiffness,
            damping=self.damping,
            force_limit=1e1,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                joint_3e1=pd_joint_pos_3e1,
                joint_1e1=pd_joint_pos_1e1,
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                joint_3e1=pd_joint_delta_pos_3e1,
                joint_1e1=pd_joint_delta_pos_1e1,
            ),
        )

        return deepcopy_dict(controller_configs)

    @property
    def tcp_pose(self):
        """Get TCP (Tool Center Point) pose"""
        return self.tcp.pose

    @property
    def controller_joint_indices(self):
        """Controller joint index mapping"""
        return self.controller.active_joint_indices

    @property
    def controller_joints(self):
        """Get controller joint names"""
        return self.controller.joints

    def set_action(self, action) -> None:
        """Set action, need to reorder joints"""
        action = action[..., self.controller_joint_indices]
        super().set_action(action)

    def get_proprioception(self):
        """Get proprioception state"""
        obs = super().get_proprioception()
        obs.update(
            {
                "palm_pose": self.palm_pose,
                "tip_poses": self.tip_poses.reshape(-1, len(self.tip_links) * 7),
                "tcp_pose": vectorize_pose(self.tcp_pose, device=self.device),
            }
        )
        return obs

    @property
    def tip_poses(self):
        """Get fingertip poses"""
        tip_poses = [vectorize_pose(link.pose, device=self.device) for link in self.tip_links]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_pose(self):
        """Get palm pose"""
        return vectorize_pose(self.palm_link.pose, device=self.device)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):

        return True

    def is_static(self, threshold: float = 0.2):
        """
        Check if the robot is static.

        Args:
            threshold (float): Velocity threshold

        Returns:
            torch.Tensor: Boolean tensor indicating if robot is static
        """
        # Get joint velocities for all joints
        qvel = self.robot.get_qvel()
        # Check if maximum absolute velocity is below threshold
        return torch.max(torch.abs(qvel), dim=1)[0] <= threshold

    def compute_contact_force_link_pairs(self) -> torch.Tensor:
        """
        Compute contact forces between link pairs.

        Returns:
            torch.Tensor: Contact forces between specified link pairs
        """
        if isinstance(self.device, str):
            if self.device != "cuda":
                raise RuntimeError("Contact force calculation only supports GPU mode.")
        elif isinstance(self.device, torch.device):
            if self.device.type != "cuda":
                raise RuntimeError("Contact force calculation only supports GPU mode.")

        if getattr(self, "query_link_pairs", None) is None:
            links_map = self.robot.links_map
            cal_first_links, cal_second_links = [], []
            for link_pair in self.pair_links_name:
                if link_pair[0] in links_map and link_pair[1] in links_map:
                    cal_first_links.extend(links_map[link_pair[0]]._bodies)
                    cal_second_links.extend(links_map[link_pair[1]]._bodies)
            if cal_first_links and cal_second_links:
                self.query_link_pairs = self.scene.px.gpu_create_contact_pair_impulse_query(
                    list(zip(cal_first_links, cal_second_links))
                )

        if self.query_link_pairs is not None:
            self.scene.px.gpu_query_contact_pair_impulses(self.query_link_pairs)
            contact_impulses = self.query_link_pairs.cuda_impulses.torch().clone().reshape(
                len(self.pair_links_name), -1, 3
            )
            contact_force = contact_impulses / self.scene.px.timestep
            return contact_force.permute(1, 0, 2).contiguous()
        else:
            # Return zero forces if no valid link pairs
            return torch.zeros(
                (self.scene.n_envs, len(self.pair_links_name), 3),
                device=self.device
            )