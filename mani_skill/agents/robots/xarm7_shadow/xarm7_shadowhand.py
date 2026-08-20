from copy import deepcopy
import sys
from pathlib import Path
from typing import Dict, List
import select
import numpy as np
import torch
import sapien
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose, vectorize_pose
from mani_skill.utils.structs.link import Link
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from . import components_name
import gymnasium as gym

@register_agent()
class XArm7ShadowHand(BaseAgent):
    uid = "xarm7_shadowhand"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xarm7/xarm7_shadowhand.urdf"
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            tip_link: dict(material="tip", patch_radius=0.1, min_patch_radius=0.1) for tip_link in components_name.dmp_links_names+['rh_palm', "rh_lfmetacarpal"]
        },
    )
    keyframes = dict(
        open=Keyframe(
            qpos=np.array([
                0.0, -0.4, 0.0, 1.0, 0.0, 0.9, -2.0,
                # ShadowHand joints (24)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0
            ]),
            pose=sapien.Pose(p=[-0.5, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])
        )
    )
    fix_root_link = True
    active_joints_name = components_name.hand_joints_name

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 2e2
        self.arm_force_limit = 500

        self.tip_link_names = components_name.useful_hand_link_names["tip"]
        self.palm_link_name = "rh_palm"
        self.forearm_link_name = "rh_forearm"
        self.need_force_links_name = [
            "rh_palm",
            *components_name.useful_hand_link_names["th"],
            *components_name.useful_hand_link_names["ff"],
            *components_name.useful_hand_link_names["mf"],
            *components_name.useful_hand_link_names["rf"],
            *components_name.useful_hand_link_names["lf"],
        ]

        super().__init__(*args, **kwargs)

    def _after_init(self):

        joint_limits = self.robot.get_qlimits()[0] 
        self.joint_limits_low = joint_limits[..., 0]
        self.joint_limits_high = joint_limits[..., 1]
        self.tip_links: List[Link] = sapien_utils.get_objs_by_names(self.robot.get_links(), self.tip_link_names)
        self.palm_link: Link = sapien_utils.get_obj_by_name(self.robot.get_links(), self.palm_link_name)
        pair_links_name = [tuple(pair) for pair in components_name.collision_pairs_xarm7]
        self.pair_links_name = list(set(pair_links_name))

        
    @property
    def _controller_configs(self):
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
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    

    @property
    def controller_joint_indices(self):
        """
        The controller joint sort is different from the robot_active_joint sort.
        Get the indices of the controller joint.
        This indices can map robot joints to controller joints. robot.joints[indices] will be the corresponding controller joints.
        e.g., call the corresponding controller qlimits by robot.qlimits[:,indices]. 
        """
        return self.controller.active_joint_indices

    
    @property
    def controller_joints(self):
        """
        Get the names of the controller joints.
        """
        return self.controller.joints
    

    def set_action(self, action) -> None:
        '''
        Actions can be defined externally through the joint order of the robot, but the order of the controller is different and needs to be converted.
        '''
        action = action[..., self.controller_joint_indices]
        super().set_action(action)


    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        obs.update(
            {
                "palm_pose": self.palm_pose,
                "tip_poses": self.tip_poses.reshape(-1, len(self.tip_links) * 7),
            }
        )

        return obs


    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger
        """
        tip_poses = [vectorize_pose(link.pose, device=self.device) for link in self.tip_links]
        return torch.stack(tip_poses, dim=-2)


    @property
    def palm_pose(self):
        """
        Get the palm pose for shadowhand
        """
        return vectorize_pose(self.palm_link.pose, device=self.device)
    
    def compute_contact_force_link_pairs(self) -> torch.Tensor:
        """
        Calculate the contact force between each pair of links by querying the contact impulse through GPU.
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
                cal_first_links.extend(links_map[link_pair[0]]._bodies)
                cal_second_links.extend(links_map[link_pair[1]]._bodies)
            self.query_link_pairs = self.scene.px.gpu_create_contact_pair_impulse_query(list(zip(cal_first_links, cal_second_links)))

        self.scene.px.gpu_query_contact_pair_impulses(self.query_link_pairs)
        contact_impulses = self.query_link_pairs.cuda_impulses.torch().clone().reshape(len(self.pair_links_name), -1, 3)
        contact_force = contact_impulses / self.scene.px.timestep
        return contact_force.permute(1, 0, 2).contiguous() # to (num_envs, num_links, 3)


    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold