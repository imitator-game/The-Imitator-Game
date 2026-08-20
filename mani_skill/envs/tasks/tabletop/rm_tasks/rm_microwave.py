from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
import trimesh
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Realman, RealmanMobileBase
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.actors.ycb import food_and_beverage, kitchen_items
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.structs import Pose, Articulation, Link
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.sim_assets_catigory import ROBOTWIN_CATEGORIES

PARTNET_COLLISION_BIT = 29

@register_env("RealmanMicrowave-v1", max_episode_steps=1000)
class RealmanMicrowaveEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["realman", "realman_mobile_base"]
    microwave_list = [
        "7349"
    ]

    random_thresh = 0.02
    goal_thresh = 0.06
    max_close_frac = 0.25

    articulation_types = ["revolute_unwrapped"]
    agent: Union[Realman, RealmanMobileBase]

    def __init__(
            self,
            *args,
            robot_uids="realman",
            robot_init_qpos_noise=0.04,
            num_envs=1,
            reconfiguration_freq=None,
            **kwargs,
    ):
        self.JSON = (
                PACKAGE_ASSET_DIR / "partnet_mobility/meta/partnet_info/info_microwave.json"
        )
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.JSON)
        self.partnet_model_ids = np.array(list(train_data.keys()))
        self.microwave_model_ids = np.array(self.microwave_list)

        # Filter YCB objects suitable for dual-arm manipulation
        self.ycb_model_ids = np.array([
            k for k in load_json(
                ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
            ).keys()
        ])

        # For demo purposes, use a subset of objects
        self.ycb_model_ids = np.array(
            food_and_beverage['cans'] + food_and_beverage['boxes'] + food_and_beverage['condiments'] +
            food_and_beverage['fruits'] + kitchen_items['dishes'] + kitchen_items['utensils']
        )
        self.robotwin_model_ids = np.array(
            ROBOTWIN_CATEGORIES['food_and_beverage']['containers']
        )

        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2 ** 20, max_rigid_patch_count=2 ** 19
            )
        )

    @property
    def _default_sensor_configs(self):
        # Multiple camera views for better perception
        pose1 = sapien_utils.look_at(eye=[0.5, 0.5, 0.8], target=[0.0, 0, 0.2])
        pose2 = sapien_utils.look_at(eye=[0.5, 0, 0.8], target=[0.0, 0, 0.2])
        pose3 = sapien_utils.look_at(eye=[0.5, -0.5, 0.8], target=[0.0, 0, 0.2])
        pose4 = sapien_utils.look_at(eye=[-0.5, 0, 0.8], target=[0.0, 0, 0.2])

        return [
            CameraConfig("base_camera_left", pose1, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("base_camera_front", pose2, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("base_camera_right", pose3, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("base_camera_back", pose4, 128, 128, np.pi / 2, 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-1.0, 0.0, 0.8], [0.0, 0.0, 0.0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # Realman robot base position
        super()._load_agent(options, sapien.Pose(
            p=[-1.0, 0, -0.65],
            q=euler2quat(0.0, 0.0, np.pi / 2)
        ))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build(random_background=False)

        # Load partnet objects
        self._microwaves: List[Actor] = []
        self.partnet_links: List[List[Link]] = []
        self.partnet_links_meshes: List[List[trimesh.Trimesh]] = []
        microwave_model_ids = self._batched_episode_rng.choice(self.microwave_model_ids)
        microwave_link_ids = self._batched_episode_rng.randint(0, 2 ** 31)
        for i, model_id in enumerate(microwave_model_ids):
            assert model_id in self.partnet_model_ids, f"model id: {model_id} is not in available ids: {self.partnet_model_ids}."
            self.origin_base = sapien.Pose(
                p=[-0.25, 0.0, 0.0],
                q=euler2quat(0.0, 0.0, -np.pi * 1 / 8) #np.pi * 1 / 8 no rotation here
            )
            self._load_microwave(
                self.origin_base, self.articulation_types, [model_id], [microwave_link_ids[i]]
            )

        # Load YCB objects
        model_ids = self._episode_rng.choice(self.ycb_model_ids, size=1, replace=False)
        # Define spawn positions for objects (within robot workspace)
        xy_poses = [
            (-0.5, -0.3),  # Front right
        ]
        self._objs: List[Actor] = []
        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
                scales=[0.6]
            )
            x, y = xy_poses[i]
            builder.initial_pose = sapien.Pose(p=[x, y, 0])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))

        # Load RoboTwin objects
        robotwin_model_ids = self._batched_episode_rng.choice(self.robotwin_model_ids)
        # Define spawn positions for objects (within robot workspace)
        self.robotwin_poses = [
            sapien.Pose(
                p=[-0.4, -0.5, 0],
                q=euler2quat(np.pi/2, 0.0, 0.0)
            )
        ]
        self._robotwin_objs: List[Actor] = []
        for i, robotwin_modelname in enumerate(robotwin_model_ids):
            robotwin_obj = create_actor(
                scene=self.scene,
                pose=self.robotwin_poses[i],
                modelname=robotwin_modelname,
                convex=True,
                model_id=get_model_id(item_name=robotwin_modelname, random_id=True),
            )
            self._robotwin_objs.append(robotwin_obj.actor)

    def _load_microwave(self, origin_base, joint_types, model_ids, link_ids):
        for i, model_id in enumerate(model_ids):
            # Build articulation
            partnet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode="microwave", scale=0.3
            )
            partnet_builder.set_scene_idxs(scene_idxs=[i])
            partnet_builder.initial_pose = origin_base
            microwave = partnet_builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(microwave)
            for link in microwave.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=PARTNET_COLLISION_BIT, bit=1
                )
            self._microwaves.append(microwave)
            self.partnet_links.append([])
            self.partnet_links_meshes.append([])

            # select semantic parts of articulations
            for link, joint in zip(microwave.links, microwave.joints):
                if joint.type[0] in joint_types:
                    self.partnet_links[-1].append(link)
                    self.partnet_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: True, mesh_name="door",
                        )[0]
                    )

            # Merge to get class Articulation
            self.microwave = Articulation.merge(self._microwaves, name="microwave")
            self.add_to_state_dict_registry(self.microwave)

            # Set links state
            qpos_max = [link.joint.limits[..., 1] for link in self.partnet_links[0]]
            self.microwave_link = Link.merge(
                [links[i % len(links)] for i, links in enumerate(self.partnet_links)], name="microwave_link",
            )
            self.microwave_link_pos = common.to_tensor(
                np.array([
                        meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                        if meshes[link_ids[i] % len(meshes)] is not None else np.array([0., 0., 0.])
                        for i, meshes in enumerate(self.partnet_links_meshes)
                ]),
                device=self.device,
            )

    def _after_reconfigure(self, options: dict):
        # Reset YCB objects height to bottom at z=0
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            # Height to place object on table
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

        # Reset articulated objects height to bottom at z=0
        self.microwave_zs = []
        for microwave in self._microwaves:
            microwave_collision_mesh = microwave.get_first_collision_mesh()
            self.microwave_zs.append(-microwave_collision_mesh.bounding_box.bounds[0, 2])
        self.microwave_zs = common.to_tensor(self.microwave_zs, device=self.device)

        # Reset robotwin objects height to bottom at z=0
        self.robotwin_zs = []
        for robotwin_obj in self._robotwin_objs:
            robotwin_collision_mesh = robotwin_obj.get_first_collision_mesh()
            self.robotwin_zs.append(-robotwin_collision_mesh.bounding_box.bounds[0, 2])
        self.robotwin_zs = common.to_tensor(self.robotwin_zs, device=self.device)

        # get the qmin qmax values of the joint corresponding to the selected links
        target_qlimits = self.microwave_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.max_close_frac

    def door_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.microwave_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.microwave_link_pos, device=self.device),
            )
        return transform_points(
            self.microwave_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.microwave_link_pos[env_idx], device=self.device),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize objects...
            for i, obj in enumerate(self._objs):
                xyz = obj.pose.p[env_idx] + torch.randn(b, 3) * 0.01
                xyz[:, 2] = self.object_zs[i]
                q = obj.pose.q[env_idx]
                obj.set_pose(Pose.create_from_pq(p=xyz, q=q))
            # Initialize objects...
            for i, obj in enumerate(self._robotwin_objs):
                xyz = torch.tensor(self.robotwin_poses[i].p) + torch.randn(3) * 0.01
                xyz[2] = self.robotwin_zs[i]
                q = torch.tensor(self.robotwin_poses[i].q)
                obj.set_pose(Pose.create_from_pq(p=xyz, q=q))

            # Initialize microwave
            microwave_xyz = self.microwave.pose.p[env_idx]
            microwave_xyz[:, 2] = self.microwave_zs[env_idx]
            q = self.microwave.pose.q[env_idx]
            self.microwave.set_pose(Pose.create_from_pq(p=microwave_xyz, q=q))

            # Initialize Realman robot
            base_qpos = torch.zeros(len(self.agent.robot.get_active_joints()))
            base_qpos[0:2] = torch.tensor([0.0, 0.0])  # head_joint1, head_joint2
            right_arm_qpos = torch.tensor([0.82383, -0.17647, 0, -1.44114, -2.09, 0.926211, 1.55079])
            left_arm_qpos = torch.tensor([-0.81842, -0.176488, 0, -1.37455, -0.875422, -0.997037, -1.54696])
            for i in range(7):
                base_qpos[2 + i * 2] = right_arm_qpos[i]  # r_joint[i+1]
                base_qpos[2 + i * 2 + 1] = left_arm_qpos[i]  # l_joint[i+1]
            for i in range(16, len(base_qpos)):
                base_qpos[i] = 0.0
            qpos = base_qpos.repeat(b).reshape(b, -1)
            if self.robot_init_qpos_noise > 0:
                arm_noise = torch.randn(b, 14) * self.robot_init_qpos_noise
                qpos[:, 2:16] += arm_noise
            self.agent.robot.set_qpos(qpos)
            head_joints = [
                self.agent.robot.get_active_joints()[0],  # head_joint1
                self.agent.robot.get_active_joints()[1]  # head_joint2
            ]
            for joint in head_joints:
                joint.set_drive_properties(
                    stiffness=10000,
                    damping=1000,
                    force_limit=1000
                )
            # GPU simulation workaround
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
            self.scene._gpu_apply_all()

    def evaluate(self):
        open_enough = self.microwave_link.joint.qpos >= self.target_qpos
        obj_to_goal_pos = self._objs[0].pose.p - torch.tensor(
            self.microwave.pose.p, device=self.device
        )
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_robot_static = self.agent.is_static(0.2)
        microwave_link_pos = self.door_link_positions()
        link_is_static = (
            torch.linalg.norm(self.microwave_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.microwave_link.linear_velocity, axis=1) <= 0.1)
        success = is_obj_placed & open_enough & link_is_static
        is_left_grasping = self.agent.is_left_grasping(self._objs[0])
        is_right_grasping = self.agent.is_right_grasping(self._objs[0])
        is_grasped = torch.logical_or(is_left_grasping, is_right_grasping)
        obs = dict(
            open_enough=open_enough,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            microwave_link_pos=microwave_link_pos,
            link_is_static=link_is_static,
            is_grasped=is_grasped,
            success=success,
        )
        return obs

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.agent.left_tcp.pose.raw_pose,
            right_tcp_pose=self.agent.right_tcp.pose.raw_pose,
            is_grasped=info["is_grasped"],
            is_obj_placed=info["is_obj_placed"],
            target_pos=torch.tensor(self.microwave.pose.p, device=self.device),
        )

        if "state" in self.obs_mode:
            obs.update(
                left_tcp_to_obj=self._objs[0].pose.p - self.agent.left_tcp.pose.p,
                right_tcp_to_obj=self._objs[0].pose.p - self.agent.right_tcp.pose.p,
                obj_pose=self._objs[0].pose.raw_pose,
                obj_to_goal_pos=info["obj_to_goal_pos"],
                left_gripper_open=self.agent.get_left_gripper_openness(),
                right_gripper_open=self.agent.get_right_gripper_openness(),
            )

        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute dense reward for learning"""
        # Distance from closest TCP to object
        left_tcp_to_obj_dist = torch.linalg.norm(
            self._objs[0].pose.p - self.agent.left_tcp.pose.p, axis=1
        )
        right_tcp_to_obj_dist = torch.linalg.norm(
            self._objs[0].pose.p - self.agent.right_tcp.pose.p, axis=1
        )

        min_tcp_dist = torch.minimum(left_tcp_to_obj_dist, right_tcp_to_obj_dist)
        reaching_reward = 1 - torch.tanh(5 * min_tcp_dist)
        reward = reaching_reward

        # Grasping reward
        is_grasped = info["is_grasped"]
        reward += is_grasped * 2

        # Placement reward
        obj_to_goal_dist = torch.linalg.norm(info["obj_to_goal_pos"], axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        # Final placement reward
        reward += info["is_obj_placed"] * is_grasped * 2

        # Static reward (robot should stop moving when object is placed)
        left_arm_vel = torch.linalg.norm(self.agent.robot.get_qvel()[..., :7], axis=1)
        right_arm_vel = torch.linalg.norm(self.agent.robot.get_qvel()[..., 7:14], axis=1)
        min_arm_vel = torch.minimum(left_arm_vel, right_arm_vel)
        static_reward = 1 - torch.tanh(5 * min_arm_vel)
        reward += static_reward * info["is_obj_placed"] * is_grasped

        # Success bonus
        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
