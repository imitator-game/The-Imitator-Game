import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, Tuple
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_robotwin_model,
)

from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
    in_region_reward,
)

# Sub-task phases for this task
REWARD_PHASES = [
    "reach_shoe_1", "grasp_shoe_1", "transport_shoe_1", "place_shoe_1",
    "reach_shoe_2", "grasp_shoe_2", "transport_shoe_2", "place_shoe_2",
]


@register_env("TwoRobotPlaceShoeBoxL3-v1", max_episode_steps=100)
class TwoRobotPlaceShoeBoxEnvL3(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.1

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        hi_res: bool = False,
        wrist_sensor: bool = False, 
        **kwargs
    ):
        self.hi_res = hi_res
        self.wrist_sensor = wrist_sensor
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.shoe_modelname, self.shoe_model_id = apply_l2_robotwin_model(
            "041_shoe",
            model_id=0,
            override_name="041_shoe",
            override_id=1,
        )
        # self.box_modelname, self.box_model_id = apply_l3_robotwin_model(
        #     "007_shoe-box",
        #     model_id=0,
        #     override_name="042_wooden_box",
        #     override_id=0,
        # )
        (
            self.box_modelname,
            self.box_model_id,
            self.box_scale,
            self.box_replace_scale,
        ) = (
            "042_wooden_box",
            0,
            (0.14, 0.13, 0.16),
            True
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
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        # Multiple camera views for better perception
        pose1 = sapien_utils.look_at(eye=[0.3, 0.3, 0.6], target=[-0.1, 0, -0.6])
        pose2 = sapien_utils.look_at(eye=[0.1, 0.0, 0.6], target=[0.0, 0, 0.0])
        pose3 = sapien_utils.look_at(eye=[0.3, -0.3, 0.6], target=[-0.1, 0, -0.6])
        pose4 = sapien_utils.look_at(eye=[-0.4, 0, 0.6], target=[0.0, 0, 0.2])

        if self.hi_res:
            return [
                CameraConfig("cam1", pose1, 640, 480, np.pi / 2, 0.01, 100),
                CameraConfig("cam2", pose2, 640, 480, np.pi / 2, 0.01, 100),
                CameraConfig("cam3", pose3, 640, 480, np.pi / 2, 0.01, 100),
                CameraConfig("zed2i", pose4, 1280, 720, np.pi * 1.1 / 2, 0.01, 100),
            ]
        else:
            return [
                CameraConfig("zed2i", pose4, 224, 224, np.pi * 1.1 / 2, 0.01, 100),
            ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.8, 0., 0.75], [0.0, 0.0, 0.25])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )
        self.agent.__init__(self.agent.agents, wrist_sensor=self.wrist_sensor)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Load shoe_1
        self.shoe_1_pose = sapien.Pose(
            p=[-0., -0.35, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        shoe_1_actor_obj = create_actor(
            scene=self.scene,
            pose=self.shoe_1_pose,
            modelname=self.shoe_modelname,
            convex=True,
            model_id=self.shoe_model_id,
            scale=(0.11, 0.11, 0.11),
            replace_scale=True,
            _idx_if_repeat=0,
        )
        self.shoe_1 = shoe_1_actor_obj.actor
        self.shoe_1.set_mass(0.2)

        # Load shoe_2
        self.shoe_2_pose = sapien.Pose(
            p=[-0., 0.35, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 2)
        )
        shoe_2_actor_obj = create_actor(
            scene=self.scene,
            pose=self.shoe_2_pose,
            modelname=self.shoe_modelname,
            convex=True,
            model_id=self.shoe_model_id,
            scale=(0.11, 0.11, 0.11),
            replace_scale=True,
            _idx_if_repeat=1,
        )
        self.shoe_2 = shoe_2_actor_obj.actor
        self.shoe_2.set_mass(0.2)

        # Load box
        self.box_pose = sapien.Pose(
            p=[0.0, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        box_actor_obj = create_actor(
            scene=self.scene,
            pose=self.box_pose,
            modelname=self.box_modelname,
            convex=True,
            model_id=self.box_model_id,
            is_static=True,
            replace_scale=self.box_replace_scale,
            scale=self.box_scale,
        )
        self.box = box_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize reward tracker
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            xyz = torch.tensor(self.shoe_1_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
            qs = torch.tensor(self.shoe_1_pose.q).repeat(b, 1)
            self.shoe_1.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.shoe_2_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.1))
            qs = torch.tensor(self.shoe_2_pose.q).repeat(b, 1)
            self.shoe_2.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.box_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.box_zs[env_idx]
            qs = torch.tensor(self.box_pose.q).repeat(b, 1)
            self.box.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.shoe_1_zs = []
        collision_mesh = self.shoe_1.get_first_collision_mesh()
        self.shoe_1_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.shoe_1_zs = common.to_tensor(self.shoe_1_zs, device=self.device)

        self.shoe_2_zs = []
        collision_mesh = self.shoe_2.get_first_collision_mesh()
        self.shoe_2_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.shoe_2_zs = common.to_tensor(self.shoe_2_zs, device=self.device)

        self.box_zs = []
        collision_mesh = self.box.get_first_collision_mesh()
        self.box_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.box_zs = common.to_tensor(self.box_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        shoe_1_to_box_dist = torch.linalg.norm(self.shoe_1.pose.p - self.box.pose.p, axis=1)
        is_shoe_1_placed = shoe_1_to_box_dist <= self.goal_thresh
        shoe_2_to_box_dist = torch.linalg.norm(self.shoe_2.pose.p - self.box.pose.p, axis=1)
        is_shoe_2_placed = shoe_2_to_box_dist <= self.goal_thresh

        is_shoe_1_grasped = self.left_agent.is_grasping(self.shoe_1)
        is_shoe_2_grasped = self.right_agent.is_grasping(self.shoe_2)
        is_robot_static = self.left_agent.is_static(0.2) and self.right_agent.is_static(0.2)

        success = torch.logical_and(is_shoe_1_placed, is_shoe_2_placed)

        result = dict(
            shoe_1_to_box_dist=shoe_1_to_box_dist,
            shoe_2_to_box_dist=shoe_2_to_box_dist,
            is_shoe_1_placed=is_shoe_1_placed,
            is_shoe_2_placed=is_shoe_2_placed,
            is_shoe_1_grasped=is_shoe_1_grasped,
            is_shoe_2_grasped=is_shoe_2_grasped,
            is_robot_static=is_robot_static,
            success=success,
        )
        # Append per-phase peak sub-rewards
        # if hasattr(self, "reward_tracker"):
        #     # result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.box.pose.p,
            is_shoe_1_grasped=info["is_shoe_1_grasped"],
            is_shoe_2_grasped=info["is_shoe_2_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                shoe_1_pose=self.shoe_1.pose.raw_pose,
                shoe_2_pose=self.shoe_2.pose.raw_pose,
                left_tcp_to_shoe_1_pos=self.shoe_1.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_shoe_2_pos=self.shoe_2.pose.p - self.right_agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        shoe_1_pos = self.shoe_1.pose.p
        shoe_2_pos = self.shoe_2.pose.p
        box_pos = self.box.pose.p

        is_shoe_1_grasped = info["is_shoe_1_grasped"]
        is_shoe_2_grasped = info["is_shoe_2_grasped"]
        is_shoe_1_placed = info["is_shoe_1_placed"]
        is_shoe_2_placed = info["is_shoe_2_placed"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Left arm: shoe_1
        r_reach_shoe_1 = reach_reward(left_tcp, shoe_1_pos, scale=5.0)
        r_reach_shoe_1 = torch.where(is_shoe_1_grasped, ones, r_reach_shoe_1)
        self.reward_tracker.update("reach_shoe_1", r_reach_shoe_1)

        r_grasp_shoe_1 = grasp_reward(left_tcp, shoe_1_pos, is_shoe_1_grasped, proximity_scale=5.0)
        self.reward_tracker.update("grasp_shoe_1", r_grasp_shoe_1)

        r_transport_shoe_1 = transport_reward(shoe_1_pos, box_pos, is_shoe_1_grasped, scale=5.0)
        r_transport_shoe_1 = torch.where(is_shoe_1_placed, ones, r_transport_shoe_1)
        self.reward_tracker.update("transport_shoe_1", r_transport_shoe_1)

        # Place reward only when transport started and released
        transport_peak_1 = self.reward_tracker._peaks["transport_shoe_1"]
        r_place_shoe_1 = is_shoe_1_placed.float() * (transport_peak_1 > 0).float() * (~is_shoe_1_grasped).float()
        r_place_shoe_1 = torch.where(is_shoe_1_placed, ones, r_place_shoe_1)
        self.reward_tracker.update("place_shoe_1", r_place_shoe_1)

        # Right arm: shoe_2
        r_reach_shoe_2 = reach_reward(right_tcp, shoe_2_pos, scale=5.0)
        r_reach_shoe_2 = torch.where(is_shoe_2_grasped, ones, r_reach_shoe_2)
        self.reward_tracker.update("reach_shoe_2", r_reach_shoe_2)

        r_grasp_shoe_2 = grasp_reward(right_tcp, shoe_2_pos, is_shoe_2_grasped, proximity_scale=5.0)
        self.reward_tracker.update("grasp_shoe_2", r_grasp_shoe_2)

        r_transport_shoe_2 = transport_reward(shoe_2_pos, box_pos, is_shoe_2_grasped, scale=5.0)
        r_transport_shoe_2 = torch.where(is_shoe_2_placed, ones, r_transport_shoe_2)
        self.reward_tracker.update("transport_shoe_2", r_transport_shoe_2)

        # Place reward only when transport started and released
        transport_peak_2 = self.reward_tracker._peaks["transport_shoe_2"]
        r_place_shoe_2 = is_shoe_2_placed.float() * (transport_peak_2 > 0).float() * (~is_shoe_2_grasped).float()
        r_place_shoe_2 = torch.where(is_shoe_2_placed, ones, r_place_shoe_2)
        self.reward_tracker.update("place_shoe_2", r_place_shoe_2)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
