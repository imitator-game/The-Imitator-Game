import numpy as np
import sapien
import torch
from typing import Any, Dict, Tuple
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    grasp_reward,
    reach_reward,
    transport_reward,
)


REWARD_PHASES = [
    "reach_left",
    "grasp_left",
    "transport_left",
    "place_left",
    "reach_right",
    "grasp_right",
    "transport_right",
    "place_right",
]


@register_env("TwoRobotPlaceCommodityRackL3-v1", max_episode_steps=200)
class TwoRobotPlaceCommodityRackEnvL3(BaseEnv):
    """
    Two robots each pick one mug and place it onto its own rack.
    - left robot: mug_left -> rack_left
    - right robot: mug_right -> rack_right
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    video_info_whitelist = {
        "reward",
        "success",
        "left_on_rack",
        "right_on_rack",
        "left_grasped",
        "right_grasped",
        "peak_r_reach_left",
        "peak_r_grasp_left",
        "peak_r_transport_left",
        "peak_r_place_left",
        "peak_r_reach_right",
        "peak_r_grasp_right",
        "peak_r_transport_right",
        "peak_r_place_right",
    }

    goal_thresh = 0.18

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

        self.mug_modelname = "039_mug"
        self.mug_left_model_id = 3
        self.mug_right_model_id = 4
        self.mug_scale = (0.08, 0.08, 0.08)

        self.rack_modelname = "040_rack"
        self.rack_model_id = 0
        self.rack_scale = (1.5, 1.5, 1.5)
        self.rack_replace_scale = True

        # Base XY layout: each arm handles its own side.
        self.mug_left_xy = (-0.2, -0.1)
        self.mug_right_xy = (-0.2, 0.1)
        self.rack_left_xy = (0.05, -0.1)
        self.rack_right_xy = (0.05, 0.1)

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
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
        pose = sapien_utils.look_at([0.8, 0.0, 0.75], [0.0, 0.0, 0.25])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )
        self.agent.__init__(self.agent.agents, wrist_sensor=self.wrist_sensor)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        mug_left_q = euler2quat(np.pi / 2, 0, 0)
        mug_right_q = euler2quat(-np.pi / 2, np.pi, 0)
        rack_q = euler2quat(np.pi / 2, 0, -np.pi)

        self.mug_left_pose = sapien.Pose(p=[self.mug_left_xy[0], self.mug_left_xy[1], 0.0], q=mug_left_q)
        self.mug_right_pose = sapien.Pose(p=[self.mug_right_xy[0], self.mug_right_xy[1], 0.0], q=mug_right_q)
        self.rack_left_pose = sapien.Pose(p=[self.rack_left_xy[0], self.rack_left_xy[1], 0.0], q=rack_q)
        self.rack_right_pose = sapien.Pose(p=[self.rack_right_xy[0], self.rack_right_xy[1], 0.0], q=rack_q)

        mug_left_obj = create_actor(
            scene=self.scene,
            pose=self.mug_left_pose,
            modelname=self.mug_modelname,
            convex=True,
            model_id=self.mug_left_model_id,
            scale=self.mug_scale,
            replace_scale=True,
            _idx_if_repeat=1,
            mass=0.5,
        )
        self.mug_left = mug_left_obj.actor

        mug_right_obj = create_actor(
            scene=self.scene,
            pose=self.mug_right_pose,
            modelname=self.mug_modelname,
            convex=True,
            model_id=self.mug_right_model_id,
            scale=self.mug_scale,
            replace_scale=True,
            _idx_if_repeat=2,
            mass=0.5,
        )
        self.mug_right = mug_right_obj.actor

        rack_left_obj = create_actor(
            scene=self.scene,
            pose=self.rack_left_pose,
            modelname=self.rack_modelname,
            convex=True,
            model_id=self.rack_model_id,
            is_static=True,
            replace_scale=self.rack_replace_scale,
            scale=self.rack_scale,
            _idx_if_repeat=3,
        )
        self.rack_left = rack_left_obj.actor

        rack_right_obj = create_actor(
            scene=self.scene,
            pose=self.rack_right_pose,
            modelname=self.rack_modelname,
            convex=True,
            model_id=self.rack_model_id,
            is_static=True,
            replace_scale=self.rack_replace_scale,
            scale=self.rack_scale,
            _idx_if_repeat=4,
        )
        self.rack_right = rack_right_obj.actor

    def _after_reconfigure(self, options: dict):
        def _compute_z(obj):
            mesh = obj.get_first_collision_mesh()
            return -mesh.bounding_box.bounds[0, 2]

        self.mug_left_z = common.to_tensor([_compute_z(self.mug_left)], device=self.device)
        self.mug_right_z = common.to_tensor([_compute_z(self.mug_right)], device=self.device)
        self.rack_left_z = common.to_tensor([_compute_z(self.rack_left)], device=self.device)
        self.rack_right_z = common.to_tensor([_compute_z(self.rack_right)], device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    REWARD_PHASES, self.num_envs, self.device
                )
            self.reward_tracker.reset(env_idx)

            mug_left_p = torch.tensor(self.mug_left_pose.p, device=self.device).repeat(b, 1)
            mug_left_p[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            mug_left_p[:, 2] = self.mug_left_z[0]
            mug_left_q = torch.tensor(self.mug_left_pose.q, device=self.device).repeat(b, 1)
            self.mug_left.set_pose(Pose.create_from_pq(p=mug_left_p, q=mug_left_q))

            mug_right_p = torch.tensor(self.mug_right_pose.p, device=self.device).repeat(b, 1)
            mug_right_p[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            mug_right_p[:, 2] = self.mug_right_z[0]
            mug_right_q = torch.tensor(self.mug_right_pose.q, device=self.device).repeat(b, 1)
            self.mug_right.set_pose(Pose.create_from_pq(p=mug_right_p, q=mug_right_q))

            rack_left_p = torch.tensor(self.rack_left_pose.p, device=self.device).repeat(b, 1)
            rack_left_p[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            rack_left_p[:, 2] = self.rack_left_z[0]
            rack_left_q = torch.tensor(self.rack_left_pose.q, device=self.device).repeat(b, 1)
            self.rack_left.set_pose(Pose.create_from_pq(p=rack_left_p, q=rack_left_q))

            rack_right_p = torch.tensor(self.rack_right_pose.p, device=self.device).repeat(b, 1)
            rack_right_p[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            rack_right_p[:, 2] = self.rack_right_z[0]
            rack_right_q = torch.tensor(self.rack_right_pose.q, device=self.device).repeat(b, 1)
            self.rack_right.set_pose(Pose.create_from_pq(p=rack_right_p, q=rack_right_q))

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        left_dist = torch.linalg.norm(self.mug_left.pose.p - self.rack_left.pose.p, axis=1)
        right_dist = torch.linalg.norm(self.mug_right.pose.p - self.rack_right.pose.p, axis=1)
        left_on_rack = left_dist <= self.goal_thresh
        right_on_rack = right_dist <= self.goal_thresh

        left_grasped = self.left_agent.is_grasping(self.mug_left) | self.right_agent.is_grasping(self.mug_left)
        right_grasped = self.left_agent.is_grasping(self.mug_right) | self.right_agent.is_grasping(self.mug_right)

        left_contact = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.mug_left, self.rack_left), axis=1
        ) > 1e-2
        right_contact = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.mug_right, self.rack_right), axis=1
        ) > 1e-2

        left_above_table = self.mug_left.pose.p[:, 2] > (self.mug_left_z + 0.08)
        right_above_table = self.mug_right.pose.p[:, 2] > (self.mug_right_z + 0.08)

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        success = (
            left_on_rack
            & right_on_rack
            & left_contact
            & right_contact
            & left_above_table
            & right_above_table
            & (~left_grasped)
            & (~right_grasped)
        )

        # # debug
        # print(f"left_dist:{left_dist}")
        # print(f"left_contact:{left_contact}")
        # print(f"self.mug_right.pose.p[:, 2]:{self.mug_right.pose.p[:, 2]}")
        # print(f"self.mug_right_z + 0.08:{self.mug_right_z + 0.08}")

        result = dict(
            left_on_rack=left_on_rack,
            right_on_rack=right_on_rack,
            left_grasped=left_grasped,
            right_grasped=right_grasped,
            is_robot_static=is_robot_static,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            mug_left_pose=self.mug_left.pose.raw_pose,
            mug_right_pose=self.mug_right.pose.raw_pose,
            rack_left_pose=self.rack_left.pose.raw_pose,
            rack_right_pose=self.rack_right.pose.raw_pose,
            left_on_rack=info["left_on_rack"],
            right_on_rack=info["right_on_rack"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        left_pos = self.mug_left.pose.p
        right_pos = self.mug_right.pose.p
        left_goal = self.rack_left.pose.p
        right_goal = self.rack_right.pose.p
        left_grasped = info["left_grasped"]
        right_grasped = info["right_grasped"]
        left_on_rack = info["left_on_rack"]
        right_on_rack = info["right_on_rack"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_left = reach_reward(left_tcp, left_pos, scale=5.0)
        r_reach_left = torch.where(left_grasped | left_on_rack | success, ones, r_reach_left)
        r_grasp_left = grasp_reward(left_tcp, left_pos, left_grasped, proximity_scale=5.0)
        r_grasp_left = torch.where(left_on_rack | success, ones, r_grasp_left)
        r_transport_left = transport_reward(left_pos, left_goal, left_grasped, scale=4.0)
        r_transport_left = torch.where(left_on_rack | success, ones, r_transport_left)
        r_place_left = torch.where(success, ones, left_on_rack.float())

        r_reach_right = reach_reward(right_tcp, right_pos, scale=5.0)
        r_reach_right = torch.where(right_grasped | right_on_rack | success, ones, r_reach_right)
        r_grasp_right = grasp_reward(right_tcp, right_pos, right_grasped, proximity_scale=5.0)
        r_grasp_right = torch.where(right_on_rack | success, ones, r_grasp_right)
        r_transport_right = transport_reward(right_pos, right_goal, right_grasped, scale=4.0)
        r_transport_right = torch.where(right_on_rack | success, ones, r_transport_right)
        r_place_right = torch.where(success, ones, right_on_rack.float())

        self.reward_tracker.update("reach_left", r_reach_left)
        self.reward_tracker.update("grasp_left", r_grasp_left)
        self.reward_tracker.update("transport_left", r_transport_left)
        self.reward_tracker.update("place_left", r_place_left)
        self.reward_tracker.update("reach_right", r_reach_right)
        self.reward_tracker.update("grasp_right", r_grasp_right)
        self.reward_tracker.update("transport_right", r_transport_right)
        self.reward_tracker.update("place_right", r_place_right)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
