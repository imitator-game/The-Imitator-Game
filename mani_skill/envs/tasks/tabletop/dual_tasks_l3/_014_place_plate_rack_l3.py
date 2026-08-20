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


REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotPlacePlateRackL3-v1", max_episode_steps=100)
class TwoRobotPlacePlateRackEnvL3(BaseEnv):
    """
    Place bowl on plate.
    Assets:
    - bowl: 002_bowl, model_id=4 (dynamic)
    - plate: 003_plate, model_id=0 (static)
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    goal_thresh = 0.05

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
        self.bowl_modelname = "002_bowl"
        self.bowl_model_id = 4
        self.plate_modelname = "003_plate"
        self.plate_model_id = 0

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

        self.bowl_pose = sapien.Pose(p=[-0.2, -0.3, 0.0], q=euler2quat(np.pi / 2, 0, 0))
        bowl_actor_obj = create_actor(
            scene=self.scene,
            pose=self.bowl_pose,
            modelname=self.bowl_modelname,
            convex=True,
            model_id=self.bowl_model_id,
            mass=0.5,
        )
        self.bowl = bowl_actor_obj.actor

        self.plate_pose = sapien.Pose(p=[-0.2, -0.1, 0.0], q=euler2quat(np.pi / 2, 0, 0))
        plate_actor_obj = create_actor(
            scene=self.scene,
            pose=self.plate_pose,
            modelname=self.plate_modelname,
            convex=True,
            model_id=self.plate_model_id,
            is_static=True,
            mass=0.5, 
        )
        self.plate = plate_actor_obj.actor

    def _after_reconfigure(self, options: dict):
        def _compute_z(obj):
            mesh = obj.get_first_collision_mesh()
            return -mesh.bounding_box.bounds[0, 2]

        self.bowl_z = common.to_tensor([_compute_z(self.bowl)], device=self.device)
        self.plate_z = common.to_tensor([_compute_z(self.plate)], device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    REWARD_PHASES, self.num_envs, self.device
                )
            self.reward_tracker.reset(env_idx)

            bowl_p = torch.tensor(self.bowl_pose.p, device=self.device).repeat(b, 1)
            bowl_p[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            bowl_p[:, 2] = self.bowl_z[0]
            bowl_q = torch.tensor(self.bowl_pose.q, device=self.device).repeat(b, 1)
            self.bowl.set_pose(Pose.create_from_pq(p=bowl_p, q=bowl_q))

            plate_p = torch.tensor(self.plate_pose.p, device=self.device).repeat(b, 1)
            plate_p[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            plate_p[:, 2] = self.plate_z[0]
            plate_q = torch.tensor(self.plate_pose.q, device=self.device).repeat(b, 1)
            self.plate.set_pose(Pose.create_from_pq(p=plate_p, q=plate_q))

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        obj_to_goal_pos = self.bowl.pose.p - self.plate.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_grasped = self.left_agent.is_grasping(self.bowl) | self.right_agent.is_grasping(self.bowl)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        success = is_obj_placed & (~is_grasped)
        result = dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        return dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            bowl_pose=self.bowl.pose.raw_pose,
            plate_pose=self.plate.pose.raw_pose,
            is_grasped=info["is_grasped"],
            is_obj_placed=info["is_obj_placed"],
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        obj_pos = self.bowl.pose.p
        goal_pos = self.plate.pose.p
        is_grasped = info["is_grasped"]
        is_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, obj_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, obj_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_placed | success, ones, r_grasp)
        r_transport = transport_reward(obj_pos, goal_pos, is_grasped, scale=5.0)
        r_transport = torch.where(is_placed | success, ones, r_transport)
        r_place = torch.where(success, ones, is_placed.float())

        self.reward_tracker.update("reach", r_reach)
        self.reward_tracker.update("grasp", r_grasp)
        self.reward_tracker.update("transport", r_transport)
        self.reward_tracker.update("place", r_place)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
