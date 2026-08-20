import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, Tuple

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    reach_reward,
    grasp_reward,
    transport_reward,
    world_aabb_from_local_mesh,
)


REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotPickAppleBasketL3-v1", max_episode_steps=100)
class TwoRobotPickAppleBasketEnvL3(BaseEnv):
    """Pick apple and place it into RobotWin plastic box."""

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    goal_thresh = 0.05
    video_info_whitelist = {
        "reward",
        "success",
        "is_grasped",
        "is_obj_placed",
        "peak_r_reach",
        "peak_r_grasp",
        "peak_r_transport",
        "peak_r_place",
    }

    apple_scale = 0.7

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

        self.apple_model_id = "013_apple"
        self.box_modelname = "062_plasticbox"
        self.box_model_id = get_model_id(self.box_modelname, model_id=2)
        self.box_scale = (1.0, 1.0, 1.0)
        self.box_replace_scale = False

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

        # Apple
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.apple_model_id}",
            scales=[self.apple_scale],
        )
        builder._mass = 0.5
        builder.initial_pose = sapien.Pose(p=[0, 0.5, 0])
        self.apple = builder.build(name="apple")

        # RobotWin box
        self.box_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(np.pi / 2, 0.0, 0.0))
        box_actor_obj = create_actor(
            scene=self.scene,
            pose=self.box_pose,
            modelname=self.box_modelname,
            model_id=self.box_model_id,
            scale=self.box_scale,
            replace_scale=self.box_replace_scale,
            convex=True,
            is_static=True,
        )
        self.box = box_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            apple_xyz = torch.zeros((b, 3), device=self.device)
            apple_xyz[:, 0] = -0.1
            apple_xyz[:, 1] = -0.25
            apple_xyz[:, 2] = self.apple_z
            apple_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            apple_q = torch.tensor([euler2quat(0.0, 0.0, np.pi / 6)] * b, device=self.device, dtype=torch.float32)
            self.apple.set_pose(Pose.create_from_pq(p=apple_xyz, q=apple_q))

            box_xyz = torch.tensor(self.box_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            box_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            box_xyz[:, 2] = self.box_z
            box_q = torch.tensor(self.box_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.box.set_pose(Pose.create_from_pq(p=box_xyz, q=box_q))
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

    def _after_reconfigure(self, options: dict):
        collision_mesh = self.apple.get_first_collision_mesh()
        self.apple_z = -collision_mesh.bounding_box.bounds[0, 2] if collision_mesh is not None else 0.02

        collision_mesh = self.box.get_first_collision_mesh()
        self.box_z = -collision_mesh.bounding_box.bounds[0, 2] if collision_mesh is not None else 0.0

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        if not hasattr(self, "_geom_center_fallback_logged"):
            self._geom_center_fallback_logged = set()
        try:
            return geom_center_from_local_mesh(obj, self.device)
        except Exception as exc:
            obj_name = getattr(obj, "name", obj.__class__.__name__)
            if obj_name not in self._geom_center_fallback_logged:
                print(f"[{self.__class__.__name__}] geom center fallback to pose.p for {obj_name}: {exc}")
                self._geom_center_fallback_logged.add(obj_name)
            return obj.pose.p.clone()

    def evaluate(self):
        box_bounds = world_aabb_from_local_mesh(self.box, self.device)
        apple_center = self._get_actor_geom_center(self.apple)
        box_xy_min = box_bounds[:, 0, :2]
        box_xy_max = box_bounds[:, 1, :2]
        is_in_xy = torch.logical_and(
            torch.logical_and(apple_center[:, 0] >= box_xy_min[:, 0], apple_center[:, 0] <= box_xy_max[:, 0]),
            torch.logical_and(apple_center[:, 1] >= box_xy_min[:, 1], apple_center[:, 1] <= box_xy_max[:, 1]),
        )
        box_z_min = box_bounds[:, 0, 2]
        box_z_max = box_bounds[:, 1, 2]
        is_in_z = torch.logical_and(
            apple_center[:, 2] >= box_z_min - 0.02,
            apple_center[:, 2] <= box_z_max + 0.08,
        )

        is_obj_placed = torch.logical_and(is_in_xy, is_in_z)
        is_grasped = self.left_agent.is_grasping(self.apple)
        is_robot_static = self.left_agent.is_static(0.2)

        result = dict(
            is_grasped=is_grasped,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=is_grasped,
            success=is_obj_placed,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.left_agent.tcp.pose.raw_pose,
            goal_pos=self.box.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                apple_pose=self.apple.pose.raw_pose,
                tcp_to_apple_pos=self.apple.pose.p - self.left_agent.tcp.pose.p,
                box_pose=self.box.pose.raw_pose,
                tcp_to_box_pos=self.box.pose.p - self.left_agent.tcp.pose.p,
                apple_to_box_pos=self.box.pose.p - self.apple.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        apple_pos = self._get_actor_geom_center(self.apple)
        box_pos = self._get_actor_geom_center(self.box)
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, apple_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, apple_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_obj_placed | success, ones, r_grasp)
        r_transport = transport_reward(apple_pos, box_pos, is_grasped, scale=5.0)
        r_transport = torch.where(is_obj_placed | success, ones, r_transport)
        r_place = is_obj_placed.float()
        r_place = torch.where(success, ones, r_place)

        self.reward_tracker.update("reach", r_reach)
        self.reward_tracker.update("grasp", r_grasp)
        self.reward_tracker.update("transport", r_transport)
        self.reward_tracker.update("place", r_place)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
