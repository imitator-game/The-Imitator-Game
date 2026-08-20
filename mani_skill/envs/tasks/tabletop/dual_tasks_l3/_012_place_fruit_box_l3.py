import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, List, Union, Tuple
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    grasp_reward,
    reach_reward,
    transport_reward,
    world_aabb_from_local_mesh,
)


REWARD_PHASES = [
    "reach_apple",
    "grasp_apple",
    "transport_apple",
    "place_apple",
    "reach_pear",
    "grasp_pear",
    "transport_pear",
    "place_pear",
]


@register_env("TwoRobotPlaceFruitBoxL3-v1", max_episode_steps=100)
class TwoRobotPlaceFruitBoxEnvL3(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh_apple = 0.1
    goal_thresh_pear = 0.2
    container_z_tolerance = 0.1
    video_info_whitelist = {
        "reward",
        "success",
        "is_apple_grasped",
        "is_pear_grasped",
        "apple_in_box",
        "pear_in_box",
        "is_obj_placed",
        "peak_r_reach_apple",
        "peak_r_grasp_apple",
        "peak_r_transport_apple",
        "peak_r_place_apple",
        "peak_r_reach_pear",
        "peak_r_grasp_pear",
        "peak_r_transport_pear",
        "peak_r_place_pear",
    }

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
        self.apple_modelname = "103_fruit"
        self.apple_model_id = 5
        self.apple_scale = (0.5, 0.5, 0.5)
        self.apple_replace_scale = False
        self.apple_mass = 0.5
        self.pear_modelname = "103_fruit"
        self.pear_model_id = 3
        self.pear_scale = (0.035, 0.035, 0.035)
        self.pear_replace_scale = True
        self.pear_mass = 0.5
        self.plasticbox_modelname = "062_plasticbox"
        self.plasticbox_model_id = 7
        self.plasticbox_scale = (0.25, 0.15, 0.25)
        self.plasticbox_replace_scale = False
        self.goalbox_modelname = "062_plasticbox"
        self.goalbox_model_id = 2
        self.goalbox_scale = (0.12, 0.12, 0.12)
        self.goalbox_replace_scale = False
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
        pear_repeat_idx = 1 if self.apple_modelname == self.pear_modelname else 0

        # Load apple
        self.apple_pose = sapien.Pose(
            p=[0.05, 0.0, 0.05],
            q=euler2quat(np.pi / 2, 0, np.pi / 3)
        )
        apple_actor_obj = create_actor(
            scene=self.scene,
            pose=self.apple_pose,
            modelname=self.apple_modelname,
            convex=True,
            model_id=self.apple_model_id,
            scale=self.apple_scale,
            replace_scale=self.apple_replace_scale,
            mass=self.apple_mass, 
        )
        self.apple = apple_actor_obj.actor

        self.pear_pose = sapien.Pose(
            p=[-0.05, 0.0, 0.05],
            q=euler2quat(np.pi / 2, 0, np.pi / 3)
        )
        pear_actor_obj = create_actor(
            scene=self.scene,
            pose=self.pear_pose,
            modelname=self.pear_modelname,
            convex=True,
            model_id=self.pear_model_id,
            scale=self.pear_scale,
            replace_scale=self.pear_replace_scale,
            _idx_if_repeat=pear_repeat_idx,
            mass=self.pear_mass, 
        )
        self.pear = pear_actor_obj.actor

        # Load basket
        self.basket_pose = sapien.Pose(
            p=[0.0, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        basket_actor_obj = create_actor(
            scene=self.scene,
            pose=self.basket_pose,
            modelname=self.plasticbox_modelname,
            convex=True,
            model_id=self.plasticbox_model_id,
            is_static=True,
            scale=self.plasticbox_scale, 
            replace_scale=True, 
            _idx_if_repeat=11,
        )
        self.basket = basket_actor_obj.actor

        # Load box
        self.box_pose = sapien.Pose(
            p=[0.0, -0.35, 0.0],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        box_actor_obj = create_actor(
            scene=self.scene,
            pose=self.box_pose,
            modelname=self.goalbox_modelname, 
            convex=True, 
            model_id=self.goalbox_model_id, 
            is_static=True, 
            scale=self.goalbox_scale, 
            replace_scale=True, 
            _idx_if_repeat=12,
        )
        self.box = box_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    REWARD_PHASES, self.num_envs, self.device
                )
            self.reward_tracker.reset(env_idx)

            xyz = torch.tensor(self.apple_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            qs = torch.tensor(self.apple_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.apple.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.pear_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            qs = torch.tensor(self.pear_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.pear.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.basket_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            qs = torch.tensor(self.basket_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.basket.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.box_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz[:, 2] = self.box_zs[env_idx]
            qs = torch.tensor(self.box_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.box.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.apple_zs = []
        collision_mesh = self.apple.get_first_collision_mesh()
        self.apple_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.apple_zs = common.to_tensor(self.apple_zs, device=self.device)

        self.pear_zs = []
        collision_mesh = self.pear.get_first_collision_mesh()
        self.pear_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.pear_zs = common.to_tensor(self.pear_zs, device=self.device)

        self.basket_zs = []
        collision_mesh = self.basket.get_first_collision_mesh()
        self.basket_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.basket_zs = common.to_tensor(self.basket_zs, device=self.device)

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

    def _is_in_container_bbox(self, obj_pos: torch.Tensor, container_bounds: torch.Tensor) -> torch.Tensor:
        if container_bounds.ndim == 3:
            lower = container_bounds[:, 0]
            upper = container_bounds[:, 1]
        else:
            lower = container_bounds[0]
            upper = container_bounds[1]
        upper = upper.clone()
        upper[..., 2] += self.container_z_tolerance
        return torch.all(torch.logical_and(obj_pos >= lower, obj_pos <= upper), dim=-1)

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        if not hasattr(self, "_geom_center_fallback_logged"):
            self._geom_center_fallback_logged = set()
        try:
            return geom_center_from_local_mesh(obj, self.device)
        except Exception as exc:
            name = getattr(obj, "name", type(obj).__name__)
            if name not in self._geom_center_fallback_logged:
                print(
                    f"[{self.__class__.__name__}] geom center fallback to pose.p for {name}: {exc}"
                )
                self._geom_center_fallback_logged.add(name)
            return obj.pose.p.clone()

    def evaluate(self):
        apple_pos = self._get_actor_geom_center(self.apple)
        pear_pos = self._get_actor_geom_center(self.pear)
        box_bounds = world_aabb_from_local_mesh(self.box, self.device)
        apple_in_box = self._is_in_container_bbox(apple_pos, box_bounds)
        pear_in_box = self._is_in_container_bbox(pear_pos, box_bounds)
        is_obj_placed = apple_in_box & pear_in_box
        is_apple_grasped = self.left_agent.is_grasping(self.apple) | self.right_agent.is_grasping(self.apple)
        is_pear_grasped = self.left_agent.is_grasping(self.pear) | self.right_agent.is_grasping(self.pear)
        is_grasped = is_apple_grasped | is_pear_grasped
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        result = dict(
            is_grasped=is_grasped,
            is_apple_grasped=is_apple_grasped,
            is_pear_grasped=is_pear_grasped,
            obj_to_goal_pos=apple_pos - self.box.pose.p,
            apple_in_box=apple_in_box,
            pear_in_box=pear_in_box,
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
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.box.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.apple.pose.raw_pose,
                left_tcp_to_obj_pos=self.apple.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.apple.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.apple.pose.p - self.box.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        apple_pos = self._get_actor_geom_center(self.apple)
        pear_pos = self._get_actor_geom_center(self.pear)
        goal_pos = self._get_actor_geom_center(self.box)
        tcp_pos = self.left_agent.tcp.pose.p
        is_apple_grasped = info["is_apple_grasped"]
        is_pear_grasped = info["is_pear_grasped"]
        apple_in_box = info["apple_in_box"]
        pear_in_box = info["pear_in_box"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_apple = reach_reward(tcp_pos, apple_pos, scale=5.0)
        r_reach_apple = torch.where(is_apple_grasped | apple_in_box | success, ones, r_reach_apple)
        r_grasp_apple = grasp_reward(tcp_pos, apple_pos, is_apple_grasped, proximity_scale=5.0)
        r_grasp_apple = torch.where(apple_in_box | success, ones, r_grasp_apple)
        r_transport_apple = transport_reward(apple_pos, goal_pos, is_apple_grasped, scale=5.0)
        r_transport_apple = torch.where(apple_in_box | success, ones, r_transport_apple)
        r_place_apple = torch.where(success, ones, apple_in_box.float())

        r_reach_pear = reach_reward(tcp_pos, pear_pos, scale=5.0)
        r_reach_pear = torch.where(is_pear_grasped | pear_in_box | success, ones, r_reach_pear)
        r_grasp_pear = grasp_reward(tcp_pos, pear_pos, is_pear_grasped, proximity_scale=5.0)
        r_grasp_pear = torch.where(pear_in_box | success, ones, r_grasp_pear)
        r_transport_pear = transport_reward(pear_pos, goal_pos, is_pear_grasped, scale=5.0)
        r_transport_pear = torch.where(pear_in_box | success, ones, r_transport_pear)
        r_place_pear = torch.where(success, ones, pear_in_box.float())

        self.reward_tracker.update("reach_apple", r_reach_apple)
        self.reward_tracker.update("grasp_apple", r_grasp_apple)
        self.reward_tracker.update("transport_apple", r_transport_apple)
        self.reward_tracker.update("place_apple", r_place_apple)
        self.reward_tracker.update("reach_pear", r_reach_pear)
        self.reward_tracker.update("grasp_pear", r_grasp_pear)
        self.reward_tracker.update("transport_pear", r_transport_pear)
        self.reward_tracker.update("place_pear", r_place_pear)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
