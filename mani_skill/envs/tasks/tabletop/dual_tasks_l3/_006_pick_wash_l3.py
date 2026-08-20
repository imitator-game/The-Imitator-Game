import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, Tuple

from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat
from mani_skill.utils.building import articulations
from mani_skill.utils.scene_builder.table.utils import create_actor
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


@register_env("TwoRobotPickWashL3-v1", max_episode_steps=100)
class TwoRobotPickWashEnvL3(BaseEnv):
    """Pick the single big bottle and place it into a plastic box."""

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    container_z_tolerance = 0.1

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

        self.bottle_modelname = "049_shampoo"
        self.bottle_model_id = 1
        self.bottle_scale = None
        self.bottle_replace_scale = False
        self.bottle_init_quat = euler2quat(np.pi / 2, 0.0, 0.0)
        self.bottle_mass = 0.5

        # Goal container: RobotWin plastic box.
        self.box_modelname = "062_plasticbox"
        self.box_model_id = get_model_id(self.box_modelname, model_id=4)
        self.box_scale = (2.0, 2.0, 2.0)
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

        bottle_actor_obj = create_actor(
            scene=self.scene,
            pose=sapien.Pose(p=[0.1, -0.1, 0.0], q=self.bottle_init_quat),
            modelname=self.bottle_modelname,
            convex=True,
            model_id=self.bottle_model_id,
            scale=self.bottle_scale,
            replace_scale=self.bottle_replace_scale,
            mass=self.bottle_mass,
        )
        self.bottle = bottle_actor_obj.actor

        # Plastic box (static goal container).
        self.box_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(np.pi / 2, 0.0, 0.0))
        box_actor_obj = create_actor(
            scene=self.scene,
            pose=self.box_pose,
            modelname=self.box_modelname,
            convex=True,
            model_id=self.box_model_id,
            scale=self.box_scale,
            replace_scale=self.box_replace_scale,
            is_static=True,
        )
        self.box = box_actor_obj.actor

    def _after_reconfigure(self, options: dict):
        bottle_collision_mesh = self.bottle.get_first_collision_mesh()
        self.bottle_z = common.to_tensor(
            [-bottle_collision_mesh.bounding_box.bounds[0, 2]], device=self.device
        )

        box_collision_mesh = self.box.get_first_collision_mesh()
        self.box_z = common.to_tensor(
            [-box_collision_mesh.bounding_box.bounds[0, 2]], device=self.device
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            bottle_xyz = torch.zeros((b, 3), device=self.device)
            bottle_xyz[:, 0] = -0.1
            bottle_xyz[:, 1] = -0.30
            bottle_xyz[:, :2] += (torch.rand((b, 2), device=self.device) * 0.04 - 0.02)
            bottle_xyz[:, 2] = self.bottle_z[env_idx]
            bottle_q = torch.tensor([self.bottle_init_quat] * b, dtype=torch.float32, device=self.device)
            self.bottle.set_pose(Pose.create_from_pq(p=bottle_xyz, q=bottle_q))

            box_xyz = torch.tensor(self.box_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            box_xyz[:, :2] += (torch.rand((b, 2), device=self.device) * 0.04 - 0.02)
            box_xyz[:, 2] = self.box_z[env_idx]
            box_q = torch.tensor(self.box_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.box.set_pose(Pose.create_from_pq(p=box_xyz, q=box_q))
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

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
            obj_name = getattr(obj, "name", obj.__class__.__name__)
            if obj_name not in self._geom_center_fallback_logged:
                print(f"[{self.__class__.__name__}] geom center fallback to pose.p for {obj_name}: {exc}")
                self._geom_center_fallback_logged.add(obj_name)
            return obj.pose.p.clone()

    def evaluate(self):
        box_bounds = world_aabb_from_local_mesh(self.box, self.device)
        bottle_center = self._get_actor_geom_center(self.bottle)
        is_obj_placed = self._is_in_container_bbox(bottle_center, box_bounds)
        result = dict(
            success=is_obj_placed,
            is_obj_placed=is_obj_placed,
            is_grasped=self.left_agent.is_grasping(self.bottle),
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
        )
        if "state" in self.obs_mode:
            obs.update(
                bottle_pose=self.bottle.pose.raw_pose,
                box_pose=self.box.pose.raw_pose,
                bottle_to_box_pos=self.box.pose.p - self.bottle.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        bottle_pos = self._get_actor_geom_center(self.bottle)
        box_pos = self._get_actor_geom_center(self.box)
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, bottle_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, bottle_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_obj_placed | success, ones, r_grasp)
        r_transport = transport_reward(bottle_pos, box_pos, is_grasped, scale=5.0)
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
        return self.compute_dense_reward(obs, action, info)
