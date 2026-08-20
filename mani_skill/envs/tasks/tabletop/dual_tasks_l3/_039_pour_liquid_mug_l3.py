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
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    above_reward,
    geom_center_from_local_mesh,
    grasp_reward,
    normalized_progress,
    reach_reward,
)


REWARD_PHASES = ["reach_liquid", "grasp_liquid", "pour_approach"]


@register_env("TwoRobotPourLiquidMugL3-v1", max_episode_steps=100)
class TwoRobotPourLiquidMugEnvL3(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.15

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
        # Fixed L3 assets (no L-level replacement logic).
        self.liquid_cup_modelname = "021_cup"
        self.liquid_cup_model_id = get_model_id(self.liquid_cup_modelname, model_id=0)
        self.target_cup_modelname = "021_cup"
        self.target_cup_model_id = get_model_id(self.target_cup_modelname, model_id=2)
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

    # Disable LR mirror hook for this standalone L3 task.
    def _maybe_apply_lr_mirror(self, env_idx: torch.Tensor, options: dict):
        return

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Load liquid
        self.liquid_pose = sapien.Pose(
            p=[-0.02, -0.1, -0.07],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        liquid_actor_obj = create_actor(
            scene=self.scene,
            pose=self.liquid_pose,
            modelname=self.liquid_cup_modelname,
            convex=True,
            model_id=self.liquid_cup_model_id,
            replace_scale=True,
            scale=(0.03, 0.2, 0.03),
            _idx_if_repeat=0,
        )
        liquid_actor_obj.set_mass(0.5)
        self.liquid = liquid_actor_obj.actor

        # Load rack
        self.rack_pose = sapien.Pose(
            p=[-0.02, -0.1, 0.05],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        rack_actor_obj = create_actor(
            scene=self.scene,
            pose=self.rack_pose,
            modelname="004_fluted-block",
            convex=True,
            is_static=True,
            model_id=1,
            scale=(0.9, 3.0, 0.9),
        )
        self.rack = rack_actor_obj.actor

        # Load cup
        self.cup_pose = sapien.Pose(
            p=[-0.05, 0.1, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 2)
        )
        cup_actor_obj = create_actor(
            scene=self.scene,
            pose=self.cup_pose,
            modelname=self.target_cup_modelname,
            convex=True,
            model_id=self.target_cup_model_id,
            is_static=True,
            # replace_scale=True,
            scale=(2.0, 2.0, 2.0),
            _idx_if_repeat=1,
        )
        self.cup = cup_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            if not hasattr(self, "near_goal"):
                self.near_goal = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.near_goal[env_idx] = False

            if not hasattr(self, "liquid_init_rot"):
                self.liquid_init_rot = torch.eye(
                    3, dtype=torch.float32, device=self.device
                ).unsqueeze(0).repeat(self.num_envs, 1, 1)

            shared_xy_offset = torch.rand((b, 2)) * 0.02
            xyz = torch.tensor(self.liquid_pose.p).repeat(b, 1)
            xyz[:, :2] += shared_xy_offset
            xyz[:, 2] = self.liquid_zs[env_idx]
            qs = torch.tensor(self.liquid_pose.q).repeat(b, 1)
            self.liquid.set_pose(Pose.create_from_pq(xyz, qs))
            self.liquid_init_rot[env_idx] = self.liquid.pose.to_transformation_matrix()[
                env_idx, :3, :3
            ]

            xyz = torch.tensor(self.rack_pose.p).repeat(b, 1)
            xyz[:, :2] += shared_xy_offset
            xyz[:, 2] = self.rack_zs[env_idx]
            qs = torch.tensor(self.rack_pose.q).repeat(b, 1)
            self.rack.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.cup_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.cup_zs[env_idx]
            qs = torch.tensor(self.cup_pose.q).repeat(b, 1)
            self.cup.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.liquid_zs = []
        collision_mesh = self.liquid.get_first_collision_mesh()
        self.liquid_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.liquid_zs = common.to_tensor(self.liquid_zs, device=self.device)

        self.rack_zs = []
        collision_mesh = self.rack.get_first_collision_mesh()
        self.rack_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.rack_zs = common.to_tensor(self.rack_zs, device=self.device)

        self.cup_zs = []
        collision_mesh = self.cup.get_first_collision_mesh()
        self.cup_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cup_zs = common.to_tensor(self.cup_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        liquid_center = self._get_actor_geom_center(self.liquid)
        liquid_pour_point = self._get_liquid_pour_point()
        cup_pos = self.cup.pose.p

        is_grasped = torch.logical_or(
            self.left_agent.is_grasping(self.liquid),
            self.right_agent.is_grasping(self.liquid),
        )

        obj_to_goal_pos = liquid_pour_point - cup_pos
        obj_to_goal_xy = torch.linalg.norm(obj_to_goal_pos[..., :2], dim=-1)
        liquid_height_above_cup = obj_to_goal_pos[..., 2]
        is_above_cup = torch.logical_and(
            obj_to_goal_xy <= 0.12,
            liquid_height_above_cup >= 0.02,
        )

        liquid_rot = self.liquid.pose.to_transformation_matrix()[:, :3, :3]
        rot_rel = torch.matmul(self.liquid_init_rot.transpose(1, 2), liquid_rot)
        trace = rot_rel[:, 0, 0] + rot_rel[:, 1, 1] + rot_rel[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
        tilt_angle = torch.arccos(cos_angle)
        is_tilted = tilt_angle >= np.deg2rad(15.0)

        currently_pouring = torch.logical_and(
            is_grasped, is_above_cup
        )
        self.near_goal = torch.logical_or(self.near_goal, currently_pouring)

        is_robot_static = torch.logical_and(
            self.left_agent.is_static(0.2), self.right_agent.is_static(0.2)
        )

        result = dict(
            is_grasped=is_grasped,
            is_above_cup=is_above_cup,
            is_tilted=is_tilted,
            liquid_height_above_cup=liquid_height_above_cup,
            currently_pouring=currently_pouring,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_above_cup,
            near_goal=self.near_goal,
            is_robot_static=is_robot_static,
            is_grasping=is_grasped,
            success=self.near_goal,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        return geom_center_from_local_mesh(obj, self.device)

    def _get_liquid_pour_point(self) -> torch.Tensor:
        if not hasattr(self, "_liquid_pour_point_local"):
            collision_mesh = self.liquid.get_first_collision_mesh(to_world_frame=False)
            bounds = common.to_tensor(
                collision_mesh.bounding_box.bounds, device=self.device
            )
            local_min, local_max = bounds[0], bounds[1]
            local_center = (local_min + local_max) * 0.5
            local_extents = local_max - local_min
            axis_idx = int(torch.argmax(local_extents).item())

            init_rot = self.liquid_init_rot[0]
            init_axis = init_rot[:, axis_idx]
            use_max = bool((init_axis[2] >= 0).item())

            local_pour_point = local_center.clone()
            local_pour_point[axis_idx] = (
                local_max[axis_idx] if use_max else local_min[axis_idx]
            )
            self._liquid_pour_point_local = local_pour_point

        local_point = self._liquid_pour_point_local.unsqueeze(0).repeat(
            len(self.liquid.pose), 1
        )
        return transform_points(
            self.liquid.pose.to_transformation_matrix().clone(), local_point
        )

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.cup.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.liquid.pose.raw_pose,
                left_tcp_to_obj_pos=self.liquid.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.liquid.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.liquid.pose.p - self.cup.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        liquid_center = self._get_actor_geom_center(self.liquid)
        liquid_pour_point = self._get_liquid_pour_point()
        cup_pos = self.cup.pose.p

        is_grasped = info["is_grasped"]
        is_above_cup = info["is_above_cup"]
        is_tilted = info["is_tilted"]
        currently_pouring = info["currently_pouring"]
        success = info["success"]

        left_dist = torch.linalg.norm(liquid_center - left_tcp, dim=-1)
        right_dist = torch.linalg.norm(liquid_center - right_tcp, dim=-1)
        use_left = left_dist <= right_dist
        active_tcp = torch.where(use_left[:, None], left_tcp, right_tcp)

        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(active_tcp, liquid_center, scale=5.0)
        r_reach = torch.where(is_grasped | currently_pouring | success, ones, r_reach)
        self.reward_tracker.update("reach_liquid", r_reach)

        r_grasp = grasp_reward(
            active_tcp, liquid_center, is_grasped, proximity_scale=5.0
        )
        r_grasp = torch.where(is_above_cup | currently_pouring | success, ones, r_grasp)
        self.reward_tracker.update("grasp_liquid", r_grasp)

        r_approach = above_reward(
            liquid_pour_point,
            cup_pos,
            min_height=0.02,
            is_grasping=is_grasped,
            h_scale=8.0,
        )
        r_approach = torch.where(is_above_cup | currently_pouring | success, ones, r_approach)
        self.reward_tracker.update("pour_approach", r_approach)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
