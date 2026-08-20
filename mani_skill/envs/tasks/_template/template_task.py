"""
Minimal L0/L1/L2 dual-arm environment template.

Levels L0/L1/L2 share ONE registered environment class:
    - L0: base scene, trajectory-level imitation.
    - L1: same objects, rearranged layout (spatial offsets).
    - L2: same task semantics, different object instances (model-id swap).

The level is chosen at runtime via L0_L3_utils (configure_dual_task_level in
the camera utils, or the --l0/--l1/--l2 flags in two_robot_run). This template
picks up an object (e.g. a YCB apple) and places it into a goal container.

How to make a new task:
    1. Copy this file into mani_skill/envs/tasks/tabletop/dual_tasks/.
    2. Change the @register_env id and class name.
    3. Replace _load_scene() assets with your own (see object_loader.py).
    4. Update _initialize_episode(), evaluate(), and compute_dense_reward().
"""

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from typing import Any, Dict, Tuple

from mani_skill.agents.multi_agent import MultiAgent
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

from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,   # L1: add a spatial offset to object xy
    apply_l2_ycb_model_id,  # L2: swap to another object instance
    apply_l3_robotwin_config,  # (used for container variants)
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    reach_reward,
    grasp_reward,
    transport_reward,
    world_aabb_from_local_mesh,
)

REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotTemplateTask-v1", max_episode_steps=100)
class TwoRobotTemplateTaskEnv(BaseEnv):
    """Template: pick a YCB apple and place it into a RoboTwin breadbasket."""

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    goal_thresh = 0.05          # success: object inside the container
    basket_height_threshold = 0.10
    apple_scale = 0.7

    def __init__(self, *args, robot_uids=("panda_wristcam", "panda_wristcam"),
                 robot_init_qpos_noise=0.02, num_envs=1, reconfiguration_freq=None,
                 hi_res: bool = False, wrist_sensor: bool = False, **kwargs):
        self.hi_res = hi_res
        self.wrist_sensor = wrist_sensor
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # L2/L3 object substitution hooks (no-op when the level is disabled).
        self.apple_model_id = apply_l2_ycb_model_id("013_apple", override_id="016_pear")
        (
            self.basket_modelname, self.basket_model_id,
            self.basket_scale, self.basket_replace_scale,
        ) = apply_l3_robotwin_config(
            "076_breadbasket", model_id=3, base_scale=(1.8, 1.8, 1.8),
            base_replace_scale=False, override_name="008_tray", override_id=2,
            override_scale=(1.2, 1.2, 1.2), override_replace_scale=False,
        )

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(*args, robot_uids=robot_uids,
                         reconfiguration_freq=reconfiguration_freq,
                         num_envs=num_envs, **kwargs)

    # ------------------------------------------------------------------ #
    # Simulation / camera configuration
    # ------------------------------------------------------------------ #
    @property
    def _default_sim_config(self):
        return SimConfig(gpu_memory_config=GPUMemoryConfig(
            found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**19,
            max_rigid_contact_count=2**21))

    @property
    def _default_sensor_configs(self):
        """External scene cameras. hi_res=False keeps only zed2i (224x224)."""
        p1 = sapien_utils.look_at(eye=[0.3, 0.3, 0.6], target=[-0.1, 0, -0.6])
        p2 = sapien_utils.look_at(eye=[0.1, 0.0, 0.6], target=[0.0, 0, 0.0])
        p3 = sapien_utils.look_at(eye=[0.3, -0.3, 0.6], target=[-0.1, 0, -0.6])
        p4 = sapien_utils.look_at(eye=[-0.4, 0, 0.6], target=[0.0, 0, 0.2])
        if self.hi_res:
            return [
                CameraConfig("cam1", p1, 640, 480, np.pi / 2, 0.01, 100),
                CameraConfig("cam2", p2, 640, 480, np.pi / 2, 0.01, 100),
                CameraConfig("cam3", p3, 640, 480, np.pi / 2, 0.01, 100),
                CameraConfig("zed2i", p4, 1280, 720, np.pi * 1.1 / 2, 0.01, 100),
            ]
        return [CameraConfig("zed2i", p4, 224, 224, np.pi * 1.1 / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig("render_camera",
                            sapien_utils.look_at([0.8, 0.0, 0.75], [0.0, 0.0, 0.25]),
                            512, 512, 1, 0.01, 100)

    # ------------------------------------------------------------------ #
    # Scene construction
    # ------------------------------------------------------------------ #
    def _load_agent(self, options: dict):
        super()._load_agent(options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])])
        self.agent.__init__(self.agent.agents, wrist_sensor=self.wrist_sensor)

    def _load_scene(self, options: dict):
        """Build the table and load all actors. See object_loader.py for the
        YCB / RoboTwin / PartNet / sketchfab loading helpers."""
        self.table_scene = TableSceneBuilder(env=self,
                                             random_background=False,
                                             robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # YCB object to grasp (dynamic actor).
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.apple_model_id}",
                                           scales=[self.apple_scale])
        builder._mass = 0.5
        builder.initial_pose = sapien.Pose(p=[0, 0.5, 0])
        self.apple = builder.build(name="apple")

        # RoboTwin goal container (static actor).
        basket_pose = sapien.Pose(p=[0, -0.5, 0], q=euler2quat(np.pi / 2, 0, 0))
        self.breadbasket = create_actor(
            scene=self.scene, pose=basket_pose, modelname=self.basket_modelname,
            scale=self.basket_scale, replace_scale=self.basket_replace_scale,
            convex=True, is_static=True, model_id=self.basket_model_id).actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Randomize per-episode poses. L1/L2 hooks are applied here."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize apple pose (with optional L1 xy offset).
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.1
            xyz[:, 1] = -0.15
            xyz[:, 2] = self.apple_z
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.1))
            qs = torch.tensor([euler2quat(0, 0, np.pi / 6)] * b,
                              device=self.device, dtype=torch.float32)
            self.apple.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Randomize basket pose (shared offset so items stay inside).
            basket_xyz = torch.zeros((b, 3), device=self.device)
            basket_xyz[:, 0] = 0.0
            basket_xyz[:, 1] = 0.1
            basket_xyz[:, 2] = self.basket_z
            basket_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            basket_xyz = apply_l1_offset_xy(basket_xyz, offset=(-0.1, 0.1))
            self.breadbasket.set_pose(Pose.create_from_pq(
                basket_xyz, torch.tensor([euler2quat(np.pi / 2, 0, 0)] * b,
                                         device=self.device, dtype=torch.float32)))

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

    def _after_reconfigure(self, options: dict):
        """Compute z so the object bottom rests on the table surface."""
        mesh = self.apple.get_first_collision_mesh()
        self.apple_z = -mesh.bounding_box.bounds[0, 2] if mesh is not None else 0.02
        self.basket_z = 0.0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        try:
            return geom_center_from_local_mesh(obj, self.device)
        except Exception:
            return obj.pose.p.clone()

    def _is_in_basket(self, obj_pos, xy_min, xy_max, z_max) -> torch.Tensor:
        in_x = torch.logical_and(obj_pos[..., 0] >= xy_min[..., 0], obj_pos[..., 0] <= xy_max[..., 0])
        in_y = torch.logical_and(obj_pos[..., 1] >= xy_min[..., 1], obj_pos[..., 1] <= xy_max[..., 1])
        in_z = obj_pos[..., 2] <= z_max + self.basket_height_threshold
        return torch.logical_and(torch.logical_and(in_x, in_y), in_z)

    # ------------------------------------------------------------------ #
    # Evaluation / observation / reward
    # ------------------------------------------------------------------ #
    def evaluate(self):
        apple_pos = self._get_actor_geom_center(self.apple)
        bounds = world_aabb_from_local_mesh(self.breadbasket, self.device)
        is_obj_placed = self._is_in_basket(apple_pos, bounds[:, 0, :2],
                                           bounds[:, 1, :2], bounds[:, 1, 2])
        is_grasped = self.left_agent.is_grasping(self.apple)
        is_robot_static = self.left_agent.is_static(0.2)
        result = dict(is_grasped=is_grasped, is_obj_placed=is_obj_placed,
                      is_robot_static=is_robot_static, is_grasping=is_grasped,
                      success=is_obj_placed)
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.left_agent.tcp.pose.raw_pose,
                   goal_pos=self.breadbasket.pose.p,
                   is_grasped=info["is_grasped"])
        if "state" in self.obs_mode:
            obs.update(apple_pose=self.apple.pose.raw_pose,
                       tcp_to_apple_pos=self.apple.pose.p - self.left_agent.tcp.pose.p,
                       breadbasket_pose=self.breadbasket.pose.raw_pose,
                       tcp_to_breadbasket_pos=self.breadbasket.pose.p - self.left_agent.tcp.pose.p,
                       apple_to_breadbasket_pos=self.breadbasket.pose.p - self.apple.pose.p)
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        apple_pos = self._get_actor_geom_center(self.apple)
        basket_pos = self._get_actor_geom_center(self.breadbasket)
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, apple_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, apple_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_obj_placed | success, ones, r_grasp)
        r_transport = transport_reward(apple_pos, basket_pos, is_grasped, scale=5.0)
        r_transport = torch.where(is_obj_placed | success, ones, r_transport)
        r_place = torch.where(success, ones, is_obj_placed.float())

        self.reward_tracker.update("reach", r_reach)
        self.reward_tracker.update("grasp", r_grasp)
        self.reward_tracker.update("transport", r_transport)
        self.reward_tracker.update("place", r_place)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
