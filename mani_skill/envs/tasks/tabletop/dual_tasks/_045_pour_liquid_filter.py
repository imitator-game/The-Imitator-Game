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
    apply_l3_robotwin_model,
    is_lr_mirror_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    above_reward,
    geom_center_from_local_mesh,
    grasp_reward,
    normalized_progress,
    reach_reward,
)
from mani_skill.utils.geometry.geometry import transform_points


REWARD_PHASES = ["reach_liquid", "grasp_liquid", "pour_approach"]


@register_env("TwoRobotPourLiquidFilter-v1", max_episode_steps=100)
class TwoRobotPourLiquidFilterEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.15
    pour_xy_thresh = 0.18
    pour_min_height = -0.02
    lift_height_thresh = 0.04
    tilt_start_deg = 20.0
    tilt_target_deg = 65.0

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
        self.liquid_modelname, self.liquid_model_id = apply_l2_robotwin_model(
            "021_cup",
            model_id=6,
            override_name="021_cup",
            override_id=7,
        )
        self.cup_modelname, self.cup_model_id = apply_l3_robotwin_model(
            "021_cup",
            model_id=5,
            override_name="002_bowl",
            override_id=5,
        )
        self.teanet_modelname, self.teanet_model_id = apply_l3_robotwin_model(
            "053_teanet",
            model_id=1,
            override_name="053_teanet",
            override_id=4,
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

        # Load liquid
        self.liquid_pose = sapien.Pose(
            p=[-0.02, -0.07, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi)
        )
        liquid_actor_obj = create_actor(
            scene=self.scene,
            pose=self.liquid_pose,
            modelname=self.liquid_modelname,
            convex=True,
            model_id=self.liquid_model_id,
            replace_scale=True,
            scale=(0.08, 0.08, 0.08),
            mass=0.5,
        )
        self.liquid = liquid_actor_obj.actor

        # Load cup
        self.cup_pose = sapien.Pose(
            p=[-0.02, 0.13, 0.0],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        cup_actor_obj = create_actor(
            scene=self.scene,
            pose=self.cup_pose,
            modelname=self.cup_modelname,
            convex=True,
            model_id=self.cup_model_id,
            is_static=True,
            _idx_if_repeat=1,
            # replace_scale=True,
            scale=(1, 1.2, 1),
        )
        self.cup = cup_actor_obj.actor

        # Load teanet
        self.teanet_pose = sapien.Pose(
            p=[-0.02, 0.16, -0.09],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        teanet_actor_obj = create_actor(
            scene=self.scene,
            pose=self.teanet_pose,
            modelname=self.teanet_modelname,
            convex=True,
            model_id=self.teanet_model_id,
            is_static=True,
            # replace_scale=True,
            scale=(0.8, 0.8, 0.8),
        )
        self.teanet = teanet_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.tensor(self.liquid_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.liquid_zs[env_idx]
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
            # euler(π/2, 0, γ): mirror is γ → π - γ. liquid γ = π → 0
            liquid_z_rot = np.pi
            if is_lr_mirror_enabled():
                liquid_z_rot = np.pi - liquid_z_rot
            qs = torch.tensor(euler2quat(np.pi / 2, 0, liquid_z_rot)).repeat(b, 1)
            self.liquid.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.cup_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.0
            xyz[:, 2] = self.cup_zs[env_idx]
            # euler(π/2, 0, γ): mirror is γ → π - γ. cup γ = 0 → π
            cup_z_rot = 0.0
            if is_lr_mirror_enabled():
                cup_z_rot = np.pi - cup_z_rot
            qs = torch.tensor(euler2quat(np.pi / 2, 0, cup_z_rot)).repeat(b, 1)
            self.cup.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.teanet_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.0
            xyz[:, 2] = self.teanet_zs[env_idx]
            # euler(π/2, 0, γ): mirror is γ → π - γ. teanet γ = 0 → π
            teanet_z_rot = 0.0
            if is_lr_mirror_enabled():
                teanet_z_rot = np.pi - teanet_z_rot
            qs = torch.tensor(euler2quat(np.pi / 2, 0, teanet_z_rot)).repeat(b, 1)
            self.teanet.set_pose(Pose.create_from_pq(xyz, qs))

            # Reset per-episode pouring trackers.
            if not hasattr(self, "ever_poured"):
                self.ever_poured = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.ever_poured[env_idx] = False

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            # Record initial liquid orientation for robust tilt detection.
            if not hasattr(self, "liquid_init_rot"):
                self.liquid_init_rot = torch.eye(
                    3, device=self.device, dtype=torch.float32
                ).unsqueeze(0).repeat(self.num_envs, 1, 1)
            self.liquid_init_rot[env_idx] = self.liquid.pose.to_transformation_matrix()[
                env_idx, :3, :3
            ]
            if not hasattr(self, "liquid_init_center_z"):
                self.liquid_init_center_z = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.float32
                )
            self.liquid_init_center_z[env_idx] = self._get_actor_geom_center(
                self.liquid
            )[env_idx, 2]

    def _after_reconfigure(self, options: dict):

        self.liquid_zs = []
        collision_mesh = self.liquid.get_first_collision_mesh()
        self.liquid_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.liquid_zs = common.to_tensor(self.liquid_zs, device=self.device)

        self.cup_zs = []
        collision_mesh = self.cup.get_first_collision_mesh()
        self.cup_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cup_zs = common.to_tensor(self.cup_zs, device=self.device)

        self.teanet_zs = []
        collision_mesh = self.teanet.get_first_collision_mesh()
        self.teanet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.teanet_zs = common.to_tensor(self.teanet_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        liquid_center = self._get_actor_geom_center(self.liquid)
        liquid_pour_point = self._get_liquid_pour_point()
        target_center = self._get_pour_target_pos(liquid_pour_point)
        obj_to_goal_pos = liquid_pour_point - target_center
        obj_to_goal_xy = torch.linalg.norm(obj_to_goal_pos[..., :2], axis=1)
        liquid_height_above_target = obj_to_goal_pos[..., 2]
        lift_height = liquid_center[..., 2] - self.liquid_init_center_z
        is_lifted = lift_height >= self.lift_height_thresh
        is_above_cup = (
            (obj_to_goal_xy <= self.pour_xy_thresh)
            & (liquid_height_above_target > self.pour_min_height)
            & is_lifted
        )
        liquid_rot = self.liquid.pose.to_transformation_matrix()[:, :3, :3]
        if not hasattr(self, "liquid_init_rot"):
            self.liquid_init_rot = liquid_rot.clone()
        rot_rel = torch.matmul(self.liquid_init_rot.transpose(1, 2), liquid_rot)
        trace = rot_rel[:, 0, 0] + rot_rel[:, 1, 1] + rot_rel[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
        tilt_angle = torch.arccos(cos_angle)
        is_tilted = tilt_angle >= np.deg2rad(self.tilt_start_deg)

        is_grasped = self.left_agent.is_grasping(self.liquid) | self.right_agent.is_grasping(self.liquid)
        currently_pouring = is_grasped & is_lifted & is_above_cup
        self.ever_poured = self.ever_poured | currently_pouring

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        result = dict(
            is_grasped=is_grasped,
            is_lifted=is_lifted,
            lift_height=lift_height,
            is_above_cup=is_above_cup,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_above_cup,
            is_tilted=is_tilted,
            liquid_height_above_target=liquid_height_above_target,
            currently_pouring=currently_pouring,
            is_robot_static=is_robot_static,
            is_grasping=is_grasped,
            success=self.ever_poured,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        return geom_center_from_local_mesh(obj, self.device)

    def _get_pour_target_candidates(self) -> torch.Tensor:
        target_pos = self._get_actor_geom_center(self.cup).clone()
        offsets = torch.tensor(
            [[-0.11, 0.0, 0.0], [0.11, 0.0, 0.0]],
            device=self.device,
            dtype=target_pos.dtype,
        )
        return target_pos[:, None, :] + offsets[None, :, :]

    def _get_pour_target_pos(self, reference_pos: torch.Tensor | None = None) -> torch.Tensor:
        target_candidates = self._get_pour_target_candidates()
        if reference_pos is None:
            return target_candidates[:, 0, :]
        xy_dist = torch.linalg.norm(
            target_candidates[..., :2] - reference_pos[:, None, :2], dim=-1
        )
        best_idx = torch.argmin(xy_dist, dim=1)
        batch_idx = torch.arange(self.num_envs, device=self.device)
        return target_candidates[batch_idx, best_idx]

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

    def _prioritize_reward_info(self, info: Dict, total_reward: torch.Tensor):
        ordered = dict(reward=total_reward.clone())
        if "success" in info:
            ordered["success"] = info["success"]
        for key in list(info.keys()):
            if key.startswith("R"):
                ordered[key] = info[key]
        for key in list(info.keys()):
            if key.startswith("peak_r_"):
                ordered[key] = info[key]
        for key, value in info.items():
            if key not in ordered:
                ordered[key] = value
        info.clear()
        info.update(ordered)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        liquid_center = self._get_actor_geom_center(self.liquid)
        liquid_pour_point = self._get_liquid_pour_point()
        target_center = self._get_pour_target_pos(liquid_pour_point)

        is_grasped = info["is_grasped"]
        is_lifted = info["is_lifted"]
        lift_height = info["lift_height"]
        is_above_cup = info["is_above_cup"]
        is_tilted = info["is_tilted"]
        currently_pouring = info["currently_pouring"]
        success = info["success"]

        left_dist = torch.linalg.norm(liquid_center - left_tcp, dim=-1)
        right_dist = torch.linalg.norm(liquid_center - right_tcp, dim=-1)
        active_tcp = torch.where((left_dist <= right_dist)[:, None], left_tcp, right_tcp)
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(active_tcp, liquid_center, scale=5.0)
        r_reach = torch.where(is_grasped | currently_pouring | success, ones, r_reach)
        self.reward_tracker.update("reach_liquid", r_reach)

        lift_progress = normalized_progress(
            lift_height,
            start=0.01,
            target=self.lift_height_thresh,
        )
        r_grasp = grasp_reward(
            active_tcp, liquid_center, is_grasped, proximity_scale=5.0
        )
        r_grasp = torch.where(
            is_grasped,
            0.5 + 0.5 * lift_progress,
            r_grasp,
        )
        r_grasp = torch.where(is_above_cup | currently_pouring | success, ones, r_grasp)
        self.reward_tracker.update("grasp_liquid", r_grasp)

        r_approach = above_reward(
            liquid_pour_point,
            target_center,
            min_height=0.01,
            is_grasping=is_grasped & is_lifted,
            h_scale=8.0,
        )
        r_approach = torch.where(is_above_cup | currently_pouring | success, ones, r_approach)
        self.reward_tracker.update("pour_approach", r_approach)

        self.reward_tracker.write_to_info(info)
        total_reward = self.reward_tracker.total()
        self._prioritize_reward_info(info, total_reward)
        return total_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
