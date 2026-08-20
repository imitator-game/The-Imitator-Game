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
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_robotwin_model,
    apply_l3_robotwin_model,
    is_lr_mirror_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    above_reward,
    RewardTracker,
)

# Sub-task phase names for this task
REWARD_PHASES = ["reach", "grasp", "pour"]


@register_env("TwoRobotPourKettle-v1", max_episode_steps=200)
class TwoRobotPourKettleEnv(BaseEnv):
    """
    Task: Left arm grasps kettle, moves it above cup, pours.

    Reward phases (from solve script):
      1. reach  — left TCP approaches kettle                      [0, 1]
      2. grasp  — left arm grasps kettle                          [0, 1]
      3. pour   — kettle positioned above cup while grasped       [0, 1]

    Completion overrides guarantee success → total = 1.0.
    total = mean( peak(phase_i) for i in 1..3 ) → [0, 1]
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    goal_thresh = 0.01
    return_thresh = 0.08

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

        self.kettle_modelname, self.kettle_model_id = apply_l2_robotwin_model(
            "091_kettle",
            model_id=5,
            override_name="091_kettle",
            override_id=2,
        )
        self.cup_modelname, self.cup_model_id = apply_l3_robotwin_model(
            "021_cup",
            model_id=0,
            override_name="021_cup",
            override_id=1,
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

        self.kettle_pose = sapien.Pose(
            p=[-0.1, -0.2, 0],
            q=euler2quat(np.pi / 2, 0.0, 0.0)
        )
        kettle_obj = create_actor(
            scene=self.scene,
            pose=self.kettle_pose,
            modelname=self.kettle_modelname,
            convex=True,
            model_id=self.kettle_model_id,
        )
        kettle_obj.set_mass(0.5)
        self.kettle = kettle_obj.actor

        self.cup_pose = sapien.Pose(
            p=[-0.1, -0., 0],
            q=euler2quat(np.pi / 2, 0.0, 0.0)
        )
        cup_obj = create_actor(
            scene=self.scene,
            pose=self.cup_pose,
            modelname=self.cup_modelname,
            convex=True,
            is_static=True,
            model_id=self.cup_model_id,
        )
        self.cup = cup_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "kettle_ever_poured"):
                self.kettle_ever_poured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.kettle_ever_poured[env_idx] = False

            # Reward tracker (3 phases)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            # Initialize Kettle
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, :2] = torch.tensor(self.kettle_pose.p[:2])
            xyz[:, 2] = self.kettle_z
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(0.1, 0.1))

            z_rotation = np.pi
            if is_lr_mirror_enabled():
                z_rotation = 0
            base_quat = euler2quat(np.pi / 2, 0.0, z_rotation)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.kettle.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            self.kettle_initial_pos = xyz.clone()

            # Initialize Cup
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, :2] = torch.tensor(self.cup_pose.p[:2])
            xyz[:, 2] = self.cup_z
            xyz[:, :2] += torch.rand((b, 2)) * 0.02

            base_quat = euler2quat(np.pi/2, 0.0, 0.0)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.cup.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):
        collision_mesh = self.kettle.get_first_collision_mesh()
        self.kettle_z = 0.0

        collision_mesh = self.cup.get_first_collision_mesh()
        self.cup_z = 0.0

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        is_grasped = self.left_agent.is_grasping(self.kettle)

        kettle_to_cup_horizontal = self.kettle.pose.p[:, :2] - self.cup.pose.p[:, :2]
        horizontal_dist = torch.linalg.norm(kettle_to_cup_horizontal, axis=1)
        kettle_height_above_cup = self.kettle.pose.p[:, 2] - self.cup.pose.p[:, 2]
        is_above_cup = torch.logical_and(
            horizontal_dist <= 0.16 + self.goal_thresh,
            kettle_height_above_cup > 0.08
        )

        unit_z = torch.tensor([0, 0, 1.0], device=self.device)
        kettle_rot = self.kettle.pose.to_transformation_matrix()[:, :3, :3]
        cos_x = torch.abs(torch.sum(kettle_rot[:, :, 0] * unit_z, dim=1))
        cos_y = torch.abs(torch.sum(kettle_rot[:, :, 1] * unit_z, dim=1))
        cos_z = torch.abs(torch.sum(kettle_rot[:, :, 2] * unit_z, dim=1))
        cos_stack = torch.stack([cos_x, cos_y, cos_z], dim=1)
        if not hasattr(self, "kettle_up_axis"):
            self.kettle_up_axis = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        init_mask = self.elapsed_steps == 0
        if torch.any(init_mask):
            self.kettle_up_axis[init_mask] = torch.argmax(cos_stack[init_mask], dim=1)
        upright_cos = cos_stack.gather(1, self.kettle_up_axis[:, None]).squeeze(1)
        is_tilted = upright_cos <= np.cos(np.deg2rad(5))

        currently_pouring = is_grasped & is_above_cup
        self.kettle_ever_poured = self.kettle_ever_poured | currently_pouring

        is_robot_static = self.left_agent.is_static(0.2)

        success = self.kettle_ever_poured

        result = dict(
            is_grasped=is_grasped,
            is_above_cup=is_above_cup,
            is_tilted=is_tilted,
            kettle_ever_poured=self.kettle_ever_poured,
            is_robot_static=is_robot_static,
            kettle_to_cup_dist=horizontal_dist,
            success=success,
        )

        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())

        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.left_agent.tcp.pose.raw_pose,
            cup_pos=self.cup.pose.p,
            is_grasped=info["is_grasped"],
            kettle_ever_poured=info["kettle_ever_poured"],
        )
        if "state" in self.obs_mode:
            obs.update(
                kettle_pose=self.kettle.pose.raw_pose,
                tcp_to_kettle_pos=self.kettle.pose.p - self.left_agent.tcp.pose.p,
                cup_pose=self.cup.pose.raw_pose,
                kettle_to_cup_pos=self.cup.pose.p - self.kettle.pose.p,
            )
        return obs

    # ═════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 3 sub-tasks, peak-tracked, arithmetic mean → [0, 1]
    #
    #  Phase 1  REACH  — left TCP approaches kettle               [0, 1]
    #  Phase 2  GRASP  — proximity / confirmed grasp              [0, 1]
    #  Phase 3  POUR   — kettle above cup (XY close + height)     [0, 1]
    #
    #  Completion overrides guarantee success → total = 1.0:
    #    - grasped       ⇒ reach = 1.0
    #    - ever_poured   ⇒ reach = grasp = pour = 1.0  (success)
    #
    #  total = mean( peak(phase_i) for i in 1..3 )
    # ═════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p       # (B, 3)
        kettle_pos = self.kettle.pose.p            # (B, 3)
        cup_pos = self.cup.pose.p                  # (B, 3)
        is_grasped = info["is_grasped"]            # (B,) bool
        success = info["success"]                  # (B,) bool = ever_poured

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Phase 1: REACH — left TCP → kettle ──────────────────────
        r_reach = reach_reward(tcp_pos, kettle_pos, scale=5.0)
        r_reach = torch.where(is_grasped | success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        # ── Phase 2: GRASP — left arm grasps kettle ─────────────────
        r_grasp = grasp_reward(tcp_pos, kettle_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(success, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        # ── Phase 3: POUR — kettle above cup while grasped ──────────
        r_pour = above_reward(
            kettle_pos,
            cup_pos,
            min_height=0.08,
            is_grasping=is_grasped,
            h_scale=5.0,
        )
        r_pour = torch.where(success, ones, r_pour)
        self.reward_tracker.update("pour", r_pour)

        # ── Diagnostics ──────────────────────────────────────────────
        self.reward_tracker.write_to_info(info)

        # ── Total = arithmetic mean of peaks → [0, 1] ───────────────
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)