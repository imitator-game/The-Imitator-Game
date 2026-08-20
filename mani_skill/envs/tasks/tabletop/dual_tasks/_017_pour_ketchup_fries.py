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
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_ycb_model_id,
    apply_l3_robotwin_model,
    is_l2_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    above_reward,
    tanh_reward,
    RewardTracker,
)

# Sub-task phase names for this task
REWARD_PHASES = ["reach", "grasp", "pour"]


@register_env("TwoRobotPourKetchupFries-v1", max_episode_steps=100)
class TwoRobotPourKetchupFriesEnv(BaseEnv):
    """
    Task: Left arm picks up ketchup bottle, moves it above fries, pours.

    Reward phases (from solve script):
      1. reach  — left TCP approaches ketchup                     [0, 1]
      2. grasp  — left arm grasps ketchup                         [0, 1]
      3. pour   — ketchup positioned above fries while grasped    [0, 1]

    Each phase has a completion override so that when the phase
    condition is met, the reward becomes exactly 1.0.
    This guarantees total = mean(1,1,1) = 1.0 on success.

    total = mean( peak(phase_i) for i in 1..3 ) → [0, 1]
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.2

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

    # ── sim / camera configs ─────────────────────────────────────────────────

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

    # ── scene loading ────────────────────────────────────────────────────────

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

        # Load ketchup
        self.ketchup_pose = sapien.Pose(
            p=[0.05, -0.1, 0.0],
            q=euler2quat(0, 0, np.pi / 2),
        )
        ketchup_ycb_id = apply_l2_ycb_model_id(
            "006_mustard_bottle", override_id="021_bleach_cleanser"
        )
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{ketchup_ycb_id}",
            scales=[1.2]
        )
        builder._mass = 0.5
        builder.initial_pose = self.ketchup_pose
        self.ketchup = builder.build(name=f"ketchup")

        # Load fries
        self.fries_pose = sapien.Pose(
            p=[0.05, 0.1, 0.0],
            q=euler2quat(0, 0, -np.pi / 2)
        )
        fries_modelname, fries_model_id = apply_l3_robotwin_model(
            "005_french-fries",
            model_id=0,
            override_name="005_french-fries",
            override_id=2,
        )
        fries_actor_obj = create_actor(
            scene=self.scene,
            pose=self.fries_pose,
            modelname=fries_modelname,
            convex=True,
            model_id=fries_model_id,
            is_static=True,
            replace_scale=True,
            scale=(0.08, 0.08, 0.08),
        )
        self.fries = fries_actor_obj.actor

    # ── episode init ─────────────────────────────────────────────────────────

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Persistent flags
            if not hasattr(self, "ketchup_ever_reached_fries"):
                self.ketchup_ever_reached_fries = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.ketchup_ever_reached_fries[env_idx] = False

            # Reward tracker (3 phases)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            # Ketchup pose
            xyz = torch.tensor(self.ketchup_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.ketchup_zs[env_idx]
            xyz = apply_l1_offset_xy(xyz, offset=(-0.2, -0.05))
            qs = torch.tensor(self.ketchup_pose.q).repeat(b, 1)
            self.ketchup.set_pose(Pose.create_from_pq(xyz, qs))

            # Fries pose
            xyz = torch.tensor(self.fries_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.fries_zs[env_idx]
            qs = torch.tensor(self.fries_pose.q).repeat(b, 1)
            self.fries.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):
        self.ketchup_zs = []
        collision_mesh = self.ketchup.get_first_collision_mesh()
        self.ketchup_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.ketchup_zs = common.to_tensor(self.ketchup_zs, device=self.device)

        self.fries_zs = []
        collision_mesh = self.fries.get_first_collision_mesh()
        self.fries_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.fries_zs = common.to_tensor(self.fries_zs, device=self.device)

    # ── agent helpers ────────────────────────────────────────────────────────

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    # ── evaluate ─────────────────────────────────────────────────────────────

    def evaluate(self):
        ketchup_to_fries_dist = torch.linalg.norm(
            self.ketchup.pose.p - self.fries.pose.p, axis=1
        )
        is_grasped = self.left_agent.is_grasping(self.ketchup)

        ketchup_to_fries_horizontal = self.ketchup.pose.p[:, :2] - self.fries.pose.p[:, :2]
        horizontal_dist = torch.linalg.norm(ketchup_to_fries_horizontal, axis=1)
        ketchup_height_above_fries = self.ketchup.pose.p[:, 2] - self.fries.pose.p[:, 2]

        is_above_fries = torch.logical_and(
            horizontal_dist <= 0.0 + self.goal_thresh,
            ketchup_height_above_fries > 0.15 if is_l2_enabled() else ketchup_height_above_fries > 0.15,
        )
        self.ketchup_ever_reached_fries = self.ketchup_ever_reached_fries | is_above_fries

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        success = self.ketchup_ever_reached_fries

        result = dict(
            is_grasped=is_grasped,
            ketchup_to_fries_dist=ketchup_to_fries_dist,
            ketchup_ever_reached_fries=self.ketchup_ever_reached_fries,
            is_robot_static=is_robot_static,
            success=success,
            is_obj_placed=self.ketchup_ever_reached_fries,
        )

        # Include sub-reward peaks for logging
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())

        return result

    # ── observations ─────────────────────────────────────────────────────────

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.ketchup.pose.p,
            is_grasped=info["is_grasped"],
            ketchup_ever_reached_fries=info["ketchup_ever_reached_fries"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.ketchup.pose.raw_pose,
                left_tcp_to_obj_pos=self.ketchup.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.ketchup.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.ketchup.pose.p - self.fries.pose.p,
            )
        return obs

    # ═════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 3 sub-tasks, peak-tracked, arithmetic mean → [0, 1]
    #
    #  Phase 1  REACH  — left TCP approaches ketchup              [0, 1]
    #  Phase 2  GRASP  — proximity / confirmed grasp              [0, 1]
    #  Phase 3  POUR   — ketchup above fries (XY close + height)  [0, 1]
    #
    #  Completion overrides guarantee success → total = 1.0:
    #    - grasped  ⇒ reach = 1.0
    #    - ever_above_fries ⇒ reach = grasp = pour = 1.0
    #
    #  total = mean( peak(phase_i) for i in 1..3 )
    # ═════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p       # (B, 3)
        ketchup_pos = self.ketchup.pose.p          # (B, 3)
        fries_pos = self.fries.pose.p              # (B, 3)
        is_grasped = info["is_grasped"]            # (B,) bool
        ever_above = info["ketchup_ever_reached_fries"]  # (B,) bool

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Phase 1: REACH — left TCP → ketchup ─────────────────────
        # Continuous: TCP proximity to ketchup
        # Completion: grasped ⇒ reach done ⇒ 1.0
        r_reach = reach_reward(tcp_pos, ketchup_pos, scale=5.0)
        r_reach = torch.where(is_grasped | ever_above, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        # ── Phase 2: GRASP — left arm grasps ketchup ────────────────
        # Continuous: proximity [0, 1), confirmed grasp → 1.0
        # Completion: ever_above ⇒ must have grasped ⇒ 1.0
        r_grasp = grasp_reward(tcp_pos, ketchup_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(ever_above, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        # ── Phase 3: POUR — ketchup above fries while grasped ───────
        # Continuous: XY proximity × height progress, gated by grasp
        # Completion: ever_above ⇒ 1.0
        r_pour = above_reward(
            ketchup_pos,
            fries_pos,
            min_height=0.12,
            is_grasping=is_grasped,
            h_scale=5.0,
        )
        r_pour = torch.where(ever_above, ones, r_pour)
        self.reward_tracker.update("pour", r_pour)

        # ── Diagnostics ──────────────────────────────────────────────
        self.reward_tracker.write_to_info(info)

        # ── Total = arithmetic mean of peaks → [0, 1] ───────────────
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)