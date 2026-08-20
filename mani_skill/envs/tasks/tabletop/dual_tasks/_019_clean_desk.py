import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, Tuple

from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_ycb_model_id,
    is_l2_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    transport_reward,
    wipe_progress_reward,
    RewardTracker,
)

# Sub-task phase names for this task
REWARD_PHASES = ["reach", "grasp", "approach", "wipe"]


@register_env("TwoRobotCleanDesk-v1", max_episode_steps=100)
class TwoRobotCleanDeskEnv(BaseEnv):
    """
    Task: Pick up sponge, wipe the desk surface (brick target) repeatedly.

    Reward phases (from solve script):
      1. reach    — TCP approaches sponge                         [0, 1]
      2. grasp    — arm grasps sponge                             [0, 1]
      3. approach — move sponge to brick (gated by grasp)         [0, 1]
      4. wipe     — accumulated cleaning steps + distance         [0, 1]

    Each phase has a completion override so that when the phase
    condition is met, the reward becomes exactly 1.0.
    This guarantees total = mean(1,1,1,1) = 1.0 on success.

    total = mean( peak(phase_i) for i in 1..4 ) → [0, 1]
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    # Clean success thresholds (from original evaluate)
    CLEAN_TARGET_STEPS = 90
    CLEAN_TARGET_DIST = 0.3

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
        self.goal_thresh = 0.12

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

        sponge_position = [0.2, 0.0, 0.0]
        sponge_rotation = [np.pi / 2, 0, 0] if is_l2_enabled() else [0, 0, 0]

        sponge_ycb_id = apply_l2_ycb_model_id("026_sponge", override_id="036_wood_block")
        sponge_scale = [0.5] if is_l2_enabled() else [0.8]
        sponge_builder = actors.get_actor_builder(
            self.scene,
            id=f'ycb:{sponge_ycb_id}',
            scales=sponge_scale,
        )
        sponge_builder._mass = 0.5
        sponge_pose = sapien.Pose(p=sponge_position, q=euler2quat(*sponge_rotation))
        sponge_builder.initial_pose = sponge_pose
        self.sponge = sponge_builder.build(name='sponge')

        brick_position = [0.0, 0.0, 0.0]
        brick_rotation = [0, 0, 0]

        brick_ycb_id = "061_foam_brick"
        brick_builder = actors.get_actor_builder(
            self.scene,
            id=f'ycb:{brick_ycb_id}',
            scales=[0.4],
        )
        brick_pose = sapien.Pose(p=brick_position, q=euler2quat(*brick_rotation))
        brick_builder.initial_pose = brick_pose
        self.brick = brick_builder.build(name='brick')

    # ── episode init ─────────────────────────────────────────────────────────

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Persistent cleaning tracking
            if not hasattr(self, "sponge_ever_cleaned_desk"):
                self.sponge_ever_cleaned_desk = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.sponge_ever_cleaned_desk[env_idx] = False

            if not hasattr(self, "sponge_last_pos"):
                self.sponge_last_pos = self.sponge.pose.p.clone()
            if not hasattr(self, "sponge_clean_dist"):
                self.sponge_clean_dist = torch.zeros(self.num_envs, device=self.device)
            if not hasattr(self, "sponge_clean_steps"):
                self.sponge_clean_steps = torch.zeros(
                    self.num_envs, dtype=torch.int32, device=self.device
                )
            self.sponge_last_pos[env_idx] = self.sponge.pose.p[env_idx].clone()
            self.sponge_clean_dist[env_idx] = 0.0
            self.sponge_clean_steps[env_idx] = 0

            # Reward tracker (4 phases)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            # Sponge pose
            xyz = self.sponge.pose.p
            qs = self.sponge.pose.q
            xyz[:, :2] += torch.tensor([-0.12, -0.17], device=self.device)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self._sponge_z
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.0))
            self.sponge.set_pose(Pose.create_from_pq(xyz, qs))

            # Brick pose
            xyz = self.brick.pose.p
            qs = self.brick.pose.q
            xyz[:, :2] += torch.tensor([-0.02, -0.02], device=self.device)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self._brick_z
            self.brick.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):
        collision_mesh = self.sponge.get_first_collision_mesh()
        self._sponge_z = common.to_tensor(
            -collision_mesh.bounding_box.bounds[0, 2], device=self.device
        )
        collision_mesh = self.brick.get_first_collision_mesh()
        self._brick_z = common.to_tensor(
            -collision_mesh.bounding_box.bounds[0, 2], device=self.device
        )

    # ── agent helpers ────────────────────────────────────────────────────────

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    # ── evaluate ─────────────────────────────────────────────────────────────

    def evaluate(self):
        is_grasped = (
            self.left_agent.is_grasping(self.sponge)
            | self.right_agent.is_grasping(self.sponge)
        )

        sponge_to_brick_dist = torch.linalg.norm(
            self.sponge.pose.p - self.brick.pose.p, axis=1
        )
        is_sponge_colliding_brick = sponge_to_brick_dist <= self.goal_thresh

        is_contact = is_sponge_colliding_brick & is_grasped
        self.sponge_ever_cleaned_desk = self.sponge_ever_cleaned_desk | is_contact

        # Accumulate clean distance and steps while in contact
        delta_xy = self.sponge.pose.p[:, :2] - self.sponge_last_pos[:, :2]
        step_dist = torch.linalg.norm(delta_xy, axis=1)
        self.sponge_clean_dist += step_dist * is_contact.float()
        self.sponge_clean_steps += is_contact.int()
        self.sponge_last_pos = self.sponge.pose.p.clone()

        is_robot_static = (
            self.left_agent.is_static(0.2)
            & self.right_agent.is_static(0.2)
        )

        success = (self.sponge_clean_steps >= self.CLEAN_TARGET_STEPS) & (
            self.sponge_clean_dist >= self.CLEAN_TARGET_DIST
        )

        result = dict(
            is_grasped=is_grasped,
            is_sponge_colliding_brick=is_sponge_colliding_brick,
            sponge_ever_cleaned_desk=self.sponge_ever_cleaned_desk,
            sponge_clean_dist=self.sponge_clean_dist,
            sponge_clean_steps=self.sponge_clean_steps,
            is_robot_static=is_robot_static,
            success=success,
            is_obj_placed=self.sponge_ever_cleaned_desk,
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
            is_grasped=info["is_grasped"],
            sponge_ever_cleaned_desk=info["sponge_ever_cleaned_desk"],
        )
        if "state" in self.obs_mode:
            obs.update(
                sponge_pose=self.sponge.pose.raw_pose,
                brick_pose=self.brick.pose.raw_pose,
                left_tcp_to_sponge_pos=self.sponge.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_sponge_pos=self.sponge.pose.p - self.right_agent.tcp.pose.p,
                sponge_to_brick_pos=self.brick.pose.p - self.sponge.pose.p,
            )
        return obs

    # ═════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 4 sub-tasks, peak-tracked, arithmetic mean → [0, 1]
    #
    #  Phase 1  REACH    — TCP approaches sponge (closer arm)     [0, 1]
    #  Phase 2  GRASP    — either arm grasps sponge               [0, 1]
    #  Phase 3  APPROACH — move sponge toward brick (gated)       [0, 1]
    #  Phase 4  WIPE     — accumulated contact steps + distance   [0, 1]
    #
    #  Completion overrides guarantee success → total = 1.0:
    #    - grasped           ⇒ reach = 1.0
    #    - ever_cleaned_desk ⇒ reach = grasp = approach = 1.0
    #    - success (wipe)    ⇒ all phases = 1.0
    #
    #  total = mean( peak(phase_i) for i in 1..4 )
    # ═════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p        # (B, 3)
        right_tcp = self.right_agent.tcp.pose.p      # (B, 3)
        sponge_pos = self.sponge.pose.p              # (B, 3)
        brick_pos = self.brick.pose.p                # (B, 3)
        is_grasped = info["is_grasped"]              # (B,) bool
        ever_cleaned = info["sponge_ever_cleaned_desk"]  # (B,) bool
        success = info["success"]                    # (B,) bool

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Phase 1: REACH — TCP → sponge (use closer arm) ──────────
        # Completion: grasped ⇒ 1.0
        left_reach = reach_reward(left_tcp, sponge_pos, scale=5.0)
        right_reach = reach_reward(right_tcp, sponge_pos, scale=5.0)
        r_reach = torch.maximum(left_reach, right_reach)
        r_reach = torch.where(is_grasped | ever_cleaned | success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        # ── Phase 2: GRASP — either arm grasps sponge ───────────────
        # Completion: ever_cleaned ⇒ must have grasped ⇒ 1.0
        left_gr = grasp_reward(left_tcp, sponge_pos, is_grasped, proximity_scale=5.0)
        right_gr = grasp_reward(right_tcp, sponge_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.maximum(left_gr, right_gr)
        r_grasp = torch.where(ever_cleaned | success, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        # ── Phase 3: APPROACH — move sponge toward brick while grasped
        # Completion: ever_cleaned ⇒ 1.0
        r_approach = transport_reward(sponge_pos, brick_pos, is_grasped, scale=5.0)
        r_approach = torch.where(ever_cleaned | success, ones, r_approach)
        self.reward_tracker.update("approach", r_approach)

        # ── Phase 4: WIPE — accumulated contact steps + distance ─────
        # Completion: success ⇒ 1.0
        r_wipe = wipe_progress_reward(
            steps=self.sponge_clean_steps,
            target_steps=self.CLEAN_TARGET_STEPS,
            dist=self.sponge_clean_dist,
            target_dist=self.CLEAN_TARGET_DIST,
        )
        r_wipe = torch.where(success, ones, r_wipe)
        self.reward_tracker.update("wipe", r_wipe)

        # ── Diagnostics ──────────────────────────────────────────────
        self.reward_tracker.write_to_info(info)

        # ── Total = arithmetic mean of peaks → [0, 1] ───────────────
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)