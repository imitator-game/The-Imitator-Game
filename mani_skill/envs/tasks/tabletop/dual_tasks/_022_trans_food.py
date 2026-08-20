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
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.scene_builder.table.utils import create_actor
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_ycb_model_id,
    apply_l3_robotwin_model,
    register_lr_mirror_euler_sxyz,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    transport_reward,
    tanh_reward,
    RewardTracker,
)

# Sub-task phase names for this task
REWARD_PHASES = ["reach", "grasp", "transport"]


@register_env("TwoRobotTransFood-v1", max_episode_steps=100)
class TwoRobotTransFoodEnv(BaseEnv):
    """
    Task: Pick up spoon from source bowl, transport to destination bowl.

    Reward phases (from solve script):
      1. reach     — left TCP approaches spoon                    [0, 1]
      2. grasp     — left arm grasps spoon                        [0, 1]
      3. transport — spoon moved to destination bowl              [0, 1]

    Completion overrides guarantee success → total = 1.0.
    total = mean( peak(phase_i) for i in 1..3 ) → [0, 1]
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

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
        self.goal_thresh = 0.2
        self.bowl_src_modelname = "002_bowl"
        self.bowl_src_model_id = get_model_id(self.bowl_src_modelname, model_id=2)
        self.bowl_modelname, self.bowl_model_id = apply_l3_robotwin_model(
            "002_bowl",
            model_id=2,
            override_name="002_bowl",
            override_id=3,
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
        bowl_upright_pose = sapien.Pose(p=[0, 0, 0], q=euler2quat(np.pi / 2, 0.0, 0.0))
        bowl_scale = (1.08, 1.08, 1.08)

        # Source bowl
        bowl_src_actor_obj = create_actor(
            scene=self.scene,
            pose=bowl_upright_pose,
            modelname=self.bowl_src_modelname,
            convex=True,
            is_static=True,
            model_id=self.bowl_src_model_id,
            scale=bowl_scale,
            _idx_if_repeat=1,
        )
        self.bowl_src = bowl_src_actor_obj.actor

        # Destination bowl
        bowl_dst_actor_obj = create_actor(
            scene=self.scene,
            pose=bowl_upright_pose,
            modelname=self.bowl_modelname,
            convex=True,
            is_static=True,
            model_id=5,
            scale=bowl_scale,
            _idx_if_repeat=2,
        )
        self.bowl_dst = bowl_dst_actor_obj.actor

        # Spoon
        spoon_ycb_id = apply_l2_ycb_model_id("031_spoon", override_id="030_fork")
        spoon_builder = actors.get_actor_builder(
            self.scene, id=f'ycb:{spoon_ycb_id}', scales=[1.1]
        )
        spoon_builder._mass = 0.5
        spoon_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        self.spoon = spoon_builder.build(name='spoon')

        self._lr_mirror_euler_z_actors = (self.bowl_src, self.bowl_dst, self.spoon)
        register_lr_mirror_euler_sxyz(self.bowl_src, (np.pi / 2, 0.0, 0.0))
        register_lr_mirror_euler_sxyz(self.bowl_dst, (np.pi / 2, 0.0, 0.0))
        register_lr_mirror_euler_sxyz(self.spoon, (np.pi, -np.pi / 7, -np.pi / 2))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "grasped_steps"):
                self.grasped_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.grasped_steps[env_idx] = 0
            if not hasattr(self, "is_spoon_ever_in_bowl_dst"):
                self.is_spoon_ever_in_bowl_dst = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.is_spoon_ever_in_bowl_dst[env_idx] = False

            # Reward tracker (3 phases)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            # Source bowl position
            src_pos = torch.zeros((b, 3), device=self.device)
            dst_pos = torch.zeros((b, 3), device=self.device)

            src_pos[:, 0] = 0
            src_pos[:, 1] = -0.15
            src_pos[:, 2] = self._bowl_z
            src_pos = apply_l1_offset_xy(src_pos, offset=(-0.1, -0.1))
            src_pos[:, :2] += torch.rand((b, 2), device=self.device) * 0.02

            dst_pos[:, 0] = 0
            dst_pos[:, 1] = 0.15
            dst_pos[:, 2] = self._bowl_z
            dst_pos[:, :2] += torch.rand((b, 2), device=self.device) * 0.02

            bowl_qs = torch.tensor(euler2quat(np.pi / 2, 0.0, 0.0), device=self.device, dtype=src_pos.dtype)
            self.bowl_src.set_pose(Pose.create_from_pq(src_pos, bowl_qs))
            self.bowl_dst.set_pose(Pose.create_from_pq(dst_pos, bowl_qs))

            # Spoon position (in/near source bowl)
            spoon_pos = src_pos.clone()
            spoon_pos[:, 2] += self._spoon_z + 0.05
            spoon_pos[:, 1] -= 0.05

            quat = euler2quat(np.pi, -np.pi/7, -np.pi/2)
            self.spoon.set_pose(Pose.create_from_pq(spoon_pos, quat))

    def _after_reconfigure(self, options: dict):
        self._bowl_z = None
        collision_mesh = self.bowl_src.get_first_collision_mesh()
        self._bowl_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

        self._spoon_z = None
        collision_mesh = self.spoon.get_first_collision_mesh()
        self._spoon_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        # FIX: use | instead of Python `or` for tensor boolean ops
        is_grasped = (
                self.left_agent.is_grasping(self.spoon)
                | self.right_agent.is_grasping(self.spoon)
        )
        self.grasped_steps += is_grasped.int()

        spoon_to_bowl_dst_dist = torch.linalg.norm(self.spoon.pose.p[:, :2] - self.bowl_dst.pose.p[:, :2], axis=1)
        is_spoon_in_bowl_dst = spoon_to_bowl_dst_dist <= 0.15
        self.is_spoon_ever_in_bowl_dst = self.is_spoon_ever_in_bowl_dst | is_spoon_in_bowl_dst
        is_spoon_ever_in_bowl_dst = self.is_spoon_ever_in_bowl_dst

        # FIX: use & instead of Python `and` for tensor boolean ops
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

        success = is_spoon_ever_in_bowl_dst

        result = dict(
            success=success,
            is_grasped=is_grasped,
            grasped_steps=self.grasped_steps,
            is_spoon_in_bowl_dst=is_spoon_in_bowl_dst,
            is_spoon_ever_in_bowl_dst=is_spoon_ever_in_bowl_dst,
            is_robot_static=is_robot_static,
        )

        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())

        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            is_grasped=info["is_grasped"],
            grasped_steps=info["grasped_steps"],
            is_spoon_in_bowl_dst=info["is_spoon_in_bowl_dst"],
        )
        if "state" in self.obs_mode:
            obs.update(
                spoon_pose=self.spoon.pose.raw_pose,
                bowl_src_pose=self.bowl_src.pose.raw_pose,
                bowl_dst_pose=self.bowl_dst.pose.raw_pose,
                tcp_to_spoon_pos=self.spoon.pose.p - self.left_agent.tcp.pose.p,
            )
        return obs

    # ═════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 3 sub-tasks, peak-tracked, arithmetic mean → [0, 1]
    #
    #  Phase 1  REACH     — TCP approaches spoon (closer arm)     [0, 1]
    #  Phase 2  GRASP     — either arm grasps spoon               [0, 1]
    #  Phase 3  TRANSPORT — spoon moved to destination bowl       [0, 1]
    #
    #  Completion overrides guarantee success → total = 1.0:
    #    - grasped       ⇒ reach = 1.0
    #    - ever_in_dst   ⇒ reach = grasp = transport = 1.0  (success)
    #
    #  total = mean( peak(phase_i) for i in 1..3 )
    # ═════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p        # (B, 3)
        right_tcp = self.right_agent.tcp.pose.p      # (B, 3)
        spoon_pos = self.spoon.pose.p                # (B, 3)
        bowl_dst_pos = self.bowl_dst.pose.p          # (B, 3)
        is_grasped = info["is_grasped"]              # (B,) bool
        success = info["success"]                    # (B,) bool = ever_in_dst

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Phase 1: REACH — TCP → spoon (use closer arm) ───────────
        left_reach = reach_reward(left_tcp, spoon_pos, scale=5.0)
        right_reach = reach_reward(right_tcp, spoon_pos, scale=5.0)
        r_reach = torch.maximum(left_reach, right_reach)
        r_reach = torch.where(is_grasped | success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        # ── Phase 2: GRASP — either arm grasps spoon ────────────────
        r_grasp = is_grasped.float()
        r_grasp = torch.where(success, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        # ── Phase 3: TRANSPORT — spoon → destination bowl ───────────
        r_transport = transport_reward(spoon_pos, bowl_dst_pos, is_grasped, scale=5.0)
        r_transport = torch.where(success, ones, r_transport)
        self.reward_tracker.update("transport", r_transport)

        # ── Diagnostics ──────────────────────────────────────────────
        self.reward_tracker.write_to_info(info)

        # ── Total = arithmetic mean of peaks → [0, 1] ───────────────
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)