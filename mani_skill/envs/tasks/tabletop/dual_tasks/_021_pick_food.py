import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, List, Tuple

from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_ycb_model_id,
    apply_l3_robotwin_model,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    transport_reward,
    tanh_reward,
    RewardTracker,
)

# Sub-task phases: left picks fruit0, right picks fruit1 — 4 phases per arm
REWARD_PHASES = [
    "reach_0", "grasp_0", "transport_0", "place_0",
    "reach_1", "grasp_1", "transport_1", "place_1",
]


@register_env("TwoRobotPickFood-v1", max_episode_steps=100)
class TwoRobotPickFoodEnv(BaseEnv):
    """
    Task: Sequentially pick up two fruits and place them into a bin.

    Reward phases (from solve script):
      Left arm (fruit 0):
        1. reach_0     — left TCP → fruit 0                  [0, 1]
        2. grasp_0     — left arm grasps fruit 0             [0, 1]
        3. transport_0 — fruit 0 → bin center                [0, 1]
        4. place_0     — fruit 0 inside bin                  [0, 1]

      Right arm (fruit 1):
        5. reach_1     — right TCP → fruit 1                 [0, 1]
        6. grasp_1     — right arm grasps fruit 1            [0, 1]
        7. transport_1 — fruit 1 → bin center                [0, 1]
        8. place_1     — fruit 1 inside bin                  [0, 1]

    Completion overrides guarantee success → total = 1.0.
    total = mean( peak(phase_i) for i in 1..8 ) → [0, 1]
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
        self.tabletrashbin_modelname, self.tabletrashbin_model_id = apply_l3_robotwin_model(
            '063_tabletrashbin',
            model_id=9,
            override_name='063_tabletrashbin',
            override_id=10,
        )
        self.goal_thresh = 0.1

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

        self._fruits_ycb_objs: List[Actor] = []
        self._fruits_ycb_poses = []

        ycb_objects = [
            apply_l2_ycb_model_id("011_banana", override_id="014_lemon"),
            apply_l2_ycb_model_id("013_apple", override_id="015_peach"),
        ]
        ycb_positions = [[0.0, -0.25, 0.0], [0.0, 0.25, 0.0]]
        ycb_rotations = [[0, 0, np.pi / 2], [0, 0, 0]]
        self._fruits_l1_offsets = [(-0.1, -0.1), (-0.1, 0.1)]

        for i, (obj_name, position, rotation) in enumerate(
                zip(ycb_objects, ycb_positions, ycb_rotations)):
            builder = actors.get_actor_builder(
                self.scene,
                id=f'ycb:{obj_name}',
                scales=[0.6]
            )
            builder._mass = 0.5
            pose = sapien.Pose(p=position, q=euler2quat(*rotation))
            builder.initial_pose = pose
            self._fruits_ycb_poses.append(pose)
            self._fruits_ycb_objs.append(builder.build(name=f'{obj_name}-{i}'))

        q = euler2quat(*[np.pi / 2, 0, 0])

        pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=q)
        bin_obj = create_actor(
            scene=self.scene,
            pose=pose,
            modelname=self.tabletrashbin_modelname,
            convex=True,
            is_static=True,
            replace_scale=True,
            scale=(0.1, 0.1, 0.1),
            model_id=self.tabletrashbin_model_id,
        )
        self.bin = bin_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Reward tracker (8 phases)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            for i, (obj, pose) in enumerate(zip(self._fruits_ycb_objs, self._fruits_ycb_poses)):
                xyz = torch.tensor(pose.p).repeat(b, 1)
                qs = torch.tensor(pose.q).repeat(b, 1)
                xyz[:, :2] += torch.rand((b, 2)) * 0.02
                xyz = apply_l1_offset_xy(xyz, offset=self._fruits_l1_offsets[i])
                xyz[:, 2] = self._obj_zs[i]
                obj.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = self.bin.pose.p
            qs = self.bin.pose.q
            xyz[:, :2] += (torch.rand((b, 2))) * 0.02
            xyz[:, 2] = self._bin_z
            self.bin.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):
        self._obj_zs = []
        for i, obj in enumerate(self._fruits_ycb_objs):
            collision_mesh = obj.get_first_collision_mesh()
            self._obj_zs.append(common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device))

        self._bin_z = None
        collision_mesh = self.bin.get_first_collision_mesh()
        self._bin_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        is_grasped_left = self.left_agent.is_grasping(self._fruits_ycb_objs[0])
        is_grasped_right = self.right_agent.is_grasping(self._fruits_ycb_objs[1])
        is_grasped = is_grasped_left | is_grasped_right

        fruit0_to_bin = self._fruits_ycb_objs[0].pose.p - self.bin.pose.p
        fruit1_to_bin = self._fruits_ycb_objs[1].pose.p - self.bin.pose.p

        dist0 = torch.linalg.norm(fruit0_to_bin[:, :2], dim=-1)
        dist1 = torch.linalg.norm(fruit1_to_bin[:, :2], dim=-1)

        is_fruit0_placed = dist0 <= self.goal_thresh
        is_fruit1_placed = dist1 <= self.goal_thresh

        is_robot_static = (
            self.left_agent.is_static(0.2)
            & self.right_agent.is_static(0.2)
        )

        success = is_fruit0_placed & is_fruit1_placed

        result = dict(
            is_grasped=is_grasped,
            is_grasped_left=is_grasped_left,
            is_grasped_right=is_grasped_right,
            is_fruit0_placed=is_fruit0_placed,
            is_fruit1_placed=is_fruit1_placed,
            is_robot_static=is_robot_static,
            success=success,
        )

        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())

        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                fruit0_pose=self._fruits_ycb_objs[0].pose.raw_pose,
                fruit1_pose=self._fruits_ycb_objs[1].pose.raw_pose,
                bin_pose=self.bin.pose.raw_pose,
                left_tcp_to_fruit0=self._fruits_ycb_objs[0].pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_fruit1=self._fruits_ycb_objs[1].pose.p - self.right_agent.tcp.pose.p,
                fruit0_to_bin=self._fruits_ycb_objs[0].pose.p - self.bin.pose.p,
                fruit1_to_bin=self._fruits_ycb_objs[1].pose.p - self.bin.pose.p,
            )
        return obs

    # ═════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 8 sub-tasks (4 per arm), peak-tracked → [0, 1]
    #
    # Left arm (fruit 0):
    #   reach_0     — left TCP → fruit 0                     [0, 1]
    #   grasp_0     — proximity / confirmed grasp            [0, 1]
    #   transport_0 — fruit 0 → bin center                   [0, 1]
    #   place_0     — fruit 0 inside bin                     [0, 1]
    #
    # Right arm (fruit 1):
    #   reach_1     — right TCP → fruit 1                    [0, 1]
    #   grasp_1     — proximity / confirmed grasp            [0, 1]
    #   transport_1 — fruit 1 → bin center                   [0, 1]
    #   place_1     — fruit 1 inside bin                     [0, 1]
    #
    # Completion overrides guarantee success → total = 1.0
    # total = mean( peak(phase_i) for i in 1..8 ) → [0, 1]
    # ═════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p         # (B, 3)
        right_tcp = self.right_agent.tcp.pose.p       # (B, 3)
        fruit0_pos = self._fruits_ycb_objs[0].pose.p  # (B, 3)
        fruit1_pos = self._fruits_ycb_objs[1].pose.p  # (B, 3)
        bin_pos = self.bin.pose.p                      # (B, 3)

        is_grasped_left = info["is_grasped_left"]      # (B,) bool
        is_grasped_right = info["is_grasped_right"]    # (B,) bool
        is_fruit0_placed = info["is_fruit0_placed"]    # (B,) bool
        is_fruit1_placed = info["is_fruit1_placed"]    # (B,) bool
        success = info["success"]                      # (B,) bool

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Left arm: fruit 0 ───────────────────────────────────────
        r_reach_0 = reach_reward(left_tcp, fruit0_pos, scale=5.0)
        r_reach_0 = torch.where(is_grasped_left | is_fruit0_placed | success, ones, r_reach_0)

        r_grasp_0 = grasp_reward(left_tcp, fruit0_pos, is_grasped_left, proximity_scale=5.0)
        r_grasp_0 = torch.where(is_fruit0_placed | success, ones, r_grasp_0)

        r_transport_0 = transport_reward(fruit0_pos, bin_pos, is_grasped_left, scale=5.0)
        r_transport_0 = torch.where(is_fruit0_placed | success, ones, r_transport_0)

        r_place_0 = is_fruit0_placed.float()
        r_place_0 = torch.where(success, ones, r_place_0)

        # ── Right arm: fruit 1 ──────────────────────────────────────
        r_reach_1 = reach_reward(right_tcp, fruit1_pos, scale=5.0)
        r_reach_1 = torch.where(is_grasped_right | is_fruit1_placed | success, ones, r_reach_1)

        r_grasp_1 = grasp_reward(right_tcp, fruit1_pos, is_grasped_right, proximity_scale=5.0)
        r_grasp_1 = torch.where(is_fruit1_placed | success, ones, r_grasp_1)

        r_transport_1 = transport_reward(fruit1_pos, bin_pos, is_grasped_right, scale=5.0)
        r_transport_1 = torch.where(is_fruit1_placed | success, ones, r_transport_1)

        r_place_1 = is_fruit1_placed.float()
        r_place_1 = torch.where(success, ones, r_place_1)

        # ── Update tracker ──────────────────────────────────────────
        self.reward_tracker.update("reach_0", r_reach_0)
        self.reward_tracker.update("grasp_0", r_grasp_0)
        self.reward_tracker.update("transport_0", r_transport_0)
        self.reward_tracker.update("place_0", r_place_0)

        self.reward_tracker.update("reach_1", r_reach_1)
        self.reward_tracker.update("grasp_1", r_grasp_1)
        self.reward_tracker.update("transport_1", r_transport_1)
        self.reward_tracker.update("place_1", r_place_1)

        # ── Diagnostics ─────────────────────────────────────────────
        self.reward_tracker.write_to_info(info)

        # ── Total = arithmetic mean of peaks → [0, 1] ──────────────
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)