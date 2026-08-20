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
    reach_reward,
    RewardTracker,
)

# Sub-task phase names for this task
REWARD_PHASES = ["reach", "press"]


@register_env("TwoRobotPressStapler-v1", max_episode_steps=100)
class TwoRobotPressStaplerEnv(BaseEnv):
    """
    Task: Move TCP to stapler and press it (make contact).

    Reward phases (from solve script):
      1. reach — TCP approaches stapler                           [0, 1]
      2. press — gripper links contact stapler (ever)             [0, 1]

    Completion overrides guarantee success → total = 1.0.
    total = mean( peak(phase_i) for i in 1..2 ) → [0, 1]
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.3

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
        self.stapler_modelname, self.stapler_model_id = apply_l2_robotwin_model(
            "048_stapler",
            model_id=1,
            override_name="048_stapler",
            override_id=0,
        )
        self.notebook_modelname, self.notebook_model_id = apply_l3_robotwin_model(
            "092_notebook",
            model_id=1,
            override_name="092_notebook",
            override_id=0,
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

        # Load stapler
        self.stapler_pose = sapien.Pose(
            p=[-0., -0.15, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi * 3/4)
        )
        stapler_actor_obj = create_actor(
            scene=self.scene,
            pose=self.stapler_pose,
            modelname=self.stapler_modelname,
            convex=True,
            model_id=self.stapler_model_id,
            is_static=True,
            scale=(0.13, 0.13, 0.13),
            replace_scale=True,
        )
        self.stapler = stapler_actor_obj.actor

        # Load file
        self.file_pose = sapien.Pose(
            p=[-0.13, 0.05, -0.02],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        file_actor_obj = create_actor(
            scene=self.scene,
            pose=self.file_pose,
            modelname=self.notebook_modelname,
            convex=True,
            model_id=self.notebook_model_id,
            replace_scale=True,
            scale=(0.18, 0.0001, 0.18),
        )
        self.file = file_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "stapler_ever_pressed_file"):
                self.stapler_ever_pressed_file = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.stapler_ever_pressed_file[env_idx] = False

            # Reward tracker (2 phases)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            xyz = torch.tensor(self.stapler_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(0.1, 0.1))
            stapler_z_rot = -np.pi * 3 / 4
            if is_lr_mirror_enabled():
                stapler_z_rot = np.pi - stapler_z_rot
            qs = torch.tensor(euler2quat(np.pi / 2, 0, stapler_z_rot)).repeat(b, 1)
            self.stapler.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.file_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.file_zs[env_idx]
            xyz = apply_l1_offset_xy(xyz, offset=(0.1, 0.1))
            qs = torch.tensor(self.file_pose.q).repeat(b, 1)
            self.file.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):
        self.stapler_zs = []
        collision_mesh = self.stapler.get_first_collision_mesh()
        self.stapler_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.stapler_zs = common.to_tensor(self.stapler_zs, device=self.device)

        self.file_zs = []
        collision_mesh = self.file.get_first_collision_mesh()
        self.file_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.file_zs = common.to_tensor(self.file_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        contact_thresh = 1e-2
        left_links = [
            self.left_agent.finger1_link,
            self.left_agent.finger2_link,
            self.left_agent.finger1pad_link,
            self.left_agent.finger2pad_link,
        ]
        right_links = [
            self.right_agent.finger1_link,
            self.right_agent.finger2_link,
            self.right_agent.finger1pad_link,
            self.right_agent.finger2pad_link,
        ]

        def _links_contact(links):
            valid_links = [link for link in links if link is not None]
            if not valid_links:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            contacts = []
            for link in valid_links:
                forces = self.scene.get_pairwise_contact_forces(link, self.stapler)
                contacts.append(torch.linalg.norm(forces, axis=1) > contact_thresh)
            return torch.stack(contacts, dim=0).any(dim=0)

        left_contact = _links_contact(left_links)
        right_contact = _links_contact(right_links)
        currently_reached = left_contact | right_contact
        stapler_to_file_dist = torch.linalg.norm(self.stapler.pose.p - self.left_agent.tcp.pose.p, axis=1)

        # FIX: use | and & instead of Python or/and for tensor booleans
        is_grasped = self.left_agent.is_grasping(self.file) | self.right_agent.is_grasping(self.file)

        self.stapler_ever_pressed_file = self.stapler_ever_pressed_file | currently_reached

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

        success = self.stapler_ever_pressed_file

        result = dict(
            is_grasped=is_grasped,
            stapler_to_file_dist=stapler_to_file_dist,
            stapler_ever_pressed_file=self.stapler_ever_pressed_file,
            is_robot_static=is_robot_static,
            success=success,
            is_obj_placed=self.stapler_ever_pressed_file,
        )

        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())

        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            stapler_pose=self.stapler.pose.raw_pose,
            is_grasped=info["is_grasped"],
            stapler_ever_pressed_file=info["stapler_ever_pressed_file"],
        )
        if "state" in self.obs_mode:
            obs.update(
                file_pose=self.file.pose.raw_pose,
                left_tcp_to_file_pos=self.file.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_file_pos=self.file.pose.p - self.right_agent.tcp.pose.p,
                file_to_stapler_pos=self.stapler.pose.p - self.file.pose.p,
            )
        return obs

    # ═════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 2 sub-tasks, peak-tracked, arithmetic mean → [0, 1]
    #
    #  Phase 1  REACH — TCP approaches stapler (closer arm)       [0, 1]
    #  Phase 2  PRESS — gripper contacts stapler (ever)           [0, 1]
    #
    #  Completion overrides guarantee success → total = 1.0
    #  total = mean( peak(phase_i) for i in 1..2 )
    # ═════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        stapler_pos = self.stapler.pose.p
        success = info["success"]  # = stapler_ever_pressed_file

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Phase 1: REACH — TCP → stapler (closer arm) ─────────────
        left_r = reach_reward(left_tcp, stapler_pos, scale=5.0)
        right_r = reach_reward(right_tcp, stapler_pos, scale=5.0)
        r_reach = torch.maximum(left_r, right_r)
        r_reach = torch.where(success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        # ── Phase 2: PRESS — ever contacted stapler ──────────────────
        r_press = success.float()
        self.reward_tracker.update("press", r_press)

        # ── Diagnostics ──────────────────────────────────────────────
        self.reward_tracker.write_to_info(info)

        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)