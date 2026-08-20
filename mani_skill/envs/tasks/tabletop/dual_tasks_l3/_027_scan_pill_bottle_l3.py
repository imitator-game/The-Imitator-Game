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
)

from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    tanh_reward,
)

# Sub-task phases for this task (L3 version: only right robot moves)
REWARD_PHASES = [
    "pick_pillbottle",
    "scan",
]


@register_env("TwoRobotScanPillBottleL3-v1", max_episode_steps=200)
class TwoRobotScanPillBottleEnvL3(BaseEnv):
    """
    **Task Description:**
    The goal is for the left robot to pick up a scanner and the right robot to pick up a medicine bottle.
    Then use the scanner to scan the pill bottle. There are two robots in this task:
    - Left robot at position [-0.9, 0, 0] - picks up scanner
    - Right robot at position [-0.3, 0, 0] - picks up pill bottle

    Both robots face the table and need to coordinate for the scanning task.

    **Randomizations:**
    - Object positions are fixed (not randomized)
    - Objects have fixed upright orientation with X-axis rotation

    **Success Conditions:**
    - Both scanner and pill bottle are grasped
    - Both objects are static (placed down)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

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

        # Scanner (robotwin object, for left robot to grasp)
        self.scanner_modelname, self.scanner_model_id = "024_scanner", 3 

        # Pill Bottle (robotwin object, for right robot to grasp)
        self.pillbottle_modelname, self.pillbottle_model_id = "080_pillbottle", 3

        self.goal_thresh = 0.25

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

        # Load scanner (robotwin object, left robot will grasp this)
        scanner_pose = sapien.Pose(
            p=[0, 0.5, 0],
            q=euler2quat(0.0, np.pi, -np.pi / 2)
        )
        scanner_obj = create_actor(
            scene=self.scene,
            pose=scanner_pose,
            modelname=self.scanner_modelname,
            convex=True,
            model_id=self.scanner_model_id,
            is_static=True,
        )
        self.scanner = scanner_obj.actor

        # Load pill bottle (robotwin object, right robot will grasp this)
        pillbottle_pose = sapien.Pose(
            p=[0, -0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        pillbottle_obj = create_actor(
            scene=self.scene,
            pose=pillbottle_pose,
            modelname=self.pillbottle_modelname,
            convex=True,
            model_id=self.pillbottle_model_id,
        )
        self.pillbottle = pillbottle_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "scanner_ever_reached_bottle"):
                self.scanner_ever_reached_bottle = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.scanner_ever_reached_bottle[env_idx] = False

            # Initialize reward tracker
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            # Initialize scanner (left robot's object) - fixed position on the left
            scanner_xyz = torch.zeros((b, 3), device=self.device)
            scanner_xyz[:, 0] = -0.1
            scanner_xyz[:, 1] = -0.15
            scanner_xyz[:, 2] = self.scanner_z
            scanner_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            scanner_xyz[:, 2] += 0.01

            # Fixed upright orientation with horizontal rotation
            base_quat = euler2quat(0.0, 0.0, np.pi)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.scanner.set_pose(Pose.create_from_pq(p=scanner_xyz, q=qs))

            # Initialize pill bottle (right robot's object) - fixed position on the right
            pillbottle_xyz = torch.zeros((b, 3), device=self.device)
            pillbottle_xyz[:, 0] = -0.1
            pillbottle_xyz[:, 1] = 0.15 
            pillbottle_xyz[:, 2] = self.pillbottle_z 
            pillbottle_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            pillbottle_xyz = apply_l1_offset_xy(pillbottle_xyz, offset=(-0.1, 0.1))

            # Fixed upright orientation
            base_quat = euler2quat(np.pi/2, 0.0, 0.0)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.pillbottle.set_pose(Pose.create_from_pq(pillbottle_xyz, qs))

    def _after_reconfigure(self, options: dict):
        # Get z-offset for scanner to place it on table surface
        collision_mesh = self.scanner.get_first_collision_mesh()
        if collision_mesh is not None:
            self.scanner_z = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            self.scanner_z = 0.02
        # Get z-offset for pill bottle to place it on table surface
        collision_mesh = self.pillbottle.get_first_collision_mesh()
        if collision_mesh is not None:
            self.pillbottle_z = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            self.pillbottle_z = 0.02

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        # 1. Check if scanner is currently near pill bottle (to update historical reach flag)
        scanner_to_bottle_dist = torch.linalg.norm(self.pillbottle.pose.p - self.scanner.pose.p, axis=1)
        is_scanner_at_bottle = scanner_to_bottle_dist <= self.goal_thresh
        self.scanner_ever_reached_bottle = self.scanner_ever_reached_bottle | is_scanner_at_bottle

        # 2. Check orientation: Pill bottle's local Z-axis should point towards World Z-axis [0, 0, 1]
        # Get the world-frame direction of the object's local Z-axis (3rd column of rotation matrix)
        pillbottle_up_vector = self.pillbottle.pose.to_transformation_matrix()[:, :3, 2]
        # Check if the Z-component is close to 1 (within ~18 degrees)
        is_pillbottle_upright = -pillbottle_up_vector[:, 1] > 0.95

        # 3. Check if robots are static
        is_left_static = self.left_agent.is_static(0.2)
        is_right_static = self.right_agent.is_static(0.2)
        is_robot_static = is_left_static and is_right_static

        # 4. Success Conditions: Ever reached bottle AND upright AND robots are static
        success = torch.logical_and(self.scanner_ever_reached_bottle, is_pillbottle_upright)

        result = dict(
            scanner_to_bottle_dist=scanner_to_bottle_dist,
            scanner_ever_reached_bottle=self.scanner_ever_reached_bottle,
            is_pillbottle_upright=is_pillbottle_upright,
            is_robot_static=is_robot_static,
            success=success,
            # For backward compatibility or extra info
            is_scanner_grasped=self.left_agent.is_grasping(self.scanner),
            is_pillbottle_grasped=self.right_agent.is_grasping(self.pillbottle),
        )
        # Append per-phase peak sub-rewards
        # if hasattr(self, "reward_tracker"):
        #     # result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            is_scanner_grasped=info["is_scanner_grasped"],
            is_pillbottle_grasped=info["is_pillbottle_grasped"],
            scanner_ever_reached_bottle=info["scanner_ever_reached_bottle"],
        )
        if "state" in self.obs_mode:
            obs.update(
                scanner_pose=self.scanner.pose.raw_pose,
                pillbottle_pose=self.pillbottle.pose.raw_pose,
                left_tcp_to_scanner_pos=self.scanner.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_pillbottle_pos=self.pillbottle.pose.p - self.right_agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        right_tcp = self.right_agent.tcp.pose.p
        scanner_pos = self.scanner.pose.p
        pillbottle_pos = self.pillbottle.pose.p

        is_pillbottle_grasped = info["is_pillbottle_grasped"]
        scanner_ever_reached = info["scanner_ever_reached_bottle"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Phase 1: PILLBOTTLE - right TCP approaches and grasps pillbottle
        r_pick_pillbottle = reach_reward(right_tcp, pillbottle_pos, scale=5.0)
        r_pick_pillbottle = torch.where(is_pillbottle_grasped, ones, r_pick_pillbottle)
        self.reward_tracker.update("pick_pillbottle", r_pick_pillbottle)

        # Phase 2: SCAN - pillbottle close to scanner
        scanner_to_bottle_dist = torch.linalg.norm(scanner_pos - pillbottle_pos, dim=-1) - 0.15
        r_scan = tanh_reward(scanner_to_bottle_dist, scale=9.0) 
        r_scan = torch.where(success, ones, r_scan)
        self.reward_tracker.update("scan", r_scan)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
