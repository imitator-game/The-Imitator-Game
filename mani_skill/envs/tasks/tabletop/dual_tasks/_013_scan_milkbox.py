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
    apply_l2_robotwin_config,
    apply_l3_robotwin_model,
)
from mani_skill.utils import common
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    grasp_reward,
    reach_reward,
    transport_reward,
)


REWARD_PHASES = [
    "reach_scanner",
    "grasp_scanner",
    "reach_milkbox",
    "grasp_milkbox",
    "bring_milkbox",
    "scan",
]


@register_env("TwoRobotScanMilkBox-v1", max_episode_steps=200)
class TwoRobotScanMilkBoxEnv(BaseEnv):
    """
    **Task Description:**
    The goal is for the left robot to pick up a scanner and the right robot to pick up a milk box.
    Then use the scanner to scan the milk box. There are two robots in this task:
    - Left robot at position [-0.9, 0, 0] - picks up scanner
    - Right robot at position [-0.3, 0, 0] - picks up milk box

    Both robots face the table and need to coordinate for the scanning task.

    **Randomizations:**
    - Object positions are fixed (not randomized)
    - Objects have fixed upright orientation with X-axis rotation

    **Success Conditions:**
    - Both scanner and milk box are grasped
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
        self.scanner_mass = 0.1
        self.milkbox_mass = 0.1

        # Scanner (robotwin object, for left robot to grasp)
        self.scanner_modelname, self.scanner_model_id = apply_l3_robotwin_model(
            "024_scanner",
            model_id=0,
            override_name="024_scanner",
            override_id=3,
        )

        # Milk Box (robotwin object, for right robot to grasp)
        (
            self.milkbox_modelname,
            self.milkbox_model_id,
            self.milkbox_scale,
            self.milkbox_replace_scale,
        ) = apply_l2_robotwin_config(
            "038_milk-box",
            model_id=0,
            base_scale=(10, 10, 10),
            base_replace_scale=False,
            override_name="038_milk-box",
            override_id=2,
            override_scale=(6, 6, 6),
            override_replace_scale=False,
        )

        self.goal_thresh = 0.15

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
        # Use euler2quat(pi/2, 0, 0) to make object upright (X-axis rotation)
        scanner_pose = sapien.Pose(
            p=[0, 0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        scanner_obj = create_actor(
            scene=self.scene,
            pose=scanner_pose,
            modelname=self.scanner_modelname,
            convex=True,
            model_id=self.scanner_model_id,
            mass=self.scanner_mass, 
        )
        self.scanner = scanner_obj.actor

        # Load milk box (robotwin object, right robot will grasp this)
        milkbox_pose = sapien.Pose(
            p=[0, -0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        milkbox_obj = create_actor(
            scene=self.scene,
            pose=milkbox_pose,
            modelname=self.milkbox_modelname,
            scale=self.milkbox_scale,  # Scale milk box
            convex=True,
            model_id=self.milkbox_model_id,
            replace_scale=self.milkbox_replace_scale,
            mass=self.milkbox_mass, 
        )
        self.milkbox = milkbox_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    REWARD_PHASES, self.num_envs, self.device
                )
            self.reward_tracker.reset(env_idx)

            if not hasattr(self, "scanner_ever_reached_box"):
                self.scanner_ever_reached_box = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.scanner_ever_reached_box[env_idx] = False

            # Initialize scanner (left robot's object) - fixed position on the left
            scanner_xyz = torch.zeros((b, 3), device=self.device)
            scanner_xyz[:, 0] = -0.1  # X position
            scanner_xyz[:, 1] = -0.15  # Y position - left side
            scanner_xyz[:, 2] = self.scanner_z  # Z position (on table surface)
            scanner_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02 
            scanner_xyz = apply_l1_offset_xy(scanner_xyz, offset=(-0.1, -0.1))

            # Fixed upright orientation with horizontal rotation
            z_rotation = np.pi / 2
            base_quat = euler2quat(np.pi/2, 0.0, z_rotation)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.scanner.set_pose(Pose.create_from_pq(p=scanner_xyz, q=qs))

            # Initialize milk box (right robot's object) - fixed position on the right
            milkbox_xyz = torch.zeros((b, 3), device=self.device)
            milkbox_xyz[:, 0] = -0.1  # X position
            milkbox_xyz[:, 1] = 0.15  # Y position - right side
            milkbox_xyz[:, 2] = self.milkbox_z  # Z position (on table surface)
            milkbox_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            milkbox_xyz = apply_l1_offset_xy(milkbox_xyz, offset=(-0.1, 0.1))

            # Fixed upright orientation
            base_quat = euler2quat(np.pi/2, 0.0, 0.0)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.milkbox.set_pose(Pose.create_from_pq(milkbox_xyz, qs))

    def _after_reconfigure(self, options: dict):
        # Get z-offset for scanner to place it on table surface
        collision_mesh = self.scanner.get_first_collision_mesh()

        self.scanner_z = 0.0  # Default height if no collision mesh

        # Get z-offset for milk box to place it on table surface
        collision_mesh = self.milkbox.get_first_collision_mesh()
        self.milkbox_z = 0.0  # Default height if no collision mesh

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        if not hasattr(self, "_geom_center_fallback_logged"):
            self._geom_center_fallback_logged = set()
        try:
            return geom_center_from_local_mesh(obj, self.device)
        except Exception as exc:
            name = getattr(obj, "name", type(obj).__name__)
            if name not in self._geom_center_fallback_logged:
                print(
                    f"[{self.__class__.__name__}] geom center fallback to pose.p for {name}: {exc}"
                )
                self._geom_center_fallback_logged.add(name)
            return obj.pose.p.clone()

    def evaluate(self):
        scanner_pos = self._get_actor_geom_center(self.scanner)
        milkbox_pos = self._get_actor_geom_center(self.milkbox)
        scanner_to_box_dist = torch.linalg.norm(milkbox_pos - scanner_pos, dim=-1)
        is_scanner_at_box = scanner_to_box_dist <= self.goal_thresh
        self.scanner_ever_reached_box = self.scanner_ever_reached_box | is_scanner_at_box
        is_left_static = self.left_agent.is_static(0.2)
        is_right_static = self.right_agent.is_static(0.2)
        is_robot_static = is_left_static & is_right_static
        success = self.scanner_ever_reached_box
        result = dict(
            scanner_to_box_dist=scanner_to_box_dist,
            scanner_ever_reached_box=self.scanner_ever_reached_box,
            is_robot_static=is_robot_static,
            success=success,
            is_scanner_grasped=self.left_agent.is_grasping(self.scanner),
            is_milkbox_grasped=self.right_agent.is_grasping(self.milkbox),
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            is_scanner_grasped=info["is_scanner_grasped"],
            is_milkbox_grasped=info["is_milkbox_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                scanner_pose=self.scanner.pose.raw_pose,
                milkbox_pose=self.milkbox.pose.raw_pose,
                left_tcp_to_scanner_pos=self.scanner.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_milkbox_pos=self.milkbox.pose.p - self.right_agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        scanner_pos = self._get_actor_geom_center(self.scanner)
        milkbox_pos = self._get_actor_geom_center(self.milkbox)
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        is_scanner_grasped = info["is_scanner_grasped"]
        is_milkbox_grasped = info["is_milkbox_grasped"]
        scanned = info["scanner_ever_reached_box"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_scanner = reach_reward(left_tcp, scanner_pos, scale=5.0)
        r_reach_scanner = torch.where(is_scanner_grasped | scanned | success, ones, r_reach_scanner)
        r_grasp_scanner = grasp_reward(left_tcp, scanner_pos, is_scanner_grasped, proximity_scale=5.0)
        r_grasp_scanner = torch.where(scanned | success, ones, r_grasp_scanner)
        r_reach_milkbox = reach_reward(right_tcp, milkbox_pos, scale=5.0)
        r_reach_milkbox = torch.where(is_milkbox_grasped | scanned | success, ones, r_reach_milkbox)
        r_grasp_milkbox = grasp_reward(right_tcp, milkbox_pos, is_milkbox_grasped, proximity_scale=5.0)
        r_grasp_milkbox = torch.where(scanned | success, ones, r_grasp_milkbox)
        r_bring_milkbox = transport_reward(milkbox_pos, scanner_pos, is_milkbox_grasped, scale=5.0)
        r_bring_milkbox = torch.where(scanned | success, ones, r_bring_milkbox)
        r_scan = torch.where(success, ones, scanned.float())

        self.reward_tracker.update("reach_scanner", r_reach_scanner)
        self.reward_tracker.update("grasp_scanner", r_grasp_scanner)
        self.reward_tracker.update("reach_milkbox", r_reach_milkbox)
        self.reward_tracker.update("grasp_milkbox", r_grasp_milkbox)
        self.reward_tracker.update("bring_milkbox", r_bring_milkbox)
        self.reward_tracker.update("scan", r_scan)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
