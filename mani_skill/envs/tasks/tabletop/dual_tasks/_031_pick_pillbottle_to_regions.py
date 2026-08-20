import numpy as np
import sapien
import torch
from typing import Any, Dict, Tuple

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.building.actors.robotwin import get_model_id
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_robotwin_model,
    is_l3_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
    exp_reward,
)

# Sub-task phases for this task
REWARD_PHASES = [
    "reach_pill_a", "grasp_pill_a", "transport_pill_a", "place_pill_a",
    "reach_pill_b", "grasp_pill_b", "transport_pill_b", "place_pill_b",
]

@register_env("TwoRobotPickPillToRegions-v1", max_episode_steps=200)
class TwoRobotPickPillToRegionsEnv(BaseEnv):
    """
    **Task Description:**
    Two panda_wristcam robots operate on a table.
    1) Pick pill A and place it into region A-1.
    2) Pick pill B and place it into region B-1.

    **Success Conditions:**
    - Pill A is in region A-1.
    - Pill B is in region B-1.
    - Both robots are static.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    PILL_MODEL_NAME = "080_pillbottle"

    pill_a_xy = (-0.25, -0.15)
    pill_b_xy = (-0.25, 0.15)
    region_a_xy = (0.10, -0.20)
    region_b_xy = (0.10, 0.20)

    region_half_size = (0.08, 0.08, 0.005)
    region_height_threshold = 0.20

    pill_scale = 1.0

    def __init__(
        self,
        *args,
        robot_uids: Tuple[str, str] = ("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise: float = 0.02,
        num_envs: int = 1,
        reconfiguration_freq=None,
        hi_res: bool = False,
        wrist_sensor: bool = False, 
        **kwargs
    ):
        self.hi_res = hi_res
        self.wrist_sensor = wrist_sensor
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.pill_a_modelname, self.pill_a_model_id = apply_l2_robotwin_model(
            self.PILL_MODEL_NAME,
            model_id=1,
            override_name=self.PILL_MODEL_NAME,
            override_id=2,
        )
        self.pill_b_modelname, self.pill_b_model_id = apply_l2_robotwin_model(
            self.PILL_MODEL_NAME,
            model_id=5,
            override_name=self.PILL_MODEL_NAME,
            override_id=4,
        )
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
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
        pose = sapien_utils.look_at([0.8, 0.0, 0.75], [0.0, 0.0, 0.25])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )
        self.agent.__init__(self.agent.agents, wrist_sensor=self.wrist_sensor)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.pill_a_pose = sapien.Pose(
            p=[self.pill_a_xy[0], self.pill_a_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0.0, 0.0),
        )
        self.pill_b_pose = sapien.Pose(
            p=[self.pill_b_xy[0], self.pill_b_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0.0, 0.0),
        )
        self.region_a_pose = sapien.Pose(
            p=[self.region_a_xy[0], self.region_a_xy[1], self.region_half_size[2]],
            q=euler2quat(0, 0, 0),
        )
        self.region_b_pose = sapien.Pose(
            p=[self.region_b_xy[0], self.region_b_xy[1], self.region_half_size[2]],
            q=euler2quat(0, 0, 0),
        )

        region_a_color = [0.9, 0.2, 0.2, 0.6] if is_l3_enabled() else [0.2, 0.7, 0.9, 0.6]
        self.region_a = actors.build_box(
            self.scene,
            half_sizes=self.region_half_size,
            color=region_a_color,
            name="region_a",
            body_type="static",
            add_collision=False,
            initial_pose=self.region_a_pose,
        )
        region_b_color = [0.2, 0.9, 0.2, 0.6] if is_l3_enabled() else [0.9, 0.7, 0.2, 0.6]
        self.region_b = actors.build_box(
            self.scene,
            half_sizes=self.region_half_size,
            color=region_b_color,
            name="region_b",
            body_type="static",
            add_collision=False,
            initial_pose=self.region_b_pose,
        )

        self.pill_a = create_actor(
            scene=self.scene,
            pose=self.pill_a_pose,
            modelname=self.pill_a_modelname,
            convex=True,
            model_id=get_model_id(self.pill_a_modelname, model_id=self.pill_a_model_id),
            scale=(self.pill_scale,) * 3,
            _idx_if_repeat=1,
            mass=0.5,
        ).actor
        self.pill_b = create_actor(
            scene=self.scene,
            pose=self.pill_b_pose,
            modelname=self.pill_b_modelname,
            convex=True,
            model_id=get_model_id(self.pill_b_modelname, model_id=self.pill_b_model_id),
            scale=(self.pill_scale,) * 3,
            _idx_if_repeat=2,
            mass=0.5,
        ).actor

    def _after_reconfigure(self, options: dict):
        def _compute_object_z(obj) -> float:
            collision_mesh = obj.get_first_collision_mesh()
            return -collision_mesh.bounding_box.bounds[0, 2]

        self._pill_a_z = common.to_tensor([_compute_object_z(self.pill_a)], device=self.device)
        self._pill_b_z = common.to_tensor([_compute_object_z(self.pill_b)], device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize reward tracker
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            region_a_xyz = torch.tensor(self.region_a_pose.p).repeat(b, 1)
            region_a_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            region_a_q = torch.tensor(self.region_a_pose.q).repeat(b, 1)
            self.region_a.set_pose(Pose.create_from_pq(p=region_a_xyz, q=region_a_q))

            region_b_xyz = torch.tensor(self.region_b_pose.p).repeat(b, 1)
            region_b_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            region_b_q = torch.tensor(self.region_b_pose.q).repeat(b, 1)
            self.region_b.set_pose(Pose.create_from_pq(p=region_b_xyz, q=region_b_q))

            pill_a_p = torch.tensor(self.pill_a_pose.p).repeat(b, 1)
            pill_a_p[:, :2] += (torch.rand((b, 2)) - 0.5) * torch.tensor([0.02, 0.02])
            pill_a_p[:, 2] = self._pill_a_z[0]
            pill_a_p = apply_l1_offset_xy(pill_a_p, offset=(0.05, -0.1))
            base_quat = euler2quat(np.pi / 2, 0.0, 0.0)
            pill_a_q = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.pill_a.set_pose(Pose.create_from_pq(p=pill_a_p, q=pill_a_q))

            pill_b_p = torch.tensor(self.pill_b_pose.p).repeat(b, 1)
            pill_b_p[:, :2] += (torch.rand((b, 2)) - 0.5) * torch.tensor([0.02, 0.02])
            pill_b_p[:, 2] = self._pill_b_z[0]
            pill_b_p = apply_l1_offset_xy(pill_b_p, offset=(0.05, 0.1))
            pill_b_q = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.pill_b.set_pose(Pose.create_from_pq(p=pill_b_p, q=pill_b_q))

    def _is_in_region(self, obj_pos: torch.Tensor, region_pos: torch.Tensor) -> torch.Tensor:
        delta = obj_pos[..., :2] - region_pos[..., :2]
        in_xy = torch.logical_and(
            torch.abs(delta[..., 0]) <= self.region_half_size[0],
            torch.abs(delta[..., 1]) <= self.region_half_size[1],
        )
        in_z = obj_pos[..., 2] <= self.region_height_threshold
        return torch.logical_and(in_xy, in_z)

    def evaluate(self):
        region_a_pos = self.region_a.pose.p
        region_b_pos = self.region_b.pose.p
        pill_a_in_region = self._is_in_region(self.pill_a.pose.p, region_a_pos)
        pill_b_in_region = self._is_in_region(self.pill_b.pose.p, region_b_pos)

        is_pill_a_grasped = self.left_agent.is_grasping(self.pill_a)
        is_pill_b_grasped = self.right_agent.is_grasping(self.pill_b)

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        success = (
            pill_a_in_region
            & pill_b_in_region
        )

        result = dict(
            pill_a_in_region=pill_a_in_region,
            pill_b_in_region=pill_b_in_region,
            is_pill_a_grasped=is_pill_a_grasped,
            is_pill_b_grasped=is_pill_b_grasped,
            is_robot_static=is_robot_static,
            success=success,
        )
        # Append per-phase peak sub-rewards
        # if hasattr(self, "reward_tracker"):
        #     result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            region_a_pose=self.region_a.pose.raw_pose,
            region_b_pose=self.region_b.pose.raw_pose,
            pill_a_pose=self.pill_a.pose.raw_pose,
            pill_b_pose=self.pill_b.pose.raw_pose,
            pill_a_in_region=info["pill_a_in_region"],
            pill_b_in_region=info["pill_b_in_region"],
            # is_pill_a_upright=info["is_pill_a_upright"],
            # is_pill_b_upright=info["is_pill_b_upright"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        pill_a_pos = self.pill_a.pose.p
        pill_b_pos = self.pill_b.pose.p
        region_a_pos = self.region_a.pose.p
        region_b_pos = self.region_b.pose.p

        is_pill_a_grasped = info["is_pill_a_grasped"]
        is_pill_b_grasped = info["is_pill_b_grasped"]
        pill_a_in_region = info["pill_a_in_region"]
        pill_b_in_region = info["pill_b_in_region"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Left arm: pill_a
        # Phase 1: REACH
        r_reach_pill_a = reach_reward(left_tcp, pill_a_pos, scale=5.0)
        r_reach_pill_a = torch.where(is_pill_a_grasped | pill_a_in_region | success, ones, r_reach_pill_a)
        self.reward_tracker.update("reach_pill_a", r_reach_pill_a)

        # Phase 2: GRASP
        r_grasp_pill_a = grasp_reward(left_tcp, pill_a_pos, is_pill_a_grasped, proximity_scale=5.0)
        r_grasp_pill_a = torch.where(pill_a_in_region | success, ones, r_grasp_pill_a)
        self.reward_tracker.update("grasp_pill_a", r_grasp_pill_a)

        # Phase 3: TRANSPORT
        r_transport_pill_a = transport_reward(pill_a_pos, region_a_pos, is_pill_a_grasped, scale=5.0)
        r_transport_pill_a = torch.where(pill_a_in_region | success, ones, r_transport_pill_a)
        self.reward_tracker.update("transport_pill_a", r_transport_pill_a)

        # Phase 4: PLACE - based on distance to region (after transport started and released)
        transport_peak_a = self.reward_tracker._peaks["transport_pill_a"]
        pill_a_to_region_dist = torch.linalg.norm(pill_a_pos[..., :2] - region_a_pos[..., :2], dim=-1)
        r_place_pill_a = exp_reward(pill_a_to_region_dist, scale=5.0) * (transport_peak_a > 0).float() * (~is_pill_a_grasped).float()
        r_place_pill_a = torch.where(pill_a_in_region | success, ones, r_place_pill_a)
        self.reward_tracker.update("place_pill_a", r_place_pill_a)

        # Right arm: pill_b
        # Phase 1: REACH
        r_reach_pill_b = reach_reward(right_tcp, pill_b_pos, scale=5.0)
        r_reach_pill_b = torch.where(is_pill_b_grasped | pill_b_in_region | success, ones, r_reach_pill_b)
        self.reward_tracker.update("reach_pill_b", r_reach_pill_b)

        # Phase 2: GRASP
        r_grasp_pill_b = grasp_reward(right_tcp, pill_b_pos, is_pill_b_grasped, proximity_scale=5.0)
        r_grasp_pill_b = torch.where(pill_b_in_region | success, ones, r_grasp_pill_b)
        self.reward_tracker.update("grasp_pill_b", r_grasp_pill_b)

        # Phase 3: TRANSPORT
        r_transport_pill_b = transport_reward(pill_b_pos, region_b_pos, is_pill_b_grasped, scale=5.0)
        r_transport_pill_b = torch.where(pill_b_in_region | success, ones, r_transport_pill_b)
        self.reward_tracker.update("transport_pill_b", r_transport_pill_b)

        # Phase 4: PLACE - based on distance to region (after transport started and released)
        transport_peak_b = self.reward_tracker._peaks["transport_pill_b"]
        pill_b_to_region_dist = torch.linalg.norm(pill_b_pos[..., :2] - region_b_pos[..., :2], dim=-1)
        r_place_pill_b = exp_reward(pill_b_to_region_dist, scale=5.0) * (transport_peak_b > 0).float() * (~is_pill_b_grasped).float()
        r_place_pill_b = torch.where(pill_b_in_region | success, ones, r_place_pill_b)
        self.reward_tracker.update("place_pill_b", r_place_pill_b)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
