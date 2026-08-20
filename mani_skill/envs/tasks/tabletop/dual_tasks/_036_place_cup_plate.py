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
    is_l2_enabled,
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
    "reach_cup_1", "grasp_cup_1", "transport_cup_1", "place_cup_1",
    "reach_cup_2", "grasp_cup_2", "transport_cup_2", "place_cup_2",
]


@register_env("TwoRobotPlaceCupPlate-v1", max_episode_steps=100)
class TwoRobotPlaceCupPlateEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.15

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
        self.cup_1_modelname, self.cup_1_model_id = apply_l2_robotwin_model(
            "022_cup-with-liquid",
            model_id=0,
            override_name="039_mug",
            override_id=5,
        )
        self.cup_2_modelname, self.cup_2_model_id = apply_l2_robotwin_model(
            "021_cup",
            model_id=6,
            override_name="021_cup",
            override_id=4,
        )
        self.tray_modelname, self.tray_model_id = apply_l3_robotwin_model(
            "008_tray",
            model_id=3,
            override_name="008_tray",
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

        # Load cup_1
        self.cup_1_pose = sapien.Pose(
            p=[-0., -0.35, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        cup_1_actor_obj = create_actor(
            scene=self.scene,
            pose=self.cup_1_pose,
            modelname=self.cup_1_modelname,
            convex=True,
            model_id=self.cup_1_model_id,
            scale=(0.04, 0.06, 0.04),
            replace_scale=True,
            _idx_if_repeat=0,
        )
        cup_1_actor_obj.set_mass(0.1)
        self.cup_1 = cup_1_actor_obj.actor

        # Load cup_2
        self.cup_2_pose = sapien.Pose(
            p=[-0., 0.35, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 2)
        )
        cup_2_actor_obj = create_actor(
            scene=self.scene,
            pose=self.cup_2_pose,
            modelname=self.cup_2_modelname,
            convex=True,
            model_id=self.cup_2_model_id,
            scale=(0.05, 0.05, 0.05),
            replace_scale=True,
            _idx_if_repeat=1,
        )
        cup_2_actor_obj.set_mass(0.1)
        self.cup_2 = cup_2_actor_obj.actor

        # Load tray
        self.tray_pose = sapien.Pose(
            p=[0.0, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 2)
        )
        tray_actor_obj = create_actor(
            scene=self.scene,
            pose=self.tray_pose,
            modelname=self.tray_modelname,
            convex=True,
            model_id=self.tray_model_id,
            is_static=True,
            replace_scale=True,
            scale=(0.2, 0.2, 0.2),
        )
        self.tray = tray_actor_obj.actor

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

            xyz = torch.tensor(self.cup_1_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
            qs = torch.tensor(self.cup_1_pose.q).repeat(b, 1)
            self.cup_1.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.cup_2_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.1))
            qs = torch.tensor(self.cup_2_pose.q).repeat(b, 1)
            self.cup_2.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.tray_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.tray_zs[env_idx]
            qs = torch.tensor(self.tray_pose.q).repeat(b, 1)
            self.tray.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.cup_1_zs = []
        collision_mesh = self.cup_1.get_first_collision_mesh()
        self.cup_1_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cup_1_zs = common.to_tensor(self.cup_1_zs, device=self.device)

        self.cup_2_zs = []
        collision_mesh = self.cup_2.get_first_collision_mesh()
        self.cup_2_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cup_2_zs = common.to_tensor(self.cup_2_zs, device=self.device)

        self.tray_zs = []
        collision_mesh = self.tray.get_first_collision_mesh()
        self.tray_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.tray_zs = common.to_tensor(self.tray_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        cup_1_to_tray_dist = torch.linalg.norm(self.cup_1.pose.p[:, :2] - self.tray.pose.p[:, :2], axis=1)
        cup_2_to_tray_dist = torch.linalg.norm(self.cup_2.pose.p[:, :2] - self.tray.pose.p[:, :2], axis=1)

        is_cup_1_placed = cup_1_to_tray_dist <= self.goal_thresh
        is_cup_2_placed = cup_2_to_tray_dist <= self.goal_thresh

        # Upright check: local Z-axis should align with world Z-axis
        # Models are loaded with pi/2 rotation around X-axis
        cup_1_up_vector = self.cup_1.pose.to_transformation_matrix()[:, :3, 2]
        is_cup_1_upright = -cup_1_up_vector[:, 0] > 0.95
        cup_2_up_vector = self.cup_2.pose.to_transformation_matrix()[:, :3, 2]
        is_cup_2_upright = cup_2_up_vector[:, 0] > 0.95

        is_cup_1_grasped = self.left_agent.is_grasping(self.cup_1)
        is_cup_2_grasped = self.right_agent.is_grasping(self.cup_2)

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

        success = (
            is_cup_1_placed
            & is_cup_2_placed
        )

        result = dict(
            is_cup_1_placed=is_cup_1_placed,
            is_cup_2_placed=is_cup_2_placed,
            is_cup_1_upright=is_cup_1_upright,
            is_cup_2_upright=is_cup_2_upright,
            is_cup_1_grasped=is_cup_1_grasped,
            is_cup_2_grasped=is_cup_2_grasped,
            is_robot_static=is_robot_static,
            success=success,
        )
        # Append per-phase peak sub-rewards
        # if hasattr(self, "reward_tracker"):
        #     # result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            tray_pos=self.tray.pose.p,
            is_cup_1_placed=info["is_cup_1_placed"],
            is_cup_2_placed=info["is_cup_2_placed"],
            is_cup_1_upright=info["is_cup_1_upright"],
            is_cup_2_upright=info["is_cup_2_upright"],
        )
        if "state" in self.obs_mode:
            obs.update(
                cup_1_pose=self.cup_1.pose.raw_pose,
                cup_2_pose=self.cup_2.pose.raw_pose,
                left_tcp_to_cup_1_pos=self.cup_1.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_cup_2_pos=self.cup_2.pose.p - self.right_agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        cup_1_pos = self.cup_1.pose.p
        cup_2_pos = self.cup_2.pose.p
        tray_pos = self.tray.pose.p
        tray_pos_cup_1 = tray_pos + torch.tensor([0.0, -0.07, 0.0], device=self.device)
        tray_pos_cup_2 = tray_pos + torch.tensor([0.0, 0.07, 0.0], device=self.device)
        if is_l2_enabled():
            tray_pos_cup_1 = tray_pos + torch.tensor([0.0, 0.07, 0.0], device=self.device)
            tray_pos_cup_2 = tray_pos + torch.tensor([0.0, -0.07, 0.0], device=self.device)

        is_cup_1_grasped = info["is_cup_1_grasped"]
        is_cup_2_grasped = info["is_cup_2_grasped"]
        is_cup_1_placed = info["is_cup_1_placed"]
        is_cup_2_placed = info["is_cup_2_placed"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Left arm: cup_1 - with proper gating
        # Phase 1: REACH - only before grasping
        reach_gate_1 = ~is_cup_1_grasped
        r_reach_cup_1 = reach_reward(left_tcp, cup_1_pos, scale=5.0) * reach_gate_1.float()
        r_reach_cup_1 = torch.where(is_cup_1_grasped, ones, r_reach_cup_1)
        self.reward_tracker.update("reach_cup_1", r_reach_cup_1)

        # Phase 2: GRASP - only after reaching, before placed
        grasp_gate_1 = is_cup_1_grasped 
        r_grasp_cup_1 = grasp_reward(left_tcp, cup_1_pos, is_cup_1_grasped, proximity_scale=5.0) * grasp_gate_1.float()
        self.reward_tracker.update("grasp_cup_1", r_grasp_cup_1)

        # Phase 3: TRANSPORT - only after grasping, before placed
        transport_gate_1 = is_cup_1_grasped
        r_transport_cup_1 = transport_reward(cup_1_pos, tray_pos_cup_1, is_cup_1_grasped, scale=5.0) * transport_gate_1.float()
        r_transport_cup_1 = torch.where(is_cup_1_placed, ones, r_transport_cup_1)
        self.reward_tracker.update("transport_cup_1", r_transport_cup_1)

        # Phase 4: PLACE - continuous reward (after transport started and released)
        transport_peak_1 = self.reward_tracker._peaks["transport_cup_1"]
        cup_1_to_tray_dist = torch.linalg.norm(cup_1_pos - tray_pos_cup_1, dim=-1)
        r_place_cup_1 = exp_reward(cup_1_to_tray_dist, scale=5.0) * (transport_peak_1 > 0).float() * (~is_cup_1_grasped).float()
        r_place_cup_1 = torch.where(is_cup_1_placed, ones, r_place_cup_1)
        self.reward_tracker.update("place_cup_1", r_place_cup_1)

        # Right arm: cup_2 - with proper gating
        # Phase 1: REACH - only before grasping
        reach_gate_2 = ~is_cup_2_grasped
        r_reach_cup_2 = reach_reward(right_tcp, cup_2_pos, scale=5.0) * reach_gate_2.float()
        r_reach_cup_2 = torch.where(is_cup_2_grasped, ones, r_reach_cup_2)
        self.reward_tracker.update("reach_cup_2", r_reach_cup_2)

        # Phase 2: GRASP - only after reaching, before placed
        grasp_gate_2 = is_cup_2_grasped 
        r_grasp_cup_2 = grasp_reward(right_tcp, cup_2_pos, is_cup_2_grasped, proximity_scale=5.0) * grasp_gate_2.float()
        self.reward_tracker.update("grasp_cup_2", r_grasp_cup_2)

        # Phase 3: TRANSPORT - only after grasping, before placed
        transport_gate_2 = is_cup_2_grasped 
        r_transport_cup_2 = transport_reward(cup_2_pos, tray_pos_cup_2, is_cup_2_grasped, scale=5.0) * transport_gate_2.float()
        r_transport_cup_2 = torch.where(is_cup_2_placed, ones, r_transport_cup_2)
        self.reward_tracker.update("transport_cup_2", r_transport_cup_2)

        # Phase 4: PLACE - continuous reward (after transport started and released)
        transport_peak_2 = self.reward_tracker._peaks["transport_cup_2"]
        cup_2_to_tray_dist = torch.linalg.norm(cup_2_pos - tray_pos_cup_2, dim=-1)
        r_place_cup_2 = exp_reward(cup_2_to_tray_dist, scale=5.0) * (transport_peak_2 > 0).float() * (~is_cup_2_grasped).float()
        r_place_cup_2 = torch.where(is_cup_2_placed, ones, r_place_cup_2)
        self.reward_tracker.update("place_cup_2", r_place_cup_2)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
