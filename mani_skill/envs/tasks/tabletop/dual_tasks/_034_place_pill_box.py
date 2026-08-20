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
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig
from mani_skill.utils.structs.types import SimConfig
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_robotwin_model,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
    exp_reward,
)

# Sub-task phases for this task
REWARD_PHASES = ["reach_pill", "grasp_pill", "transport_pill"]


@register_env("TwoRobotPlacePillBox-v1", max_episode_steps=200)
class TwoRobotPlacePillBoxEnv(BaseEnv):
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

    pill_xy = (-0., -0.35)
    box_xy = (0., -0.0)
    goal_thresh = 0.15

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
        self.pillbottle_modelname, self.pillbottle_model_id = apply_l2_robotwin_model(
            "080_pillbottle",
            model_id=3,
            override_name="080_pillbottle",
            override_id=5,
        )
        self.plasticbox_modelname, self.plasticbox_model_id = apply_l2_robotwin_model(
            "062_plasticbox", 
            model_id=0,
            override_name="062_plasticbox",
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

        self.pill_pose = sapien.Pose(
            p=[self.pill_xy[0], self.pill_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0.0, 0.0),
        )
        self.box_pose = sapien.Pose(
            p=[self.box_xy[0], self.box_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 2),
        )

        self.box = create_actor(
            scene=self.scene,
            pose=self.pill_pose,
            modelname=self.plasticbox_modelname,
            convex=True,
            model_id=self.plasticbox_model_id,
            is_static=True,
            scale=(1.5, 1.5, 1.5),
        ).actor

        self.pill = create_actor(
            scene=self.scene,
            pose=self.pill_pose,
            modelname=self.pillbottle_modelname,
            convex=True,
            model_id=self.pillbottle_model_id,
            scale=(0.6, 0.08, 0.6),
        ).actor

    def _after_reconfigure(self, options: dict):
        def _compute_object_z(obj) -> float:
            collision_mesh = obj.get_first_collision_mesh()
            return -collision_mesh.bounding_box.bounds[0, 2]

        self._pill_z = common.to_tensor([_compute_object_z(self.pill)], device=self.device)
        self._box_z = common.to_tensor([_compute_object_z(self.box)], device=self.device)

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

            box_xyz = torch.tensor(self.box_pose.p).repeat(b, 1)
            box_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            box_xyz[:, 2] = self._box_z[0]
            box_q = torch.tensor(self.box_pose.q).repeat(b, 1)
            self.box.set_pose(Pose.create_from_pq(p=box_xyz, q=box_q))

            pill_p = torch.tensor(self.pill_pose.p).repeat(b, 1)
            pill_p[:, :2] += torch.rand((b, 2)) * 0.02
            pill_p[:, 2] = self._pill_z[0]
            pill_p = apply_l1_offset_xy(pill_p, offset=(-0.1, -0.1))
            base_quat = euler2quat(np.pi / 2, 0.0, 0.0)
            pill_q = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.pill.set_pose(Pose.create_from_pq(p=pill_p, q=pill_q))

    def evaluate(self):
        obj_to_goal_pos = self.box.pose.p - self.pill.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_grasped = self.right_agent.is_grasping(self.pill)
        is_robot_static = self.left_agent.is_static(0.2) and self.right_agent.is_static(0.2)
        success = is_obj_placed

        result = dict(
            pill_in_region=is_obj_placed,
            is_grasped=is_grasped,
            is_robot_static=is_robot_static,
            is_obj_placed=is_obj_placed,
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
            box_pose=self.box.pose.raw_pose,
            pill_pose=self.pill.pose.raw_pose,
            is_obj_placed=info["is_obj_placed"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        pill_pos = self.pill.pose.p
        box_pos = self.box.pose.p

        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)
        tcp_pill_dist = torch.linalg.norm(left_tcp - pill_pos, dim=-1)
        is_pill_held = tcp_pill_dist < 0.05  # heuristic for whether pill is held, used for reward gating

        # Phase 1: REACH_PILL
        r_reach_pill = reach_reward(left_tcp, pill_pos, scale=5.0)
        r_reach_pill = torch.where(is_pill_held, ones, r_reach_pill)
        self.reward_tracker.update("reach_pill", r_reach_pill)

        # Phase 2: GRASP_PILL
        r_grasp_pill = grasp_reward(left_tcp, pill_pos, is_pill_held, proximity_scale=5.0)
        self.reward_tracker.update("grasp_pill", r_grasp_pill)

        # Phase 3: TRANSPORT_PILL
        r_transport_pill = transport_reward(pill_pos, box_pos, is_pill_held, scale=5.0)
        r_transport_pill = torch.where(success, ones, r_transport_pill)
        self.reward_tracker.update("transport_pill", r_transport_pill)

        # Phase 4: PLACE_PILL (removed - user requested)
        # Place reward removed per user request

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
