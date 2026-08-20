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
    tanh_reward,
    transport_reward,
)

# Sub-task phases for this task
REWARD_PHASES = ["reach_pill", "grasp_pill", "pour_approach", "transport_pill"]


@register_env("TwoRobotPlacePillBoxL3-v1", max_episode_steps=200)
class TwoRobotPlacePillBoxEnvL3(BaseEnv):
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
    goal_thresh_height = 0.05
    goal_thresh_dist = 0.1

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
        self.plasticbox_modelname, self.plasticbox_model_id = "062_plasticbox", 0

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

        pill_actor_obj = create_actor(
            scene=self.scene,
            pose=self.pill_pose,
            modelname=self.pillbottle_modelname,
            convex=True,
            model_id=self.pillbottle_model_id,
            scale=(0.75, 0.75, 0.75),
        ) 
        self.pill = pill_actor_obj.actor
        self.pill.set_mass(0.1)

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

            if not hasattr(self, "ever_poured"):
                self.ever_poured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.ever_poured[env_idx] = False

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

            bias = torch.rand((b, 2)) * 0.02

            pill_p = torch.tensor(self.pill_pose.p).repeat(b, 1)
            pill_p[:, :2] += bias
            pill_p[:, 2] = self._pill_z[0] 
            pill_p = apply_l1_offset_xy(pill_p, offset=(-0.1, -0.1))
            base_quat = euler2quat(np.pi / 2, 0.0, 0.0)
            pill_q = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.pill.set_pose(Pose.create_from_pq(p=pill_p, q=pill_q))


    def evaluate(self):
        is_robot_static = self.left_agent.is_static(0.2) and self.right_agent.is_static(0.2)
        is_above_box = self.pill.pose.p[:, 2] - self.box.pose.p[:, 2] > self.goal_thresh_height
        is_grasped = self.left_agent.is_grasping(self.pill) or self.right_agent.is_grasping(self.pill)
        unit_z = torch.tensor([0, 0, 1.0], device=self.device)
        pill_rot = self.pill.pose.to_transformation_matrix()[:, :3, :3]
        cos_x = torch.abs(torch.sum(pill_rot[:, :, 0] * unit_z, dim=1))
        cos_y = torch.abs(torch.sum(pill_rot[:, :, 1] * unit_z, dim=1))
        cos_z = torch.abs(torch.sum(pill_rot[:, :, 2] * unit_z, dim=1))
        cos_stack = torch.stack([cos_x, cos_y, cos_z], dim=1)
        if not hasattr(self, "pill_up_axis"):
            self.pill_up_axis = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        init_mask = self.elapsed_steps == 0
        if torch.any(init_mask):
            self.pill_up_axis[init_mask] = torch.argmax(cos_stack[init_mask], dim=1)
        upright_cos = cos_stack.gather(1, self.pill_up_axis[:, None]).squeeze(1)
        is_tilted = upright_cos <= np.cos(np.deg2rad(15))
        _box_pos_xy = self.box.pose.p[:, :2] + torch.tensor([0, -0.16], device=self.device)
        is_reaching = torch.linalg.norm(self.pill.pose.p[:,:2] - _box_pos_xy, dim=1) < self.goal_thresh_dist

        currently_pouring = is_grasped & is_above_box & is_tilted & is_reaching
        self.ever_poured = self.ever_poured | currently_pouring

        success = self.ever_poured

        result = dict(
            is_grasped=is_grasped,
            is_above_box=is_above_box,
            is_tilted=is_tilted,
            upright_cos=upright_cos,
            is_robot_static=is_robot_static,
            success=success,
            pill_pos=self.pill.pose.p,
            box_pos=self.box.pose.p,
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
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        pill_pos = info["pill_pos"]
        box_pos = info["box_pos"]

        is_grasped = info["is_grasped"]
        is_above_box = info["is_above_box"]
        is_tilted = info["is_tilted"]
        upright_cos = info["upright_cos"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Use whichever arm is closer
        left_to_pill_dist = torch.linalg.norm(left_tcp - pill_pos, dim=-1)
        right_to_pill_dist = torch.linalg.norm(right_tcp - pill_pos, dim=-1)
        closer_tcp = torch.where(
            left_to_pill_dist < right_to_pill_dist,
            left_tcp, right_tcp
        )

        # Phase 1: REACH_PILL - approach the pill
        r_reach_pill = reach_reward(closer_tcp, pill_pos, scale=5.0)
        r_reach_pill = torch.where(is_grasped | success, ones, r_reach_pill)
        self.reward_tracker.update("reach_pill", r_reach_pill)

        # Phase 2: GRASP_PILL - grasp the pill
        r_grasp_pill = is_grasped.float()
        self.reward_tracker.update("grasp_pill", r_grasp_pill)

        # Phase 3: POUR_APPROACH - pill above box
        pill_to_box_xy_dist = torch.linalg.norm(pill_pos[..., :2] - box_pos[..., :2], dim=-1)
        r_approach = tanh_reward(pill_to_box_xy_dist, scale=5.0) * is_grasped.float()
        r_approach = torch.where(is_above_box | success, ones, r_approach)
        self.reward_tracker.update("pour_approach", r_approach)

        # Phase 5: Transport reward removed per user request
        box_pos_for_transport = box_pos + torch.tensor([0, -0.16, 0.12], device=self.device)  # encourage getting pill close to box opening
        r_transport_pill = transport_reward(pill_pos, box_pos_for_transport, is_grasped, scale=5.0)
        r_transport_pill = torch.where(success, ones, r_transport_pill)
        self.reward_tracker.update("transport_pill", r_transport_pill)
        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
