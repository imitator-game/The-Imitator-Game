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
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.building.actors.robotwin import get_model_id
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_quat_offset,
    apply_l2_robotwin_model,
    apply_l3_robotwin_model,
    apply_l2_robotwin_config,
)

from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
    tanh_reward,
)

# Sub-task phases: left arm picks A, right arm picks B — 4 phases per arm
REWARD_PHASES = [
    "reach_a", "grasp_a", "transport_a", "place_a",
    "reach_b", "grasp_b", "transport_b", "place_b",
]


@register_env("TwoRobotPickFruitsToPlate-v1", max_episode_steps=200)
class TwoRobotPickFruitsToPlateEnv(BaseEnv):
    """
    **Task Description:**
    Two panda_wristcam robots operate on a table.
    1) Pick fruit A from plate A and place it into the goal plate.
    2) Pick fruit B from plate B and place it into the goal plate.

    **Success Conditions:**
    - Both fruits are inside the plate region.
    - Both robots are static.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    FRUIT_MODEL_NAME = "103_fruit"
    PLATE_MODEL_NAME = "003_plate"

    plate_left_xy = (-0.25, -0.20)
    plate_right_xy = (-0.25, 0.20)
    plate_goal_xy = (0.02, 0.0)

    plate_radius = 0.13
    plate_height_threshold = 0.10

    plate_scale = 1.5
    fruit_a_scale = 0.5
    fruit_b_scale = 0.5

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
        self.fruit_a_modelname, self.fruit_a_model_id = apply_l2_robotwin_model(
            "103_fruit",
            model_id=2,
            override_name="069_vagetable",
            override_id=0,
        )
        self.fruit_a_l2_q_offset = euler2quat(np.pi / 2, 0.0, 0.0)
        self.fruit_b_modelname, self.fruit_b_model_id, self.fruit_b_scale, self.fruit_b_replace_scale = apply_l2_robotwin_config(
            modelname="103_fruit",
            model_id=1,
            override_name="069_vagetable",
            override_id=2,
            base_scale=self.fruit_b_scale,
            override_scale=1.0,
        )
        self.plate_modelname, self.plate_model_id = apply_l3_robotwin_model(
            "003_plate",
            model_id=0,
            override_name="008_tray",
            override_id=0,
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
        self.plate_left_pose = sapien.Pose(
            p=[self.plate_left_xy[0], self.plate_left_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2),
        )
        self.plate_right_pose = sapien.Pose(
            p=[self.plate_right_xy[0], self.plate_right_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2),
        )
        self.plate_goal_pose = sapien.Pose(
            p=[self.plate_goal_xy[0], self.plate_goal_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2),
        )

        self.plate_left = create_actor(
            scene=self.scene,
            pose=self.plate_left_pose,
            modelname=self.PLATE_MODEL_NAME,
            convex=True,
            model_id=get_model_id(self.PLATE_MODEL_NAME, model_id=0),
            is_static=True,
            scale=(self.plate_scale,) * 3,
            _idx_if_repeat=1,
        ).actor
        self.plate_right = create_actor(
            scene=self.scene,
            pose=self.plate_right_pose,
            modelname=self.PLATE_MODEL_NAME,
            convex=True,
            model_id=get_model_id(self.PLATE_MODEL_NAME, model_id=0),
            is_static=True,
            scale=(self.plate_scale,) * 3,
            _idx_if_repeat=2,
        ).actor
        self.plate = create_actor(
            scene=self.scene,
            pose=self.plate_goal_pose,
            modelname=self.plate_modelname,
            convex=True,
            model_id=self.plate_model_id,
            is_static=True,
            scale=(self.plate_scale,) * 3,
            _idx_if_repeat=3,
        ).actor

        self.fruit_a_pose = sapien.Pose(
            p=[self.plate_left_xy[0], self.plate_left_xy[1], self.plate_left_pose.p[2] - 0.01],
            q=euler2quat(0, 0, np.pi / 2),
        )
        self.fruit_b_pose = sapien.Pose(
            p=[self.plate_right_xy[0], self.plate_right_xy[1], self.plate_right_pose.p[2] - 0.01],
            q=euler2quat(0, 0, np.pi / 2),
        )
        self.fruit_a = create_actor(
            scene=self.scene,
            pose=self.fruit_a_pose,
            modelname=self.fruit_a_modelname,
            convex=True,
            model_id=self.fruit_a_model_id,
            scale=(self.fruit_a_scale,) * 3,
            _idx_if_repeat=4,
            mass=0.5
        ).actor
        self.fruit_b = create_actor(
            scene=self.scene,
            pose=self.fruit_b_pose,
            modelname=self.fruit_b_modelname,
            convex=True,
            model_id=self.fruit_b_model_id,
            scale=(self.fruit_b_scale,) * 3,
            _idx_if_repeat=5,
            mass=0.5
        ).actor

    def _after_reconfigure(self, options: dict):
        def _compute_object_z(obj) -> float:
            collision_mesh = obj.get_first_collision_mesh()
            return -collision_mesh.bounding_box.bounds[0, 2]

        self._fruit_a_z = common.to_tensor([_compute_object_z(self.fruit_a)], device=self.device)
        self._fruit_b_z = common.to_tensor([_compute_object_z(self.fruit_b)], device=self.device)
        self._plate_left_z = common.to_tensor([_compute_object_z(self.plate_left)], device=self.device)
        self._plate_right_z = common.to_tensor([_compute_object_z(self.plate_right)], device=self.device)
        self._plate_z = common.to_tensor([_compute_object_z(self.plate)], device=self.device)

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

            # ── reward tracker (lazy init) ──
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    REWARD_PHASES, self.num_envs, self.device
                )
            self.reward_tracker.reset(env_idx)

            plate_xyz = torch.tensor(self.plate_goal_pose.p).repeat(b, 1)
            plate_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            plate_xyz[:, 2] = self._plate_z[0]
            plate_xyz = apply_l1_offset_xy(plate_xyz, offset=(-0.1, 0.0))
            plate_q = torch.tensor(self.plate_goal_pose.q).repeat(b, 1)
            self.plate.set_pose(Pose.create_from_pq(p=plate_xyz, q=plate_q))

            plate_left_xyz = torch.tensor(self.plate_left_pose.p).repeat(b, 1)
            plate_left_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            plate_left_xyz[:, 2] = self._plate_left_z[0]
            plate_left_xyz = apply_l1_offset_xy(plate_left_xyz, offset=(0.1, -0.15))
            plate_left_q = torch.tensor(self.plate_left_pose.q).repeat(b, 1)
            self.plate_left.set_pose(Pose.create_from_pq(p=plate_left_xyz, q=plate_left_q))

            plate_right_xyz = torch.tensor(self.plate_right_pose.p).repeat(b, 1)
            plate_right_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            plate_right_xyz[:, 2] = self._plate_right_z[0]
            plate_right_xyz = apply_l1_offset_xy(plate_right_xyz, offset=(0.1, 0.15))
            plate_right_q = torch.tensor(self.plate_right_pose.q).repeat(b, 1)
            self.plate_right.set_pose(Pose.create_from_pq(p=plate_right_xyz, q=plate_right_q))

            fruit_a_p = torch.zeros((b, 3))
            fruit_a_p[:, 0] = plate_left_xyz[:, 0]
            fruit_a_p[:, 1] = plate_left_xyz[:, 1]
            fruit_a_p[:, :2] += torch.rand((b, 2)) * 0.02
            fruit_a_p[:, 2] = self._fruit_a_z[0] + plate_left_xyz[:, 2] + 0.02
            fruit_a_q = torch.tensor(self.fruit_a_pose.q).repeat(b, 1)
            fruit_a_q = apply_l2_quat_offset(fruit_a_q, offset_q=self.fruit_a_l2_q_offset)
            self.fruit_a.set_pose(Pose.create_from_pq(p=fruit_a_p, q=fruit_a_q))

            fruit_b_p = torch.zeros((b, 3))
            fruit_b_p[:, 0] = plate_right_xyz[:, 0]
            fruit_b_p[:, 1] = plate_right_xyz[:, 1]
            fruit_b_p[:, :2] += torch.rand((b, 2)) * 0.02
            fruit_b_p[:, 2] = self._fruit_b_z[0] + plate_right_xyz[:, 2] + 0.02
            fruit_b_q = torch.tensor(self.fruit_b_pose.q).repeat(b, 1)
            self.fruit_b.set_pose(Pose.create_from_pq(p=fruit_b_p, q=fruit_b_q))

    def _is_in_plate(self, obj_pos: torch.Tensor, plate_pos: torch.Tensor) -> torch.Tensor:
        xy_dist = torch.linalg.norm(obj_pos[..., :2] - plate_pos[..., :2], dim=-1)
        in_xy = xy_dist <= self.plate_radius
        in_z = obj_pos[..., 2] <= self.plate_height_threshold
        return torch.logical_and(in_xy, in_z)

    def evaluate(self):
        plate_pos = self.plate.pose.p
        fruit_a_pos = self.fruit_a.pose.p
        fruit_b_pos = self.fruit_b.pose.p

        fruit_a_in_plate = self._is_in_plate(fruit_a_pos, plate_pos)
        fruit_b_in_plate = self._is_in_plate(fruit_b_pos, plate_pos)

        is_left_static = self.left_agent.is_static(0.2)
        is_right_static = self.right_agent.is_static(0.2)
        is_robot_static = is_left_static & is_right_static

        is_a_grasped = self.left_agent.is_grasping(self.fruit_a)
        is_b_grasped = self.right_agent.is_grasping(self.fruit_b)

        success = fruit_a_in_plate & fruit_b_in_plate

        result = dict(
            fruit_a_in_plate=fruit_a_in_plate,
            fruit_b_in_plate=fruit_b_in_plate,
            is_a_grasped=is_a_grasped,
            is_b_grasped=is_b_grasped,
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
            plate_pose=self.plate.pose.raw_pose,
            fruit_a_pose=self.fruit_a.pose.raw_pose,
            fruit_b_pose=self.fruit_b.pose.raw_pose,
            fruit_a_in_plate=info["fruit_a_in_plate"],
            fruit_b_in_plate=info["fruit_b_in_plate"],
        )
        if "state" in self.obs_mode:
            obs.update(
                left_tcp_to_fruit_a=self.fruit_a.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_fruit_b=self.fruit_b.pose.p - self.right_agent.tcp.pose.p,
                fruit_a_to_plate=self.plate.pose.p - self.fruit_a.pose.p,
                fruit_b_to_plate=self.plate.pose.p - self.fruit_b.pose.p,
            )
        return obs

    # ═══════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 8 sub-tasks (4 per arm), peak-tracked → [0, 1]
    #
    # Left arm (fruit A):
    #   reach_a     — left TCP → fruit A                     [0, 1]
    #   grasp_a     — proximity / confirmed grasp            [0, 1]
    #   transport_a — fruit A → goal plate center            [0, 1]
    #   place_a     — fruit A inside goal plate              [0, 1]
    #
    # Right arm (fruit B):
    #   reach_b     — right TCP → fruit B                    [0, 1]
    #   grasp_b     — proximity / confirmed grasp            [0, 1]
    #   transport_b — fruit B → goal plate center            [0, 1]
    #   place_b     — fruit B inside goal plate              [0, 1]
    #
    # Completion overrides guarantee success → total = 1.0
    # total = mean( peak(phase_i) for i in 1..8 ) → [0, 1]
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        fruit_a_pos = self.fruit_a.pose.p
        fruit_b_pos = self.fruit_b.pose.p
        plate_pos = self.plate.pose.p

        is_a_grasped = info["is_a_grasped"]
        is_b_grasped = info["is_b_grasped"]
        a_in_plate = info["fruit_a_in_plate"]
        b_in_plate = info["fruit_b_in_plate"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Left arm: fruit A ───────────────────────────────────────
        r_reach_a = reach_reward(left_tcp, fruit_a_pos, scale=5.0)
        r_reach_a = torch.where(is_a_grasped | a_in_plate | success, ones, r_reach_a)

        r_grasp_a = grasp_reward(left_tcp, fruit_a_pos, is_a_grasped, proximity_scale=5.0)
        r_grasp_a = torch.where(a_in_plate | success, ones, r_grasp_a)

        r_transport_a = transport_reward(fruit_a_pos, plate_pos, is_a_grasped, scale=5.0)
        r_transport_a = torch.where(a_in_plate | success, ones, r_transport_a)

        r_place_a = a_in_plate.float()
        r_place_a = torch.where(success, ones, r_place_a)

        # ── Right arm: fruit B ──────────────────────────────────────
        r_reach_b = reach_reward(right_tcp, fruit_b_pos, scale=5.0)
        r_reach_b = torch.where(is_b_grasped | b_in_plate | success, ones, r_reach_b)

        r_grasp_b = grasp_reward(right_tcp, fruit_b_pos, is_b_grasped, proximity_scale=5.0)
        r_grasp_b = torch.where(b_in_plate | success, ones, r_grasp_b)

        r_transport_b = transport_reward(fruit_b_pos, plate_pos, is_b_grasped, scale=5.0)
        r_transport_b = torch.where(b_in_plate | success, ones, r_transport_b)

        r_place_b = b_in_plate.float()
        r_place_b = torch.where(success, ones, r_place_b)

        # ── Update tracker ──────────────────────────────────────────
        self.reward_tracker.update("reach_a", r_reach_a)
        self.reward_tracker.update("grasp_a", r_grasp_a)
        self.reward_tracker.update("transport_a", r_transport_a)
        self.reward_tracker.update("place_a", r_place_a)

        self.reward_tracker.update("reach_b", r_reach_b)
        self.reward_tracker.update("grasp_b", r_grasp_b)
        self.reward_tracker.update("transport_b", r_transport_b)
        self.reward_tracker.update("place_b", r_place_b)

        # ── Diagnostics ─────────────────────────────────────────────
        self.reward_tracker.write_to_info(info)

        # ── Total = arithmetic mean of peaks → [0, 1] ──────────────
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)