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
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_quat_offset,
    apply_l2_ycb_model_id,
    apply_l3_partnet_ids,
    is_l3_enabled,
    is_lr_mirror_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    transport_reward,
    tanh_reward,
    RewardTracker,
)

# Sub-task phases: left picks tennis → box1, right picks golf → box2
REWARD_PHASES = [
    "reach_t", "grasp_t", "transport_t", "place_t",
    "reach_g", "grasp_g", "transport_g", "place_g",
]


@register_env("TwoRobotPickTennisBallGolfBall-v1", max_episode_steps=100)
class TwoRobotPickTennisBallGolfBallEnv(BaseEnv):
    """
    Task: Left picks tennis ball → box1, Right picks golf ball → box2.

    Reward phases (from solve script):
      Left arm (tennis ball):
        1. reach_t     — left TCP → tennis ball              [0, 1]
        2. grasp_t     — left arm grasps tennis ball         [0, 1]
        3. transport_t — tennis ball → box1                  [0, 1]
        4. place_t     — tennis ball in box1                 [0, 1]

      Right arm (golf ball):
        5. reach_g     — right TCP → golf ball               [0, 1]
        6. grasp_g     — right arm grasps golf ball          [0, 1]
        7. transport_g — golf ball → box2                    [0, 1]
        8. place_g     — golf ball in box2                   [0, 1]

    Completion overrides guarantee success → total = 1.0.
    total = mean( peak(phase_i) for i in 1..8 ) → [0, 1]
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    goal_thresh = 0.05

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
        self.tennis_ball_model_id = apply_l2_ycb_model_id(
            "056_tennis_ball", override_id="055_baseball"
        )
        self.tennis_ball_l2_q_offset = euler2quat(0.0, 0.0, np.pi / 2)
        self.golf_ball_model_id = apply_l2_ycb_model_id(
            "058_golf_ball", override_id="057_racquetball"
        )
        self.box_model_id = apply_l3_partnet_ids(["100671"], override_ids=["102456"])[0]

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

        # Load tennis ball
        self._tennis_balls = []
        for i in range(self.num_envs):
            tennis_ball_builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{self.tennis_ball_model_id}",
            )
            tennis_ball_builder._mass = 0.5
            tennis_ball_builder.initial_pose = sapien.Pose(p=[0, 0.2, 0])
            tennis_ball_builder.set_scene_idxs([i])
            self._tennis_balls.append(tennis_ball_builder.build(name=f"{self.tennis_ball_model_id}-{i}"))
            self.remove_from_state_dict_registry(self._tennis_balls[-1])

        from mani_skill.utils.structs.actor import Actor
        self.tennis_ball = Actor.merge(self._tennis_balls, name="ycb_tennis_ball")
        self.add_to_state_dict_registry(self.tennis_ball)

        # Load golf ball
        self._golf_balls = []
        for i in range(self.num_envs):
            golf_ball_builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{self.golf_ball_model_id}",
            )
            golf_ball_builder._mass = 0.5
            golf_ball_builder.initial_pose = sapien.Pose(p=[0, -0.2, 0])
            golf_ball_builder.set_scene_idxs([i])
            self._golf_balls.append(golf_ball_builder.build(name=f"{self.golf_ball_model_id}-{i}"))
            self.remove_from_state_dict_registry(self._golf_balls[-1])

        self.golf_ball = Actor.merge(self._golf_balls, name="ycb_golf_ball")
        self.add_to_state_dict_registry(self.golf_ball)

        # Load two partnet boxes
        self._boxes = []
        for i in range(self.num_envs):
            box1_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{self.box_model_id}", mode="box", scale=0.15
            )
            box1_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            box1_builder.set_scene_idxs([i])
            box1 = box1_builder.build(name=f"box1-{i}")
            self.remove_from_state_dict_registry(box1)

            box2_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{self.box_model_id}", mode="box", scale=0.15
            )
            box2_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            box2_builder.set_scene_idxs([i])
            box2 = box2_builder.build(name=f"box2-{i}")
            self.remove_from_state_dict_registry(box2)

            self._boxes.append([box1, box2])

        from mani_skill.utils.structs import Articulation
        self.box1 = Articulation.merge([boxes[0] for boxes in self._boxes], name="box1")
        self.box2 = Articulation.merge([boxes[1] for boxes in self._boxes], name="box2")
        self.add_to_state_dict_registry(self.box1)
        self.add_to_state_dict_registry(self.box2)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.left_agent.robot.set_pose(sapien.Pose(p=[-0.60, -0.3, 0]))
            self.right_agent.robot.set_pose(sapien.Pose(p=[-0.60, 0.3, 0]))

            # Reward tracker (8 phases)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            # Initialize tennis ball
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.25
            xyz[:, 1] = -0.45
            xyz[:, 2] = self.tennis_ball_zs[env_idx]
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.05, -0.05))
            qs = torch.tensor([[1, 0, 0, 0]] * b, device=self.device, dtype=torch.float32)
            qs = apply_l2_quat_offset(qs, offset_q=self.tennis_ball_l2_q_offset)
            self.tennis_ball.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Initialize golf ball
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.25
            xyz[:, 1] = 0.45
            xyz[:, 2] = self.golf_ball_zs[env_idx]
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.05, 0.05))
            qs = torch.tensor([[1, 0, 0, 0]] * b, device=self.device, dtype=torch.float32)
            self.golf_ball.set_pose(Pose.create_from_pq(xyz, qs))

            # Initialize first box
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.15
            xyz[:, 1] = -0.13
            xyz[:, 2] = self.box_zs[env_idx]
            xyz = apply_l1_offset_xy(xyz, offset=(-0.05, -0.05))
            box1_z_rot = -np.pi / 10
            if is_lr_mirror_enabled():
                box1_z_rot = -box1_z_rot
            base_quat = euler2quat(0, 0, box1_z_rot)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.box1.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Initialize second box
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.15
            xyz[:, 1] = 0.13
            xyz[:, 2] = self.box_zs[env_idx]
            xyz = apply_l1_offset_xy(xyz, offset=(-0.05, 0.05))
            box2_z_rot = np.pi / 10
            if is_lr_mirror_enabled():
                box2_z_rot = -box2_z_rot
            base_quat = euler2quat(0, 0, box2_z_rot)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.box2.set_pose(Pose.create_from_pq(xyz, qs))

            lid0_angle = 0.8 if is_l3_enabled() else 1.2
            self.set_box_lid_angles(self.box1, lid0_angle=lid0_angle, env_idx=env_idx)
            self.set_box_lid_angles(self.box2, lid0_angle=lid0_angle, env_idx=env_idx)

            self.target_lid0_closed = 0

    def set_box_lid_angles(self, box, lid0_angle: float, env_idx: torch.Tensor = None):
        b = len(env_idx)
        qpos = torch.zeros((b, 1), device=self.device, dtype=torch.float32)
        qpos[:, 0] = lid0_angle
        box.set_qpos(qpos)

    def _after_reconfigure(self, options: dict):
        self.tennis_ball_zs = []
        for tennis_ball_obj in self._tennis_balls:
            collision_mesh = tennis_ball_obj.get_first_collision_mesh()
            if collision_mesh is not None:
                self.tennis_ball_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
            else:
                self.tennis_ball_zs.append(0.033)
        self.tennis_ball_zs = common.to_tensor(self.tennis_ball_zs, device=self.device)

        self.golf_ball_zs = []
        for golf_ball_obj in self._golf_balls:
            collision_mesh = golf_ball_obj.get_first_collision_mesh()
            if collision_mesh is not None:
                self.golf_ball_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
            else:
                self.golf_ball_zs.append(0.0215)
        self.golf_ball_zs = common.to_tensor(self.golf_ball_zs, device=self.device)

        self.box_zs = []
        for boxes in self._boxes:
            box = boxes[0]
            collision_mesh = box.get_first_collision_mesh()
            if collision_mesh is not None:
                self.box_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
            else:
                self.box_zs.append(0.1)
        self.box_zs = common.to_tensor(self.box_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        tennis_to_box1_dist = torch.linalg.norm(self.tennis_ball.pose.p - self.box1.pose.p, axis=1)
        golf_to_box2_dist = torch.linalg.norm(self.golf_ball.pose.p - self.box2.pose.p, axis=1)

        tennis_in_box1 = tennis_to_box1_dist <= 0.1
        golf_in_box2 = golf_to_box2_dist <= 0.1

        box1_lid0_closed = self.box1.qpos[:, 0] <= (self.target_lid0_closed + 0.3)
        box2_lid0_closed = self.box2.qpos[:, 0] <= (self.target_lid0_closed + 0.3)

        is_robot_static = torch.logical_and(
            self.left_agent.is_static(0.2),
            self.right_agent.is_static(0.2)
        )

        is_tennis_grasped = self.left_agent.is_grasping(self.tennis_ball)
        is_golf_grasped = self.right_agent.is_grasping(self.golf_ball)

        success = torch.logical_and(tennis_in_box1, golf_in_box2)

        result = dict(
            tennis_in_box1=tennis_in_box1,
            golf_in_box2=golf_in_box2,
            is_robot_static=is_robot_static,
            is_tennis_grasped=is_tennis_grasped,
            is_golf_grasped=is_golf_grasped,
            success=success,
        )

        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())

        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            tennis_in_box1=info["tennis_in_box1"],
            golf_in_box2=info["golf_in_box2"],
        )
        if "state" in self.obs_mode:
            obs.update(
                tennis_ball_pose=self.tennis_ball.pose.raw_pose,
                golf_ball_pose=self.golf_ball.pose.raw_pose,
                box1_pose=self.box1.pose.raw_pose,
                box2_pose=self.box2.pose.raw_pose,
                left_tcp_to_tennis_ball=self.tennis_ball.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_golf_ball=self.golf_ball.pose.p - self.right_agent.tcp.pose.p,
            )
        return obs

    # ═════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 8 sub-tasks (4 per arm), peak-tracked → [0, 1]
    #
    # Left arm (tennis ball → box1):
    #   reach_t     — left TCP → tennis ball                 [0, 1]
    #   grasp_t     — left arm grasps tennis ball            [0, 1]
    #   transport_t — tennis ball → box1                     [0, 1]
    #   place_t     — tennis ball in box1                    [0, 1]
    #
    # Right arm (golf ball → box2):
    #   reach_g     — right TCP → golf ball                  [0, 1]
    #   grasp_g     — right arm grasps golf ball             [0, 1]
    #   transport_g — golf ball → box2                       [0, 1]
    #   place_g     — golf ball in box2                      [0, 1]
    #
    # Completion overrides guarantee success → total = 1.0
    # total = mean( peak(phase_i) for i in 1..8 ) → [0, 1]
    # ═════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        tennis_pos = self.tennis_ball.pose.p
        golf_pos = self.golf_ball.pose.p
        box1_pos = self.box1.pose.p
        box2_pos = self.box2.pose.p

        is_tennis_grasped = info["is_tennis_grasped"]
        is_golf_grasped = info["is_golf_grasped"]
        tennis_in = info["tennis_in_box1"]
        golf_in = info["golf_in_box2"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Left arm: tennis ball → box1 ────────────────────────────
        r_reach_t = reach_reward(left_tcp, tennis_pos, scale=5.0)
        r_reach_t = torch.where(is_tennis_grasped | tennis_in | success, ones, r_reach_t)

        r_grasp_t = grasp_reward(left_tcp, tennis_pos, is_tennis_grasped, proximity_scale=5.0)
        r_grasp_t = torch.where(tennis_in | success, ones, r_grasp_t)

        r_transport_t = transport_reward(tennis_pos, box1_pos, is_tennis_grasped, scale=5.0)
        r_transport_t = torch.where(tennis_in | success, ones, r_transport_t)

        r_place_t = tennis_in.float()
        r_place_t = torch.where(success, ones, r_place_t)

        # ── Right arm: golf ball → box2 ─────────────────────────────
        r_reach_g = reach_reward(right_tcp, golf_pos, scale=5.0)
        r_reach_g = torch.where(is_golf_grasped | golf_in | success, ones, r_reach_g)

        r_grasp_g = grasp_reward(right_tcp, golf_pos, is_golf_grasped, proximity_scale=5.0)
        r_grasp_g = torch.where(golf_in | success, ones, r_grasp_g)

        r_transport_g = transport_reward(golf_pos, box2_pos, is_golf_grasped, scale=5.0)
        r_transport_g = torch.where(golf_in | success, ones, r_transport_g)

        r_place_g = golf_in.float()
        r_place_g = torch.where(success, ones, r_place_g)

        # ── Update tracker ──────────────────────────────────────────
        self.reward_tracker.update("reach_t", r_reach_t)
        self.reward_tracker.update("grasp_t", r_grasp_t)
        self.reward_tracker.update("transport_t", r_transport_t)
        self.reward_tracker.update("place_t", r_place_t)

        self.reward_tracker.update("reach_g", r_reach_g)
        self.reward_tracker.update("grasp_g", r_grasp_g)
        self.reward_tracker.update("transport_g", r_transport_g)
        self.reward_tracker.update("place_g", r_place_g)

        # ── Diagnostics ─────────────────────────────────────────────
        self.reward_tracker.write_to_info(info)

        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)