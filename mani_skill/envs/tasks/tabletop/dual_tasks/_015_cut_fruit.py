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
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l3_robotwin_model,
    is_l2_enabled,
    is_l3_enabled,
    is_lr_mirror_enabled,
    mirror_quat,
)
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    tanh_reward,
    reach_reward,
    grasp_reward,
)

# Sub-task phase names for this task
REWARD_PHASES = ["reach", "grasp", "cut"]


@register_env("TwoRobotCutFruit-v1", max_episode_steps=100)
class TwoRobotCutFruitEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.025

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
        self.vagetable_modelname, self.vagetable_model_id = apply_l3_robotwin_model(
            "069_vagetable",
            model_id=6,
            override_name="069_vagetable",
            override_id=5,
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

        # Load carrot
        self.carrot_pose = sapien.Pose(
            p=[0.0, 0.0, 0.03],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        carrot_actor_obj = create_actor(
            scene=self.scene,
            pose=self.carrot_pose,
            modelname=self.vagetable_modelname,
            convex=True,
            model_id=self.vagetable_model_id,
            scale=(0.18, 0.18, 0.18),
            replace_scale=True,
            is_static=True
        )
        self.carrot = carrot_actor_obj.actor

        # Load knife
        self.knife_pose = sapien.Pose(
            p=[-0.1, -0.2, 0.06],
            q=euler2quat(0, -np.pi / 2, 0)
        )
        if is_l2_enabled():
            p = np.array(self.knife_pose.p)
            p[1] -= 0.0
            extra_q = euler2quat(np.pi / 2, np.pi / 2, 0)
            extra_q = torch.tensor(extra_q, dtype=torch.float32, device=self.device)
            base_q = torch.tensor(self.knife_pose.q, dtype=torch.float32, device=self.device)
            q = quaternion_multiply(extra_q, base_q).cpu().numpy()
            self.knife_pose = sapien.Pose(p=p.tolist(), q=q)
        if is_l2_enabled():
            knife_ycb_id = "032_knife"
            knife_builder = actors.get_actor_builder(
                self.scene, id=f"ycb:{knife_ycb_id}", scales=[1.6]
            )
            knife_builder._mass = 0.2
            knife_builder.initial_pose = self.knife_pose
            self.knife = knife_builder.build(name="knife")
            self.knife_grasp_target = self.knife
        else:
            knife_actor_obj = create_actor(
                scene=self.scene,
                pose=self.knife_pose,
                modelname="034_knife",
                convex=True,
                model_id=0,
                scale=(0.8, 0.8, 0.8),
            )
            knife_actor_obj.set_mass(0.2)
            self.knife = knife_actor_obj.actor
            self.knife_grasp_target = self.knife

        # Load board
        self.board_pose = sapien.Pose(
            p=[0.0, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        board_actor_obj = create_actor(
            scene=self.scene,
            pose=self.board_pose,
            modelname="104_board",
            convex=True,
            model_id=2,
            is_static=True,
            replace_scale=True,
            scale=(0.3, 0.3, 0.3),
        )
        self.board = board_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # ── historical flags ──
            if not hasattr(self, "knife_ever_cut_fruit"):
                self.knife_ever_cut_fruit = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.knife_ever_cut_fruit[env_idx] = False

            # ── reward tracker ──
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            xyz = torch.tensor(self.knife_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.05, 0.0))
            qs = torch.tensor(self.knife_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            if is_lr_mirror_enabled():
                if is_l3_enabled():
                    qs = torch.tensor(
                        euler2quat(0, np.pi / 2, np.pi),
                        device=self.device,
                        dtype=torch.float32,
                    ).repeat(b, 1)
                else:
                    qs = mirror_quat(qs)
            self.knife.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            xyz = torch.tensor(self.carrot_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            carrot_z_rot = -np.pi / 2
            if is_lr_mirror_enabled():
                carrot_z_rot = np.pi - carrot_z_rot
            qs = torch.tensor(
                euler2quat(np.pi / 2, 0, carrot_z_rot),
                device=self.device,
                dtype=torch.float32,
            ).repeat(b, 1)
            self.carrot.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.board_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.board_zs[env_idx]
            board_z_rot = 0.0
            if is_lr_mirror_enabled():
                board_z_rot = np.pi - board_z_rot
            qs = torch.tensor(
                euler2quat(np.pi / 2, 0, board_z_rot),
                device=self.device,
                dtype=torch.float32,
            ).repeat(b, 1)
            self.board.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):
        self.knife_zs = []
        collision_mesh = self.knife.get_first_collision_mesh()
        self.knife_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.knife_zs = common.to_tensor(self.knife_zs, device=self.device)

        self.carrot_zs = []
        collision_mesh = self.carrot.get_first_collision_mesh()
        self.carrot_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.carrot_zs = common.to_tensor(self.carrot_zs, device=self.device)

        self.board_zs = []
        collision_mesh = self.board.get_first_collision_mesh()
        self.board_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.board_zs = common.to_tensor(self.board_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    # ═══════════════════════════════════════════════════════════════════════════
    # EVALUATE
    # ═══════════════════════════════════════════════════════════════════════════

    def evaluate(self):
        impulses = self.scene.get_pairwise_contact_impulses(
            self.knife_grasp_target, self.carrot
        )
        is_knife_at_carrot = torch.linalg.norm(impulses, dim=1) > 0.02
        if is_l2_enabled():
            is_knife_at_carrot = torch.linalg.norm(impulses, dim=1) > 0.005
        is_grasped = self.left_agent.is_grasping(self.knife_grasp_target)
        self.knife_ever_cut_fruit = self.knife_ever_cut_fruit | (
            is_knife_at_carrot & is_grasped
        )

        is_left_static = self.left_agent.is_static(0.2)
        is_right_static = self.right_agent.is_static(0.2)
        is_robot_static = is_left_static & is_right_static

        success = self.knife_ever_cut_fruit

        result = dict(
            is_grasped=is_grasped,
            knife_ever_cut_fruit=self.knife_ever_cut_fruit,
            is_robot_static=is_robot_static,
            success=success,
        )
        # Append per-phase peak sub-rewards
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    # ═══════════════════════════════════════════════════════════════════════════
    # OBSERVATIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.carrot.pose.p,
            is_grasped=info["is_grasped"],
            knife_ever_cut_fruit=info["knife_ever_cut_fruit"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.knife.pose.raw_pose,
                left_tcp_to_obj_pos=self.knife.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.knife.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.carrot.pose.p - self.knife.pose.p,
            )
        return obs

    # ═══════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 3 sub-tasks, peak-tracked, arithmetic mean → [0, 1]
    #
    #  Phase 1  REACH  — left TCP approaches knife                [0, 1]
    #  Phase 2  GRASP  — proximity / confirmed grasp              [0, 1]
    #  Phase 3  CUT    — transport knife → carrot + contact       [0, 1]
    #
    #  Completion overrides guarantee success → total = 1.0:
    #    - grasped     ⇒ reach = 1.0
    #    - ever_cut    ⇒ reach = grasp = cut = 1.0  (success)
    #
    #  total = mean( peak(phase_i) for i in 1..3 )
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p         # (B, 3)
        knife_pos = self.knife.pose.p                # (B, 3)
        carrot_pos = self.carrot.pose.p              # (B, 3)
        is_grasped = info["is_grasped"]              # (B,) bool
        success = info["success"]                    # (B,) bool  = ever_cut

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Phase 1: REACH ──────────────────────────────────────────
        r_reach = reach_reward(tcp_pos, knife_pos, scale=5.0)
        r_reach = torch.where(is_grasped | success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        # ── Phase 2: GRASP ──────────────────────────────────────────
        r_grasp = grasp_reward(tcp_pos, knife_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(success, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        # ── Phase 3: CUT ────────────────────────────────────────────
        knife_to_carrot_dist = torch.linalg.norm(knife_pos - carrot_pos, dim=-1)
        r_cut = tanh_reward(knife_to_carrot_dist, scale=5.0) * is_grasped.float()
        r_cut = torch.where(success, ones, r_cut)
        self.reward_tracker.update("cut", r_cut)

        # ── Diagnostics ─────────────────────────────────────────────
        self.reward_tracker.write_to_info(info)

        # ── Total = arithmetic mean of peaks → [0, 1] ──────────────
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)