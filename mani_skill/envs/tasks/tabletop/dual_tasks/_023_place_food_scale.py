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
    apply_l2_robotwin_config,
    apply_l3_robotwin_model,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    transport_reward,
    RewardTracker,
)

# Sub-task phase names for this task
REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotPlaceFoodScale-v1", max_episode_steps=100)
class TwoRobotPlaceFoodScaleEnv(BaseEnv):
    """
    Task: Pick up food, place it on the electronic scale.

    Reward phases (from solve script):
      1. reach     — TCP approaches food                          [0, 1]
      2. grasp     — arm grasps food                              [0, 1]
      3. transport — food moved toward scale (gated by grasp)     [0, 1]
      4. place     — food on scale                                [0, 1]

    Completion overrides guarantee success → total = 1.0.
    total = mean( peak(phase_i) for i in 1..4 ) → [0, 1]
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.2

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
        self.vagetable_modelname, self.vagetable_model_id, self.vagetable_scale, self.vagetable_replace_scale = apply_l2_robotwin_config(
            modelname="069_vagetable",
            model_id=3,
            override_name="069_vagetable",
            override_id=2,
            base_scale=(0.5, 0.5, 0.5),
            override_scale=(1, 1, 1),
        )
        self.electronicscale_modelname, self.electronicscale_model_id = apply_l3_robotwin_model(
            "072_electronicscale",
            model_id=0,
            override_name="072_electronicscale",
            override_id=1,
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

        # Load food
        self.food_pose = sapien.Pose(
            p=[-0.1, -0.35, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        food_actor_obj = create_actor(
            scene=self.scene,
            pose=self.food_pose,
            modelname=self.vagetable_modelname,
            convex=True,
            model_id=self.vagetable_model_id,
            scale=self.vagetable_scale,
        )
        food_actor_obj.set_mass(0.5)
        self.food = food_actor_obj.actor

        # Load electronicscale
        self.electronicscale_pose = sapien.Pose(
            p=[-0.1, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        electronicscale_actor_obj = create_actor(
            scene=self.scene,
            pose=self.electronicscale_pose,
            modelname=self.electronicscale_modelname,
            convex=True,
            model_id=self.electronicscale_model_id,
            is_static=True,
            replace_scale=True,
            scale=(0.25, 0.25, 0.25),
        )
        self.electronicscale = electronicscale_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Reward tracker
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            xyz = torch.tensor(self.food_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
            qs = torch.tensor(self.food_pose.q).repeat(b, 1)
            self.food.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.electronicscale_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.electronicscale_zs[env_idx]
            qs = torch.tensor(self.electronicscale_pose.q).repeat(b, 1)
            self.electronicscale.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):
        self.food_zs = []
        collision_mesh = self.food.get_first_collision_mesh()
        self.food_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.food_zs = common.to_tensor(self.food_zs, device=self.device)

        self.electronicscale_zs = []
        collision_mesh = self.electronicscale.get_first_collision_mesh()
        self.electronicscale_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.electronicscale_zs = common.to_tensor(self.electronicscale_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        obj_to_goal_pos = self.food.pose.p - self.electronicscale.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        # FIX: use | and & instead of Python or/and for tensor booleans
        is_grasped = self.left_agent.is_grasping(self.food) | self.right_agent.is_grasping(self.food)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

        result = dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed,
        )

        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())

        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.food.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.food.pose.raw_pose,
                left_tcp_to_obj_pos=self.food.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.food.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.food.pose.p - self.electronicscale.pose.p,
            )
        return obs

    # ═════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 4 sub-tasks, peak-tracked, arithmetic mean → [0, 1]
    #
    #  Phase 1  REACH     — TCP approaches food (closer arm)      [0, 1]
    #  Phase 2  GRASP     — either arm grasps food                [0, 1]
    #  Phase 3  TRANSPORT — food → scale (gated by grasp)         [0, 1]
    #  Phase 4  PLACE     — food on scale                         [0, 1]
    #
    #  Completion overrides guarantee success → total = 1.0
    #  total = mean( peak(phase_i) for i in 1..4 )
    # ═════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        food_pos = self.food.pose.p
        scale_pos = self.electronicscale.pose.p
        is_grasped = info["is_grasped"]
        is_placed = info["is_obj_placed"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Phase 1: REACH — TCP → food (closer arm) ────────────────
        left_r = reach_reward(left_tcp, food_pos, scale=5.0)
        right_r = reach_reward(right_tcp, food_pos, scale=5.0)
        r_reach = torch.maximum(left_r, right_r)
        r_reach = torch.where(is_grasped | success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        # ── Phase 2: GRASP — either arm grasps food ─────────────────
        left_gr = grasp_reward(left_tcp, food_pos, is_grasped, proximity_scale=5.0)
        right_gr = grasp_reward(right_tcp, food_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.maximum(left_gr, right_gr)
        r_grasp = torch.where(success, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        # ── Phase 3: TRANSPORT — food → scale while grasped ─────────
        r_transport = transport_reward(food_pos, scale_pos, is_grasped, scale=5.0)
        r_transport = torch.where(success, ones, r_transport)
        self.reward_tracker.update("transport", r_transport)

        # ── Phase 4: PLACE — food on scale ───────────────────────────
        r_place = is_placed.float()
        r_place = torch.where(success, ones, r_place)
        self.reward_tracker.update("place", r_place)

        # ── Diagnostics ──────────────────────────────────────────────
        self.reward_tracker.write_to_info(info)

        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)