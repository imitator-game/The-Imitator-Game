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
    apply_l2_quat_offset,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    transport_reward,
    RewardTracker,
)

# Sub-task phase names for this task
REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotPlaceBurgerTray-v1", max_episode_steps=100)
class TwoRobotPlaceBurgerTrayEnv(BaseEnv):
    """
    Task: Pick up burger, place it on the tray.

    Reward phases (from solve script):
      1. reach     — TCP approaches burger                        [0, 1]
      2. grasp     — arm grasps burger                            [0, 1]
      3. transport — burger moved toward tray (gated by grasp)    [0, 1]
      4. place     — burger on tray                               [0, 1]

    Completion overrides guarantee success → total = 1.0.
    total = mean( peak(phase_i) for i in 1..4 ) → [0, 1]
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.12

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
        self.hamburg_modelname, self.hamburg_model_id = apply_l2_robotwin_model(
            "006_hamburg",
            model_id=4,
            override_name="006_hamburg",
            override_id=5,
        )
        self.french_fries_modelname, self.french_fries_model_id = apply_l3_robotwin_model(
            "005_french-fries",
            model_id=0,
            override_name="005_french-fries",
            override_id=2,
        )
        self.bottle_modelname, self.bottle_model_id = apply_l3_robotwin_model(
            "114_bottle",
            model_id=3,
            override_name="001_bottle",
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

        # Load burger
        self.burger_pose = sapien.Pose(
            p=[-0., -0.35, 0.0],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        self.burger_pose.q = apply_l2_quat_offset(self.burger_pose.q, offset_q=euler2quat(0, np.pi / 2, 0))
        burger_actor_obj = create_actor(
            scene=self.scene,
            pose=self.burger_pose,
            modelname=self.hamburg_modelname,
            convex=True,
            model_id=self.hamburg_model_id,
            scale=(0.04, 0.04, 0.04),
            replace_scale=True,
        )
        burger_actor_obj.set_mass(0.9)
        self.burger = burger_actor_obj.actor

        # Load fries
        self.fries_pose = sapien.Pose(
            p=[-0., 0.2, 0.025],
            q=euler2quat(0, 0, -np.pi / 2)
        )
        fries_actor_obj = create_actor(
            scene=self.scene,
            pose=self.fries_pose,
            modelname=self.french_fries_modelname,
            convex=True,
            model_id=self.french_fries_model_id,
            is_static=True,
            scale=(0.08, 0.08, 0.08),
            replace_scale=True,
        )
        self.fries = fries_actor_obj.actor

        # Load coke
        self.coke_pose = sapien.Pose(
            p=[0.1, -0., 0.025],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        coke_actor_obj = create_actor(
            scene=self.scene,
            pose=self.coke_pose,
            modelname=self.bottle_modelname,
            convex=True,
            model_id=self.bottle_model_id,
            is_static=True,
            scale=(0.1, 0.1, 0.1),
            replace_scale=True,
        )
        self.coke = coke_actor_obj.actor

        # Load tray
        self.tray_pose = sapien.Pose(
            p=[0.0, 0.1, 0.01],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        tray_actor_obj = create_actor(
            scene=self.scene,
            pose=self.tray_pose,
            modelname=self.tray_modelname,
            convex=True,
            model_id=self.tray_model_id,
            is_static=True,
            replace_scale=True,
            scale=(0.25, 0.25, 0.25),
        )
        self.tray = tray_actor_obj.actor

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

            xyz = torch.tensor(self.burger_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(0.1, 0.1))
            qs = torch.tensor(self.burger_pose.q).repeat(b, 1)
            self.burger.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.fries_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            qs = torch.tensor(self.fries_pose.q).repeat(b, 1)
            self.fries.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.coke_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            qs = torch.tensor(self.coke_pose.q).repeat(b, 1)
            self.coke.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.tray_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.tray_zs[env_idx]
            qs = torch.tensor(self.tray_pose.q).repeat(b, 1)
            self.tray.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):
        self.burger_zs = []
        collision_mesh = self.burger.get_first_collision_mesh()
        self.burger_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.burger_zs = common.to_tensor(self.burger_zs, device=self.device)

        self.fries_zs = []
        collision_mesh = self.fries.get_first_collision_mesh()
        self.fries_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.fries_zs = common.to_tensor(self.fries_zs, device=self.device)

        self.coke_zs = []
        collision_mesh = self.coke.get_first_collision_mesh()
        self.coke_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.coke_zs = common.to_tensor(self.coke_zs, device=self.device)

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
        obj_to_goal_pos = self.burger.pose.p - self.tray.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        # FIX: use | and & instead of Python or/and for tensor booleans
        is_grasped = self.left_agent.is_grasping(self.burger) | self.right_agent.is_grasping(self.burger)
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
            goal_pos=self.burger.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.burger.pose.raw_pose,
                left_tcp_to_obj_pos=self.burger.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.burger.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.burger.pose.p - self.tray.pose.p,
            )
        return obs

    # ═════════════════════════════════════════════════════════════════════════
    # DENSE REWARD — 4 sub-tasks, peak-tracked, arithmetic mean → [0, 1]
    #
    #  Phase 1  REACH     — TCP approaches burger (closer arm)    [0, 1]
    #  Phase 2  GRASP     — either arm grasps burger              [0, 1]
    #  Phase 3  TRANSPORT — burger → tray (gated by grasp)        [0, 1]
    #  Phase 4  PLACE     — burger on tray                        [0, 1]
    #
    #  Completion overrides guarantee success → total = 1.0
    #  total = mean( peak(phase_i) for i in 1..4 )
    # ═════════════════════════════════════════════════════════════════════════

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        burger_pos = self.burger.pose.p
        tray_pos = self.tray.pose.p
        is_grasped = info["is_grasped"]
        is_placed = info["is_obj_placed"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # ── Phase 1: REACH — TCP → burger (closer arm) ──────────────
        left_r = reach_reward(left_tcp, burger_pos, scale=5.0)
        right_r = reach_reward(right_tcp, burger_pos, scale=5.0)
        r_reach = torch.maximum(left_r, right_r)
        r_reach = torch.where(is_grasped | success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        # ── Phase 2: GRASP — either arm grasps burger ───────────────
        left_gr = grasp_reward(left_tcp, burger_pos, is_grasped, proximity_scale=5.0)
        right_gr = grasp_reward(right_tcp, burger_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.maximum(left_gr, right_gr)
        r_grasp = torch.where(success, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        # ── Phase 3: TRANSPORT — burger → tray while grasped ────────
        r_transport = transport_reward(burger_pos, tray_pos, is_grasped, scale=5.0)
        r_transport = torch.where(success, ones, r_transport)
        self.reward_tracker.update("transport", r_transport)

        # ── Phase 4: PLACE — burger on tray ──────────────────────────
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