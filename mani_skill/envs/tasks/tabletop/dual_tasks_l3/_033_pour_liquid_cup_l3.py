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
    is_lr_mirror_enabled,
)

from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    tanh_reward,
)

# Sub-task phases for this task
REWARD_PHASES = ["reach_bottle", "grasp_bottle", "pour_approach"]


@register_env("TwoRobotPourLiquidCupL3-v1", max_episode_steps=100)
class TwoRobotPourLiquidCupEnvL3(BaseEnv):
    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.1

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
        self.pillbottle_modelname, self.pillbottle_model_id = apply_l2_robotwin_model(
            "080_pillbottle",
            model_id=3,
            override_name="001_bottle",
            override_id=0,
        )
        self.cup_modelname, self.cup_model_id = "039_mug", 5

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
                found_lost_pairs_capacity=2 ** 25,
                max_rigid_patch_count=2 ** 19,
                max_rigid_contact_count=2 ** 21,
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

        # Load liquid
        liquid_q = euler2quat(np.pi / 2, 0, 0)
        # if is_l2_enabled():
        #     extra_q = euler2quat(0, 0, np.pi / 2)
        #     liquid_q = (sapien.Pose(q=extra_q) * sapien.Pose(q=liquid_q)).q
        self.liquid_pose = sapien.Pose(
            p=[-0.05, -0.0, 0.0],
            q=liquid_q
        )
        liquid_actor_obj = create_actor(
            scene=self.scene,
            pose=self.liquid_pose,
            modelname=self.pillbottle_modelname,
            convex=True,
            model_id=self.pillbottle_model_id,
            replace_scale=True,
            scale=(0.06, 0.06, 0.06),
        )
        liquid_actor_obj.set_mass(0.5)
        self.liquid = liquid_actor_obj.actor

        # Load cup
        self.cup_pose = sapien.Pose(
            p=[0.0, 0.1, 0.0],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        cup_actor_obj = create_actor(
            scene=self.scene,
            pose=self.cup_pose,
            modelname=self.cup_modelname,
            convex=True,
            model_id=self.cup_model_id,
            is_static=True,
            # replace_scale=True,
            scale=(0.6, 0.8, 0.6),
        )
        self.cup = cup_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "liquid_ever_poured"):
                self.liquid_ever_poured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.liquid_ever_poured[env_idx] = False

            # Initialize reward tracker
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            xyz = torch.tensor(self.liquid_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.liquid_zs[env_idx]
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
            # For euler(π/2, 0, γ) objects, correct y-mirror is γ → π - γ
            liquid_z_rot = 0.0
            if is_lr_mirror_enabled():
                liquid_z_rot = np.pi - liquid_z_rot
            qs = torch.tensor(euler2quat(np.pi / 2, 0, liquid_z_rot)).repeat(b, 1)
            self.liquid.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.cup_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.cup_zs[env_idx]
            qs = torch.tensor(self.cup_pose.q).repeat(b, 1)
            self.cup.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.liquid_zs = []
        collision_mesh = self.liquid.get_first_collision_mesh()
        self.liquid_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.liquid_zs = common.to_tensor(self.liquid_zs, device=self.device)

        self.cup_zs = []
        collision_mesh = self.cup.get_first_collision_mesh()
        self.cup_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cup_zs = common.to_tensor(self.cup_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        # 1. Check current state
        is_grasped = self.left_agent.is_grasping(self.liquid) or self.right_agent.is_grasping(self.liquid)

        liquid_height_above_cup = self.liquid.pose.p[:, 2] - self.cup.pose.p[:, 2]

        is_above_cup = (liquid_height_above_cup > self.goal_thresh)
        unit_z = torch.tensor([0, 0, 1.0], device=self.device)
        liquid_rot = self.liquid.pose.to_transformation_matrix()[:, :3, :3]
        cos_x = torch.abs(torch.sum(liquid_rot[:, :, 0] * unit_z, dim=1))
        cos_y = torch.abs(torch.sum(liquid_rot[:, :, 1] * unit_z, dim=1))
        cos_z = torch.abs(torch.sum(liquid_rot[:, :, 2] * unit_z, dim=1))
        cos_stack = torch.stack([cos_x, cos_y, cos_z], dim=1)
        if not hasattr(self, "liquid_up_axis"):
            self.liquid_up_axis = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        init_mask = self.elapsed_steps == 0
        if torch.any(init_mask):
            self.liquid_up_axis[init_mask] = torch.argmax(cos_stack[init_mask], dim=1)
        upright_cos = cos_stack.gather(1, self.liquid_up_axis[:, None]).squeeze(1)
        is_tilted = upright_cos <= np.cos(np.deg2rad(15))

        # 2. Update historical flag: EVER (grasped AND above cup)
        currently_pouring = is_grasped & is_above_cup & is_tilted
        self.liquid_ever_poured = self.liquid_ever_poured | currently_pouring

        # 3. Robot should be static at the end
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

        # 4. Success: ever poured and robot static
        success = self.liquid_ever_poured

        # debug
        # print(
        #     "Eval Info: ",
        #     "liquid_height_above_cup=", liquid_height_above_cup,
        #     "is_above_cup=", is_above_cup,
        #     "is_tilted=", is_tilted,
        #     "currently_pouring=", currently_pouring,
        #     "liquid_ever_poured=", self.liquid_ever_poured,
        #     "success=", success,
        # )

        result = dict(
            is_grasped=is_grasped,
            is_above_cup=is_above_cup,
            is_tilted=is_tilted,
            upright_cos=upright_cos,
            liquid_ever_poured=self.liquid_ever_poured,
            is_robot_static=is_robot_static,
            success=success,
            liquid_height_above_cup=liquid_height_above_cup,
            cup_pos=self.cup.pose.p,
            liquid_pos=self.liquid.pose.p,
        )
        # Append per-phase peak sub-rewards
        # if hasattr(self, "reward_tracker"):
        #     # result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            cup_pos=self.cup.pose.p,
            is_grasped=info["is_grasped"],
            liquid_ever_poured=info["liquid_ever_poured"],
        )
        if "state" in self.obs_mode:
            obs.update(
                liquid_pose=self.liquid.pose.raw_pose,
                left_tcp_to_liquid_pos=self.liquid.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_liquid_pos=self.liquid.pose.p - self.right_agent.tcp.pose.p,
                cup_pose=self.cup.pose.raw_pose,
                liquid_to_cup_pos=self.cup.pose.p - self.liquid.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        liquid_pos = info["liquid_pos"]
        cup_pos = info["cup_pos"]

        is_grasped = info["is_grasped"]
        is_above_cup = info["is_above_cup"]
        is_tilted = info["is_tilted"]
        upright_cos = info["upright_cos"]
        liquid_ever_poured = info["liquid_ever_poured"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Phase 1: REACH_BOTTLE - approach the bottle
        left_tcp_to_liquid_dist = torch.linalg.norm(liquid_pos - left_tcp, axis=1)
        right_tcp_to_liquid_dist = torch.linalg.norm(liquid_pos - right_tcp, axis=1)
        closer_tcp = torch.where(
            left_tcp_to_liquid_dist < right_tcp_to_liquid_dist, left_tcp, right_tcp
        )
        r_reach_bottle = reach_reward(closer_tcp, liquid_pos, scale=5.0)
        r_reach_bottle = torch.where(is_grasped | liquid_ever_poured | success, ones, r_reach_bottle)
        self.reward_tracker.update("reach_bottle", r_reach_bottle)

        # Phase 2: GRASP_BOTTLE
        r_grasp_bottle = is_grasped.float()
        r_grasp_bottle = torch.where(liquid_ever_poured | success, ones, r_grasp_bottle)
        self.reward_tracker.update("grasp_bottle", r_grasp_bottle)

        # Phase 3: POUR_APPROACH - bottle above cup
        liquid_to_cup_xy_dist = torch.linalg.norm(liquid_pos[..., :2] - cup_pos[..., :2], dim=-1)
        r_approach = tanh_reward(liquid_to_cup_xy_dist, scale=5.0) * is_grasped.float()
        r_approach = torch.where(is_above_cup | liquid_ever_poured | success, ones, r_approach)
        self.reward_tracker.update("pour_approach", r_approach)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
