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
    apply_l2_robotwin_config,
    apply_l3_robotwin_config,
    is_l3_enabled,
    is_lr_mirror_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
    exp_reward,
)

# Sub-task phases for this task
REWARD_PHASES = ["reach_mug", "grasp_mug", "transport_mug", "place_mug"]


@register_env("TwoRobotPlaceMugRack-v1", max_episode_steps=100)
class TwoRobotPlaceMugRackEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.20

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
        self.goal_thresh = 0.40 if is_l3_enabled() else 0.16
        # self.mug_modelname, self.mug_model_id = apply_l2_robotwin_model(
        #     "039_mug",
        #     model_id=11,
        #     override_name="021_cup",
        #     override_id=10,
        # )
        (
            self.mug_modelname,
            self.mug_model_id,
            self.mug_scale,
            self.mug_replace_scale,
        ) = apply_l2_robotwin_config(
            "039_mug",
            model_id=11,
            base_scale=(1, 1, 1),
            base_replace_scale=True,
            override_name="039_mug",
            override_id=4,
            override_scale=(0.08, 0.08, 0.08),
            override_replace_scale=True,
        )
        (
            self.rack_modelname,
            self.rack_model_id,
            self.rack_scale,
            self.rack_replace_scale,
        ) = apply_l3_robotwin_config(
            "040_rack",
            model_id=0,
            base_scale=(1.5, 1.5, 1.5),
            base_replace_scale=True,
            override_name="013_dumbbell-rack",
            override_id=3,
            override_scale=(5, 2, 5),
            override_replace_scale=False,
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

        # Load mug
        self.mug_pose = sapien.Pose(
            p=[-0., -0.35, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 4)
        )
        mug_actor_obj = create_actor(
            scene=self.scene,
            pose=self.mug_pose,
            modelname=self.mug_modelname,
            convex=True,
            model_id=self.mug_model_id,
            scale=self.mug_scale,
            replace_scale=True,
        )
        mug_actor_obj.set_mass(0.05)
        self.mug = mug_actor_obj.actor

        # Load rack
        if is_l3_enabled():
            self.rack_pose = sapien.Pose(
                p=[0.05, 0.1, 0.55],
                q=euler2quat(0, np.pi / 2, np.pi / 1.5),
            )
        else:
            self.rack_pose = sapien.Pose(
                p=[-0.05, -0.05, 0.0],
                q=euler2quat(np.pi / 2, 0, -np.pi * 3 / 4),
            )
        rack_actor_obj = create_actor(
            scene=self.scene,
            pose=self.rack_pose,
            modelname=self.rack_modelname,
            convex=True,
            model_id=self.rack_model_id,
            is_static=True,
            replace_scale=self.rack_replace_scale,
            scale=self.rack_scale,
        )
        self.rack = rack_actor_obj.actor

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

            xyz = torch.tensor(self.mug_pose.p).repeat(b, 1)
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            # For euler(π/2, 0, γ) objects, y-mirror: γ -> π - γ
            mug_z_rot = np.pi / 4
            if is_lr_mirror_enabled():
                mug_z_rot = np.pi - mug_z_rot
            qs = torch.tensor(euler2quat(np.pi / 2, 0, mug_z_rot)).repeat(b, 1)
            self.mug.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.rack_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.rack_zs[env_idx]
            if is_l3_enabled():
                # For euler(0, π/2, γ) objects, y-mirror: γ -> -γ
                rack_z_rot = np.pi / 1.5
                if is_lr_mirror_enabled():
                    rack_z_rot = -rack_z_rot
                    rack_z_rot += np.pi  # keep mirrored rack facing expected side
                qs = torch.tensor(euler2quat(0, np.pi / 2, rack_z_rot)).repeat(b, 1)
            else:
                # For euler(π/2, 0, γ) objects, y-mirror: γ -> π - γ
                rack_z_rot = -np.pi * 3 / 4
                if is_lr_mirror_enabled():
                    rack_z_rot = np.pi - rack_z_rot
                    rack_z_rot += np.pi  # keep mirrored rack facing expected side
                qs = torch.tensor(euler2quat(np.pi / 2, 0, rack_z_rot)).repeat(b, 1)
            self.rack.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.mug_zs = []
        collision_mesh = self.mug.get_first_collision_mesh()
        self.mug_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.mug_zs = common.to_tensor(self.mug_zs, device=self.device)

        self.rack_zs = []
        collision_mesh = self.rack.get_first_collision_mesh()
        self.rack_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.rack_zs = common.to_tensor(self.rack_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        obj_to_goal_pos = self.mug.pose.p - self.rack.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_grasped = self.left_agent.is_grasping(self.mug) or self.right_agent.is_grasping(self.mug)
        contact_forces = self.scene.get_pairwise_contact_forces(self.mug, self.rack)
        is_mug_contact_rack = torch.linalg.norm(contact_forces, axis=1) > 1e-2

        # Check if mug is off the table
        is_mug_above_table = self.mug.pose.p[:, 2] > (self.mug_zs + 0.08) # 5cm above table

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

        success = is_mug_contact_rack & is_mug_above_table & (~is_grasped)

        result = dict(
            is_grasped=is_grasped,
            is_mug_contact_rack=is_mug_contact_rack,
            is_mug_above_table=is_mug_above_table,
            is_obj_placed=is_obj_placed,
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
            mug_pose=self.mug.pose.raw_pose,
            rack_pose=self.rack.pose.raw_pose,
            is_grasped=info["is_grasped"],
            is_mug_above_table=info["is_mug_above_table"],
        )
        if "state" in self.obs_mode:
            obs.update(
                left_tcp_to_obj_pos=self.mug.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.mug.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.mug.pose.p - self.rack.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        mug_pos = self.mug.pose.p
        rack_pos = self.rack.pose.p + torch.tensor([-0.07, 0, 0.15], device=self.device)  # use top of rack as target
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        is_mug_above_table = info["is_mug_above_table"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Phase 1: REACH_MUG - only before grasping
        reach_gate = ~is_grasped
        r_reach_mug = reach_reward(left_tcp, mug_pos, scale=5) * reach_gate.float()
        r_reach_mug = torch.where(is_grasped, ones, r_reach_mug)
        self.reward_tracker.update("reach_mug", r_reach_mug)

        # Phase 2: GRASP_MUG - only after reaching, before placed
        grasp_gate = is_grasped & ~is_obj_placed
        r_grasp_mug = grasp_reward(left_tcp, mug_pos, is_grasped, proximity_scale=5) * grasp_gate.float()
        self.reward_tracker.update("grasp_mug", r_grasp_mug)

        # Phase 3: TRANSPORT_MUG - only after grasping, before placed
        transport_gate = is_grasped & ~is_obj_placed
        r_transport_mug = transport_reward(mug_pos, rack_pos, is_grasped, scale=3) * transport_gate.float()
        r_transport_mug = torch.where(success, ones, r_transport_mug)  # full reward if placed
        self.reward_tracker.update("transport_mug", r_transport_mug)

        # Phase 4: PLACE_MUG - continuous reward (after transport started and released)
        transport_peak = self.reward_tracker._peaks["transport_mug"]
        mug_to_rack_dist = torch.linalg.norm(mug_pos - rack_pos, dim=-1)
        r_place_mug = exp_reward(mug_to_rack_dist, scale=3) * (transport_peak > 0).float() * (~is_grasped).float()
        r_place_mug = torch.where(success, ones, r_place_mug)
        self.reward_tracker.update("place_mug", r_place_mug)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
