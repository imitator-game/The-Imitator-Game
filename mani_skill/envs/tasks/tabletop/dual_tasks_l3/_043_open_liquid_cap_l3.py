import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, List, Union, Tuple
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    grasp_reward,
    normalized_progress,
    place_reward,
    reach_reward,
    transport_reward,
)


REWARD_PHASES = ["reach_cap", "grasp_cap", "open_cap", "place_cap"]


@register_env("TwoRobotOpenLiquidCapL3-v1", max_episode_steps=100)
class TwoRobotOpenLiquidCapEnvL3(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.1
    open_height_thresh = 0.06

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
        # Fixed L0 assets for L3 standalone task (no L-level replacement logic).
        self.woodenblock_modelname, self.woodenblock_model_id = "021_cup", 4
        self.cup_modelname, self.cup_model_id = "021_cup", 5
        self.plate_modelname, self.plate_model_id = "008_tray", 0
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

    # Disable task-level LR mirror for this standalone L3 task.
    def _maybe_apply_lr_mirror(self, env_idx: torch.Tensor, options: dict):
        self._lr_mirror_applied_this_reset = False
        return

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Load liquid
        self.liquid_pose = sapien.Pose(
            p=[-0.0, -0., -0.07],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        liquid_actor_obj = create_actor(
            scene=self.scene,
            pose=self.liquid_pose,
            modelname=self.cup_modelname,
            convex=True,
            model_id=self.cup_model_id,
            is_static=True,
            replace_scale=True,
            scale=(0.03, 0.2, 0.03),
            _idx_if_repeat=1,
        )
        self.liquid = liquid_actor_obj.actor

        # Load cap
        self.cap_pose = sapien.Pose(
            p=[0.00, 0.00, -0.28],
            q=euler2quat(-np.pi / 2, 0, np.pi)
        )
        cap_actor_obj = create_actor(
            scene=self.scene,
            pose=self.cap_pose,
            modelname=self.woodenblock_modelname,
            convex=True,
            model_id=self.woodenblock_model_id,
            # replace_scale=True,
            scale=(0.5, 0.7, 0.5),
            _idx_if_repeat=2,
        )
        self.cap = cap_actor_obj.actor
        self.cap.set_mass(0.1)

        # Load rack
        self.rack_pose = sapien.Pose(
            p=[-0.0, 0.0, 0.05],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        rack_actor_obj = create_actor(
            scene=self.scene,
            pose=self.rack_pose,
            modelname="004_fluted-block",
            convex=True,
            is_static=True,
            model_id=1,
            scale=(0.9, 3.0, 0.9),
        )
        self.rack = rack_actor_obj.actor


        # Load plate
        self.plate_pose = sapien.Pose(
            p=[-0.0, -0.25, 0.0],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        plate_actor_obj = create_actor(
            scene=self.scene,
            pose=self.plate_pose,
            modelname=self.plate_modelname,
            convex=True,
            is_static=True,
            model_id=self.plate_model_id,
            scale=(1.0, 1.0, 1.0),
            _idx_if_repeat=3,
        )
        self.plate = plate_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            if not hasattr(self, "cap_ever_opened"):
                self.cap_ever_opened = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.cap_ever_opened[env_idx] = False

            if not hasattr(self, "cap_initial_pos"):
                self.cap_initial_pos = torch.zeros(
                    (self.num_envs, 3), dtype=torch.float32, device=self.device
                )

            shared_xy_offset = torch.rand((b, 2)) * 0.02
            xyz = torch.tensor(self.liquid_pose.p).repeat(b, 1)
            xyz[:, :2] += shared_xy_offset
            xyz[:, 2] = self.liquid_zs[env_idx]
            qs = torch.tensor(self.liquid_pose.q).repeat(b, 1)
            self.liquid.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.rack_pose.p).repeat(b, 1)
            xyz[:, :2] += shared_xy_offset
            xyz[:, 2] = self.rack_zs[env_idx]
            qs = torch.tensor(self.rack_pose.q).repeat(b, 1)
            self.rack.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.cap_pose.p).repeat(b, 1)
            xyz[:, :2] += shared_xy_offset
            xyz[:, 2] = self.cap_zs[env_idx]
            qs = torch.tensor(self.cap_pose.q).repeat(b, 1)
            self.cap.set_pose(Pose.create_from_pq(xyz, qs))
            self.cap_initial_pos[env_idx] = self.cap.pose.p[env_idx]

            xyz = torch.tensor(self.plate_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.plate_zs[env_idx]
            qs = torch.tensor(self.plate_pose.q).repeat(b, 1)
            self.plate.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.liquid_zs = []
        collision_mesh = self.liquid.get_first_collision_mesh()
        self.liquid_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.liquid_zs = common.to_tensor(self.liquid_zs, device=self.device)

        self.rack_zs = []
        collision_mesh = self.rack.get_first_collision_mesh()
        self.rack_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.rack_zs = common.to_tensor(self.rack_zs, device=self.device)

        self.cap_zs = []
        collision_mesh = self.cap.get_first_collision_mesh()
        self.cap_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cap_zs = common.to_tensor(self.cap_zs, device=self.device)

        self.plate_zs = []
        collision_mesh = self.plate.get_first_collision_mesh()
        self.plate_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.plate_zs = common.to_tensor(self.plate_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        obj_to_goal_pos = self.cap.pose.p - self.plate.pose.p
        cap_to_liquid_pos = self.cap.pose.p - self.liquid.pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)
        is_obj_placed = obj_to_goal_dist <= self.goal_thresh
        left_grasped = self.left_agent.is_grasping(self.cap)
        right_grasped = self.right_agent.is_grasping(self.cap)
        is_grasped = left_grasped | right_grasped
        cap_lift_height = self.cap.pose.p[:, 2] - self.cap_initial_pos[:, 2]
        is_cap_opened = cap_lift_height >= self.open_height_thresh
        self.cap_ever_opened = self.cap_ever_opened | is_cap_opened
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

        result = dict(
            is_grasped=is_grasped,
            left_grasped=left_grasped,
            right_grasped=right_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            obj_to_goal_dist=obj_to_goal_dist,
            cap_to_liquid_pos=cap_to_liquid_pos,
            cap_lift_height=cap_lift_height,
            is_cap_opened=is_cap_opened,
            cap_ever_opened=self.cap_ever_opened,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=is_grasped,
            success=is_obj_placed & self.cap_ever_opened & (~is_grasped),
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.plate.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.plate.pose.raw_pose,
                left_tcp_to_obj_pos=self.plate.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.plate.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.plate.pose.p - self.cap.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        cap_pos = self.cap.pose.p
        plate_pos = self.plate.pose.p
        left_grasped = self.left_agent.is_grasping(self.cap)
        is_grasped = info["is_grasped"]
        ever_opened = info["cap_ever_opened"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(left_tcp, cap_pos, scale=5.0)
        r_reach = torch.where(is_grasped | ever_opened | success, ones, r_reach)
        self.reward_tracker.update("reach_cap", r_reach)

        r_grasp = grasp_reward(left_tcp, cap_pos, left_grasped, proximity_scale=5.0)
        r_grasp = torch.where(ever_opened | success, ones, r_grasp)
        self.reward_tracker.update("grasp_cap", r_grasp)

        r_open = normalized_progress(
            info["cap_lift_height"], start=0.0, target=self.open_height_thresh
        ) * is_grasped.float()
        r_open = torch.where(ever_opened | success, ones, r_open)
        self.reward_tracker.update("open_cap", r_open)

        r_transport = transport_reward(cap_pos, plate_pos, is_grasped, scale=5.0)
        r_release = place_reward(cap_pos, plate_pos, is_grasped, scale=15.0)
        r_place = torch.maximum(r_transport, r_release)
        r_place = torch.where(success, ones, r_place)
        self.reward_tracker.update("place_cap", r_place)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
