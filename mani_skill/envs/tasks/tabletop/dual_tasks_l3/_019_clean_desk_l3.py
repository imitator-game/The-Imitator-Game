import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, Tuple

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
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    transport_reward,
    wipe_progress_reward,
    above_reward,
    tanh_reward,
    RewardTracker,
)

REWARD_PHASES = ['reach', 'grasp', 'approach', 'wipe']


@register_env("TwoRobotCleanDeskL3-v1", max_episode_steps=100)
class TwoRobotCleanDeskEnvL3(BaseEnv):
    """
        **Task Description:**
        The goal is to pick up a tissue and wipe the surface of the pot three consecutive times.
        Each wipe requires the tissue to make contact with the pot and move across its surface.
        After completing three wipes, the tissue must be placed back to its original position.

        **Randomizations:**
        The initial position of the tissue are randomized within a reachable area.
        The position and orientation of the pot are randomized on the workspace.

        **Success Conditions:**
        Just For Testing.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

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
        self.goal_thresh = 0.12

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
        # global WARNED_ONCE
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Load tissue
        self.tissue_modelname, self.tissue_model_id = (
            "023_tissue-box",
            3,
        )
        self.tissue_pose = sapien.Pose(
            p=[0.1, -0.05, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 4)
        )
        tissue_actor_obj = create_actor(
            scene=self.scene,
            pose=self.tissue_pose,
            modelname=self.tissue_modelname,
            convex=True,
            model_id=self.tissue_model_id,
            scale=(0.2, 0.13, 0.06),
            replace_scale=True,
        )
        self.tissue = tissue_actor_obj.actor
        self.tissue.set_mass(0.5)

        brick_position = [0.0, 0.0, 0.0]
        brick_rotation = [0, 0, 0]

        brick_ycb_id = override_id = "061_foam_brick"
        brick_builder = actors.get_actor_builder(
            self.scene,
            id=f'ycb:{brick_ycb_id}',
            scales=[0.4]
        )
        brick_pose = sapien.Pose(p=brick_position, q=euler2quat(*brick_rotation))
        brick_builder.initial_pose = brick_pose
        self.brick = brick_builder.build(name=f'brick')

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "tissue_ever_cleaned_desk"):
                self.tissue_ever_cleaned_desk = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.tissue_ever_cleaned_desk[env_idx] = False
            if not hasattr(self, "tissue_last_pos"):
                self.tissue_last_pos = self.tissue.pose.p.clone()
            if not hasattr(self, "tissue_clean_dist"):
                self.tissue_clean_dist = torch.zeros(self.num_envs, device=self.device)
            if not hasattr(self, "tissue_clean_steps"):
                self.tissue_clean_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.tissue_last_pos[env_idx] = self.tissue.pose.p[env_idx].clone()
            self.tissue_clean_dist[env_idx] = 0.0
            self.tissue_clean_steps[env_idx] = 0

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

            # TODO: adding pose randomization
            xyz = self.tissue.pose.p
            qs = self.tissue.pose.q
            xyz[:, :2] += torch.tensor([-0.12, -0.17], device=self.device)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self._tissue_z
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.0))
            # qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.tissue.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = self.brick.pose.p
            qs = self.brick.pose.q
            xyz[:, :2] += torch.tensor([-0.02, -0.02], device=self.device)
            xyz[:, :2] += (torch.rand((b, 2))) * 0.02
            xyz[:, 2] = self._brick_z
            # qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.brick.set_pose(Pose.create_from_pq(xyz, qs))

    # Usually called after load_scene
    def _after_reconfigure(self, options: dict):
        self._tissue_z = None
        collision_mesh = self.tissue.get_first_collision_mesh()
        # this value is used to set object pose so the bottom is at z=0
        self._tissue_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

        self._brick_z = None
        collision_mesh = self.brick.get_first_collision_mesh()
        # this value is used to set object pose so the bottom is at z=0
        self._brick_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        is_grasped = (
                self.left_agent.is_grasping(self.tissue)
                | self.right_agent.is_grasping(self.tissue)
        )

        # 1. Check if tissue is currently close to brick (to update historical flag)
        tissue_to_brick_dist = torch.linalg.norm(self.tissue.pose.p - self.brick.pose.p, axis=1)
        is_tissue_colliding_brick = tissue_to_brick_dist <= self.goal_thresh

        # Historical condition: impulse > 0.01 AND grasped
        is_contact = is_tissue_colliding_brick & is_grasped
        self.tissue_ever_cleaned_desk = self.tissue_ever_cleaned_desk | is_contact

        # Accumulate clean distance and steps while in contact.
        delta_xy = self.tissue.pose.p[:, :2] - self.tissue_last_pos[:, :2]
        step_dist = torch.linalg.norm(delta_xy, axis=1)
        self.tissue_clean_dist += step_dist * is_contact.float()
        self.tissue_clean_steps += is_contact.int()
        self.tissue_last_pos = self.tissue.pose.p.clone()

        is_robot_static = (
                self.left_agent.is_static(0.2)
                & self.right_agent.is_static(0.2)
        )

        # Success: sufficient cleaning steps and motion while in contact
        success = (self.tissue_clean_steps >= 90) & (self.tissue_clean_dist >= 0.3)

        result = dict(
            is_grasped=is_grasped,
            is_tissue_colliding_brick=is_tissue_colliding_brick,
            tissue_ever_cleaned_desk=self.tissue_ever_cleaned_desk,
            tissue_clean_dist=self.tissue_clean_dist,
            tissue_clean_steps=self.tissue_clean_steps,
            is_robot_static=is_robot_static,
            success=success,
            # For backward compatibility with reward function
            is_obj_placed=self.tissue_ever_cleaned_desk,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            is_grasped=info["is_grasped"],
            tissue_ever_cleaned_desk=info["tissue_ever_cleaned_desk"],
        )
        if "state" in self.obs_mode:
            obs.update(
                tissue_pose=self.tissue.pose.raw_pose,
                brick_pose=self.brick.pose.raw_pose,
                left_tcp_to_tissue_pos=self.tissue.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_tissue_pos=self.tissue.pose.p - self.right_agent.tcp.pose.p,
                tissue_to_brick_pos=self.brick.pose.p - self.tissue.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        tissue_pos = self.tissue.pose.p
        brick_pos = self.brick.pose.p
        is_grasped = info["is_grasped"]
        ever_cleaned = info["tissue_ever_cleaned_desk"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        left_r = reach_reward(left_tcp, tissue_pos, scale=5.0)
        right_r = reach_reward(right_tcp, tissue_pos, scale=5.0)
        r_reach = torch.maximum(left_r, right_r)
        r_reach = torch.where(is_grasped | ever_cleaned | success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        left_gr = grasp_reward(left_tcp, tissue_pos, is_grasped, proximity_scale=5.0)
        right_gr = grasp_reward(right_tcp, tissue_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.maximum(left_gr, right_gr)
        r_grasp = torch.where(ever_cleaned | success, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        r_approach = transport_reward(tissue_pos, brick_pos, is_grasped, scale=5.0)
        r_approach = torch.where(ever_cleaned | success, ones, r_approach)
        self.reward_tracker.update("approach", r_approach)

        r_wipe = wipe_progress_reward(self.tissue_clean_steps, 90, self.tissue_clean_dist, 0.3)
        r_wipe = torch.where(success, ones, r_wipe)
        self.reward_tracker.update("wipe", r_wipe)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)