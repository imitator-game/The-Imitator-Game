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
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.scene_builder.table.utils import create_actor
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l3_robotwin_model,
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

REWARD_PHASES = ['reach', 'grasp', 'transport']


@register_env("TwoRobotTransFoodL3-v1", max_episode_steps=100)
class TwoRobotTransFoodEnvL3(BaseEnv):
    """
        **Task Description:**
        The goal is to sequentially pick up two fruits and place them into a bowl_src.
        The fruits must be picked up one at a time, and each fruit must be
        successfully placed inside the bowl_src before proceeding to the next.

        **Randomizations:**
        The types of fruits are randomly selected for each episode.(To Be Done)
        The initial positions and orientations of the fruits are randomized.
        The position and orientation of the bowl_src are randomized within the workspace.
        The order in which the fruits are picked may vary across episodes.

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
        self.goal_thresh = 0.2
        self.bowl_src_modelname = "002_bowl"
        self.bowl_src_model_id = get_model_id(self.bowl_src_modelname, model_id=2)
        self.bowl_modelname, self.bowl_model_id = apply_l3_robotwin_model(
            "002_bowl",
            model_id=2,
            override_name="002_bowl",
            override_id=3,
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
        bowl_upright_pose = sapien.Pose(p=[0, 0, 0], q=euler2quat(np.pi / 2, 0.0, 0.0))
        bowl_scale = (1.08, 1.08, 1.08)

        # === 1. Load the source bowl (used to hold food and spoon) ===
        bowl_src_actor_obj = create_actor(
            scene=self.scene,
            pose=bowl_upright_pose,
            modelname=self.bowl_src_modelname,
            convex=True,
            model_id=self.bowl_src_model_id,
            scale=bowl_scale,
            _idx_if_repeat=1,
        )
        bowl_src_actor_obj.set_mass(0.4)
        self.bowl_src = bowl_src_actor_obj.actor

        # === 2. Load the target bowl (empty) ===
        bowl_dst_actor_obj = create_actor(
            scene=self.scene,
            pose=bowl_upright_pose,
            modelname=self.bowl_modelname,
            convex=True,
            is_static=True,
            model_id=5,
            scale=bowl_scale,
            _idx_if_repeat=2,
        )
        self.bowl_dst = bowl_dst_actor_obj.actor

        # === 3. Load food ===

        food_builder = actors.get_actor_builder(
            self.scene, id=f'ycb:026_sponge', scales=[0.2, 0.2, 0.2]
        )
        food_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        # self.food = food_builder.build(name='food')


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize the grasp step counter
            if not hasattr(self, "grasped_steps"):
                self.grasped_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.grasped_steps[env_idx] = 0
            if not hasattr(self, "is_bowl_src_ever_in_bowl_dst"):
                self.is_bowl_src_ever_in_bowl_dst = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.is_bowl_src_ever_in_bowl_dst[env_idx] = False

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

            # === 1. Set the positions of the two bowls ===
            # Set base positions: one on the left, one on the right
            # Y axis: -0.2 (left / source bowl), 0.2 (right / target bowl)
            src_pos = torch.zeros((b, 3), device=self.device)
            dst_pos = torch.zeros((b, 3), device=self.device)

            # Source bowl position (green bowl)
            src_pos[:, 0] = 0  # X
            src_pos[:, 1] = -0.15  # Y (left)
            src_pos[:, 2] = self._bowl_z  # Z (on the ground)
            src_pos = apply_l1_offset_xy(src_pos, offset=(-0.1, -0.1))
            # Add a little random noise
            src_pos[:, :2] += torch.rand((b, 2), device=self.device) * 0.02

            # Target bowl position (gray bowl)
            dst_pos[:, 0] = 0
            dst_pos[:, 1] = 0.15  # Y (right)
            dst_pos[:, 2] = self._bowl_z
            dst_pos[:, :2] += torch.rand((b, 2), device=self.device) * 0.02

            # Apply position and orientation (bowl opening facing up)
            bowl_qs = torch.tensor(euler2quat(np.pi / 2, 0.0, 0.0), device=self.device, dtype=src_pos.dtype)
            self.bowl_src.set_pose(Pose.create_from_pq(src_pos, bowl_qs))
            self.bowl_dst.set_pose(Pose.create_from_pq(dst_pos, bowl_qs))

            # === 2. Set the food (inside the source bowl) ===
            # food_pos = src_pos.clone() # Copy the source bowl position
            # # Raise Z: bowl bottom thickness + slightly hovering so it falls in
            # # Assume bowl bottom thickness is about 1-2cm, give 5cm to ensure it is above the bowl
            # food_pos[:, 2] += self._food_z +0.005
            # food_pos[:, 1] += 0.02
            # quat = euler2quat(5*np.pi/6 ,0, 0)
            # self.food.set_pose(Pose.create_from_pq(food_pos,quat))

    def _after_reconfigure(self, options: dict):

        self._bowl_z = None
        collision_mesh = self.bowl_src.get_first_collision_mesh()
        # this value is used to set object pose so the bottom is at z=0
        self._bowl_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

        # self._food_z = None
        # collision_mesh = self.food.get_first_collision_mesh()
        # # this value is used to set object pose so the bottom is at z=0
        # self._food_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        is_grasped = (
                self.left_agent.is_grasping(self.bowl_src)
                | self.right_agent.is_grasping(self.bowl_src)
        )
        # Record grasp steps (vectorized environment, so accumulate directly here)
        self.grasped_steps += is_grasped.int()

        # Compute the current distance to the target bowl
        bowl_src_to_bowl_dst_dist = torch.linalg.norm(self.bowl_src.pose.p[:, :2] - self.bowl_dst.pose.p[:, :2], axis=1)
        is_bowl_src_in_bowl_dst = bowl_src_to_bowl_dst_dist <= 0.15
        self.is_bowl_src_ever_in_bowl_dst = self.is_bowl_src_ever_in_bowl_dst | is_bowl_src_in_bowl_dst
        is_bowl_src_ever_in_bowl_dst = self.is_bowl_src_ever_in_bowl_dst

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

        success = is_bowl_src_ever_in_bowl_dst

        result = dict(
            success=success,
            is_grasped=is_grasped,
            grasped_steps=self.grasped_steps,
            is_bowl_src_in_bowl_dst=is_bowl_src_in_bowl_dst,
            is_bowl_src_ever_in_bowl_dst=is_bowl_src_ever_in_bowl_dst,
            is_robot_static=is_robot_static,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            is_grasped=info["is_grasped"],
            grasped_steps=info["grasped_steps"],
            is_bowl_src_in_bowl_dst=info["is_bowl_src_in_bowl_dst"],
        )
        if "state" in self.obs_mode:
            obs.update(
                bowl_src_pose=self.bowl_src.pose.raw_pose,
                bowl_dst_pose=self.bowl_dst.pose.raw_pose,
                tcp_to_bowl_src_pos=self.bowl_src.pose.p - self.left_agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        bowl_src_pos = self.bowl_src.pose.p
        bowl_dst_pos = self.bowl_dst.pose.p
        is_grasped = info["is_grasped"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        left_r = reach_reward(left_tcp, bowl_src_pos, scale=5.0)
        right_r = reach_reward(right_tcp, bowl_src_pos, scale=5.0)
        r_reach = torch.maximum(left_r, right_r)
        r_reach = torch.where(is_grasped | success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        left_gr = grasp_reward(left_tcp, bowl_src_pos, is_grasped, proximity_scale=5.0)
        right_gr = grasp_reward(right_tcp, bowl_src_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.maximum(left_gr, right_gr)
        r_grasp = torch.where(success, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        r_transport = transport_reward(bowl_src_pos, bowl_dst_pos, is_grasped, scale=5.0)
        r_transport = torch.where(success, ones, r_transport)
        self.reward_tracker.update("transport", r_transport)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)