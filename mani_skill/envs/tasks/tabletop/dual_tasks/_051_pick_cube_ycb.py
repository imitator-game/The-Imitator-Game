"""
TwoRobotPickCubeYCB-v1: pick a cube and place it at the YCB-object goal.

Reference implementation adapted from the ManiSkill upstream dual-arm
pick-and-place task to the Imitator Game conventions: hi_res/wrist_sensor
cameras, RewardTracker, and L0_L3_utils level hooks.

Task: a green cube is within reach of the LEFT robot and a merged YCB object
marks the goal position which is within reach of the RIGHT robot. The two
robots cooperate: the left robot grasps the cube and the right robot receives
it / finalizes placement at the goal.
"""

import numpy as np
import sapien
import torch
from typing import Any, Dict, Tuple

from mani_skill.agents.multi_agent import MultiAgent
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

from mani_skill import ASSET_DIR

from mani_skill.envs.tasks.tabletop.utils.reward_utils import RewardTracker

WARNED_ONCE = False

REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotPickCubeYCB-v1", max_episode_steps=100)
class TwoRobotPickCubeYCBEnv(BaseEnv):
    """Pick up a cube and move it to a goal position (YCB object as goal)."""

    _sample_video_link = ("https://github.com/haosulab/ManiSkill/raw/refs/heads/main/"
                          "figures/environment_demos/TwoRobotPickCube-v1_rt.mp4")

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids=("panda_wristcam", "panda_wristcam"),
                 robot_init_qpos_noise=0.02, num_envs=1, reconfiguration_freq=None,
                 hi_res: bool = False, wrist_sensor: bool = False, **kwargs):
        self.hi_res = hi_res
        self.wrist_sensor = wrist_sensor
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None

        # YCB models used to build the merged goal object.
        self.all_model_ids = np.array(
            [
                k
                for k in load_json(
                    ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
                ).keys()
                if k
                not in [
                    "022_windex_bottle",
                    "028_skillet_lid",
                    "029_plate",
                    "059_chain",
                ]
            ]
        )
        self.all_model_ids = np.array(["029_plate"])

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(*args, robot_uids=robot_uids,
                         reconfiguration_freq=reconfiguration_freq,
                         num_envs=num_envs, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(gpu_memory_config=GPUMemoryConfig(
            found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**19,
            max_rigid_contact_count=2**21))

    @property
    def _default_sensor_configs(self):
        p1 = sapien_utils.look_at(eye=[0.3, 0.3, 0.6], target=[-0.1, 0, -0.6])
        p2 = sapien_utils.look_at(eye=[0.1, 0.0, 0.6], target=[0.0, 0, 0.0])
        p3 = sapien_utils.look_at(eye=[0.3, -0.3, 0.6], target=[-0.1, 0, -0.6])
        p4 = sapien_utils.look_at(eye=[-0.4, 0, 0.6], target=[0.0, 0, 0.2])
        if self.hi_res:
            return [
                CameraConfig("cam1", p1, 640, 480, np.pi / 2, 0.01, 100),
                CameraConfig("cam2", p2, 640, 480, np.pi / 2, 0.01, 100),
                CameraConfig("cam3", p3, 640, 480, np.pi / 2, 0.01, 100),
                CameraConfig("zed2i", p4, 1280, 720, np.pi * 1.1 / 2, 0.01, 100),
            ]
        return [CameraConfig("zed2i", p4, 224, 224, np.pi * 1.1 / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig("render_camera",
                            sapien_utils.look_at([0.8, 0.0, 0.75], [0.0, 0.0, 0.25]),
                            512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])])
        self.agent.__init__(self.agent.agents, wrist_sensor=self.wrist_sensor)

    def _load_scene(self, options: dict):
        global WARNED_ONCE
        self.table_scene = TableSceneBuilder(env=self,
                                             robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        model_ids = self._batched_episode_rng.choice(self.all_model_ids, replace=True)
        if (self.num_envs > 1 and self.num_envs < len(self.all_model_ids)
                and self.reconfiguration_freq <= 0 and not WARNED_ONCE):
            WARNED_ONCE = True
            print("Fewer parallel envs than YCB models: some models will be "
                  "skipped unless reconfiguration_freq >= 1.")

        self._objs: list[Actor] = []
        self.obj_heights = []
        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
            builder.initial_pose = sapien.Pose(p=[-0.2, 0.0, 0])
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])

        # Green cube: the object to move.
        self.cube = actors.build_cube(self.scene, half_size=self.cube_half_size,
                                      color=[0, 1, 0, 1], name="cube",
                                      initial_pose=sapien.Pose(
                                          p=[-0.2, -0.2, self.cube_half_size]))
        # Merged YCB object: the goal position.
        self.obj = Actor.merge(self._objs, name="ycb_object")
        self.add_to_state_dict_registry(self.obj)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = self.obj.pose.p
            xyz[:, :2] += torch.rand((b, 2)) * 0.2
            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            xyz = self.cube.pose.p
            xyz[:, :2] += torch.rand((b, 2)) * 0.2
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        obj_to_goal_pos = self.cube.pose.p - self.obj.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_grasped = self.left_agent.is_grasping(self.obj) or self.right_agent.is_grasping(self.obj)
        is_robot_static = self.left_agent.is_static(0.2) and self.right_agent.is_static(0.2)
        result = dict(is_grasped=is_grasped, obj_to_goal_pos=obj_to_goal_pos,
                      is_obj_placed=is_obj_placed, is_robot_static=is_robot_static,
                      is_grasping=is_grasped,
                      success=torch.logical_and(is_obj_placed, is_robot_static))
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
                   right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
                   goal_pos=self.obj.pose.p, is_grasped=info["is_grasped"])
        if "state" in self.obs_mode:
            obs.update(obj_pose=self.obj.pose.raw_pose,
                       left_tcp_to_obj_pos=self.obj.pose.p - self.left_agent.tcp.pose.p,
                       right_tcp_to_obj_pos=self.obj.pose.p - self.right_agent.tcp.pose.p,
                       cube_pose=self.cube.pose.raw_pose,
                       left_tcp_to_cube_pos=self.cube.pose.p - self.left_agent.tcp.pose.p,
                       right_tcp_to_cube_pos=self.cube.pose.p - self.right_agent.tcp.pose.p,
                       cube_to_goal_pos=self.obj.pose.p - self.cube.pose.p)
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_cube_dist = torch.linalg.norm(
            self.cube.pose.p - self.left_agent.tcp.pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_cube_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        cube_to_goal_dist = torch.linalg.norm(
            self.cube.pose.p - self.obj.pose.p, axis=1)
        place_reward = 1 - torch.tanh(5 * cube_to_goal_dist)
        reward += place_reward * is_grasped
        reward += info["is_obj_placed"] * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.left_agent.robot.get_qvel()[..., :-2], axis=1))
        reward += static_reward * info["is_obj_placed"] * is_grasped
        reward[info["success"]] = 6
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 21
