import numpy as np
import sapien
import torch

from typing import Any, Dict, Tuple
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
    wipe_progress_reward,
)

REWARD_PHASES = [
    "reach_grind",
    "grasp_grind",
    "approach_bowl",
    "press_juicer",
]


@register_env("TwoRobotPressJuicerL3-v1", max_episode_steps=100)
class TwoRobotPressJuicerEnvL3(BaseEnv):
    # Refactored to use the same scene/eval style as TwoRobotGrindFood L2.
    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    goal_thresh = 0.15
    press_xy_thresh = 0.08
    press_z_thresh = 0.12
    press_motion_thresh = 0.003
    press_steps_required = 2
    press_motion_target = 0.03

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

        # Fixed to the L2 asset combination of task 044.
        self.grind_modelname = "086_woodenblock"
        self.grind_model_id = 1
        self.bowl_modelname = "002_bowl"
        self.bowl_model_id = 2
        # Static fruit inside bowl (robotwin asset).
        self.fruit_modelname = "035_apple"
        self.fruit_model_id = get_model_id(self.fruit_modelname, model_id=0)
        self.fruit_scale = (0.7, 0.7, 0.7)
        # Tunable offset (world frame) from bowl center.
        self.fruit_in_bowl_offset = np.array([0.0, -0.02, 0.02], dtype=np.float32)

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
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
        pose = sapien_utils.look_at([0.8, 0.0, 0.75], [0.0, 0.0, 0.25])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )
        self.agent.__init__(self.agent.agents, wrist_sensor=self.wrist_sensor)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # Grind object (movable)
        self.grind_pose = sapien.Pose(
            p=[0.09, -0.23, 0.05],
            q=euler2quat(np.pi / 2, 0, np.pi / 3),
        )
        grind_actor_obj = create_actor(
            scene=self.scene,
            pose=self.grind_pose,
            modelname=self.grind_modelname,
            convex=True,
            model_id=self.grind_model_id,
            scale=(1, 1.5, 1),
            replace_scale=True,
        )
        self.grind = grind_actor_obj.actor
        grind_actor_obj.set_mass(0.5)

        # Static block
        self.block_pose = sapien.Pose(
            p=[0.09, -0.24, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2),
        )
        block_actor_obj = create_actor(
            scene=self.scene,
            pose=self.block_pose,
            modelname="004_fluted-block",
            convex=True,
            model_id=1,
            is_static=True,
            scale=(1.05, 1.05, 1.05),
        )
        self.block = block_actor_obj.actor

        # Static bowl
        self.bowl_pose = sapien.Pose(
            p=[0.05, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2),
        )
        bowl_actor_obj = create_actor(
            scene=self.scene,
            pose=self.bowl_pose,
            modelname=self.bowl_modelname,
            convex=True,
            model_id=self.bowl_model_id,
            is_static=True,
            scale=(1.3, 1.3, 1.3),
        )
        self.bowl = bowl_actor_obj.actor

        # Static fruit, posed into the bowl during episode initialization.
        self.fruit_pose = sapien.Pose(
            p=[self.bowl_pose.p[0], self.bowl_pose.p[1], self.bowl_pose.p[2] + float(self.fruit_in_bowl_offset[2])],
            q=euler2quat(0, 0, 0),
        )
        fruit_actor_obj = create_actor(
            scene=self.scene,
            pose=self.fruit_pose,
            modelname=self.fruit_modelname,
            convex=True,
            model_id=self.fruit_model_id,
            is_static=True,
            scale=self.fruit_scale,
            replace_scale=False,
        )
        self.fruit = fruit_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            # Keep deterministic placement as in task 044 when only L2 is enabled.
            self.grind.set_pose(Pose.create_from_pq(torch.tensor(self.grind_pose.p), torch.tensor(self.grind_pose.q)))
            self.block.set_pose(Pose.create_from_pq(torch.tensor(self.block_pose.p), torch.tensor(self.block_pose.q)))

            bowl_xyz = torch.tensor(self.bowl_pose.p)
            bowl_xyz[:2] += torch.rand(2) * 0.02
            bowl_xyz[2] = self.bowl_zs[env_idx]
            self.bowl.set_pose(Pose.create_from_pq(bowl_xyz, torch.tensor(self.bowl_pose.q)))

            # Keep the fruit inside the bowl by following bowl XY.
            fruit_xyz = bowl_xyz.clone()
            fruit_xyz[0] += float(self.fruit_in_bowl_offset[0])
            fruit_xyz[1] += float(self.fruit_in_bowl_offset[1])
            fruit_xyz[2] = bowl_xyz[2] + self.fruit_zs[env_idx] + float(self.fruit_in_bowl_offset[2])
            self.fruit.set_pose(Pose.create_from_pq(fruit_xyz, torch.tensor(self.fruit_pose.q)))

            if not hasattr(self, "grind_finish"):
                self.grind_finish = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if not hasattr(self, "press_motion_steps"):
                self.press_motion_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            if not hasattr(self, "press_motion_dist"):
                self.press_motion_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            if not hasattr(self, "prev_grind_xy"):
                self.prev_grind_xy = self.grind.pose.p[:, :2].clone()
            self.grind_finish[env_idx] = False
            self.press_motion_steps[env_idx] = 0
            self.press_motion_dist[env_idx] = 0.0
            self.prev_grind_xy[env_idx] = self.grind.pose.p[env_idx, :2]
            if not hasattr(self, "prev_grind_pos"):
                self.prev_grind_pos = self.grind.pose.p.clone()
            self.prev_grind_pos[env_idx] = self.grind.pose.p[env_idx]

    def _after_reconfigure(self, options: dict):
        self.bowl_zs = []
        collision_mesh = self.bowl.get_first_collision_mesh()
        self.bowl_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.bowl_zs = common.to_tensor(self.bowl_zs, device=self.device)
        self.fruit_zs = []
        fruit_collision_mesh = self.fruit.get_first_collision_mesh()
        self.fruit_zs.append(-fruit_collision_mesh.bounding_box.bounds[0, 2])
        self.fruit_zs = common.to_tensor(self.fruit_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        obj_to_goal_pos = self.grind.pose.p - self.bowl.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        left_grasped = self.left_agent.is_grasping(self.grind)
        right_grasped = self.right_agent.is_grasping(self.grind)
        is_grasped = torch.logical_or(left_grasped, right_grasped)

        curr_grind_pos = self.grind.pose.p
        if not hasattr(self, "prev_grind_pos") or self.prev_grind_pos.shape != curr_grind_pos.shape:
            self.prev_grind_pos = curr_grind_pos.clone()
        grind_step_motion = torch.linalg.norm(curr_grind_pos - self.prev_grind_pos, axis=1)
        grind_vertical_motion = torch.abs(curr_grind_pos[:, 2] - self.prev_grind_pos[:, 2])
        obj_to_goal_xy = torch.linalg.norm(obj_to_goal_pos[:, :2], axis=1)
        obj_to_goal_z = obj_to_goal_pos[:, 2]
        is_press_pose = (
            (obj_to_goal_xy <= self.press_xy_thresh)
            & (torch.abs(obj_to_goal_z) <= self.press_z_thresh)
        )
        currently_pressing = is_grasped & is_press_pose & (
            grind_vertical_motion >= self.press_motion_thresh
        )

        if not hasattr(self, "grind_finish") or self.grind_finish.shape != currently_pressing.shape:
            self.grind_finish = torch.zeros_like(currently_pressing)
        self.press_motion_steps += currently_pressing.int()
        self.press_motion_dist += grind_vertical_motion * currently_pressing.float()
        self.grind_finish = self.grind_finish | (
            (self.press_motion_steps >= self.press_steps_required)
            & (self.press_motion_dist >= self.press_motion_target)
        )
        self.prev_grind_xy = curr_grind_pos[:, :2].clone()
        self.prev_grind_pos = curr_grind_pos.clone()

        is_robot_static = torch.logical_and(self.left_agent.is_static(0.2), self.right_agent.is_static(0.2))
        success = self.grind_finish | currently_pressing
        result = dict(
            is_grasped=is_grasped,
            left_grasped=left_grasped,
            right_grasped=right_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            obj_to_goal_xy=obj_to_goal_xy,
            obj_to_goal_z=obj_to_goal_z,
            is_press_pose=is_press_pose,
            currently_pressing=currently_pressing,
            grind_finish=self.grind_finish,
            press_motion_steps=self.press_motion_steps,
            press_motion_dist=self.press_motion_dist,
            grind_step_motion=grind_step_motion,
            grind_vertical_motion=grind_vertical_motion,
            is_robot_static=is_robot_static,
            is_grasping=is_grasped,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.grind.pose.p,
            is_grasped=info["is_grasped"],
            currently_pressing=info["currently_pressing"],
            grind_finish=info["grind_finish"],
            press_motion_steps=info["press_motion_steps"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.grind.pose.raw_pose,
                left_tcp_to_obj_pos=self.grind.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.grind.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.grind.pose.p - self.bowl.pose.p,
            )
        return obs

    def _prioritize_reward_info(self, info: Dict, total_reward: torch.Tensor):
        ordered = dict(reward=total_reward.clone())
        if "success" in info:
            ordered["success"] = info["success"]
        for key in list(info.keys()):
            if key.startswith("R"):
                ordered[key] = info[key]
        for key in list(info.keys()):
            if key.startswith("peak_r_"):
                ordered[key] = info[key]
        for key, value in info.items():
            if key not in ordered:
                ordered[key] = value
        info.clear()
        info.update(ordered)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        grind_pos = self.grind.pose.p
        bowl_target = self.bowl.pose.p.clone()
        bowl_target[:, 2] += 0.02
        success = info["success"]
        is_grasped = info["is_grasped"]
        left_grasped = info["left_grasped"]
        ever_pressed = info["grind_finish"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(left_tcp, grind_pos, scale=5.0)
        r_reach = torch.where(is_grasped | ever_pressed | success, ones, r_reach)
        self.reward_tracker.update("reach_grind", r_reach)

        r_grasp = grasp_reward(
            left_tcp, grind_pos, left_grasped, proximity_scale=5.0
        )
        r_grasp = torch.where(ever_pressed | success, ones, r_grasp)
        self.reward_tracker.update("grasp_grind", r_grasp)

        r_approach = transport_reward(grind_pos, bowl_target, is_grasped, scale=5.0)
        r_approach = torch.where(ever_pressed | success, ones, r_approach)
        self.reward_tracker.update("approach_bowl", r_approach)

        r_press = wipe_progress_reward(
            steps=info["press_motion_steps"],
            target_steps=self.press_steps_required,
            dist=info["press_motion_dist"],
            target_dist=self.press_motion_target,
        )
        r_press = torch.where(success, ones, r_press)
        self.reward_tracker.update("press_juicer", r_press)

        self.reward_tracker.write_to_info(info)
        total_reward = torch.where(success, ones, self.reward_tracker.total())
        self._prioritize_reward_info(info, total_reward)
        return total_reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
