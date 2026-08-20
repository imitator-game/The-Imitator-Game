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
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_ycb_model_id,
    apply_l3_ycb_model_id,
    is_lr_mirror_enabled,
    mirror_xyz,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    grasp_reward,
    reach_reward,
    tanh_reward,
    transport_reward,
)


REWARD_PHASES = [
    "reach_knife",
    "grasp_knife",
    "transport_knife",
    "place_knife",
    "reach_fork",
    "grasp_fork",
    "transport_fork",
    "place_fork",
]


@register_env("TwoRobotKnifeBowlForkL3-v1", max_episode_steps=100)
class TwoRobotKnifeBowlForkEnvL3(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.15

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
        # Replace brush-pen with woodenblock (base1 variant).
        self.knife_modelname = "086_woodenblock"
        self.knife_model_id = get_model_id(self.knife_modelname, model_id=1)
        # Tunable anisotropic scale to make it thinner/longer.
        # Increase the 3rd value to make it longer; decrease first two to make it slimmer.
        self.knife_scale = (0.3, 3, 0.3)
        # Brush-on-bowl fine-tuning params (meters/radians).
        self.brush_on_bowl_offset = np.array([0.08, -0.08, 0.0], dtype=np.float32)
        self.brush_on_bowl_euler = np.array([0.0, 0.0, np.pi/4], dtype=np.float32)
        # self.all_model_ids = np.array(["005_tomato_soup_can"])
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
        
        # Load fork
        appliance_model_id = apply_l2_ycb_model_id(
            "030_fork", override_id="031_spoon"
        )
        appliance_builder = actors.get_actor_builder(
            self.scene,
            id=f'ycb:{appliance_model_id}',
            scales=[1.0]
        )
        # Fork base pose.
        appliance_builder._mass = 0.5
        self.fork_pose = sapien.Pose(p=[-0.2, 0.2, 0.0], q=euler2quat(0, 0, 0))
        appliance_builder.initial_pose = self.fork_pose
        self.fork = appliance_builder.build(name=f'{appliance_model_id}')
        
        # Woodenblock base pose (actual on-bowl placement is set in _initialize_episode).
        self.knife_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(0, 0, 0))
        knife_obj = create_actor(
            scene=self.scene,
            pose=self.knife_pose,
            modelname=self.knife_modelname,
            convex=True,
            model_id=self.knife_model_id,
            scale=self.knife_scale,
            replace_scale=False,
            mass=0.5,
        )
        self.knife = knife_obj.actor
        
        # Load bowl
        appliance_model_id = apply_l3_ycb_model_id(
            "024_bowl", override_id="029_plate"
        )
        appliance_builder = actors.get_actor_builder(
            self.scene,
            id=f'ycb:{appliance_model_id}',
            scales=[1.0]
        )

        self.bowl_pose = sapien.Pose(p=[0.0, 0.0, -0.02], q=euler2quat(0, 0, 0))
        appliance_builder.initial_pose = self.bowl_pose
        self.bowl = appliance_builder.build_static(name=f'{appliance_model_id}')
        
        self.objects = [self.bowl, self.fork, self.knife]
        self._object_initial_poses = [self.bowl_pose, self.fork_pose, self.knife_pose]


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

            # 1) Bowl
            bowl_xyz = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
            bowl_xyz[:, 0] = float(self.bowl_pose.p[0])
            bowl_xyz[:, 1] = float(self.bowl_pose.p[1])
            bowl_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            bowl_xyz[:, 2] = self.bowl_z
            bowl_q = torch.tensor(self.bowl_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.bowl.set_pose(Pose.create_from_pq(p=bowl_xyz, q=bowl_q))

            # 2) Fork
            fork_xyz = torch.zeros((b, 3), device=self.device, dtype=torch.float32)
            fork_xyz[:, 0] = float(self.fork_pose.p[0])
            fork_xyz[:, 1] = float(self.fork_pose.p[1])
            fork_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            fork_xyz[:, 2] = self.fork_z
            fork_xyz = apply_l1_offset_xy(fork_xyz, offset=(0.0, -0.1))
            fork_q = torch.tensor(self.fork_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.fork.set_pose(Pose.create_from_pq(p=fork_xyz, q=fork_q))

            # 3) Woodenblock: on top of bowl, same XY, with tunable offset and rotation.
            brush_xyz = bowl_xyz.clone()
            brush_xyz[:, 0] += float(self.brush_on_bowl_offset[0])
            brush_xyz[:, 1] += float(self.brush_on_bowl_offset[1])
            brush_xyz[:, 2] = (
                self.bowl_z
                + self.bowl_height
                + self.knife_z
                + float(self.brush_on_bowl_offset[2])
            )
            brush_q_np = euler2quat(
                float(self.brush_on_bowl_euler[0]),
                float(self.brush_on_bowl_euler[1]),
                float(self.brush_on_bowl_euler[2]),
            )
            brush_q = torch.tensor(brush_q_np, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.knife.set_pose(Pose.create_from_pq(p=brush_xyz, q=brush_q))

    def _after_reconfigure(self, options: dict):
        bowl_mesh = self.bowl.get_first_collision_mesh()
        if bowl_mesh is not None:
            bowl_bounds = bowl_mesh.bounding_box.bounds
            self.bowl_z = common.to_tensor(-bowl_bounds[0, 2], device=self.device)
            self.bowl_height = common.to_tensor(
                bowl_bounds[1, 2] - bowl_bounds[0, 2], device=self.device
            )
        else:
            self.bowl_z = common.to_tensor(0.02, device=self.device)
            self.bowl_height = common.to_tensor(0.06, device=self.device)

        fork_mesh = self.fork.get_first_collision_mesh()
        if fork_mesh is not None:
            fork_bounds = fork_mesh.bounding_box.bounds
            self.fork_z = common.to_tensor(-fork_bounds[0, 2], device=self.device)
        else:
            self.fork_z = common.to_tensor(0.01, device=self.device)

        knife_mesh = self.knife.get_first_collision_mesh()
        if knife_mesh is not None:
            knife_bounds = knife_mesh.bounding_box.bounds
            self.knife_z = common.to_tensor(-knife_bounds[0, 2], device=self.device)
        else:
            self.knife_z = common.to_tensor(0.01, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        fork_pose = self.fork.pose.p
        knife_pose = self.knife.pose.p
        fork_goal_pose = self._get_fork_goal_pos()
        knife_goal_pose = self._get_knife_goal_pos()

        fork_to_goal_pos = fork_pose - fork_goal_pose
        knife_to_goal_pos = knife_pose - knife_goal_pose
        fork_to_bowl_right_dist = torch.linalg.norm(fork_to_goal_pos, dim=-1)
        knife_to_bowl_left_dist = torch.linalg.norm(knife_to_goal_pos, dim=-1)
        is_fork_close_to_bowl_right = fork_to_bowl_right_dist <= self.goal_thresh
        is_knife_close_to_bowl_left = knife_to_bowl_left_dist <= self.goal_thresh

        is_fork_grasped = self.right_agent.is_grasping(self.fork)
        is_knife_grasped = self.left_agent.is_grasping(self.knife)

        is_robot_static = torch.logical_and(
            self.left_agent.is_static(0.2), self.right_agent.is_static(0.2)
        )

        success = torch.logical_and(
            is_fork_close_to_bowl_right, is_knife_close_to_bowl_left
        )

        result = dict(
            is_fork_grasped=is_fork_grasped,
            is_knife_grasped=is_knife_grasped,
            is_fork_close_to_bowl_right=is_fork_close_to_bowl_right,
            is_knife_close_to_bowl_left=is_knife_close_to_bowl_left,
            fork_goal_pose=fork_goal_pose,
            knife_goal_pose=knife_goal_pose,
            fork_to_goal_pos=fork_to_goal_pos,
            knife_to_goal_pos=knife_to_goal_pos,
            fork_to_bowl_right_dist=fork_to_bowl_right_dist,
            knife_to_bowl_left_dist=knife_to_bowl_left_dist,
            # Legacy aliases kept for compatibility with older logs/readers.
            is_fork_close_to_bowl_left=is_fork_close_to_bowl_right,
            is_knife_close_to_bowl_right=is_knife_close_to_bowl_left,
            fork_to_bowl_left_dist=fork_to_bowl_right_dist,
            knife_to_bowl_right_dist=knife_to_bowl_left_dist,
            is_robot_static=is_robot_static,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_knife_goal_pos(self) -> torch.Tensor:
        offset = np.array([0.115, -0.2, 0.015], dtype=np.float32)
        if is_lr_mirror_enabled():
            offset = mirror_xyz(offset)
        return self.bowl.pose.p + torch.tensor(
            offset, device=self.device, dtype=self.bowl.pose.p.dtype
        ).unsqueeze(0).repeat(self.num_envs, 1)

    def _get_fork_goal_pos(self) -> torch.Tensor:
        offset = np.array([0.015, 0.2, -0.005], dtype=np.float32)
        if is_lr_mirror_enabled():
            offset = mirror_xyz(offset)
        return self.bowl.pose.p + torch.tensor(
            offset, device=self.device, dtype=self.bowl.pose.p.dtype
        ).unsqueeze(0).repeat(self.num_envs, 1)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            bowl_pos=self.bowl.pose.p,
            fork_pos=self.fork.pose.p,
            knife_pos=self.knife.pose.p,
            is_fork_grasped=info["is_fork_grasped"],
            is_knife_grasped=info["is_knife_grasped"],
            is_fork_close_to_bowl_right=info["is_fork_close_to_bowl_right"],
            is_knife_close_to_bowl_left=info["is_knife_close_to_bowl_left"],
            fork_to_bowl_right_dist=info["fork_to_bowl_right_dist"],
            knife_to_bowl_left_dist=info["knife_to_bowl_left_dist"],
        )
        # if "state" in self.obs_mode:
        #     obs.update(
        #         bowl_pose=self.bowl.pose.raw_pose,
        #         left_tcp_to_obj_pos=self.bowl.pose.p - self.left_agent.tcp.pose.p,
        #         right_tcp_to_obj_pos=self.bowl.pose.p - self.right_agent.tcp.pose.p,
        #         obj_to_goal_pos=self.bookcase.pose.p - self.bowl.pose.p,
        #     )
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
        right_tcp = self.right_agent.tcp.pose.p
        fork_pos = self.fork.pose.p
        knife_pos = self.knife.pose.p
        fork_goal_pose = info["fork_goal_pose"]
        knife_goal_pose = info["knife_goal_pose"]

        is_fork_grasped = info["is_fork_grasped"]
        is_knife_grasped = info["is_knife_grasped"]
        fork_placed = info["is_fork_close_to_bowl_right"]
        knife_placed = info["is_knife_close_to_bowl_left"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_knife = reach_reward(left_tcp, knife_pos, scale=5.0)
        r_reach_knife = torch.where(
            is_knife_grasped | knife_placed | success, ones, r_reach_knife
        )
        r_grasp_knife = grasp_reward(
            left_tcp, knife_pos, is_knife_grasped, proximity_scale=5.0
        )
        r_grasp_knife = torch.where(knife_placed | success, ones, r_grasp_knife)
        r_transport_knife = transport_reward(
            knife_pos, knife_goal_pose, is_knife_grasped, scale=5.0
        )
        r_transport_knife = torch.where(
            knife_placed | success, ones, r_transport_knife
        )
        r_place_knife = tanh_reward(info["knife_to_bowl_left_dist"], scale=8.0) * (
            ~is_knife_grasped
        ).float()
        r_place_knife = torch.where(knife_placed | success, ones, r_place_knife)

        r_reach_fork = reach_reward(right_tcp, fork_pos, scale=5.0)
        r_reach_fork = torch.where(
            is_fork_grasped | fork_placed | success, ones, r_reach_fork
        )
        r_grasp_fork = grasp_reward(
            right_tcp, fork_pos, is_fork_grasped, proximity_scale=5.0
        )
        r_grasp_fork = torch.where(fork_placed | success, ones, r_grasp_fork)
        r_transport_fork = transport_reward(
            fork_pos, fork_goal_pose, is_fork_grasped, scale=5.0
        )
        r_transport_fork = torch.where(fork_placed | success, ones, r_transport_fork)
        r_place_fork = tanh_reward(info["fork_to_bowl_right_dist"], scale=8.0) * (
            ~is_fork_grasped
        ).float()
        r_place_fork = torch.where(fork_placed | success, ones, r_place_fork)

        self.reward_tracker.update("reach_knife", r_reach_knife)
        self.reward_tracker.update("grasp_knife", r_grasp_knife)
        self.reward_tracker.update("transport_knife", r_transport_knife)
        self.reward_tracker.update("place_knife", r_place_knife)
        self.reward_tracker.update("reach_fork", r_reach_fork)
        self.reward_tracker.update("grasp_fork", r_grasp_fork)
        self.reward_tracker.update("transport_fork", r_transport_fork)
        self.reward_tracker.update("place_fork", r_place_fork)

        self.reward_tracker.write_to_info(info)
        total_reward = torch.where(success, ones, self.reward_tracker.total())
        self._prioritize_reward_info(info, total_reward)
        return total_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
