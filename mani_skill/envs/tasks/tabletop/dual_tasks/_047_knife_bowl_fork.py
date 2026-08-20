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
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_ycb_model_id,
    apply_l3_robotwin_model,
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
    "reach_fork",
    "grasp_fork",
    "transport_fork",
    "place_fork",
    "reach_knife",
    "grasp_knife",
    "transport_knife",
    "place_knife",
]


@register_env("TwoRobotKnifeBowlFork-v1", max_episode_steps=100)
class TwoRobotKnifeBowlForkEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.035

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
        # Per-axis scale for YCB utensils. Increase Z to make them thicker.
        self.fork_scale_xyz = [1.0, 1.0, 1.3]
        self.knife_scale_xyz = [1.0, 1.0, 1.3]
        self.bowl_modelname, self.bowl_model_id = apply_l3_robotwin_model(
            "002_bowl",
            model_id=2,
            override_name="002_bowl",
            override_id=3,
        )
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
            scales=self.fork_scale_xyz,
        )
        self.fork_pose = sapien.Pose(p=[-0.2, -0.2, 0.0], q=euler2quat(0, 0, 0))
        appliance_builder.initial_pose = self.fork_pose
        self.fork = appliance_builder.build(name=f'{appliance_model_id}')
        self.fork.set_mass(0.5)
        
        # Load knife
        appliance_model_id = apply_l2_ycb_model_id(
            "032_knife", override_id="033_spatula"
        )
        appliance_builder = actors.get_actor_builder(
            self.scene,
            id=f'ycb:{appliance_model_id}',
            scales=self.knife_scale_xyz,
        )
        self.knife_pose = sapien.Pose(p=[-0.2, 0.2, 0.0], q=euler2quat(0, 0, 0))
        appliance_builder.initial_pose = self.knife_pose
        self.knife = appliance_builder.build(name=f'{appliance_model_id}')
        self.knife.set_mass(0.5)
        
        # Load bowl (robotwin asset with L3 id replacement)
        self.bowl_pose = sapien.Pose(p=[0.0, 0.0, -0.02], q=euler2quat(np.pi / 2, 0, 0))
        bowl_actor_obj = create_actor(
            scene=self.scene,
            pose=self.bowl_pose,
            modelname=self.bowl_modelname,
            convex=True,
            is_static=True,
            model_id=self.bowl_model_id,
        )
        self.bowl = bowl_actor_obj.actor
        
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
            knife_offset = (0.0, -0.1)
            fork_offset = (0.0, 0.1)
            for object, init_pose in zip(self.objects, self._object_initial_poses):
                xyz = torch.tensor(
                    init_pose.p, device=self.device, dtype=torch.float32
                ).repeat(b, 1)
                xyz[:, :2] += torch.rand((b, 2)) * 0.02
                xyz[:, 2] = self.objects_zs[object][env_idx]
                if object is self.knife:
                    xyz = apply_l1_offset_xy(xyz, offset=knife_offset)
                elif object is self.fork:
                    xyz = apply_l1_offset_xy(xyz, offset=fork_offset)
                qs = torch.tensor(init_pose.q).repeat(b, 1)
                object.set_pose(Pose.create_from_pq(p=xyz, q=qs))
                
            # xyz = torch.tensor(self.bowl.pose.p)
            # xyz[:, :2] += torch.rand((b, 2)) * 0.02
            # xyz[:, 2] = self.bowl_zs[env_idx]
            # qs = torch.tensor(self.bowl.pose.q)
            # self.bowl.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            
            # xyz = torch.tensor(self.fork.pose.p)
            # xyz[:, :2] += torch.rand((b, 2)) * 0.02
            # xyz[:, 2] = self.fork_zs[env_idx]
            # qs = torch.tensor(self.fork.pose.q)
            # self.fork.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            
            # xyz = torch.tensor(self.knife.pose.p)
            # xyz[:, :2] += torch.rand((b, 2)) * 0.02
            # xyz[:, 2] = self.knife_zs[env_idx]
            # qs = torch.tensor(self.knife.pose.q)
            # self.knife.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def _after_reconfigure(self, options: dict):
        # self.book_zs = []
        # collision_mesh = self.book.get_first_collision_mesh()
        # self.book_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        # self.book_zs = common.to_tensor(self.book_zs, device=self.device)

        # self.bookcase_zs = []
        # collision_mesh = self.bookcase.get_first_collision_mesh()
        # self.bookcase_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        # self.bookcase_zs = common.to_tensor(self.bookcase_zs, device=self.device)
        self.objects_zs = {}
        for object in self.objects:
            self.objects_zs[object] = []
            collision_mesh = object.get_first_collision_mesh()
            self.objects_zs[object].append(-collision_mesh.bounding_box.bounds[0, 2])
            self.objects_zs[object] = common.to_tensor(
                self.objects_zs[object], device=self.device
            )
            

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

        is_fork_grasped = self.left_agent.is_grasping(self.fork)
        is_knife_grasped = self.right_agent.is_grasping(self.knife)

        fork_to_goal_pos = fork_pose - fork_goal_pose
        knife_to_goal_pos = knife_pose - knife_goal_pose
        fork_to_bowl_left_dist = torch.linalg.norm(fork_to_goal_pos, dim=-1)
        knife_to_bowl_right_dist = torch.linalg.norm(knife_to_goal_pos, dim=-1)
        is_fork_close_to_bowl_left = fork_to_bowl_left_dist <= self.goal_thresh
        is_knife_close_to_bowl_right = knife_to_bowl_right_dist <= self.goal_thresh

        is_robot_static = torch.logical_and(
            self.left_agent.is_static(0.2),
            self.right_agent.is_static(0.2),
        )

        result = dict(
            is_fork_grasped=is_fork_grasped,
            is_knife_grasped=is_knife_grasped,
            fork_goal_pose=fork_goal_pose,
            knife_goal_pose=knife_goal_pose,
            fork_to_goal_pos=fork_to_goal_pos,
            knife_to_goal_pos=knife_to_goal_pos,
            is_fork_close_to_bowl_left=is_fork_close_to_bowl_left,
            is_knife_close_to_bowl_right=is_knife_close_to_bowl_right,
            fork_to_bowl_left_dist=fork_to_bowl_left_dist,
            knife_to_bowl_right_dist=knife_to_bowl_right_dist,
            is_robot_static=is_robot_static,
            success=torch.logical_and(
                is_fork_close_to_bowl_left, is_knife_close_to_bowl_right
            ),
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_fork_goal_pos(self) -> torch.Tensor:
        offset = np.array([0.0, -0.2, -0.02], dtype=np.float32)
        if is_lr_mirror_enabled():
            offset = mirror_xyz(offset)
        return self.bowl.pose.p + torch.tensor(
            offset, device=self.device, dtype=self.bowl.pose.p.dtype
        ).unsqueeze(0).repeat(self.num_envs, 1)

    def _get_knife_goal_pos(self) -> torch.Tensor:
        offset = np.array([0.0, 0.2, -0.02], dtype=np.float32)
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
            is_fork_close_to_bowl_left=info["is_fork_close_to_bowl_left"],
            is_knife_close_to_bowl_right=info["is_knife_close_to_bowl_right"],
            fork_to_bowl_left_dist=info["fork_to_bowl_left_dist"],
            knife_to_bowl_right_dist=info["knife_to_bowl_right_dist"],
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
        fork_placed = info["is_fork_close_to_bowl_left"]
        knife_placed = info["is_knife_close_to_bowl_right"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_fork = reach_reward(left_tcp, fork_pos, scale=5.0)
        r_reach_fork = torch.where(
            is_fork_grasped | fork_placed | success, ones, r_reach_fork
        )
        r_grasp_fork = grasp_reward(
            left_tcp, fork_pos, is_fork_grasped, proximity_scale=5.0
        )
        r_grasp_fork = torch.where(fork_placed | success, ones, r_grasp_fork)
        r_transport_fork = transport_reward(
            fork_pos, fork_goal_pose, is_fork_grasped, scale=5.0
        )
        r_transport_fork = torch.where(fork_placed | success, ones, r_transport_fork)
        r_place_fork = tanh_reward(info["fork_to_bowl_left_dist"], scale=8.0) * (
            ~is_fork_grasped
        ).float()
        r_place_fork = torch.where(fork_placed | success, ones, r_place_fork)

        r_reach_knife = reach_reward(right_tcp, knife_pos, scale=5.0)
        r_reach_knife = torch.where(
            is_knife_grasped | knife_placed | success, ones, r_reach_knife
        )
        r_grasp_knife = grasp_reward(
            right_tcp, knife_pos, is_knife_grasped, proximity_scale=5.0
        )
        r_grasp_knife = torch.where(knife_placed | success, ones, r_grasp_knife)
        r_transport_knife = transport_reward(
            knife_pos, knife_goal_pose, is_knife_grasped, scale=5.0
        )
        r_transport_knife = torch.where(
            knife_placed | success, ones, r_transport_knife
        )
        r_place_knife = tanh_reward(info["knife_to_bowl_right_dist"], scale=8.0) * (
            ~is_knife_grasped
        ).float()
        r_place_knife = torch.where(knife_placed | success, ones, r_place_knife)

        self.reward_tracker.update("reach_fork", r_reach_fork)
        self.reward_tracker.update("grasp_fork", r_grasp_fork)
        self.reward_tracker.update("transport_fork", r_transport_fork)
        self.reward_tracker.update("place_fork", r_place_fork)
        self.reward_tracker.update("reach_knife", r_reach_knife)
        self.reward_tracker.update("grasp_knife", r_grasp_knife)
        self.reward_tracker.update("transport_knife", r_transport_knife)
        self.reward_tracker.update("place_knife", r_place_knife)

        self.reward_tracker.write_to_info(info)
        total_reward = torch.where(success, ones, self.reward_tracker.total())
        self._prioritize_reward_info(info, total_reward)
        return total_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
