import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, Tuple

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_robotwin_config,
    apply_l3_robotwin_model,
    is_lr_mirror_enabled,
    mirror_xyz,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    above_reward,
    geom_center_from_local_mesh,
    grasp_reward,
    normalized_progress,
    reach_reward,
)

REWARD_PHASES = ["reach_cup2", "grasp_cup2", "pour_approach"]

@register_env("TwoRobotPourCupL3-v1", max_episode_steps=200)
class TwoRobotPourCupEnvL3(BaseEnv):
    """
    **Task Description:**

    The task consists of multiple phases:
    1. Grasp the kettle
    2. Move kettle above the cup
    3. Tilt kettle to perform pouring motion
    4. Return kettle to original position
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    # Task-specific thresholds
    goal_thresh = 0.05  # Threshold for checking if kettle is above cup
    return_thresh = 0.15  # Threshold for checking if kettle returned to original position

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
        self.cup1_modelname, self.cup1_model_id = apply_l3_robotwin_model(
            "021_cup",
            model_id=4,
            override_name="021_cup",
            override_id=10,
        )
        # L2-only size tuning for cup2. Adjust this factor as needed.
        self.cup2_l2_scale_factor = 1.0
        (
            self.cup2_modelname,
            self.cup2_model_id,
            self.cup2_scale,
            self.cup2_replace_scale,
        ) = apply_l2_robotwin_config(
            "021_cup",
            model_id=6,
            base_scale=None,
            base_replace_scale=False,
            override_name="021_cup",
            override_id=1,
            override_scale=(
                self.cup2_l2_scale_factor,
                self.cup2_l2_scale_factor,
                self.cup2_l2_scale_factor,
            ),
            override_replace_scale=False,
        )
        self.teanet_modelname, self.teanet_model_id = apply_l3_robotwin_model(
            "053_teanet",
            model_id=5,
            override_name="053_teanet",
            override_id=4,
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

        cup1_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(np.pi / 2, 0, 0))
        cup1_actor_obj = create_actor(
            self.scene,
            pose=cup1_pose,
            modelname=self.cup1_modelname,
            convex=True,
            model_id=self.cup1_model_id,
            is_static=True,
            _idx_if_repeat=0,
        )
        cup1_actor_obj.set_mass(1)
        self.cup1 = cup1_actor_obj.actor

        # Load Cup (target for pouring)
        cup2_pose = sapien.Pose(p=[0.0, -0.2, 0.0], q=euler2quat(np.pi / 2, 0, 0))
        cup2_actor_obj = create_actor(
            self.scene,
            pose=cup2_pose,
            modelname=self.cup2_modelname,
            convex=True,
            model_id=self.cup2_model_id,
            scale=self.cup2_scale,
            replace_scale=self.cup2_replace_scale,
            _idx_if_repeat=1,
        )
        cup2_actor_obj.set_mass(1.0)
        self.cup2 = cup2_actor_obj.actor

        # Load filter and keep it above cup1 in +y direction, matching _045 placement.
        self.teanet_pose = sapien.Pose(
            p=[cup1_pose.p[0], cup1_pose.p[1] + 0.03, -0.09],
            q=euler2quat(np.pi / 2, 0, 0),
        )
        teanet_actor_obj = create_actor(
            self.scene,
            pose=self.teanet_pose,
            modelname=self.teanet_modelname,
            convex=True,
            model_id=self.teanet_model_id,
            is_static=True,
            scale=(1.2, 1.2, 1.2),
        )
        self.teanet = teanet_actor_obj.actor
        
        self.objects = [self.cup1, self.cup2]
        self._object_initial_poses = [cup1_pose, cup2_pose]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "cup2_ever_poured"):
                self.cup2_ever_poured = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.cup2_ever_poured[env_idx] = False

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    REWARD_PHASES, self.num_envs, self.device
                )
            self.reward_tracker.reset(env_idx)

            if not hasattr(self, "cup2_init_rot"):
                self.cup2_init_rot = torch.eye(
                    3, dtype=torch.float32, device=self.device
                ).unsqueeze(0).repeat(self.num_envs, 1, 1)
            if not hasattr(self, "cup2_initial_pos"):
                self.cup2_initial_pos = torch.zeros(
                    (self.num_envs, 3), dtype=torch.float32, device=self.device
                )

            for object, init_pose in zip(self.objects, self._object_initial_poses):
                xyz = torch.tensor(object.pose.p)
                
                xyz[:, :2] += torch.rand((b, 2)) * 0.02
                xyz[:, 2] = self.objects_zs[object][env_idx]
                xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
                qs = torch.tensor(init_pose.q).repeat(b, 1)
                object.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            teanet_xyz = self.cup1.pose.p.clone()
            teanet_xyz[:, 1] += 0.04
            teanet_xyz[:, 2] = self.teanet_zs[env_idx]
            teanet_z_rot = 0.0
            if is_lr_mirror_enabled():
                teanet_z_rot = np.pi - teanet_z_rot
            teanet_q = torch.tensor(
                euler2quat(np.pi / 2, 0, teanet_z_rot),
                device=self.device,
            ).repeat(b, 1)
            self.teanet.set_pose(Pose.create_from_pq(p=teanet_xyz, q=teanet_q))
            self.cup2_initial_pos[env_idx] = self.cup2.pose.p[env_idx]
            self.cup2_init_rot[env_idx] = self.cup2.pose.to_transformation_matrix()[
                env_idx, :3, :3
            ]

    def _after_reconfigure(self, options: dict):
        self.objects_zs = {}
        for object in self.objects:
            self.objects_zs[object] = []
            collision_mesh = object.get_first_collision_mesh()
            self.objects_zs[object].append(-collision_mesh.bounding_box.bounds[0, 2])
            self.objects_zs[object] = common.to_tensor(
                self.objects_zs[object], device=self.device
            )
        self.teanet_zs = []
        collision_mesh = self.teanet.get_first_collision_mesh()
        self.teanet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.teanet_zs = common.to_tensor(self.teanet_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        is_grasped = self.left_agent.is_grasping(self.cup2)
        target_center = self._get_pour_target_pos()
        cup2_center = self._get_actor_geom_center(self.cup2)
        cup2_to_target = cup2_center - target_center
        horizontal_dist = torch.linalg.norm(cup2_to_target[..., :2], dim=-1)
        cup2_height_above_target = cup2_to_target[..., 2]
        is_above_cup = torch.logical_and(
            horizontal_dist <= self.goal_thresh,
            cup2_height_above_target > 0.05,
        )

        cup2_rot = self.cup2.pose.to_transformation_matrix()[:, :3, :3]
        rot_rel = torch.matmul(self.cup2_init_rot.transpose(1, 2), cup2_rot)
        trace = rot_rel[:, 0, 0] + rot_rel[:, 1, 1] + rot_rel[:, 2, 2]
        cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
        tilt_angle = torch.arccos(cos_angle)
        is_tilted = tilt_angle >= np.deg2rad(15.0)

        currently_pouring = is_grasped & is_above_cup
        self.cup2_ever_poured = self.cup2_ever_poured | currently_pouring

        cup2_to_initial = cup2_center - self.cup2_initial_pos
        return_dist = torch.linalg.norm(cup2_to_initial, dim=-1)
        is_returned = return_dist <= self.return_thresh

        is_robot_static = self.left_agent.is_static(0.2)
        success = self.cup2_ever_poured

        result = dict(
            is_grasped=is_grasped,
            is_above_cup=is_above_cup,
            is_tilted=is_tilted,
            cup2_ever_poured=self.cup2_ever_poured,
            currently_pouring=currently_pouring,
            is_returned=is_returned,
            is_robot_static=is_robot_static,
            kettle_to_cup_dist=horizontal_dist,
            return_dist=return_dist,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        return geom_center_from_local_mesh(obj, self.device)

    def _get_pour_target_pos(self) -> torch.Tensor:
        target_pos = self._get_actor_geom_center(self.cup1).clone()
        offset = mirror_xyz(np.array([0.0, -0.1, 0.0], dtype=np.float32))
        offset = torch.tensor(offset, device=self.device, dtype=target_pos.dtype)
        target_pos = target_pos + offset.unsqueeze(0).repeat(self.num_envs, 1)
        return target_pos

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.left_agent.tcp.pose.raw_pose,
            cup_pos=self.cup1.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                cup2_pose=self.cup2.pose.raw_pose,
                tcp_to_cup2_pos=self.cup2.pose.p - self.left_agent.tcp.pose.p,
                cup_pose=self.cup1.pose.raw_pose,
                tcp_to_cup_pos=self.cup1.pose.p - self.left_agent.tcp.pose.p,
                cup2_to_cup1_pos=self.cup1.pose.p - self.cup2.pose.p,
                cup2_to_initial_pos=self.cup2.pose.p - self.cup2_initial_pos,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        cup2_center = self._get_actor_geom_center(self.cup2)
        target_center = self._get_pour_target_pos()

        is_grasped = info["is_grasped"]
        is_above_cup = info["is_above_cup"]
        is_tilted = info["is_tilted"]
        currently_pouring = info["currently_pouring"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, cup2_center, scale=5.0)
        r_reach = torch.where(is_grasped | currently_pouring | success, ones, r_reach)
        self.reward_tracker.update("reach_cup2", r_reach)

        r_grasp = grasp_reward(
            tcp_pos, cup2_center, is_grasped, proximity_scale=5.0
        )
        r_grasp = torch.where(is_above_cup | currently_pouring | success, ones, r_grasp)
        self.reward_tracker.update("grasp_cup2", r_grasp)

        r_approach = above_reward(
            cup2_center,
            target_center,
            min_height=0.05,
            is_grasping=is_grasped,
            h_scale=8.0,
        )
        r_approach = torch.where(is_above_cup | currently_pouring | success, ones, r_approach)
        self.reward_tracker.update("pour_approach", r_approach)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
