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
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l3_robotwin_model,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    grasp_reward,
    reach_reward,
    transport_reward,
)


REWARD_PHASES = ["reach_brush", "grasp_brush", "clean_approach", "clean_wipe"]


@register_env("TwoRobotCleanCupL3-v1", max_episode_steps=100)
class TwoRobotCleanCupEnvL3(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
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
        self.brush_modelname, self.brush_model_id = "117_whiteboard-eraser", 0
        self.cup_modelname, self.cup_model_id = apply_l3_robotwin_model(
            "021_cup",
            model_id=5,
            override_name="021_cup",
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

        # Load brush
        self.brush_pose = sapien.Pose(
            p=[-0.23, -0.35, 0.03],
            q=euler2quat(np.pi / 2, np.pi, 0)
        )
        brush_actor_obj = create_actor(
            scene=self.scene,
            pose=self.brush_pose,
            modelname=self.brush_modelname,
            convex=True,
            model_id=self.brush_model_id,
            scale=(0.1, 0.08, 0.05),
            replace_scale=True,
        )
        self.brush = brush_actor_obj.actor
        self.brush.set_mass(0.5)

        # Load cup
        self.cup_pose = sapien.Pose(
            p=[-0.23, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        cup_actor_obj = create_actor(
            scene=self.scene,
            pose=self.cup_pose,
            modelname=self.cup_modelname,
            convex=True,
            model_id=self.cup_model_id,
            is_static=True,
            replace_scale=True,
            scale=(0.1, 0.1, 0.1),
        )
        self.cup = cup_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "near_point_a"):
                self.near_point_a = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            if not hasattr(self, "near_point_b"):
                self.near_point_b = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.near_point_a[env_idx] = False
            self.near_point_b[env_idx] = False

            if not hasattr(self, "brush_clean_steps"):
                self.brush_clean_steps = torch.zeros(
                    self.num_envs, dtype=torch.int32, device=self.device
                )
            if not hasattr(self, "brush_clean_dist"):
                self.brush_clean_dist = torch.zeros(self.num_envs, device=self.device)
            if not hasattr(self, "brush_last_pos"):
                self.brush_last_pos = self.brush.pose.p.clone()
            self.brush_clean_steps[env_idx] = 0
            self.brush_clean_dist[env_idx] = 0.0
            self.brush_last_pos[env_idx] = self.brush.pose.p[env_idx].clone()

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            xyz = torch.tensor(self.brush_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(0.1, -0.1))
            qs = torch.tensor(self.brush_pose.q).repeat(b, 1)
            self.brush.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.cup_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.cup_zs[env_idx]
            qs = torch.tensor(self.cup_pose.q).repeat(b, 1)
            self.cup.set_pose(Pose.create_from_pq(xyz, qs))
            self._init_clean_points(env_idx)

    def _init_clean_points(self, env_idx: torch.Tensor):
        if not hasattr(self, "clean_point_a"):
            self.clean_point_a = torch.zeros((self.num_envs, 3), device=self.device)
            self.clean_point_b = torch.zeros((self.num_envs, 3), device=self.device)
        obb = get_actor_obb(self.brush)
        approaching = np.array([0, 0, -1])
        target_closing = self.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.025,
        )
        grasp_pose = self.left_agent.build_grasp_pose(approaching, grasp_info["closing"], self.brush.pose.sp.p)
        goal_pose = sapien.Pose(self.cup.pose.sp.p, grasp_pose.q) * sapien.Pose([0, 0, -0.05])
        point_a_pose = goal_pose * sapien.Pose([-0.1, 0.0, -0.08], euler2quat(0, -np.pi / 2, 0))
        point_b_pose = goal_pose * sapien.Pose([0.1, 0.0, -0.08], euler2quat(0, -np.pi / 2, 0))
        point_a = torch.tensor(point_a_pose.p, device=self.device)
        point_b = torch.tensor(point_b_pose.p, device=self.device)
        self.clean_point_a[env_idx] = point_a
        self.clean_point_b[env_idx] = point_b

    def _after_reconfigure(self, options: dict):

        self.brush_zs = []
        collision_mesh = self.brush.get_first_collision_mesh()
        self.brush_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.brush_zs = common.to_tensor(self.brush_zs, device=self.device)

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
        point_a = self.clean_point_a
        point_b = self.clean_point_b
        left_grasped = self.left_agent.is_grasping(self.brush)
        right_grasped = self.right_agent.is_grasping(self.brush)
        is_grasped = left_grasped | right_grasped

        active_tcp_pos = torch.where(
            right_grasped[:, None],
            self.right_agent.tcp.pose.p,
            self.left_agent.tcp.pose.p,
        )
        dist_a = torch.linalg.norm(active_tcp_pos - point_a, axis=1)
        dist_b = torch.linalg.norm(active_tcp_pos - point_b, axis=1)
        near_a = (dist_a <= 0.05) & is_grasped
        near_b = (dist_b <= 0.05) & is_grasped
        is_cleaning = is_grasped & (torch.minimum(dist_a, dist_b) <= 0.08)
        self.near_point_a = self.near_point_a | near_a
        self.near_point_b = self.near_point_b | near_b

        step_dist = torch.linalg.norm(self.brush.pose.p - self.brush_last_pos, axis=1)
        self.brush_clean_dist += step_dist * is_cleaning.float()
        self.brush_clean_steps += is_cleaning.int()
        self.brush_last_pos = self.brush.pose.p.clone()

        is_obj_placed = near_a | near_b
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        wipe_progress = 0.5 * self.near_point_a.float() + 0.5 * self.near_point_b.float()

        result = dict(
            is_grasped=is_grasped,
            left_grasped=left_grasped,
            right_grasped=right_grasped,
            point_a=point_a,
            point_b=point_b,
            active_tcp_pos=active_tcp_pos,
            dist_a=dist_a,
            dist_b=dist_b,
            near_point_a=self.near_point_a,
            near_point_b=self.near_point_b,
            is_cleaning=is_cleaning,
            brush_clean_steps=self.brush_clean_steps,
            brush_clean_dist=self.brush_clean_dist,
            wipe_progress=wipe_progress,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=is_grasped,
            success=self.near_point_a & self.near_point_b,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.brush.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.brush.pose.raw_pose,
                left_tcp_to_obj_pos=self.brush.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.brush.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.brush.pose.p - self.cup.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        brush_pos = self.brush.pose.p
        left_grasped = self.left_agent.is_grasping(self.brush)
        right_grasped = self.right_agent.is_grasping(self.brush)
        is_grasped = info["is_grasped"]
        visited_any = self.near_point_a | self.near_point_b
        success = info["success"]

        active_tcp_pos = torch.where(
            right_grasped[:, None],
            right_tcp,
            left_tcp,
        )
        target_point = torch.where(
            self.near_point_a[:, None],
            self.clean_point_b,
            self.clean_point_a,
        )
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(left_tcp, brush_pos, scale=5.0)
        r_reach = torch.where(is_grasped | visited_any | success, ones, r_reach)
        self.reward_tracker.update("reach_brush", r_reach)

        r_grasp = grasp_reward(
            left_tcp, brush_pos, left_grasped, proximity_scale=5.0
        )
        r_grasp = torch.where(visited_any | success, ones, r_grasp)
        self.reward_tracker.update("grasp_brush", r_grasp)

        r_approach = transport_reward(
            active_tcp_pos, target_point, is_grasped, scale=10.0
        )
        r_approach = torch.where(visited_any | success, ones, r_approach)
        self.reward_tracker.update("clean_approach", r_approach)

        r_wipe = 0.5 * self.near_point_a.float() + 0.5 * self.near_point_b.float()
        r_wipe = torch.where(success, ones, r_wipe)
        self.reward_tracker.update("clean_wipe", r_wipe)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
