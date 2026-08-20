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
    apply_l2_robotwin_model,
    apply_l3_robotwin_model,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    grasp_reward,
    reach_reward,
    transport_reward,
    wipe_progress_reward,
)


REWARD_PHASES = ["reach_grind", "grasp_grind", "approach_bowl", "grind_motion"]


@register_env("TwoRobotGrindFood-v1", max_episode_steps=100)
class TwoRobotGrindFoodEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.15
    grind_motion_thresh = 0.003
    grind_steps_required = 1
    grind_motion_target = 0.01
    grind_engage_thresh = 0.08

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
        self.grind_modelname, self.grind_model_id = apply_l2_robotwin_model(
            "018_microphone",
            model_id=4,
            override_name="018_microphone",
            override_id=1,
        )
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

        # Load grind
        self.grind_pose = sapien.Pose(
            p=[0.05, -0.25, 0.13],
            q=euler2quat(0, 0, -np.pi / 2)
        )
        grind_actor_obj = create_actor(
            scene=self.scene,
            pose=self.grind_pose,
            modelname=self.grind_modelname,
            convex=True,
            model_id=self.grind_model_id,
            scale=(0.1, 0.1, 0.1),
            replace_scale=True,
        )
        self.grind = grind_actor_obj.actor
        grind_actor_obj.set_mass(0.5)

        # Load block
        self.block_pose = sapien.Pose(
            p=[0.09, -0.24, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        block_actor_obj = create_actor(
            scene=self.scene,
            pose=self.block_pose,
            modelname="004_fluted-block",
            convex=True,
            model_id=1,
            is_static=True,
            scale=(1.05, 1.05, 1.05),
            # replace_scale=True,
        )
        self.block = block_actor_obj.actor

        # Load bowl
        self.bowl_pose = sapien.Pose(
            p=[0.05, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        bowl_actor_obj = create_actor(
            scene=self.scene,
            pose=self.bowl_pose,
            modelname=self.bowl_modelname,
            convex=True,
            model_id=self.bowl_model_id,
            is_static=True,
            # replace_scale=True,
            scale=(1.3, 1.3, 1.3),
        )
        self.bowl = bowl_actor_obj.actor

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

            xyz = torch.tensor(self.grind_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.0
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0))
            qs = torch.tensor(self.grind_pose.q).repeat(b, 1)
            self.grind.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.block_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.0
            qs = torch.tensor(self.block_pose.q).repeat(b, 1)
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0))
            self.block.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.bowl_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.bowl_zs[env_idx]
            qs = torch.tensor(self.bowl_pose.q).repeat(b, 1)
            self.bowl.set_pose(Pose.create_from_pq(xyz, qs))

            if not hasattr(self, "grind_ever_done"):
                self.grind_ever_done = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            if not hasattr(self, "prev_grind_pos"):
                self.prev_grind_pos = self.grind.pose.p.clone()
            if not hasattr(self, "grind_motion_dist"):
                self.grind_motion_dist = torch.zeros(self.num_envs, device=self.device)
            if not hasattr(self, "grind_motion_steps"):
                self.grind_motion_steps = torch.zeros(
                    self.num_envs, dtype=torch.int32, device=self.device
                )
            self.grind_ever_done[env_idx] = False
            self.prev_grind_pos[env_idx] = self.grind.pose.p[env_idx].clone()
            self.grind_motion_dist[env_idx] = 0.0
            self.grind_motion_steps[env_idx] = 0

    def _after_reconfigure(self, options: dict):

        self.grind_zs = []
        collision_mesh = self.grind.get_first_collision_mesh()
        self.grind_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.grind_zs = common.to_tensor(self.grind_zs, device=self.device)

        self.bowl_zs = []
        collision_mesh = self.bowl.get_first_collision_mesh()
        self.bowl_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.bowl_zs = common.to_tensor(self.bowl_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        obj_to_goal_pos = self.grind.pose.p - self.bowl.pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)
        is_obj_placed = obj_to_goal_dist <= self.goal_thresh
        left_tcp_to_grind = torch.linalg.norm(
            self.left_agent.tcp.pose.p - self.grind.pose.p, axis=1
        )
        right_tcp_to_grind = torch.linalg.norm(
            self.right_agent.tcp.pose.p - self.grind.pose.p, axis=1
        )
        left_grasped = self.left_agent.is_grasping(self.grind)
        right_grasped = self.right_agent.is_grasping(self.grind)
        is_grasped = left_grasped | right_grasped
        is_engaged = is_grasped | (
            torch.minimum(left_tcp_to_grind, right_tcp_to_grind)
            <= self.grind_engage_thresh
        )

        step_motion = torch.linalg.norm(self.grind.pose.p - self.prev_grind_pos, axis=1)
        currently_grinding = is_engaged & is_obj_placed & (
            step_motion >= self.grind_motion_thresh
        )
        self.grind_motion_dist += step_motion * currently_grinding.float()
        self.grind_motion_steps += currently_grinding.int()
        self.grind_ever_done = self.grind_ever_done | (
            (self.grind_motion_steps >= self.grind_steps_required)
            & (self.grind_motion_dist >= self.grind_motion_target)
        )
        self.prev_grind_pos = self.grind.pose.p.clone()

        is_robot_static = torch.logical_and(
            self.left_agent.is_static(0.2), self.right_agent.is_static(0.2)
        )
        result = dict(
            is_grasped=is_grasped,
            is_engaged=is_engaged,
            left_grasped=left_grasped,
            right_grasped=right_grasped,
            left_tcp_to_grind=left_tcp_to_grind,
            right_tcp_to_grind=right_tcp_to_grind,
            obj_to_goal_pos=obj_to_goal_pos,
            obj_to_goal_dist=obj_to_goal_dist,
            is_obj_placed=is_obj_placed,
            currently_grinding=currently_grinding,
            grind_ever_done=self.grind_ever_done,
            grind_motion_steps=self.grind_motion_steps,
            grind_motion_dist=self.grind_motion_dist,
            grind_step_motion=step_motion,
            is_robot_static=is_robot_static,
            is_grasping=is_grasped,
            success=self.grind_ever_done,
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
            currently_grinding=info["currently_grinding"],
            grind_ever_done=info["grind_ever_done"],
            grind_motion_steps=info["grind_motion_steps"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.grind.pose.raw_pose,
                left_tcp_to_obj_pos=self.grind.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.grind.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.grind.pose.p - self.bowl.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        grind_pos = self.grind.pose.p
        bowl_pos = self.bowl.pose.p
        is_grasped = info["is_engaged"]
        ever_done = info["grind_ever_done"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)
        left_dist = torch.linalg.norm(grind_pos - left_tcp, dim=-1)
        right_dist = torch.linalg.norm(grind_pos - right_tcp, dim=-1)
        tcp_pos = torch.where((left_dist <= right_dist)[:, None], left_tcp, right_tcp)

        r_reach = reach_reward(tcp_pos, grind_pos, scale=5.0)
        r_reach = torch.where(is_grasped | ever_done | success, ones, r_reach)
        self.reward_tracker.update("reach_grind", r_reach)

        r_grasp = grasp_reward(
            tcp_pos, grind_pos, is_grasped, proximity_scale=5.0
        )
        r_grasp = torch.where(ever_done | success, ones, r_grasp)
        self.reward_tracker.update("grasp_grind", r_grasp)

        r_approach = transport_reward(grind_pos, bowl_pos, is_grasped, scale=5.0)
        r_approach = torch.where(ever_done | success, ones, r_approach)
        self.reward_tracker.update("approach_bowl", r_approach)

        r_grind = wipe_progress_reward(
            steps=self.grind_motion_steps,
            target_steps=self.grind_steps_required,
            dist=self.grind_motion_dist,
            target_dist=self.grind_motion_target,
        )
        r_grind = torch.where(success, ones, r_grind)
        self.reward_tracker.update("grind_motion", r_grind)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
