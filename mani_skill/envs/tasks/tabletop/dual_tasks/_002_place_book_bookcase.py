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
    apply_l2_robotwin_config,
    apply_l3_robotwin_config,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    reach_reward,
    grasp_reward,
    transport_reward,
)


REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotPlaceBookBookcase-v1", max_episode_steps=100)
class TwoRobotPlaceBookBookcaseEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.08

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
        # self.all_model_ids = np.array(["005_tomato_soup_can"])
        self.book_mass = 0.5
        (
            self.bookcase_modelname,
            self.bookcase_model_id,
            self.bookcase_scale,
            self.bookcase_replace_scale,
        ) = apply_l3_robotwin_config(
            "014_bookcase",
            model_id=2,
            base_scale=(0.18, 0.18, 0.18),
            base_replace_scale=True,
            override_name="014_bookcase",
            override_id=1,
            override_scale=(0.18, 0.18, 0.18),
            override_replace_scale=True,
        )
        (
            self.book_modelname,
            self.book_model_id,
            self.book_scale,
            self.book_replace_scale,
        ) = apply_l2_robotwin_config(
            "043_book",
            model_id=0,
            base_scale=(0.8, 0.8, 0.8),
            base_replace_scale=False,
            override_name="043_book",
            override_id=1,
            override_scale=(0.4, 0.4, 0.4),
            override_replace_scale=False,
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

        # Load bookcase
        self.bookcase_pose = sapien.Pose(
            p=[0.1, 0.0, -0.08],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        bookcase_actor_obj = create_actor(
            scene=self.scene,
            pose=self.bookcase_pose,
            modelname=self.bookcase_modelname,
            convex=True,
            model_id=self.bookcase_model_id,
            scale=self.bookcase_scale,
            replace_scale=self.bookcase_replace_scale,
            is_static=True
        )
        self.bookcase = bookcase_actor_obj.actor

        # Load book
        self.book_pose = sapien.Pose(
            p=[0.0, -0.5, 0.0],
            q=euler2quat(0, 0, 0)
        )
        book_actor_obj = create_actor(
            scene=self.scene,
            pose=self.book_pose,
            modelname=self.book_modelname,
            convex=True,
            model_id=self.book_model_id,
            scale=self.book_scale,
            replace_scale=self.book_replace_scale,
            mass=self.book_mass, 
        )
        self.book = book_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.tensor(self.book_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz[:, 2] = self.book_zs[env_idx]
            xyz = apply_l1_offset_xy(xyz, offset=(-0.15, 0.1))
            qs = torch.tensor(self.book_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.book.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            xyz = torch.tensor(self.bookcase_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz[:, 2] = self.bookcase_zs[env_idx] + xyz[:, 2]
            qs = torch.tensor(self.bookcase_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.bookcase.set_pose(Pose.create_from_pq(xyz, qs))
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

    def _after_reconfigure(self, options: dict):
        self.book_zs = []
        collision_mesh = self.book.get_first_collision_mesh()
        self.book_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.book_zs = common.to_tensor(self.book_zs, device=self.device)

        self.bookcase_zs = []
        collision_mesh = self.bookcase.get_first_collision_mesh()
        self.bookcase_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.bookcase_zs = common.to_tensor(self.bookcase_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        if not hasattr(self, "_geom_center_fallback_logged"):
            self._geom_center_fallback_logged = set()
        try:
            return geom_center_from_local_mesh(obj, self.device)
        except Exception as exc:
            obj_name = getattr(obj, "name", obj.__class__.__name__)
            if obj_name not in self._geom_center_fallback_logged:
                print(
                    f"[{self.__class__.__name__}] geom center fallback to pose.p for {obj_name}: {exc}"
                )
                self._geom_center_fallback_logged.add(obj_name)
            return obj.pose.p.clone()

    def evaluate(self):
        obj_to_goal_pos = self.bookcase.pose.p - self.book.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos[:, :2], axis=1) <= self.goal_thresh
        is_grasped = self.left_agent.is_grasping(self.book) | self.right_agent.is_grasping(self.book)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        result = dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=is_grasped,
            success=is_obj_placed,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.bookcase.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.book.pose.raw_pose,
                left_tcp_to_obj_pos=self.book.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.book.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.bookcase.pose.p - self.book.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        book_pos = self._get_actor_geom_center(self.book)
        goal_pos = self._get_actor_geom_center(self.bookcase)
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, book_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, book_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_obj_placed | success, ones, r_grasp)
        r_transport = transport_reward(book_pos, goal_pos, is_grasped, scale=5.0)
        r_transport = torch.where(is_obj_placed | success, ones, r_transport)
        r_place = is_obj_placed.float()
        r_place = torch.where(success, ones, r_place)

        self.reward_tracker.update("reach", r_reach)
        self.reward_tracker.update("grasp", r_grasp)
        self.reward_tracker.update("transport", r_transport)
        self.reward_tracker.update("place", r_place)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
