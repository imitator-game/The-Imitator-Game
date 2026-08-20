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
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    reach_reward,
    grasp_reward,
    transport_reward,
    world_aabb_from_local_mesh,
)


REWARD_PHASES = [
    "clear_reach",
    "clear_grasp",
    "clear_transport",
    "clear_place",
    "book_reach",
    "book_grasp",
    "book_transport",
    "book_place",
]


@register_env("TwoRobotPlaceBookBookcaseL3-v1", max_episode_steps=100)
class TwoRobotPlaceBookBookcaseEnvL3(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.08
    container_z_tolerance = 0.1
    video_info_whitelist = {
        "reward",
        "success",
        "is_grasped",
        "is_obj_placed",
        "is_book2_grasped",
        "is_book2_cleared",
        "peak_r_clear_reach",
        "peak_r_clear_grasp",
        "peak_r_clear_transport",
        "peak_r_clear_place",
        "peak_r_book_reach",
        "peak_r_book_grasp",
        "peak_r_book_transport",
        "peak_r_book_place",
    }

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
        # No level-switch logic: fixed assets for this task.
        self.bookcase_modelname = "042_wooden_box"
        self.bookcase_model_id = get_model_id(self.bookcase_modelname, model_id=0)
        self.bookcase_scale = (1.5, 1.5, 1.5)
        self.bookcase_replace_scale = False
        self.book_modelname = "043_book"
        self.book_model_id = get_model_id(self.book_modelname, model_id=0)
        self.book_scale = (0.8, 0.8, 0.8)
        self.book_mass = 0.5
        self.book_replace_scale = False
        # Extra book (distractor) placed horizontally on top of the box.
        self.book2_modelname = "043_book"
        self.book2_model_id = get_model_id(self.book2_modelname, model_id=1)
        self.book2_scale = (1.2, 0.4, 1.2)  # tunable
        self.book2_mass = 0.5
        # self.book2_static_friction = 5.0
        # self.book2_dynamic_friction = 5.0
        self.book2_replace_scale = False
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
            p=[0.0, 0.0, 0.0],
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
            p=[0.0, -0.6, 0.0],
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

        # Load second book
        self.book2_pose = sapien.Pose(
            p=[0.0, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 2),
        )
        book2_actor_obj = create_actor(
            scene=self.scene,
            pose=self.book2_pose,
            modelname=self.book2_modelname,
            convex=True,
            model_id=self.book2_model_id,
            scale=self.book2_scale,
            replace_scale=self.book2_replace_scale,
            _idx_if_repeat=2,
            mass=self.book2_mass, 
        )
        self.book2 = book2_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.tensor(self.book_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz[:, 2] = self.book_zs[env_idx]
            qs = torch.tensor(self.book_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.book.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            xyz = torch.tensor(self.bookcase_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz[:, 2] = self.bookcase_zs[env_idx] + xyz[:, 2]
            qs = torch.tensor(self.bookcase_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.bookcase.set_pose(Pose.create_from_pq(xyz, qs))

            # Place second book horizontally on top of the box.
            box_bounds = world_aabb_from_local_mesh(self.bookcase, self.device)
            box_top_z = box_bounds[:, 1, 2]
            box_center_x = (box_bounds[:, 0, 0] + box_bounds[:, 1, 0]) * 0.5
            box_center_y = (box_bounds[:, 0, 1] + box_bounds[:, 1, 1]) * 0.5

            book2_xyz = torch.zeros((b, 3), device=self.device)
            book2_xyz[:, 0] = box_center_x + 0.02
            book2_xyz[:, 1] = box_center_y
            book2_xyz[:, 2] = box_top_z + self.book2_zs[env_idx] + 0.005
            book2_q = torch.tensor(self.book2_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.book2.set_pose(Pose.create_from_pq(p=book2_xyz, q=book2_q))
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

        self.book2_zs = []
        collision_mesh = self.book2.get_first_collision_mesh()
        self.book2_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.book2_zs = common.to_tensor(self.book2_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def _is_in_container_bbox(self, obj_pos: torch.Tensor, container_bounds: torch.Tensor) -> torch.Tensor:
        if container_bounds.ndim == 3:
            lower = container_bounds[:, 0]
            upper = container_bounds[:, 1]
        else:
            lower = container_bounds[0]
            upper = container_bounds[1]
        upper = upper.clone()
        upper[..., 2] += self.container_z_tolerance
        return torch.all(torch.logical_and(obj_pos >= lower, obj_pos <= upper), dim=-1)

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        if not hasattr(self, "_geom_center_fallback_logged"):
            self._geom_center_fallback_logged = set()
        try:
            return geom_center_from_local_mesh(obj, self.device)
        except Exception as exc:
            obj_name = getattr(obj, "name", obj.__class__.__name__)
            if obj_name not in self._geom_center_fallback_logged:
                print(f"[{self.__class__.__name__}] geom center fallback to pose.p for {obj_name}: {exc}")
                self._geom_center_fallback_logged.add(obj_name)
            return obj.pose.p.clone()

    def evaluate(self):
        obj_to_goal_pos = self.bookcase.pose.p - self.book.pose.p
        box_bounds = world_aabb_from_local_mesh(self.bookcase, self.device)
        book_center = self._get_actor_geom_center(self.book)
        is_obj_placed = self._is_in_container_bbox(book_center, box_bounds)
        is_grasped = self.left_agent.is_grasping(self.book) | self.right_agent.is_grasping(self.book)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        book2_center = self._get_actor_geom_center(self.book2)
        clear_goal = self._get_actor_geom_center(self.bookcase) + torch.tensor(
            [-0.3, 0.0, 0.05], device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        book2_cleared = torch.linalg.norm(book2_center - clear_goal, dim=-1) <= 0.18
        result = dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=is_grasped,
            is_book2_grasped=self.left_agent.is_grasping(self.book2),
            is_book2_cleared=book2_cleared,
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
                book2_pose=self.book2.pose.raw_pose,
                left_tcp_to_obj_pos=self.book.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.book.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.bookcase.pose.p - self.book.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        book_pos = self._get_actor_geom_center(self.book)
        book2_pos = self._get_actor_geom_center(self.book2)
        goal_pos = self._get_actor_geom_center(self.bookcase)
        clear_goal = goal_pos + torch.tensor([-0.3, 0.0, 0.05], device=self.device, dtype=torch.float32).unsqueeze(0)
        is_grasped = info["is_grasped"]
        is_book2_grasped = info["is_book2_grasped"]
        is_book2_cleared = info["is_book2_cleared"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_clear_reach = reach_reward(tcp_pos, book2_pos, scale=5.0)
        r_clear_reach = torch.where(is_book2_grasped | is_book2_cleared | success, ones, r_clear_reach)
        r_clear_grasp = grasp_reward(tcp_pos, book2_pos, is_book2_grasped, proximity_scale=5.0)
        r_clear_grasp = torch.where(is_book2_cleared | success, ones, r_clear_grasp)
        r_clear_transport = transport_reward(book2_pos, clear_goal, is_book2_grasped, scale=5.0)
        r_clear_transport = torch.where(is_book2_cleared | success, ones, r_clear_transport)
        r_clear_place = is_book2_cleared.float()
        r_clear_place = torch.where(success, ones, r_clear_place)

        r_book_reach = reach_reward(tcp_pos, book_pos, scale=5.0)
        r_book_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_book_reach)
        r_book_grasp = grasp_reward(tcp_pos, book_pos, is_grasped, proximity_scale=5.0)
        r_book_grasp = torch.where(is_obj_placed | success, ones, r_book_grasp)
        r_book_transport = transport_reward(book_pos, goal_pos, is_grasped, scale=5.0)
        r_book_transport = torch.where(is_obj_placed | success, ones, r_book_transport)
        r_book_place = is_obj_placed.float()
        r_book_place = torch.where(success, ones, r_book_place)

        self.reward_tracker.update("clear_reach", r_clear_reach)
        self.reward_tracker.update("clear_grasp", r_clear_grasp)
        self.reward_tracker.update("clear_transport", r_clear_transport)
        self.reward_tracker.update("clear_place", r_clear_place)
        self.reward_tracker.update("book_reach", r_book_reach)
        self.reward_tracker.update("book_grasp", r_book_grasp)
        self.reward_tracker.update("book_transport", r_book_transport)
        self.reward_tracker.update("book_place", r_book_place)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
