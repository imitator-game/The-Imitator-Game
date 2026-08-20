import numpy as np
import sapien
import torch
from typing import Any, Dict, Tuple

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.building import actors
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_ycb_model_id,
    apply_l3_robotwin_config,
    is_l2_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    grasp_reward,
    reach_reward,
    transport_reward,
    world_aabb_from_local_mesh,
)


REWARD_PHASES = [
    "reach_apple",
    "grasp_apple",
    "transport_apple",
    "place_apple",
    "reach_banana",
    "grasp_banana",
    "transport_banana",
    "place_banana",
]


@register_env("TwoRobotPickAppleBananaToBaskets-v1", max_episode_steps=200)
class TwoRobotPickAppleBananaToBasketsEnv(BaseEnv):
    """
    **Task Description:**
    Two panda_wristcam robots operate on a table.
    1) Pick the apple outside basket A and place it into basket A.
    2) Pick the banana outside basket B and place it into basket B.

    **Success Conditions:**
    - Apple is in the apple basket region.
    - Banana is in the banana basket region.
    - Both robots are static.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    video_info_whitelist = {
        "reward",
        "success",
        "apple_in_basket",
        "banana_in_basket",
        "is_apple_grasped",
        "is_banana_grasped",
        "peak_r_reach_apple",
        "peak_r_grasp_apple",
        "peak_r_transport_apple",
        "peak_r_place_apple",
        "peak_r_reach_banana",
        "peak_r_grasp_banana",
        "peak_r_transport_banana",
        "peak_r_place_banana",
    }

    APPLE_MODEL_NAME = "035_apple"
    BANANA_MODEL_ID = "011_banana"
    BASKET_MODEL_NAME = "076_breadbasket"

    apple_out_xy = (-0.15, -0.15)
    banana_out_xy = (-0.15, 0.15)
    basket_apple_xy = (0.10, -0.25)
    basket_banana_xy = (0.10, 0.25)

    basket_radius = 0.12
    basket_height_threshold = 0.10

    apple_scale = 0.7
    # banana_scale = 1.0
    basket_scale = 1.5

    def __init__(
        self,
        *args,
        robot_uids: Tuple[str, str] = ("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise: float = 0.02,
        num_envs: int = 1,
        reconfiguration_freq=None,
        hi_res: bool = False,
        wrist_sensor: bool = False, 
        **kwargs
    ):
        self.hi_res = hi_res
        self.wrist_sensor = wrist_sensor
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.apple_model_id = apply_l2_ycb_model_id(
            "013_apple",
            override_id="015_peach",
        )
        self.banana_model_id = apply_l2_ycb_model_id(
            "011_banana",
            override_id="014_lemon",
        )
        (
            self.basket_modelname,
            self.basket_model_id,
            self.basket_scale_cfg,
            self.basket_replace_scale,
        ) = apply_l3_robotwin_config(
            "076_breadbasket",
            model_id=3,
            base_scale=(self.basket_scale,) * 3,
            base_replace_scale=False,
            override_name="008_tray",
            override_id=2,
            override_scale=(self.basket_scale,) * 3,
            override_replace_scale=False,
        )
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

        self.apple_out_pose = sapien.Pose(
            p=[self.apple_out_xy[0], self.apple_out_xy[1], 0.0],
            q=euler2quat(0, 0, 0),
        )
        self.banana_out_pose = sapien.Pose(
            p=[self.banana_out_xy[0], self.banana_out_xy[1], 0.0],
            q=euler2quat(0, 0, -np.pi / 2),
        )
        self.basket_apple_pose = sapien.Pose(
            p=[self.basket_apple_xy[0], self.basket_apple_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2),
        )
        self.apple_in_basket_pose = sapien.Pose(
            p=[self.basket_apple_xy[0], self.basket_apple_xy[1], self.basket_apple_pose.p[2]-0.01],
            q=euler2quat(0, 0, 0),
        )
        self.basket_banana_pose = sapien.Pose(
            p=[self.basket_banana_xy[0], self.basket_banana_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2),
        )

        self.basket_apple = create_actor(
            scene=self.scene,
            pose=self.basket_apple_pose,
            modelname=self.basket_modelname,
            convex=True,
            model_id=self.basket_model_id,
            is_static=True,
            scale=self.basket_scale_cfg,
            replace_scale=self.basket_replace_scale,
            _idx_if_repeat=1,
        ).actor
        self.basket_banana = create_actor(
            scene=self.scene,
            pose=self.basket_banana_pose,
            modelname=self.basket_modelname,
            convex=True,
            model_id=self.basket_model_id,
            is_static=True,
            scale=self.basket_scale_cfg,
            replace_scale=self.basket_replace_scale,
            _idx_if_repeat=2,
        ).actor
        apple_in_builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.apple_model_id}",
            scales=[self.apple_scale],
        )
        apple_in_builder.initial_pose = self.apple_in_basket_pose
        self.apple_in = apple_in_builder.build(name="apple_in")
        apple_out_builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.apple_model_id}",
            scales=[self.apple_scale],
        )
        apple_out_builder._mass = 0.5
        apple_out_builder.initial_pose = self.apple_out_pose
        self.apple_out = apple_out_builder.build(name="apple_out")

        banana_builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.banana_model_id}")
        banana_builder._mass = 0.5
        banana_builder.initial_pose = self.banana_out_pose
        self.banana_out = banana_builder.build(name="banana_out")
        banana_in_builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.banana_model_id}")
        banana_in_builder.initial_pose = self.basket_banana_pose
        self.banana_in = banana_in_builder.build(name="banana_in")

    def _after_reconfigure(self, options: dict):
        def _compute_object_z(obj) -> float:
            collision_mesh = obj.get_first_collision_mesh()
            return -collision_mesh.bounding_box.bounds[0, 2]

        self._apple_in_z = common.to_tensor([_compute_object_z(self.apple_in)], device=self.device)
        self._apple_out_z = common.to_tensor([_compute_object_z(self.apple_out)], device=self.device)
        self._banana_in_z = common.to_tensor([_compute_object_z(self.banana_in)], device=self.device)
        self._banana_out_z = common.to_tensor([_compute_object_z(self.banana_out)], device=self.device)
        self._basket_apple_z = common.to_tensor([_compute_object_z(self.basket_apple)], device=self.device)
        self._basket_banana_z = common.to_tensor([_compute_object_z(self.basket_banana)], device=self.device)

        def _compute_xy_bounds(obj) -> Tuple[torch.Tensor, torch.Tensor]:
            collision_mesh = obj.get_first_collision_mesh(to_world_frame=True)
            bounds = collision_mesh.bounding_box.bounds
            xy_min = common.to_tensor(bounds[0, :2], device=self.device)
            xy_max = common.to_tensor(bounds[1, :2], device=self.device)
            return xy_min, xy_max

        self._basket_apple_xy_min, self._basket_apple_xy_max = _compute_xy_bounds(self.basket_apple)
        self._basket_banana_xy_min, self._basket_banana_xy_max = _compute_xy_bounds(self.basket_banana)
        self._compute_xy_bounds = _compute_xy_bounds

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    REWARD_PHASES, self.num_envs, self.device
                )
            self.reward_tracker.reset(env_idx)

            basket_apple_xyz = torch.tensor(self.basket_apple_pose.p).repeat(b, 1)
            basket_apple_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            basket_apple_xyz[:, 2] = self._basket_apple_z[0]
            basket_apple_q = torch.tensor(self.basket_apple_pose.q).repeat(b, 1)
            self.basket_apple.set_pose(Pose.create_from_pq(p=basket_apple_xyz, q=basket_apple_q))

            basket_banana_xyz = torch.tensor(self.basket_banana_pose.p).repeat(b, 1)
            basket_banana_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            basket_banana_xyz[:, 2] = self._basket_banana_z[0]
            basket_banana_q = torch.tensor(self.basket_banana_pose.q).repeat(b, 1)
            self.basket_banana.set_pose(Pose.create_from_pq(p=basket_banana_xyz, q=basket_banana_q))
            self._basket_apple_xy_min, self._basket_apple_xy_max = self._compute_xy_bounds(self.basket_apple)
            self._basket_banana_xy_min, self._basket_banana_xy_max = self._compute_xy_bounds(self.basket_banana)

            apple_in_p = torch.tensor(self.basket_apple_pose.p).repeat(b, 1)
            apple_in_p[:, :2] += torch.rand((b, 2)) * 0.02
            apple_in_p[:, 2] = self._apple_in_z[0] + basket_apple_xyz[:, 2] + 0.02
            apple_in_q = torch.tensor(self.apple_in_basket_pose.q).repeat(b, 1)
            self.apple_in.set_pose(Pose.create_from_pq(p=apple_in_p, q=apple_in_q))

            apple_out_p = torch.tensor(self.apple_out_pose.p).repeat(b, 1)
            apple_out_p[:, :2] += torch.rand((b, 2)) * 0.02
            apple_out_p[:, 2] = self._apple_out_z[0]
            apple_out_p = apply_l1_offset_xy(apple_out_p, offset=(-0.1, 0.0))
            apple_out_q = torch.tensor(self.apple_out_pose.q).repeat(b, 1)
            self.apple_out.set_pose(Pose.create_from_pq(p=apple_out_p, q=apple_out_q))

            banana_in_p = torch.tensor(self.basket_banana_pose.p).repeat(b, 1)
            banana_in_p[:, :2] += torch.rand((b, 2)) * 0.02
            banana_in_p[:, 2] = self._banana_in_z[0] + basket_banana_xyz[:, 2] + 0.02
            banana_in_q = torch.tensor(self.basket_banana_pose.q).repeat(b, 1)
            self.banana_in.set_pose(Pose.create_from_pq(p=banana_in_p, q=banana_in_q))

            banana_out_p = torch.tensor(self.banana_out_pose.p).repeat(b, 1)
            banana_out_p[:, :2] += torch.rand((b, 2)) * 0.02
            banana_out_p[:, 2] = self._banana_out_z[0]
            banana_out_p = apply_l1_offset_xy(banana_out_p, offset=(-0.1, 0.0))
            banana_out_q = torch.tensor(self.banana_out_pose.q).repeat(b, 1)
            self.banana_out.set_pose(Pose.create_from_pq(p=banana_out_p, q=banana_out_q))

    def _is_in_basket(
        self,
        obj_pos: torch.Tensor,
        basket_xy_min: torch.Tensor,
        basket_xy_max: torch.Tensor,
        basket_z_max: torch.Tensor,
    ) -> torch.Tensor:
        in_x = torch.logical_and(obj_pos[..., 0] >= basket_xy_min[..., 0], obj_pos[..., 0] <= basket_xy_max[..., 0])
        in_y = torch.logical_and(obj_pos[..., 1] >= basket_xy_min[..., 1], obj_pos[..., 1] <= basket_xy_max[..., 1])
        in_xy = torch.logical_and(in_x, in_y)
        in_z = obj_pos[..., 2] <= basket_z_max + self.basket_height_threshold
        return torch.logical_and(in_xy, in_z)

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        if not hasattr(self, "_geom_center_fallback_logged"):
            self._geom_center_fallback_logged = set()
        try:
            return geom_center_from_local_mesh(obj, self.device)
        except Exception as exc:
            name = getattr(obj, "name", type(obj).__name__)
            if name not in self._geom_center_fallback_logged:
                print(
                    f"[{self.__class__.__name__}] geom center fallback to pose.p for {name}: {exc}"
                )
                self._geom_center_fallback_logged.add(name)
            return obj.pose.p.clone()

    def evaluate(self):
        apple_out_pos = self._get_actor_geom_center(self.apple_out)
        banana_out_pos = self._get_actor_geom_center(self.banana_out)
        # if is_l2_enabled():
        #     apple_basket_bounds = world_aabb_from_local_mesh(self.basket_banana, self.device)
        #     banana_basket_bounds = world_aabb_from_local_mesh(self.basket_apple, self.device)
        # else:
        #     apple_basket_bounds = world_aabb_from_local_mesh(self.basket_apple, self.device)
        #     banana_basket_bounds = world_aabb_from_local_mesh(self.basket_banana, self.device)
        apple_basket_bounds = world_aabb_from_local_mesh(self.basket_apple, self.device)
        banana_basket_bounds = world_aabb_from_local_mesh(self.basket_banana, self.device)

        apple_basket_xy_min, apple_basket_xy_max = apple_basket_bounds[:, 0, :2], apple_basket_bounds[:, 1, :2]
        banana_basket_xy_min, banana_basket_xy_max = banana_basket_bounds[:, 0, :2], banana_basket_bounds[:, 1, :2]
        apple_basket_z_max = apple_basket_bounds[:, 1, 2]
        banana_basket_z_max = banana_basket_bounds[:, 1, 2]
        apple_in_basket = self._is_in_basket(
            apple_out_pos,
            apple_basket_xy_min,
            apple_basket_xy_max,
            apple_basket_z_max,
        )
        banana_in_basket = self._is_in_basket(
            banana_out_pos,
            banana_basket_xy_min,
            banana_basket_xy_max,
            banana_basket_z_max,
        )

        is_apple_grasped = self.left_agent.is_grasping(self.apple_out)
        is_banana_grasped = self.right_agent.is_grasping(self.banana_out)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        success = torch.logical_and(apple_in_basket, banana_in_basket)

        # # debug
        # print(f"apple_out_pos: {apple_out_pos}")
        # print(f"banana_out_pos: {banana_out_pos}")
        # print(f"apple_basket_bounds: {apple_basket_bounds}")
        # print(f"banana_basket_bounds: {banana_basket_bounds}")

        result = dict(
            apple_in_basket=apple_in_basket,
            banana_in_basket=banana_in_basket,
            is_apple_grasped=is_apple_grasped,
            is_banana_grasped=is_banana_grasped,
            is_robot_static=is_robot_static,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            basket_apple_pose=self.basket_apple.pose.raw_pose,
            basket_banana_pose=self.basket_banana.pose.raw_pose,
            apple_in_pose=self.apple_in.pose.raw_pose,
            apple_out_pose=self.apple_out.pose.raw_pose,
            banana_in_pose=self.banana_in.pose.raw_pose,
            banana_out_pose=self.banana_out.pose.raw_pose,
            apple_in_basket=info["apple_in_basket"],
            banana_in_basket=info["banana_in_basket"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        apple_pos = self._get_actor_geom_center(self.apple_out)
        banana_pos = self._get_actor_geom_center(self.banana_out)
        if is_l2_enabled():
            apple_goal = self._get_actor_geom_center(self.basket_banana)
            banana_goal = self._get_actor_geom_center(self.basket_apple)
        else:
            apple_goal = self._get_actor_geom_center(self.basket_apple)
            banana_goal = self._get_actor_geom_center(self.basket_banana)

        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        is_apple_grasped = info["is_apple_grasped"]
        is_banana_grasped = info["is_banana_grasped"]
        apple_in_basket = info["apple_in_basket"]
        banana_in_basket = info["banana_in_basket"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_apple = reach_reward(left_tcp, apple_pos, scale=5.0)
        r_reach_apple = torch.where(is_apple_grasped | apple_in_basket | success, ones, r_reach_apple)
        r_grasp_apple = grasp_reward(left_tcp, apple_pos, is_apple_grasped, proximity_scale=5.0)
        r_grasp_apple = torch.where(apple_in_basket | success, ones, r_grasp_apple)
        r_transport_apple = transport_reward(apple_pos, apple_goal, is_apple_grasped, scale=5.0)
        r_transport_apple = torch.where(apple_in_basket | success, ones, r_transport_apple)
        r_place_apple = torch.where(success, ones, apple_in_basket.float())

        r_reach_banana = reach_reward(right_tcp, banana_pos, scale=5.0)
        r_reach_banana = torch.where(is_banana_grasped | banana_in_basket | success, ones, r_reach_banana)
        r_grasp_banana = grasp_reward(right_tcp, banana_pos, is_banana_grasped, proximity_scale=5.0)
        r_grasp_banana = torch.where(banana_in_basket | success, ones, r_grasp_banana)
        r_transport_banana = transport_reward(banana_pos, banana_goal, is_banana_grasped, scale=5.0)
        r_transport_banana = torch.where(banana_in_basket | success, ones, r_transport_banana)
        r_place_banana = torch.where(success, ones, banana_in_basket.float())

        self.reward_tracker.update("reach_apple", r_reach_apple)
        self.reward_tracker.update("grasp_apple", r_grasp_apple)
        self.reward_tracker.update("transport_apple", r_transport_apple)
        self.reward_tracker.update("place_apple", r_place_apple)
        self.reward_tracker.update("reach_banana", r_reach_banana)
        self.reward_tracker.update("grasp_banana", r_grasp_banana)
        self.reward_tracker.update("transport_banana", r_transport_banana)
        self.reward_tracker.update("place_banana", r_place_banana)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
