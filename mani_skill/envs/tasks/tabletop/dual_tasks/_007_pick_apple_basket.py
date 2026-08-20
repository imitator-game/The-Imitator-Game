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
from mani_skill.utils import common
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_ycb_model_id,
    apply_l3_robotwin_config,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    reach_reward,
    grasp_reward,
    transport_reward,
    world_aabb_from_local_mesh,
)


REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotPickAppleBasket-v1", max_episode_steps=100)
class TwoRobotPickAppleBasketEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up an apple (YCB object) and place it into a breadbasket container.
    There are two robots in this task, both positioned side-by-side in front of the table:
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    goal_thresh = 0.05  # Threshold for checking if apple is inside breadbasket
    basket_height_threshold = 0.10
    apple_scale = 0.7

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

        # Breadbasket (robotwin object, container/goal)
        (
            self.breadbasket_modelname,
            self.breadbasket_model_id,
            self.breadbasket_scale,
            self.breadbasket_replace_scale,
        ) = apply_l3_robotwin_config(
            "076_breadbasket",
            model_id=3,
            base_scale=(1.8, 1.8, 1.8),
            base_replace_scale=False,
            override_name="008_tray",
            override_id=2,
            override_scale=(1.2, 1.2, 1.2),
            override_replace_scale=False,
        )

        # Apple (YCB object, to grasp)
        self.apple_model_id = apply_l2_ycb_model_id("013_apple", override_id="016_pear")

        # Additional YCB objects in basket
        self.banana_model_id = "011_banana"
        self.lemon_model_id = "014_lemon"

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

        # Load apple (YCB object to grasp)
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.apple_model_id}",
            scales=[self.apple_scale],
        )
        builder._mass = 0.5
        builder.initial_pose = sapien.Pose(p=[0, 0.5, 0])
        apple_actor = builder.build(name="apple")
        self.apple = apple_actor

        # Load banana (YCB object in basket)
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.banana_model_id}",
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.5, 0])
        banana_actor = builder.build(name="banana")
        self.banana = banana_actor

        # Load lemon (YCB object in basket)
        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{self.lemon_model_id}",
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.5, 0])
        lemon_actor = builder.build(name="lemon")
        self.lemon = lemon_actor

        # Load breadbasket (robotwin container/goal)
        # Use euler2quat(pi/2, 0, 0) to make object upright (X-axis rotation)
        breadbasket_pose = sapien.Pose(
            p=[0, -0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        breadbasket_obj = create_actor(
            scene=self.scene,
            pose=breadbasket_pose,
            modelname=self.breadbasket_modelname,
            scale=self.breadbasket_scale,
            replace_scale=self.breadbasket_replace_scale,
            convex=True,
            is_static=True,
            model_id=self.breadbasket_model_id,
        )
        self.breadbasket = breadbasket_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Generate shared random offset for basket and items inside it (same as spoon task)
            basket_shared_xy_offset = torch.rand((b, 2), device=self.device) * 0.02  # Per-env shared random offset

            # Initialize apple (object to grasp) - with independent randomization
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.1
            xyz[:, 1] = -0.15
            xyz[:, 2] = self.apple_z
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02  # Per-env random offset
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.1))

            z_rotation = np.pi / 6
            base_quat = euler2quat(0.0, 0.0, z_rotation)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.apple.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Initialize breadbasket (goal container) - with shared randomization
            basket_xyz = torch.zeros((b, 3), device=self.device)
            basket_xyz[:, 0] = 0.0
            basket_xyz[:, 1] = 0.1
            basket_xyz[:, 2] = self.breadbasket_z 
            basket_xyz[:, :2] += basket_shared_xy_offset 

            # Fixed upright orientation - no Z rotation randomization
            base_quat = euler2quat(np.pi/2, 0.0, 0.0)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.breadbasket.set_pose(Pose.create_from_pq(basket_xyz, qs))
            self._basket_xy_min, self._basket_xy_max = self._compute_xy_bounds(self.breadbasket)

            # Initialize banana in basket - positioned inside basket, maintain relative position
            banana_xyz = torch.zeros((b, 3), device=self.device)
            banana_xyz[:, 0] = 0.0 + 0.06  # Base position + relative offset inside basket
            banana_xyz[:, 1] = 0.1 - 0.02  # Base position + relative offset inside basket
            banana_xyz[:, 2] = self.breadbasket_z + 0.05 
            banana_xyz[:, :2] += basket_shared_xy_offset

            banana_quat = euler2quat(0.0, 0.0, np.pi)  # Fixed rotation
            banana_qs = torch.tensor([banana_quat] * b, device=self.device, dtype=torch.float32)
            self.banana.set_pose(Pose.create_from_pq(p=banana_xyz, q=banana_qs))

            # Initialize lemon in basket - positioned inside basket, maintain relative position
            lemon_xyz = torch.zeros((b, 3), device=self.device)
            lemon_xyz[:, 0] = 0.0 + 0.03  # Base position + relative offset inside basket
            lemon_xyz[:, 1] = 0.1 + (-0.02)  # Base position + relative offset inside basket
            lemon_xyz[:, 2] = self.breadbasket_z + 0.05  # Slightly above basket bottom
            lemon_xyz[:, :2] += basket_shared_xy_offset  # Apply same shared random offset

            lemon_quat = euler2quat(0.0, 0.0, -np.pi / 6)  # Fixed rotation
            lemon_qs = torch.tensor([lemon_quat] * b, device=self.device, dtype=torch.float32)
            self.lemon.set_pose(Pose.create_from_pq(p=lemon_xyz, q=lemon_qs))
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

    def _after_reconfigure(self, options: dict):
        # Get z-offset for apple to place it on table surface
        collision_mesh = self.apple.get_first_collision_mesh()
        if collision_mesh is not None:
            self.apple_z = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            self.apple_z = 0.02  # Default height if no collision mesh

        # Get z-offset for breadbasket to place it on table surface
        collision_mesh = self.breadbasket.get_first_collision_mesh()
        self.breadbasket_z = 0.0  # Default height if no collision mesh

        def _compute_xy_bounds(obj) -> Tuple[torch.Tensor, torch.Tensor]:
            collision_mesh = obj.get_first_collision_mesh(to_world_frame=True)
            bounds = collision_mesh.bounding_box.bounds
            xy_min = common.to_tensor(bounds[0, :2], device=self.device)
            xy_max = common.to_tensor(bounds[1, :2], device=self.device)
            return xy_min, xy_max

        self._compute_xy_bounds = _compute_xy_bounds
        self._basket_xy_min, self._basket_xy_max = _compute_xy_bounds(self.breadbasket)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

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
            obj_name = getattr(obj, "name", obj.__class__.__name__)
            if obj_name not in self._geom_center_fallback_logged:
                print(
                    f"[{self.__class__.__name__}] geom center fallback to pose.p for {obj_name}: {exc}"
                )
                self._geom_center_fallback_logged.add(obj_name)
            return obj.pose.p.clone()

    def evaluate(self):
        apple_pos = self._get_actor_geom_center(self.apple)
        basket_center = self._get_actor_geom_center(self.breadbasket)
        basket_bounds = world_aabb_from_local_mesh(self.breadbasket, self.device)
        basket_xy_min = basket_bounds[:, 0, :2]
        basket_xy_max = basket_bounds[:, 1, :2]
        basket_z_max = basket_bounds[:, 1, 2]
        apple_to_basket_xy_dist = torch.linalg.norm(
            basket_center[:, :2] - apple_pos[:, :2], axis=1
        )
        apple_to_basket_z_offset = apple_pos[:, 2] - basket_center[:, 2]
        apple_to_basket_z_dist = torch.abs(apple_to_basket_z_offset)

        is_obj_placed = self._is_in_basket(
            apple_pos,
            basket_xy_min,
            basket_xy_max,
            basket_z_max,
        )

        is_grasped = self.left_agent.is_grasping(self.apple)
        is_robot_static = self.left_agent.is_static(0.2)
        
        result = dict(
            is_grasped=is_grasped,
            apple_to_basket_xy_dist=apple_to_basket_xy_dist,
            apple_to_basket_z_dist=apple_to_basket_z_dist,
            apple_to_basket_z_offset=apple_to_basket_z_offset,
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
            tcp_pose=self.left_agent.tcp.pose.raw_pose,
            goal_pos=self.breadbasket.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                apple_pose=self.apple.pose.raw_pose,
                tcp_to_apple_pos=self.apple.pose.p - self.left_agent.tcp.pose.p,
                breadbasket_pose=self.breadbasket.pose.raw_pose,
                tcp_to_breadbasket_pos=self.breadbasket.pose.p - self.left_agent.tcp.pose.p,
                apple_to_breadbasket_pos=self.breadbasket.pose.p - self.apple.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        apple_pos = self._get_actor_geom_center(self.apple)
        basket_pos = self._get_actor_geom_center(self.breadbasket)
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, apple_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, apple_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_obj_placed | success, ones, r_grasp)
        r_transport = transport_reward(apple_pos, basket_pos, is_grasped, scale=5.0)
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
