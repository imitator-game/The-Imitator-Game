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
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.building.actors.sketchfab import create_sketchfab_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig
from mani_skill.utils.structs.types import SimConfig
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    grasp_reward,
    reach_reward,
    transport_reward,
)


REWARD_PHASES = [
    "reach_apple",
    "grasp_apple",
    "transport_apple",
    "place_apple",
    "reach_weight",
    "grasp_weight",
    "transport_weight",
    "place_weight",
]


@register_env("TwoRobotPickAppleToScaleL3-v1", max_episode_steps=200)
class TwoRobotPickAppleToScaleEnvL3(BaseEnv):
    """
    **Task Description:**
    Two panda_wristcam robots operate on a table.
    1) Pick the apple.
    2) Place the apple onto the electronic scale.

    **Success Conditions:**
    - Apple is on the scale region.
    - Both robots are static.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    video_info_whitelist = {
        "reward",
        "success",
        "apple_on_scale",
        "weight_on_scale",
        "is_apple_grasped",
        "is_weight_grasped",
        "peak_r_reach_apple",
        "peak_r_grasp_apple",
        "peak_r_transport_apple",
        "peak_r_place_apple",
        "peak_r_reach_weight",
        "peak_r_grasp_weight",
        "peak_r_transport_weight",
        "peak_r_place_weight",
    }

    APPLE_MODEL_NAME = "035_apple"
    SCALE_OBJECT_KEY = "balance_scale"

    # pending: add weight?
    WEIGHT_OBJECT_KEY = "weight_on_scale"

    apple_xy = (-0.15, -0.25)
    scale_xy = (0.0, 0.05)

    weight_xy = (-0.2, 0.5)

    scale_radius = 0.09
    plate_goal_radius = 0.06
    plate_goal_z_tolerance = 0.05
    # scale_plate_center_xy_pose = sapien.Pose(p=[-0.183, -0.279, 0.0])
    scale_height_threshold = 0.05

    apple_scale = 1.0
    scale_scale = 1.0

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
        # Fixed assets, no level-switch logic.
        self.apple_modelname = "035_apple"
        self.apple_model_id = get_model_id(self.apple_modelname, model_id=0)
        self.apple_scale_cfg = (self.apple_scale,) * 3
        self.apple_replace_scale = False

        self.scale_object_key = self.SCALE_OBJECT_KEY

        self.weight_object_key = self.WEIGHT_OBJECT_KEY
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

        self.apple_pose = sapien.Pose(
            p=[self.apple_xy[0], self.apple_xy[1], 0.0],
            q=euler2quat(0, 0, 0),
        )
        self.scale_pose = sapien.Pose(
            p=[self.scale_xy[0], self.scale_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2),
        )
        self.weight_pose = sapien.Pose(
            p=[self.weight_xy[0], self.weight_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 3),
        )

        self.scale = create_sketchfab_actor(
            scene=self.scene,
            object_key=self.scale_object_key,
            pose=self.scale_pose,
            name="balance_scale",
            is_static=True,
        )
        self.apple = create_actor(
            scene=self.scene,
            pose=self.apple_pose,
            modelname=self.apple_modelname,
            convex=True,
            model_id=self.apple_model_id,
            scale=self.apple_scale_cfg,
            replace_scale=self.apple_replace_scale,
            mass=0.5
        ).actor
        # self._lr_mirror_target_actors = [self.scale]

        # pending: add weight?
        self.weight = create_sketchfab_actor(
            scene=self.scene, 
            object_key=self.weight_object_key, 
            pose=self.weight_pose, 
            is_static=False, 
        )

    def _after_reconfigure(self, options: dict):
        def _compute_object_z(obj) -> float:
            collision_mesh = obj.get_first_collision_mesh()
            return -collision_mesh.bounding_box.bounds[0, 2]

        self._apple_z = common.to_tensor([_compute_object_z(self.apple)], device=self.device)
        self._scale_z = common.to_tensor([_compute_object_z(self.scale)], device=self.device)
        self._weight_z = common.to_tensor([_compute_object_z(self.weight)], device=self.device)

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

            scale_xyz = torch.tensor(self.scale_pose.p).repeat(b, 1)
            scale_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            scale_xyz[:, 2] = self._scale_z[0]
            scale_q = torch.tensor(self.scale_pose.q).repeat(b, 1)
            self.scale.set_pose(Pose.create_from_pq(p=scale_xyz, q=scale_q))

            apple_p = torch.tensor(self.apple_pose.p).repeat(b, 1)
            apple_p[:, :2] += torch.rand((b, 2)) * 0.02
            apple_p[:, 2] = self._apple_z[0]
            apple_q = torch.tensor(self.apple_pose.q).repeat(b, 1)
            self.apple.set_pose(Pose.create_from_pq(p=apple_p, q=apple_q))

            weight_p = torch.tensor(self.weight_pose.p).repeat(b, 1)
            weight_p[:, :2] += torch.rand((b, 2)) * 0.02
            weight_p[:, 2] = self._weight_z[0]
            weight_q = torch.tensor(self.weight_pose.q).repeat(b, 1)
            self.weight.set_pose(Pose.create_from_pq(p=weight_p, q=weight_q))

    def _is_near_goal(self, obj_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
        xy_dist = torch.linalg.norm(obj_pos[..., :2] - goal_pos[..., :2], axis=1)
        in_xy = xy_dist <= self.plate_goal_radius
        z_dist = torch.abs(obj_pos[..., 2] - goal_pos[..., 2])
        in_z = z_dist <= self.plate_goal_z_tolerance
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
        scale_pos = self._get_actor_geom_center(self.scale)
        apple_center = self._get_actor_geom_center(self.apple)
        weight_center = self._get_actor_geom_center(self.weight)

        # Match the current motion-planning targets: apple goes to the left plate,
        # weight goes to the right plate.
        apple_goal = scale_pos + common.to_tensor([[0.02, -0.1, 0.1]], device=self.device).repeat(self.num_envs, 1)
        weight_goal = scale_pos + common.to_tensor([[0.02, 0.1, 0.1]], device=self.device).repeat(self.num_envs, 1)

        apple_on_scale = self._is_near_goal(apple_center, goal_pos=apple_goal)
        weight_on_scale = self._is_near_goal(weight_center, goal_pos=weight_goal)
        success = apple_on_scale & weight_on_scale
        is_apple_grasped = self.left_agent.is_grasping(self.apple)
        is_weight_grasped = self.right_agent.is_grasping(self.weight)
        result = dict(
            apple_on_scale=apple_on_scale,
            weight_on_scale=weight_on_scale,
            apple_goal_pos=apple_goal,
            weight_goal_pos=weight_goal,
            is_apple_grasped=is_apple_grasped,
            is_weight_grasped=is_weight_grasped,
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
            scale_pose=self.scale.pose.raw_pose,
            apple_pose=self.apple.pose.raw_pose,
            weight_pose=self.weight.pose.raw_pose,
            apple_on_scale=info["apple_on_scale"],
            weight_on_scale=info["weight_on_scale"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        apple_pos = self._get_actor_geom_center(self.apple)
        weight_pos = self._get_actor_geom_center(self.weight)
        scale_pos = self._get_actor_geom_center(self.scale)
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        apple_goal = info["apple_goal_pos"]
        weight_goal = info["weight_goal_pos"]
        is_apple_grasped = info["is_apple_grasped"]
        is_weight_grasped = info["is_weight_grasped"]
        apple_on_scale = info["apple_on_scale"]
        weight_on_scale = info["weight_on_scale"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_apple = reach_reward(left_tcp, apple_pos, scale=5.0)
        r_reach_apple = torch.where(is_apple_grasped | apple_on_scale | success, ones, r_reach_apple)
        r_grasp_apple = grasp_reward(left_tcp, apple_pos, is_apple_grasped, proximity_scale=5.0)
        r_grasp_apple = torch.where(apple_on_scale | success, ones, r_grasp_apple)
        r_transport_apple = transport_reward(apple_pos, apple_goal, is_apple_grasped, scale=5.0)
        r_transport_apple = torch.where(apple_on_scale | success, ones, r_transport_apple)
        r_place_apple = torch.where(success, ones, apple_on_scale.float())

        r_reach_weight = reach_reward(right_tcp, weight_pos, scale=5.0)
        r_reach_weight = torch.where(is_weight_grasped | weight_on_scale | success, ones, r_reach_weight)
        r_grasp_weight = grasp_reward(right_tcp, weight_pos, is_weight_grasped, proximity_scale=5.0)
        r_grasp_weight = torch.where(weight_on_scale | success, ones, r_grasp_weight)
        r_transport_weight = transport_reward(weight_pos, weight_goal, is_weight_grasped, scale=5.0)
        r_transport_weight = torch.where(weight_on_scale | success, ones, r_transport_weight)
        r_place_weight = torch.where(success, ones, weight_on_scale.float())

        self.reward_tracker.update("reach_apple", r_reach_apple)
        self.reward_tracker.update("grasp_apple", r_grasp_apple)
        self.reward_tracker.update("transport_apple", r_transport_apple)
        self.reward_tracker.update("place_apple", r_place_apple)
        self.reward_tracker.update("reach_weight", r_reach_weight)
        self.reward_tracker.update("grasp_weight", r_grasp_weight)
        self.reward_tracker.update("transport_weight", r_transport_weight)
        self.reward_tracker.update("place_weight", r_place_weight)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
