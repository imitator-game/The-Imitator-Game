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
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_quat_offset,
    apply_l2_robotwin_config,
    apply_l3_robotwin_config,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    grasp_reward,
    reach_reward,
    transport_reward,
)


REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotPickAppleToScale-v1", max_episode_steps=200)
class TwoRobotPickAppleToScaleEnv(BaseEnv):
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

    APPLE_MODEL_NAME = "035_apple"
    SCALE_MODEL_NAME = "072_electronicscale"

    apple_xy = (-0.15, -0.25)
    scale_xy = (-0.15, 0.05)

    scale_radius = 0.09
    # scale_plate_center_xy_pose = sapien.Pose(p=[-0.183, -0.279, 0.0])
    scale_height_threshold = 0.20

    apple_scale = 1.0
    scale_scale = 0.15

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
        (
            self.apple_modelname,
            self.apple_model_id,
            self.apple_scale_cfg,
            self.apple_replace_scale,
        ) = apply_l2_robotwin_config(
            "035_apple",
            model_id=0,
            base_scale=(self.apple_scale,) * 3,
            base_replace_scale=False,
            override_name="103_fruit",
            override_id=0,
            override_scale=(0.03, 0.03, 0.03),
            override_replace_scale=True,
        )
        self.apple_l2_q_offset = euler2quat(np.pi / 2, 0.0, 0.0)
        (
            self.scale_modelname,
            self.scale_model_id,
            self.scale_scale_cfg,
            self.scale_replace_scale,
        ) = apply_l3_robotwin_config(
            self.SCALE_MODEL_NAME,
            model_id=0,
            base_scale=(self.scale_scale,) * 3,
            base_replace_scale=True,
            override_name=self.SCALE_MODEL_NAME,
            override_id=5,
            override_scale=(self.scale_scale,) * 3,
            override_replace_scale=True,
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

        self.apple_pose = sapien.Pose(
            p=[self.apple_xy[0], self.apple_xy[1], 0.0],
            q=euler2quat(0, 0, 0),
        )
        self.scale_pose = sapien.Pose(
            p=[self.scale_xy[0], self.scale_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2),
        )

        self.scale = create_actor(
            scene=self.scene,
            pose=self.scale_pose,
            modelname=self.scale_modelname,
            convex=True,
            model_id=self.scale_model_id,
            is_static=True,
            replace_scale=self.scale_replace_scale,
            scale=self.scale_scale_cfg,
        ).actor
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

    def _after_reconfigure(self, options: dict):
        def _compute_object_z(obj) -> float:
            collision_mesh = obj.get_first_collision_mesh()
            return -collision_mesh.bounding_box.bounds[0, 2]

        self._apple_z = common.to_tensor([_compute_object_z(self.apple)], device=self.device)
        self._scale_z = common.to_tensor([_compute_object_z(self.scale)], device=self.device)

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
            apple_p = apply_l1_offset_xy(apple_p, offset=(-0.1, 0.0))
            apple_q = torch.tensor(self.apple_pose.q).repeat(b, 1)
            apple_q = apply_l2_quat_offset(apple_q, offset_q=self.apple_l2_q_offset)
            self.apple.set_pose(Pose.create_from_pq(p=apple_p, q=apple_q))

    def _is_on_scale(self, obj_pos: torch.Tensor, scale_pos: torch.Tensor) -> torch.Tensor:
        xy_dist = torch.linalg.norm(obj_pos[..., :2] - scale_pos[..., :2], axis=1)
        in_xy = xy_dist <= self.scale_radius
        in_z = obj_pos[..., 2] <= self.scale_height_threshold
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
        apple_on_scale = self._is_on_scale(apple_center, scale_pos=scale_pos)
        is_grasped = self.left_agent.is_grasping(self.apple)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        success = apple_on_scale
        result = dict(
            apple_on_scale=apple_on_scale,
            is_grasped=is_grasped,
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
            scale_pose=self.scale.pose.raw_pose,
            apple_pose=self.apple.pose.raw_pose,
            apple_on_scale=info["apple_on_scale"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        apple_pos = self._get_actor_geom_center(self.apple)
        scale_pos = self._get_actor_geom_center(self.scale)
        tcp_pos = self.left_agent.tcp.pose.p
        is_grasped = info["is_grasped"]
        apple_on_scale = info["apple_on_scale"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, apple_pos, scale=5.0)
        r_reach = torch.where(is_grasped | apple_on_scale | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, apple_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(apple_on_scale | success, ones, r_grasp)
        r_transport = transport_reward(apple_pos, scale_pos, is_grasped, scale=5.0)
        r_transport = torch.where(apple_on_scale | success, ones, r_transport)
        r_place = torch.where(success, ones, apple_on_scale.float())

        self.reward_tracker.update("reach", r_reach)
        self.reward_tracker.update("grasp", r_grasp)
        self.reward_tracker.update("transport", r_transport)
        self.reward_tracker.update("place", r_place)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
