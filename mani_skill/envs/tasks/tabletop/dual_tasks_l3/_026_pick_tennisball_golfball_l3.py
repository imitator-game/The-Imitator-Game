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
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    transport_reward,
    RewardTracker,
)

REWARD_PHASES = ['reach_t','grasp_t','transport_t','place_t','reach_g','grasp_g','transport_g','place_g']


@register_env("TwoRobotPickTennisBallGolfBallL3-v1", max_episode_steps=200)
class TwoRobotPickTennisBallGolfBallEnvL3(BaseEnv):
    """
    **Task Description:**
    Two panda_wristcam robots operate on a table.
    1) Pick the tennis outside basket A and place it into basket A.
    2) Pick the golf outside basket B and place it into basket B.

    **Success Conditions:**
    - tennis is in the tennis basket region.
    - golf is in the golf basket region.
    - Both robots are static.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    tennis_out_xy = (-0.25, -0.15)
    golf_out_xy = (-0.25, 0.15)
    basket_tennis_xy = (0., -0.1)
    basket_golf_xy = (0., 0.1)

    basket_radius = 0.12
    basket_height_threshold = 0.10

    tennis_scale = 1.2
    # golf_scale = 1.0
    basket_scale = 1.0

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
        self.tennis_model_id = "056_tennis_ball"
        self.golf_model_id = "058_golf_ball"
        (
            self.basket_modelname,
            self.basket_model_id,
            self.basket_scale_cfg,
            self.basket_replace_scale,
        ) = (
            "062_plasticbox",
            7,
            [1.0] * 3,
            False,
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

        self.tennis_out_pose = sapien.Pose(
            p=[self.tennis_out_xy[0], self.tennis_out_xy[1], 0.0],
            q=euler2quat(0, 0, 0),
        )
        self.golf_out_pose = sapien.Pose(
            p=[self.golf_out_xy[0], self.golf_out_xy[1], 0.0],
            q=euler2quat(0, 0, 0),
        )
        self.basket_tennis_pose = sapien.Pose(
            p=[self.basket_tennis_xy[0], self.basket_tennis_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, 0.),
        )
        self.tennis_in_basket_pose = sapien.Pose(
            p=[self.basket_tennis_xy[0], self.basket_tennis_xy[1], self.basket_tennis_pose.p[2]-0.01],
            q=euler2quat(0, 0, 0),
        )
        self.basket_golf_pose = sapien.Pose(
            p=[self.basket_golf_xy[0], self.basket_golf_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi),
        )

        self.basket_tennis = create_actor(
            scene=self.scene,
            pose=self.basket_tennis_pose,
            modelname=self.basket_modelname,
            convex=True,
            model_id=self.basket_model_id,
            is_static=True,
            scale=self.basket_scale_cfg,
            replace_scale=self.basket_replace_scale,
            _idx_if_repeat=1,
        ).actor
        self.basket_golf = create_actor(
            scene=self.scene,
            pose=self.basket_golf_pose,
            modelname=self.basket_modelname,
            convex=True,
            model_id=self.basket_model_id,
            is_static=True,
            scale=self.basket_scale_cfg,
            replace_scale=self.basket_replace_scale,
            _idx_if_repeat=2,
        ).actor
        tennis_in_builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.tennis_model_id}")
        tennis_in_builder.initial_pose = self.tennis_in_basket_pose
        self.tennis_in = tennis_in_builder.build(name="tennis_in")
        tennis_out_builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.tennis_model_id}")
        tennis_out_builder._mass = 0.5
        tennis_out_builder.initial_pose = self.tennis_out_pose
        self.tennis_out = tennis_out_builder.build(name="tennis_out")

        golf_builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.golf_model_id}")
        golf_builder._mass = 0.5
        golf_builder.initial_pose = self.golf_out_pose
        self.golf_out = golf_builder.build(name="golf_out")
        golf_in_builder = actors.get_actor_builder(self.scene, id=f"ycb:{self.golf_model_id}")
        golf_in_builder.initial_pose = self.basket_golf_pose
        self.golf_in = golf_in_builder.build(name="golf_in")

    def _after_reconfigure(self, options: dict):
        def _compute_object_z(obj) -> float:
            collision_mesh = obj.get_first_collision_mesh()
            return -collision_mesh.bounding_box.bounds[0, 2]

        self._tennis_in_z = common.to_tensor([_compute_object_z(self.tennis_in)], device=self.device)
        self._tennis_out_z = common.to_tensor([_compute_object_z(self.tennis_out)], device=self.device)
        self._golf_in_z = common.to_tensor([_compute_object_z(self.golf_in)], device=self.device)
        self._golf_out_z = common.to_tensor([_compute_object_z(self.golf_out)], device=self.device)
        self._basket_tennis_z = common.to_tensor([_compute_object_z(self.basket_tennis)], device=self.device)
        self._basket_golf_z = common.to_tensor([_compute_object_z(self.basket_golf)], device=self.device)

        def _compute_xy_bounds(obj) -> Tuple[torch.Tensor, torch.Tensor]:
            collision_mesh = obj.get_first_collision_mesh(to_world_frame=True)
            bounds = collision_mesh.bounding_box.bounds
            xy_min = common.to_tensor(bounds[0, :2], device=self.device)
            xy_max = common.to_tensor(bounds[1, :2], device=self.device)
            return xy_min, xy_max

        self._basket_tennis_xy_min, self._basket_tennis_xy_max = _compute_xy_bounds(self.basket_tennis)
        self._basket_golf_xy_min, self._basket_golf_xy_max = _compute_xy_bounds(self.basket_golf)
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
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

            basket_tennis_xyz = torch.tensor(self.basket_tennis_pose.p).repeat(b, 1)
            basket_tennis_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            basket_tennis_xyz[:, 2] = self._basket_tennis_z[0]
            basket_tennis_q = torch.tensor(self.basket_tennis_pose.q).repeat(b, 1)
            self.basket_tennis.set_pose(Pose.create_from_pq(p=basket_tennis_xyz, q=basket_tennis_q))

            basket_golf_xyz = torch.tensor(self.basket_golf_pose.p).repeat(b, 1)
            basket_golf_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            basket_golf_xyz[:, 2] = self._basket_golf_z[0]
            basket_golf_q = torch.tensor(self.basket_golf_pose.q).repeat(b, 1)
            self.basket_golf.set_pose(Pose.create_from_pq(p=basket_golf_xyz, q=basket_golf_q))
            self._basket_tennis_xy_min, self._basket_tennis_xy_max = self._compute_xy_bounds(self.basket_tennis)
            self._basket_golf_xy_min, self._basket_golf_xy_max = self._compute_xy_bounds(self.basket_golf)

            tennis_in_p = torch.tensor(self.basket_tennis_pose.p).repeat(b, 1)
            tennis_in_p[:, :2] += torch.rand((b, 2)) * 0.02
            tennis_in_p[:, 2] = self._tennis_in_z[0] + basket_tennis_xyz[:, 2] + 0.02
            tennis_in_q = torch.tensor(self.tennis_in_basket_pose.q).repeat(b, 1)
            self.tennis_in.set_pose(Pose.create_from_pq(p=tennis_in_p, q=tennis_in_q))

            tennis_out_p = torch.tensor(self.tennis_out_pose.p).repeat(b, 1)
            tennis_out_p[:, :2] += torch.rand((b, 2)) * 0.02
            tennis_out_p[:, 2] = self._tennis_out_z[0]
            tennis_out_p = apply_l1_offset_xy(tennis_out_p, offset=(-0.1, 0.0))
            tennis_out_q = torch.tensor(self.tennis_out_pose.q).repeat(b, 1)
            self.tennis_out.set_pose(Pose.create_from_pq(p=tennis_out_p, q=tennis_out_q))

            golf_in_p = torch.tensor(self.basket_golf_pose.p).repeat(b, 1)
            golf_in_p[:, :2] += torch.rand((b, 2)) * 0.02
            golf_in_p[:, 2] = self._golf_in_z[0] + basket_golf_xyz[:, 2] + 0.02
            golf_in_q = torch.tensor(self.basket_golf_pose.q).repeat(b, 1)
            self.golf_in.set_pose(Pose.create_from_pq(p=golf_in_p, q=golf_in_q))

            golf_out_p = torch.tensor(self.golf_out_pose.p).repeat(b, 1)
            golf_out_p[:, :2] += torch.rand((b, 2)) * 0.02
            golf_out_p[:, 2] = self._golf_out_z[0]
            golf_out_p = apply_l1_offset_xy(golf_out_p, offset=(-0.1, 0.0))
            golf_out_q = torch.tensor(self.golf_out_pose.q).repeat(b, 1)
            self.golf_out.set_pose(Pose.create_from_pq(p=golf_out_p, q=golf_out_q))

    def _is_in_basket(
        self,
        obj_pos: torch.Tensor,
        basket_xy_min: torch.Tensor,
        basket_xy_max: torch.Tensor,
    ) -> torch.Tensor:
        in_x = torch.logical_and(obj_pos[..., 0] >= basket_xy_min[0], obj_pos[..., 0] <= basket_xy_max[0])
        in_y = torch.logical_and(obj_pos[..., 1] >= basket_xy_min[1], obj_pos[..., 1] <= basket_xy_max[1])
        in_xy = torch.logical_and(in_x, in_y)
        in_z = obj_pos[..., 2] <= self.basket_height_threshold
        return torch.logical_and(in_xy, in_z)

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        collision_mesh = obj.get_first_collision_mesh(to_world_frame=True)
        bounds = collision_mesh.bounding_box.bounds
        center = (bounds[0] + bounds[1]) * 0.5
        return common.to_tensor(center, device=self.device)

    def evaluate(self):
        tennis_out_pos = self._get_actor_geom_center(self.tennis_out).repeat(self.num_envs, 1)
        golf_out_pos = self._get_actor_geom_center(self.golf_out).repeat(self.num_envs, 1)
        tennis_in_basket = self._is_in_basket(
            tennis_out_pos,
            self._basket_tennis_xy_min,
            self._basket_tennis_xy_max,
        )
        golf_in_basket = self._is_in_basket(
            golf_out_pos,
            self._basket_golf_xy_min,
            self._basket_golf_xy_max,
        )
        is_tennis_grasped = self.left_agent.is_grasping(self.tennis_out)
        is_golf_grasped = self.right_agent.is_grasping(self.golf_out)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        success = tennis_in_basket & golf_in_basket
        result = dict(
            tennis_in_basket=tennis_in_basket,
            golf_in_basket=golf_in_basket,
            is_tennis_grasped=is_tennis_grasped,
            is_golf_grasped=is_golf_grasped,
            is_robot_static=is_robot_static,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            basket_tennis_pose=self.basket_tennis.pose.raw_pose,
            basket_golf_pose=self.basket_golf.pose.raw_pose,
            tennis_in_pose=self.tennis_in.pose.raw_pose,
            tennis_out_pose=self.tennis_out.pose.raw_pose,
            golf_in_pose=self.golf_in.pose.raw_pose,
            golf_out_pose=self.golf_out.pose.raw_pose,
            tennis_in_basket=info["tennis_in_basket"],
            golf_in_basket=info["golf_in_basket"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        tennis_pos = self.tennis_out.pose.p
        golf_pos = self.golf_out.pose.p
        basket_t_pos = self.basket_tennis.pose.p
        basket_g_pos = self.basket_golf.pose.p
        is_tg = info["is_tennis_grasped"]
        is_gg = info["is_golf_grasped"]
        t_in = info["tennis_in_basket"]
        g_in = info["golf_in_basket"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_t = reach_reward(left_tcp, tennis_pos, scale=5.0)
        r_reach_t = torch.where(is_tg | t_in | success, ones, r_reach_t)
        r_grasp_t = grasp_reward(left_tcp, tennis_pos, is_tg, proximity_scale=5.0)
        r_grasp_t = torch.where(t_in | success, ones, r_grasp_t)
        r_transport_t = transport_reward(tennis_pos, basket_t_pos, is_tg, scale=5.0)
        r_transport_t = torch.where(t_in | success, ones, r_transport_t)
        r_place_t = t_in.float()
        r_place_t = torch.where(success, ones, r_place_t)

        r_reach_g = reach_reward(right_tcp, golf_pos, scale=5.0)
        r_reach_g = torch.where(is_gg | g_in | success, ones, r_reach_g)
        r_grasp_g = grasp_reward(right_tcp, golf_pos, is_gg, proximity_scale=5.0)
        r_grasp_g = torch.where(g_in | success, ones, r_grasp_g)
        r_transport_g = transport_reward(golf_pos, basket_g_pos, is_gg, scale=5.0)
        r_transport_g = torch.where(g_in | success, ones, r_transport_g)
        r_place_g = g_in.float()
        r_place_g = torch.where(success, ones, r_place_g)

        for name, val in [("reach_t",r_reach_t),("grasp_t",r_grasp_t),("transport_t",r_transport_t),("place_t",r_place_t),("reach_g",r_reach_g),("grasp_g",r_grasp_g),("transport_g",r_transport_g),("place_g",r_place_g)]:
            self.reward_tracker.update(name, val)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)