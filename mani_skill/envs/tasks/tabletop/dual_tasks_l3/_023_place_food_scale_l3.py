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
    reach_reward,
    grasp_reward,
    transport_reward,
    wipe_progress_reward,
    above_reward,
    tanh_reward,
    RewardTracker,
)

REWARD_PHASES = ['reach_v','grasp_v','transport_v','place_v','reach_w','grasp_w','transport_w','place_w']


@register_env("TwoRobotPlaceFoodScaleL3-v1", max_episode_steps=100)
class TwoRobotPlaceFoodScaleEnvL3(BaseEnv):
    """
    **Task Description:**
    Two panda_wristcam robots operate on a table.
    1) Pick the food.
    2) Place the food onto the electronic scale.

    **Success Conditions:**
    - Food is on the scale region.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    VEGETABLE_MODEL_NAME = "069_vagetable"
    SCALE_OBJECT_KEY = "balance_scale"

    # pending: add weight?
    WEIGHT_OBJECT_KEY = "weight_on_scale"

    vegetable_xy = (-0.15, -0.17)
    scale_xy = (0.0, 0.05)

    weight_xy = (-0.2, 0.4)

    scale_radius = 0.09
    plate_goal_radius = 0.08
    plate_goal_z_tolerance = 0.05
    # scale_plate_center_xy_pose = sapien.Pose(p=[-0.183, -0.279, 0.0])
    scale_height_threshold = 0.05

    vegetable_scale = 0.5
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
        self.vegetable_modelname = "069_vagetable"
        self.vegetable_model_id = get_model_id(self.vegetable_modelname, model_id=1)
        self.vegetable_scale_cfg = (self.vegetable_scale,) * 3
        self.vegetable_replace_scale = False

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

        self.vegetable_pose = sapien.Pose(
            p=[self.vegetable_xy[0], self.vegetable_xy[1], 0.0],
            q=euler2quat(-np.pi / 2, 0, 0),
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
        self.vegetable = create_actor(
            scene=self.scene,
            pose=self.vegetable_pose,
            modelname=self.vegetable_modelname,
            convex=True,
            model_id=self.vegetable_model_id,
            scale=self.vegetable_scale_cfg,
            replace_scale=self.vegetable_replace_scale,
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

        self._vegetable_z = common.to_tensor([_compute_object_z(self.vegetable)], device=self.device)
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

            scale_xyz = torch.tensor(self.scale_pose.p).repeat(b, 1)
            scale_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            scale_xyz[:, 2] = self._scale_z[0]
            scale_q = torch.tensor(self.scale_pose.q).repeat(b, 1)
            self.scale.set_pose(Pose.create_from_pq(p=scale_xyz, q=scale_q))

            vegetable_p = torch.tensor(self.vegetable_pose.p).repeat(b, 1)
            vegetable_p[:, :2] += torch.rand((b, 2)) * 0.02
            vegetable_p[:, 2] = self._vegetable_z[0]
            vegetable_q = torch.tensor(self.vegetable_pose.q).repeat(b, 1)
            self.vegetable.set_pose(Pose.create_from_pq(p=vegetable_p, q=vegetable_q))

            weight_p = torch.tensor(self.weight_pose.p).repeat(b, 1)
            weight_p[:, :2] += torch.rand((b, 2)) * 0.02
            weight_p[:, 2] = self._weight_z[0]
            weight_q = torch.tensor(self.weight_pose.q).repeat(b, 1)
            self.weight.set_pose(Pose.create_from_pq(p=weight_p, q=weight_q))

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

    def _is_near_goal(self, obj_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
        xy_dist = torch.linalg.norm(obj_pos[..., :2] - goal_pos[..., :2], axis=1)
        in_xy = xy_dist <= self.plate_goal_radius
        z_dist = torch.abs(obj_pos[..., 2] - goal_pos[..., 2])
        in_z = z_dist <= self.plate_goal_z_tolerance
        return torch.logical_and(in_xy, in_z)

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        collision_mesh = obj.get_first_collision_mesh(to_world_frame=True)
        bounds = collision_mesh.bounding_box.bounds
        center = (bounds[0] + bounds[1]) * 0.5
        return common.to_tensor(center, device=self.device)

    def evaluate(self):
        scale_pos = self._get_actor_geom_center(self.scale).repeat(self.num_envs, 1)
        vegetable_center = self._get_actor_geom_center(self.vegetable).repeat(self.num_envs, 1)
        weight_center = self._get_actor_geom_center(self.weight).repeat(self.num_envs, 1)

        # Match the current motion-planning targets: vegetable goes to the left plate,
        # weight goes to the right plate.
        vegetable_goal = scale_pos + common.to_tensor(
            [[0.02, -0.1, 0.1]], device=self.device
        ).repeat(self.num_envs, 1)
        weight_goal = scale_pos + common.to_tensor(
            [[0.02, 0.1, 0.1]], device=self.device
        ).repeat(self.num_envs, 1)

        vegetable_on_scale = self._is_near_goal(vegetable_center, goal_pos=vegetable_goal)
        weight_on_scale = self._is_near_goal(weight_center, goal_pos=weight_goal)
        success = vegetable_on_scale & weight_on_scale
        # # debug
        # apple_xy_dist = torch.linalg.norm(apple_center[..., :2] - apple_goal[..., :2], axis=1)
        # weight_xy_dist = torch.linalg.norm(weight_center[..., :2] - weight_goal[..., :2], axis=1)
        # apple_z_dist = torch.abs(apple_center[..., 2] - apple_goal[..., 2])
        # weight_z_dist = torch.abs(weight_center[..., 2] - weight_goal[..., 2])
        # print(
        #     "[Task9 evaluate]",
        #     {
        #         "scale_center": scale_pos[0].detach().cpu().numpy().tolist(),
        #         "apple_center": apple_center[0].detach().cpu().numpy().tolist(),
        #         "apple_goal": apple_goal[0].detach().cpu().numpy().tolist(),
        #         "apple_xy_dist": float(apple_xy_dist[0].detach().cpu().item()),
        #         "apple_z_dist": float(apple_z_dist[0].detach().cpu().item()),
        #         "apple_on_scale": bool(apple_on_scale[0].detach().cpu().item()),
        #         "weight_center": weight_center[0].detach().cpu().numpy().tolist(),
        #         "weight_goal": weight_goal[0].detach().cpu().numpy().tolist(),
        #         "weight_xy_dist": float(weight_xy_dist[0].detach().cpu().item()),
        #         "weight_z_dist": float(weight_z_dist[0].detach().cpu().item()),
        #         "weight_on_scale": bool(weight_on_scale[0].detach().cpu().item()),
        #         "success": bool(success[0].detach().cpu().item()),
        #     },
        # )
        is_v_grasped = self.left_agent.is_grasping(self.vegetable)
        is_w_grasped = self.right_agent.is_grasping(self.weight)
        result = dict(
            vegetable_on_scale=vegetable_on_scale,
            weight_on_scale=weight_on_scale,
            vegetable_goal_pos=vegetable_goal,
            weight_goal_pos=weight_goal,
            is_v_grasped=is_v_grasped,
            is_w_grasped=is_w_grasped,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            scale_pose=self.scale.pose.raw_pose,
            vegetable_pose=self.vegetable.pose.raw_pose,
            weight_pose=self.weight.pose.raw_pose,
            vegetable_on_scale=info["vegetable_on_scale"],
            weight_on_scale=info["weight_on_scale"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        veg_pos = self.vegetable.pose.p
        weight_pos = self.weight.pose.p
        veg_goal = info["vegetable_goal_pos"]
        w_goal = info["weight_goal_pos"]
        is_v_grasped = info["is_v_grasped"]
        is_w_grasped = info["is_w_grasped"]
        v_on = info["vegetable_on_scale"]
        w_on = info["weight_on_scale"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_v = reach_reward(left_tcp, veg_pos, scale=5.0)
        r_reach_v = torch.where(is_v_grasped | v_on | success, ones, r_reach_v)
        r_grasp_v = grasp_reward(left_tcp, veg_pos, is_v_grasped, proximity_scale=5.0)
        r_grasp_v = torch.where(v_on | success, ones, r_grasp_v)
        r_transport_v = transport_reward(veg_pos, veg_goal, is_v_grasped, scale=5.0)
        r_transport_v = torch.where(v_on | success, ones, r_transport_v)
        r_place_v = v_on.float()
        r_place_v = torch.where(success, ones, r_place_v)

        r_reach_w = reach_reward(right_tcp, weight_pos, scale=5.0)
        r_reach_w = torch.where(is_w_grasped | w_on | success, ones, r_reach_w)
        r_grasp_w = grasp_reward(right_tcp, weight_pos, is_w_grasped, proximity_scale=5.0)
        r_grasp_w = torch.where(w_on | success, ones, r_grasp_w)
        r_transport_w = transport_reward(weight_pos, w_goal, is_w_grasped, scale=5.0)
        r_transport_w = torch.where(w_on | success, ones, r_transport_w)
        r_place_w = w_on.float()
        r_place_w = torch.where(success, ones, r_place_w)

        for name, val in [("reach_v",r_reach_v),("grasp_v",r_grasp_v),("transport_v",r_transport_v),("place_v",r_place_v),("reach_w",r_reach_w),("grasp_w",r_grasp_w),("transport_w",r_transport_w),("place_w",r_place_w)]:
            self.reward_tracker.update(name, val)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)