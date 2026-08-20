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
    is_l2_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
    exp_reward,
)

# Sub-task phases for this task
REWARD_PHASES = ["reach_brush", "grasp_brush", "transport_brush", "place_brush"]


@register_env("TwoRobotPlaceBrushRest-v1", max_episode_steps=100)
class TwoRobotPlaceBrushRestEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.1

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
        self.brush_pen_modelname, self.brush_pen_model_id = apply_l2_robotwin_model(
            "093_brush-pen",
            model_id=5,
            override_name="093_brush-pen",
            override_id=1,
        )
        self.book_modelname, self.book_model_id = "043_book", 1

        self.rest_modelname, self.rest_model_id = "094_rest", 1
        
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
        brush_q = euler2quat(np.pi / 2, 0, -np.pi / 2)
        brush_scale = (0.25, 0.25, 0.25)
        if is_l2_enabled() and self.brush_pen_model_id == 1:
            extra_q = euler2quat(0, np.pi / 10, np.pi/2)
            brush_q = (sapien.Pose(q=extra_q) * sapien.Pose(q=brush_q)).q
            brush_scale = (0.15, 0.15, 0.15)
        self.brush_pose = sapien.Pose(
            p=[-0., -0., 0.02],
            q=brush_q
        )
        brush_actor_obj = create_actor(
            scene=self.scene,
            pose=self.brush_pose,
            modelname=self.brush_pen_modelname,
            convex=True,
            model_id=self.brush_pen_model_id,
            scale=brush_scale,
            replace_scale=True,
        )
        brush_actor_obj.set_mass(0.05)
        self.brush = brush_actor_obj.actor

        # Load book
        self.book_pose = sapien.Pose(
            p=[0.0, 0.15, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi)
        )
        book_actor_obj = create_actor(
            scene=self.scene,
            pose=self.book_pose,
            modelname=self.book_modelname,
            convex=True,
            model_id=self.book_model_id,
            is_static=True,
            scale=(1.2, 1.2, 1.2),
        )
        self.book = book_actor_obj.actor

        # Load rest
        self.rest_pose = sapien.Pose(
            p=[0.1, -0.35, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        rest_actor_obj = create_actor(
            scene=self.scene,
            pose=self.rest_pose,
            modelname=self.rest_modelname,
            convex=True,
            model_id=self.rest_model_id,
            is_static=True,
            # replace_scale=True,
            scale=(1.3, 1.3, 1.3),
        )
        self.rest = rest_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize reward tracker
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            xyz = torch.tensor(self.brush_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.1))
            qs = torch.tensor(self.brush_pose.q).repeat(b, 1)
            self.brush.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.book_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            qs = torch.tensor(self.book_pose.q).repeat(b, 1)
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.1))
            self.book.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.rest_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.rest_zs[env_idx]
            qs = torch.tensor(self.rest_pose.q).repeat(b, 1)
            self.rest.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.brush_zs = []
        collision_mesh = self.brush.get_first_collision_mesh()
        self.brush_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.brush_zs = common.to_tensor(self.brush_zs, device=self.device)

        self.book_zs = []
        collision_mesh = self.book.get_first_collision_mesh()
        self.book_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.book_zs = common.to_tensor(self.book_zs, device=self.device)

        self.rest_zs = []
        collision_mesh = self.rest.get_first_collision_mesh()
        self.rest_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.rest_zs = common.to_tensor(self.rest_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        obj_to_goal_pos = self.brush.pose.p - self.rest.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_grasped = self.left_agent.is_grasping(self.brush) or self.right_agent.is_grasping(self.brush)
        is_robot_static = self.left_agent.is_static(0.2) and self.right_agent.is_static(0.2)

        result = dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=self.left_agent.is_grasping(self.brush) or self.right_agent.is_grasping(self.brush),
            success=is_obj_placed,
        )
        # Append per-phase peak sub-rewards
        # if hasattr(self, "reward_tracker"):
        #     # result.update(self.reward_tracker.get_peak_dict())
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
                obj_to_goal_pos=self.brush.pose.p - self.rest.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        brush_pos = self.brush.pose.p
        rest_pos = self.rest.pose.p
        rest_pos = rest_pos + torch.tensor([-0.05, 0, 0], device=self.device)  

        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]

        # Phase 1: REACH_BRUSH - only before grasping
        reach_gate = ~is_grasped
        r_reach_brush = reach_reward(left_tcp, brush_pos, scale=5.0) * reach_gate.float()
        r_reach_brush = torch.where(is_grasped, torch.ones_like(r_reach_brush), r_reach_brush)
        self.reward_tracker.update("reach_brush", r_reach_brush)

        # Phase 2: GRASP_BRUSH - only after reaching, before placed
        grasp_gate = is_grasped
        r_grasp_brush = grasp_reward(left_tcp, brush_pos, is_grasped, proximity_scale=5.0) * grasp_gate.float()
        self.reward_tracker.update("grasp_brush", r_grasp_brush)

        # Phase 3: TRANSPORT_BRUSH - only after grasping, before placed
        transport_gate = is_grasped
        r_transport_brush = transport_reward(brush_pos[:, :2], rest_pos[:, :2], is_grasped, scale=5.0) * transport_gate.float()
        r_transport_brush = torch.where(is_obj_placed, torch.ones_like(r_transport_brush), r_transport_brush)
        self.reward_tracker.update("transport_brush", r_transport_brush)

        # Phase 4: PLACE_BRUSH - continuous reward (after transport started and released)
        transport_peak = self.reward_tracker._peaks["transport_brush"]
        brush_to_rest_dist = torch.linalg.norm(brush_pos[:, :2] - rest_pos[:, :2], dim=-1)
        r_place_brush = exp_reward(brush_to_rest_dist, scale=5.0) * (transport_peak > 0).float() * (~is_grasped).float()
        r_place_brush = torch.where(is_obj_placed, torch.ones_like(r_place_brush), r_place_brush)
        self.reward_tracker.update("place_brush", r_place_brush)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
