import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, Tuple
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

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
    apply_l2_quat_offset,
    apply_l2_robotwin_config,
    is_l2_enabled,
    is_l3_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    grasp_reward,
    reach_reward,
    transport_reward,
)


REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotPlacePlateRack-v1", max_episode_steps=100)
class TwoRobotPlacePlateRackEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.15

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
        self.plate_mass = 0.1
        self.l2_plate_x_offset = -0.05
        self.l2_plate_y_offset = 0.1
        self.l2_plate_q_offset = euler2quat(0.0, 0.0, 0.0)
        self.l2_rack_x_offset = -0.05
        self.l2_rack_y_offset = 0.0
        self.l2_rack_q_offset = euler2quat(-np.pi / 6, 0.0, 0.0)
        self.l2_board_x_offset = 0.0
        self.l2_board_y_offset = 0.0
        (
            self.plate_modelname,
            self.plate_model_id,
            self.plate_scale,
            self.plate_replace_scale,
        ) = apply_l2_robotwin_config(
            "003_plate",
            model_id=0,
            base_scale=(1, 1, 1),
            base_replace_scale=False,
            override_name="008_tray",
            override_id=2,
            override_scale=(0.7, 0.7, 0.7),
            override_replace_scale=False,
        )
        (
            self.board_modelname,
            self.board_model_id,
            self.board_scale,
            self.board_replace_scale,
        ) = apply_l2_robotwin_config(
            "104_board",
            model_id=0,
            base_scale=(0.2, 0.7, 0.2),
            base_replace_scale=True,
            override_name="086_woodenblock",
            override_id=4,
            override_scale=(0.7, 0.7, 0.7),
            override_replace_scale=True,
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

    # Task 14 L2 only swaps objects; keep the original non-mirrored layout.
    def _maybe_apply_lr_mirror(self, env_idx: torch.Tensor, options: dict):
        self._lr_mirror_applied_this_reset = False
        return

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Load plate
        self.plate_pose = sapien.Pose(
            p=[-0., -0.3, 0.11],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        plate_actor_obj = create_actor(
            scene=self.scene,
            pose=self.plate_pose,
            modelname=self.plate_modelname,
            convex=True,
            model_id=self.plate_model_id,
            scale=self.plate_scale,
            replace_scale=self.plate_replace_scale,
            mass=self.plate_mass, 
        )
        self.plate = plate_actor_obj.actor

        # Load board
        self.board_pose = sapien.Pose(
            p=[-0.05, -0.2, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi)
        )
        board_actor_obj = create_actor(
            scene=self.scene,
            pose=self.board_pose,
            modelname=self.board_modelname,
            convex=True,
            model_id=self.board_model_id,
            is_static=True,
            replace_scale=self.board_replace_scale,
            scale=self.board_scale,
        )
        self.board = board_actor_obj.actor

        # Load rack
        self.rack_pose = sapien.Pose(
            p=[0.15, 0.1, 0.65],
            q=euler2quat(0, np.pi / 2, np.pi / 2)
        )
        rack_model_id = 1 if is_l3_enabled() else 3
        rack_actor_obj = create_actor(
            scene=self.scene,
            pose=self.rack_pose,
            modelname="013_dumbbell-rack",
            convex=True,
            model_id=rack_model_id,
            is_static=True,
            # replace_scale=True,
            scale=(5, 2, 5),
        )
        self.rack = rack_actor_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    REWARD_PHASES, self.num_envs, self.device
                )
            self.reward_tracker.reset(env_idx)
            l2_plate_x_offset = self.l2_plate_x_offset if is_l2_enabled() else 0.0
            l2_plate_y_offset = self.l2_plate_y_offset if is_l2_enabled() else 0.0
            l2_rack_x_offset = self.l2_rack_x_offset if is_l2_enabled() else 0.0
            l2_rack_y_offset = self.l2_rack_y_offset if is_l2_enabled() else 0.0
            l2_board_x_offset = self.l2_board_x_offset if is_l2_enabled() else 0.0
            l2_board_y_offset = self.l2_board_y_offset if is_l2_enabled() else 0.0

            xyz = torch.tensor(self.plate_pose.p, device=self.device).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.1))
            xyz[:, 0] += l2_plate_x_offset
            xyz[:, 1] += l2_plate_y_offset
            qs = torch.tensor(self.plate_pose.q, device=self.device).repeat(b, 1)
            qs = apply_l2_quat_offset(qs, offset_q=self.l2_plate_q_offset)
            self.plate.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.board_pose.p, device=self.device).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.1))
            xyz[:, 0] += l2_board_x_offset
            xyz[:, 1] += l2_board_y_offset
            qs = torch.tensor(self.board_pose.q, device=self.device).repeat(b, 1)
            self.board.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.rack_pose.p, device=self.device).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz[:, 0] += l2_rack_x_offset
            xyz[:, 1] += l2_rack_y_offset
            xyz[:, 2] = self.rack_zs[env_idx]
            rack_q = np.array(self.rack_pose.q, dtype=np.float32)
            if is_l2_enabled():
                rack_q = qmult(rack_q, np.array(self.l2_rack_q_offset, dtype=np.float32))
            qs = torch.tensor(rack_q, device=self.device).repeat(b, 1)
            self.rack.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.plate_zs = []
        collision_mesh = self.plate.get_first_collision_mesh()
        self.plate_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.plate_zs = common.to_tensor(self.plate_zs, device=self.device)

        self.rack_zs = []
        collision_mesh = self.rack.get_first_collision_mesh()
        self.rack_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.rack_zs = common.to_tensor(self.rack_zs, device=self.device)

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
            name = getattr(obj, "name", type(obj).__name__)
            if name not in self._geom_center_fallback_logged:
                print(
                    f"[{self.__class__.__name__}] geom center fallback to pose.p for {name}: {exc}"
                )
                self._geom_center_fallback_logged.add(name)
            return obj.pose.p.clone()

    def evaluate(self):
        plate_center = self._get_actor_geom_center(self.plate)
        rack_center = self._get_actor_geom_center(self.rack)
        obj_to_goal_pos = plate_center - rack_center
        obj_to_goal_xy = obj_to_goal_pos[:, :2]
        is_obj_placed = torch.linalg.norm(obj_to_goal_xy, axis=1) <= self.goal_thresh
        is_grasped = self.left_agent.is_grasping(self.plate) | self.right_agent.is_grasping(self.plate)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        result = dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            obj_to_goal_xy=obj_to_goal_xy, 
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
            goal_pos=self.plate.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.plate.pose.raw_pose,
                left_tcp_to_obj_pos=self.plate.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.plate.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.plate.pose.p - self.rack.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        plate_pos = self._get_actor_geom_center(self.plate)
        rack_pos = self._get_actor_geom_center(self.rack)
        tcp_pos = self.left_agent.tcp.pose.p
        is_grasped = info["is_grasped"]
        is_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, plate_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, plate_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_placed | success, ones, r_grasp)
        r_transport = transport_reward(plate_pos, rack_pos, is_grasped, scale=4.0)
        r_transport = torch.where(is_placed | success, ones, r_transport)
        r_place = torch.where(success, ones, is_placed.float())

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
