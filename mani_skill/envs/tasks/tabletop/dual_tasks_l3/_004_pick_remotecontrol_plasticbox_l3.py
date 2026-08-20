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
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    reach_reward,
    grasp_reward,
    transport_reward,
    normalized_progress,
    world_aabb_from_local_mesh,
)


REWARD_PHASES = ["reach", "grasp", "lift", "transport", "place"]


@register_env("TwoRobotPickRemoteControlL3-v1", max_episode_steps=100)
class TwoRobotPickRemoteControlEnvL3(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a remotecontrol and place it into a plasticbox container. There are two robots in this task,
    both positioned side-by-side in front of the table (similar to PickCube layout):
    - Left robot at position [-0.9, 0, 0]
    - Right robot at position [-0.3, 0, 0]
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    goal_thresh = 0.05
    container_z_tolerance = 0.1

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
        # Fixed asset config (no level-switch logic).
        self.remotecontrol_modelname = "079_remotecontrol"
        self.remotecontrol_model_id = get_model_id(self.remotecontrol_modelname, model_id=0)
        self.remotecontrol_scale = None
        self.remotecontrol_replace_scale = False
        self.remotecontrol_mass = 0.1
        self.plasticbox_modelname = "062_plasticbox"
        self.plasticbox_model_id = get_model_id(self.plasticbox_modelname, model_id=7)
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

        # Load remotecontrol (object to grasp and move) - on the right side
        remotecontrol_pose = sapien.Pose(
            p=[0, 0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        remotecontrol_obj = create_actor(
            scene=self.scene,
            pose=remotecontrol_pose,
            modelname=self.remotecontrol_modelname,
            convex=True,
            model_id=self.remotecontrol_model_id,
            scale=self.remotecontrol_scale,
            replace_scale=self.remotecontrol_replace_scale,
            mass=self.remotecontrol_mass, 
        )
        self.remotecontrol = remotecontrol_obj.actor

        # Load plasticbox (goal container) - on the left side
        plasticbox_pose = sapien.Pose(
            p=[0, -0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        plasticbox_obj = create_actor(
            scene=self.scene,
            pose=plasticbox_pose,
            modelname=self.plasticbox_modelname,
            convex=True,
            model_id=self.plasticbox_model_id,
            is_static=True
        )
        self.plasticbox = plasticbox_obj.actor

        # Load displaystand
        displaystand_pose = sapien.Pose(
            p=[-0.1, -0.4, 0.0],
            q=euler2quat(np.pi / 2, 0.0, -np.pi / 2),
        )
        displaystand_obj = create_actor(
            scene=self.scene,
            pose=displaystand_pose,
            modelname="074_displaystand",
            convex=True,
            model_id=2,
            scale=(0.06, 0.06, 0.06),
            replace_scale=True,
            is_static=True,
        )
        self.displaystand = displaystand_obj.actor

    #     self.cube = self._build_cube(half_size=0.02)

    # def _build_cube(self, half_size):
    #     builder = self.scene.create_actor_builder()
    #     builder.add_box_collision(half_size=(half_size, half_size, half_size), density=1000)
    #     builder.add_box_visual(half_size=(half_size, half_size, half_size), material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]))
    #     cube = builder.build(name="cube")
    #     return cube

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # self.left_agent.robot.set_pose(sapien.Pose(p=[-0.60, -0.35, 0]))
            # self.right_agent.robot.set_pose(sapien.Pose(p=[-0.60, 0.35, 0]))

            # Initialize remotecontrol (object to grasp) - with randomization
            displaystand_xyz = torch.zeros((b, 3), device=self.device)
            displaystand_xyz[:, 0] = 0.0
            displaystand_xyz[:, 1] = -0.2
            displaystand_xyz[:, 2] = self.displaystand_z
            displaystand_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            displaystand_q = torch.tensor(
                [euler2quat(np.pi / 2, 0.0, -np.pi / 2)] * b,
                device=self.device,
                dtype=torch.float32,
            )
            self.displaystand.set_pose(Pose.create_from_pq(p=displaystand_xyz, q=displaystand_q))

            displaystand_bounds = world_aabb_from_local_mesh(self.displaystand, self.device)
            displaystand_top_z = displaystand_bounds[:, 1, 2]
            displaystand_center_x = (displaystand_bounds[:, 0, 0] + displaystand_bounds[:, 1, 0]) * 0.5
            displaystand_center_y = (displaystand_bounds[:, 0, 1] + displaystand_bounds[:, 1, 1]) * 0.5

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = displaystand_center_x - 0.05
            xyz[:, 1] = displaystand_center_y
            xyz[:, 2] = displaystand_top_z + self.remotecontrol_z + 0.002
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.01
            # Fixed upright orientation with horizontal rotation (no Z rotation randomization)
            # euler2quat(roll, pitch, yaw) - modify the third parameter (yaw/Z-axis) to change horizontal rotation
            z_rotation = np.pi / 6
            base_quat = euler2quat(np.pi/2, 0.0, z_rotation)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.remotecontrol.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Initialize plasticbox (goal container) - with independent randomization
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = 0.0  # X position
            xyz[:, 1] = 0.1  # Y position - left side
            xyz[:, 2] = self.plasticbox_z  # Z position (on table surface)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02  # Per-env random offset
            base_quat = euler2quat(np.pi/2, 0.0, 0.0)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.plasticbox.set_pose(Pose.create_from_pq(xyz, qs))
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

            # # Initialize cube - additional object
            # xyz = torch.zeros((b, 3), device=self.device)
            # xyz[:, 0] = 0.0  # X position
            # xyz[:, 1] = 0.3  # Y position - right side
            # xyz[:, 2] = self.remotecontrol_z  # Z position (same as remotecontrol)
            # base_quat = euler2quat(0.0, 0.0, 0.0)
            # qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            # self.cube.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):
        # Get z-offset for remotecontrol to place it on table surface
        collision_mesh = self.remotecontrol.get_first_collision_mesh()
        if collision_mesh is not None:
            self.remotecontrol_z = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            self.remotecontrol_z = 0.02  # Default height if no collision mesh

        # Get z-offset for plasticbox to place it on table surface
        collision_mesh = self.plasticbox.get_first_collision_mesh()
        if collision_mesh is not None:
            self.plasticbox_z = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            self.plasticbox_z = 0.05  # Default height if no collision mesh

        # Get z-offset for displaystand to place it on table surface
        collision_mesh = self.displaystand.get_first_collision_mesh()
        if collision_mesh is not None:
            self.displaystand_z = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            self.displaystand_z = 0.05

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
        remotecontrol_to_plasticbox_pos = self.plasticbox.pose.p - self.remotecontrol.pose.p
        plasticbox_bounds = world_aabb_from_local_mesh(self.plasticbox, self.device)
        remotecontrol_center = self._get_actor_geom_center(self.remotecontrol)
        is_obj_placed = self._is_in_container_bbox(remotecontrol_center, plasticbox_bounds)
        is_grasped = self.left_agent.is_grasping(self.remotecontrol)
        is_robot_static = self.left_agent.is_static(0.2)
        result = dict(
            is_grasped=is_grasped,
            remotecontrol_to_plasticbox_pos=remotecontrol_to_plasticbox_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=self.left_agent.is_grasping(self.remotecontrol),
            success=is_obj_placed,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.left_agent.tcp.pose.raw_pose,
            goal_pos=self.plasticbox.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                remotecontrol_pose=self.remotecontrol.pose.raw_pose,
                tcp_to_remotecontrol_pos=self.remotecontrol.pose.p - self.left_agent.tcp.pose.p,
                plasticbox_pose=self.plasticbox.pose.raw_pose,
                tcp_to_plasticbox_pos=self.plasticbox.pose.p - self.left_agent.tcp.pose.p,
                remotecontrol_to_plasticbox_pos=self.plasticbox.pose.p - self.remotecontrol.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        obj_pos = self._get_actor_geom_center(self.remotecontrol)
        goal_pos = self._get_actor_geom_center(self.plasticbox)
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, obj_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, obj_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_obj_placed | success, ones, r_grasp)
        lift_height = obj_pos[:, 2] - self.remotecontrol_z
        r_lift = normalized_progress(lift_height, start=0.01, target=0.12) * is_grasped.float()
        r_lift = torch.where(is_obj_placed | success, ones, r_lift)
        r_transport = transport_reward(obj_pos, goal_pos, is_grasped, scale=5.0)
        r_transport = torch.where(is_obj_placed | success, ones, r_transport)
        r_place = is_obj_placed.float()
        r_place = torch.where(success, ones, r_place)

        self.reward_tracker.update("reach", r_reach)
        self.reward_tracker.update("grasp", r_grasp)
        self.reward_tracker.update("lift", r_lift)
        self.reward_tracker.update("transport", r_transport)
        self.reward_tracker.update("place", r_place)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
