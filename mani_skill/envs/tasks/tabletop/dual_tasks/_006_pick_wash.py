import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, List, Tuple

from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    is_l2_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    reach_reward,
    grasp_reward,
    transport_reward,
)


REWARD_PHASES = ["reach", "grasp", "transport", "place"]


@register_env("TwoRobotPickWash-v1", max_episode_steps=100)
class TwoRobotPickWashEnv(BaseEnv):
    """
    **Task Description:**
    Recreation of a specific scene with 3 bottles/objects aligned on a table between two robots.
    This environment focuses on scene construction for data collection.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

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
        self.bottle_modelname = "049_shampoo"
        self.bottle_model_ids = [3, 1, 6]
        if is_l2_enabled():
            self.bottle_model_ids[1] = 4
        self.bottle_scale = None
        self.bottle_replace_scale = False
        self.bottle_mass = 0.5
        self.bottle_init_quat = euler2quat(np.pi / 2, 0.0, 0.0)
        
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

        self.bottles: List[Actor] = []
        bottle_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=self.bottle_init_quat)
        for i, bottle_model_id in enumerate(self.bottle_model_ids):
            bottle_obj = create_actor(
                scene=self.scene,
                pose=bottle_pose,
                modelname=self.bottle_modelname,
                convex=True,
                model_id=bottle_model_id,
                scale=self.bottle_scale,
                replace_scale=self.bottle_replace_scale,
                mass=self.bottle_mass,
                _idx_if_repeat=i,
            )
            self.bottles.append(bottle_obj.actor)

    def _after_reconfigure(self, options: dict):
        # Compute each object's physical height so it rests flush on the table
        self.bottle_zs = []
        for obj in self.bottles:
            collision_mesh = obj.get_first_collision_mesh()
            self.bottle_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.bottle_zs = common.to_tensor(self.bottle_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # [Key Change] Set the positions of the three objects
            # Our robots are located at Y=-1 (left) and Y=1 (right)
            # So the three bottles should be arranged along the Y axis
            
            # Define the target positions [x, y] of the three objects
            # Object 1 (left): Y = -0.25 (near the left arm)
            # Object 2 (middle): Y = 0.0   (center)
            # Object 3 (right): Y = 0.25  (near the right arm)
            base_positions = torch.tensor([
                [-0.1, -0.2], 
                [-0.1, -0.4], 
                [-0.1, 0.1]
            ], device=self.device)

            for i, bottle in enumerate(self.bottles):
                # Base position
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, :2] = base_positions[i]
                
                # Add slight random noise (simulating real placement error)
                # Range ±2cm
                xyz[:, :2] += (torch.rand((b, 2), device=self.device) * 0.04 - 0.02)
                
                # Set the height Z
                xyz[:, 2] = self.bottle_zs[i]

                if i == 1:
                    xyz = apply_l1_offset_xy(xyz,  offset=(-0.10, -0.10))
                
                # Random rotation (around the Z axis only)
                #qs = random_quaternions(b, lock_x=True, lock_y=True)
                qs = torch.tensor(
                    [self.bottle_init_quat] * b,
                    dtype=torch.float32,
                    device=self.device,
                )
                bottle.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

            # Initialize the robots
            # This is usually handled by the parent class or via agent reset, but may need
            # to be triggered manually in a custom init.
            # If you use the standard ManiSkill flow, BaseEnv handles agent reset;
            # here we only need to handle the objects.

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
            obj_name = getattr(obj, "name", obj.__class__.__name__)
            if obj_name not in self._geom_center_fallback_logged:
                print(
                    f"[{self.__class__.__name__}] geom center fallback to pose.p for {obj_name}: {exc}"
                )
                self._geom_center_fallback_logged.add(obj_name)
            return obj.pose.p.clone()

    def evaluate(self):
        goal_thresh = 0.08

        small_left_pos = self.bottles[0].pose.p
        big_pos = self.bottles[1].pose.p
        small_right_pos = self.bottles[2].pose.p
        mid_xy = (small_left_pos[:, :2] + small_right_pos[:, :2]) * 0.5
        dist_to_mid = torch.linalg.norm(big_pos[:, :2] - mid_xy, axis=1)
        is_obj_placed = dist_to_mid <= goal_thresh
        result = dict(
            success=is_obj_placed,
            is_obj_placed=is_obj_placed,
            dist_to_mid=dist_to_mid,
            is_grasped=self.left_agent.is_grasping(self.bottles[1]),
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        # Return pose information for all bottles
        obs = dict()
        if "state" in self.obs_mode:
            for i, bottle in enumerate(self.bottles):
                obs[f"bottle_{i}_pose"] = bottle.pose.raw_pose
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        source_obj = self.bottles[1]
        source_pos = self._get_actor_geom_center(source_obj)
        other_pos_0 = self._get_actor_geom_center(self.bottles[0])
        other_pos_2 = self._get_actor_geom_center(self.bottles[2])
        target_pos = (other_pos_0 + other_pos_2) * 0.5
        target_pos[:, 2] = source_pos[:, 2]

        tcp_pos = self.left_agent.tcp.pose.p
        obj_pos = source_pos
        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, obj_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, obj_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_obj_placed | success, ones, r_grasp)
        r_transport = transport_reward(obj_pos, target_pos, is_grasped, scale=5.0)
        r_transport = torch.where(is_obj_placed | success, ones, r_transport)
        r_place = is_obj_placed.float()
        r_place = torch.where(success, ones, r_place)

        self.reward_tracker.update("reach", r_reach)
        self.reward_tracker.update("grasp", r_grasp)
        self.reward_tracker.update("transport", r_transport)
        self.reward_tracker.update("place", r_place)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action, info)
