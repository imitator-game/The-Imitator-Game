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
from mani_skill.utils.building.actors.robotwin import get_model_id
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_robotwin_model
)

from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
    exp_reward,
)

# Sub-task phases for this task
REWARD_PHASES = [
    "reach_pill_a", "grasp_pill_a", "transport_pill_a", "place_pill_a",
    "reach_pill_b", "grasp_pill_b", "transport_pill_b", "place_pill_b",
]

@register_env("TwoRobotPickPillToRegionsL3-v1", max_episode_steps=200)
class TwoRobotPickPillToRegionsEnvL3(BaseEnv):
    """
    **Task Description:**
    Two panda_wristcam robots operate on a table.
    1) Pick pill A and place it onto the display stand.
    2) Pick pill B and place it onto the display stand.

    **Success Conditions:**
    - Pill A is on the display stand.
    - Pill B is on the display stand.
    - Both robots are static.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]

    PILL_MODEL_NAME = "080_pillbottle"

    pill_a_xy = (-0.25, -0.15)
    pill_b_xy = (-0.25, 0.15)

    pill_scale = 1.0

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
        self.pill_a_modelname, self.pill_a_model_id = apply_l2_robotwin_model(
            self.PILL_MODEL_NAME,
            model_id=1,
            override_name=self.PILL_MODEL_NAME,
            override_id=2,
        )
        self.pill_b_modelname, self.pill_b_model_id = apply_l2_robotwin_model(
            self.PILL_MODEL_NAME,
            model_id=5,
            override_name=self.PILL_MODEL_NAME,
            override_id=4,
        )

        # Display Stands (robotwin objects, sharing XY with manual Z offsets)
        self.displaystand_modelname = "074_displaystand"
        self.displaystand_bottom_model_id = 4  # Bottom display stand
        self.displaystand_top_model_id = 3
        self.displaystand_top_scale_factor = 2.0
        # Shared XY placement and simple manual Z offsets (adjust if you want more/less separation)
        self.displaystand_xy = (0, 0)
        self.displaystand_bottom_offset = -0.01
        self.displaystand_top_offset = -0.01

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )
    
    def _create_displaystand(self, model_id: int, actor_name: str, scale_factor: float = 2, is_static: bool = False):
        """Helper function to create a display stand actor with unique name"""
        from pathlib import Path
        import json
        from mani_skill.utils.structs.robotwin_actor import Actor

        def get_glb_or_obj_file(modeldir, model_id):
            modeldir = Path(modeldir)
            file = modeldir / f"base{model_id}.glb" if model_id is not None else modeldir / "base.glb"
            if not file.exists():
                file = modeldir / f"textured{model_id}.obj" if model_id is not None else modeldir / "textured.obj"
            return file

        from mani_skill import ASSET_DIR
        modeldir = ASSET_DIR / "robotwin" / "objects" / self.displaystand_modelname
        json_file_path = modeldir / f"model_data{model_id}.json"

        # Get collision and visual file paths
        collision_file = get_glb_or_obj_file(modeldir / "collision", model_id) if (modeldir / "collision").exists() else get_glb_or_obj_file(modeldir, model_id)
        visual_file = get_glb_or_obj_file(modeldir / "visual", model_id) if (modeldir / "visual").exists() else get_glb_or_obj_file(modeldir, model_id)

        # Load model data and calculate scale
        with open(json_file_path, "r") as file:
            model_data = json.load(file)

        base_scale = model_data.get('scale', (0.08, 0.08, 0.08))
        final_scale = tuple(s * scale_factor for s in base_scale)

        # Build actor
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static" if is_static else "dynamic")
        builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=final_scale)
        builder.add_visual_from_file(filename=str(visual_file), scale=final_scale)

        quat = euler2quat(np.pi/2, 0.0, 0.0)
        pose = sapien.Pose(p=[0, -0.5, 0], q=quat.tolist())
        mesh = builder.build(name=actor_name)
        mesh.set_pose(pose)

        return Actor(mesh, model_data).actor

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

        self.pill_a_pose = sapien.Pose(
            p=[self.pill_a_xy[0], self.pill_a_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0.0, 0.0),
        )
        self.pill_b_pose = sapien.Pose(
            p=[self.pill_b_xy[0], self.pill_b_xy[1], 0.0],
            q=euler2quat(np.pi / 2, 0.0, 0.0),
        )

        self.pill_a = create_actor(
            scene=self.scene,
            pose=self.pill_a_pose,
            modelname=self.pill_a_modelname,
            convex=True,
            model_id=get_model_id(self.pill_a_modelname, model_id=self.pill_a_model_id),
            scale=(self.pill_scale,) * 3,
            _idx_if_repeat=1,
            mass=0.5,
        ).actor
        self.pill_b = create_actor(
            scene=self.scene,
            pose=self.pill_b_pose,
            modelname=self.pill_b_modelname,
            convex=True,
            model_id=get_model_id(self.pill_b_modelname, model_id=self.pill_b_model_id),
            scale=(self.pill_scale,) * 3,
            _idx_if_repeat=2,
            mass=0.5,
        ).actor
        
        self.pill_a.set_mass(0.5)
        self.pill_b.set_mass(0.5)

        # Load two display stands with unique names (as static objects)
        self.displaystand_bottom = self._create_displaystand(
            model_id=self.displaystand_bottom_model_id,
            actor_name="displaystand_bottom",
            scale_factor=2.5,
            is_static=True
        )
        self.displaystand_top = self._create_displaystand(
            model_id=self.displaystand_top_model_id,
            actor_name="displaystand_top",
            scale_factor=self.displaystand_top_scale_factor,
            is_static=True
        )

    def _after_reconfigure(self, options: dict):
        def _compute_object_z(obj) -> float:
            collision_mesh = obj.get_first_collision_mesh()
            return -collision_mesh.bounding_box.bounds[0, 2]

        self._pill_a_z = common.to_tensor([_compute_object_z(self.pill_a)], device=self.device)
        self._pill_b_z = common.to_tensor([_compute_object_z(self.pill_b)], device=self.device)

        # Get z-offset for bottom display stand to place it on table surface
        collision_mesh = self.displaystand_bottom.get_first_collision_mesh()
        if collision_mesh is not None:
            self.displaystand_bottom_z = -collision_mesh.bounding_box.bounds[0, 2] + self.displaystand_bottom_offset
            self.displaystand_bottom_height = (
                collision_mesh.bounding_box.bounds[1, 2] - collision_mesh.bounding_box.bounds[0, 2]
            )
        else:
            self.displaystand_bottom_z = 0.02 + self.displaystand_bottom_offset  # Default height if no collision mesh
            self.displaystand_bottom_height = 0.1

        # Get z-offset for top display stand (stacked on top of bottom stand)
        collision_mesh = self.displaystand_top.get_first_collision_mesh()
        if collision_mesh is not None:
            top_stand_bottom_offset = -collision_mesh.bounding_box.bounds[0, 2]
            self.displaystand_top_height = (
                collision_mesh.bounding_box.bounds[1, 2] - collision_mesh.bounding_box.bounds[0, 2]
            )
        else:
            top_stand_bottom_offset = 0.02
            self.displaystand_top_height = 0.1

        # Stack top on bottom: bottom's z + bottom's height + top's bottom offset + manual offset
        self.displaystand_top_z = (
            self.displaystand_bottom_z
            + self.displaystand_bottom_height
            + top_stand_bottom_offset
            + self.displaystand_top_offset
        )

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

            # Initialize reward tracker
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            pill_a_p = torch.tensor(self.pill_a_pose.p).repeat(b, 1)
            pill_a_p[:, :2] += (torch.rand((b, 2)) - 0.5) * torch.tensor([0.02, 0.02])
            pill_a_p[:, 2] = self._pill_a_z[0]
            pill_a_p = apply_l1_offset_xy(pill_a_p, offset=(0.05, -0.1))
            base_quat = euler2quat(np.pi / 2, 0.0, 0.0)
            pill_a_q = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.pill_a.set_pose(Pose.create_from_pq(p=pill_a_p, q=pill_a_q))

            pill_b_p = torch.tensor(self.pill_b_pose.p).repeat(b, 1)
            pill_b_p[:, :2] += (torch.rand((b, 2)) - 0.5) * torch.tensor([0.02, 0.02])
            pill_b_p[:, 2] = self._pill_b_z[0]
            pill_b_p = apply_l1_offset_xy(pill_b_p, offset=(0.05, 0.1))
            pill_b_q = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.pill_b.set_pose(Pose.create_from_pq(p=pill_b_p, q=pill_b_q))

            # Define shared base position for both display stands
            displaystand_base_xy = torch.zeros((b, 2), device=self.device)
            displaystand_base_xy[:, 0] = self.displaystand_xy[0]  # X position
            displaystand_base_xy[:, 1] = self.displaystand_xy[1]  # Y position
            displaystand_base_xy[:, :2] += torch.rand((b, 2)) * 0.02

            # Initialize bottom display stand using base position
            displaystand_bottom_xyz = torch.zeros((b, 3), device=self.device)
            displaystand_bottom_xyz[:, 0] = displaystand_base_xy[:, 0]
            displaystand_bottom_xyz[:, 1] = displaystand_base_xy[:, 1]
            displaystand_bottom_xyz[:, 2] = self.displaystand_bottom_z  # Z position (on table surface)

            # Fixed upright orientation
            base_quat = euler2quat(np.pi/2, 0.0, np.pi/2)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.displaystand_bottom.set_pose(Pose.create_from_pq(displaystand_bottom_xyz, qs))

            # Initialize top display stand - shares XY with bottom, manual Z offset
            displaystand_top_xyz = torch.zeros((b, 3), device=self.device)
            displaystand_top_xyz[:, 0] = displaystand_base_xy[:, 0] + 0.05
            displaystand_top_xyz[:, 1] = displaystand_base_xy[:, 1]
            displaystand_top_xyz[:, 2] = self.displaystand_top_z  # Manual Z offset (see _after_reconfigure)

            # Fixed upright orientation (same as bottom)
            self.displaystand_top.set_pose(Pose.create_from_pq(displaystand_top_xyz, qs))

    def evaluate(self):

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        # Check if jam jar is grasped by left robot
        is_pill_a_grasped = self.left_agent.is_grasping(self.pill_a)

        # Check if milk box is grasped by right robot
        is_pill_b_grasped = self.right_agent.is_grasping(self.pill_b)

        # Check if both robots are static
        is_left_static = self.left_agent.is_static(0.2)
        is_right_static = self.right_agent.is_static(0.2)

        # Check orientation: Local Z-axis should point towards World Z-axis [0, 0, 1]
        # In ManiSkill, common.quaternion_apply can transform local vectors to world
        unit_z = torch.tensor([0, 0, 1.0], device=self.device)
        
        # Some Axis are not aligned with the world Z-axis

        # Check if both items are above table (not dropped)
        is_pill_a_above_table = self.pill_a.pose.p[:, 2] > 0.1
        is_pill_b_above_table = self.pill_b.pose.p[:, 2] > 0.1

        pill_a_to_stand_dist = torch.linalg.norm(
            self.pill_a.pose.p - (sapien.Pose(self.displaystand_top.pose.sp.p) * sapien.Pose([0.0, -0.1, 0.0])).p, axis=1
        )
        pill_b_to_stand_dist = torch.linalg.norm(
            self.pill_b.pose.p - (sapien.Pose(self.displaystand_top.pose.sp.p) * sapien.Pose([0.0, 0.1, 0.0])).p, axis=1
        )
        is_pill_a_on_stand = pill_a_to_stand_dist < 0.20
        is_pill_b_on_stand = pill_b_to_stand_dist < 0.20

        success = is_pill_a_above_table & is_pill_b_above_table & is_pill_a_on_stand & is_pill_b_on_stand

        result = dict(
            is_pill_a_grasped=is_pill_a_grasped,
            is_pill_b_grasped=is_pill_b_grasped,
            is_pill_a_above_table=is_pill_a_above_table,
            is_pill_b_above_table=is_pill_b_above_table,
            is_pill_a_on_stand=is_pill_a_on_stand,
            is_pill_b_on_stand=is_pill_b_on_stand,
            is_left_static=is_left_static,
            is_right_static=is_right_static,
            success=success,
            # For reward computation
            pill_a_to_stand_dist=pill_a_to_stand_dist,
            pill_b_to_stand_dist=pill_b_to_stand_dist,
        )
        # Append per-phase peak sub-rewards
        # if hasattr(self, "reward_tracker"):
        #     # result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            pill_a_pose=self.pill_a.pose.raw_pose,
            pill_b_pose=self.pill_b.pose.raw_pose,
            is_pill_a_grasped=info["is_pill_a_grasped"],
            is_pill_b_grasped=info["is_pill_b_grasped"],
            is_pill_a_above_table=info["is_pill_a_above_table"],
            is_pill_b_above_table=info["is_pill_b_above_table"],
            is_pill_a_on_stand=info["is_pill_a_on_stand"],
            is_pill_b_on_stand=info["is_pill_b_on_stand"],
            is_left_static=info["is_left_static"],
            is_right_static=info["is_right_static"],
            success=info["success"],
            # is_pill_a_upright=info["is_pill_a_upright"],
            # is_pill_b_upright=info["is_pill_b_upright"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        pill_a_pos = self.pill_a.pose.p
        pill_b_pos = self.pill_b.pose.p
        # Target is the display stand (approximate center)
        stand_pos = self.displaystand_top.pose.p
        stand_pos_pill_a = self.displaystand_top.pose.p + torch.tensor([0.0, -0.1, 0.0], device=self.device)
        stand_pos_pill_b = self.displaystand_top.pose.p + torch.tensor([0.0, 0.1, 0.0], device=self.device)

        is_pill_a_grasped = info["is_pill_a_grasped"]
        is_pill_b_grasped = info["is_pill_b_grasped"]
        is_pill_a_on_stand = info["is_pill_a_on_stand"]
        is_pill_b_on_stand = info["is_pill_b_on_stand"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Left arm: pill_a - with proper sequential gating
        # Phase 1: REACH - only before grasping
        reach_gate_a = ~is_pill_a_grasped
        r_reach_pill_a = reach_reward(left_tcp[:, :2], pill_a_pos[:, :2], scale=5.0) * reach_gate_a.float()
        r_reach_pill_a = torch.where(is_pill_a_grasped, ones, r_reach_pill_a)
        self.reward_tracker.update("reach_pill_a", r_reach_pill_a)

        # Phase 2: GRASP - only after reaching (grasped), before on stand
        grasp_gate_a = is_pill_a_grasped 
        r_grasp_pill_a = grasp_reward(left_tcp[:, :2], pill_a_pos[:, :2], is_pill_a_grasped, proximity_scale=5.0) * grasp_gate_a.float()
        self.reward_tracker.update("grasp_pill_a", r_grasp_pill_a)

        # Phase 3: TRANSPORT - only after grasping, before on stand
        transport_gate_a = is_pill_a_grasped 
        r_transport_pill_a = transport_reward(pill_a_pos[:, :2], stand_pos_pill_a[:, :2], is_pill_a_grasped, scale=5.0) * transport_gate_a.float()
        r_transport_pill_a = torch.where(r_transport_pill_a>0.99, ones, r_transport_pill_a)
        self.reward_tracker.update("transport_pill_a", r_transport_pill_a)

        # Phase 4: PLACE - continuous reward based on distance to stand (after transport started and released)
        transport_peak_a = self.reward_tracker._peaks["transport_pill_a"]
        pill_a_to_stand_dist = torch.linalg.norm(pill_a_pos[:, :2] - stand_pos_pill_a[:, :2], dim=-1)
        r_place_pill_a = exp_reward(pill_a_to_stand_dist, scale=5.0) * (transport_peak_a > 0).float() * (~is_pill_a_grasped).float()
        r_place_pill_a = torch.where(r_place_pill_a>0.99, ones, r_place_pill_a)
        self.reward_tracker.update("place_pill_a", r_place_pill_a)

        # Right arm: pill_b - with proper sequential gating
        # Phase 1: REACH - only before grasping
        reach_gate_b = ~is_pill_b_grasped
        r_reach_pill_b = reach_reward(right_tcp[:, :2], pill_b_pos[:, :2], scale=5.0) * reach_gate_b.float()
        r_reach_pill_b = torch.where(is_pill_b_grasped, ones, r_reach_pill_b)
        self.reward_tracker.update("reach_pill_b", r_reach_pill_b)

        # Phase 2: GRASP - only after reaching (grasped), before on stand
        grasp_gate_b = is_pill_b_grasped 
        r_grasp_pill_b = grasp_reward(right_tcp[:, :2], pill_b_pos[:, :2], is_pill_b_grasped, proximity_scale=5.0) * grasp_gate_b.float()
        self.reward_tracker.update("grasp_pill_b", r_grasp_pill_b)

        # Phase 3: TRANSPORT - only after grasping, before on stand
        transport_gate_b = is_pill_b_grasped 
        r_transport_pill_b = transport_reward(pill_b_pos[:, :2], stand_pos_pill_b[:, :2], is_pill_b_grasped, scale=5.0) * transport_gate_b.float()
        r_transport_pill_b = torch.where(r_transport_pill_b>0.99, ones, r_transport_pill_b)
        self.reward_tracker.update("transport_pill_b", r_transport_pill_b)

        # Phase 4: PLACE - continuous reward based on distance to stand (after transport started and released)
        transport_peak_b = self.reward_tracker._peaks["transport_pill_b"]
        pill_b_to_stand_dist = torch.linalg.norm(pill_b_pos[:, :2] - stand_pos_pill_b[:, :2], dim=-1)
        r_place_pill_b = exp_reward(pill_b_to_stand_dist, scale=3.0) * (transport_peak_b > 0).float() * (~is_pill_b_grasped).float()
        r_place_pill_b = torch.where(r_place_pill_b>0.99, ones, r_place_pill_b)
        self.reward_tracker.update("place_pill_b", r_place_pill_b)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
