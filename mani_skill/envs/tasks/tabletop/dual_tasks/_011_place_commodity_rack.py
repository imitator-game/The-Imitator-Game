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
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_robotwin_config,
    apply_l3_robotwin_config,
    is_l3_enabled,
)
from mani_skill.utils import common
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    grasp_reward,
    reach_reward,
    transport_reward,
)


REWARD_PHASES = [
    "reach_jamjar",
    "grasp_jamjar",
    "transport_jamjar",
    "place_jamjar",
    "reach_milkbox",
    "grasp_milkbox",
    "transport_milkbox",
    "place_milkbox",
]


@register_env("TwoRobotPlaceCommodityRack-v1", max_episode_steps=200)
class TwoRobotPlaceCommodityRackEnv(BaseEnv):
    """
    **Task Description:**
    The goal is for robots to pick up commodity items (jam jar and milk box) and place them on a display stand.
    There are two display stands sharing the same XY (model_id=4 and model_id=3) with a simple manual Z offset.
    There are two robots in this task:
    - Left robot at position [-0.9, 0, 0] - picks up jam jar and places on top display stand
    - Right robot at position [-0.3, 0, 0] - picks up milk box and places on top display stand

    Both robots face the table.

    **Randomizations:**
    - Object positions have ±2cm XY randomization (same as bookcase task)
    - Objects have fixed upright orientation with X-axis rotation

    **Success Conditions:**
    - Both jam jar and milk box are placed on the top display stand
    - Both robots are static (task completed)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    video_info_whitelist = {
        "reward",
        "success",
        "is_jamjar_grasped",
        "is_milkbox_grasped",
        "is_jamjar_on_stand",
        "is_milkbox_on_stand",
        "peak_r_reach_jamjar",
        "peak_r_grasp_jamjar",
        "peak_r_transport_jamjar",
        "peak_r_place_jamjar",
        "peak_r_reach_milkbox",
        "peak_r_grasp_milkbox",
        "peak_r_transport_milkbox",
        "peak_r_place_milkbox",
    }

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

        # Jam Jar (robotwin object, for left robot to grasp)
        (
            self.jamjar_modelname,
            self.jamjar_model_id,
            self.jamjar_scale,
            self.jamjar_replace_scale,
        ) = apply_l2_robotwin_config(
            "031_jam-jar",
            model_id=0,
            override_name="031_jam-jar",
            override_id=4,
        )

        # Milk Box (robotwin object, for right robot to grasp)
        (
            self.milkbox_modelname,
            self.milkbox_model_id,
            self.milkbox_scale,
            self.milkbox_replace_scale,
        ) = apply_l2_robotwin_config(
            "038_milk-box",
            model_id=0,
            base_scale=(10, 10, 10),
            base_replace_scale=False,
            override_name="038_milk-box",
            override_id=3,
            override_scale=(6, 6, 6),
            override_replace_scale=False,
        )

        # Display Stands (robotwin objects, sharing XY with manual Z offsets)
        self.displaystand_modelname = "074_displaystand"
        self.displaystand_bottom_model_id = 4  # Bottom display stand
        _, self.displaystand_top_model_id, _, _ = apply_l3_robotwin_config(
            self.displaystand_modelname,
            model_id=3,
            override_name=self.displaystand_modelname,
            override_id=1,
        )
        self.displaystand_top_scale_factor = 1.0 if is_l3_enabled() else 2.0
        # Shared XY placement and simple manual Z offsets (adjust if you want more/less separation)
        self.displaystand_xy = (0, 0)
        self.displaystand_bottom_offset = -0.01
        self.displaystand_top_offset = -0.01

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

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Load jam jar (robotwin object, left robot will grasp this)
        jamjar_pose = sapien.Pose(
            p=[0, 0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        jamjar_obj = create_actor(
            scene=self.scene,
            pose=jamjar_pose,
            modelname=self.jamjar_modelname,
            convex=True,
            model_id=self.jamjar_model_id,
            scale=(0.5, 0.5, 0.5),
        )
        jamjar_obj.set_mass(0.5)
        self.jamjar = jamjar_obj.actor

        # Load milk box (robotwin object, right robot will grasp this)
        milkbox_pose = sapien.Pose(
            p=[0, -0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        milkbox_obj = create_actor(
            scene=self.scene,
            pose=milkbox_pose,
            modelname=self.milkbox_modelname,
            convex=True,
            model_id=self.milkbox_model_id,
            scale=self.milkbox_scale,
            replace_scale=self.milkbox_replace_scale,
        )
        milkbox_obj.set_mass(0.5)
        self.milkbox = milkbox_obj.actor

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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    REWARD_PHASES, self.num_envs, self.device
                )
            self.reward_tracker.reset(env_idx)

            # Initialize jam jar (left robot's object to pick) - with randomization
            jamjar_xyz = torch.zeros((b, 3), device=self.device)
            jamjar_xyz[:, 0] = -0.3  # X position
            jamjar_xyz[:, 1] = -0.15  # Y position - left side
            jamjar_xyz[:, 2] = self.jamjar_z  # Z position (on table surface)
            jamjar_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02  # Per-env random offset
            jamjar_xyz = apply_l1_offset_xy(jamjar_xyz, offset=(0.05, -0.1))

            # Fixed upright orientation
            base_quat = euler2quat(np.pi/2, 0.0, 0.0)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.jamjar.set_pose(Pose.create_from_pq(p=jamjar_xyz, q=qs))

            # Initialize milk box (right robot's object to pick) - with randomization
            milkbox_xyz = torch.zeros((b, 3), device=self.device)
            milkbox_xyz[:, 0] = -0.3  # X position
            milkbox_xyz[:, 1] = 0.15  # Y position - right side
            milkbox_xyz[:, 2] = self.milkbox_z  # Z position (on table surface)
            milkbox_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02  # Per-env random offset
            milkbox_xyz = apply_l1_offset_xy(milkbox_xyz, offset=(0.05, 0.1))


            # Fixed upright orientation
            self.milkbox.set_pose(Pose.create_from_pq(p=milkbox_xyz, q=qs))

            # Define shared base position for both display stands
            displaystand_base_xy = torch.zeros((b, 2), device=self.device)
            displaystand_base_xy[:, 0] = self.displaystand_xy[0]  # X position
            displaystand_base_xy[:, 1] = self.displaystand_xy[1]  # Y position
            displaystand_base_xy[:, :2] += torch.rand((b, 2), device=self.device) * 0.02

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

    def _after_reconfigure(self, options: dict):
        # Get z-offset for jam jar to place it on table surface
        collision_mesh = self.jamjar.get_first_collision_mesh()
        if collision_mesh is not None:
            self.jamjar_z = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            self.jamjar_z = 0.02  # Default height if no collision mesh

        # Get z-offset for milk box to place it on table surface
        collision_mesh = self.milkbox.get_first_collision_mesh()
        if collision_mesh is not None:
            self.milkbox_z = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            self.milkbox_z = 0.02  # Default height if no collision mesh

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
        is_jamjar_grasped = self.left_agent.is_grasping(self.jamjar)
        is_milkbox_grasped = self.right_agent.is_grasping(self.milkbox)
        is_left_static = self.left_agent.is_static(0.2)
        is_right_static = self.right_agent.is_static(0.2)
        jamjar_pos = self._get_actor_geom_center(self.jamjar)
        milkbox_pos = self._get_actor_geom_center(self.milkbox)
        stand_pos = self._get_actor_geom_center(self.displaystand_bottom)
        is_jamjar_above_table = jamjar_pos[:, 2] > (stand_pos[:, 2] + 0.02)
        is_milkbox_above_table = milkbox_pos[:, 2] > (stand_pos[:, 2] + 0.02)
        jamjar_to_stand_dist = torch.linalg.norm(
            jamjar_pos[:, :2] - stand_pos[:, :2], dim=-1
        )
        milkbox_to_stand_dist = torch.linalg.norm(
            milkbox_pos[:, :2] - stand_pos[:, :2], dim=-1
        )
        is_jamjar_on_stand = jamjar_to_stand_dist < 0.20
        is_milkbox_on_stand = milkbox_to_stand_dist < 0.20

        success = is_jamjar_above_table & is_milkbox_above_table & is_jamjar_on_stand & is_milkbox_on_stand

        result = dict(
            is_jamjar_grasped=is_jamjar_grasped,
            is_milkbox_grasped=is_milkbox_grasped,
            is_jamjar_above_table=is_jamjar_above_table,
            is_milkbox_above_table=is_milkbox_above_table,
            is_jamjar_on_stand=is_jamjar_on_stand,
            is_milkbox_on_stand=is_milkbox_on_stand,
            is_left_static=is_left_static,
            is_right_static=is_right_static,
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
            is_jamjar_grasped=info["is_jamjar_grasped"],
            is_milkbox_grasped=info["is_milkbox_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                jamjar_pose=self.jamjar.pose.raw_pose,
                milkbox_pose=self.milkbox.pose.raw_pose,
                displaystand_bottom_pose=self.displaystand_bottom.pose.raw_pose,
                displaystand_top_pose=self.displaystand_top.pose.raw_pose,
                left_tcp_to_jamjar_pos=self.jamjar.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_milkbox_pos=self.milkbox.pose.p - self.right_agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        jamjar_pos = self._get_actor_geom_center(self.jamjar)
        milkbox_pos = self._get_actor_geom_center(self.milkbox)
        stand_pos = self._get_actor_geom_center(self.displaystand_bottom)
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        is_jamjar_grasped = info["is_jamjar_grasped"]
        is_milkbox_grasped = info["is_milkbox_grasped"]
        jamjar_on_stand = info["is_jamjar_on_stand"]
        milkbox_on_stand = info["is_milkbox_on_stand"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_jamjar = reach_reward(left_tcp, jamjar_pos, scale=5.0)
        r_reach_jamjar = torch.where(is_jamjar_grasped | jamjar_on_stand | success, ones, r_reach_jamjar)
        r_grasp_jamjar = grasp_reward(left_tcp, jamjar_pos, is_jamjar_grasped, proximity_scale=5.0)
        r_grasp_jamjar = torch.where(jamjar_on_stand | success, ones, r_grasp_jamjar)
        r_transport_jamjar = transport_reward(jamjar_pos, stand_pos, is_jamjar_grasped, scale=4.0)
        r_transport_jamjar = torch.where(jamjar_on_stand | success, ones, r_transport_jamjar)
        r_place_jamjar = torch.where(success, ones, jamjar_on_stand.float())

        r_reach_milkbox = reach_reward(right_tcp, milkbox_pos, scale=5.0)
        r_reach_milkbox = torch.where(is_milkbox_grasped | milkbox_on_stand | success, ones, r_reach_milkbox)
        r_grasp_milkbox = grasp_reward(right_tcp, milkbox_pos, is_milkbox_grasped, proximity_scale=5.0)
        r_grasp_milkbox = torch.where(milkbox_on_stand | success, ones, r_grasp_milkbox)
        r_transport_milkbox = transport_reward(milkbox_pos, stand_pos, is_milkbox_grasped, scale=4.0)
        r_transport_milkbox = torch.where(milkbox_on_stand | success, ones, r_transport_milkbox)
        r_place_milkbox = torch.where(success, ones, milkbox_on_stand.float())

        self.reward_tracker.update("reach_jamjar", r_reach_jamjar)
        self.reward_tracker.update("grasp_jamjar", r_grasp_jamjar)
        self.reward_tracker.update("transport_jamjar", r_transport_jamjar)
        self.reward_tracker.update("place_jamjar", r_place_jamjar)
        self.reward_tracker.update("reach_milkbox", r_reach_milkbox)
        self.reward_tracker.update("grasp_milkbox", r_grasp_milkbox)
        self.reward_tracker.update("transport_milkbox", r_transport_milkbox)
        self.reward_tracker.update("place_milkbox", r_place_milkbox)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
