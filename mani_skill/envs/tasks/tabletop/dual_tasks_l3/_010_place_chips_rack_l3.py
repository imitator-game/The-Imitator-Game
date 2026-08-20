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
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils import common
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    grasp_reward,
    normalized_progress,
    reach_reward,
    transport_reward,
)


REWARD_PHASES = ["reach", "grasp", "lift", "transport", "place"]


@register_env("TwoRobotPlaceChipsRackL3-v1", max_episode_steps=200)
class TwoRobotPlaceChipsRackEnvL3(BaseEnv):
    """
    **Task Description:**
    The goal is for the left robot to pick up a chips tub and place it on a display stand.
    There are two display stands sharing the same XY (model_id=4 and model_id=3) with a simple manual Z offset.
    There are two robots in this task:
    - Left robot at position [-0.9, 0, 0] - picks up chips and places on top display stand
    - Right robot at position [-0.3, 0, 0] - stays idle

    Both robots face the table.

    **Randomizations:**
    - Object positions are fixed (not randomized)
    - Objects have fixed upright orientation with X-axis rotation

    **Success Conditions:**
    - Chips tub is placed on the top display stand
    - Left robot is static (task completed)
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

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

        # Chips Tub (robotwin object, for left robot to grasp)
        self.chipstub_modelname = "025_chips-tub"
        self.chipstub_model_id = get_model_id(self.chipstub_modelname, model_id=3)
        self.chipstub_scale = (1.0, 1.0, 1.0)
        self.chipstub_replace_scale = False

        # Display Stands (robotwin objects, sharing XY with manual Z offsets)
        self.displaystand_modelname = "074_displaystand"
        self.displaystand_bottom_model_id = 4  # Bottom display stand
        self.displaystand_top_model_id = 3     # Top display stand
        # Shared XY placement and simple manual Z offsets (adjust if you want more/less separation)
        self.displaystand_xy = (0, 0)
        self.displaystand_bottom_offset = -0.01
        self.displaystand_top_offset = -0.01
        # Coffee Box (robotwin object, placed on display stand)
        self.coffee_box_modelname = "113_coffee-box"
        self.coffee_box_model_id = get_model_id(self.coffee_box_modelname, model_id=2)
        self.coffee_box_scale = None
        self.coffee_box_replace_scale = False
        self.coffee_box_on_stand_offset = -0.01
        # Bottle (robotwin object, placed on display stand)
        self.bottle_modelname = "001_bottle"
        self.bottle_model_id = get_model_id(self.bottle_modelname, model_id=0)
        self.bottle_scale = None
        self.bottle_replace_scale = False
        self.bottle_on_stand_offset = -0.01

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

        # Load chips tub (robotwin object, left robot will grasp this)
        chipstub_pose = sapien.Pose(
            p=[0, 0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        chipstub_obj = create_actor(
            scene=self.scene,
            pose=chipstub_pose,
            modelname=self.chipstub_modelname,
            convex=True,
            model_id=self.chipstub_model_id,
            scale=self.chipstub_scale,
            replace_scale=self.chipstub_replace_scale,
            mass=0.5
        )
        # chipstub_obj.set_mass(0.5)
        self.chipstub = chipstub_obj.actor

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
            scale_factor=2.0,
            is_static=True
        )

        # Load coffee box (robotwin object, placed on display stand)
        coffee_box_pose = sapien.Pose(
            p=[0, -0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        coffee_box_obj = create_actor(
            scene=self.scene,
            pose=coffee_box_pose,
            modelname=self.coffee_box_modelname,
            convex=True,
            model_id=self.coffee_box_model_id,
            is_static=True,
        )
        self.coffee_box = coffee_box_obj.actor

        # Load bottle (robotwin object, placed on display stand)
        bottle_pose = sapien.Pose(
            p=[0, -0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        bottle_obj = create_actor(
            scene=self.scene,
            pose=bottle_pose,
            modelname=self.bottle_modelname,
            convex=True,
            model_id=self.bottle_model_id,
            is_static=True,
        )
        self.bottle = bottle_obj.actor

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    REWARD_PHASES, self.num_envs, self.device
                )
            self.reward_tracker.reset(env_idx)
            if not hasattr(self, "_chipstub_start_height"):
                self._chipstub_start_height = torch.zeros(
                    self.num_envs, device=self.device
                )

            # Initialize chips tub (left robot's object to pick) - fixed position on the left
            chipstub_xyz = torch.zeros((b, 3), device=self.device)
            chipstub_xyz[:, 0] = -0.3  # X position
            chipstub_xyz[:, 1] = -0.15  # Y position - left side
            chipstub_xyz[:, 2] = self.chipstub_z  # Z position (on table surface)
            chipstub_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02

            # Fixed upright orientation
            base_quat = euler2quat(np.pi/2, 0.0, 0.0)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.chipstub.set_pose(Pose.create_from_pq(p=chipstub_xyz, q=qs))
            self._chipstub_start_height[env_idx] = chipstub_xyz[:, 2]

            # Define shared base position for both display stands (can be randomized in future)
            displaystand_base_xy = torch.zeros((b, 2), device=self.device)
            displaystand_base_xy[:, 0] = self.displaystand_xy[0]  # X position
            displaystand_base_xy[:, 1] = self.displaystand_xy[1]  # Y position
            displaystand_base_xy[:, :2] += torch.rand((b, 2), device=self.device) * 0.02 # per-env shared offset

            # Initialize bottom display stand using base position
            displaystand_bottom_xyz = torch.zeros((b, 3), device=self.device)
            displaystand_bottom_xyz[:, 0] = displaystand_base_xy[:, 0]
            displaystand_bottom_xyz[:, 1] = displaystand_base_xy[:, 1]
            displaystand_bottom_xyz[:, 2] = self.displaystand_bottom_z  # Z position (on table surface)

            # Fixed upright orientation
            base_quat = euler2quat(np.pi/2, 0.0, -np.pi/2)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.displaystand_bottom.set_pose(Pose.create_from_pq(displaystand_bottom_xyz, qs))

            # Initialize top display stand - shares XY with bottom, manual Z offset
            displaystand_top_xyz = torch.zeros((b, 3), device=self.device)
            displaystand_top_xyz[:, 0] = displaystand_base_xy[:, 0] + 0.05
            displaystand_top_xyz[:, 1] = displaystand_base_xy[:, 1]
            displaystand_top_xyz[:, 2] = self.displaystand_top_z  # Manual Z offset (see _after_reconfigure)

            # Fixed upright orientation (same as bottom)
            self.displaystand_top.set_pose(Pose.create_from_pq(displaystand_top_xyz, qs))

            # Initialize coffee box - placed on display stand using shared XY
            coffee_box_xyz = torch.zeros((b, 3), device=self.device)
            coffee_box_xyz[:, 0] = displaystand_base_xy[:, 0] + 0.05
            coffee_box_xyz[:, 1] = displaystand_base_xy[:, 1] + 0.05
            coffee_box_xyz[:, 2] = self.coffee_box_z

            coffee_box_quat = euler2quat(np.pi/2, 0.0, 0.0)
            coffee_box_qs = torch.tensor(
                [coffee_box_quat] * b, device=self.device, dtype=torch.float32
            )
            self.coffee_box.set_pose(Pose.create_from_pq(p=coffee_box_xyz, q=coffee_box_qs))

            # Initialize bottle - placed on display stand using shared XY
            bottle_xyz = torch.zeros((b, 3), device=self.device)
            bottle_xyz[:, 0] = displaystand_base_xy[:, 0] + 0.05
            bottle_xyz[:, 1] = displaystand_base_xy[:, 1] - 0.05
            bottle_xyz[:, 2] = self.bottle_z

            bottle_quat = euler2quat(np.pi/2, 0.0, 0.0)
            bottle_qs = torch.tensor(
                [bottle_quat] * b, device=self.device, dtype=torch.float32
            )
            self.bottle.set_pose(Pose.create_from_pq(p=bottle_xyz, q=bottle_qs))

    def _after_reconfigure(self, options: dict):
        # Get z-offset for chips tub to place it on table surface
        collision_mesh = self.chipstub.get_first_collision_mesh()
        if collision_mesh is not None:
            self.chipstub_z = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            self.chipstub_z = 0.02  # Default height if no collision mesh

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

        # Get z-offset for coffee box to place it on the TOP display stand (all static objects)
        collision_mesh = self.coffee_box.get_first_collision_mesh()
        if collision_mesh is not None:
            coffee_box_bottom_offset = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            coffee_box_bottom_offset = 0.02
        self.coffee_box_z = (
            self.displaystand_top_z
            + self.displaystand_top_height
            + coffee_box_bottom_offset
            + self.coffee_box_on_stand_offset
        )

        # Get z-offset for bottle to place it on the TOP display stand (all static objects)
        collision_mesh = self.bottle.get_first_collision_mesh()
        if collision_mesh is not None:
            bottle_bottom_offset = -collision_mesh.bounding_box.bounds[0, 2]
        else:
            bottle_bottom_offset = 0.02
        self.bottle_z = (
            self.displaystand_top_z
            + self.displaystand_top_height
            + bottle_bottom_offset
            + self.bottle_on_stand_offset
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
        is_chipstub_grasped = self.left_agent.is_grasping(self.chipstub)
        is_left_static = self.left_agent.is_static(0.2)
        chipstub_pos = self._get_actor_geom_center(self.chipstub)
        stand_pos = self._get_actor_geom_center(self.displaystand_bottom)
        chipstub_to_stand_xy = torch.linalg.norm(chipstub_pos[:, :2] - stand_pos[:, :2], dim=-1)
        is_chipstub_on_stand = chipstub_to_stand_xy < 0.16
        is_chipstub_above_table = chipstub_pos[:, 2] > (stand_pos[:, 2] + 0.02)
        success = is_chipstub_on_stand & is_chipstub_above_table
        result = dict(
            is_chipstub_grasped=is_chipstub_grasped,
            is_chipstub_on_stand=is_chipstub_on_stand,
            is_chipstub_above_table=is_chipstub_above_table,
            is_left_static=is_left_static,
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
            is_chipstub_grasped=info["is_chipstub_grasped"],
            is_chipstub_on_stand=info["is_chipstub_on_stand"],
        )
        if "state" in self.obs_mode:
            obs.update(
                chipstub_pose=self.chipstub.pose.raw_pose,
                displaystand_bottom_pose=self.displaystand_bottom.pose.raw_pose,
                displaystand_top_pose=self.displaystand_top.pose.raw_pose,
                left_tcp_to_chipstub_pos=self.chipstub.pose.p - self.left_agent.tcp.pose.p,
                chipstub_to_stand_pos=self.displaystand_top.pose.p - self.chipstub.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        chipstub_pos = self._get_actor_geom_center(self.chipstub)
        stand_pos = self._get_actor_geom_center(self.displaystand_bottom)
        tcp_pos = self.left_agent.tcp.pose.p
        is_grasped = info["is_chipstub_grasped"]
        is_placed = info["is_chipstub_on_stand"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, chipstub_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, chipstub_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_placed | success, ones, r_grasp)
        lift_height = chipstub_pos[:, 2] - self._chipstub_start_height
        r_lift = normalized_progress(lift_height, 0.0, 0.08) * is_grasped.float()
        r_lift = torch.where(is_placed | success, ones, r_lift)
        r_transport = transport_reward(chipstub_pos, stand_pos, is_grasped, scale=4.0)
        r_transport = torch.where(is_placed | success, ones, r_transport)
        r_place = torch.where(success, ones, is_placed.float())

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
