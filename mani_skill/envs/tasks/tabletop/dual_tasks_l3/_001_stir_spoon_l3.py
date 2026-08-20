import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from typing import Any, Dict, Tuple

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import apply_l3_robotwin_model
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    reach_reward,
    grasp_reward,
    tanh_reward,
    return_reward,
    normalized_progress,
)


REWARD_PHASES = ["reach", "grasp", "lift", "approach_bowl", "stir", "return"]


@register_env("TwoRobotStirSpoonL3-v1", max_episode_steps=120)
class TwoRobotStirSpoonEnvL3(BaseEnv):
    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    video_info_whitelist = {
        "reward",
        "success",
        "is_grasped",
        "peak_r_reach",
        "peak_r_grasp",
        "peak_r_lift",
        "peak_r_approach_bowl",
        "peak_r_stir",
        "peak_r_return",
    }

    env_id = "TwoRobotStirSpoonL3-v1"
    tool_modelname = "086_woodenblock"

    stir_reach_xy_thresh = 0.08
    stir_success_steps = 80

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
        self.mug_modelname = "039_mug"
        self.mug_model_id = 10
        self.tool_mass = 0.5
        self.tool_static_friction = 5.0
        self.tool_dynamic_friction = 5.0
        self.tool_model_id = get_model_id(self.tool_modelname, model_id=1)
        self.tool_scale = (0.3, 1.7, 0.3)
        self.tool_replace_scale = False
        self.bowl_modelname, self.bowl_model_id = apply_l3_robotwin_model(
            "002_bowl",
            model_id=5,
            override_name="062_plasticbox",
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

        self.mug_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(np.pi / 2, 0.0, np.pi / 2))
        self.bowl_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(np.pi / 2, 0.0, 0.0))
        self.tool_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(np.pi / 2, 0.0, 0.0))

        self.mug = create_actor(
            scene=self.scene,
            pose=self.mug_pose,
            modelname=self.mug_modelname,
            convex=True,
            model_id=self.mug_model_id,
            is_static=True,
        ).actor
        self.bowl = create_actor(
            scene=self.scene,
            pose=self.bowl_pose,
            modelname=self.bowl_modelname,
            convex=True,
            model_id=self.bowl_model_id,
            is_static=True,
        ).actor

        self.spoon = create_actor(
            scene=self.scene,
            pose=self.tool_pose,
            modelname=self.tool_modelname,
            convex=True,
            model_id=self.tool_model_id,
            scale=self.tool_scale,
            replace_scale=self.tool_replace_scale,
            mass=self.tool_mass,
        ).actor
        tool_material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=self.tool_static_friction,
            dynamic_friction=self.tool_dynamic_friction,
            restitution=0.0,
        )
        for body in self.spoon._bodies:
            for cs in body.get_collision_shapes():
                cs.set_physical_material(tool_material)

    def _after_reconfigure(self, options: dict):
        def _compute_object_z(obj) -> float:
            mesh = obj.get_first_collision_mesh()
            return -mesh.bounding_box.bounds[0, 2] if mesh is not None else 0.0

        self._mug_z = common.to_tensor([_compute_object_z(self.mug)], device=self.device)
        self._bowl_z = common.to_tensor([_compute_object_z(self.bowl)], device=self.device)
        self._tool_z = common.to_tensor([_compute_object_z(self.spoon)], device=self.device)

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

            shared_xy_offset = torch.rand((b, 2), device=self.device) * 0.02

            mug_xyz = torch.zeros((b, 3), device=self.device)
            mug_xyz[:, 0] = -0.12
            mug_xyz[:, 1] = -0.25
            mug_xyz[:, 2] = self._mug_z[0]
            mug_xyz[:, :2] += shared_xy_offset
            mug_q = torch.tensor(self.mug_pose.q, dtype=torch.float32, device=self.device).repeat(b, 1)
            self.mug.set_pose(Pose.create_from_pq(p=mug_xyz, q=mug_q))

            bowl_xyz = torch.zeros((b, 3), device=self.device)
            bowl_xyz[:, 0] = 0.03
            bowl_xyz[:, 1] = -0.13
            bowl_xyz[:, 2] = self._bowl_z[0]
            bowl_xyz[:, :2] += shared_xy_offset
            bowl_q = torch.tensor(self.bowl_pose.q, dtype=torch.float32, device=self.device).repeat(b, 1)
            self.bowl.set_pose(Pose.create_from_pq(p=bowl_xyz, q=bowl_q))

            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[:, 0] = mug_xyz[:, 0] + 0.015
            tool_xyz[:, 1] = mug_xyz[:, 1]
            tool_xyz[:, 2] = mug_xyz[:, 2] + 0.11
            tool_q = torch.tensor(self.tool_pose.q, dtype=torch.float32, device=self.device).repeat(b, 1)
            self.spoon.set_pose(Pose.create_from_pq(p=tool_xyz, q=tool_q))
            if not hasattr(self, "spoon_home_pos"):
                self.spoon_home_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.spoon_home_pos[env_idx] = tool_xyz

            if not hasattr(self, "spoon_ever_reached_bowl"):
                self.spoon_ever_reached_bowl = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            if not hasattr(self, "spoon_last_pos"):
                self.spoon_last_pos = self.spoon.pose.p.clone()
            if not hasattr(self, "spoon_stir_dist"):
                self.spoon_stir_dist = torch.zeros(self.num_envs, device=self.device)
            if not hasattr(self, "spoon_stir_steps"):
                self.spoon_stir_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)

            self.spoon_ever_reached_bowl[env_idx] = False
            self.spoon_last_pos[env_idx] = self.spoon.pose.p[env_idx].clone()
            self.spoon_stir_dist[env_idx] = 0.0
            self.spoon_stir_steps[env_idx] = 0
            self.reward_tracker.reset(env_idx)

            # Keep MP waypoints consistent with the current initialized tool pose.
            spoon_pose_sp = self.spoon.pose.sp
            base_pose = sapien.Pose(
                p=np.array(spoon_pose_sp.p, copy=True),
                q=np.array(spoon_pose_sp.q, copy=True),
            )
            tcp_q = np.array(self.left_agent.tcp.pose.sp.q, copy=True)
            self.grasp_pose = sapien.Pose(p=base_pose.p + np.array([0.0, 0.0, 0.0]), q=tcp_q)
            self.reach_pose = sapien.Pose(p=self.grasp_pose.p + np.array([0.0, 0.0, 0.10]), q=tcp_q)
            self.lift_pose_tilted = sapien.Pose(p=self.grasp_pose.p + np.array([0.0, 0.0, 0.16]), q=tcp_q)
            self.lift_pose_vertical = sapien.Pose(
                p=self.lift_pose_tilted.p,
                q=tcp_q,
            )

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
        spoon_to_bowl_xy_dist = torch.linalg.norm(self.bowl.pose.p[:, :2] - self.spoon.pose.p[:, :2], axis=1)
        is_grasped = self.left_agent.is_grasping(self.spoon) | self.right_agent.is_grasping(self.spoon)
        is_spoon_at_bowl = (spoon_to_bowl_xy_dist <= self.stir_reach_xy_thresh) & is_grasped
        self.spoon_ever_reached_bowl = self.spoon_ever_reached_bowl | is_spoon_at_bowl

        delta_xy = self.spoon.pose.p[:, :2] - self.spoon_last_pos[:, :2]
        step_dist = torch.linalg.norm(delta_xy, axis=1)
        self.spoon_stir_dist += step_dist * is_spoon_at_bowl.float()
        self.spoon_stir_steps += is_spoon_at_bowl.int()
        self.spoon_last_pos = self.spoon.pose.p.clone()

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        success = self.spoon_stir_steps >= self.stir_success_steps

        result = dict(
            is_grasped=is_grasped,
            spoon_ever_reached_bowl=self.spoon_ever_reached_bowl,
            spoon_stir_dist=self.spoon_stir_dist,
            spoon_stir_steps=self.spoon_stir_steps,
            is_robot_static=is_robot_static,
            success=success,
            is_obj_placed=self.spoon_ever_reached_bowl,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            mug_pose=self.mug.pose.raw_pose,
            bowl_pose=self.bowl.pose.raw_pose,
            spoon_pose=self.spoon.pose.raw_pose,
            is_grasped=info["is_grasped"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        spoon_pos = self._get_actor_geom_center(self.spoon)
        mug_pos = self._get_actor_geom_center(self.mug)
        bowl_pos = self._get_actor_geom_center(self.bowl)
        is_grasped = info["is_grasped"]
        reached_bowl = info["spoon_ever_reached_bowl"]
        stir_steps = info["spoon_stir_steps"].float()
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, spoon_pos, scale=5.0)
        r_reach = torch.where(is_grasped | reached_bowl | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, spoon_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(reached_bowl | success, ones, r_grasp)
        lift_height = spoon_pos[:, 2] - mug_pos[:, 2]
        r_lift = normalized_progress(lift_height, start=0.05, target=0.18) * is_grasped.float()
        r_lift = torch.where(reached_bowl | success, ones, r_lift)
        spoon_to_bowl = torch.linalg.norm(spoon_pos - bowl_pos, dim=-1)
        r_approach_bowl = tanh_reward(spoon_to_bowl, scale=5.0) * is_grasped.float()
        r_approach_bowl = torch.where(reached_bowl | success, ones, r_approach_bowl)
        r_stir = normalized_progress(stir_steps, start=0.0, target=float(self.stir_success_steps))
        r_stir = torch.where(success, ones, r_stir)
        r_return = return_reward(spoon_pos, self.spoon_home_pos, reached_bowl, scale=5.0)
        r_return = torch.where(success, ones, r_return)

        self.reward_tracker.update("reach", r_reach)
        self.reward_tracker.update("grasp", r_grasp)
        self.reward_tracker.update("lift", r_lift)
        self.reward_tracker.update("approach_bowl", r_approach_bowl)
        self.reward_tracker.update("stir", r_stir)
        self.reward_tracker.update("return", r_return)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
