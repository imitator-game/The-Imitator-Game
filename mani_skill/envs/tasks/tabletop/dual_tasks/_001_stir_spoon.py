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
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_ycb_model_id,
    apply_l3_robotwin_model,
    is_l2_enabled,
)
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply
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


@register_env("TwoRobotStirSpoon-v1", max_episode_steps=100)
class TwoRobotStirSpoonEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to use the spoon from the mug to stir inside the bowl. There are two robots in this task,
    both positioned side-by-side in front of the table (similar to PickCube layout):
    - Left robot at position [-0.60, -0.3, 0]
    - Right robot at position [-0.60, 0.3, 0]

    Both robots face the table and can collaborate to manipulate objects.

    **Randomizations:**
    - mug has a fixed 30-degree rotation on the horizontal plane
    - Object positions are FIXED (not randomized):
      - mug at [-0.1, -0.1, z]
      - bowl at [0.1, 0.1, z]

    **Success Conditions:**
    - The spoon from the mug is used to stir inside the bowl
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    goal_thresh = 0.05
    env_id = "TwoRobotStirSpoon-v1"
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
        self.mug_modelname = "039_mug"
        self.mug_model_id = 10
        self.bowl_modelname, self.bowl_model_id = apply_l3_robotwin_model(
            "002_bowl",
            model_id=5,
            override_name="062_plasticbox",
        )
        self.spoon_model_id = apply_l2_ycb_model_id("031_spoon", override_id="030_fork")
        self.l2_spoon_offset_x = 0.02
        self.l2_spoon_offset_z = 0.0
        self.l2_spoon_extra_euler = (0.0, np.pi/30, np.pi/17)
        self._origin_waypoints_defined = False
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

        # Load mug (contains spoon) - on the right side
        # Use euler2quat(pi/2, 0, 0) to make object upright (X-axis rotation)
        mug_pose = sapien.Pose(
            p=[0, 0.5, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        mug_obj = create_actor(
            scene=self.scene,
            pose=mug_pose,
            modelname=self.mug_modelname,
            convex=True,
            is_static=True,  # Make mug static (won't move)
            model_id=self.mug_model_id,
        )
        self.mug = mug_obj.actor

        # Load bowl (stirring target) - on the left side
        bowl_pose = sapien.Pose(
            p=[0, -0.0, 0],
            q=euler2quat(np.pi/2, 0.0, 0.0)
        )
        bowl_obj = create_actor(
            scene=self.scene,
            pose=bowl_pose,
            modelname=self.bowl_modelname,
            convex=True,
            is_static=True,
            model_id=self.bowl_model_id,
        )
        self.bowl = bowl_obj.actor

        # Load spoon (utensil in mug) - using YCB loading method
        self._spoons = []
        for i in range(self.num_envs):
            spoon_builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{self.spoon_model_id}",
            )
            spoon_builder.initial_pose = sapien.Pose(p=[0, 0.2, 0])
            spoon_builder.set_scene_idxs([i])
            self._spoons.append(spoon_builder.build(name=f"{self.spoon_model_id}-{i}"))
            self.remove_from_state_dict_registry(self._spoons[-1])

        from mani_skill.utils.structs.actor import Actor
        self.spoon = Actor.merge(self._spoons, name="ycb_spoon")
        self.add_to_state_dict_registry(self.spoon)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "spoon_ever_reached_bowl"):
                self.spoon_ever_reached_bowl = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.spoon_ever_reached_bowl[env_idx] = False
            if not hasattr(self, "spoon_last_pos"):
                self.spoon_last_pos = self.spoon.pose.p.clone()
            if not hasattr(self, "spoon_stir_dist"):
                self.spoon_stir_dist = torch.zeros(self.num_envs, device=self.device)
            if not hasattr(self, "spoon_stir_steps"):
                self.spoon_stir_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.spoon_last_pos[env_idx] = self.spoon.pose.p[env_idx].clone()
            self.spoon_stir_dist[env_idx] = 0.0
            self.spoon_stir_steps[env_idx] = 0
            self.reward_tracker.reset(env_idx)

            # Generate shared random offset for all objects (same as bookcase: ±2cm)
            shared_xy_offset = torch.rand((b, 2), device=self.device) * 0.02  # Per-env shared random offset

            # Initialize mug (contains spoon)
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.12  # X position
            xyz[:, 1] = -0.12  # Y position
            xyz[:, 2] = self.mug_z  # Z position (on table surface)
            xyz[:, :2] += shared_xy_offset  # Apply shared random offset
            xyz = apply_l1_offset_xy(xyz, offset=(-0.15, 0.1))
            # Fixed upright orientation with fixed Z rotation (no randomization)
            base_quat = euler2quat(np.pi/2, 0.0, np.pi/2)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.mug.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            bowl_xy_offset = torch.rand((b, 2), device=self.device) * 0.02
            # Initialize bowl - maintain relative position to mug
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.12 + 0.02  # Base position + relative offset
            xyz[:, 1] = -0.12 + 0.12  # Base position + relative offset
            xyz[:, 2] = self.bowl_z  # Z position (on table surface)
            xyz[:, :2] += bowl_xy_offset  # Apply same shared random offset
            # Fixed upright orientation
            base_quat = euler2quat(np.pi/2, 0.0, 0.0)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.bowl.set_pose(Pose.create_from_pq(xyz, qs))

            # Initialize spoon - maintain relative position to mug
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.12 + 0.02  # Base position + relative offset
            xyz[:, 1] = -0.12  # Base position
            xyz[:, 2] = self.mug_z + 0.11  # Z position (in mug)
            xyz[:, :2] += shared_xy_offset  # Apply same shared random offset
            xyz = apply_l1_offset_xy(xyz, offset=(-0.15, 0.1))
            # Fixed orientation for spoon in mug
            base_quat = euler2quat(-np.pi/3, -np.pi/2, np.pi)
            if is_l2_enabled():
                xyz[:, 0] += self.l2_spoon_offset_x
                xyz[:, 2] += self.l2_spoon_offset_z
                extra_quat = euler2quat(*self.l2_spoon_extra_euler)
                extra_q = torch.tensor(extra_quat, device=self.device, dtype=torch.float32)
                base_q = torch.tensor(base_quat, device=self.device, dtype=torch.float32)
                base_quat = quaternion_multiply(extra_q, base_q).cpu().numpy()
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.spoon.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            if not hasattr(self, "spoon_home_pos"):
                self.spoon_home_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.spoon_home_pos[env_idx] = xyz

            if not self._origin_waypoints_defined:
                origin_base_quat = euler2quat(-np.pi/3, -np.pi/2, np.pi)
                self.origin_spoon_base = sapien.Pose(
                    p=[-0.10, -0.12, 0.11 + self.mug_z],  # Fixed reference position
                    q=tuple(origin_base_quat),  # Fixed reference orientation
                )
                self.origin_grasp_pose = sapien.Pose(
                    p=[-0.086, -0.18, 0.16],
                    q=[-0.207, 0.978, 0.004, 0.011],
                )
                reach_position = self.origin_grasp_pose.p + np.array([0, 0, 0.1])
                self.origin_reach_pose = sapien.Pose(
                    p=reach_position,
                    q=self.origin_grasp_pose.q,
                )
                lift_position = self.origin_grasp_pose.p + np.array([0, 0, 0.15])
                self.origin_lift_tilted_pose = sapien.Pose(
                    p=lift_position,
                    q=self.origin_grasp_pose.q,
                )
                vertical_orientation_quat = euler2quat(np.pi, 0, 0)
                self.origin_lift_vertical_pose = sapien.Pose(
                    p=lift_position,
                    q=tuple(vertical_orientation_quat),
                )
                self._origin_waypoints_defined = True

            # Get the current spoon pose (which includes randomization)
            spoon_pose_sp = self.spoon.pose.sp
            # Use a copy of the current spoon pose as the new base frame
            self._update_spoon_waypoints(
                sapien.Pose(
                    p=np.array(spoon_pose_sp.p, copy=True),
                    q=np.array(spoon_pose_sp.q, copy=True),
                )
            )

    def _transform_point(self, origin_base: sapien.Pose, new_base: sapien.Pose, origin_pos: sapien.Pose):
        from scipy.spatial.transform import Rotation
        # Convert quaternions to rotation objects
        R_O = Rotation.from_quat([origin_base.q[1], origin_base.q[2], origin_base.q[3], origin_base.q[0]])  # scipy uses (x,y,z,w) format
        R_O_new = Rotation.from_quat([new_base.q[1], new_base.q[2], new_base.q[3], new_base.q[0]])

        # Calculate relative position of A with respect to O
        relative_pos = origin_pos.p - origin_base.p

        # Calculate relative rotation of A with respect to O
        R_A = Rotation.from_quat([origin_pos.q[1], origin_pos.q[2], origin_pos.q[3], origin_pos.q[0]])
        relative_rot = R_O.inv() * R_A

        # Calculate new position of A
        relative_pos_rotated = R_O_new.apply(R_O.inv().apply(relative_pos))
        p_A_new = new_base.p + relative_pos_rotated

        # Calculate new rotation of A
        R_A_new = R_O_new * relative_rot
        q_A_new = R_A_new.as_quat()  # Returns (x,y,z,w)

        # Convert back to (w,x,y,z) format
        q_A_new = np.array([q_A_new[3], q_A_new[0], q_A_new[1], q_A_new[2]])

        new_pos = sapien.Pose(
            p=p_A_new,
            q=q_A_new
        )
        return new_pos

    def _update_spoon_waypoints(self, new_base: sapien.Pose):
        if not self._origin_waypoints_defined:
            return
        self.grasp_pose = self._transform_point(self.origin_spoon_base, new_base, self.origin_grasp_pose)
        self.reach_pose = self._transform_point(self.origin_spoon_base, new_base, self.origin_reach_pose)
        self.lift_pose_tilted = self._transform_point(self.origin_spoon_base, new_base, self.origin_lift_tilted_pose)
        self.lift_pose_vertical = self._transform_point(self.origin_spoon_base, new_base, self.origin_lift_vertical_pose)

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

    def _after_lr_mirror(self, env_idx: torch.Tensor, options: dict):
        # Recompute waypoints to match the mirrored spoon position (rotation unchanged).
        if self._origin_waypoints_defined:
            spoon_pose_sp = self.spoon.pose.sp
            self._update_spoon_waypoints(
                sapien.Pose(
                    p=np.array(spoon_pose_sp.p, copy=True),
                    q=np.array(spoon_pose_sp.q, copy=True),
                )
            )

        # Keep last position aligned with the mirrored spoon pose.
        if hasattr(self, "spoon_last_pos"):
            self.spoon_last_pos[env_idx] = self.spoon.pose.p[env_idx].clone()

    def _after_reconfigure(self, options: dict):
        # Get z-offset for mug to place it on table surface
        collision_mesh = self.mug.get_first_collision_mesh()
        if collision_mesh is not None:
            self.mug_z = -collision_mesh.bounding_box.bounds[0, 2]   # Add small offset to prevent falling
        else:
            self.mug_z = 0.02  # Default height if no collision mesh

        # Get z-offset for bowl to place it on table surface
        collision_mesh = self.bowl.get_first_collision_mesh()
        self.bowl_z = 0.0  # Default height if no collision mesh

        # Get z-offset for YCB spoon to place it on table surface
        self.spoon_zs = []
        for spoon_obj in self._spoons:
            collision_mesh = spoon_obj.get_first_collision_mesh()
            if collision_mesh is not None:
                self.spoon_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
            else:
                self.spoon_zs.append(0.01)  # Default height if no collision mesh
        self.spoon_zs = common.to_tensor(self.spoon_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        # 1. Check if spoon is currently near bowl (to update historical reach flag)
        # Calculate distance only on the XY plane
        spoon_to_bowl_xy_dist = torch.linalg.norm(self.bowl.pose.p[:, :2] - self.spoon.pose.p[:, :2], axis=1)
        is_grasped = self.left_agent.is_grasping(self.spoon) | self.right_agent.is_grasping(self.spoon)
        is_spoon_at_bowl = (spoon_to_bowl_xy_dist <= 0.08) & is_grasped
        self.spoon_ever_reached_bowl = self.spoon_ever_reached_bowl | is_spoon_at_bowl

        # 2. Accumulate stirring motion while at bowl
        delta_xy = self.spoon.pose.p[:, :2] - self.spoon_last_pos[:, :2]
        step_dist = torch.linalg.norm(delta_xy, axis=1)
        self.spoon_stir_dist += step_dist * is_spoon_at_bowl.float()
        self.spoon_stir_steps += is_spoon_at_bowl.int()
        self.spoon_last_pos = self.spoon.pose.p.clone()

        # 3. Check if robot is static
        is_robot_static = self.left_agent.is_static(0.2)

        # Success: reached bowl and performed enough stirring motion
        success = self.spoon_stir_steps >= 80

        result = dict(
            is_grasped=is_grasped,
            spoon_ever_reached_bowl=self.spoon_ever_reached_bowl,
            spoon_stir_dist=self.spoon_stir_dist,
            spoon_stir_steps=self.spoon_stir_steps,
            # is_robot_static=is_robot_static,
            success=success,
            is_obj_placed=self.spoon_ever_reached_bowl,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.left_agent.tcp.pose.raw_pose,
            goal_pos=self.bowl.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                mug_pose=self.mug.pose.raw_pose,
                tcp_to_mug_pos=self.mug.pose.p - self.left_agent.tcp.pose.p,
                bowl_pose=self.bowl.pose.raw_pose,
                tcp_to_bowl_pos=self.bowl.pose.p - self.left_agent.tcp.pose.p,
                mug_to_bowl_pos=self.bowl.pose.p - self.mug.pose.p,
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

        spoon_to_bowl_dist = torch.linalg.norm(spoon_pos - bowl_pos, dim=-1)
        r_approach_bowl = tanh_reward(spoon_to_bowl_dist, scale=5.0) * is_grasped.float()
        r_approach_bowl = torch.where(reached_bowl | success, ones, r_approach_bowl)

        r_stir = normalized_progress(stir_steps, start=0.0, target=80.0)
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

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
