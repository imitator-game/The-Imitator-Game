from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
from mani_skill.agents.robots.realman.realman import Realman, RealmanMobileBase
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("RealmanPickCubeYCB-v1", max_episode_steps=100, asset_download_ids=["ycb"])
class RealmanPickCubeYCBEnv(BaseEnv):
    """
    **Task Description:**
    Use Realman dual-arm robot to pick up a random object sampled from the YCB dataset
    and move it to a target position. The robot can use one or both arms depending on
    the object size and weight.

    **Randomizations:**
    - the object's xy position is randomized on top of a table in the region [0.3, 0.2] x [-0.2, -0.2]
    - the object's z-axis rotation is randomized
    - the object geometry is randomized by randomly sampling any YCB object (during reconfiguration)
    - target positions are randomized within reachable workspace

    **Success Conditions:**
    - the object position is within goal_thresh (default 0.025) euclidean distance of the goal position
    - at least one robot arm is static (q velocity < 0.2)

    **Goal Specification:**
    - 3D goal position (visualized in human renders)
    """

    SUPPORTED_ROBOTS = ["realman", "realman_mobile_base"]
    agent: Union[Realman, RealmanMobileBase]

    cube_half_size = 0.02
    goal_thresh = 0.06

    def __init__(
            self,
            *args,
            robot_uids="realman",
            robot_init_qpos_noise=0.04,
            num_envs=1,
            reconfiguration_freq=None,
            **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None

        # Filter YCB objects suitable for dual-arm manipulation
        self.all_model_ids = np.array([
            k for k in load_json(
                ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
            ).keys()
            if k not in [
                "022_windex_bottle",  # Too tall
                "028_skillet_lid",  # Too flat
                "029_plate",  # Too flat
                "059_chain",  # Hard to grasp
            ]
        ])

        # For demo purposes, use a subset of objects
        self.all_model_ids = np.array([
            "005_tomato_soup_can",
            "006_mustard_bottle",
            "010_potted_meat_can",
            "065-d_cups",
            "003_cracker_box",
        ])

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
                max_rigid_contact_count=2 ** 20, max_rigid_patch_count=2 ** 19
            )
        )

    @property
    def _default_sensor_configs(self):
        # Multiple camera views for better perception
        pose1 = sapien_utils.look_at(eye=[0.5, 0.5, 0.8], target=[0.0, 0, 0.2])
        pose2 = sapien_utils.look_at(eye=[0.5, 0, 0.8], target=[0.0, 0, 0.2])
        pose3 = sapien_utils.look_at(eye=[0.5, -0.5, 0.8], target=[0.0, 0, 0.2])
        pose4 = sapien_utils.look_at(eye=[-0.5, 0, 0.8], target=[0.0, 0, 0.2])

        return [
            CameraConfig("base_camera_left", pose1, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("base_camera_front", pose2, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("base_camera_right", pose3, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("base_camera_back", pose4, 128, 128, np.pi / 2, 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-1.0, 0.0, 0.8], [0.0, 0.0, 0.0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # Realman robot base position
        super()._load_agent(options, sapien.Pose(
            p=[-1.0, 0, -0.7],
            q=euler2quat(0.0, 0.0, np.pi / 2)
        ))

    def _load_scene(self, options: dict):
        # Build table scene
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Load YCB objects
        model_ids = self._episode_rng.choice(self.all_model_ids, size=2, replace=False)

        # Define spawn positions for objects (within robot workspace)
        xy_poses = [
            (-0.4, 0.15),  # Front left
            (-0.4, -0.15),  # Front right
        ]

        self._objs: List[Actor] = []
        self.obj_heights = []

        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            x, y = xy_poses[i]
            builder.initial_pose = sapien.Pose(p=[x, y, 0])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))

        # Add colored cubes
        self.cube1 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="red_cube",
            initial_pose=sapien.Pose(p=[-0.3, 0.2, self.cube_half_size]),
        )

        self.cube2 = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 0, 1, 1],
            name="blue_cube",
            initial_pose=sapien.Pose(p=[-0.3, -0.2, self.cube_half_size]),
        )

        # Define source and target objects
        self.source_objs = [self._objs[0], self._objs[1], self.cube1, self.cube2]
        self.target_objs = self.source_objs.copy()

        # Randomly select source and target
        import random
        self.source_obj = random.choice(self.source_objs)
        available_targets = [obj for obj in self.target_objs if obj != self.source_obj]
        self.target_obj = random.choice(available_targets)

        # Special case for pouring tasks
        self.is_pour = False
        if "cup" in self.source_obj.name or "bottle" in self.source_obj.name:
            self.is_pour = random.choice([True, False])

        self._generate_target_pose()

    def _generate_target_pose(self):
        """Generate a valid target pose within robot's workspace"""
        # Define reachable workspace for Realman
        x_range = (-0.2, 0.5)
        y_range = (-0.3, 0.3)

        # Ensure target is away from source
        while True:
            target_x = self._episode_rng.uniform(*x_range)
            target_y = self._episode_rng.uniform(*y_range)
            target_pos = np.array([target_x, target_y, 0])

            # Check distance from source
            dist_to_source = np.linalg.norm(
                target_pos[:2] - self.source_obj.pose.p.cpu().numpy()[0][:2]
            )
            if dist_to_source > 0.15:  # Minimum separation
                break

        self.target_pos = target_pos

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            # Height to place object on table
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)

            # Initialize objects...
            for i, obj in enumerate(self._objs):
                xyz = obj.pose.p + torch.randn_like(obj.pose.p) * 0.01
                xyz[:, 2] = self.object_zs[i]
                obj.pose.p = xyz

            # Initialize cubes...
            xyz = self.cube1.pose.p + torch.randn_like(self.cube1.pose.p) * 0.01
            xyz[:, 2] = self.cube_half_size
            self.cube1.pose.p = xyz

            xyz = self.cube2.pose.p + torch.randn_like(self.cube2.pose.p) * 0.01
            xyz[:, 2] = self.cube_half_size
            self.cube2.pose.p = xyz

            # Initialize Realman robot with fixed head joints
            qpos = np.zeros(len(self.agent.robot.get_active_joints()))

            # Head joints - fixed at 0
            qpos[0:2] = [0.0, 0.0]  # head_joint1, head_joint2

            # Arm joints are interleaved: r1,l1,r2,l2,r3,l3...
            # Default right arm pose
            right_arm_qpos = [0.82383, -0.17647, 0, -1.44114, -2.09, 0.926211, 1.55079]
            # Default left arm pose
            left_arm_qpos = [-0.81842, -0.176488, 0, -1.37455, -0.875422, -0.997037, -1.54696]

            # Set the interleaved arm joints
            for i in range(7):
                qpos[2 + i * 2] = right_arm_qpos[i]  # r_joint[i+1]
                qpos[2 + i * 2 + 1] = left_arm_qpos[i]  # l_joint[i+1]

            # Gripper joints - set to the open position
            for i in range(16, len(qpos)):
                qpos[i] = 0.0  # Default open position

            # Only add noise to arm joints, not the head
            if self.robot_init_qpos_noise > 0:
                arm_noise = self._episode_rng.normal(0, self.robot_init_qpos_noise, 14)
                qpos[2:16] += arm_noise  # Only affects arm joints

            self.agent.reset(qpos)

            # Fix the drive properties of the head joints so they stay still
            head_joints = [
                self.agent.robot.get_active_joints()[0],  # head_joint1
                self.agent.robot.get_active_joints()[1]  # head_joint2
            ]

            for joint in head_joints:
                joint.set_drive_properties(
                    stiffness=10000,  # Very high stiffness
                    damping=1000,  # High damping
                    force_limit=1000
                )

    def evaluate(self):
        """Evaluate if the task is completed successfully"""
        # Check object placement
        obj_to_goal_pos = self.source_obj.pose.p - torch.tensor(
            self.target_pos, device=self.device
        )
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh

        # Check if any arm is grasping
        is_left_grasping = self.agent.is_left_grasping(self.source_obj)
        is_right_grasping = self.agent.is_right_grasping(self.source_obj)
        is_grasped = torch.logical_or(is_left_grasping, is_right_grasping)

        # Check if robot is static (at least one arm should be static)
        is_robot_static = self.agent.is_static(0.2)

        # Success: object placed and robot static
        success = torch.logical_and(is_obj_placed, is_robot_static)

        return dict(
            is_grasped=is_grasped,
            is_left_grasping=is_left_grasping,
            is_right_grasping=is_right_grasping,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=success,
        )

    def _get_obs_extra(self, info: Dict):
        """Get extra observations"""
        obs = dict(
            left_tcp_pose=self.agent.left_tcp.pose.raw_pose,
            right_tcp_pose=self.agent.right_tcp.pose.raw_pose,
            is_grasped=info["is_grasped"],
            is_obj_placed=info["is_obj_placed"],
            target_pos=torch.tensor(self.target_pos, device=self.device),
        )

        if "state" in self.obs_mode:
            obs.update(
                left_tcp_to_obj=self.source_obj.pose.p - self.agent.left_tcp.pose.p,
                right_tcp_to_obj=self.source_obj.pose.p - self.agent.right_tcp.pose.p,
                obj_pose=self.source_obj.pose.raw_pose,
                obj_to_goal_pos=info["obj_to_goal_pos"],
                left_gripper_open=self.agent.get_left_gripper_openness(),
                right_gripper_open=self.agent.get_right_gripper_openness(),
            )

        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """Compute dense reward for learning"""
        # Distance from closest TCP to object
        left_tcp_to_obj_dist = torch.linalg.norm(
            self.source_obj.pose.p - self.agent.left_tcp.pose.p, axis=1
        )
        right_tcp_to_obj_dist = torch.linalg.norm(
            self.source_obj.pose.p - self.agent.right_tcp.pose.p, axis=1
        )

        min_tcp_dist = torch.minimum(left_tcp_to_obj_dist, right_tcp_to_obj_dist)
        reaching_reward = 1 - torch.tanh(5 * min_tcp_dist)
        reward = reaching_reward

        # Grasping reward
        is_grasped = info["is_grasped"]
        reward += is_grasped * 2

        # Placement reward
        obj_to_goal_dist = torch.linalg.norm(info["obj_to_goal_pos"], axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        # Final placement reward
        reward += info["is_obj_placed"] * is_grasped * 2

        # Static reward (robot should stop moving when object is placed)
        left_arm_vel = torch.linalg.norm(self.agent.robot.get_qvel()[..., :7], axis=1)
        right_arm_vel = torch.linalg.norm(self.agent.robot.get_qvel()[..., 7:14], axis=1)
        min_arm_vel = torch.minimum(left_arm_vel, right_arm_vel)
        static_reward = 1 - torch.tanh(5 * min_arm_vel)
        reward += static_reward * info["is_obj_placed"] * is_grasped

        # Success bonus
        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8