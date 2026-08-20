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
from mani_skill.utils.building import articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_robotwin_model,
    apply_l3_partnet_ids,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    grasp_reward,
    normalized_progress,
    reach_reward,
    tanh_reward,
    transport_reward,
)


REWARD_PHASES = [
    "reach_food",
    "grasp_food",
    "transport_food",
    "place_food",
    "close_approach",
    "close_box",
]


@register_env("TwoRobotPutBox-v1", max_episode_steps=100)
class TwoRobotPutBoxEnv(BaseEnv):
    """
    **Task Description:**
    Two robots working with a single tennis ball and a single box.
    Both robots are positioned side-by-side in front of the table:
    - Left robot at position [-0.60, -0.3, 0]
    - Right robot at position [-0.60, 0.3, 0]

    Both robots face the table and can collaborate to manipulate the object.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    goal_thresh = 0.1
    lid_open_angle = 1.0
    lid_closed_thresh = 0.3

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
        self.food_ever_in_box = None
        self.food_modelname, self.food_model_id = apply_l2_robotwin_model(
            "075_bread",
            model_id=3,
            override_name="006_hamburg",
            override_id=5,
        )
        self.box_model_id = apply_l3_partnet_ids(
            ["100671"], override_ids=["102456"]
        )[0]

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

        # Load tennis ball (YCB 056_food)
        # food_builder = actors.get_actor_builder(
        #     self.scene, id=f'ycb:026_sponge', scales=[0.2,0.2,0.2]
        # )
        # food_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        # self.food = food_builder.build(name='food')

        food_pose = sapien.Pose(
            p=[0.0, 0.0, 0.0],
            q=euler2quat(0, 0, np.pi / 2)
        )
        food_actor_obj = create_actor(
            scene=self.scene,
            pose=food_pose,
            modelname=self.food_modelname,
            convex=True,
            model_id=self.food_model_id,
            scale=[1.0,1.0,1.0]
        )
        self.food = food_actor_obj.actor
        food_actor_obj.set_mass(0.5)



        # Removed Golf Ball loading code here

        # Load one partnet box
        self._boxes = []
        for i in range(self.num_envs):
            # First box (Only one now)
            box1_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{self.box_model_id}", mode="box", scale=0.15
            )
            box1_builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            box1_builder.set_scene_idxs([i])
            box1 = box1_builder.build(name=f"box1-{i}")
            self.remove_from_state_dict_registry(box1)

            # Removed Second box loading code here

            self._boxes.append(box1) # Changed from [box1, box2] to just box1

        # Merge all boxes into one articulation
        from mani_skill.utils.structs import Articulation
        self.box1 = Articulation.merge(self._boxes, name="box1") # Removed list comprehension logic for multiple boxes
        # Removed self.box2 merging
        self.add_to_state_dict_registry(self.box1)
        # Removed self.box2 registry addition

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Override robot positions set by TableSceneBuilder
            # Place both robots side-by-side in front of the table
            self.left_agent.robot.set_pose(sapien.Pose(p=[-0.60, -0.3, 0]))
            self.right_agent.robot.set_pose(sapien.Pose(p=[-0.60, 0.3, 0]))

            # Initialize tennis ball - on the left side (or center since it's alone)
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.25  
            xyz[:, 1] = -0.30 # Changed Y position slightly towards center
            xyz[:, 2] = self._food_z
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(0.05, -0.1))
            # Identity quaternion for no rotation
            qs = euler2quat(np.pi/2 ,0, np.pi/2)
            self.food.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Removed Golf Ball initialization code here

            # Initialize first box - behind tennis ball
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.1
            xyz[:, 1] = 0.0 # Centered Y for the single box
            xyz[:, 2] = self.box_zs[env_idx]
            # Upright orientation with rotation
            base_quat = euler2quat(0, 0, -np.pi / 10)
            qs = torch.tensor([base_quat] * b, device=self.device, dtype=torch.float32)
            self.box1.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Removed Second Box initialization code here

            # Set box lid angles (Box 100671 has 1 revolute joint)
            self.set_box_lid_angles(
                self.box1, lid0_angle=self.lid_open_angle, env_idx=env_idx
            )
            # Removed self.box2 lid setting

            # Define target lid angles for success condition
            self.target_lid0_closed = 0  # Closed position for lid0

            if self.food_ever_in_box is None:
                self.food_ever_in_box = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            self.food_ever_in_box[env_idx] = False

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

    def set_box_lid_angles(self, box, lid0_angle: float, env_idx: torch.Tensor = None):
        """
        Directly set the angle of the box lid.

        Box 100671 has 1 revolute joint.

        Args:
            box: The box articulation (box1)
            lid0_angle: Angle for joint_0 in radians
            env_idx: Environment indices
        """
        b = len(env_idx)
        # Create qpos tensor: [batch_size, num_joints]
        # Box 100671 has 1 active joint
        qpos = torch.zeros((b, 1), device=self.device, dtype=torch.float32)
        qpos[:, 0] = lid0_angle  # joint_0 (the lid)
        box.set_qpos(qpos)

    def _after_reconfigure(self, options: dict):
        # Get z-offset for tennis ball to place it on table surface
        self._food_z = None
        collision_mesh = self.food.get_first_collision_mesh()
        # this value is used to set object pose so the bottom is at z=0
        self._food_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

        # Removed Golf Ball z-offset calculation

        # Get z-offset for boxes to place them on table surface
        self.box_zs = []
        for box in self._boxes: # Changed from 'for boxes in self._boxes' since it's a flat list now
            # box = boxes[0] # Removed unboxing since it's already a box object
            collision_mesh = box.get_first_collision_mesh()
            if collision_mesh is not None:
                self.box_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
            else:
                self.box_zs.append(0.1)  # Default height for box
        self.box_zs = common.to_tensor(self.box_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        # Check grasping status
        is_left_grasping = self.left_agent.is_grasping(self.food)
        is_right_grasping = self.right_agent.is_grasping(self.food)
        is_grasped = is_left_grasping | is_right_grasping

        food_goal_pos = self._get_food_goal_pos()
        food_to_box_pos = self.food.pose.p - food_goal_pos

        # Check if food is in box
        food_box_dist = torch.linalg.norm(food_to_box_pos, dim=-1)
        is_food_in_box = food_box_dist <= self.goal_thresh
        is_food_placed = is_food_in_box & (~is_grasped)
        self.food_ever_in_box = self.food_ever_in_box | is_food_placed

        lid_angle = self.box1.qpos[:, 0]
        close_progress = normalized_progress(
            lid_angle,
            start=self.lid_open_angle,
            target=self.target_lid0_closed,
        )
        box1_lid0_closed = lid_angle <= self.lid_closed_thresh

        # Check if robots are static
        is_robot_static = torch.logical_and(
            self.left_agent.is_static(0.2),
            self.right_agent.is_static(0.2)
        )

        success = self.food_ever_in_box

        result = dict(
            is_left_grasping=is_left_grasping,
            is_right_grasping=is_right_grasping,
            is_grasped=is_grasped,
            food_box_dist=food_box_dist,
            food_to_box_pos=food_to_box_pos,
            is_food_in_box=is_food_in_box,
            is_food_placed=is_food_placed,
            food_ever_in_box=self.food_ever_in_box,
            lid_angle=lid_angle,
            close_progress=close_progress,
            box1_lid0_closed=box1_lid0_closed,
            is_box_closed=box1_lid0_closed,
            is_robot_static=is_robot_static,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_food_goal_pos(self) -> torch.Tensor:
        return self.box1.pose.p

    def _get_transport_target_pos(self) -> torch.Tensor:
        offset = torch.tensor(
            [0.0, 0.0, 0.15],
            device=self.device,
            dtype=self.box1.pose.p.dtype,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        return self.box1.pose.p + offset

    def _get_close_approach_target_pos(self) -> torch.Tensor:
        offset = torch.tensor(
            [0.15, 0.0, 0.12],
            device=self.device,
            dtype=self.box1.pose.p.dtype,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        return self.box1.pose.p + offset

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            is_left_grasping=info["is_left_grasping"],
            is_right_grasping=info["is_right_grasping"],
        )
        if "state" in self.obs_mode:
            obs.update(
                food_pose=self.food.pose.raw_pose,
                # Removed golf_ball_pose
                box1_pose=self.box1.pose.raw_pose,
                # Removed box2_pose
                left_tcp_to_food=self.food.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_food=self.food.pose.p - self.right_agent.tcp.pose.p, # Changed from golf ball to tennis ball for right agent obs
            )
        return obs

    def _prioritize_reward_info(self, info: Dict, total_reward: torch.Tensor):
        ordered = dict(reward=total_reward.clone())
        if "success" in info:
            ordered["success"] = info["success"]
        for key in list(info.keys()):
            if key.startswith("R"):
                ordered[key] = info[key]
        for key in list(info.keys()):
            if key.startswith("peak_r_"):
                ordered[key] = info[key]
        for key, value in info.items():
            if key not in ordered:
                ordered[key] = value
        info.clear()
        info.update(ordered)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        food_pos = self.food.pose.p
        transport_target = self._get_transport_target_pos()
        close_target = self._get_close_approach_target_pos()

        is_left_grasping = info["is_left_grasping"]
        is_grasped = info["is_grasped"]
        food_box_dist = info["food_box_dist"]
        food_ever_in_box = info["food_ever_in_box"]
        lid_angle = info["lid_angle"]
        box_closed = info["box1_lid0_closed"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(left_tcp, food_pos, scale=5.0)
        r_reach = torch.where(is_left_grasping | food_ever_in_box, ones, r_reach)
        self.reward_tracker.update("reach_food", r_reach)

        r_grasp = grasp_reward(
            left_tcp, food_pos, is_left_grasping, proximity_scale=5.0
        )
        r_grasp = torch.where(food_ever_in_box, ones, r_grasp)
        self.reward_tracker.update("grasp_food", r_grasp)

        r_transport = transport_reward(
            food_pos, transport_target, is_left_grasping, scale=5.0
        )
        r_transport = torch.where(food_ever_in_box, ones, r_transport)
        self.reward_tracker.update("transport_food", r_transport)

        r_place = tanh_reward(food_box_dist, scale=5.0) * (~is_grasped).float()
        r_place = torch.where(food_ever_in_box, ones, r_place)
        self.reward_tracker.update("place_food", r_place)

        r_close_approach = reach_reward(left_tcp, close_target, scale=6.0)
        r_close_approach = r_close_approach * food_ever_in_box.float()
        r_close_approach = torch.where(box_closed, ones, r_close_approach)
        self.reward_tracker.update("close_approach", r_close_approach)

        r_close = normalized_progress(
            lid_angle,
            start=self.lid_open_angle,
            target=self.target_lid0_closed,
        ) * food_ever_in_box.float()
        r_close = torch.where(box_closed & food_ever_in_box, ones, r_close)
        self.reward_tracker.update("close_box", r_close)

        self.reward_tracker.write_to_info(info)
        total_reward = self.reward_tracker.total()
        self._prioritize_reward_info(info, total_reward)
        return total_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
