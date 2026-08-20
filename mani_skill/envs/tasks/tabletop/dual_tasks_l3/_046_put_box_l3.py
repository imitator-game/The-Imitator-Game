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
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_robotwin_model,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    grasp_reward,
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


@register_env("TwoRobotPutBoxL3-v1", max_episode_steps=100)
class TwoRobotPutBoxEnvL3(BaseEnv):
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
    goal_thresh = 0.2
    tray_closed_thresh = 0.09

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
        # Fixed RobotWin container assets (no L3 partnet replacement).
        self.plasticbox_modelname = "062_plasticbox"
        self.plasticbox_model_id = get_model_id(self.plasticbox_modelname, model_id=7)
        self.tray_modelname = "008_tray"
        self.tray_model_id = get_model_id(self.tray_modelname, model_id=0)
        # Tunable scales (multiplier over model_data.json scale).
        self.plasticbox_scale = (1.2, 1.2, 1.2)
        self.tray_scale = (1.2, 1.0, 1.3)
        # Tray local/world fine-tune offset relative to plasticbox anchor.
        self.tray_offset = np.array([0.15, 0.0, 0.0], dtype=np.float32)

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
            q=euler2quat(0, 0, 0)
        )
        food_actor_obj = create_actor(
            scene=self.scene,
            pose=food_pose,
            modelname=self.food_modelname,
            convex=True,
            model_id=self.food_model_id,
            scale=[1.0, 1.0, 1.0]
        )
        food_actor_obj.set_mass(0.5)
        self.food = food_actor_obj.actor
        plasticbox_actor_obj = create_actor(
            scene=self.scene,
            pose=sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(np.pi / 2, 0.0, 0.0)),
            modelname=self.plasticbox_modelname,
            convex=True,
            model_id=self.plasticbox_model_id,
            is_static=True,
            scale=self.plasticbox_scale,
            replace_scale=False,
        )
        self.plasticbox = plasticbox_actor_obj.actor

        tray_actor_obj = create_actor(
            scene=self.scene,
            pose=sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(np.pi / 2, 0.0, np.pi)),
            modelname=self.tray_modelname,
            convex=True,
            model_id=self.tray_model_id,
            is_static=False,
            scale=self.tray_scale,
            replace_scale=False,
        )
        tray_actor_obj.set_mass(0.5)
        self.tray = tray_actor_obj.actor
        # Keep legacy name for solution/reward compatibility.
        self.box1 = self.plasticbox

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Override robot positions set by TableSceneBuilder
            # Place both robots side-by-side in front of the table
            self.left_agent.robot.set_pose(sapien.Pose(p=[-0.60, -0.3, 0]))
            self.right_agent.robot.set_pose(sapien.Pose(p=[-0.60, 0.3, 0]))

            # Initialize movable PartNet box (used as the grasp target)
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 0] = -0.25  
            xyz[:, 1] = -0.30 # Changed Y position slightly towards center
            xyz[:, 2] = self._food_z
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(0.05, -0.1))
            qs = euler2quat(np.pi/2 ,0, np.pi/2)
            self.food.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Initialize plasticbox at previous box anchor pose.
            box_xyz = torch.zeros((b, 3), device=self.device)
            box_xyz[:, 0] = -0.1
            box_xyz[:, 1] = 0.0
            box_xyz[:, 2] = self.plasticbox_z
            box_q = torch.tensor(
                [euler2quat(np.pi / 2, 0.0, 0.0)] * b,
                device=self.device,
                dtype=torch.float32,
            )
            self.plasticbox.set_pose(Pose.create_from_pq(p=box_xyz, q=box_q))

            # Initialize tray at same XY, above plasticbox, with extra tunable XYZ offset.
            tray_xyz = box_xyz.clone()
            tray_xyz[:, 0] += float(self.tray_offset[0])
            tray_xyz[:, 1] += float(self.tray_offset[1])
            tray_xyz[:, 2] = (
                self.plasticbox_z
                + self.plasticbox_height
                + self.tray_z
                + float(self.tray_offset[2])
            )
            tray_q = torch.tensor(
                [euler2quat(np.pi / 2, 0.0, np.pi)] * b,
                device=self.device,
                dtype=torch.float32,
            )
            self.tray.set_pose(Pose.create_from_pq(p=tray_xyz, q=tray_q))

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

            if not hasattr(self, "tray_open_dist"):
                self.tray_open_dist = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.float32
                )
            closed_target = box_xyz.clone()
            closed_target[:, 2] = (
                self.plasticbox.pose.p[env_idx, 2]
                + self.plasticbox_height
                + self.tray_z
                + float(self.tray_offset[2])
            )
            self.tray_open_dist[env_idx] = torch.linalg.norm(
                tray_xyz - closed_target, dim=-1
            )

    def set_box_lid_angles(self, box, lid0_angle: float, env_idx: torch.Tensor = None):
        """
        Directly set the angle of box joint_0 if available.

        Args:
            box: The box articulation (box1)
            lid0_angle: Angle for joint_0 in radians
            env_idx: Environment indices
        """
        b = len(env_idx)
        dof = int(getattr(box, "max_dof", 0))
        if dof <= 0:
            return
        qpos = torch.zeros((b, dof), device=self.device, dtype=torch.float32)
        qpos[:, 0] = lid0_angle
        box.set_qpos(qpos)

    def _after_reconfigure(self, options: dict):
        # Get z-offset for tennis ball to place it on table surface
        self._food_z = None
        collision_mesh = self.food.get_first_collision_mesh()
        # this value is used to set object pose so the bottom is at z=0
        self._food_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

        collision_mesh = self.plasticbox.get_first_collision_mesh()
        if collision_mesh is not None:
            bounds = collision_mesh.bounding_box.bounds
            self.plasticbox_z = common.to_tensor(-bounds[0, 2], device=self.device)
            self.plasticbox_height = common.to_tensor(bounds[1, 2] - bounds[0, 2], device=self.device)
        else:
            self.plasticbox_z = common.to_tensor(0.05, device=self.device)
            self.plasticbox_height = common.to_tensor(0.08, device=self.device)

        collision_mesh = self.tray.get_first_collision_mesh()
        if collision_mesh is not None:
            bounds = collision_mesh.bounding_box.bounds
            self.tray_z = common.to_tensor(-bounds[0, 2], device=self.device)
        else:
            self.tray_z = common.to_tensor(0.02, device=self.device)

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

        tray_closed_target = self._get_tray_closed_target_pos()
        tray_to_target_pos = self.tray.pose.p - tray_closed_target
        tray_target_dist = torch.linalg.norm(tray_to_target_pos, dim=-1)
        tray_close_progress = torch.clamp(
            1.0 - tray_target_dist / torch.clamp(self.tray_open_dist, min=1e-6),
            0.0,
            1.0,
        )
        box1_lid0_closed = tray_target_dist <= self.tray_closed_thresh

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
            tray_target_dist=tray_target_dist,
            tray_to_target_pos=tray_to_target_pos,
            close_progress=tray_close_progress,
            box1_lid0_closed=box1_lid0_closed,
            is_box_closed=box1_lid0_closed,
            is_robot_static=is_robot_static,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_food_goal_pos(self) -> torch.Tensor:
        offset = torch.tensor(
            [-0.05, 0.0, 0.0],
            device=self.device,
            dtype=self.box1.pose.p.dtype,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        return self.box1.pose.p + offset

    def _get_transport_target_pos(self) -> torch.Tensor:
        offset = torch.tensor(
            [-0.1, 0.0, 0.20],
            device=self.device,
            dtype=self.box1.pose.p.dtype,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        return self.box1.pose.p + offset

    def _get_close_approach_target_pos(self) -> torch.Tensor:
        offset = torch.tensor(
            [0.10, 0.0, 0.145],
            device=self.device,
            dtype=self.box1.pose.p.dtype,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        return self.box1.pose.p + offset

    def _get_tray_closed_target_pos(self) -> torch.Tensor:
        target = self.box1.pose.p.clone()
        target[:, 2] = (
            self.box1.pose.p[:, 2]
            + self.plasticbox_height
            + self.tray_z
            + float(self.tray_offset[2])
        )
        return target

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
        tray_target_dist = info["tray_target_dist"]
        tray_closed = info["box1_lid0_closed"]
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
        r_close_approach = torch.where(tray_closed, ones, r_close_approach)
        self.reward_tracker.update("close_approach", r_close_approach)

        r_close = torch.clamp(
            1.0 - tray_target_dist / torch.clamp(self.tray_open_dist, min=1e-6),
            0.0,
            1.0,
        ) * food_ever_in_box.float()
        r_close = torch.where(tray_closed & food_ever_in_box, ones, r_close)
        self.reward_tracker.update("close_box", r_close)

        self.reward_tracker.write_to_info(info)
        total_reward = self.reward_tracker.total()
        self._prioritize_reward_info(info, total_reward)
        return total_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
