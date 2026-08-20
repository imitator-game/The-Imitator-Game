import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, Tuple
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.building.actors.sketchfab import create_sketchfab_actor
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_ycb_model_id,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    grasp_reward,
    reach_reward,
    transport_reward,
)


REWARD_PHASES = [
    "reach_cube",
    "grasp_cube",
    "transport_cube",
    "place_cube",
    "reach_rubik",
    "grasp_rubik",
    "transport_rubik",
    "place_rubik",
]


@register_env("TwoRobotPutCubeOnScaleL3-v1", max_episode_steps=100)
class TwoRobotPutCubeOnScaleEnvL3(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.025
    scale_radius = 0.10
    scale_height_threshold = 0.20
    plate_goal_radius = 0.06
    plate_goal_z_tolerance = 0.05
    SCALE_OBJECT_KEY = "balance_scale"

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
        self.scale_object_key = self.SCALE_OBJECT_KEY
        # Scene tuning knobs for sketchfab balance scale.
        self.scale_base_pos = np.array([0.0, 0.05, 0.0], dtype=np.float32)
        self.scale_pose_offset = np.array([0.0, -0.05, 0.0], dtype=np.float32)
        self.scale_yaw_offset = 0.0
        # self.all_model_ids = np.array(["005_tomato_soup_can"])
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
                found_lost_pairs_capacity=2 ** 25,
                max_rigid_patch_count=2 ** 19,
                max_rigid_contact_count=2 ** 21,
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

        # Load sketchfab balance scale (same as task 009 L3).
        self.scale_pose = sapien.Pose(
            p=(self.scale_base_pos + self.scale_pose_offset).tolist(),
            q=euler2quat(np.pi / 2, 0, -np.pi / 2 + self.scale_yaw_offset),
        )
        self.scale = create_sketchfab_actor(
            scene=self.scene,
            object_key=self.scale_object_key,
            pose=self.scale_pose,
            name="balance_scale",
            is_static=True,
        )

        # load cube to put on scale
        appliance_model_id = apply_l2_ycb_model_id(
            "061_foam_brick", override_id="077_rubiks_cube"
        )
        appliance_builder = actors.get_actor_builder(
            self.scene,
            id=f'ycb:{appliance_model_id}',
            scales=[0.8]
        )
        appliance_builder._mass = 0.5
        appliance_pose = sapien.Pose(p=[0.0, -0.4, -0.01], q=euler2quat(0, 0, np.pi / 2))
        appliance_builder.initial_pose = appliance_pose
        self.cube = appliance_builder.build(name=f'{appliance_model_id}')

        # Add an extra Robotwin Rubik's cube on the opposite-y side of the foam brick.
        self.rubik_pose = sapien.Pose(
            p=[0.0, 0.4, -0.01],
            q=euler2quat(0, 0, np.pi / 2),
        )
        rubik_actor_obj = create_actor(
            scene=self.scene,
            pose=self.rubik_pose,
            modelname="073_rubikscube",
            convex=True,
            model_id=1,
        )
        rubik_actor_obj.set_mass(0.5)
        self.rubik = rubik_actor_obj.actor

        self.objects = [self.scale, self.cube, self.rubik]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            xyz = torch.tensor(self.scale_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.objects_zs[self.scale][env_idx]
            original_q = torch.tensor(self.scale_pose.q)
            self.scale.set_pose(Pose.create_from_pq(p=xyz, q=original_q))

            xyz = torch.tensor(self.cube.pose.p)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.objects_zs[self.cube][env_idx]
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
            original_q = torch.tensor(self.cube.pose.sp.q)
            self.cube.set_pose(Pose.create_from_pq(p=xyz, q=original_q))
            cube_xyz = xyz.clone()

            # Keep rubik cube at opposite y position of foam brick.
            rubik_xyz = cube_xyz.clone()
            rubik_xyz[:, 1] = -cube_xyz[:, 1]
            rubik_xyz[:, 2] = self.objects_zs[self.rubik][env_idx]
            original_q = torch.tensor(self.rubik.pose.sp.q)
            self.rubik.set_pose(Pose.create_from_pq(p=rubik_xyz, q=original_q))

    def _after_reconfigure(self, options: dict):
        self.objects_zs = {}
        for object in self.objects:
            self.objects_zs[object] = []
            collision_mesh = object.get_first_collision_mesh()
            self.objects_zs[object].append(-collision_mesh.bounding_box.bounds[0, 2])
            self.objects_zs[object] = common.to_tensor(
                self.objects_zs[object], device=self.device
            )

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        scale_center = self._get_actor_geom_center(self.scale)
        cube_center = self._get_actor_geom_center(self.cube)
        rubik_center = self._get_actor_geom_center(self.rubik)
        cube_goal = scale_center + common.to_tensor(
            [[0.02, -0.10, 0.10]], device=self.device
        ).repeat(self.num_envs, 1)
        rubik_goal = scale_center + common.to_tensor(
            [[0.02, 0.11, 0.10]], device=self.device
        ).repeat(self.num_envs, 1)
        is_cube_grasped = self.left_agent.is_grasping(self.cube)
        is_rubik_grasped = self.right_agent.is_grasping(self.rubik)
        is_cube_on_scale = self._is_near_goal(cube_center, goal_pos=cube_goal)
        is_rubik_on_scale = self._is_near_goal(rubik_center, goal_pos=rubik_goal)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        success = torch.logical_and(is_cube_on_scale, is_rubik_on_scale)
        result = dict(
            is_cube_grasped=is_cube_grasped,
            is_rubik_grasped=is_rubik_grasped,
            is_cube_on_scale=is_cube_on_scale,
            is_rubik_on_scale=is_rubik_on_scale,
            cube_goal_pos=cube_goal,
            rubik_goal_pos=rubik_goal,
            cube_to_goal_pos=cube_goal - cube_center,
            rubik_to_goal_pos=rubik_goal - rubik_center,
            cube_to_scale_dist=torch.linalg.norm(cube_center - cube_goal, axis=1),
            rubik_to_scale_dist=torch.linalg.norm(rubik_center - rubik_goal, axis=1),
            is_robot_static=is_robot_static,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _is_near_goal(self, obj_pos: torch.Tensor, goal_pos: torch.Tensor) -> torch.Tensor:
        xy_dist = torch.linalg.norm(obj_pos[..., :2] - goal_pos[..., :2], axis=1)
        in_xy = xy_dist <= self.plate_goal_radius
        z_dist = torch.abs(obj_pos[..., 2] - goal_pos[..., 2])
        in_z = z_dist <= self.plate_goal_z_tolerance
        return torch.logical_and(in_xy, in_z)

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

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            scale_pos=self.scale.pose.p,
            cube_pos=self.cube.pose.p,
            rubik_pos=self.rubik.pose.p,
            is_cube_grasped=info["is_cube_grasped"],
            is_rubik_grasped=info["is_rubik_grasped"],
            is_cube_on_scale=info["is_cube_on_scale"],
            is_rubik_on_scale=info["is_rubik_on_scale"],
            cube_to_scale_dist=info["cube_to_scale_dist"],
            rubik_to_scale_dist=info["rubik_to_scale_dist"],
        )
        return obs

    def compute_dense_reward(self, obs: Any, action, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        cube_center = self._get_actor_geom_center(self.cube)
        rubik_center = self._get_actor_geom_center(self.rubik)

        cube_goal = info["cube_goal_pos"]
        rubik_goal = info["rubik_goal_pos"]
        is_cube_grasped = info["is_cube_grasped"]
        is_rubik_grasped = info["is_rubik_grasped"]
        is_cube_on_scale = info["is_cube_on_scale"]
        is_rubik_on_scale = info["is_rubik_on_scale"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_cube = reach_reward(left_tcp, cube_center, scale=5.0)
        r_reach_cube = torch.where(
            is_cube_grasped | is_cube_on_scale | success,
            ones,
            r_reach_cube,
        )
        self.reward_tracker.update("reach_cube", r_reach_cube)

        r_grasp_cube = grasp_reward(
            left_tcp, cube_center, is_cube_grasped, proximity_scale=5.0
        )
        r_grasp_cube = torch.where(
            is_cube_on_scale | success,
            ones,
            r_grasp_cube,
        )
        self.reward_tracker.update("grasp_cube", r_grasp_cube)

        r_transport_cube = transport_reward(
            cube_center, cube_goal, is_cube_grasped, scale=5.0
        )
        r_transport_cube = torch.where(
            is_cube_on_scale | success,
            ones,
            r_transport_cube,
        )
        self.reward_tracker.update("transport_cube", r_transport_cube)

        r_place_cube = is_cube_on_scale.float()
        r_place_cube = torch.where(success, ones, r_place_cube)
        self.reward_tracker.update("place_cube", r_place_cube)

        r_reach_rubik = reach_reward(right_tcp, rubik_center, scale=5.0)
        r_reach_rubik = torch.where(
            is_rubik_grasped | is_rubik_on_scale | success,
            ones,
            r_reach_rubik,
        )
        self.reward_tracker.update("reach_rubik", r_reach_rubik)

        r_grasp_rubik = grasp_reward(
            right_tcp, rubik_center, is_rubik_grasped, proximity_scale=5.0
        )
        r_grasp_rubik = torch.where(
            is_rubik_on_scale | success,
            ones,
            r_grasp_rubik,
        )
        self.reward_tracker.update("grasp_rubik", r_grasp_rubik)

        r_transport_rubik = transport_reward(
            rubik_center, rubik_goal, is_rubik_grasped, scale=5.0
        )
        r_transport_rubik = torch.where(
            is_rubik_on_scale | success,
            ones,
            r_transport_rubik,
        )
        self.reward_tracker.update("transport_rubik", r_transport_rubik)

        r_place_rubik = is_rubik_on_scale.float()
        r_place_rubik = torch.where(success, ones, r_place_rubik)
        self.reward_tracker.update("place_rubik", r_place_rubik)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
            self, obs: Any, action, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
