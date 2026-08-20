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
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    is_l2_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
    exp_reward,
)

# Sub-task phases for this task
REWARD_PHASES = ["reach_screwdriver", "grasp_screwdriver", "transport_screwdriver", "touch_cube"]


@register_env("TwoRobotPlaceScrewdriver-v1", max_episode_steps=100)
class TwoRobotPlaceScrewdriverEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.15

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
        self.screwdriver_modelname, self.screwdriver_model_id = "032_screwdriver", 0

        self.displaystand_modelname, self.displaystand_model_id = "074_displaystand", 0

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

        # Load screwdriver
        screwdriver_q = euler2quat(0, 0, -np.pi / 2)
        screwdriver_p = np.array([-0.23, -0.35, 0.03], dtype=np.float32)
        self.screwdriver_pose = sapien.Pose(
            p=screwdriver_p.tolist(),
            q=screwdriver_q
        )
        screwdriver_actor_obj = create_actor(
            scene=self.scene,
            pose=self.screwdriver_pose,
            modelname=self.screwdriver_modelname,
            convex=True,
            model_id=self.screwdriver_model_id,
            scale=(0.13, 0.13, 0.13),
            replace_scale=True,
        )
        self.screwdriver = screwdriver_actor_obj.actor
        self.screwdriver.set_mass(0.5)

        # Load Displaystand
        self.displaystand_pose = sapien.Pose(
            p=[-0.23, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        displaystand_actor_obj = create_actor(
            scene=self.scene,
            pose=self.displaystand_pose,
            modelname=self.displaystand_modelname,
            convex=True,
            model_id=self.displaystand_model_id,
            is_static=True,
            replace_scale=True,
            scale=(0.2, 0.2, 0.2),
        )
        self.displaystand = displaystand_actor_obj.actor

        # Load cube
        self.cube_pose = sapien.Pose(
            p=[-0.23, 0.0, 0.16],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        color = [1, 0, 0, 1]
        if is_l2_enabled():
            color = [1, 1, 0, 1]
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=color,
            name="cube",
            body_type="static",
            initial_pose=self.cube_pose,
        )
        

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

            if not hasattr(self, "screwdriver_ever_touched_cube"):
                self.screwdriver_ever_touched_cube = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.screwdriver_ever_touched_cube[env_idx] = False

            xyz = torch.tensor(self.screwdriver_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.0
            xyz = apply_l1_offset_xy(xyz, offset=(0.1, -0.1))
            qs = torch.tensor(self.screwdriver_pose.q).repeat(b, 1)
            self.screwdriver.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.cube_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.0
            qs = torch.tensor(self.cube_pose.q).repeat(b, 1)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.displaystand_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.0
            xyz[:, 2] = self.displaystand_zs[env_idx]
            qs = torch.tensor(self.displaystand_pose.q).repeat(b, 1)
            self.displaystand.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.screwdriver_zs = []
        collision_mesh = self.screwdriver.get_first_collision_mesh()
        self.screwdriver_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.screwdriver_zs = common.to_tensor(self.screwdriver_zs, device=self.device)

        self.cube_zs = []
        collision_mesh = self.cube.get_first_collision_mesh()
        self.cube_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cube_zs = common.to_tensor(self.cube_zs, device=self.device)

        self.displaystand_zs = []
        collision_mesh = self.displaystand.get_first_collision_mesh()
        self.displaystand_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.displaystand_zs = common.to_tensor(self.displaystand_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        # 1. Check for contact using geom-center distance threshold
        screwdriver_center = self._get_actor_geom_center(self.screwdriver).repeat(self.num_envs, 1)
        cube_center = self._get_actor_geom_center(self.cube).repeat(self.num_envs, 1)
        screwdriver_to_cube_dist = torch.linalg.norm(screwdriver_center - cube_center, axis=1)
        is_colliding = screwdriver_to_cube_dist <= 0.16
        
        is_grasped = self.left_agent.is_grasping(self.screwdriver) or self.right_agent.is_grasping(self.screwdriver)
        
        # Condition: colliding while grasped
        currently_touched = is_colliding & is_grasped
        self.screwdriver_ever_touched_cube = self.screwdriver_ever_touched_cube | currently_touched

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        
        success = self.screwdriver_ever_touched_cube

        # debug
        # print(
        #     "Eval Info: ",
        #     "screwdriver_to_cube_dist=", screwdriver_to_cube_dist,
        #     "is_colliding=", is_colliding,
        #     "is_grasped=", is_grasped,
        #     "currently_touched=", currently_touched,
        #     "screwdriver_ever_touched_cube=", self.screwdriver_ever_touched_cube,
        #     "success=", success,
        # )

        result = dict(
            is_grasped=is_grasped,
            screwdriver_ever_touched_cube=self.screwdriver_ever_touched_cube,
            is_robot_static=is_robot_static,
            success=success,
        )
        # Append per-phase peak sub-rewards
        # if hasattr(self, "reward_tracker"):
        #     # result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        collision_mesh = obj.get_first_collision_mesh(to_world_frame=True)
        bounds = collision_mesh.bounding_box.bounds
        center = (bounds[0] + bounds[1]) * 0.5
        return common.to_tensor(center, device=self.device)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            screwdriver_pose=self.screwdriver.pose.raw_pose,
            cube_pose=self.cube.pose.raw_pose,
            is_grasped=info["is_grasped"],
            screwdriver_ever_touched_cube=info["screwdriver_ever_touched_cube"],
        )
        if "state" in self.obs_mode:
            obs.update(
                left_tcp_to_obj_pos=self.screwdriver.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.screwdriver.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.screwdriver.pose.p - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        screwdriver_pos = self.screwdriver.pose.p
        cube_pos = self.cube.pose.p

        is_grasped = info["is_grasped"]
        screwdriver_ever_touched_cube = info["screwdriver_ever_touched_cube"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Use whichever arm is closer
        left_tcp_to_obj_dist = torch.linalg.norm(screwdriver_pos - left_tcp, axis=1)
        right_tcp_to_obj_dist = torch.linalg.norm(screwdriver_pos - right_tcp, axis=1)
        closer_tcp = torch.where(left_tcp_to_obj_dist < right_tcp_to_obj_dist, left_tcp, right_tcp)

        # Phase 1: REACH_SCREWDRIVER - only before grasping
        reach_gate = ~is_grasped
        r_reach_screwdriver = reach_reward(closer_tcp, screwdriver_pos, scale=5.0) * reach_gate.float()
        r_reach_screwdriver = torch.where(is_grasped, ones, r_reach_screwdriver)
        self.reward_tracker.update("reach_screwdriver", r_reach_screwdriver)

        # Phase 2: GRASP_SCREWDRIVER - only after reaching, before touching cube
        r_grasp_screwdriver = grasp_reward(closer_tcp, screwdriver_pos, is_grasped, proximity_scale=5.0) 
        r_grasp_screwdriver = torch.where(is_grasped, ones, r_grasp_screwdriver)
        self.reward_tracker.update("grasp_screwdriver", r_grasp_screwdriver)

        # Phase 3: TRANSPORT_SCREWDRIVER - only after grasping, before touching cube
        transport_gate = is_grasped 
        r_transport_screwdriver = transport_reward(screwdriver_pos[:, :2], cube_pos[:, :2], is_grasped, scale=5.0) * transport_gate.float()
        r_transport_screwdriver = torch.where(success, ones, r_transport_screwdriver)
        self.reward_tracker.update("transport_screwdriver", r_transport_screwdriver)

        # Phase 4: TOUCH_CUBE - historical achievement (continuous based on distance)
        r_touch = success.float() * is_grasped.float()
        self.reward_tracker.update("touch_cube", r_touch)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
