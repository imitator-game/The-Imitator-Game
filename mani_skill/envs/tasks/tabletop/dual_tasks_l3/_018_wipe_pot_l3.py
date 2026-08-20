import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, Tuple

from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from transforms3d.euler import euler2quat
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    transport_reward,
    wipe_progress_reward,
    above_reward,
    tanh_reward,
    RewardTracker,
)

REWARD_PHASES = ['reach', 'grasp', 'approach', 'wipe']


@register_env("TwoRobotWipePotL3-v1", max_episode_steps=100)
class TwoRobotWipePotEnvL3(BaseEnv):
    """
        **Task Description:**
        The goal is to pick up a brush and wipe the surface of the pot three consecutive times.
        Each wipe requires the brush to make contact with the pot and move across its surface.
        After completing three wipes, the brush must be placed back to its original position.

        **Randomizations:**
        The initial position of the brush are randomized within a reachable area.
        The position and orientation of the pot are randomized on the workspace.

        **Success Conditions:**
        Just For Testing.
    """

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
        self.goal_thresh = 0.22

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
        # global WARNED_ONCE
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Load brush
        self.brush_modelname, self.brush_model_id = (
            "083_brush",
            0,
        )
        self.brush_pose = sapien.Pose(
            p=[0.1, -0.2, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 2)
        )
        brush_actor_obj = create_actor(
            scene=self.scene,
            pose=self.brush_pose,
            modelname=self.brush_modelname,
            convex=True,
            model_id=self.brush_model_id,
            scale=(0.13, 0.13, 0.13),
            replace_scale=True,
        )
        self.brush = brush_actor_obj.actor
        self.brush.set_mass(0.5)

        partnet_category = 'Kitchenpot'
        partnet_model_id = "100051"
        partnet_position = [0.0, 0.0, 0.0]
        partnet_rotation = [0, 0, np.pi/2]

        pose = sapien.Pose(p=partnet_position, q=euler2quat(*partnet_rotation))

        partnet_builder = articulations.get_articulation_builder(
            self.scene,
            f'partnet-mobility:{partnet_model_id}',
            mode=partnet_category.lower(),
            scale=0.1
        )
        partnet_builder.set_scene_idxs(scene_idxs=[0])
        partnet_builder.initial_pose = pose
        self.pot = partnet_builder.build(name=f'{partnet_category}-{partnet_model_id}', fix_root_link=True)

        PARTNET_COLLISION_BIT = 29

        # Set collision properties
        for link in self.pot.links:
            link.set_collision_group_bit(
                group=2, bit_idx=PARTNET_COLLISION_BIT, bit=1
            )

            # Add safety boundaries
            if partnet_category.lower() in ['microwave', 'oven', 'dishwasher', 'refrigerator', 'cabinet',
                                            'storagefurniture']:
                link.set_collision_group_bit(group=2, bit_idx=28, bit=1)
            elif partnet_category.lower() in ['box', 'suitcase', 'bucket', 'trashcan']:
                link.set_collision_group_bit(group=2, bit_idx=27, bit=1)
            else:
                link.set_collision_group_bit(group=2, bit_idx=26, bit=1)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "brush_ever_wipe_pot"):
                self.brush_ever_wipe_pot = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.brush_ever_wipe_pot[env_idx] = False
            if not hasattr(self, "brush_last_pos"):
                self.brush_last_pos = self.brush.pose.p.clone()
            if not hasattr(self, "brush_wipe_dist"):
                self.brush_wipe_dist = torch.zeros(self.num_envs, device=self.device)
            if not hasattr(self, "brush_wipe_steps"):
                self.brush_wipe_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
            self.brush_last_pos[env_idx] = self.brush.pose.p[env_idx].clone()
            self.brush_wipe_dist[env_idx] = 0.0
            self.brush_wipe_steps[env_idx] = 0

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

            # TODO: adding pose randomization
            xyz = self.brush.pose.p
            qs = self.brush.pose.q
            xyz[:, :2] += torch.tensor([-0.1, -0.15])
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self._brush_z
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, 0.0))
            # qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.brush.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = self.pot.pose.p
            qs = self.pot.pose.q
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self._pot_z
            # qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.pot.set_pose(Pose.create_from_pq(xyz, qs))


    def _after_reconfigure(self, options: dict):
        self._brush_z = None
        collision_mesh = self.brush.get_first_collision_mesh()
        # this value is used to set object pose so the bottom is at z=0
        self._brush_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

        self._pot_z = None
        collision_mesh = self.pot.get_first_collision_mesh()
        # this value is used to set object pose so the bottom is at z=0
        self._pot_z = common.to_tensor(-collision_mesh.bounding_box.bounds[0, 2], device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        is_grasped = self.left_agent.is_grasping(self.brush)

        brush_to_pot_dist = torch.linalg.norm(self.brush.pose.p - self.pot.pose.p, axis=1)
        is_brush_on_pot = brush_to_pot_dist < self.goal_thresh
        is_contact = is_brush_on_pot & is_grasped
        self.brush_ever_wipe_pot = self.brush_ever_wipe_pot | is_contact

        # Accumulate wipe distance and steps while in contact.
        delta_xy = self.brush.pose.p[:, :2] - self.brush_last_pos[:, :2]
        step_dist = torch.linalg.norm(delta_xy, axis=1)
        self.brush_wipe_dist += step_dist * is_contact.float()
        self.brush_wipe_steps += is_contact.int()
        self.brush_last_pos = self.brush.pose.p.clone()

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

        success = (self.brush_wipe_steps >= 80) & (self.brush_wipe_dist >= 0.2)

        result = dict(
            is_grasped=is_grasped,
            is_brush_on_pot=is_brush_on_pot,
            brush_to_pot_dist=brush_to_pot_dist,
            brush_wipe_dist=self.brush_wipe_dist,
            brush_wipe_steps=self.brush_wipe_steps,
            success=success,
            brush_ever_wipe_pot=self.brush_ever_wipe_pot,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            is_grasped=info["is_grasped"],
        )
        # if "state" in self.obs_mode:
        #     obs.update(
        #     )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        brush_pos = self.brush.pose.p
        pot_pos = self.pot.pose.p
        is_grasped = info["is_grasped"]
        ever_on_pot = info.get("brush_ever_wipe_pot", self.brush_ever_wipe_pot)
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, brush_pos, scale=5.0)
        r_reach = torch.where(is_grasped | ever_on_pot | success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        r_grasp = grasp_reward(tcp_pos, brush_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(ever_on_pot | success, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        r_approach = transport_reward(brush_pos, pot_pos, is_grasped, scale=5.0)
        r_approach = torch.where(ever_on_pot | success, ones, r_approach)
        self.reward_tracker.update("approach", r_approach)

        r_wipe = wipe_progress_reward(self.brush_wipe_steps, 80, self.brush_wipe_dist, 0.2)
        r_wipe = torch.where(success, ones, r_wipe)
        self.reward_tracker.update("wipe", r_wipe)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)