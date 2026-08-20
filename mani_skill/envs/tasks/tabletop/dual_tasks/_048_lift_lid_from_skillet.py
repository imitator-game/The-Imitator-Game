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
    apply_l2_ycb_model_id,
    apply_l2_robotwin_model,
    apply_l3_robotwin_model,
    is_lr_mirror_enabled,
    mirror_xyz,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    grasp_reward,
    reach_reward,
    tanh_reward,
    transport_reward,
)


REWARD_PHASES = [
    "reach_lid",
    "grasp_lid",
    "lift_lid",
    "transport_lid",
    "place_lid",
]


@register_env("TwoRobotLiftLidFromSkillet-v1", max_episode_steps=100)
class TwoRobotLiftLidFromSkilletEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.025

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
        
        # Load fork
        appliance_model_id = apply_l2_ycb_model_id(
            "028_skillet_lid", override_id="028_skillet_lid"
        )
        appliance_builder = actors.get_actor_builder(
            self.scene,
            id=f'ycb:{appliance_model_id}',
            scales=[0.6]
        )
        appliance_pose = sapien.Pose(p=[0.03, 0.02, -0.05], q=euler2quat(0, 0, np.pi / 2))
        self.lid_init_pose = appliance_pose
        appliance_builder.initial_pose = appliance_pose
        self.lid = appliance_builder.build(name=f'{appliance_model_id}')
        self.lid.set_mass(0.5)
        
        appliance_model_name = "106_skillet"  # Example: 106_skillet
        base_skillet_id = 1
        skillet_modelname, skillet_model_id = apply_l2_robotwin_model(
            appliance_model_name,
            model_id=base_skillet_id,
            override_name=appliance_model_name,
            override_id=3,
        )
        skillet_modelname, skillet_model_id = apply_l3_robotwin_model(
            skillet_modelname,
            model_id=skillet_model_id,
            override_name="074_displaystand",
            override_id=4,
        )
        appliance_pose = sapien.Pose(
            p=[0.0, 0.0, 0.0],
            q=euler2quat(np.pi/2, 0, np.pi/2)
        )
        self.skillet_init_pose = appliance_pose
        appliance_actor_obj = create_actor(
            scene=self.scene,
            pose=appliance_pose,
            modelname=skillet_modelname,
            convex=True,
            is_static=True,
            model_id=skillet_model_id,
        )
        self.skillet = appliance_actor_obj.actor
        
        self.objects = [self.lid, self.skillet]



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xyz_rand = torch.rand((b, 2)) * 0.02
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            for object, init_pose in [
                (self.lid, self.lid_init_pose),
                (self.skillet, self.skillet_init_pose),
            ]:
                xyz = torch.tensor(
                    init_pose.p, device=self.device, dtype=torch.float32
                ).repeat(b, 1)
                xyz[:, :2] += xyz_rand
                xyz[:, 2] = self.objects_zs[object][env_idx]
                xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
                original_q = torch.tensor(
                    init_pose.q, device=self.device, dtype=torch.float32
                ).repeat(b, 1)
                object.set_pose(Pose.create_from_pq(p=xyz, q=original_q))
                

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
        lid_pos = self.lid.pose.p
        skillet_pos = self.skillet.pose.p
        place_target = self._get_place_target_pos()
        lid_to_place_pos = lid_pos - place_target
        lid_to_place_dist = torch.linalg.norm(lid_to_place_pos, dim=-1)
        lid_to_skillet_xy = torch.linalg.norm(
            lid_pos[..., :2] - skillet_pos[..., :2], dim=-1
        )

        is_lid_grasped = self.left_agent.is_grasping(self.lid)
        is_robot_static = torch.logical_and(
            self.left_agent.is_static(0.2),
            self.right_agent.is_static(0.2),
        )

        lid_table_z = torch.full(
            (lid_pos.shape[0],),
            float(self.objects_zs[self.lid][0]),
            device=self.device,
        )
        is_lid_on_table = lid_pos[..., 2] <= lid_table_z + 0.03
        is_lid_away_from_skillet = lid_to_skillet_xy >= 0.10
        is_lid_placed = (lid_to_place_dist <= self.goal_thresh) & (~is_lid_grasped)

        success = is_lid_placed

        result = dict(
            is_lid_grasped=is_lid_grasped,
            is_lid_on_table=is_lid_on_table,
            is_lid_away_from_skillet=is_lid_away_from_skillet,
            is_lid_placed=is_lid_placed,
            lid_to_skillet_xy=lid_to_skillet_xy,
            lid_to_place_pos=lid_to_place_pos,
            lid_to_place_dist=lid_to_place_dist,
            is_robot_static=is_robot_static,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_lift_target_pos(self) -> torch.Tensor:
        offset = np.array([0.0, 0.0, 0.10], dtype=np.float32)
        if is_lr_mirror_enabled():
            offset = mirror_xyz(offset)
        return self.skillet.pose.p + torch.tensor(
            offset, device=self.device, dtype=self.skillet.pose.p.dtype
        ).unsqueeze(0).repeat(self.num_envs, 1)

    def _get_transport_target_pos(self) -> torch.Tensor:
        offset = np.array([0.0, -0.25, 0.10], dtype=np.float32)
        if is_lr_mirror_enabled():
            offset = mirror_xyz(offset)
        return self.skillet.pose.p + torch.tensor(
            offset, device=self.device, dtype=self.skillet.pose.p.dtype
        ).unsqueeze(0).repeat(self.num_envs, 1)

    def _get_place_target_pos(self) -> torch.Tensor:
        offset = np.array([0.0, -0.25, 0.01], dtype=np.float32)
        if is_lr_mirror_enabled():
            offset = mirror_xyz(offset)
        return self.skillet.pose.p + torch.tensor(
            offset, device=self.device, dtype=self.skillet.pose.p.dtype
        ).unsqueeze(0).repeat(self.num_envs, 1)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            lid_pos=self.lid.pose.p,
            skillet_pos=self.skillet.pose.p,
            is_lid_grasped=info["is_lid_grasped"],
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
        lid_pos = self.lid.pose.p
        lift_target = self._get_lift_target_pos()
        transport_target = self._get_transport_target_pos()

        is_lid_grasped = info["is_lid_grasped"]
        lid_placed = info["is_lid_placed"]
        lid_to_place_dist = info["lid_to_place_dist"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(left_tcp, lid_pos, scale=5.0)
        r_reach = torch.where(is_lid_grasped | lid_placed | success, ones, r_reach)
        self.reward_tracker.update("reach_lid", r_reach)

        r_grasp = grasp_reward(
            left_tcp, lid_pos, is_lid_grasped, proximity_scale=5.0
        )
        r_grasp = torch.where(lid_placed | success, ones, r_grasp)
        self.reward_tracker.update("grasp_lid", r_grasp)

        r_lift = transport_reward(lid_pos, lift_target, is_lid_grasped, scale=6.0)
        r_lift = torch.where(lid_placed | success, ones, r_lift)
        self.reward_tracker.update("lift_lid", r_lift)

        r_transport = transport_reward(
            lid_pos, transport_target, is_lid_grasped, scale=5.0
        )
        r_transport = torch.where(lid_placed | success, ones, r_transport)
        self.reward_tracker.update("transport_lid", r_transport)

        r_place = tanh_reward(lid_to_place_dist, scale=8.0) * (~is_lid_grasped).float()
        r_place = torch.where(lid_placed | success, ones, r_place)
        self.reward_tracker.update("place_lid", r_place)

        self.reward_tracker.write_to_info(info)
        total_reward = torch.where(success, ones, self.reward_tracker.total())
        self._prioritize_reward_info(info, total_reward)
        return total_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
