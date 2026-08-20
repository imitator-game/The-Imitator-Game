import numpy as np
import sapien
import torch
import trimesh

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, List, Tuple, Optional
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import articulations
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs import Pose, Articulation, Link
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_partnet_ids,
    is_lr_mirror_enabled,
    mirror_pose,
    mirror_panda_qpos_y,
)

from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    tanh_reward,
    exp_reward,
)

PARTNET_COLLISION_BIT = 29

# Sub-task phases for this task
REWARD_PHASES = ["open"]

@register_env("TwoRobotOpenBoxL3-v1", max_episode_steps=100)
class TwoRobotOpenBoxEnvL3(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    min_open_frac = 0.45
    articulation_types = ["revolute_unwrapped"]

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
        self.JSON = (
                PACKAGE_ASSET_DIR / "partnet_mobility/meta/partnet_info/info_box.json"
        )
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.JSON)
        self.partnet_model_ids = np.array(list(train_data.keys()))
        self.box_model_ids = apply_l2_partnet_ids(
            ["100189"], override_ids=["100191"]
        )
        self.box_model_ids = ["100141"]
        
        self.l0_init_joint_qpos = -0.3
        self.l3_init_joint_qpos = -1
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

        # Load box
        self._boxs: List[Actor] = []
        self.partnet_links: List[List[Link]] = []
        self.partnet_links_meshes: List[List[trimesh.Trimesh]] = []
        box_model_ids = self._batched_episode_rng.choice(self.box_model_ids)
        box_link_ids = self._batched_episode_rng.randint(0, 2 ** 31)
        for i, model_id in enumerate(box_model_ids):
            assert model_id in self.partnet_model_ids, f"model id: {model_id} is not in available ids: {self.partnet_model_ids}."
            self.origin_base = sapien.Pose(
                p=[0., -0.1, -0.0],
                q=euler2quat(0, 0, np.pi * 3/4)
            )
            self._load_box(
                self.origin_base, self.articulation_types, [model_id], [box_link_ids[i]]
            )

        self.waypoint1 = sapien.Pose([-0.0207413, -0.161137, 0.25], [-0.0705284, 0.919453, 0.386573, 0.013893])
        self.waypoint2 = sapien.Pose([-0.132653, -0.0874181, 0.27], [0.31947, 0.86108, 0.347494, 0.189018])
        self.waypoint3 = sapien.Pose([-0.20096, 0.0649945, 0.37], [0.325356, 0.866503, 0.341442, 0.163507])
        self.left_init_pose = sapien.Pose([0.0256184, -0.315653, 0.185474],
                                           [-0.017689, 0.999802, 0.00428796, 0.00798423])
        if is_lr_mirror_enabled():
            self.waypoint1 = mirror_pose(self.waypoint1, mode="full")
            self.waypoint2 = mirror_pose(self.waypoint2, mode="full")
            self.waypoint3 = mirror_pose(self.waypoint3, mode="full")
            self.left_init_pose = mirror_pose(self.left_init_pose, mode="full")

    def _load_box(self, origin_base, joint_types, model_ids, link_ids):
        for i, model_id in enumerate(model_ids):
            # Build articulation
            partnet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode="box", scale=0.25
            )
            partnet_builder.set_scene_idxs(scene_idxs=[i])
            partnet_builder.initial_pose = origin_base
            box = partnet_builder.build(name=f"{model_id}-{i}", fix_root_link=True)
            self.remove_from_state_dict_registry(box)
            for link in box.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=PARTNET_COLLISION_BIT, bit=1
                )
            self._boxs.append(box)
            self.partnet_links.append([])
            self.partnet_links_meshes.append([])

            # selecting semantic parts of articulations
            for link, joint in zip(box.links, box.joints):
                if joint.type[0] in joint_types:
                    self.partnet_links[-1].append(link)
                    # save the first mesh in the link object that correspond with a lid
                    self.partnet_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: "lid" in render_shape.name,
                            mesh_name="lid",
                        )[0]
                    )

            # Merge to get class Articulation
            self.box = Articulation.merge(self._boxs, name="box")
            self.add_to_state_dict_registry(self.box)
            # Set links state
            qpos = []
            init_joint_qpos = self.l3_init_joint_qpos
            for i in range(len(self.partnet_links[0])):
                target_qlimits = self.partnet_links[0][i].joint.limits  # [b, 1, 2]
                qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
                qpos.append(init_joint_qpos)
            self.partnet_links[0][0].joint.set_qpos(qpos)
            self.partnet_link = Link.merge(
                [links[link_ids[i] % len(links)] for i, links in enumerate(self.partnet_links)],
                name="partnet_link",
            )
            self.partnet_link_pos = common.to_tensor(
                np.array(
                    [
                        meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                        if meshes[link_ids[i] % len(meshes)] is not None else np.array([0., 0., 0.])
                        for i, meshes in enumerate(self.partnet_links_meshes)
                    ]
                ),
                device=self.device,
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

            box_xyz = self.box.pose.p[env_idx]
            box_xyz[:, :2] += torch.rand((b, 2)) * 0.02
            box_xyz = apply_l1_offset_xy(box_xyz, offset=(-0.1, -0.1))
            box_xyz[:, 2] = self.box_zs[env_idx]
            # For euler(0, 0, γ) objects, correct y-mirror is γ → -γ
            box_z_rot = np.pi * 3 / 4
            if is_lr_mirror_enabled():
                box_z_rot = -box_z_rot
            q = torch.tensor(euler2quat(0, 0, box_z_rot), device=self.device).unsqueeze(0).expand(b, -1)
            self.box.set_pose(Pose.create_from_pq(p=box_xyz, q=q))

            init_qpos = [-0.21923073, 1.0975947, 0.40320083, -0.9610004, 0.5436107, 1.2193075, 0.09658707, 0.04, 0.04]
            if is_lr_mirror_enabled():
                init_qpos = mirror_panda_qpos_y(init_qpos)
                init_qpos[6] += np.pi / 2  # Extra π/2 to keep gripper perpendicular to mirrored box
            self.left_agent.robot.set_qpos(init_qpos)

    def _after_reconfigure(self, options: dict):
        self.box_zs = []
        for box in self._boxs:
            box_collision_mesh = box.get_first_collision_mesh()
            self.box_zs.append(-box_collision_mesh.bounding_box.bounds[0, 2])
        self.box_zs = common.to_tensor(self.box_zs, device=self.device)

        target_qlimits = self.partnet_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.min_open_frac
        self.max_qpos = qmax

    def partnet_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.partnet_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.partnet_link_pos, device=self.device),
            )
        return transform_points(
            self.partnet_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.partnet_link_pos[env_idx], device=self.device),
        )

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        open_enough = self.partnet_link.joint.qpos >= self.target_qpos
        partnet_link_pos = self.partnet_link_positions()
        link_is_static = (torch.linalg.norm(self.partnet_link.angular_velocity, axis=1) <= 1
                         ) & (torch.linalg.norm(self.partnet_link.linear_velocity, axis=1) <= 0.1)
        is_robot_static = self.left_agent.is_static(0.2) and self.right_agent.is_static(0.2)

        is_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        result = dict(
            partnet_link_pos=partnet_link_pos,
            open_enough=open_enough,
            is_robot_static=is_robot_static,
            is_grasped=is_grasped,
            success=open_enough,
        )
        # Append per-phase peak sub-rewards
        # if hasattr(self, "reward_tracker"):
        #     # result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            tcp_to_lid_pos=info["partnet_link_pos"] - self.left_agent.tcp.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_lid_pos=info["partnet_link_pos"] - self.left_agent.tcp.pose.p,
                target_link_qpos=self.partnet_link.joint.qpos,
                target_lid_pos=info["partnet_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Phase 1: OPEN - lid opened
        qpos = self.partnet_link.joint.qpos
        r_open = exp_reward(torch.abs(qpos - self.max_qpos), scale=2.0)
        self.reward_tracker.update("open", r_open)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
