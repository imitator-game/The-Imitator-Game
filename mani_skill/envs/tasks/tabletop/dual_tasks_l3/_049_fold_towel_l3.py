import numpy as np
import sapien
import torch
import trimesh

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, List, Union, Tuple
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs import Pose, Articulation, Link
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import mirror_xyz
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
    tanh_reward,
    normalized_progress,
    release_and_static_reward,
    geom_center_from_local_mesh,
    world_aabb_from_local_mesh,
)

PARTNET_COLLISION_BIT = 29
REWARD_PHASES = [
    "reach_first_fold",
    "grasp_first_fold",
    "fold_towel",
    "reach_second_fold",
    "transport_towel",
    "place_towel",
]

@register_env("TwoRobotFoldTowelL3-v1", max_episode_steps=100)
class TwoRobotFoldTowelEnvL3(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.15
    container_z_tolerance = 0.08
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
                PACKAGE_ASSET_DIR / "partnet_mobility/meta/partnet_info/info_laptop.json"
        )
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.basket_modelname = "062_plasticbox"
        self.basket_model_id = 9
        train_data = load_json(self.JSON)
        self.partnet_model_ids = np.array(list(train_data.keys()))
        self.cloth_model_ids = np.array(["10040"])
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

        # Load cloth
        self._cloths: List[Actor] = []
        self.partnet_links: List[List[Link]] = []
        self.partnet_links_meshes: List[List[trimesh.Trimesh]] = []
        cloth_model_ids = self._batched_episode_rng.choice(self.cloth_model_ids)
        cloth_link_ids = self._batched_episode_rng.randint(0, 2 ** 31)
        for i, model_id in enumerate(cloth_model_ids):
            assert model_id in self.partnet_model_ids, f"model id: {model_id} is not in available ids: {self.partnet_model_ids}."
            self.origin_base = sapien.Pose(
                p=[-0.05, -0.12, -0.12],
                q=euler2quat(0, 0, 0)
            )
            self._load_cloth(
                self.origin_base, self.articulation_types, [model_id], [cloth_link_ids[i]]
            )

        # Load basket
        self.basket_pose = sapien.Pose(
            p=[0.1, 0.0, 0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        basket_actor_obj = create_actor(
            scene=self.scene,
            pose=self.basket_pose,
            modelname=self.basket_modelname,  # 062_plasticbox
            convex=True,
            model_id=self.basket_model_id,
            scale=(0.15, 0.15, 0.15),
            replace_scale=True,
            is_static=True
        )
        self.basket = basket_actor_obj.actor

        # Load displaystand
        self.displaystand_pose = sapien.Pose(
            p=[-0.13, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, np.pi / 2)
        )
        displaystand_actor_obj = create_actor(
            scene=self.scene,
            pose=self.displaystand_pose,
            modelname="074_displaystand",
            convex=True,
            model_id=2,
            scale=(0.16, 0.16, 0.16),
            replace_scale=True,
            is_static=True
        )
        self.displaystand = displaystand_actor_obj.actor

    def _load_cloth(self, origin_base, joint_types, model_ids, link_ids):
        for i, model_id in enumerate(model_ids):
            # Build articulation
            partnet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode="laptop", scale=0.15
            )
            partnet_builder.set_scene_idxs(scene_idxs=[i])
            partnet_builder.initial_pose = origin_base
            cloth = partnet_builder.build(name=f"{model_id}-{i}", fix_root_link=False)
            cloth.set_mass(0.3)
            self.remove_from_state_dict_registry(cloth)
            for link in cloth.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=PARTNET_COLLISION_BIT, bit=1
                )
            self._cloths.append(cloth)
            self.partnet_links.append([])
            self.partnet_links_meshes.append([])

            # selecting semantic parts of articulations
            for link, joint in zip(cloth.links, cloth.joints):
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
            self.cloth = Articulation.merge(self._cloths, name="cloth")
            self.add_to_state_dict_registry(self.cloth)
            # Set links state
            qpos_min = []
            for i in range(len(self.partnet_links[0])):
                target_qlimits = self.partnet_links[0][i].joint.limits  # [b, 1, 2]
                qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
                qpos_min.append(qmax)
            self.partnet_links[0][0].joint.set_qpos(qpos_min)
            self.partnet_link = Link.merge(
                [links[link_ids[i] % len(links)] for i, links in enumerate(self.partnet_links)],
                name="lid_link",
            )
            self.lid_link_pos = common.to_tensor(
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
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)
            cloth_xyz = self.cloth.pose.p[env_idx]
            cloth_xyz[:, 2] = self.cloth_zs[env_idx]
            q = self.cloth.pose.q[env_idx]
            self.cloth.set_pose(Pose.create_from_pq(p=cloth_xyz, q=q))

            xyz = torch.tensor(self.basket_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.basket_zs[env_idx] + xyz[:, 2]
            qs = torch.tensor(self.basket_pose.q).repeat(b, 1)
            self.basket.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.displaystand_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.displaystand_zs[env_idx] + xyz[:, 2]
            qs = torch.tensor(self.displaystand_pose.q).repeat(b, 1)
            self.displaystand.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):
        self.cloth_zs = []
        for cloth in self._cloths:
            cloth_collision_mesh = cloth.get_first_collision_mesh()
            self.cloth_zs.append(-cloth_collision_mesh.bounding_box.bounds[0, 2])
        self.cloth_zs = common.to_tensor(self.cloth_zs, device=self.device)

        self.basket_zs = []
        collision_mesh = self.basket.get_first_collision_mesh()
        self.basket_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.basket_zs = common.to_tensor(self.basket_zs, device=self.device)

        self.displaystand_zs = []
        collision_mesh = self.displaystand.get_first_collision_mesh()
        self.displaystand_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.displaystand_zs = common.to_tensor(self.displaystand_zs, device=self.device)

        # Fold target: use partnet joint limits similar to open_box
        target_qlimits = self.partnet_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * 0.5
        self.max_qpos = qmax

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        cloth_center = self._get_cloth_geom_center()
        basket_center = self._get_actor_geom_center(self.basket)
        basket_bounds = world_aabb_from_local_mesh(self.basket, self.device)
        obj_to_goal_pos = basket_center - cloth_center
        cloth_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, dim=-1)
        is_obj_placed = self._is_in_container_bbox(cloth_center, basket_bounds)
        qpos = self.partnet_link.joint.qpos
        if qpos.ndim > 1:
            qpos = qpos[..., 0]
        target_qpos = self.target_qpos
        if target_qpos.ndim > 1:
            target_qpos = target_qpos[..., 0]
        max_qpos = self.max_qpos
        if max_qpos.ndim > 1:
            max_qpos = max_qpos[..., 0]
        fold_progress = normalized_progress(qpos, start=max_qpos, target=target_qpos)
        fold_enough = qpos <= target_qpos
        is_grasped = self.left_agent.is_grasping(self.partnet_link)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        result = dict(
            cloth_center=cloth_center,
            basket_center=basket_center,
            obj_to_goal_pos=obj_to_goal_pos,
            cloth_to_goal_dist=cloth_to_goal_dist,
            is_grasped=is_grasped,
            is_obj_placed=is_obj_placed,
            fold_progress=fold_progress,
            fold_enough=fold_enough,
            is_robot_static=is_robot_static,
            success=is_obj_placed & fold_enough & (~is_grasped),
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.basket.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cloth.pose.raw_pose,
                left_tcp_to_obj_pos=self.cloth.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.cloth.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.basket.pose.p - self.cloth.pose.p,
            )
        return obs

    def _is_in_container_bbox(self, obj_pos: torch.Tensor, container_bounds: torch.Tensor) -> torch.Tensor:
        if container_bounds.ndim == 3:
            lower = container_bounds[:, 0]
            upper = container_bounds[:, 1]
        else:
            lower = container_bounds[0]
            upper = container_bounds[1]
        upper = upper.clone()
        upper[..., 2] += self.container_z_tolerance
        return torch.all(torch.logical_and(obj_pos >= lower, obj_pos <= upper), dim=-1)

    def _get_cloth_geom_center(self) -> torch.Tensor:
        return transform_points(
            self.partnet_link.pose.to_transformation_matrix().clone(),
            common.to_tensor(self.lid_link_pos, device=self.device),
        )

    def _get_actor_geom_center(self, obj) -> torch.Tensor:
        try:
            return geom_center_from_local_mesh(obj, self.device)
        except Exception:
            return obj.pose.p.clone()

    def _get_first_grasp_target_pos(self) -> torch.Tensor:
        offset = [0.10, 0.05, 0.02]
        if bool(getattr(self, "_lr_mirror_applied_this_reset", False)):
            offset = mirror_xyz(offset)
        offset = torch.tensor(offset, device=self.device, dtype=self.cloth.pose.p.dtype)
        return self.cloth.pose.p + offset.unsqueeze(0)

    def _get_second_grasp_target_pos(self) -> torch.Tensor:
        offset = [-0.05, 0.05, 0.02]
        if bool(getattr(self, "_lr_mirror_applied_this_reset", False)):
            offset = mirror_xyz(offset)
        offset = torch.tensor(offset, device=self.device, dtype=self.cloth.pose.p.dtype)
        return self.cloth.pose.p + offset.unsqueeze(0)

    def _get_transport_target_pos(self) -> torch.Tensor:
        offset = [0.01, 0.02, 0.06]
        if bool(getattr(self, "_lr_mirror_applied_this_reset", False)):
            offset = mirror_xyz(offset)
        offset = torch.tensor(offset, device=self.device, dtype=self.basket.pose.p.dtype)
        return self.basket.pose.p + offset.unsqueeze(0)

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
        cloth_center = info["cloth_center"]
        first_target = self._get_first_grasp_target_pos()
        second_target = self._get_second_grasp_target_pos()
        transport_target = self._get_transport_target_pos()
        is_grasped = info["is_grasped"]
        fold_progress = info["fold_progress"]
        fold_enough = info["fold_enough"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach_first = reach_reward(left_tcp, first_target, scale=5.0)
        r_reach_first = torch.where(is_grasped | (fold_progress > 0.05) | success, ones, r_reach_first)
        self.reward_tracker.update("reach_first_fold", r_reach_first)

        r_grasp_first = grasp_reward(
            left_tcp, first_target, is_grasped, proximity_scale=5.0
        )
        r_grasp_first = torch.where((fold_progress > 0.05) | success, ones, r_grasp_first)
        self.reward_tracker.update("grasp_first_fold", r_grasp_first)

        r_fold = fold_progress * is_grasped.float()
        r_fold = torch.where(fold_enough | is_obj_placed | success, ones, r_fold)
        self.reward_tracker.update("fold_towel", r_fold)

        r_reach_second = reach_reward(left_tcp, second_target, scale=5.0) * fold_progress
        r_reach_second = torch.where((fold_enough & is_grasped) | is_obj_placed | success, ones, r_reach_second)
        self.reward_tracker.update("reach_second_fold", r_reach_second)

        r_transport = transport_reward(
            cloth_center, transport_target, is_grasped & fold_enough, scale=5.0
        )
        r_transport = torch.where(is_obj_placed | success, ones, r_transport)
        self.reward_tracker.update("transport_towel", r_transport)

        release_static = release_and_static_reward(
            is_grasped, info["is_robot_static"], is_obj_placed
        )
        r_place = 0.5 * tanh_reward(info["cloth_to_goal_dist"], scale=5.0) + 0.5 * release_static
        r_place = r_place * fold_enough.float()
        r_place = torch.where(success, ones, r_place)
        self.reward_tracker.update("place_towel", r_place)

        self.reward_tracker.write_to_info(info)
        total_reward = torch.where(success, ones, self.reward_tracker.total())
        self._prioritize_reward_info(info, total_reward)
        return total_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
