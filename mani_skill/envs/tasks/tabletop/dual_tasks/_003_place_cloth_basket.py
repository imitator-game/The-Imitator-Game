import numpy as np
import sapien
import torch
import trimesh

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, List, Tuple
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import articulations
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs import Pose, Articulation, Link
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_partnet_ids,
    apply_l3_robotwin_model,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    reach_reward,
    grasp_reward,
    transport_reward,
    world_aabb_from_local_mesh,
)

PARTNET_COLLISION_BIT = 29
REWARD_PHASES = ["reach", "grasp", "transport", "place"]

@register_env("TwoRobotPlaceClothBasket-v1", max_episode_steps=100)
class TwoRobotPlaceClothBasketEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.08
    articulation_types = ["revolute_unwrapped"]
    container_z_tolerance = 0.1

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
        train_data = load_json(self.JSON)
        self.partnet_model_ids = np.array(list(train_data.keys()))
        self.cloth_model_ids = np.array(['11242'])
        self.basket_modelname, self.basket_model_id = apply_l3_robotwin_model(
            "076_breadbasket",
            model_id=3,
            override_name="062_plasticbox",
        )
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
        cloth_model_ids = self._batched_episode_rng.choice(
            apply_l2_partnet_ids(self.cloth_model_ids, override_ids=["11248"])
        )
        cloth_link_ids = self._batched_episode_rng.randint(0, 2 ** 31)
        for i, model_id in enumerate(cloth_model_ids):
            assert model_id in self.partnet_model_ids, f"model id: {model_id} is not in available ids: {self.partnet_model_ids}."
            self.origin_base = sapien.Pose(
                p=[-0.1, -0.1, -0.12],
                q=euler2quat(np.pi, 0, 0)
            )
            self._load_cloth(
                self.origin_base, self.articulation_types, [model_id], [cloth_link_ids[i]]
            )

        # Load basket
        self.basket_pose = sapien.Pose(
            p=[0.15, 0.0, 0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        basket_actor_obj = create_actor(
            scene=self.scene,
            pose=self.basket_pose,
            modelname=self.basket_modelname,
            convex=True,
            model_id=self.basket_model_id,
            scale=(0.2, 0.2, 0.2),
            replace_scale=True,
            is_static=True
        )
        self.basket = basket_actor_obj.actor

        # Load displaystand
        self.displaystand_pose = sapien.Pose(
            p=[-0.13, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
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
                self.scene, f"partnet-mobility:{model_id}", mode="laptop", scale=0.18
            )
            partnet_builder.set_scene_idxs(scene_idxs=[i])
            partnet_builder.initial_pose = origin_base
            cloth = partnet_builder.build(name=f"{model_id}-{i}", fix_root_link=False)
            cloth.set_mass(0.1)
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
                qpos_min.append(qmin)
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
            cloth_xyz = self.cloth.pose.p[env_idx]
            cloth_xyz[:, 2] = self.cloth_zs[env_idx]
            q = self.cloth.pose.q[env_idx]
            cloth_xyz = apply_l1_offset_xy(cloth_xyz, offset=(-0.10, 0.10))
            self.cloth.set_pose(Pose.create_from_pq(p=cloth_xyz, q=q))

            xyz = torch.tensor(self.basket_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz[:, 2] = self.basket_zs[env_idx] + xyz[:, 2]
            qs = torch.tensor(self.basket_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.basket.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.displaystand_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            xyz[:, 2] = self.displaystand_zs[env_idx] + xyz[:, 2]
            qs = torch.tensor(self.displaystand_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            xyz = apply_l1_offset_xy(xyz, offset=(-0.10, 0.10))
            self.displaystand.set_pose(Pose.create_from_pq(xyz, qs))
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

    # def _after_lr_mirror(self, env_idx: torch.Tensor, options: dict):
    #     # Cloth's original rotation euler(π, 0, 0) is symmetric about XZ plane,
    #     # so mirror_quat has no effect. We need to manually rotate 180° around Z
    #     # to flip the grasp side.
    #     z_180_q = torch.tensor(
    #         euler2quat(0, 0, np.pi), device=self.device, dtype=torch.float32
    #     )
    #     current_q = self.cloth.pose.q
    #     # Apply Z-180 rotation: new_q = z_180_q * current_q
    #     new_q = quaternion_multiply(z_180_q.unsqueeze(0), current_q)
    #     p = self.cloth.pose.p
    #     self.cloth.set_pose(Pose.create_from_pq(p=p, q=new_q))

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

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

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
        centers = []
        for cloth in self._cloths:
            mesh = cloth.get_first_collision_mesh(to_world_frame=True)
            if mesh is None:
                centers.append(cloth.pose.sp.p)
                continue
            try:
                center = np.asarray(mesh.center_mass, dtype=np.float32)
                if not np.all(np.isfinite(center)):
                    raise ValueError("non-finite center_mass")
            except Exception:
                try:
                    center = np.asarray(mesh.centroid, dtype=np.float32)
                    if not np.all(np.isfinite(center)):
                        raise ValueError("non-finite centroid")
                except Exception:
                    bounds = np.asarray(mesh.bounding_box.bounds, dtype=np.float32)
                    center = (bounds[0] + bounds[1]) * 0.5
            centers.append(center)
        return common.to_tensor(np.stack(centers, axis=0), device=self.device)

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

    def evaluate(self):
        obj_to_goal_pos = self.basket.pose.p - self.cloth.pose.p
        basket_bounds = world_aabb_from_local_mesh(self.basket, self.device)
        cloth_center = self._get_cloth_geom_center()
        is_obj_placed = self._is_in_container_bbox(cloth_center, basket_bounds)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        result = dict(
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
            result.update(self.reward_tracker.get_current_dict())
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

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        cloth_center = self._get_cloth_geom_center()
        basket_pos = self._get_actor_geom_center(self.basket)
        is_grasped = self.left_agent.is_grasping(self.partnet_link)
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, cloth_center, scale=5.0)
        r_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_reach)
        r_grasp = grasp_reward(tcp_pos, cloth_center, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_obj_placed | success, ones, r_grasp)
        r_transport = transport_reward(cloth_center, basket_pos, is_grasped, scale=5.0)
        r_transport = torch.where(is_obj_placed | success, ones, r_transport)
        r_place = is_obj_placed.float()
        r_place = torch.where(success, ones, r_place)

        self.reward_tracker.update("reach", r_reach)
        self.reward_tracker.update("grasp", r_grasp)
        self.reward_tracker.update("transport", r_transport)
        self.reward_tracker.update("place", r_place)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
