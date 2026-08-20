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
from mani_skill.utils.structs import Articulation, Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.building.actors.robotwin import get_model_id
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    geom_center_from_local_mesh,
    reach_reward,
    grasp_reward,
    transport_reward,
    world_aabb_from_local_mesh,
)

PARTNET_COLLISION_BIT = 29
REWARD_PHASES = [
    "clear_reach",
    "clear_grasp",
    "clear_transport",
    "clear_place",
    "cloth_reach",
    "cloth_grasp",
    "cloth_transport",
    "cloth_place",
]


@register_env("TwoRobotPlaceClothBasketL3-v1", max_episode_steps=100)
class TwoRobotPlaceClothBasketEnvL3(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    goal_thresh = 0.08
    articulation_types = ["revolute_unwrapped"]
    container_z_tolerance = 0.1
    video_info_whitelist = {
        "reward",
        "success",
        "is_obj_placed",
        "is_book2_grasped",
        "is_book2_cleared",
        "peak_r_clear_reach",
        "peak_r_clear_grasp",
        "peak_r_clear_transport",
        "peak_r_clear_place",
        "peak_r_cloth_reach",
        "peak_r_cloth_grasp",
        "peak_r_cloth_transport",
        "peak_r_cloth_place",
    }

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
        self.JSON = PACKAGE_ASSET_DIR / "partnet_mobility/meta/partnet_info/info_laptop.json"
        self.robot_init_qpos_noise = robot_init_qpos_noise

        train_data = load_json(self.JSON)
        self.partnet_model_ids = np.array(list(train_data.keys()))
        self.cloth_model_ids = np.array(["11248"])

        # Fixed assets: align with task 002 box setup.
        self.bookcase_modelname = "042_wooden_box"
        self.bookcase_model_id = get_model_id(self.bookcase_modelname, model_id=0)
        self.bookcase_scale = (1.5, 1.5, 1.5)
        self.bookcase_replace_scale = False

        # A horizontal blocker book on top of the box (same pattern as task 002).
        self.book2_modelname = "043_book"
        self.book2_model_id = get_model_id(self.book2_modelname, model_id=1)
        self.book2_scale = (1.2, 0.4, 1.2)
        self.book2_mass = 0.5
        self.book2_replace_scale = False

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

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
        pose = sapien_utils.look_at([0.8, 0.0, 0.75], [0.0, 0.0, 0.25])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )
        self.agent.__init__(self.agent.agents, wrist_sensor=self.wrist_sensor)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        # Load cloth articulation.
        self._cloths: List[Articulation] = []
        self.partnet_links: List[List[Link]] = []
        self.partnet_links_meshes: List[List[trimesh.Trimesh]] = []
        cloth_model_ids = self._batched_episode_rng.choice(self.cloth_model_ids)
        cloth_link_ids = self._batched_episode_rng.randint(0, 2**31)
        for i, model_id in enumerate(cloth_model_ids):
            assert (
                model_id in self.partnet_model_ids
            ), f"model id: {model_id} is not in available ids: {self.partnet_model_ids}."
            cloth_base_pose = sapien.Pose(p=[-0.1, -0.1, -0.12], q=euler2quat(np.pi, 0, 0))
            self._load_cloth(cloth_base_pose, self.articulation_types, [model_id], [cloth_link_ids[i]])

        # Load wooden box (target container), same setup as task 002.
        self.bookcase_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(np.pi / 2, 0, -np.pi / 2))
        bookcase_actor_obj = create_actor(
            scene=self.scene,
            pose=self.bookcase_pose,
            modelname=self.bookcase_modelname,
            convex=True,
            model_id=self.bookcase_model_id,
            scale=self.bookcase_scale,
            replace_scale=self.bookcase_replace_scale,
            is_static=True,
        )
        self.bookcase = bookcase_actor_obj.actor

        self.book2_pose = sapien.Pose(p=[0.0, 0.0, 0.0], q=euler2quat(np.pi / 2, 0, np.pi / 2))
        book2_actor_obj = create_actor(
            scene=self.scene,
            pose=self.book2_pose,
            modelname=self.book2_modelname,
            convex=True,
            model_id=self.book2_model_id,
            scale=self.book2_scale,
            replace_scale=self.book2_replace_scale,
            _idx_if_repeat=2,
            mass=self.book2_mass,
        )
        self.book2 = book2_actor_obj.actor

        # Keep the original displaystand unchanged.
        self.displaystand_pose = sapien.Pose(
            p=[-0.13, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2),
        )
        displaystand_actor_obj = create_actor(
            scene=self.scene,
            pose=self.displaystand_pose,
            modelname="074_displaystand",
            convex=True,
            model_id=2,
            scale=(0.16, 0.16, 0.16),
            replace_scale=True,
            is_static=True,
        )
        self.displaystand = displaystand_actor_obj.actor

        # Compatibility alias for older code paths.
        self.basket = self.bookcase

    def _load_cloth(self, origin_base, joint_types, model_ids, link_ids):
        for i, model_id in enumerate(model_ids):
            partnet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode="laptop", scale=0.15
            )
            partnet_builder.set_scene_idxs(scene_idxs=[i])
            partnet_builder.initial_pose = origin_base
            cloth = partnet_builder.build(name=f"{model_id}-{i}", fix_root_link=False)
            self.remove_from_state_dict_registry(cloth)
            for link in cloth.links:
                link.set_collision_group_bit(group=2, bit_idx=PARTNET_COLLISION_BIT, bit=1)
            self._cloths.append(cloth)
            self.partnet_links.append([])
            self.partnet_links_meshes.append([])

            for link, joint in zip(cloth.links, cloth.joints):
                if joint.type[0] in joint_types:
                    self.partnet_links[-1].append(link)
                    self.partnet_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: "lid" in render_shape.name,
                            mesh_name="lid",
                        )[0]
                    )

            self.cloth = Articulation.merge(self._cloths, name="cloth")
            self.add_to_state_dict_registry(self.cloth)

            qpos_min = []
            for j in range(len(self.partnet_links[0])):
                target_qlimits = self.partnet_links[0][j].joint.limits
                qmin, _ = target_qlimits[..., 0], target_qlimits[..., 1]
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
                        if meshes[link_ids[i] % len(meshes)] is not None
                        else np.array([0.0, 0.0, 0.0])
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
            cloth_xyz[:, 0] -= 0.10
            cloth_xyz[:, 1] += 0.10
            cloth_q = self.cloth.pose.q[env_idx]
            self.cloth.set_pose(Pose.create_from_pq(p=cloth_xyz, q=cloth_q))

            # Same initialization pattern/positioning as task 002 for box.
            box_xyz = torch.tensor(self.bookcase_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            box_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            box_xyz[:, 2] = self.bookcase_zs[env_idx] + box_xyz[:, 2]
            box_q = torch.tensor(self.bookcase_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.bookcase.set_pose(Pose.create_from_pq(box_xyz, box_q))

            box_bounds = world_aabb_from_local_mesh(self.bookcase, self.device)
            box_top_z = box_bounds[:, 1, 2]
            box_center_x = (box_bounds[:, 0, 0] + box_bounds[:, 1, 0]) * 0.5
            box_center_y = (box_bounds[:, 0, 1] + box_bounds[:, 1, 1]) * 0.5

            book2_xyz = torch.zeros((b, 3), device=self.device)
            book2_xyz[:, 0] = box_center_x + 0.02
            book2_xyz[:, 1] = box_center_y
            book2_xyz[:, 2] = box_top_z + self.book2_zs[env_idx] + 0.005
            book2_q = torch.tensor(self.book2_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.book2.set_pose(Pose.create_from_pq(p=book2_xyz, q=book2_q))

            displaystand_xyz = torch.tensor(self.displaystand_pose.p, device=self.device, dtype=torch.float32).repeat(b, 1)
            displaystand_xyz[:, :2] += torch.rand((b, 2), device=self.device) * 0.02
            displaystand_xyz[:, 2] = self.displaystand_zs[env_idx] + displaystand_xyz[:, 2]
            displaystand_xyz[:, 0] -= 0.10
            displaystand_xyz[:, 1] += 0.10
            displaystand_q = torch.tensor(self.displaystand_pose.q, device=self.device, dtype=torch.float32).repeat(b, 1)
            self.displaystand.set_pose(Pose.create_from_pq(displaystand_xyz, displaystand_q))
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

    def _after_reconfigure(self, options: dict):
        self.cloth_zs = []
        for cloth in self._cloths:
            cloth_collision_mesh = cloth.get_first_collision_mesh()
            self.cloth_zs.append(-cloth_collision_mesh.bounding_box.bounds[0, 2])
        self.cloth_zs = common.to_tensor(self.cloth_zs, device=self.device)

        self.bookcase_zs = []
        collision_mesh = self.bookcase.get_first_collision_mesh()
        self.bookcase_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.bookcase_zs = common.to_tensor(self.bookcase_zs, device=self.device)

        self.book2_zs = []
        collision_mesh = self.book2.get_first_collision_mesh()
        self.book2_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.book2_zs = common.to_tensor(self.book2_zs, device=self.device)

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
                print(f"[{self.__class__.__name__}] geom center fallback to pose.p for {obj_name}: {exc}")
                self._geom_center_fallback_logged.add(obj_name)
            return obj.pose.p.clone()

    def evaluate(self):
        box_bounds = world_aabb_from_local_mesh(self.bookcase, self.device)
        cloth_center = self._get_cloth_geom_center()
        is_obj_placed = self._is_in_container_bbox(cloth_center, box_bounds)
        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        book2_center = self._get_actor_geom_center(self.book2)
        clear_goal = self._get_actor_geom_center(self.bookcase) + torch.tensor(
            [-0.1, -0.5, 0.2], device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        book2_cleared = torch.linalg.norm(book2_center - clear_goal, dim=-1) <= 0.2
        result = dict(
            obj_to_goal_pos=self.bookcase.pose.p - self.cloth.pose.p,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_book2_grasped=self.left_agent.is_grasping(self.book2),
            is_book2_cleared=book2_cleared,
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
            goal_pos=self.bookcase.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cloth.pose.raw_pose,
                book2_pose=self.book2.pose.raw_pose,
                left_tcp_to_obj_pos=self.cloth.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.cloth.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.bookcase.pose.p - self.cloth.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        cloth_pos = self._get_cloth_geom_center()
        box_pos = self._get_actor_geom_center(self.bookcase)
        book2_pos = self._get_actor_geom_center(self.book2)
        clear_goal = box_pos + torch.tensor([-0.1, -0.5, 0.2], device=self.device, dtype=torch.float32).unsqueeze(0)
        is_book2_grasped = info["is_book2_grasped"]
        is_book2_cleared = info["is_book2_cleared"]
        is_grasped = self.left_agent.is_grasping(self.partnet_link)
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_clear_reach = reach_reward(tcp_pos, book2_pos, scale=5.0)
        r_clear_reach = torch.where(is_book2_grasped | is_book2_cleared | success, ones, r_clear_reach)
        r_clear_grasp = grasp_reward(tcp_pos, book2_pos, is_book2_grasped, proximity_scale=5.0)
        r_clear_grasp = torch.where(is_book2_cleared | success, ones, r_clear_grasp)
        r_clear_transport = transport_reward(book2_pos, clear_goal, is_book2_grasped, scale=5.0)
        r_clear_transport = torch.where(is_book2_cleared | success, ones, r_clear_transport)
        r_clear_place = is_book2_cleared.float()
        r_clear_place = torch.where(success, ones, r_clear_place)

        r_cloth_reach = reach_reward(tcp_pos, cloth_pos, scale=5.0)
        r_cloth_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_cloth_reach)
        r_cloth_grasp = grasp_reward(tcp_pos, cloth_pos, is_grasped, proximity_scale=5.0)
        r_cloth_grasp = torch.where(is_obj_placed | success, ones, r_cloth_grasp)
        r_cloth_transport = transport_reward(cloth_pos, box_pos, is_grasped, scale=5.0)
        r_cloth_transport = torch.where(is_obj_placed | success, ones, r_cloth_transport)
        r_cloth_place = is_obj_placed.float()
        r_cloth_place = torch.where(success, ones, r_cloth_place)

        self.reward_tracker.update("clear_reach", r_clear_reach)
        self.reward_tracker.update("clear_grasp", r_clear_grasp)
        self.reward_tracker.update("clear_transport", r_clear_transport)
        self.reward_tracker.update("clear_place", r_clear_place)
        self.reward_tracker.update("cloth_reach", r_cloth_reach)
        self.reward_tracker.update("cloth_grasp", r_cloth_grasp)
        self.reward_tracker.update("cloth_transport", r_cloth_transport)
        self.reward_tracker.update("cloth_place", r_cloth_place)
        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
