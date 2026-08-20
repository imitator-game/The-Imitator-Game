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
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs import Pose, Articulation, Link
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l3_robotwin_model,
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

REWARD_PHASES = ['reach', 'press']

PARTNET_COLLISION_BIT = 29


@register_env("TwoRobotPourKetchupFriesL3-v1", max_episode_steps=100)
class TwoRobotPourKetchupFriesEnvL3(BaseEnv):
    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.15
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
                PACKAGE_ASSET_DIR / "partnet_mobility/meta/partnet_info/info_dispenser.json"
        )
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.JSON)
        self.partnet_model_ids = np.array(list(train_data.keys()))
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

        # Load dispenser
        self._dispensers: List[Actor] = []
        self.partnet_links: List[List[Link]] = []
        self.partnet_links_meshes: List[List[trimesh.Trimesh]] = []
        dispenser_model_ids = self._batched_episode_rng.choice(["101490"])
        dispenser_link_ids = self._batched_episode_rng.randint(0, 2 ** 31)
        for i, model_id in enumerate(dispenser_model_ids):
            assert model_id in self.partnet_model_ids, f"model id: {model_id} is not in available ids: {self.partnet_model_ids}."
            self.origin_base = sapien.Pose(
                p=[0.1, -0.1, -0.0],
                q=euler2quat(0, 0, -np.pi / 2)
            )
            self._load_dispenser(
                self.origin_base, self.articulation_types, [model_id], [dispenser_link_ids[i]]
            )

        # Load fries
        self.fries_pose = sapien.Pose(
            p=[0.05, 0.0, 0.0],
            q=euler2quat(0, 0, -np.pi / 2)
        )
        fries_modelname, fries_model_id = apply_l3_robotwin_model(
            "005_french-fries",
            model_id=0,
            override_name="005_french-fries",
            override_id=2,
        )
        fries_actor_obj = create_actor(
            scene=self.scene,
            pose=self.fries_pose,
            modelname=fries_modelname,
            convex=True,
            model_id=fries_model_id,
            is_static=True,
            replace_scale=True,
            scale=(0.08, 0.08, 0.08),
        )
        self.fries = fries_actor_obj.actor

    def _load_dispenser(self, origin_base, joint_types, model_ids, link_ids):
        for i, model_id in enumerate(model_ids):
            # Build articulation
            partnet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode="dispenser", scale=0.14
            )
            partnet_builder.set_scene_idxs(scene_idxs=[i])
            partnet_builder.initial_pose = origin_base
            dispenser = partnet_builder.build(name=f"{model_id}-{i}", fix_root_link=True)
            dispenser.set_mass(0.5)
            self.remove_from_state_dict_registry(dispenser)
            for link in dispenser.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=PARTNET_COLLISION_BIT, bit=1
                )
            self._dispensers.append(dispenser)
            self.partnet_links.append([])
            self.partnet_links_meshes.append([])

            # selecting semantic parts of articulations
            for link, joint in zip(dispenser.links, dispenser.joints):
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
            self.dispenser = Articulation.merge(self._dispensers, name="dispenser")
            self.add_to_state_dict_registry(self.dispenser)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if not hasattr(self, "dispenser_ever_reached_fries"):
                self.dispenser_ever_reached_fries = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.dispenser_ever_reached_fries[env_idx] = False

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

            self.table_scene.initialize(env_idx)
            dispenser_xyz = self.dispenser.pose.p[env_idx]
            dispenser_xyz[:, 2] = self.dispenser_zs[env_idx]
            q = self.dispenser.pose.q[env_idx]
            self.dispenser.set_pose(Pose.create_from_pq(p=dispenser_xyz, q=q))

            xyz = torch.tensor(self.fries_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.fries_zs[env_idx]
            qs = torch.tensor(self.fries_pose.q).repeat(b, 1)
            self.fries.set_pose(Pose.create_from_pq(xyz, qs))

    def _after_reconfigure(self, options: dict):

        self.dispenser_zs = []
        for dispenser in self._dispensers:
            dispenser_collision_mesh = dispenser.get_first_collision_mesh()
            self.dispenser_zs.append(-dispenser_collision_mesh.bounding_box.bounds[0, 2])
        self.dispenser_zs = common.to_tensor(self.dispenser_zs, device=self.device)

        self.fries_zs = []
        collision_mesh = self.fries.get_first_collision_mesh()
        self.fries_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.fries_zs = common.to_tensor(self.fries_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        tcp_to_obj_dist = torch.linalg.norm(
            self.dispenser.pose.p - self.left_agent.tcp.pose.p, axis=1
        )

        # Condition: distance < threshold AND grasping
        currently_reached = (tcp_to_obj_dist <= self.goal_thresh)
        self.dispenser_ever_reached_fries = self.dispenser_ever_reached_fries | currently_reached

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)
        success = self.dispenser_ever_reached_fries

        result = dict(
            dispenser_to_fries_dist=tcp_to_obj_dist,
            dispenser_ever_reached_fries=self.dispenser_ever_reached_fries,
            is_robot_static=is_robot_static,
            success=success,
            # For backward compatibility with reward function
            is_obj_placed=self.dispenser_ever_reached_fries,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.dispenser.pose.p,
            dispenser_ever_reached_fries=info["dispenser_ever_reached_fries"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.dispenser.pose.raw_pose,
                left_tcp_to_obj_pos=self.dispenser.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.dispenser.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.dispenser.pose.p - self.fries.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        dispenser_pos = self.dispenser.pose.p
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, dispenser_pos, scale=5.0)
        r_reach = torch.where(success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        r_press = success.float()
        self.reward_tracker.update("press", r_press)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)