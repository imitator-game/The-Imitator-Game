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
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs import Pose, Articulation, Link
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.utils.building.actors.robotwin import get_model_id

from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    reach_reward,
    grasp_reward,
    transport_reward,
    wipe_progress_reward,
    above_reward,
    tanh_reward,
    RewardTracker,
)

REWARD_PHASES = ['reach', 'cut']

PARTNET_COLLISION_BIT = 29


@register_env("TwoRobotCutFruitL3-v1", max_episode_steps=100)
class TwoRobotCutFruitEnvL3(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.4
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
                PACKAGE_ASSET_DIR / "partnet_mobility/meta/partnet_info/info_scissors.json"
        )
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.JSON)
        self.partnet_model_ids = np.array(list(train_data.keys()))
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

        # Load carrot
        self.carrot_pose = sapien.Pose(
            p=[0.0, 0.0, 0.03],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        carrot_actor_obj = create_actor(
            scene=self.scene,
            pose=self.carrot_pose,
            modelname="069_vagetable",
            convex=True,
            model_id=6,
            scale=(0.18, 0.18, 0.18),
            replace_scale=True,
            is_static=True
        )
        self.carrot = carrot_actor_obj.actor

        # Load scissors
        self._scissors: List[Actor] = []
        self.partnet_links: List[List[Link]] = []
        self.partnet_links_meshes: List[List[trimesh.Trimesh]] = []
        scissor_model_ids = self._batched_episode_rng.choice(["10495"])
        scissor_link_ids = self._batched_episode_rng.randint(0, 2 ** 31)
        for i, model_id in enumerate(scissor_model_ids):
            assert model_id in self.partnet_model_ids, f"model id: {model_id} is not in available ids: {self.partnet_model_ids}."
            self.origin_base = sapien.Pose(
                p=[-0.0, -0.25, -0.12],
                q=euler2quat(-np.pi / 2, 0, np.pi)
            )
            self._load_scissor(
                self.origin_base, self.articulation_types, [model_id], [scissor_link_ids[i]]
            )

        # Load board
        self.board_pose = sapien.Pose(
            p=[0.0, 0.0, 0.0],
            q=euler2quat(np.pi / 2, 0, 0)
        )
        board_actor_obj = create_actor(
            scene=self.scene,
            pose=self.board_pose,
            modelname="104_board",
            convex=True,
            model_id=2,
            is_static=True,
            replace_scale=True,
            scale=(0.3, 0.3, 0.3),
        )
        self.board = board_actor_obj.actor

        self.left_init_pose = sapien.Pose([0.0256184, -0.315653, 0.185474],
                                          [-0.017689, 0.999802, 0.00428796, 0.00798423])

    def _load_scissor(self, origin_base, joint_types, model_ids, link_ids):
        for i, model_id in enumerate(model_ids):
            # Build articulation
            partnet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode="scissors", scale=0.2
            )
            partnet_builder.set_scene_idxs(scene_idxs=[i])
            partnet_builder.initial_pose = origin_base
            scissor = partnet_builder.build(name=f"{model_id}-{i}", fix_root_link=False)
            scissor.set_mass(0.5)
            self.remove_from_state_dict_registry(scissor)
            for link in scissor.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=PARTNET_COLLISION_BIT, bit=1
                )
            self._scissors.append(scissor)
            self.partnet_links.append([])
            self.partnet_links_meshes.append([])

            # selecting semantic parts of articulations
            for link, joint in zip(scissor.links, scissor.joints):
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
            self.scissor = Articulation.merge(self._scissors, name="scissor")
            self.add_to_state_dict_registry(self.scissor)
            # Set links state
            qpos_min = []
            for i in range(len(self.partnet_links[0])):
                target_qlimits = self.partnet_links[0][i].joint.limits  # [b, 1, 2]
                qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
                qpos_min.append(-0.2)
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

            if not hasattr(self, "scissor_ever_cut_fruit"):
                self.scissor_ever_cut_fruit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.scissor_ever_cut_fruit[env_idx] = False

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(REWARD_PHASES, self.num_envs, self.device)
            self.reward_tracker.reset(env_idx)

            self.table_scene.initialize(env_idx)
            scissor_xyz = self.scissor.pose.p[env_idx]
            scissor_xyz[:, 2] = self.scissor_zs[env_idx]
            q = self.scissor.pose.q[env_idx]
            self.scissor.set_pose(Pose.create_from_pq(p=scissor_xyz, q=q))

            xyz = torch.tensor(self.carrot_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            qs = torch.tensor(self.carrot_pose.q).repeat(b, 1)
            self.carrot.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.board_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.board_zs[env_idx]
            qs = torch.tensor(self.board_pose.q).repeat(b, 1)
            self.board.set_pose(Pose.create_from_pq(xyz, qs))

            left_qpos = [-0.72280455, 0.53875965, 0.65284795, -1.7075238, 1.1945488, 1.4370397, -1.3505738, 0.04, 0.04]
            self.left_agent.robot.set_qpos(left_qpos)

    def _after_reconfigure(self, options: dict):
        self.scissor_zs = []
        for scissor in self._scissors:
            scissor_collision_mesh = scissor.get_first_collision_mesh()
            self.scissor_zs.append(-scissor_collision_mesh.bounding_box.bounds[0, 2])
        self.scissor_zs = common.to_tensor(self.scissor_zs, device=self.device)

        self.carrot_zs = []
        collision_mesh = self.carrot.get_first_collision_mesh()
        self.carrot_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.carrot_zs = common.to_tensor(self.carrot_zs, device=self.device)

        self.board_zs = []
        collision_mesh = self.board.get_first_collision_mesh()
        self.board_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.board_zs = common.to_tensor(self.board_zs, device=self.device)

    @property
    def left_agent(self) -> PandaWristCam:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> PandaWristCam:
        return self.agent.agents[1]

    def evaluate(self):
        # 1. Check if scissor is currently colliding with carrot (to update historical reach flag)
        # Use contact impulses to check for collision
        obj_to_goal_pos = self.scissor.pose.p - self.carrot.pose.p
        is_scissor_at_carrot = torch.linalg.norm(obj_to_goal_pos, axis=1) <= 0.355
        tcp_to_scissor_pos = self.left_agent.tcp.pose.p - self.carrot.pose.p
        is_agent_grasp_scissor = torch.linalg.norm(tcp_to_scissor_pos, axis=1) <= self.goal_thresh
        self.scissor_ever_cut_fruit = self.scissor_ever_cut_fruit | (is_scissor_at_carrot & is_agent_grasp_scissor)

        # 2. Check if robots are static
        is_left_static = self.left_agent.is_static(0.2)
        is_right_static = self.right_agent.is_static(0.2)
        is_robot_static = is_left_static & is_right_static

        # 3. Success Conditions: Ever collided with fruit AND robots are static
        success = self.scissor_ever_cut_fruit

        result = dict(
            is_grasped=is_agent_grasp_scissor,
            scissor_ever_cut_fruit=self.scissor_ever_cut_fruit,
            is_robot_static=is_robot_static,
            success=success,
            # For backward compatibility with reward function
            is_obj_placed=self.scissor_ever_cut_fruit,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.carrot.pose.p,
            is_grasped=info["is_grasped"],
            scissor_ever_cut_fruit=info["scissor_ever_cut_fruit"],
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.scissor.pose.raw_pose,
                left_tcp_to_obj_pos=self.scissor.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.scissor.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.carrot.pose.p - self.scissor.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.left_agent.tcp.pose.p
        scissor_pos = self.scissor.pose.p
        success = info["success"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(tcp_pos, scissor_pos, scale=5.0)
        r_reach = torch.where(success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        r_cut = success.float()
        self.reward_tracker.update("cut", r_cut)

        self.reward_tracker.write_to_info(info)
        return self.reward_tracker.total()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)