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
    apply_l3_partnet_ids,
    is_l2_enabled,
)
from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    tanh_reward,
    normalized_progress,
)

PARTNET_COLLISION_BIT = 29
REWARD_PHASES = ["reach_juicer", "press_juicer"]

@register_env("TwoRobotPressJuicer-v1", max_episode_steps=100)
class TwoRobotPressJuicerEnv(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    max_close_frac = 0.25
    articulation_types = ["revolute_unwrapped"]
    goal_thresh = 0.15
    press_contact_thresh = 0.14

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
                PACKAGE_ASSET_DIR / "partnet_mobility/meta/partnet_info/info_coffeemachine.json"
        )
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.JSON)
        self.partnet_model_ids = np.array(list(train_data.keys()))
        self.juicer_model_ids = np.array(["103105"])
        self.juicer_model_ids = apply_l2_partnet_ids(
            self.juicer_model_ids,
            override_ids=["103134"],
        )
        self.juicer_model_ids = apply_l3_partnet_ids(
            self.juicer_model_ids,
            override_ids=["103064"],
        )
        # Tunable articulation scale for L2 replacement juicer.
        self.juicer_scale_default = 0.25
        self.juicer_scale_l2 = 0.16
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
        pose = sapien_utils.look_at([0.8, 0, 0.75], [0.0, 0.0, 0.25])
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

        # Load juicer
        self._juicers: List[Actor] = []
        self.partnet_links: List[List[Link]] = []
        self.partnet_links_meshes: List[List[trimesh.Trimesh]] = []
        juicer_model_ids = self._batched_episode_rng.choice(self.juicer_model_ids)
        juicer_link_ids = self._batched_episode_rng.randint(0, 2 ** 31)
        for i, model_id in enumerate(juicer_model_ids):
            assert model_id in self.partnet_model_ids, f"model id: {model_id} is not in available ids: {self.partnet_model_ids}."
            self.origin_base = sapien.Pose(
                p=[0.2, 0.0, -0.0],
                q=euler2quat(0, 0, 0)
            )
            self._load_juicer(
                self.origin_base, self.articulation_types, [model_id], [juicer_link_ids[i]]
            )

        self.waypoint1 = sapien.Pose([0.0846275, 0.00043541, 0.228448], [-0.0185337, 0.999788, 0.00386176, 0.00809702])
        self.waypoint2 = sapien.Pose([0.0844223, 0.00291058, 0.120203], [-0.0235407, 0.937102, 0.0253386, 0.347338])

    def _load_juicer(self, origin_base, joint_types, model_ids, link_ids):
        for i, model_id in enumerate(model_ids):
            # Build articulation
            juicer_scale = self.juicer_scale_l2 if is_l2_enabled() else self.juicer_scale_default
            partnet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode="coffeemachine", scale=juicer_scale
            )
            partnet_builder.set_scene_idxs(scene_idxs=[i])
            partnet_builder.initial_pose = origin_base
            juicer = partnet_builder.build(name=f"{model_id}-{i}", fix_root_link=True)
            self.remove_from_state_dict_registry(juicer)
            for link in juicer.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=PARTNET_COLLISION_BIT, bit=1
                )
            self._juicers.append(juicer)
            self.partnet_links.append([])
            self.partnet_links_meshes.append([])

            # selecting semantic parts of articulations
            def _select_mesh(link):
                mesh = link.generate_mesh(
                    filter=lambda _, render_shape: "lid" in render_shape.name,
                    mesh_name="lid",
                )[0]
                if mesh is None:
                    mesh = link.generate_mesh(
                        filter=lambda *_: True,
                        mesh_name="all",
                    )[0]
                return mesh

            for link, joint in zip(juicer.links, juicer.joints):
                if joint.type[0] in joint_types:
                    self.partnet_links[-1].append(link)
                    # save the first mesh in the link object that correspond with a lid
                    self.partnet_links_meshes[-1].append(_select_mesh(link))

            if not self.partnet_links[-1]:
                for link, joint in zip(juicer.links, juicer.joints):
                    if joint.type[0] != "fixed":
                        self.partnet_links[-1].append(link)
                        self.partnet_links_meshes[-1].append(_select_mesh(link))

            if not self.partnet_links[-1]:
                raise ValueError(f"No valid articulation links found for model_id={model_id}")

            # Merge to get class Articulation
            self.juicer = Articulation.merge(self._juicers, name="juicer")
            self.add_to_state_dict_registry(self.juicer)
            self.juicer.set_mass(0.5)
            # Set links state
            qpos = []
            for i in range(len(self.partnet_links[0])):
                target_qlimits = self.partnet_links[0][i].joint.limits  # [b, 1, 2]
                qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
                qpos.append(qmin)
            # self.partnet_links[0][0].joint.set_qpos(qpos)
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

            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)
            if not hasattr(self, "juicer_ever_pressed"):
                self.juicer_ever_pressed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.juicer_ever_pressed[env_idx] = False
            if not hasattr(self, "init_joint_qpos"):
                self.init_joint_qpos = torch.zeros(self.num_envs, device=self.device)

            juicer_xyz = self.juicer.pose.p[env_idx]
            shared_xy_offset = torch.rand((b, 2)) * 0.02
            juicer_xyz[:, :2] += shared_xy_offset
            juicer_xyz = apply_l1_offset_xy(juicer_xyz, offset=(0.0, -0.2))
            shared_xy_offset = apply_l1_offset_xy(shared_xy_offset, offset=(0.0, -0.2))
            juicer_xyz[:, 2] = self.juicer_zs[env_idx]
            q = self.juicer.pose.q[env_idx]
            self.juicer.set_pose(Pose.create_from_pq(p=juicer_xyz, q=q))
            waypoint_offset = torch.tensor(
                [shared_xy_offset[0, 0], shared_xy_offset[0, 1], 0.0],
                device=self.device,
            )
            self.waypoint1 = sapien.Pose(self.waypoint1.p + waypoint_offset.cpu().numpy(), self.waypoint1.q)
            self.waypoint2 = sapien.Pose(self.waypoint2.p + waypoint_offset.cpu().numpy(), self.waypoint2.q)
            qpos = self.partnet_link.joint.qpos
            if qpos.ndim > 1:
                qpos = qpos[..., 0]
            self.init_joint_qpos[env_idx] = qpos[env_idx]

    def _after_reconfigure(self, options: dict):
        self.juicer_zs = []
        for juicer in self._juicers:
            juicer_collision_mesh = juicer.get_first_collision_mesh()
            self.juicer_zs.append(-juicer_collision_mesh.bounding_box.bounds[0, 2])
        self.juicer_zs = common.to_tensor(self.juicer_zs, device=self.device)

        target_qlimits = self.partnet_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.max_close_frac

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
        partnet_link_pos = self.partnet_link_positions()
        tcp_to_lid_dist = torch.linalg.norm(self.left_agent.tcp.pose.p - partnet_link_pos, axis=1)
        qpos = self.partnet_link.joint.qpos
        if qpos.ndim > 1:
            qpos = qpos[..., 0]
        target_qpos = self.target_qpos
        if target_qpos.ndim > 1:
            target_qpos = target_qpos[..., 0]
        press_progress = normalized_progress(
            qpos, start=self.init_joint_qpos, target=target_qpos
        )
        close_enough = tcp_to_lid_dist <= self.press_contact_thresh
        pressed_enough = qpos <= target_qpos
        currently_pressing = close_enough
        self.juicer_ever_pressed = self.juicer_ever_pressed | currently_pressing

        is_robot_static = self.left_agent.is_static(0.2) & self.right_agent.is_static(0.2)

        success = self.juicer_ever_pressed

        result = dict(
            partnet_link_pos=partnet_link_pos,
            tcp_to_lid_dist=tcp_to_lid_dist,
            target_link_qpos=qpos,
            target_press_qpos=target_qpos,
            press_progress=press_progress,
            close_enough=close_enough,
            pressed_enough=pressed_enough,
            currently_pressing=currently_pressing,
            is_robot_static=is_robot_static,
            juicer_ever_pressed=self.juicer_ever_pressed,
            is_obj_placed=self.juicer_ever_pressed,
            success=success,
        )
        if hasattr(self, "reward_tracker"):
            result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            tcp_to_lid_pos=info["partnet_link_pos"] - self.left_agent.tcp.pose.p,
            juicer_ever_pressed=info["juicer_ever_pressed"],
        )
        if "state" in self.obs_mode:
            obs.update(
                target_link_qpos=info["target_link_qpos"],
                target_lid_pos=info["partnet_link_pos"],
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
        success = info["success"]
        ever_pressed = info["juicer_ever_pressed"]
        close_enough = info["close_enough"]
        press_progress = info["press_progress"]
        ones = torch.ones(self.num_envs, device=self.device)

        r_reach = reach_reward(left_tcp, info["partnet_link_pos"], scale=5.0)
        r_reach = torch.where(ever_pressed | success, ones, r_reach)
        self.reward_tracker.update("reach_juicer", r_reach)

        press_contact_reward = tanh_reward(info["tcp_to_lid_dist"], scale=8.0)
        r_press = torch.maximum(press_contact_reward, press_progress)
        r_press = torch.where(close_enough, ones, r_press)
        r_press = torch.where(success, ones, r_press)
        self.reward_tracker.update("press_juicer", r_press)

        self.reward_tracker.write_to_info(info)
        total_reward = torch.where(success, ones, self.reward_tracker.total())
        self._prioritize_reward_info(info, total_reward)
        return total_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
