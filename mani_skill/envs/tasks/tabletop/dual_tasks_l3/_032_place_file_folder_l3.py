import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from typing import Any, Dict, Tuple
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
from mani_skill.utils.structs import Pose, Articulation, Link
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table.utils import create_actor
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    apply_l1_offset_xy,
    apply_l2_robotwin_model,
    is_lr_mirror_enabled,
    mirror_pose,
)

from mani_skill.envs.tasks.tabletop.utils.reward_utils import (
    RewardTracker,
    reach_reward,
    grasp_reward,
    transport_reward,
)

# Sub-task phases for this task
REWARD_PHASES = ["reach", "grasp", "transport", "place"]

@register_env("TwoRobotPlaceFileFolderL3-v1", max_episode_steps=100)
class TwoRobotPlaceFileFolderEnvL3(BaseEnv):

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[PandaWristCam, PandaWristCam]]
    cube_half_size = 0.02
    goal_thresh = 0.25
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
                PACKAGE_ASSET_DIR / "partnet_mobility/meta/partnet_info/info_suitcase.json"
        )
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.notebook_modelname, self.notebook_model_id = apply_l2_robotwin_model(
            "092_notebook",
            model_id=1,
            override_name="092_notebook",
            override_id=2,
        )
        train_data = load_json(self.JSON)
        self.partnet_model_ids = np.array(list(train_data.keys()))
        self.tray_modelname = "008_tray"
        self.tray_model_id = 3

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

        # Load tray
        self.tray_pose = sapien.Pose(
            p=[0.0, 0.15, 0.01],
            q=euler2quat(np.pi / 2, 0, -np.pi / 2)
        )
        tray_actor_obj = create_actor(
            scene=self.scene,
            pose=self.tray_pose,
            modelname=self.tray_modelname,
            convex=True,
            model_id=self.tray_model_id,
            is_static=True,
            replace_scale=True,
            scale=(0.2, 0.2, 0.2),
        )
        self.tray = tray_actor_obj.actor


        # Load file
        self.file_pose = sapien.Pose(
            p=[0., -0.275, 0.12],
            q=euler2quat(-np.pi / 2, 0, np.pi)
        )
        file_actor_obj = create_actor(
            scene=self.scene,
            pose=self.file_pose,
            modelname=self.notebook_modelname,
            convex=True,
            model_id=self.notebook_model_id,
            scale=(0.1, 0.05, 0.13),
            replace_scale=True,
        )
        file_actor_obj.set_mass(0.2)
        self.file = file_actor_obj.actor

        # Load displaystand
        self.displaystand_pose = sapien.Pose(
            p=[0., -0.2, 0.0],
            q=euler2quat(np.pi / 2, 0, 0)
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

        self.waypoint1 = sapien.Pose([0.03, 0.2, 0.22474], [-0.017689, 0.999802, 0.00428796, 0.00798423])
        self.waypoint2 = sapien.Pose([0.03, 0.1, 0.16501], [-0.37829, 0.924493, -0.00569388, 0.0466624])
        self.waypoint3 = sapien.Pose([0.03, 0., 0.253789], [-0.017689, 0.999802, 0.00428796, 0.00798423])
        # self.right_init_pose = sapien.Pose([1.8695e-02, 3.7931e-01, 1.8484e-01],
        #                                   [7.6055e-03, 9.9940e-01, 3.3919e-02, 9.8156e-04])
        if is_lr_mirror_enabled():
            self.waypoint1 = mirror_pose(self.waypoint1, mode="full")
            self.waypoint2 = mirror_pose(self.waypoint2, mode="full")
            self.waypoint3 = mirror_pose(self.waypoint3, mode="full")
            # self.right_init_pose = mirror_pose(self.right_init_pose, mode="full")
        # Flip only the 3rd sxyz parameter for folder when mirroring.
        # self._lr_mirror_euler_z_articulations = (self.folder,)
        # register_lr_mirror_euler_sxyz(self.folder, (0, np.pi / 2, np.pi / 2))

    def _load_folder(self, origin_base, joint_types, model_ids, link_ids):
        for i, model_id in enumerate(model_ids):
            # Build articulation
            partnet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}", mode="suitcase", scale=0.2
            )
            partnet_builder.set_scene_idxs(scene_idxs=[i])
            partnet_builder.initial_pose = origin_base
            folder = partnet_builder.build(name=f"{model_id}-{i}", fix_root_link=True)
            self.remove_from_state_dict_registry(folder)
            for link in folder.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=PARTNET_COLLISION_BIT, bit=1
                )
            self._folders.append(folder)
            self.partnet_links.append([])
            self.partnet_links_meshes.append([])

            # selecting semantic parts of articulations
            for link, joint in zip(folder.links, folder.joints):
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
            self.folder = Articulation.merge(self._folders, name="folder")
            self.add_to_state_dict_registry(self.folder)
            # Set links state
            qpos_min = []
            for i in range(len(self.partnet_links[0])):
                target_qlimits = self.partnet_links[0][i].joint.limits  # [b, 1, 2]
                qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
                qpos_min.append(-0.3)
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

            # Initialize reward tracker
            if not hasattr(self, "reward_tracker"):
                self.reward_tracker = RewardTracker(
                    phase_names=REWARD_PHASES,
                    num_envs=self.num_envs,
                    device=self.device,
                )
            self.reward_tracker.reset(env_idx)

            xyz = torch.tensor(self.tray_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.tray_zs[env_idx]
            qs = torch.tensor(self.tray_pose.q).repeat(b, 1)
            self.tray.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.file_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
            qs = torch.tensor(self.file_pose.q).repeat(b, 1)
            self.file.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.tensor(self.displaystand_pose.p).repeat(b, 1)
            xyz[:, :2] += torch.rand((b, 2)) * 0.02
            xyz[:, 2] = self.displaystand_zs[env_idx] + xyz[:, 2]
            xyz = apply_l1_offset_xy(xyz, offset=(-0.1, -0.1))
            qs = torch.tensor(self.displaystand_pose.q).repeat(b, 1)
            self.displaystand.set_pose(Pose.create_from_pq(xyz, qs))

            # right_qpos = [0.40373468, 0.618651, -0.36374113, -1.8591677, -0.75380105, 2.2495472, 1.4026432, 0.039999984, 0.039999977]
            # if is_lr_mirror_enabled():
            #     right_qpos = mirror_panda_qpos_y(right_qpos)
            #     right_qpos[6] += np.pi / 2  # Extra π/2 to keep gripper perpendicular to mirrored objects
            # self.right_agent.robot.set_qpos(right_qpos)

    def _after_reconfigure(self, options: dict):
        self.tray_zs = []
        collision_mesh = self.tray.get_first_collision_mesh()
        self.tray_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.tray_zs = common.to_tensor(self.tray_zs, device=self.device)

        self.file_zs = []
        collision_mesh = self.file.get_first_collision_mesh()
        self.file_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.file_zs = common.to_tensor(self.file_zs, device=self.device)

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
        obj_to_goal_pos = self.file.pose.p - self.tray.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_robot_static = self.left_agent.is_static(0.2) and self.right_agent.is_static(0.2)
        is_grasped = self.left_agent.is_grasping(self.file)

        result = dict(
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasped=is_grasped,
            success=is_obj_placed,
        )
        # Append per-phase peak sub-rewards
        # if hasattr(self, "reward_tracker"):
        #     # result.update(self.reward_tracker.get_peak_dict())
        return result

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            left_tcp_pose=self.left_agent.tcp.pose.raw_pose,
            right_tcp_pose=self.right_agent.tcp.pose.raw_pose,
            goal_pos=self.tray.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.file.pose.raw_pose,
                left_tcp_to_obj_pos=self.file.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_obj_pos=self.file.pose.p - self.right_agent.tcp.pose.p,
                obj_to_goal_pos=self.file.pose.p - self.tray.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        left_tcp = self.left_agent.tcp.pose.p
        file_pos = self.file.pose.p
        tray_pos = self.tray.pose.p

        is_grasped = info["is_grasped"]
        is_obj_placed = info["is_obj_placed"]
        success = info["success"]

        ones = torch.ones(self.num_envs, device=self.device)

        # Phase 1: REACH - left TCP approaches file
        r_reach = reach_reward(left_tcp, file_pos, scale=5.0)
        r_reach = torch.where(is_grasped | is_obj_placed | success, ones, r_reach)
        self.reward_tracker.update("reach", r_reach)

        # Phase 2: GRASP - proximity / confirmed grasp
        r_grasp = grasp_reward(left_tcp, file_pos, is_grasped, proximity_scale=5.0)
        r_grasp = torch.where(is_obj_placed | success, ones, r_grasp)
        self.reward_tracker.update("grasp", r_grasp)

        # Phase 3: TRANSPORT - file to tray
        r_transport = transport_reward(file_pos, tray_pos, is_grasped, scale=5.0)
        r_transport = torch.where(is_obj_placed | success, ones, r_transport)
        self.reward_tracker.update("transport", r_transport)

        # Phase 4: PLACE - file placed on tray (only when transport started and released)
        transport_peak = self.reward_tracker._peaks["transport"]
        r_place = is_obj_placed.float() * (transport_peak > 0).float() * (~is_grasped).float()
        r_place = torch.where(success, ones, r_place)
        self.reward_tracker.update("place", r_place)

        # Diagnostics
        self.reward_tracker.write_to_info(info)

        # Total = arithmetic mean of peaks -> [0, 1]
        return self.reward_tracker.total()
    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)
