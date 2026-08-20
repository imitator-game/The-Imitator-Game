import os
import os.path as osp
from pathlib import Path
from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


# TODO (stao): make the build and initialize api consistent with other scenes
class TableSceneBuilder(SceneBuilder):
    def build(
            self,
            random_background=False,
            eval_mode=False,
    ):
        self.height = 0.75
        # RoboTwin Settings
        table_height_bias = 0
        random_table_height = 0
        self.table_z_bias = (np.random.uniform(low=-random_table_height, high=0) + table_height_bias)
        self.clean_background_rate = 0
        self.create_table_and_wall(
            table_height=self.height - 0.05 / 2, # minus table thickness
            random_background=random_background,
            eval_mode=eval_mode,
        )

        self.scene_objects: List[sapien.Entity] = [self.table, self.ground, self.wall]

    def create_table_and_wall(
            self,
            table_xy_bias=[0, 0],
            table_height=0.9,
            random_background=False,
            eval_mode=False
    ):
        self.table_xy_bias = table_xy_bias
        table_height += self.table_z_bias

        if random_background:
            texture_type = "seen" if not eval_mode else "unseen"
            directory_path = str(Path.home() / ".maniskill" / "data" / "robotwin" / f"background_texture/{texture_type}")
            file_count = len(
                [name for name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, name))])

            # wall_texture, table_texture = random.randint(0, file_count - 1), random.randint(0, file_count - 1)
            wall_texture = np.random.randint(0, file_count)
            table_texture = np.random.randint(0, file_count)
            ground_texture = np.random.randint(0, file_count)

            wall_texture, table_texture, ground_texture = (
                f"{texture_type}/{wall_texture}",
                f"{texture_type}/{table_texture}",
                f"{texture_type}/{ground_texture}",
            )
            if np.random.rand() <= self.clean_background_rate:
                wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                ground_texture = None
        else:
            wall_texture, table_texture, ground_texture = None, None, None

        from mani_skill.utils.scene_builder.table.utils import create_table, create_box, create_ground
        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0.8, 0, -0.3], q=euler2quat(0.0, 0.0, np.pi / 2)),
            half_size=[3, 0.05, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=wall_texture,
            is_static=True,
        )

        self.table = create_table(
            self.scene,
            sapien.Pose(
                p=[table_xy_bias[0], table_xy_bias[1], table_height],
                q=euler2quat(0.0, 0, 0.0)
            ),
            length=2.0,
            width=1.2,
            height=table_height,
            thickness=0.05,
            is_static=True,
            texture_id=table_texture,
        )

        self.ground = create_ground(
            self.scene,
            floor_width=100,
            altitude=-self.height,
            texture_id=ground_texture,
        )

    def initialize(self, env_idx: torch.Tensor):
        b = len(env_idx)
        self.table.set_pose(
            sapien.Pose(p=[-0.12, 0, -self.height], q=euler2quat(0, 0, np.pi / 2))
        )
        if self.env.robot_uids == "panda":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_wristcam":
            from mani_skill.envs.tasks import FoldSuitcaseEnv
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
            )
            if isinstance(self.env, FoldSuitcaseEnv):
                qpos = np.array(
                    [
                        0.0 + np.random.uniform(-1, 1) * 0.3,
                        -np.pi * 1 / 8 + np.random.uniform(-1, 1) * 0.1,
                        0 + np.random.uniform(-1, 1) * 0.3,
                        -np.pi * 6 / 8 + np.random.uniform(-1, 1) * 0.1,
                        0 + np.random.uniform(-1, 1) * 0.2,
                        np.pi * 3 / 4 + np.random.uniform(-1, 1) * 0.3,
                        np.pi / 4,
                        0.04,
                        0.04
                    ]
                )
            # fmt: on
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in [
            "xarm6_allegro_left",
            "xarm6_allegro_right",
            "xarm6_robotiq",
            "xarm6_nogripper",
        ]:
            qpos = self.env.agent.keyframes["rest"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.522, 0, 0]))
        elif self.env.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == ("panda", "panda"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif self.env.robot_uids == ("panda_wristcam", "panda_wristcam"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([-0.60, 0.35, 0], q=euler2quat(0, 0, 0))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([-0.60, -0.35, 0], q=euler2quat(0, 0, 0))
            )
        elif (
            "dclaw" in self.env.robot_uids
            or "allegro" in self.env.robot_uids
            or "trifinger" in self.env.robot_uids
        ):
            # Need to specify the robot qpos for each sub-scenes using tensor api
            pass
        elif self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in ["widowxai", "widowxai_wristcam"]:
            qpos = self.env.agent.keyframes["ready_to_grasp"].qpos
            self.env.agent.reset(qpos)
        elif self.env.robot_uids == "so100":
            qpos = np.array([0, np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, 1.0])
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose([-0.725, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )
