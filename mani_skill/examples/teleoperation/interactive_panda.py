"""
Interactive teleoperation template (panda-type arms).

Reference implementation for collecting demonstrations by hand with a panda
(or panda_wristcam) robot. Reuses the click-and-drag TransformWindow of the
SAPIEN viewer plus keyboard shortcuts for micro-movements and the gripper.

How it works:
    - A "ghost" panda arm is dragged in the viewer with the mouse gizmo.
    - Pressing [n] runs motion planning to move the real robot to that pose.
    - [g] toggles the gripper; arrow keys + [u]/[j] nudge the ghost arm.
    - [c] finishes the current episode (saves the trajectory), [q] quits.

Run:
    python -m mani_skill.examples.teleoperation.interactive_panda \
        -e TwoRobotPickCubeYCB-v1 -r panda --save-video

Recorded demos: demos/{env_id}/teleop/trajectory.h5 (+ .json)
"""

from dataclasses import dataclass
from typing import Annotated

import gymnasium as gym
import h5py
import json
import sapien.core as sapien
import sapien.utils.viewer
import tyro
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
    PandaStickMotionPlanningSolver,
)
import mani_skill.trajectory.utils as trajectory_utils
import numpy as np


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda"
    """Supported: panda, panda_wristcam, panda_stick"""
    record_dir: str = "demos"
    """directory to save the recorded demonstrations and optional videos"""
    save_video: bool = False
    viewer_shader: str = "rt-fast"
    video_saving_shader: str = "rt-fast"


def parse_args() -> Args:
    return tyro.cli(Args)


def _make_planner(env, debug=False, vis=False):
    """Create the motion planner for the requested robot uid."""
    robot_uid = env.unwrapped.robot_uids
    if robot_uid == "panda_stick":
        return PandaStickMotionPlanningSolver(
            env, debug=debug, vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False, print_env_info=False,
            joint_acc_limits=0.5, joint_vel_limits=0.5)
    assert robot_uid in ("panda", "panda_wristcam"), robot_uid
    return PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False,
        joint_acc_limits=0.5, joint_vel_limits=0.5)


def solve(env: BaseEnv, debug=False, vis=False):
    """Interactive loop: drag the ghost arm, then press [n] to execute."""
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"]
    planner = _make_planner(env, debug, vis)
    robot_has_gripper = env.unwrapped.robot_uids != "panda_stick"

    viewer = env.render_human()
    gripper_open = True

    def select_panda_hand():
        viewer.select_entity(
            sapien_utils.get_obj_by_name(env.agent.robot.links, "panda_hand")._objs[0].entity)

    select_panda_hand()
    transform_window = next(
        p for p in viewer.plugins if isinstance(p, sapien.utils.viewer.viewer.TransformWindow))

    while True:
        env.render_human()
        execute = False
        if viewer.window.key_press("h"):
            print("""Keys:
  h          print help
  g          toggle gripper open/close
  u / j      move ghost arm up / down
  arrow keys move ghost arm in the arrow direction
  n          execute motion plan to the ghost pose
  c          finish episode and continue to the next one
  q          quit (saves trajectories and videos)""")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        elif viewer.window.key_press("n"):
            execute = True
        elif viewer.window.key_press("g") and robot_has_gripper:
            if gripper_open:
                gripper_open = False
                planner.close_gripper()
            else:
                gripper_open = True
                planner.open_gripper()
        elif viewer.window.key_press("u"):
            select_panda_hand()
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[0, 0, -0.01])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            select_panda_hand()
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[0, 0, +0.01])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            select_panda_hand()
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[+0.01, 0, 0])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            select_panda_hand()
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[-0.01, 0, 0])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            select_panda_hand()
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[0, -0.01, 0])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            select_panda_hand()
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[0, +0.01, 0])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()

        if execute:
            # Gizmo -> TCP z-offset is hardcoded per robot type.
            z_offset = 0.15 if env.unwrapped.robot_uids == "panda_stick" else 0.1
            result = planner.move_to_pose_with_screw(
                transform_window._gizmo_pose * sapien.Pose([0, 0, z_offset]),
                dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                planner.follow_path(result)
            else:
                print("Plan failed or too long; try a closer sub-goal")
            execute = False


def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/teleop/"
    env = gym.make(args.env_id, obs_mode=args.obs_mode, control_mode="pd_joint_pos",
                   render_mode="rgb_array", reward_mode="none", enable_shadow=True,
                   viewer_camera_configs=dict(shader_pack=args.viewer_shader))
    env = RecordEpisode(env, output_dir=output_dir, trajectory_name="trajectory",
                        save_video=False, info_on_video=False,
                        source_type="teleoperation",
                        source_desc="teleoperation via click+drag")
    num_trajs = 0
    seed = 0
    env.reset(seed=seed)
    while True:
        print(f"Collecting trajectory {num_trajs + 1}, seed={seed}")
        code = solve(env, debug=False, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            continue
        elif code == "restart":
            env.reset(seed=seed, options=dict(save_trajectory=False))

    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    print(f"Trajectories saved to {h5_file_path}")

    if args.save_video:
        print(f"Saving videos to {output_dir}")
        with h5py.File(h5_file_path) as trajectory_data, open(json_file_path) as f:
            json_data = json.load(f)
        env = gym.make(args.env_id, obs_mode=args.obs_mode,
                       control_mode="pd_joint_pos", render_mode="rgb_array",
                       reward_mode="none",
                       human_render_camera_configs=dict(shader_pack=args.video_saving_shader))
        env = RecordEpisode(env, output_dir=output_dir, trajectory_name="trajectory",
                            save_video=True, info_on_video=False,
                            save_trajectory=False, video_fps=30)
        with h5py.File(h5_file_path) as trajectory_data:
            for episode in json_data["episodes"]:
                traj_id = f"traj_{episode['episode_id']}"
                data = trajectory_data[traj_id]
                env.reset(**episode["reset_kwargs"])
                env.base_env.set_state_dict(
                    trajectory_utils.dict_to_list_of_dicts(data["env_states"])[0])
                for action in np.array(data["actions"]):
                    env.step(action)
        env.close()


if __name__ == "__main__":
    main(parse_args())
