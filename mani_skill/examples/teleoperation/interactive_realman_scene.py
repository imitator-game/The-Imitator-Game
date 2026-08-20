import argparse
from ast import parse
from typing import Annotated
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.examples.motionplanning.realman.motionplanner import \
    RealmanArmMotionPlanningSolver
import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "RealmanPickCubeYCB-v1"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "realman"
    """The robot to use. Currently only 'realman' and 'realman_mobile_base' are supported"""
    record_dir: str = "demos"
    """directory to record the demonstration data and optionally videos"""
    save_video: bool = False
    """whether to save the videos of the demonstrations after collecting them all"""
    viewer_shader: str = "rt-fast"
    """the shader to use for the viewer. 'default' is fast but lower-quality shader, 'rt' and 'rt-fast' are the ray tracing shaders"""
    video_saving_shader: str = "rt-fast"
    """the shader to use for the videos of the demonstrations. 'minimal' is the fast shader, 'rt' and 'rt-fast' are the ray tracing shaders"""


def parse_args() -> Args:
    return tyro.cli(Args)


def fix_head_joints(env):
    """Fix head joints to prevent drift - improved version"""
    # Get current qpos
    current_qpos = env.agent.robot.get_qpos()
    if hasattr(current_qpos, 'cpu'):
        current_qpos = current_qpos.cpu().numpy()

    # Find head joint indices dynamically
    active_joints = env.agent.robot.get_active_joints()
    head_indices = []
    for i, joint in enumerate(active_joints):
        if "head_joint" in joint.get_name():
            head_indices.append(i)

    # If no head joints found by name, assume first two joints (as in original code)
    if not head_indices and len(active_joints) >= 2:
        head_indices = [0, 1]

    # Set head joints to fixed position (0.0)
    new_qpos = current_qpos.copy()[0]
    for idx in head_indices:
        new_qpos[idx] = 0.0

    # Apply the new qpos
    env.agent.robot.set_qpos(new_qpos)

    # Set high stiffness and damping for head joints to prevent drift
    for idx in head_indices:
        active_joints[idx].set_drive_properties(
            stiffness=10000,  # High stiffness
            damping=1000,  # High damping
            force_limit=1000
        )


def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/drag_teleop/"
    env = gym.make(
        args.env_id,
        robot_uids=args.robot_uid,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader),
        sim_backend="physx_cpu",
        render_backend="gpu",
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system",
    )
    num_trajs = 0
    seed = 0
    env.reset(seed=seed)

    # Fix head joints at initialization
    fix_head_joints(env)

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
            # Fix head joints after reset
            fix_head_joints(env)
            continue
        elif code == "restart":
            env.reset(seed=seed, options=dict(save_trajectory=False))
            # Fix head joints after reset
            fix_head_joints(env)
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    print(f"Trajectories saved to {h5_file_path}")
    if args.save_video:
        print(f"Saving videos to {output_dir}")

        trajectory_data = h5py.File(h5_file_path)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        env = gym.make(
            args.env_id,
            robot_uids=args.robot_uid,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="none",
            human_render_camera_configs=dict(shader_pack=args.video_saving_shader),
        )
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video=True,
            info_on_video=False,
            save_trajectory=False,
            video_fps=30
        )
        for episode in json_data["episodes"]:
            traj_id = f"traj_{episode['episode_id']}"
            data = trajectory_data[traj_id]
            env.reset(**episode["reset_kwargs"])
            env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

            env.base_env.set_state_dict(env_states_list[0])
            for action in np.array(data["actions"]):
                env.step(action)

        trajectory_data.close()
        env.close()
        del env


def solve(env: BaseEnv, debug=False, vis=False):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    robot_has_gripper = True
    left_planner = RealmanArmMotionPlanningSolver(
        env,
        arm_side="left",
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_acc_limits=0.5,
        joint_vel_limits=0.5,
    )
    right_planner = RealmanArmMotionPlanningSolver(
        env,
        arm_side="right",
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_acc_limits=0.5,
        joint_vel_limits=0.5,
    )
    viewer = env.render_human()
    planner = right_planner
    arm_to_control = 1

    last_checkpoint_state = None
    gripper_open = True

    # Store current TCP poses
    left_tcp_entity = sapien_utils.get_obj_by_name(env.unwrapped.agent.robot.links, "l_link7")._objs[0].entity
    right_tcp_entity = sapien_utils.get_obj_by_name(env.unwrapped.agent.robot.links, "r_link7")._objs[0].entity

    def select_realman_tcp():
        if arm_to_control == 0:
            viewer.select_entity(left_tcp_entity)
        else:
            viewer.select_entity(right_tcp_entity)

    select_realman_tcp()

    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin

    # Track the last target pose
    last_target_pose = None

    # Fix head joints at the start of solve
    fix_head_joints(env)

    current_full_qpos = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
    left_planner.last_commanded_qpos = current_full_qpos.copy()
    right_planner.last_commanded_qpos = current_full_qpos.copy()

    while True:
        transform_window.enabled = True
        env.render_human()

        # Get current TCP pose
        if arm_to_control == 0:
            current_tcp_pose = env.unwrapped.agent.left_tcp.pose
        else:
            current_tcp_pose = env.unwrapped.agent.right_tcp.pose

        execute_current_pose = False

        if viewer.window.key_press("0"):
            arm_to_control = 0
            # Synchronize target position before switching
            if hasattr(right_planner, 'last_commanded_qpos') and right_planner.last_commanded_qpos is not None:
                left_planner.last_commanded_qpos = right_planner.last_commanded_qpos.copy()
            planner = left_planner
            select_realman_tcp()
            # Fix gizmo initialization
            tcp_pose = env.unwrapped.agent.left_tcp.pose
            if hasattr(tcp_pose, 'to_transformation_matrix'):
                if hasattr(tcp_pose.to_transformation_matrix(), '__getitem__'):
                    transform_window.gizmo_matrix = tcp_pose.to_transformation_matrix()[0]
                else:
                    transform_window.gizmo_matrix = tcp_pose.to_transformation_matrix()

        elif viewer.window.key_press("1"):
            arm_to_control = 1
            # Synchronize target position before switching
            if hasattr(left_planner, 'last_commanded_qpos') and left_planner.last_commanded_qpos is not None:
                right_planner.last_commanded_qpos = left_planner.last_commanded_qpos.copy()
            planner = right_planner
            select_realman_tcp()
            # Fix gizmo initialization
            tcp_pose = env.unwrapped.agent.right_tcp.pose
            if hasattr(tcp_pose, 'to_transformation_matrix'):
                if hasattr(tcp_pose.to_transformation_matrix(), '__getitem__'):
                    transform_window.gizmo_matrix = tcp_pose.to_transformation_matrix()[0]
                else:
                    transform_window.gizmo_matrix = tcp_pose.to_transformation_matrix()
        elif viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            0: control left arm
            1: control right arm
            g: toggle gripper to close/open (if there is a gripper)
            u: move the realman tcp up
            j: move the realman tcp down
            arrow_keys: move the realman tcp in the direction of the arrow keys
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost realman arm
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data. Save trajectories and optionally videos.
            """)
            pass
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g") and robot_has_gripper:
            if gripper_open:
                gripper_open = False
                if arm_to_control == 0:
                    _, reward, _, _, info = planner.close_gripper()
                else:
                    _, reward, _, _, info = planner.close_gripper()
            else:
                gripper_open = True
                if arm_to_control == 0:
                    _, reward, _, _, info = planner.open_gripper()
                else:
                    _, reward, _, _, info = planner.open_gripper()
            # Fix head joints after gripper action
            fix_head_joints(env)
        elif viewer.window.key_press("u"):
            select_realman_tcp()
            # Move in world Z direction
            current_mat = transform_window._gizmo_pose.to_transformation_matrix()
            current_mat[2, 3] += 0.01  # Move up in world Z
            transform_window.gizmo_matrix = current_mat
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            select_realman_tcp()
            # Move down in world Z direction
            current_mat = transform_window._gizmo_pose.to_transformation_matrix()
            current_mat[2, 3] -= 0.01  # Move down in world Z
            transform_window.gizmo_matrix = current_mat
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            select_realman_tcp()
            # Move in world +X direction
            current_mat = transform_window._gizmo_pose.to_transformation_matrix()
            current_mat[0, 3] += 0.01  # Move in +X
            transform_window.gizmo_matrix = current_mat
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            select_realman_tcp()
            # Move in world -X direction
            current_mat = transform_window._gizmo_pose.to_transformation_matrix()
            current_mat[0, 3] -= 0.01  # Move in -X
            transform_window.gizmo_matrix = current_mat
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            select_realman_tcp()
            # Move in world -Y direction
            current_mat = transform_window._gizmo_pose.to_transformation_matrix()
            current_mat[1, 3] -= 0.01  # Move in -Y
            transform_window.gizmo_matrix = current_mat
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            select_realman_tcp()
            # Move in world +Y direction
            current_mat = transform_window._gizmo_pose.to_transformation_matrix()
            current_mat[1, 3] += 0.01  # Move in +Y
            transform_window.gizmo_matrix = current_mat
            transform_window.update_ghost_objects()

        if execute_current_pose:
            # Get the target pose from gizmo
            target_pose = transform_window._gizmo_pose
            last_target_pose = target_pose

            # Plan to the target pose
            result = planner.move_to_pose_with_screw(target_pose, dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _, _, info = planner.follow_path(result, refine_steps=0)

                # Fix head joints after movement to prevent drift
                fix_head_joints(env)

                # Update gizmo to new TCP position after movement
                if arm_to_control == 0:
                    new_tcp_pose = env.unwrapped.agent.left_tcp.pose
                else:
                    new_tcp_pose = env.unwrapped.agent.right_tcp.pose
                transform_window.gizmo_matrix = np.array(new_tcp_pose.to_transformation_matrix())[0]
            else:
                if result == -1:
                    print("Plan failed")
                else:
                    print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False

        # Periodically fix head joints to prevent gradual drift
        # This is called less frequently to avoid performance impact
        if hasattr(env, '_elapsed_steps') and env._elapsed_steps % 10 == 0:
            fix_head_joints(env)

    return args


if __name__ == "__main__":
    main(parse_args())