#!/usr/bin/env python3

import argparse
from ast import parse
from typing import Annotated, Optional
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

# ROS imports
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Float64MultiArray, Header
import threading
from scipy.spatial.transform import Rotation


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "RealmanPickCubeYCB-v1"
    obs_mode: str = "rgb+depth"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "realman"
    record_dir: str = "demos"
    save_video: bool = True
    viewer_shader: str = "rt-fast"
    video_saving_shader: str = "rt-fast"
    vr_control_mode: str = "joint_pose"  # or "ee_pose", Changed default to ee_pose
    use_ros: bool = True  # Flag to enable ROS


class VRRealmanROSController:
    def __init__(self, env: BaseEnv, args: Args):
        self.env = env
        self.args = args

        # Initialize ROS if enabled
        if args.use_ros:
            if not rospy.core.is_initialized():
                rospy.init_node('vr_realman_controller', anonymous=True)

            # Publishers
            self.joint_state_pub = rospy.Publisher('/io_teleop/joint_states', JointState, queue_size=1)
            self.gripper_state_pub = rospy.Publisher('/io_teleop/gripper_states', JointState, queue_size=1)

            # Subscribers
            self.ee_pose_sub = rospy.Subscriber('/io_teleop/target_ee_poses', PoseArray, self.ee_pose_callback)
            self.joint_cmd_sub = rospy.Subscriber('/io_teleop/joint_cmd', JointState, self.joint_cmd_callback)
            self.gripper_cmd_sub = rospy.Subscriber('/io_teleop/target_gripper_status', JointState,
                                                    self.gripper_cmd_callback)

            # State storage
            self.target_ee_poses = None
            self.target_joint_cmd = None
            self.target_gripper_status = [0.0, 0.0]

            # Publishing thread
            self.pub_thread = threading.Thread(target=self.publish_loop)
            self.pub_thread.daemon = True
            self.pub_running = True
            self.pub_thread.start()

            rospy.loginfo("VR Realman ROS Controller initialized")

    def ee_pose_callback(self, msg: PoseArray):
        """Callback for target end-effector poses"""
        if len(msg.poses) >= 2:
            self.target_ee_poses = msg
            # rospy.loginfo(f"Received target EE poses: {msg.poses[:2]}")

    def joint_cmd_callback(self, msg: JointState):
        """Callback for joint commands"""
        self.target_joint_cmd = msg
        # rospy.loginfo(f"Received joint command: {self.target_joint_cmd}")

    def gripper_cmd_callback(self, msg: JointState):
        """Callback for gripper commands"""
        if len(msg.position) >= 2:
            self.target_gripper_status = list(msg.position[:2])
            # rospy.loginfo(f"Received gripper command: {self.target_gripper_status}")

    def publish_loop(self):
        """Publishing thread to send robot state"""
        rate = rospy.Rate(100)  # 100 Hz

        while self.pub_running and not rospy.is_shutdown():
            self.publish_robot_state()
            rate.sleep()

    def publish_robot_state(self):
        """Publish current robot state to ROS topics"""
        robot = self.env.unwrapped.agent.robot

        # Get joint positions
        qpos = robot.get_qpos()[0]
        left_arm_indices = [3, 5, 7, 9, 11, 13, 15]  # l_joint1-7
        left_arm_current = [float(qpos[i]) for i in left_arm_indices]

        # Right arm joints (extracted in actual order)
        right_arm_indices = [2, 4, 6, 8, 10, 12, 14]  # r_joint1-7
        right_arm_current = [float(qpos[i]) for i in right_arm_indices]
        # Skip head joints (first 2) and get arm joints
        arm_joints = right_arm_current + left_arm_current  # 14 joints total
        # print(f"arm_joints: {arm_joints}")

        # Publish joint states
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [
            'r_joint1', 'r_joint2', 'r_joint3', 'r_joint4', 'r_joint5', 'r_joint6', 'r_joint7',
            'l_joint1', 'l_joint2', 'l_joint3', 'l_joint4', 'l_joint5', 'l_joint6', 'l_joint7',
        ]
        joint_msg.position = arm_joints
        joint_msg.velocity = [0.0] * 14
        joint_msg.effort = [0.0] * 14
        self.joint_state_pub.publish(joint_msg)
        # rospy.loginfo(f"Publish joint states: {arm_joints}")

        # Publish gripper states
        gripper_msg = JointState()
        gripper_msg.header.stamp = rospy.Time.now()
        gripper_msg.name = ['gripper_0', 'gripper_1']  # [right, left]
        gripper_msg.position = self.target_gripper_status
        self.gripper_state_pub.publish(gripper_msg)
        # rospy.loginfo(f"Publish gripper states: {self.target_gripper_status}")

    def pose_array_to_sapien_pose(self, pose_msg: Pose):
        """Convert ROS Pose to Sapien Pose"""
        # Position
        p = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])

        # Orientation (ROS uses xyzw, Sapien uses wxyz)
        q = np.array([
            pose_msg.orientation.w,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z
        ])

        return sapien.Pose(p, q)

    def get_ee_control_action(self):
        """Get control action from EE poses via ROS"""
        print(f"self.target_ee_poses: {self.target_ee_poses}")
        if self.target_ee_poses is None or len(self.target_ee_poses.poses) < 2:
            return None

        # Convert ROS poses to Sapien poses
        right_target_pose = self.pose_array_to_sapien_pose(self.target_ee_poses.poses[0])
        print(f"get right_target_pose: {right_target_pose}")
        left_target_pose = self.pose_array_to_sapien_pose(self.target_ee_poses.poses[1])
        print(f"get left_target_pose: {left_target_pose}")

        return {
            'right_pose': right_target_pose,
            'left_pose': left_target_pose,
            'gripper_status': self.target_gripper_status
        }

    def get_joint_control_action(self):
        """Get control action from joint commands via ROS"""
        if self.target_joint_cmd is None:
            return None

        # Extract joint positions
        joint_positions = np.array(self.target_joint_cmd.position[:14])

        # Create action array
        action = np.zeros(18)
        action[9:16] = joint_positions[7:14]  # left arm
        action[2:9] = joint_positions[0:7]  # right arm
        action[16] = self.target_gripper_status[1]  # left gripper
        action[17] = self.target_gripper_status[0]  # right gripper
        # action = np.array([
        #         # head fixed to 0
        #         0.0, 0.0,  # head_joint1-2
        #         # right arm: r_joint1-7
        #         0.82383, -0.17647, 0, -1.44114, -2.09, 0.926211, -1.55079,
        #         # left arm: l_joint1-7
        #         -0.81842, -0.176488, 0, -1.37455, -0.875422, -0.997037, -1.54696,
        #         # grippers
        #         0, 0
        #     ])

        # Inverse mapping: restore the qpos format from the action
        qpos_reconstructed = np.zeros(18)  # Assume qpos length of 16, adjust as needed

        # Define joint indices (consistent with publish_robot_state)
        left_arm_indices = [3, 5, 7, 9, 11, 13, 15]  # l_joint1-7
        right_arm_indices = [2, 4, 6, 8, 10, 12, 14]  # r_joint1-7

        # Extract joint data from the action and place it back into the original positions
        qpos_reconstructed[right_arm_indices] = action[2:9]  # Left arm: action[2:9] -> qpos[3,5,7,9,11,13,15]
        qpos_reconstructed[left_arm_indices] = action[9:16]  # Right arm: action[9:16] -> qpos[2,4,6,8,10,12,14]

        qpos_reconstructed[16] = self.target_gripper_status[1]  # left gripper
        qpos_reconstructed[17] = self.target_gripper_status[0]  # right gripper

        return qpos_reconstructed

    def shutdown(self):
        """Clean shutdown"""
        if self.args.use_ros:
            self.pub_running = False
            if hasattr(self, 'pub_thread') and self.pub_thread.is_alive():
                self.pub_thread.join()


def parse_args() -> Args:
    return tyro.cli(Args)


def solve(env: BaseEnv, args: Args, debug=False, vis=False):
    """Main control loop with ROS integration"""
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    # Initialize ROS controller
    ros_controller = None
    if args.use_ros:
        ros_controller = VRRealmanROSController(env, args)

    # Initialize motion planners
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

    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin

    execute_current_pose = False

    try:
        while True:
            transform_window.enabled = True
            env.render_human()

            if viewer.window.key_press("h"):
                print("""Available commands:
                n: execute command via ROS
                z: stop execute command
                c: stop this episode and record the trajectory
                q: quit the script
                """)
            elif viewer.window.key_press("q"):
                return "quit"
            elif viewer.window.key_press("c"):
                return "continue"
            elif viewer.window.key_press("n"):
                execute_current_pose = True
                rospy.loginfo("Started ROS control mode")
            elif viewer.window.key_press("z"):
                execute_current_pose = False
                rospy.loginfo("Stopped ROS control mode")

            if execute_current_pose and args.use_ros and ros_controller:
                if args.vr_control_mode == "ee_pose":
                    # Get EE control action
                    control_data = ros_controller.get_ee_control_action()

                    if control_data is not None:
                        # Plan right arm
                        right_result = right_planner.move_to_pose_with_screw(
                            control_data['right_pose'], dry_run=True
                        )
                        if right_result != -1 and len(right_result["position"]) < 150:
                            _, reward, _, _, info = right_planner.follow_path(right_result, refine_steps=0)

                        # Plan left arm
                        left_result = left_planner.move_to_pose_with_screw(
                            control_data['left_pose'], dry_run=True
                        )
                        if left_result != -1 and len(left_result["position"]) < 150:
                            _, reward, _, _, info = left_planner.follow_path(left_result, refine_steps=0)

                elif args.vr_control_mode == "joint_pose":
                    # Get joint control action
                    action = ros_controller.get_joint_control_action()

                    if action is not None:
                        print(f"deploy action: {action}")
                        obs, reward, terminated, truncated, info = env.step(action)
                        env.render_human()
                        if terminated or truncated:
                            rospy.logwarn("Episode ended")
                            execute_current_pose = False

                # Small delay to prevent tight loop
                rospy.sleep(0.01)

    finally:
        if ros_controller:
            ros_controller.shutdown()

    return args


def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/vr_teleop/"
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader),
        sim_backend="physx_cpu",
        render_backend="gpu",
        max_episode_steps=100000,
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="vr_trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the vr system",
        max_steps_per_video=100000,
    )

    num_trajs = 0
    seed = 0
    env.reset(seed=seed)

    while True:
        print(f"Collecting trajectory {num_trajs + 1}, seed={seed}")
        code = solve(env, args, debug=False, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
        elif code == "restart":
            env.reset(seed=seed, options=dict(save_trajectory=False))

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


if __name__ == "__main__":
    main(parse_args())