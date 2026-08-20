import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotFoldBoxEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: TwoRobotFoldBoxEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    left_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[0].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=0
    )
    right_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[1].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=1
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    left_init_pose = env.left_init_pose
    right_init_pose = env.right_agent.tcp.pose

    # -------------------------------------------------------------------------- #
    # Close (dynamic waypoints from lid position to handle L2 models)
    # -------------------------------------------------------------------------- #
    lid_pos = env.partnet_link_positions()[0].cpu().numpy()
    robot_pos = env.left_agent.robot.pose.p[0].cpu().numpy()
    dir_xy = lid_pos[:2] - robot_pos[:2]
    norm = np.linalg.norm(dir_xy)
    if norm < 1e-6:
        dir_xy = np.array([0.0, 1.0])
    else:
        dir_xy = dir_xy / norm
    approach_pos = np.array([lid_pos[0], lid_pos[1], lid_pos[2] + 0.05])
    approach_pos[:2] -= dir_xy * 0.14
    push_pos = np.array([lid_pos[0], lid_pos[1], lid_pos[2] + 0.05])
    push_pos[:2] -= dir_xy * 0.02
    retreat_pos = approach_pos + np.array([0.0, 0.0, 0.10])

    waypoint1 = sapien.Pose(approach_pos, left_init_pose.q)
    waypoint2 = sapien.Pose(push_pos, left_init_pose.q)
    waypoint3 = sapien.Pose(retreat_pos, left_init_pose.q)

    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    current_pose = env.left_agent.tcp.pose
    mid_pos = (current_pose.p[0].cpu().numpy() + waypoint1.p) * 0.5
    waypoint1_mid = sapien.Pose(mid_pos, waypoint1.q)
    waypoint1_mid_rot = waypoint1_mid * sapien.Pose([0.0, -0.1, -0.02], euler2quat(0, 0, -np.deg2rad(30)))
    left_planner.move_to_pose_with_screw(waypoint1_mid_rot, other_gripper_state=right_planner.gripper_state)
    # left_planner.move_to_pose_with_screw(waypoint1, other_gripper_state=right_planner.gripper_state)
    waypoint2_adjust = waypoint2 * sapien.Pose([-0.1, -0.12, -0.01], euler2quat(-np.deg2rad(20), 0, -np.deg2rad(30)))
    left_planner.move_to_pose_with_screw(waypoint2_adjust, other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    waypoint2_3 = waypoint2_adjust * sapien.Pose([0.0, 0.0, -0.1], euler2quat(0, 0, 0)) 
    left_planner.move_to_pose_with_screw(waypoint2_3, other_gripper_state=right_planner.gripper_state)
    #left_planner.move_to_pose_with_screw(waypoint3, other_gripper_state=right_planner.gripper_state)
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
