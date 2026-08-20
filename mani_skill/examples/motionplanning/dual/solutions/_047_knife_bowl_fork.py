import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotKnifeBowlForkEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import is_lr_mirror_enabled, mirror_xyz

def solve(env: TwoRobotKnifeBowlForkEnv, seed=None, debug=False, vis=False):
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
    lr_mirror = is_lr_mirror_enabled()
    
    # move fork (left arm -> left side, y negative)
    fork_obb = get_actor_obb(env.fork)

    approaching = np.array([0, 0, -1])
    initial_pose = env.left_agent.tcp.pose
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        fork_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, center)

    fork_goal_ori = euler2quat(0, 0, 0)
    rel_pose = env.fork.pose.sp.inv() * grasp_pose
    fork_goal_offset = np.array([0, -0.2, -0.02], dtype=np.float32)
    if lr_mirror:
        fork_goal_offset = mirror_xyz(fork_goal_offset)
    fork_goal_pos = env.bowl.pose.sp.p + fork_goal_offset
    goal_pose = sapien.Pose(fork_goal_pos, fork_goal_ori) * rel_pose
    mid_goal_pose = goal_pose * sapien.Pose([0, 0, -0.05])

    reach_pose1 = grasp_pose * sapien.Pose([0, 0, -0.05])
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(mid_goal_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(goal_pose, other_gripper_state=right_planner.gripper_state)
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(initial_pose, other_gripper_state=right_planner.gripper_state)

    # move knife (right arm -> right side, y positive)
    knife_obb = get_actor_obb(env.knife)

    approaching = np.array([0, 0, -1])
    initial_pose = env.right_agent.tcp.pose
    target_closing = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        knife_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.right_agent.build_grasp_pose(approaching, closing, center)

    knife_goal_ori = euler2quat(0, 0, 0)
    rel_pose = env.knife.pose.sp.inv() * grasp_pose
    knife_goal_offset = np.array([0, 0.2, -0.02], dtype=np.float32)
    if lr_mirror:
        knife_goal_offset = mirror_xyz(knife_goal_offset)
    knife_goal_pos = env.bowl.pose.sp.p + knife_goal_offset
    goal_pose = sapien.Pose(knife_goal_pos, knife_goal_ori) * rel_pose
    mid_goal_pose = goal_pose * sapien.Pose([0, 0, -0.05])

    reach_pose1 = grasp_pose * sapien.Pose([0, 0, -0.05])
    right_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=left_planner.gripper_state)
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(mid_goal_pose, other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(goal_pose, other_gripper_state=left_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(initial_pose, other_gripper_state=left_planner.gripper_state)
    
    return left_res, right_res
