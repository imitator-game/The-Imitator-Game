import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPlaceFruitBoxEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import is_l2_enabled, is_l1_enabled

OPEN = 1
SETTLE_STEPS = 10


def _hold_action(env, gripper_state=OPEN):
    env = env.unwrapped
    left_qpos = env.agent.agents[0].robot.qpos[0][:7].cpu().numpy()
    right_qpos = env.agent.agents[1].robot.qpos[0][:7].cpu().numpy()
    left_action = np.hstack([left_qpos, gripper_state])
    right_action = np.hstack([right_qpos, gripper_state])
    return {
        "panda_wristcam-0": left_action,
        "panda_wristcam-1": right_action,
    }


def _settle_env(env, steps=SETTLE_STEPS):
    hold_action = _hold_action(env, gripper_state=OPEN)
    for _ in range(steps):
        env.step(hold_action)


def solve(env: TwoRobotPlaceFruitBoxEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    _settle_env(env, steps=SETTLE_STEPS)
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

    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose

    # -------------------------------------------------------------------------- #
    # Apple
    # -------------------------------------------------------------------------- #
    obb = get_actor_obb(env.apple)
    approaching = np.array([0, 0, -1])
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.apple.pose.sp.p)
    if is_l1_enabled(): 
        grasp_target_pose = grasp_pose * sapien.Pose([0, 0, 0.01])
    elif is_l2_enabled(): 
        grasp_target_pose = grasp_pose * sapien.Pose([0, 0, -0.03])
    else: 
        grasp_target_pose = grasp_pose * sapien.Pose([0, 0, 0.03])
    goal_pose = sapien.Pose(env.box.pose.sp.p, grasp_pose.q)
    reach_pose2 = grasp_pose * sapien.Pose([0, -0.02, -0.2])
    reach_pose1 = grasp_pose * sapien.Pose([0, -0.02, -0.15])
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(grasp_target_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_planner.gripper_state)
    reach_pose3 = goal_pose * sapien.Pose([-0.05, 0, -0.2])
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(goal_pose * sapien.Pose([-0.05, 0, -0.08]), other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Pear
    # -------------------------------------------------------------------------- #
    obb = get_actor_obb(env.pear)
    approaching = np.array([0, 0, -1])
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.pear.pose.sp.p)
    goal_pose = sapien.Pose(env.box.pose.sp.p, grasp_pose.q)
    reach_pose2 = grasp_pose * sapien.Pose([0.0, 0, -0.25])
    reach_pose1 = grasp_pose * sapien.Pose([0.0, 0, -0.15])
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0.0, 0, -0.03]), other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_planner.gripper_state)
    reach_pose3 = goal_pose * sapien.Pose([0.05, 0, -0.25])
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(goal_pose * sapien.Pose([0.05, 0, -0.08]), other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
