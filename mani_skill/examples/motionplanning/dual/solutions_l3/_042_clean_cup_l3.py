import numpy as np
import sapien

from mani_skill.envs.tasks import TwoRobotCleanCupEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: TwoRobotCleanCupEnvL3, seed=None, debug=False, vis=False):
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

    obb = get_actor_obb(env.brush)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.brush.pose.sp.p)
    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.2])

    wipe_point_a = env.clean_point_a[0].detach().cpu().numpy()
    wipe_point_b = env.clean_point_b[0].detach().cpu().numpy()
    wipe_pose_1 = sapien.Pose(wipe_point_a, grasp_pose.q)
    wipe_pose_2 = sapien.Pose(wipe_point_b, grasp_pose.q)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([-0.05, 0.0, 0.0]), other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(lift_pose, other_gripper_state=right_planner.gripper_state)
    for _ in range(3):
        left_planner.move_to_pose_with_screw(wipe_pose_1, other_gripper_state=right_planner.gripper_state)
        left_planner.move_to_pose_with_screw(wipe_pose_2, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(lift_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([-0.05, 0.02, 0.0]), other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
