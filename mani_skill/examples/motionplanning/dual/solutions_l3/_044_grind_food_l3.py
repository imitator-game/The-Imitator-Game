import numpy as np
import sapien

from mani_skill.envs.tasks import TwoRobotGrindFoodEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: TwoRobotGrindFoodEnvL3, seed=None, debug=False, vis=False):
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

    obb = get_actor_obb(env.grind)

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

    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.grind.pose.sp.p)
    goal_pose = sapien.Pose(
        p=env.bowl.pose.sp.p + np.array([0.0, 0.0, 0.0], dtype=np.float32),
        q=grasp_pose.q,
    )
    grasp_target_world_offset = np.array([0.0, 0.0, 0.2], dtype=np.float32)
    lift_world_offset = np.array([0.0, 0.0, 0.4], dtype=np.float32)
    goal_reach_world_offset = np.array([0.0, 0.0, 0.4], dtype=np.float32)
    goal_place_world_offset = np.array([0.0, 0.0, 0.2], dtype=np.float32)
    reach_world_offset = np.array([0.0, 0.0, 0.3], dtype=np.float32)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose1 = sapien.Pose(
        p=grasp_pose.p + reach_world_offset,
        q=grasp_pose.q,
    )
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    grasp_target_pose = sapien.Pose(
        p=grasp_pose.p + grasp_target_world_offset,
        q=grasp_pose.q,
    )
    left_planner.move_to_pose_with_screw(
        grasp_target_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    lift_pose = sapien.Pose(
        p=grasp_pose.p + lift_world_offset,
        q=grasp_pose.q,
    )
    left_planner.move_to_pose_with_screw(
        lift_pose,
        other_gripper_state=right_planner.gripper_state,
    )

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    reach_pose3 = sapien.Pose(
        p=goal_pose.p + goal_reach_world_offset,
        q=goal_pose.q,
    )
    goal_place_pose = sapien.Pose(
        p=goal_pose.p + goal_place_world_offset,
        q=goal_pose.q,
    )
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(
        goal_place_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(
        goal_place_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)


    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
