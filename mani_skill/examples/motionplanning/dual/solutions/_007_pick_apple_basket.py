import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPickAppleBasketEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver


def solve(env: TwoRobotPickAppleBasketEnv, seed=None, debug=False, vis=False):
    """
    Solution for TwoRobotPickAppleBasket-v1 environment.

    Task: Pick up apple and place it into breadbasket, then return to initial position.

    Phases:
    1. Reach above apple
    2. Grasp apple
    3. Lift apple
    4. Move to above basket
    5. Lower into basket
    6. Release apple
    7. Retract
    8. Return to initial pose
    """
    env.reset(seed=seed)

    # Initialize both planners (dual-arm setup requirement)
    left_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[0].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=0  # Left robot
    )
    right_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[1].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=1  # Right robot
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # Store initial TCP poses for final return
    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose

    # -------------------------------------------------------------------------- #
    # Define grasp pose with fixed orientation (no OBB, no rotation)
    # -------------------------------------------------------------------------- #
    approaching = np.array([0, 0, -1])  # From above (downward)

    # Extract current TCP closing direction and make it orthogonal to approaching
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # Gram-Schmidt orthogonalization: remove the component parallel to approaching
    target_closing = target_closing - (target_closing @ approaching) * approaching
    target_closing = target_closing / np.linalg.norm(target_closing)  # Normalize

    # Build grasp pose using orthogonal closing direction
    grasp_pose = env.left_agent.build_grasp_pose(approaching, target_closing, env.apple.pose.sp.p)

    # Define goal pose (basket position)
    goal_pose = sapien.Pose(env.breadbasket.pose.sp.p, grasp_pose.q) * sapien.Pose([0, 0, -0.05])

    # -------------------------------------------------------------------------- #
    # Phase 1: Reach above apple
    # -------------------------------------------------------------------------- #
    reach_pose1 = grasp_pose * sapien.Pose([0, 0, -0.05])
    left_planner.move_to_pose_with_screw(
        reach_pose1,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 2: Descend and grasp apple
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        grasp_pose,
        other_gripper_state=right_planner.gripper_state
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 3: Lift apple
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        lift_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 4: Move to above basket
    # -------------------------------------------------------------------------- #
    reach_pose2 = goal_pose * sapien.Pose([-0.05, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        reach_pose2,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 5: Lower into basket
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        goal_pose * sapien.Pose([-0.05, 0, -0.02]),
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 6: Release apple
    # -------------------------------------------------------------------------- #
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 7: Retract
    # -------------------------------------------------------------------------- #
    retract_pose = goal_pose * sapien.Pose([-0.05, 0, -0.15])  # Move up 10cm
    left_planner.move_to_pose_with_screw(
        retract_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 8: Return to initial pose
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        left_init_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # Cleanup
    left_planner.close()
    right_planner.close()

    # Return results (right robot didn't perform actions, but we need both results)
    return left_res, left_res
