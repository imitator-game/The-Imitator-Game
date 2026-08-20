import numpy as np
import sapien
from transforms3d.euler import euler2quat
import time

from mani_skill.envs.tasks import TwoRobotScanMilkBoxEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver


def solve(env: TwoRobotScanMilkBoxEnvL3, seed=None, debug=False, vis=False):
    """
    Solution for TwoRobotScanMilkBox-v1 environment.

    Task: Right robot picks milkbox and executes scan/place routine.

    Phases:
    1. Right robot: Pick up milkbox
    2. Right robot: Move milkbox to scan position
    3. Right robot: Place milkbox down
    """
    env.reset(seed=seed)

    # Initialize both planners
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

    env = env.unwrapped

    # Store initial TCP poses for final return
    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose

    scanner_pose_p = env.scanner.pose.sp.p

    # -------------------------------------------------------------------------- #
    # Define grasp poses for milkbox (right robot)
    # -------------------------------------------------------------------------- #
    approaching_right = np.array([0, 0, -1])  # From above (downward)

    # Extract current TCP closing direction and make it orthogonal to approaching
    target_closing_right = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # Gram-Schmidt orthogonalization
    target_closing_right = target_closing_right - (target_closing_right @ approaching_right) * approaching_right
    target_closing_right = target_closing_right / np.linalg.norm(target_closing_right)

    # Build grasp pose for milkbox
    grasp_pose_right = env.right_agent.build_grasp_pose(
        approaching_right, target_closing_right, env.milkbox.pose.sp.p
    )

    # -------------------------------------------------------------------------- #
    # Phase 1: Right robot picks up milkbox
    # -------------------------------------------------------------------------- #
    # Reach above milkbox
    reach_pose_right = grasp_pose_right * sapien.Pose([0, 0, -0.15])
    right_planner.move_to_pose_with_screw(
        reach_pose_right,
        other_gripper_state=left_planner.gripper_state
    )

    # Descend and grasp milkbox
    right_planner.move_to_pose_with_screw(
        grasp_pose_right * sapien.Pose([0, 0, -0.03]),
        other_gripper_state=left_planner.gripper_state
    )
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)

    # Lift milkbox
    lift_pose_right = grasp_pose_right * sapien.Pose([0, 0, -0.15])
    right_planner.move_to_pose_with_screw(
        lift_pose_right,
        other_gripper_state=left_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 2: Right robot moves milkbox to scanning position
    # -------------------------------------------------------------------------- #
    # Tilt angle for scanning (30 degrees)
    tilt_angle_1 = np.pi / 4
    tilt_angle_2 = np.pi / 4

    # Milkbox tilts inward (towards left/scanner direction)
    # Apply roll rotation around x-axis (left-right tilt)
    milkbox_scan_pose = sapien.Pose(
        p=scanner_pose_p,
        q=grasp_pose_right.q
    ) * sapien.Pose(q=euler2quat(-tilt_angle_1, 0, 0))  # Roll inward (tilt left)

    milkbox_scan_pose = milkbox_scan_pose * sapien.Pose(p=[0, -0.05, -0.1])

    right_planner.move_to_pose_with_screw(
        milkbox_scan_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 3: Right robot straightens milkbox and places it down at scan position
    # -------------------------------------------------------------------------- #
    # Step 1: Return to upright pose (remove tilt) at scan position
    milkbox_upright_pose = sapien.Pose(
        p=[0, 0.15, 0.15],
        q=grasp_pose_right.q
    )
    right_planner.move_to_pose_with_screw(
        milkbox_upright_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Step 2: Lower to place position (descend to table height)
    milkbox_place_pose = sapien.Pose(
        p=[0, 0.15, 0.05],  # Lower to near-table height
        q=grasp_pose_right.q
    )
    right_planner.move_to_pose_with_screw(
        milkbox_place_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Step 3: Release milkbox
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    # Step 4: Retract upward
    right_planner.move_to_pose_with_screw(
        milkbox_upright_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Return to initial poses
    # -------------------------------------------------------------------------- #
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    right_res = right_planner.move_to_pose_with_screw(
        right_init_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Cleanup
    left_planner.close()
    right_planner.close()

    return left_res, right_res
