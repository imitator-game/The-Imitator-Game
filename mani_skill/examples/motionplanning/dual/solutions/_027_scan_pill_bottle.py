import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotScanPillBottleEnv
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import is_lr_mirror_enabled, mirror_pose
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver


def solve(env: TwoRobotScanPillBottleEnv, seed=None, debug=False, vis=False):
    """
    Phases:
    1. Left robot: Pick up scanner
    2. Right robot: Pick up pill bottle
    3. Right robot: Move pill bottle to [0, -0.1, 0.15]
    4. Left robot: Move scanner to [0, 0.1, 0.15]
    5. Wait 1 second
    6. Right robot: Place pill bottle back
    7. Left robot: Place scanner back
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

    # -------------------------------------------------------------------------- #
    # Define grasp poses for scanner (left robot)
    # -------------------------------------------------------------------------- #
    approaching_left = np.array([0, 0, -1])  # From above (downward)

    # Extract current TCP closing direction and make it orthogonal to approaching
    target_closing_left = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # Gram-Schmidt orthogonalization
    target_closing_left = target_closing_left - (target_closing_left @ approaching_left) * approaching_left
    target_closing_left = target_closing_left / np.linalg.norm(target_closing_left)

    grasp_pose_left = env.left_agent.build_grasp_pose(
        approaching_left, target_closing_left, env.scanner.pose.sp.p
    )

    # -------------------------------------------------------------------------- #
    # Define grasp poses for pill bottle (right robot)
    # -------------------------------------------------------------------------- #
    approaching_right = np.array([0, 0, -1])  # From above (downward)

    # Extract current TCP closing direction and make it orthogonal to approaching
    target_closing_right = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # Gram-Schmidt orthogonalization
    target_closing_right = target_closing_right - (target_closing_right @ approaching_right) * approaching_right
    target_closing_right = target_closing_right / np.linalg.norm(target_closing_right)

    # Build grasp pose for pill bottle
    grasp_pose_right = env.right_agent.build_grasp_pose(
        approaching_right, target_closing_right, env.pillbottle.pose.sp.p
    )

    # -------------------------------------------------------------------------- #
    # Phase 1: Left robot picks up scanner
    # -------------------------------------------------------------------------- #
    # Reach above scanner
    reach_pose_left = grasp_pose_left * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        reach_pose_left,
        other_gripper_state=right_planner.gripper_state
    )

    # Descend and grasp scanner
    left_planner.move_to_pose_with_screw(
        grasp_pose_left * sapien.Pose([0, 0, -0.03]),
        other_gripper_state=right_planner.gripper_state
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # Lift scanner
    lift_pose_left = grasp_pose_left * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        lift_pose_left,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 2: Right robot picks up pill bottle
    # -------------------------------------------------------------------------- #
    # Reach above pill bottle
    reach_pose_right = grasp_pose_right * sapien.Pose([0, 0, -0.25])
    right_planner.move_to_pose_with_screw(
        reach_pose_right,
        other_gripper_state=left_planner.gripper_state
    )

    # Descend and grasp pill bottle
    right_planner.move_to_pose_with_screw(
        grasp_pose_right * sapien.Pose([0, 0, -0.12]),
        other_gripper_state=left_planner.gripper_state
    )
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)

    # Lift pill bottle
    lift_pose_right = grasp_pose_right * sapien.Pose([0, 0, -0.15])
    right_planner.move_to_pose_with_screw(
        lift_pose_right,
        other_gripper_state=left_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 3: Right robot moves pill bottle to scanning position [0, 0.1, 0.15]
    # with inward tilt for scanning
    # -------------------------------------------------------------------------- #
    # Tilt angle for scanning (30 degrees)
    tilt_angle_1 = np.pi / 4
    tilt_angle_2 = np.pi / 4

    # Pill bottle tilts inward (towards left/scanner direction)
    # Apply roll rotation around x-axis (left-right tilt)
    pillbottle_scan_pose = sapien.Pose(
        p=[0, 0.1, 0.15],
        q=grasp_pose_right.q
    ) * sapien.Pose(q=euler2quat(-tilt_angle_1, 0, 0))  # Roll inward (tilt left)
    if is_lr_mirror_enabled():
        pillbottle_scan_pose = mirror_pose(pillbottle_scan_pose, mode="full")

    right_planner.move_to_pose_with_screw(
        pillbottle_scan_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 4: Left robot moves scanner to scanning position [0, -0.1, 0.15]
    # with inward tilt for scanning
    # -------------------------------------------------------------------------- #
    # Scanner tilts inward (towards right/pill bottle direction)
    # Apply roll rotation around x-axis (opposite direction)
    scanner_scan_pose = sapien.Pose(
        p=[-0.04, -0.04, 0.25],
        q=grasp_pose_left.q
    ) * sapien.Pose(q=euler2quat(tilt_angle_2, 0, 0))  # Roll inward (tilt right)
    if is_lr_mirror_enabled():
        scanner_scan_pose = mirror_pose(scanner_scan_pose, mode="full")

    left_planner.move_to_pose_with_screw(
        scanner_scan_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 5: Wait 1 second (scanning simulation)
    # -------------------------------------------------------------------------- #
    # Hold current positions for approximately 1 second
    # Use move_to_pose_with_screw with current poses to maintain position
    # for _ in range(3):  # Execute 3 small movements of ~0.3s each
    #     left_planner.move_to_pose_with_screw(
    #         scanner_scan_pose,
    #         other_gripper_state=right_planner.gripper_state
    #     )
    #     right_planner.move_to_pose_with_screw(
    #         pillbottle_scan_pose,
    #         other_gripper_state=left_planner.gripper_state
    #     )

    # -------------------------------------------------------------------------- #
    # Phase 6: Right robot straightens pill bottle and places it down at scan position
    # -------------------------------------------------------------------------- #
    # Step 1: Return to upright pose (remove tilt) at scan position
    pillbottle_upright_pose = sapien.Pose(
        p=[0, 0.15, 0.15],
        q=grasp_pose_right.q
    )
    if is_lr_mirror_enabled():
        pillbottle_upright_pose = mirror_pose(pillbottle_upright_pose, mode="full")
    right_planner.move_to_pose_with_screw(
        pillbottle_upright_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Step 2: Lower to place position (descend to table height)
    pillbottle_place_pose = sapien.Pose(
        p=[0, 0.15, 0.12],  # Lower to near-table height
        q=grasp_pose_right.q
    )
    if is_lr_mirror_enabled():
        pillbottle_place_pose = mirror_pose(pillbottle_place_pose, mode="full")
    right_planner.move_to_pose_with_screw(
        pillbottle_place_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Step 3: Release pill bottle
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    # Step 4: Retract upward
    right_planner.move_to_pose_with_screw(
        pillbottle_upright_pose * sapien.Pose([0, 0, -0.1]),
        other_gripper_state=left_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 7: Left robot straightens scanner and places it down at scan position
    # -------------------------------------------------------------------------- #
    # Step 1: Return to upright pose (remove tilt) at scan position
    scanner_upright_pose = sapien.Pose(
        p=[0, -0.15, 0.15],
        q=grasp_pose_left.q
    )
    if is_lr_mirror_enabled():
        scanner_upright_pose = mirror_pose(scanner_upright_pose, mode="full")
    left_planner.move_to_pose_with_screw(
        scanner_upright_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # Step 2: Lower to place position (descend to table height)
    scanner_place_pose = sapien.Pose(
        p=[0, -0.15, 0.05],  # Lower to near-table height
        q=grasp_pose_left.q
    )
    if is_lr_mirror_enabled():
        scanner_place_pose = mirror_pose(scanner_place_pose, mode="full")
    left_planner.move_to_pose_with_screw(
        scanner_place_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # Step 3: Release scanner
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    # Step 4: Retract upward
    left_planner.move_to_pose_with_screw(
        scanner_upright_pose * sapien.Pose([0, 0, -0.1]),
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Return to initial poses
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        left_init_pose,
        other_gripper_state=right_planner.gripper_state
    )

    right_res = right_planner.move_to_pose_with_screw(
        right_init_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Cleanup
    left_planner.close()
    right_planner.close()

    return left_res, right_res
