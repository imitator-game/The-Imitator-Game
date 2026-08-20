import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPourKettleEnv
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import mirror_sign
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver


def solve(env: TwoRobotPourKettleEnv, seed=None, debug=False, vis=False):
    """
    Phases:
    1. Grasp kettle
    2. Lift kettle
    3. Move to above cup
    4. Pour (tilt kettle)
    5. Straighten kettle
    6. Return to original position
    7. Place kettle down
    8. Release and retract
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

    # Store initial kettle position for return phase
    kettle_initial_pos = env.kettle.pose.sp.p.copy()

    # -------------------------------------------------------------------------- #
    # Define grasp pose with simple Z-axis rotation (no OBB)
    # -------------------------------------------------------------------------- #
    approaching = np.array([0, 0, -1])  # From above (downward)

    # Set closing direction based on desired Z-axis rotation
    # closing defines the Y-axis of gripper (open/close direction)
    # For z_rotation_angle = 0: closing = [0, 1, 0] (gripper opens along Y-axis)
    # For z_rotation_angle = π/2: closing = [-1, 0, 0] (gripper opens along -X-axis)
    z_rotation_angle = -np.pi / 2  # Adjust this to change gripper orientation

    # Calculate closing direction from Z-axis rotation
    closing = np.array([
        -np.sin(z_rotation_angle),  # X component
        np.cos(z_rotation_angle),   # Y component
        0                           # Z component (always 0, horizontal)
    ])

    # Build grasp pose
    grasp_pose_base = env.left_agent.build_grasp_pose(
        approaching,
        closing,
        env.kettle.pose.sp.p
    )

    # Apply custom height offset
    custom_offset = np.array([0, 0, 0.16])
    custom_position = env.kettle.pose.sp.p + custom_offset
    grasp_pose = sapien.Pose(p=custom_position, q=grasp_pose_base.q)

    # -------------------------------------------------------------------------- #
    # Phase 1: Reach above kettle
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        reach_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 2: Descend and grasp kettle
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        grasp_pose,
        other_gripper_state=right_planner.gripper_state
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 3: Lift kettle
    # -------------------------------------------------------------------------- #
    # lift_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
    # left_planner.move_to_pose_with_screw(
    #     lift_pose,
    #     other_gripper_state=right_planner.gripper_state
    # )

    # -------------------------------------------------------------------------- #
    # Phase 4: Move to above cup (ready for pour)
    # -------------------------------------------------------------------------- #
    # Move to cup position with backward offset in gripper frame
    above_cup_pose_base = sapien.Pose(
        env.cup.pose.sp.p + np.array([0, 0, 0.25]),  # 25cm above cup
        grasp_pose.q
    )
    # Add backward offset in gripper local frame (local X = world Y)
    backward_offset = sapien.Pose([mirror_sign(-0.10), 0, 0])
    above_cup_pose = above_cup_pose_base * backward_offset

    left_planner.move_to_pose_with_screw(
        above_cup_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 5: Pour (tilt kettle)
    # -------------------------------------------------------------------------- #
    # Create a tilted pose for pouring
    pour_rotation = euler2quat(0, mirror_sign(-np.pi / 6), 0)
    current_quat = grasp_pose.q

    # Combine rotations: first the grasp orientation, then the pour tilt
    from scipy.spatial.transform import Rotation as R
    grasp_rot = R.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
    pour_rot = R.from_quat([pour_rotation[1], pour_rotation[2], pour_rotation[3], pour_rotation[0]])
    combined_rot = grasp_rot * pour_rot
    combined_quat = combined_rot.as_quat()  # [x, y, z, w]
    combined_quat_sapien = [combined_quat[3], combined_quat[0], combined_quat[1], combined_quat[2]]  # [w, x, y, z]

    # Pour pose: keep the same position as above_cup_pose, only change orientation
    # This minimizes movement during pour - mainly rotation, minimal translation
    pour_pose = sapien.Pose(
        above_cup_pose.p,  # Same position as Phase 4
        combined_quat_sapien  # Only change: tilted orientation
    )
    left_planner.move_to_pose_with_screw(
        pour_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # Hold pouring position briefly (simulate pouring)
    # Move slightly to simulate pouring motion
    for _ in range(3):
        left_planner.move_to_pose_with_screw(
            pour_pose,
            other_gripper_state=right_planner.gripper_state
        )

    # -------------------------------------------------------------------------- #
    # Phase 6: Straighten kettle (return to upright)
    # -------------------------------------------------------------------------- #
    # left_planner.move_to_pose_with_screw(
    #     above_cup_pose,
    #     other_gripper_state=right_planner.gripper_state
    # )

    # -------------------------------------------------------------------------- #
    # Phase 7: Return to original position
    # -------------------------------------------------------------------------- #
    # Move back to above the original kettle position
    return_above_pose = sapien.Pose(
        kettle_initial_pos + np.array([0, 0, 0.30]), 
        grasp_pose.q
    )
    left_planner.move_to_pose_with_screw(
        return_above_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 8: Place kettle down at original position
    # -------------------------------------------------------------------------- #
    place_pose = sapien.Pose(
        kettle_initial_pos + np.array([0, 0, 0.15]),
        grasp_pose.q
    )
    left_planner.move_to_pose_with_screw(
        place_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 9: Release kettle
    # -------------------------------------------------------------------------- #
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 10: Retract
    # -------------------------------------------------------------------------- #
    retract_pose = place_pose * sapien.Pose([0, 0, -0.1])  # Move up 10cm
    left_planner.move_to_pose_with_screw(
        retract_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 11: Return to initial pose
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
