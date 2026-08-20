import numpy as np
import sapien

from mani_skill.envs.tasks import TwoRobotStirSpoonEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import is_l2_enabled


def solve(env: TwoRobotStirSpoonEnv, seed=None, debug=False, vis=False):
    """
    Solution for TwoRobotStirSpoon-v1 environment.

    Task: Right robot picks up spoon from mug, stirs in bowl (3 counterclockwise circles),
    and returns spoon to mug.

    Both planners are initialized but only the right robot executes actions.
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

    env = env.unwrapped

    # Store initial TCP poses for final return
    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose

    spoon_initial_position = env.spoon.pose.sp.p.copy()

    # Waypoints are defined in the env and transformed with the spoon pose
    grasp_pose = env.grasp_pose
    if is_l2_enabled():
        grasp_pose = sapien.Pose(
            p=np.array(grasp_pose.p) + np.array([0.0, 0.0, -0.01]),
            q=grasp_pose.q,
        )
    reach_pose = env.reach_pose
    lift_pose_tilted = env.lift_pose_tilted
    lift_pose_vertical = env.lift_pose_vertical
    vertical_orientation = np.array(lift_pose_vertical.q, dtype=np.float32)

    # -------------------------------------------------------------------------- #
    # Phase 1: Reach above spoon in mug (10cm higher with same orientation)
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        reach_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 2: Descend and grasp spoon
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        grasp_pose,
        other_gripper_state=right_planner.gripper_state
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 3: Lift spoon out of mug
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        lift_pose_tilted,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 3.5: Rotate to vertical orientation (gripper pointing down)
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        lift_pose_vertical,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 4: Move to above bowl (with vertical orientation)
    # -------------------------------------------------------------------------- #
    bowl_center = env.bowl.pose.sp.p
    above_bowl_pose = sapien.Pose(
        bowl_center + np.array([0, 0, 0.30]),  # 15cm above bowl
        vertical_orientation  # Use vertical orientation
    )
    left_planner.move_to_pose_with_screw(
        above_bowl_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 5: Descend into bowl (SKIPPED - go directly to stirring)
    # -------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------- #
    # Phase 6: Stir in bowl - 3 counterclockwise circles (with vertical orientation)
    # -------------------------------------------------------------------------- #
    stir_radius = 0.02  # 4cm radius for stirring circle
    num_circles = 3    
    points_per_circle = 16
    stir_height = bowl_center[2] + 0.21  # 8cm above bowl bottom (inside bowl)
    stir_center_pose = sapien.Pose(
        np.array([bowl_center[0], bowl_center[1], stir_height]),
        vertical_orientation  # Use vertical orientation
    )

    for circle in range(num_circles):
        for i in range(points_per_circle + 1): 
            angle = 2 * np.pi * i / points_per_circle  
            x_offset = stir_radius * np.cos(angle)
            y_offset = stir_radius * np.sin(angle)

            stir_point = sapien.Pose(
                np.array([
                    bowl_center[0] + x_offset,
                    bowl_center[1] + y_offset,
                    stir_height
                ]),
                vertical_orientation  # Use vertical orientation
            )
            left_planner.move_to_pose_with_screw(
                stir_point,
                other_gripper_state=right_planner.gripper_state
            )

    # -------------------------------------------------------------------------- #
    # Phase 7: Return to center and lift out of bowl
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        stir_center_pose,
        other_gripper_state=right_planner.gripper_state
    )
    left_planner.move_to_pose_with_screw(
        above_bowl_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 8: Move back above mug (with vertical orientation)
    # -------------------------------------------------------------------------- #
    mug_center = env.mug.pose.sp.p
    above_mug_pose = sapien.Pose(
        mug_center + np.array([0, 0, 0.45]),  # 15cm above mug
        vertical_orientation  # Use vertical orientation
    )
    left_planner.move_to_pose_with_screw(
        above_mug_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 9: Descend into mug (with vertical orientation)
    # -------------------------------------------------------------------------- #
    # Return spoon to its original XY position in the mug
    return_pose = sapien.Pose(
        np.array([spoon_initial_position[0], spoon_initial_position[1], mug_center[2] + 0.25]),  # Use spoon's original XY
        vertical_orientation  # Use vertical orientation
    )
    left_planner.move_to_pose_with_screw(
        return_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 10: Release spoon
    # -------------------------------------------------------------------------- #
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 11: Retract from mug
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        above_mug_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 12: Return to initial pose
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        left_init_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # Cleanup
    left_planner.close()
    right_planner.close()

    # Return results (left robot didn't perform actions, but we need both results)
    return left_res, left_res
