import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.dual_tasks._011_place_commodity_rack import TwoRobotPlaceCommodityRackEnv
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import is_l2_enabled, is_lr_mirror_enabled
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver


def solve(env: TwoRobotPlaceCommodityRackEnv, seed=None, debug=False, vis=False):
    """
    Solution for TwoRobotPlaceCommodityRack-v1 environment.

    Sequential execution:
    STEP 1 - LEFT ROBOT:
      → Pick jam jar
      → Place on display stand

    STEP 2 - RIGHT ROBOT:
      → Pick milk box
      → Place on display stand
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

    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose
    swap_goal_targets = is_l2_enabled() and is_lr_mirror_enabled()
    left_goal_offset = np.array([-0.1, 0.075, 0.0], dtype=np.float32)
    right_goal_offset = np.array([-0.1, -0.075, 0.0], dtype=np.float32)
    if swap_goal_targets:
        left_goal_offset, right_goal_offset = right_goal_offset, left_goal_offset

    # STEP 1: LEFT ROBOT - Pick Jam Jar and Place on Display Stand

    # -------------------------------------------------------------------------- #
    # Phase 1: Define grasp pose with fixed orientation (no OBB, no rotation)
    # -------------------------------------------------------------------------- #
    approaching = np.array([0, 0, -1])  # From above (downward)

    # Extract current TCP closing direction and make it orthogonal to approaching
    left_target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # Gram-Schmidt orthogonalization: remove the component parallel to approaching
    left_target_closing = left_target_closing - (left_target_closing @ approaching) * approaching
    left_target_closing = left_target_closing / np.linalg.norm(left_target_closing)  # Normalize

    # Build grasp pose using orthogonal closing direction
    jamjar_grasp_pose = env.left_agent.build_grasp_pose(approaching, left_target_closing, env.jamjar.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Phase 2: Left robot picks up jam jar
    # -------------------------------------------------------------------------- #
    jamjar_prep_pose = jamjar_grasp_pose * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        jamjar_prep_pose,
        other_gripper_state=right_planner.gripper_state
    )

    left_planner.move_to_pose_with_screw(
        jamjar_grasp_pose * sapien.Pose([0, 0, -0.02]),
        other_gripper_state=right_planner.gripper_state
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    jamjar_lift_pose = jamjar_grasp_pose * sapien.Pose([0, 0, -0.18])
    left_planner.move_to_pose_with_screw(
        jamjar_lift_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 3: Move to above display stand
    # -------------------------------------------------------------------------- #
    jamjar_goal_pose = sapien.Pose(env.displaystand_bottom.pose.sp.p, jamjar_grasp_pose.q) * sapien.Pose(left_goal_offset)

    # Move to above displaystand
    jamjar_reach_pose = jamjar_goal_pose * sapien.Pose([0, 0, -0.3])
    left_planner.move_to_pose_with_screw(
        jamjar_reach_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 4: Lower and place jam jar on display stand
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        jamjar_goal_pose * sapien.Pose([0, 0, -0.14]),
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 5: Release jam jar
    # -------------------------------------------------------------------------- #
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 6: Retract upward
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        jamjar_reach_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 7: Left robot returns to initial pose
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(
        left_init_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # STEP 2: RIGHT ROBOT - Pick Milk Box and Place on Display Stand

    # -------------------------------------------------------------------------- #
    # Phase 1: Define grasp pose with fixed orientation (no OBB, no rotation)
    # -------------------------------------------------------------------------- #
    right_target_closing = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # Gram-Schmidt orthogonalization: remove the component parallel to approaching
    right_target_closing = right_target_closing - (right_target_closing @ approaching) * approaching
    right_target_closing = right_target_closing / np.linalg.norm(right_target_closing)  # Normalize

    # Build grasp pose using orthogonal closing direction
    milkbox_grasp_pose = env.right_agent.build_grasp_pose(approaching, right_target_closing, env.milkbox.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Phase 2: Right robot picks up milk box
    # -------------------------------------------------------------------------- #
    # Preparatory pose - high above milk box
    milkbox_prep_pose = milkbox_grasp_pose * sapien.Pose([0, 0, -0.15])
    right_planner.move_to_pose_with_screw(
        milkbox_prep_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Descend and grasp milk box
    right_planner.move_to_pose_with_screw(
        milkbox_grasp_pose * sapien.Pose([0, 0, -0.06]),
        other_gripper_state=left_planner.gripper_state
    )
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)

    # Lift milk box
    milkbox_lift_pose = milkbox_grasp_pose * sapien.Pose([0, 0, -0.15])
    right_planner.move_to_pose_with_screw(
        milkbox_lift_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 3: Move to above display stand
    # -------------------------------------------------------------------------- #
    milkbox_goal_pose = sapien.Pose(env.displaystand_bottom.pose.sp.p, milkbox_grasp_pose.q) * sapien.Pose(right_goal_offset)

    # Move to above displaystand
    milkbox_reach_pose = milkbox_goal_pose * sapien.Pose([0, 0, -0.3])
    right_planner.move_to_pose_with_screw(
        milkbox_reach_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 4: Lower and place milk box on display stand
    # -------------------------------------------------------------------------- #
    right_planner.move_to_pose_with_screw(
        milkbox_goal_pose * sapien.Pose([0, 0, -0.14]),
        other_gripper_state=left_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 5: Release milk box
    # -------------------------------------------------------------------------- #
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 6: Retract upward
    # -------------------------------------------------------------------------- #
    right_planner.move_to_pose_with_screw(
        milkbox_reach_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # ========================================================================== #
    # Final: Right robot returns to initial pose
    # ========================================================================== #
    right_res = right_planner.move_to_pose_with_screw(
        right_init_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Cleanup
    left_planner.close()
    right_planner.close()

    return left_res, right_res
