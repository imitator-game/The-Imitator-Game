import numpy as np
import sapien
from transforms3d.euler import euler2quat
import time

from mani_skill.envs.tasks import TwoRobotPlaceChipsRackEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver


def solve(env: TwoRobotPlaceChipsRackEnv, seed=None, debug=False, vis=False):
    """
    Solution for TwoRobotPlaceChipsRack-v1 environment.

    Task: Left robot picks up chips tub and places it on bottom display stand.

    Phases:
    1. Pick up chips tub
    2. Move to above bottom display stand
    3. Lower and place chips tub on bottom display stand
    4. Release chips tub
    5. Retract upward
    6. Return to initial pose
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

    approaching = np.array([0, 0, -1])

    # Extract current TCP closing direction and make it orthogonal to approaching
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # Gram-Schmidt orthogonalization: remove the component parallel to approaching
    target_closing = target_closing - (target_closing @ approaching) * approaching
    target_closing = target_closing / np.linalg.norm(target_closing)  # Normalize

    # Build grasp pose using orthogonal closing direction
    grasp_pose = env.left_agent.build_grasp_pose(approaching, target_closing, env.chipstub.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Phase 1: Left robot picks up chips tub
    # -------------------------------------------------------------------------- #
    prep_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        prep_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # # Reach above chips tub
    reach_pose1 = grasp_pose * sapien.Pose([0, 0, -0.1])
    left_planner.move_to_pose_with_screw(
        reach_pose1,
        other_gripper_state=right_planner.gripper_state
    )

    # Descend and grasp chips tub
    left_planner.move_to_pose_with_screw(
        grasp_pose * sapien.Pose([0, 0, -0.06]),
        other_gripper_state=right_planner.gripper_state
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # Lift chips tub
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        lift_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 2: Move to above bottom display stand
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.displaystand_bottom.pose.sp.p, grasp_pose.q) * sapien.Pose([-0.1, 0, 0])

    # Move to above displaystand
    reach_pose2 = goal_pose * sapien.Pose([0, 0, -0.3])
    left_planner.move_to_pose_with_screw(
        reach_pose2,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 3: Lower and place chips tub on bottom display stand
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        goal_pose * sapien.Pose([0, 0, -0.16]),
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 4: Release chips tub
    # -------------------------------------------------------------------------- #
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 5: Retract upward
    # -------------------------------------------------------------------------- #
    retract_pose = goal_pose * sapien.Pose([0, 0, -0.3])
    left_planner.move_to_pose_with_screw(
        retract_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 6: Return to initial poses
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        left_init_pose,
        other_gripper_state=right_planner.gripper_state
    )

    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    # Cleanup
    left_planner.close()
    right_planner.close()

    return left_res, right_res
