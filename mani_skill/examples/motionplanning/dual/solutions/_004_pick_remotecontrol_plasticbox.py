import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPickRemoteControlEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


def solve(env: TwoRobotPickRemoteControlEnv, seed=None, debug=False, vis=False):
    """
    Solution for TwoRobotPickRemoteControl-v1 environment.

    Task: Left robot (at position [-0.60, -0.3, 0]) picks up remotecontrol
    and places it into plasticbox.

    Both planners are initialized but only the left robot executes actions.
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

    left_init_pose = env.left_agent.tcp.pose

    # Get object oriented bounding box for remotecontrol
    obb = get_actor_obb(env.remotecontrol)

    # Define grasp approach direction (from above)
    approaching = np.array([0, 0, -1])
    # Get closing direction from tcp pose
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # Compute grasp information
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]

    # Build grasp pose for remotecontrol
    grasp_pose = env.left_agent.build_grasp_pose(
        approaching,
        closing,
        env.remotecontrol.pose.sp.p
    )
    # Lower the grasp pose to be closer to the table for thin objects
    grasp_pose = grasp_pose * sapien.Pose([0, 0, 0.03])

    # Define target pose for placing into plasticbox
    # Place slightly above plasticbox center
    goal_pose = sapien.Pose(
        env.plasticbox.pose.sp.p + np.array([0, 0, 0.05]),  # 5cm above plasticbox
        grasp_pose.q
    )

    # -------------------------------------------------------------------------- #
    # Phase 1: Reach above remotecontrol
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])  # 10cm above grasp position
    left_planner.move_to_pose_with_screw(
        reach_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 2: Descend and grasp remotecontrol
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        grasp_pose,
        other_gripper_state=right_planner.gripper_state
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 3: Lift remotecontrol
    # -------------------------------------------------------------------------- #
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.25])  # Lift 15cm
    left_planner.move_to_pose_with_screw(
        lift_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 4: Move to above plasticbox
    # -------------------------------------------------------------------------- #
    above_box_pose = goal_pose * sapien.Pose([0, 0, -0.06])  # 10cm above goal
    left_planner.move_to_pose_with_screw(
        above_box_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # -------------------------------------------------------------------------- #
    # Phase 5: Descend into plasticbox
    # -------------------------------------------------------------------------- #
    # left_planner.move_to_pose_with_screw(
    #     goal_pose,
    #     other_gripper_state=right_planner.gripper_state
    # )

    # -------------------------------------------------------------------------- #
    # Phase 6: Release remotecontrol
    # -------------------------------------------------------------------------- #
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Phase 7: Retract
    # -------------------------------------------------------------------------- #
    retract_pose = goal_pose * sapien.Pose([0, 0, -0.125])
    left_planner.move_to_pose_with_screw(
        retract_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # Cleanup
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    # Return results (right robot didn't perform actions, but we need both results)
    return left_res, right_res
