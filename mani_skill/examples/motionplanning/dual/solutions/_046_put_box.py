import numpy as np
import sapien
from mani_skill.envs.tasks.tabletop.dual_tasks._046_put_box import TwoRobotPutBoxEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


def solve(env: TwoRobotPutBoxEnv, seed=None, debug=False, vis=False):
    """
    Solution for TwoRobotPutBox-v1 (Modified: Single Ball, Single Box).
    
    Constraint: Only LEFT ROBOT moves. RIGHT ROBOT stays static.
    
    Sequence:
    1. Left Robot picks Tennis Ball.
    2. Left Robot places Ball into Box1.
    3. Left Robot closes Box1 Lid.
    """
    env.reset(seed=seed)

    # -------------------------------------------------------------------------- #
    # 1. Initialize Planners
    # -------------------------------------------------------------------------- #
    # Left Robot (Active)
    left_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[0].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=0
    )
    # Right Robot (Passive/Static) - We init it just to pass its state to collision checker
    right_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[1].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=1
    )

    FINGER_LENGTH = 0.05
    env = env.unwrapped
    
    # Define objects
    food = env.food
    target_box = env.box1

    left_init_pose = env.left_agent.tcp.pose

    # ========================================================================== #
    # STEP 1: Pick Tennis Ball (Left Arm)
    # ========================================================================== #
    
    # 1.1 Calculate Grasp Pose
    tennis_obb = get_actor_obb(food)
    
    # Approach from top (Z axis is -1)
    approaching = np.array([0, 0, -1])
    # Closing direction based on current end-effector orientation
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    tennis_grasp_info = compute_grasp_info_by_obb(
        tennis_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )

    # Build grasp pose
    tennis_grasp_pose = env.left_agent.build_grasp_pose(
        approaching,
        tennis_grasp_info["closing"],
        food.pose.sp.p
    )
    # Adjustment: Lower slightly to ensure firm grip on sphere
    tennis_grasp_pose = tennis_grasp_pose * sapien.Pose([0, 0, 0.02])
    
    # Pre-grasp pose (Hover 10cm above)
    tennis_reach_pose = tennis_grasp_pose * sapien.Pose([0, 0, -0.1])

    # 1.2 Execute Pick
    # Note: We always pass right_planner.gripper_state to ensure we don't hit the static arm
    
    # Move to Pre-grasp
    left_planner.move_to_pose_with_screw(
        tennis_reach_pose, other_gripper_state=right_planner.gripper_state
    )

    # Move to Grasp
    left_planner.move_to_pose_with_screw(
        tennis_grasp_pose, other_gripper_state=right_planner.gripper_state
    )
    
    # Close Gripper
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # Lift Up
    tennis_lift_pose = tennis_grasp_pose * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        tennis_lift_pose, other_gripper_state=right_planner.gripper_state
    )

    # ========================================================================== #
    # STEP 2: Place into Box1 (Left Arm)
    # ========================================================================== #
    
    # 2.1 Calculate Place Pose
    box_pose = target_box.pose.sp
    # Offset: Center of box + 12cm height
    # Note: Applying offset in World Frame assuming box is roughly flat
    # If using local frame: box_pose * sapien.Pose([0, 0, 0.12])
    place_position = box_pose.p + np.array([0, 0, 0.15]) 
    
    place_pose = sapien.Pose(
        place_position,
        tennis_grasp_pose.q # Keep same rotation as grasp
    )

    # 2.2 Execute Place
    # Move to Place location
    left_planner.move_to_pose_with_screw(
        place_pose, other_gripper_state=right_planner.gripper_state
    )

    # Open Gripper
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    # Retract Up (Safety)
    retract_pose = place_pose * sapien.Pose([0, 0, -0.05])
    left_planner.move_to_pose_with_screw(
        retract_pose, other_gripper_state=right_planner.gripper_state
    )

    # ========================================================================== #
    # STEP 3: Close Box Lid (Left Arm)
    # ========================================================================== #
    # Strategy: 
    # 1. Move high above box
    # 2. Move to the "Back" of the box (assuming lid hinges at front or we push from back)
    #    Based on previous code: Push from X+0.15 to X-0.15
    # 3. Lower down
    # 4. Push forward
    
    # 3.1 Prepare for Pushing
    # Raise hand high (25cm above box)
    high_pose = sapien.Pose(
        p=box_pose.p + np.array([0, 0, 0.25]),
        q=tennis_grasp_pose.q
    )
    left_planner.move_to_pose_with_screw(
        high_pose, other_gripper_state=right_planner.gripper_state
    )
    
    # Make a fist (Close gripper) to push effectively
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # 3.2 Move to Start Position (Behind Box)
    # Offset: 15cm behind (X direction positive) relative to box center
    push_start_pos = box_pose.p + np.array([0.15, 0, 0.12]) # Height 12cm
    push_start_pose = sapien.Pose(
        p=push_start_pos,
        q=tennis_grasp_pose.q
    )
    
    # Go to high position at start X,Y first to avoid hitting lid
    push_start_high = sapien.Pose(
        p=box_pose.p + np.array([0.15, 0, 0.25]),
        q=tennis_grasp_pose.q
    )
    left_planner.move_to_pose_with_screw(
        push_start_high, other_gripper_state=right_planner.gripper_state
    )
    
    # Lower down to pushing height
    left_planner.move_to_pose_with_screw(
        push_start_pose, other_gripper_state=right_planner.gripper_state
    )

    # 3.3 Execute Push
    # Push forward (along -X axis) by 20cm (from +0.15 to -0.05)
    # This assumes the robot is at -X, and we are pushing towards the robot
    push_end_pose = sapien.Pose(
        p=push_start_pose.p + np.array([-0.20, 0, 0]), 
        q=push_start_pose.q
    )
    
    left_planner.move_to_pose_with_screw(
        push_end_pose, other_gripper_state=right_planner.gripper_state
    )

    # 3.4 Retract
    # Move back up/away
    left_planner.move_to_pose_with_screw(
        push_start_high, other_gripper_state=right_planner.gripper_state
    )

    # Return results (Right arm result is dummy/None essentially)
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    # Return to initial pose
    left_planner.move_to_pose_with_screw(
        left_init_pose, other_gripper_state=right_planner.gripper_state
    )

    left_planner.close()
    right_planner.close()

    return left_res, right_res