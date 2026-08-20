import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.dual_tasks._026_pick_tennisball_golfball import TwoRobotPickTennisBallGolfBallEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


def solve(env: TwoRobotPickTennisBallGolfBallEnv, seed=None, debug=False, vis=False):
    """
    Solution for TwoRobotPickTennisBallGolfBall-v1 environment.

    Sequential execution (NO parallel):

    STEP 1 - LEFT ROBOT completes FULL sequence:
      → Pick tennis ball
      → Place into box1
      → IMMEDIATELY close box1 lid0

    STEP 2 - RIGHT ROBOT completes FULL sequence:
      → Pick golf ball
      → Place into box2
      → IMMEDIATELY close box2 lid0
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
        multi_robot_id=0  # Left robot at [-0.60, -0.3, 0]
    )
    right_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[1].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=1  # Right robot at [-0.60, 0.3, 0]
    )

    FINGER_LENGTH = 0.05
    env = env.unwrapped

    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose

    # ========================================================================== #
    # STEP 1: LEFT ROBOT - Pick Tennis Ball and Place in Box1
    # ========================================================================== #

    # Get tennis ball oriented bounding box
    tennis_obb = get_actor_obb(env.tennis_ball)

    # Define grasp approach (from above)
    approaching = np.array([0, 0, -1])
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # Compute grasp information
    tennis_grasp_info = compute_grasp_info_by_obb(
        tennis_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )

    # Build grasp pose for tennis ball
    tennis_grasp_pose = env.left_agent.build_grasp_pose(
        approaching,
        tennis_grasp_info["closing"],
        env.tennis_ball.pose.sp.p
    )
    # Adjust for sphere - grasp slightly lower
    tennis_grasp_pose = tennis_grasp_pose * sapien.Pose([0, 0, 0.01])

    # Phase 1: Reach above tennis ball
    tennis_reach_pose = tennis_grasp_pose * sapien.Pose([0, 0, -0.1])
    left_planner.move_to_pose_with_screw(
        tennis_reach_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # Phase 2: Grasp tennis ball
    left_planner.move_to_pose_with_screw(
        tennis_grasp_pose,
        other_gripper_state=right_planner.gripper_state
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # Phase 3: Lift tennis ball
    tennis_lift_pose = tennis_grasp_pose * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        tennis_lift_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # Phase 4: Move above box1 and place
    # Place ball in the open half (lid0 side) of the box
    # Use box's local coordinate system to add offset along its Y-axis
    box1_pose = env.box1.pose.sp
    # Offset in box's local coordinates
    local_offset1 = sapien.Pose(p=[0, 0, 0.12], q=[1, 0, 0, 0])
    place_position1 = (box1_pose * local_offset1).p  # Transform to world coordinates
    place_pose1 = sapien.Pose(
        place_position1,
        tennis_grasp_pose.q
    )
    left_planner.move_to_pose_with_screw(
        place_pose1,
        other_gripper_state=right_planner.gripper_state
    )

    # Phase 5: Release tennis ball
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)


    # ========================================================================== #
    # STEP 1 (continued): LEFT ROBOT - IMMEDIATELY Close Box1 Lid0
    # ========================================================================== #
    # After placing tennis ball, LEFT robot immediately closes the lid
    # Strategy: Keep gripper orientation → Move high → Move to behind box → Push forward 5cm

    # Step 1: Raise hand high (keep same orientation as placing)
    high_pose1 = sapien.Pose(
        p=box1_pose.p + np.array([0, 0, 0.25]),  # 25cm above box center
        q=tennis_grasp_pose.q  # Keep same gripper orientation
    )
    left_planner.move_to_pose_with_screw(
        high_pose1,
        other_gripper_state=right_planner.gripper_state
    )

    # Step 2: Close gripper to form a fist for pushing
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # Step 3a: Move horizontally to behind box (maintain high altitude)
    behind_box1_high = sapien.Pose(
        p=box1_pose.p + np.array([0.15, 0, 0.25]),  # 15cm behind, maintain 25cm height
        q=tennis_grasp_pose.q  # Keep same gripper orientation
    )
    left_planner.move_to_pose_with_screw(
        behind_box1_high,
        other_gripper_state=right_planner.gripper_state
    )

    # Step 3b: Descend to working height
    behind_box1_pose = sapien.Pose(
        p=box1_pose.p + np.array([0.15, 0, 0.12]),  # Descend to 10cm height
        q=tennis_grasp_pose.q  # Keep same gripper orientation
    )
    left_planner.move_to_pose_with_screw(
        behind_box1_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # Step 4: Push forward 5cm to close lid
    push_forward_pose1 = sapien.Pose(
        p=behind_box1_pose.p + np.array([-0.15, 0, 0]),  # Move forward 5cm along X
        q=behind_box1_pose.q
    )
    left_planner.move_to_pose_with_screw(
        push_forward_pose1,
        other_gripper_state=right_planner.gripper_state
    )

    # Step 5: Retract back
    left_planner.move_to_pose_with_screw(
        left_init_pose,
        other_gripper_state=right_planner.gripper_state
    )

    # ========================================================================== #
    # RIGHT ROBOT: Pick Golf Ball and Place in Box2
    # ========================================================================== #

    # Get golf ball oriented bounding box
    golf_obb = get_actor_obb(env.golf_ball)

    # Define grasp approach (from above)
    right_target_closing = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    # Compute grasp information
    golf_grasp_info = compute_grasp_info_by_obb(
        golf_obb,
        approaching=approaching,
        target_closing=right_target_closing,
        depth=FINGER_LENGTH,
    )

    # Build grasp pose for golf ball
    golf_grasp_pose = env.right_agent.build_grasp_pose(
        approaching,
        golf_grasp_info["closing"],
        env.golf_ball.pose.sp.p
    )
    # Adjust for small sphere
    golf_grasp_pose = golf_grasp_pose * sapien.Pose([0, 0, 0.005])

    # Phase 1: Reach above golf ball
    golf_reach_pose = golf_grasp_pose * sapien.Pose([0, 0, -0.1])
    right_planner.move_to_pose_with_screw(
        golf_reach_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Phase 2: Grasp golf ball
    right_planner.move_to_pose_with_screw(
        golf_grasp_pose,
        other_gripper_state=left_planner.gripper_state
    )
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)

    # Phase 3: Lift golf ball
    golf_lift_pose = golf_grasp_pose * sapien.Pose([0, 0, -0.12])
    right_planner.move_to_pose_with_screw(
        golf_lift_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Phase 4: Move above box2 and place
    # Place ball in the open half (lid0 side) of the box
    # Use box's local coordinate system to add offset along its Y-axis
    box2_pose = env.box2.pose.sp
    # Offset in box's local coordinates
    local_offset2 = sapien.Pose(p=[0, 0, 0.12], q=[1, 0, 0, 0])
    place_position2 = (box2_pose * local_offset2).p  # Transform to world coordinates
    place_pose2 = sapien.Pose(
        place_position2,
        golf_grasp_pose.q
    )
    right_planner.move_to_pose_with_screw(
        place_pose2,
        other_gripper_state=left_planner.gripper_state
    )

    # Phase 5: Release golf ball
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    # ========================================================================== #
    # STEP 2 (continued): RIGHT ROBOT - IMMEDIATELY Close Box2 Lid0
    # ========================================================================== #
    # After placing golf ball, RIGHT robot immediately closes the lid
    # Strategy: Keep gripper orientation → Move high → Move to behind box → Push forward 5cm

    # Step 1: Raise hand high (keep same orientation as placing)
    high_pose2 = sapien.Pose(
        p=box2_pose.p + np.array([0, 0, 0.25]),  # 25cm above box center
        q=golf_grasp_pose.q  # Keep same gripper orientation
    )
    right_planner.move_to_pose_with_screw(
        high_pose2,
        other_gripper_state=left_planner.gripper_state
    )

    # Step 2: Close gripper to form a fist for pushing
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)

    # Step 3a: Move horizontally to behind box (maintain high altitude)
    behind_box2_high = sapien.Pose(
        p=box2_pose.p + np.array([0.15, 0, 0.25]),  # 15cm behind, maintain 25cm height
        q=golf_grasp_pose.q  # Keep same gripper orientation
    )
    right_planner.move_to_pose_with_screw(
        behind_box2_high,
        other_gripper_state=left_planner.gripper_state
    )

    # Step 3b: Descend to working height
    behind_box2_pose = sapien.Pose(
        p=box2_pose.p + np.array([0.15, 0, 0.12]),  # Descend to 10cm height
        q=golf_grasp_pose.q  # Keep same gripper orientation
    )
    right_planner.move_to_pose_with_screw(
        behind_box2_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Step 4: Push forward 5cm to close lid
    push_forward_pose2 = sapien.Pose(
        p=behind_box2_pose.p + np.array([-0.15, 0, 0]),  # Move forward 5cm along X
        q=behind_box2_pose.q
    )
    right_planner.move_to_pose_with_screw(
        push_forward_pose2,
        other_gripper_state=left_planner.gripper_state
    )

    # Step 5: Retract back
    right_planner.move_to_pose_with_screw(
        right_init_pose,
        other_gripper_state=left_planner.gripper_state
    )

    # Final open to log result tensors for both arms
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    # Cleanup
    left_planner.close()
    right_planner.close()

    return left_res, right_res
