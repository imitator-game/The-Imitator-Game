import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPickCubeYCBEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: TwoRobotPickCubeYCBEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    left_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[0].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=0
    )
    right_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[1].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=1
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # -----------------------------
    # 0) Cache init poses (both arms)
    # -----------------------------
    left_init_pose = env.left_agent.tcp.pose  # batched Pose
    right_init_pose = env.right_agent.tcp.pose

    # -----------------------------
    # 0.1) Build grasp pose from contact_points_pose[0]
    # -----------------------------
    obb_left = get_actor_obb(env._fruits_ycb_objs[0], to_world_frame=True, vis=False)
    obb_right = get_actor_obb(env._fruits_ycb_objs[1], to_world_frame=True, vis=False)

    approaching = np.array([0.0, 0.0, -1.0])  # world -Z

    target_closing_left = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_closing_right = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info_left = compute_grasp_info_by_obb(
        obb=obb_left,
        approaching=approaching,
        target_closing=target_closing_left,
        depth=FINGER_LENGTH,
        ortho=True,
    )

    grasp_info_right = compute_grasp_info_by_obb(
        obb=obb_right,
        approaching=approaching,
        target_closing=target_closing_right,
        depth=FINGER_LENGTH,
        ortho=True,
    )

    grasp_pose_left = env.left_agent.build_grasp_pose(
        approaching=grasp_info_left["approaching"],
        closing=grasp_info_left["closing"],
        center=grasp_info_left["center"],
    )

    grasp_pose_right = env.left_agent.build_grasp_pose(
        approaching=grasp_info_right["approaching"],
        closing=grasp_info_right["closing"],
        center=grasp_info_right["center"],
    )

    goal_pose_left = env.left_agent.build_grasp_pose(grasp_info_left["approaching"], grasp_info_left["closing"], env.bin.pose.sp.p)
    goal_pose_right = env.right_agent.build_grasp_pose(grasp_info_right["approaching"], grasp_info_right["closing"], env.bin.pose.sp.p)

    # -----------------------------
    # 1) Left Arm Reach (pre-grasp approach)
    # -----------------------------
    pre_grasp_pose_left = grasp_pose_left * sapien.Pose([0, 0, -0.08])
    left_planner.move_to_pose_with_screw(pre_grasp_pose_left, other_gripper_state=right_planner.gripper_state)

    # -----------------------------
    # 2) Left Arm Grasp
    # -----------------------------
    left_planner.move_to_pose_with_screw(grasp_pose_left, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    lift_pose_left = grasp_pose_left * sapien.Pose([0, 0, -0.20])
    left_planner.move_to_pose_with_screw(lift_pose_left, other_gripper_state=right_planner.gripper_state)

    # -----------------------------
    # 3) Left Arm Reach goal region
    # -----------------------------
    ABOVE_BIN = 0.20

    reach_goal_pose_left = goal_pose_left * sapien.Pose([0, 0, -ABOVE_BIN])
    left_planner.move_to_pose_with_screw(reach_goal_pose_left, other_gripper_state=right_planner.gripper_state)

    reach_goal_pose_down_left = reach_goal_pose_left * sapien.Pose([0, 0, 0.075])
    left_planner.move_to_pose_with_screw(reach_goal_pose_down_left, other_gripper_state=right_planner.gripper_state)

    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    left_planner.move_to_pose_with_screw(reach_goal_pose_left, other_gripper_state=right_planner.gripper_state)

    left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)

    # -----------------------------
    # 4) Right Arm Reach (pre-grasp approach)
    # -----------------------------
    pre_grasp_pose_right= grasp_pose_right * sapien.Pose([0, 0, -0.08])
    right_planner.move_to_pose_with_screw(pre_grasp_pose_right, other_gripper_state=left_planner.gripper_state)

    # -----------------------------
    # 5) Right Arm Grasp
    # -----------------------------
    right_planner.move_to_pose_with_screw(grasp_pose_right, other_gripper_state=left_planner.gripper_state)
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)

    lift_pose_right = grasp_pose_right * sapien.Pose([0, 0, -0.20])
    right_planner.move_to_pose_with_screw(lift_pose_right, other_gripper_state=left_planner.gripper_state)

    # -----------------------------
    # 6) Left Arm Reach goal region
    # -----------------------------
    ABOVE_BIN = 0.2
    reach_goal_pose_right = goal_pose_right * sapien.Pose([0, 0, -ABOVE_BIN])

    right_planner.move_to_pose_with_screw(reach_goal_pose_right, other_gripper_state=left_planner.gripper_state)

    reach_goal_pose_down_right = reach_goal_pose_right * sapien.Pose([0, 0, 0.075])
    right_planner.move_to_pose_with_screw(reach_goal_pose_down_right, other_gripper_state=left_planner.gripper_state)

    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    right_planner.move_to_pose_with_screw(reach_goal_pose_right, other_gripper_state=left_planner.gripper_state)

    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.move_to_pose_with_screw(right_init_pose, other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
