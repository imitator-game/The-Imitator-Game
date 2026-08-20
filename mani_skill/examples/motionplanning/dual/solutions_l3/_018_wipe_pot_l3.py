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
    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose

    # -----------------------------
    # 0.1) Build grasp pose from contact_points_pose[0]
    # -----------------------------
    obb = get_actor_obb(env.brush, to_world_frame=True, vis=False)

    approaching = np.array([0.0, 0.0, -1.0])

    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb=obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
        ortho=True,
    )

    grasp_pose = env.left_agent.build_grasp_pose(
        approaching=grasp_info["approaching"],
        closing=grasp_info["closing"],
        center=grasp_info["center"],
    )*sapien.Pose([-0.01, 0, 0])

    goal_pose = (env.left_agent.build_grasp_pose(grasp_info["approaching"], grasp_info["closing"], env.pot.pose.sp.p) *
                 sapien.Pose(q=euler2quat(np.pi / 4, 0.0, 0.0)))

    # -----------------------------
    # 1) Reach (pre-grasp approach)
    # -----------------------------
    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.08])
    left_planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=right_planner.gripper_state)

    # -----------------------------
    # 2) Grasp
    # -----------------------------
    left_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.08])
    left_planner.move_to_pose_with_screw(lift_pose, other_gripper_state=right_planner.gripper_state)

    # -----------------------------
    # 3) Wipe
    # -----------------------------
    RADIUS = 0.12
    pot_pos = env.pot.pose.sp.p
    point_a_offset_x = -0.04
    point_a_offset_y = -0.09
    point_b_offset_x = 0.04
    point_b_offset_y = -0.09
    wipe_pose_1 = sapien.Pose(pot_pos + np.array([point_a_offset_x, point_a_offset_y, 0.1]), goal_pose.q)
    wipe_pose_2 = sapien.Pose(pot_pos + np.array([point_b_offset_x, point_b_offset_y, 0.1]), goal_pose.q)

    for i in range(3):
        left_planner.move_to_pose_with_screw(wipe_pose_1, other_gripper_state=right_planner.gripper_state)
        left_planner.move_to_pose_with_screw(wipe_pose_2, other_gripper_state=right_planner.gripper_state)

    # -----------------------------
    # 4) Back to initial placement region, put down brush
    # -----------------------------
    back_above_place = pre_grasp_pose
    left_planner.move_to_pose_with_screw(back_above_place, other_gripper_state=right_planner.gripper_state)

    place_pose = grasp_pose
    left_planner.move_to_pose_with_screw(place_pose, other_gripper_state=right_planner.gripper_state)

    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=right_planner.gripper_state)

    # -----------------------------
    # 5) Arm back to init pose
    # -----------------------------
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
