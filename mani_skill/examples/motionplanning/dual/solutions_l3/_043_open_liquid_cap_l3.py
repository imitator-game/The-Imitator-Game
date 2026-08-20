import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotOpenLiquidCapEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: TwoRobotOpenLiquidCapEnvL3, seed=None, debug=False, vis=False):
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

    obb = get_actor_obb(env.cap)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose
    # Only use OBB for grasp height (z); keep x/y from current object pose.
    cap_p = np.array(env.cap.pose.sp.p, dtype=np.float32)
    grasp_center = np.array([cap_p[0], cap_p[1], center[2]], dtype=np.float32)
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, grasp_center)
    # Side-grasp pattern (same style as _039_pour_liquid_mug_l3).
    side_grasp_q = euler2quat(np.pi / 2, np.pi / 2, 0)
    reach_pose1 = grasp_pose * sapien.Pose([0.00, 0.07, 0.03], side_grasp_q)
    grasp_pre_pose = grasp_pose * sapien.Pose([0.00, 0.0, 0.03], side_grasp_q)
    # Keep current grasp orientation, lift only along world +Z after close.
    lift_delta_z = 0.12
    grasp_pre_p = np.array(grasp_pre_pose.p, dtype=np.float32)
    lift_pose = sapien.Pose(
        [grasp_pre_p[0], grasp_pre_p[1], grasp_pre_p[2] + lift_delta_z],
        grasp_pre_pose.q,
    )

    # Place on plate (world-frame target poses).
    plate_p = np.array(env.plate.pose.sp.p, dtype=np.float32)
    place_xy_offset = np.array([0.0, 0.0], dtype=np.float32)
    place_above_z = 0.22
    place_release_z = 0.12
    place_above_pose = sapien.Pose(
        [plate_p[0] + place_xy_offset[0], plate_p[1] + place_xy_offset[1], plate_p[2] + place_above_z],
        lift_pose.q,
    )
    place_pose = sapien.Pose(
        [plate_p[0] + place_xy_offset[0], plate_p[1] + place_xy_offset[1], plate_p[2] + place_release_z],
        lift_pose.q,
    )

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(grasp_pre_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(lift_pose, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(place_above_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(place_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(place_above_pose, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
