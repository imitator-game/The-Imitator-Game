import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPlacePlateRackEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def _move_to_pose_with_rrt_dual(
    planner: PandaArmMotionPlanningSolver,
    pose: sapien.Pose,
    other_gripper_state,
):
    result = planner.move_to_pose_with_RRTConnect(pose, dry_run=True)
    if result == -1:
        return -1
    return planner.follow_path(result, other_gripper_state=other_gripper_state)


def _build_plan_context(env: TwoRobotPlacePlateRackEnv):
    finger_length = 0.025
    obb = get_actor_obb(env.plate)
    approaching = np.array([0, 0, -1])
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=finger_length,
    )
    closing = grasp_info["closing"]
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.plate.pose.sp.p)
    goal_pose = sapien.Pose(env.rack.pose.sp.p, grasp_pose.q)

    offset_from_rack = -0.05
    rot_angle = np.pi * 14.0 / 32.0
    reach_pose1 = grasp_pose * sapien.Pose(
        [0.0, 0.18, -0.15], euler2quat(np.pi / 2, 0, 0)
    )
    reach_pose2 = grasp_pose * sapien.Pose(
        [0.0, 0.18, 0.0], euler2quat(np.pi / 2, 0, 0)
    )
    grasp_touch_pose = grasp_pose * sapien.Pose(
        [0.0, 0.05, -0.02], euler2quat(np.pi / 2, 0, 0)
    )
    reach_pose3 = grasp_pose * sapien.Pose(
        [0.0, 0.15, -0.2], euler2quat(np.pi / 2, 0, 0)
    )

    context = {
        "left_init_pose": env.left_agent.tcp.pose,
        "right_init_pose": env.right_agent.tcp.pose,
        "grasp_pose": grasp_pose,
        "goal_pose": goal_pose,
        "reach_pose1": reach_pose1,
        "reach_pose2": reach_pose2,
        "grasp_touch_pose": grasp_touch_pose,
        "reach_pose3": reach_pose3,
        "reach_pose4": goal_pose * sapien.Pose(
            [offset_from_rack, 0.05, -0.55], euler2quat(rot_angle, np.pi / 2, 0)
        ),
        "place_pose": goal_pose * sapien.Pose(
            [offset_from_rack, 0.05, -0.45], euler2quat(rot_angle, np.pi / 2, 0)
        ),
        "reach_pose5": goal_pose * sapien.Pose(
            [offset_from_rack - 0.042, 0.15, -0.4], euler2quat(rot_angle, np.pi / 2, 0)
        ),
        "reach_pose6": goal_pose * sapien.Pose(
            [offset_from_rack - 0.042, 0.15, -0.5], euler2quat(rot_angle, np.pi / 2, 0)
        ),
    }
    return context


def _solve_l0_l1(
    env: TwoRobotPlacePlateRackEnv,
    left_planner: PandaArmMotionPlanningSolver,
    right_planner: PandaArmMotionPlanningSolver,
):
    ctx = _build_plan_context(env)

    left_planner.move_to_pose_with_screw(ctx["reach_pose1"], other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(ctx["reach_pose2"], other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(
        ctx["grasp_touch_pose"],
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    left_res = _move_to_pose_with_rrt_dual(left_planner, ctx["reach_pose3"], right_planner.gripper_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, ctx["reach_pose4"], right_planner.gripper_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, ctx["place_pose"], right_planner.gripper_state)
    if left_res == -1:
        return -1, -1
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    left_res = _move_to_pose_with_rrt_dual(left_planner, ctx["reach_pose5"], right_planner.gripper_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, ctx["reach_pose6"], right_planner.gripper_state)
    if left_res == -1:
        return -1, -1

    left_res = _move_to_pose_with_rrt_dual(
        left_planner, ctx["left_init_pose"], right_planner.gripper_state
    )
    if left_res == -1:
        return -1, -1
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    return left_res, right_res


def solve(env: TwoRobotPlacePlateRackEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    left_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[0].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=0,
    )
    right_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[1].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=1,
    )

    env = env.unwrapped
    return _solve_l0_l1(env, left_planner, right_planner)
