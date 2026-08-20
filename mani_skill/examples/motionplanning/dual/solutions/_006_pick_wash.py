import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.dual_tasks._006_pick_wash import TwoRobotPickWashEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


def _move_to_pose_with_rrt_dual(
    planner: PandaArmMotionPlanningSolver,
    pose: sapien.Pose,
    other_gripper_state,
):
    result = planner.move_to_pose_with_RRTConnect(pose, dry_run=True)
    if result == -1:
        return -1
    return planner.follow_path(result, other_gripper_state=other_gripper_state)


def solve(env: TwoRobotPickWashEnv, seed=None, debug=False, vis=False):
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
    left_init_pose = env.left_agent.tcp.pose
    right_state = right_planner.gripper_state

    source_obj = env.bottles[1]
    other_obj_1 = env.bottles[2]
    other_obj_2 = env.bottles[0]

    target_pos = (other_obj_1.pose.p + other_obj_2.pose.p) / 2
    target_pos[0, 2] = source_obj.pose.p[0, 2]

    source_p = np.array(source_obj.pose.sp.p).reshape(-1, 3)[0]
    fixed_q = np.array(left_init_pose.q[0])
    grasp_pose = sapien.Pose(p=source_p, q=fixed_q) * sapien.Pose([0, 0, -0.05])

    goal_center = np.array(target_pos[0, 0:3].cpu())
    goal_pose = sapien.Pose(p=goal_center + np.array([0.0, 0.0, 0.05]), q=grasp_pose.q)

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
    grasp_target_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.4])
    pre_place_pose = goal_pose * sapien.Pose([0, 0, -0.4])
    place_pose = goal_pose * sapien.Pose([0, 0, -0.1])

    left_res = _move_to_pose_with_rrt_dual(left_planner, reach_pose, right_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, grasp_target_pose, right_state)
    if left_res == -1:
        return -1, -1
    left_planner.close_gripper(other_gripper_state=right_state)

    left_res = _move_to_pose_with_rrt_dual(left_planner, lift_pose, right_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, pre_place_pose, right_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, place_pose, right_state)
    if left_res == -1:
        return -1, -1
    left_res = left_planner.open_gripper(other_gripper_state=right_state)

    left_res = _move_to_pose_with_rrt_dual(left_planner, pre_place_pose, right_state)
    if left_res == -1:
        return -1, -1
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    left_res = _move_to_pose_with_rrt_dual(left_planner, left_init_pose, right_state)
    if left_res == -1:
        return -1, -1

    left_planner.close()
    right_planner.close()
    return left_res, right_res
