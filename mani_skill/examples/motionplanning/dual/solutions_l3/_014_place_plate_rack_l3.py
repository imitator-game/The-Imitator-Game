import numpy as np
import sapien

from mani_skill.envs.tasks import TwoRobotPlacePlateRackEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb


def _move_to_pose_with_rrt_dual(
    planner: PandaArmMotionPlanningSolver,
    pose: sapien.Pose,
    other_gripper_state,
):
    result = planner.move_to_pose_with_RRTConnect(pose, dry_run=True)
    if result == -1:
        return -1
    return planner.follow_path(result, other_gripper_state=other_gripper_state)


def solve(env: TwoRobotPlacePlateRackEnvL3, seed=None, debug=False, vis=False):
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
    right_init_pose = env.right_agent.tcp.pose

    obb = get_actor_obb(env.bowl)
    approaching = np.array([0, 0, -1])
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.025,
    )
    closing = grasp_info["closing"]
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.bowl.pose.sp.p)
    grasp_pose = grasp_pose * sapien.Pose(p=[-0.03, -0.03, -0.04])

    pre_grasp = grasp_pose * sapien.Pose([0, 0, -0.12])
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.18])
    goal_pose = sapien.Pose(env.plate.pose.sp.p, grasp_pose.q) * sapien.Pose([0.0, -0.04, -0.08])
    pre_place = goal_pose * sapien.Pose([0, 0, -0.12])

    left_res = _move_to_pose_with_rrt_dual(left_planner, pre_grasp, right_planner.gripper_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, grasp_pose, right_planner.gripper_state)
    if left_res == -1:
        return -1, -1
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_res = _move_to_pose_with_rrt_dual(left_planner, lift_pose, right_planner.gripper_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, pre_place, right_planner.gripper_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, goal_pose, right_planner.gripper_state)
    if left_res == -1:
        return -1, -1
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_res = _move_to_pose_with_rrt_dual(left_planner, pre_place, right_planner.gripper_state)
    if left_res == -1:
        return -1, -1

    left_res = _move_to_pose_with_rrt_dual(left_planner, left_init_pose, right_planner.gripper_state)
    if left_res == -1:
        return -1, -1
    # Right arm does not execute manipulation in this task; avoid zero-length RRT path.
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    return left_res, right_res
