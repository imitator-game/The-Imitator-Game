import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPlaceClothBasketEnvL3
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


def _remove_book2_from_box(env, planner: PandaArmMotionPlanningSolver, other_planner: PandaArmMotionPlanningSolver):
    """Step 1: same style as task 002, move horizontal book2 away from box top."""
    right_state = other_planner.gripper_state

    book2_drop_center = env.bookcase.pose.sp.p + np.array([-0.1, -0.5, 0.2], dtype=np.float32)
    obb_book2 = get_actor_obb(env.book2)
    approach_book2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    closing_book2 = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    grasp_info_book2 = compute_grasp_info_by_obb(
        obb_book2,
        approaching=approach_book2,
        target_closing=closing_book2,
        depth=0.025,
    )

    grasp_pose_book2 = env.left_agent.build_grasp_pose(
        approach_book2, grasp_info_book2["closing"], env.book2.pose.sp.p
    )
    grasp_pose_book2 = grasp_pose_book2 * sapien.Pose([0, 0, -0.1])
    reach_pose_book2 = grasp_pose_book2 * sapien.Pose([0, 0, -0.05])
    lift_pose_book2 = grasp_pose_book2 * sapien.Pose([0, -0.1, -0.05])
    pre_drop_pose_book2 = sapien.Pose(book2_drop_center, grasp_pose_book2.q) * sapien.Pose([0.05, -0.05, 0])
    drop_pose_book2 = sapien.Pose(book2_drop_center, grasp_pose_book2.q) * sapien.Pose([0, -0.05, 0])
    rotate_before_release_pose_book2 = drop_pose_book2 * sapien.Pose(
        [0, 0, 0], euler2quat(-np.pi / 10, 0, 0)
    )
    after_drop_pose_book2 = sapien.Pose(drop_pose_book2.p, grasp_pose_book2.q) * sapien.Pose([0, 0, -0.1])

    planner.move_to_pose_with_screw(reach_pose_book2, other_gripper_state=right_state)
    planner.move_to_pose_with_screw(grasp_pose_book2, other_gripper_state=right_state)
    planner.close_gripper(other_gripper_state=right_state)
    planner.move_to_pose_with_screw(lift_pose_book2, other_gripper_state=right_state)
    planner.move_to_pose_with_screw(pre_drop_pose_book2, other_gripper_state=right_state)
    planner.move_to_pose_with_screw(drop_pose_book2, other_gripper_state=right_state)
    planner.move_to_pose_with_screw(rotate_before_release_pose_book2, other_gripper_state=right_state)
    planner.open_gripper(other_gripper_state=right_state)
    planner.move_to_pose_with_screw(after_drop_pose_book2, other_gripper_state=right_state)
    planner.move_to_pose_with_screw(pre_drop_pose_book2, other_gripper_state=right_state)


def solve(env: TwoRobotPlaceClothBasketEnvL3, seed=None, debug=False, vis=False):
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

    # Step 1: move the top horizontal book away from the box.
    _remove_book2_from_box(env, left_planner, right_planner)
    # left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)

    # Step 2: keep the original cloth flow, but target the wooden box.
    base_cloth = sapien.Pose(env.cloth.pose.sp.p, np.array(left_init_pose.q[0]))
    base_box = sapien.Pose(env.bookcase.pose.sp.p, np.array(left_init_pose.q[0]))

    reach_pose1 = base_cloth * sapien.Pose(p=[0.05, 0.2, -0.15], q=euler2quat(0, 0, 0))
    grasp_pose = base_cloth * sapien.Pose(p=[-0.05, 0.07, 0.0], q=euler2quat(np.pi / 2, 0, 0))
    reach_pose2 = grasp_pose * sapien.Pose([0, -0.15, 0])
    reach_pose3 = base_box * sapien.Pose(p=[0, 0.15, -0.2], q=euler2quat(np.pi / 2, 0, 0))
    goal_pose = base_box * sapien.Pose(p=[0.03, 0.0, -0.2], q=euler2quat(np.pi / 3, 0, 0))
    reach_pose4 = base_box * sapien.Pose(p=[0, 0.15, -0.2], q=euler2quat(np.pi * 3 / 4, 0, 0))

    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_planner.gripper_state)
    # Use RRTConnect for the final placement stage to avoid TOPP failures in screw planning.
    left_planner_res = _move_to_pose_with_rrt_dual(
        left_planner, reach_pose3, right_planner.gripper_state
    )
    if left_planner_res == -1:
        return -1, -1
    left_planner_res = _move_to_pose_with_rrt_dual(
        left_planner, goal_pose, right_planner.gripper_state
    )
    if left_planner_res == -1:
        return -1, -1
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    # left_planner_res = _move_to_pose_with_rrt_dual(
    #     left_planner, reach_pose4, right_planner.gripper_state
    # )
    # if left_planner_res == -1:
    #     return -1, -1

    left_res = _move_to_pose_with_rrt_dual(
        left_planner, left_init_pose, right_planner.gripper_state
    )
    if left_res == -1:
        return -1, -1
    # Temporary workaround: skip right-arm return-to-init to avoid empty-trajectory
    # edge case in follow_path (n_step == 0).
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
