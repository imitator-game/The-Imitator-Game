import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPlaceBookBookcaseEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb


def _pick_and_place(
    env,
    planner: PandaArmMotionPlanningSolver,
    other_planner: PandaArmMotionPlanningSolver,
    obj,
    place_center_xyz: np.ndarray,
    finger_length: float = 0.025,
    approaching: np.ndarray = None,
    target_closing: np.ndarray = None,
):
    if approaching is None:
        approaching = np.array([0, 0, -1], dtype=np.float32)
    if target_closing is None:
        target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    obb = get_actor_obb(obj)
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=finger_length,
    )
    closing = grasp_info["closing"]
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, obj.pose.sp.p) * sapien.Pose([0, -0.02, 0])

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    transit_pose = grasp_pose * sapien.Pose([-0.1, 0, -0.2])
    goal_pose = sapien.Pose(place_center_xyz, grasp_pose.q) * sapien.Pose([0, 0, -0.02])
    pre_goal_pose = goal_pose * sapien.Pose([-0.1, 0, -0.3])
    place_pose = goal_pose * sapien.Pose([0, 0, -0.25])

    planner.move_to_pose_with_screw(reach_pose, other_gripper_state=other_planner.gripper_state)
    grasp_target_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(grasp_target_pose, other_gripper_state=other_planner.gripper_state)
    planner.close_gripper(other_gripper_state=other_planner.gripper_state)

    planner.move_to_pose_with_screw(transit_pose, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(pre_goal_pose, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(place_pose, other_gripper_state=other_planner.gripper_state)
    planner.open_gripper(other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(pre_goal_pose, other_gripper_state=other_planner.gripper_state)


def solve(env: TwoRobotPlaceBookBookcaseEnvL3, seed=None, debug=False, vis=False):
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

    # Step 1: remove the horizontal book2 from top of the wooden box.
    book2_drop_center = env.bookcase.pose.sp.p + np.array([-0.3, 0.0, 0.05], dtype=np.float32)
    obb_book2 = get_actor_obb(env.book2)
    approach_book2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # grasp along +x
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
    grasp_pose_book2 = grasp_pose_book2 * sapien.Pose([0, 0, -0.11])
    reach_pose_book2 = grasp_pose_book2 * sapien.Pose([0, 0, -0.05])
    lift_pose_book2 = grasp_pose_book2 * sapien.Pose([0, -0.1, -0.05])
    pre_drop_pose_book2 = sapien.Pose(book2_drop_center, grasp_pose_book2.q) * sapien.Pose(
        [0.05, -0.15, 0]
    )
    drop_pose_book2 = sapien.Pose(book2_drop_center, grasp_pose_book2.q) * sapien.Pose(
        [0, -0.15, 0]
    )
    rotate_before_release_pose_book2 = drop_pose_book2 * sapien.Pose(
        [0, 0, 0], euler2quat(-np.pi / 10, 0, 0)
    )

    left_planner.move_to_pose_with_screw(reach_pose_book2, other_gripper_state=right_state)
    left_planner.move_to_pose_with_screw(grasp_pose_book2, other_gripper_state=right_state)
    left_planner.close_gripper(other_gripper_state=right_state)
    left_planner.move_to_pose_with_screw(lift_pose_book2, other_gripper_state=right_state)
    left_planner.move_to_pose_with_screw(pre_drop_pose_book2, other_gripper_state=right_state)
    left_planner.move_to_pose_with_screw(drop_pose_book2, other_gripper_state=right_state)
    left_planner.move_to_pose_with_screw(
        rotate_before_release_pose_book2, other_gripper_state=right_state
    )
    left_planner.open_gripper(other_gripper_state=right_state)
    left_planner.move_to_pose_with_screw(pre_drop_pose_book2, other_gripper_state=right_state)
    left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_state)

    # Step 2: continue the original flow - place the main book into the wooden box.
    box_center = env.bookcase.pose.sp.p
    _pick_and_place(env, left_planner, right_planner, obj=env.book, place_center_xyz=box_center)
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    return left_res, right_res
