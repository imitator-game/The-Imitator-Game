import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.dual_tasks_l3._011_place_commodity_rack_l3 import (
    TwoRobotPlaceCommodityRackEnvL3,
)
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


FINGER_LENGTH = 0.025

def _pick_and_place_mug(
    planner: PandaArmMotionPlanningSolver,
    other_planner: PandaArmMotionPlanningSolver,
    agent,
    mug,
    rack,
    place_offset: np.ndarray,
    side_sign: float,
):
    
    obb = get_actor_obb(mug)

    approaching = np.array([0, 0, -1])

    target_closing = agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing = grasp_info["closing"]
    grasp_pose = agent.build_grasp_pose(approaching, closing, mug.pose.sp.p)
    goal_pose = sapien.Pose(np.array(rack.pose.sp.p) + np.array(place_offset), grasp_pose.q)
    reach_pose1 = sapien.Pose(
        np.array(grasp_pose.p) + np.array([0.0, 0.08 * side_sign, 0.2]),
        (euler2quat(np.pi, 0, 0)),
    )
    grasp_touch_pose = sapien.Pose(
        np.array(grasp_pose.p) + np.array([0.0, 0.08 * side_sign, 0.07]),
        (euler2quat(np.pi, 0, 0)),
    )
    reach_pose2 = sapien.Pose(
        np.array(grasp_pose.p) + np.array([-0.05, 0.0, 0.2]),
        (grasp_pose * sapien.Pose(q=euler2quat(0, 0, 3 * np.pi / 4 * side_sign))).q,
    )
    reach_pose3 = sapien.Pose(
        np.array(goal_pose.p) + np.array([-0.15, side_sign * 0.11, 0.23]),
        (goal_pose * sapien.Pose(q=euler2quat(-side_sign * np.pi / 4, -np.pi / 8, np.pi * side_sign / 3))).q,
    )
    place_pose = sapien.Pose(
        np.array(goal_pose.p) + np.array([-0.05, side_sign * 0.11, 0.16]),
        (goal_pose * sapien.Pose(q=euler2quat(-side_sign * np.pi / 4, -np.pi / 8, np.pi * side_sign / 3))).q,
    )

    planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(grasp_touch_pose, other_gripper_state=other_planner.gripper_state)
    planner.close_gripper(other_gripper_state=other_planner.gripper_state)
    # planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(place_pose, other_gripper_state=other_planner.gripper_state)
    planner.open_gripper(other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=other_planner.gripper_state)


def solve(env: TwoRobotPlaceCommodityRackEnvL3, seed=None, debug=False, vis=False):
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

    # Left mug -> left rack
    _pick_and_place_mug(
        planner=left_planner,
        other_planner=right_planner,
        agent=env.left_agent,
        mug=env.mug_left,
        rack=env.rack_left,
        place_offset=np.array([0.0, 0.0, 0.02]),
        side_sign=-1.0,
    )
    left_res = left_planner.move_to_pose_with_screw(
        left_init_pose, other_gripper_state=right_planner.gripper_state
    )

    # Right mug -> right rack
    _pick_and_place_mug(
        planner=right_planner,
        other_planner=left_planner,
        agent=env.right_agent,
        mug=env.mug_right,
        rack=env.rack_right,
        place_offset=np.array([0.0, 0.0, 0.02]),
        side_sign=1.0,
    )
    right_res = right_planner.move_to_pose_with_screw(
        right_init_pose, other_gripper_state=left_planner.gripper_state
    )

    left_planner.close()
    right_planner.close()
    return left_res, right_res
