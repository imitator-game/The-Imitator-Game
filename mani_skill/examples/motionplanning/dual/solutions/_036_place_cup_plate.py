import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPlaceCupPlateEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    mirror_offset_pose,
    is_l2_enabled
)

def solve(env: TwoRobotPlaceCupPlateEnv, seed=None, debug=False, vis=False):
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
    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose

    # -------------------------------------------------------------------------- #
    # cup_1
    # -------------------------------------------------------------------------- #

    obb = get_actor_obb(env.cup_1)

    approaching = np.array([0, 0, -1])
    # we can build a simple grasp pose using this information for Panda
    target_closing = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # Gram-Schmidt orthogonalization
    target_closing = target_closing - (target_closing @ approaching) * approaching
    target_closing = target_closing / np.linalg.norm(target_closing)


    grasp_pose = env.left_agent.build_grasp_pose(approaching, target_closing, env.cup_1.pose.sp.p)
    goal_pose = sapien.Pose(env.tray.pose.sp.p, grasp_pose.q) * mirror_offset_pose(sapien.Pose([0, 0, -0.05]))
    reach_pose2 = grasp_pose * mirror_offset_pose(sapien.Pose([0.0, 0, -0.2]))
    reach_pose1 = grasp_pose * mirror_offset_pose(sapien.Pose([0.0, 0, -0.15]))
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(grasp_pose * mirror_offset_pose(sapien.Pose([0.0, 0, -0.08])), other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_planner.gripper_state)
    reach_pose3 = goal_pose * mirror_offset_pose(sapien.Pose([0.0, 0.07, -0.2]))
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(goal_pose * mirror_offset_pose(sapien.Pose([0.0, 0.07, -0.06])), other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # cup_2
    # -------------------------------------------------------------------------- #

    obb = get_actor_obb(env.cup_2)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # Gram-Schmidt orthogonalization
    target_closing = target_closing - (target_closing @ approaching) * approaching
    target_closing = target_closing / np.linalg.norm(target_closing)


    grasp_pose = env.right_agent.build_grasp_pose(approaching, target_closing, env.cup_2.pose.sp.p)
    goal_pose = sapien.Pose(env.tray.pose.sp.p, grasp_pose.q) * mirror_offset_pose(sapien.Pose([0, 0, -0.05]))
    reach_pose2 = grasp_pose * mirror_offset_pose(sapien.Pose([-0.0, -0.0, -0.2]))
    reach_pose1 = grasp_pose * mirror_offset_pose(sapien.Pose([-0.0, -0.0, -0.15]))
    right_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(grasp_pose * mirror_offset_pose(sapien.Pose([-0.0, -0.0, -0.04])), other_gripper_state=left_planner.gripper_state)
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=left_planner.gripper_state)
    reach_pose3 = goal_pose * mirror_offset_pose(sapien.Pose([-0.0, -0.07, -0.2]))
    right_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(goal_pose * mirror_offset_pose(sapien.Pose([-0.0, -0.07, -0.02])), other_gripper_state=left_planner.gripper_state)
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=left_planner.gripper_state)
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.move_to_pose_with_screw(right_init_pose, other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
