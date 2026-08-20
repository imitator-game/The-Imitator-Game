import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPlaceFileFolderEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import mirror_offset_pose

def solve(env: TwoRobotPlaceFileFolderEnv, seed=None, debug=False, vis=False):
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
    right_init_pose = env.right_init_pose

    # -------------------------------------------------------------------------- #
    # Open
    # -------------------------------------------------------------------------- #
    right_planner.close_gripper(other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(env.waypoint1, other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(env.waypoint2, other_gripper_state=left_planner.gripper_state)
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(env.waypoint3, other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(right_init_pose, other_gripper_state=left_planner.gripper_state)


    base_file = sapien.Pose(env.file.pose.sp.p, np.array(left_init_pose.q[0]))
    base_folder = sapien.Pose(env.folder.pose.sp.p, np.array(left_init_pose.q[0]))

    reach_pose1 = base_file * mirror_offset_pose(
        sapien.Pose(p=[0.0, 0.1, -0.15], q=euler2quat(0, 0, 0))
    )
    grasp_pose = base_file * mirror_offset_pose(
        sapien.Pose(p=[0.0, 0.02, 0.0], q=euler2quat(np.pi / 2, 0, 0))
    )
    reach_pose2 = grasp_pose * mirror_offset_pose(sapien.Pose([0, -0.15, 0.]))
    reach_pose3 = base_folder * mirror_offset_pose(
        sapien.Pose(p=[0, 0.15, -0.2], q=euler2quat(np.pi / 2, 0, 0))
    )
    goal_pose = base_folder * mirror_offset_pose(
        sapien.Pose(p=[0, 0.15, -0.1], q=euler2quat(np.pi / 6, 0, 0))
    )
    reach_pose4 = base_folder * mirror_offset_pose(
        sapien.Pose(p=[0, 0.15, -0.2], q=euler2quat(np.pi / 8, 0, 0))
    )

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(goal_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose4, other_gripper_state=right_planner.gripper_state)
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
