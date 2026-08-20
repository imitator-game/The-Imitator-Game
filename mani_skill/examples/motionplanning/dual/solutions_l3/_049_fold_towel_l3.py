import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotFoldTowelEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

def solve(env: TwoRobotFoldTowelEnvL3, seed=None, debug=False, vis=False):
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

    env = env.unwrapped

    left_init_pose = env.left_agent.tcp.pose
    base_cloth = sapien.Pose(env.cloth.pose.sp.p, np.array(left_init_pose.q[0]))
    base_basket = sapien.Pose(env.basket.pose.sp.p, np.array(left_init_pose.q[0]))
    reach_pose1 = base_cloth * sapien.Pose(p=[0.1, 0.2, -0.15], q=euler2quat(0, 0, 0))
    grasp_pose1 = base_cloth * sapien.Pose(p=[0.1, 0.05, 0.0], q=euler2quat(np.pi / 2, 0, 0))
    reach_pose2 = base_cloth * sapien.Pose(p=[-0.05, 0.05, -0.05], q=euler2quat(np.pi / 2, np.pi / 2, 0))
    reach_pose3 = base_cloth * sapien.Pose(p=[-0.07, 0.05, -0.02], q=euler2quat(np.pi / 2, np.pi * 3 / 4, 0))
    reach_pose4 = base_cloth * sapien.Pose(p=[-0.1, 0.13, -0.02], q=euler2quat(np.pi / 2, np.pi * 2.5 / 4, 0))
    reach_pose5 = base_cloth * sapien.Pose(p=[-0.07, 0.13, 0.0], q=euler2quat(np.pi / 2, 0, 0))
    grasp_pose2 = base_cloth * sapien.Pose(p=[-0.05, 0.05, 0.0], q=euler2quat(np.pi / 2, 0, 0))
    reach_pose6 = base_cloth * sapien.Pose(p=[-0.05, 0.05, -0.05], q=euler2quat(np.pi / 2, 0, 0))
    goal_pose = base_basket * sapien.Pose(p=[0.01, 0.02, -0.2], q=euler2quat(np.pi / 4, np.pi / 4, 0))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(grasp_pose1 * sapien.Pose([-0.02, 0, -0.03]), other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Fold
    # -------------------------------------------------------------------------- #
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose2 * sapien.Pose([0.01, 0, -0.03]), other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose3 * sapien.Pose([0.01, 0, -0.03]), other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose4, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose5, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(grasp_pose2, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose6, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    # left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_planner.gripper_state)
    # left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(goal_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    # left_planner.move_to_pose_with_screw(reach_pose4, other_gripper_state=right_planner.gripper_state)
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
