import numpy as np
import sapien

from mani_skill.envs.tasks import TwoRobotPickAppleBasketEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


def solve(env: TwoRobotPickAppleBasketEnvL3, seed=None, debug=False, vis=False):
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

    approaching = np.array([0, 0, -1], dtype=np.float32)
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    target_closing = target_closing - (target_closing @ approaching) * approaching
    target_closing = target_closing / np.linalg.norm(target_closing)

    grasp_pose = env.left_agent.build_grasp_pose(approaching, target_closing, env.apple.pose.sp.p)

    goal_pose = sapien.Pose(env.box.pose.sp.p, grasp_pose.q) * sapien.Pose([0, 0, -0.05])

    reach_pose1 = grasp_pose * sapien.Pose([0, 0, -0.05])
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_state)

    left_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=right_state)
    left_planner.close_gripper(other_gripper_state=right_state)

    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(lift_pose, other_gripper_state=right_state)

    reach_pose2 = goal_pose * sapien.Pose([-0.05, 0, -0.15])
    left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_state)

    place_pose = goal_pose * sapien.Pose([-0.05, 0, -0.02])
    left_planner.move_to_pose_with_screw(place_pose, other_gripper_state=right_state)

    left_res = left_planner.open_gripper(other_gripper_state=right_state)

    retract_pose = goal_pose * sapien.Pose([-0.05, 0, -0.15])
    left_planner.move_to_pose_with_screw(retract_pose, other_gripper_state=right_state)
    left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_state)

    left_planner.close()
    right_planner.close()

    return left_res, left_res
