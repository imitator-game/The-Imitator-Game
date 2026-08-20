from mani_skill.envs.tasks import TwoRobotLiftLidFromSkilletEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver


def solve(env: TwoRobotLiftLidFromSkilletEnvL3, seed=None, debug=False, vis=False):
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
    left_init_pose = env.left_init_pose

    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(env.waypoint1, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(env.waypoint2, other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(env.waypoint3, other_gripper_state=right_planner.gripper_state)
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    return left_res, right_res
