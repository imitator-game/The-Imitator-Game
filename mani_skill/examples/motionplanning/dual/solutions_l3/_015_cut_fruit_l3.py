import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotCutFruitEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: TwoRobotCutFruitEnvL3, seed=None, debug=False, vis=False):
    options = {
    "reconfigure": True
    }
    env.reset(options=options, seed=seed)
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

    scissor_target = getattr(env, "scissor_grasp_target", env.scissor)

    left_init_pose = env.left_init_pose
    right_init_pose = env.right_agent.tcp.pose
    grasp_pose = sapien.Pose(np.array(env.left_agent.tcp.pose.p)[0], np.array(env.left_agent.tcp.pose.q)[0])
    reach_pose = sapien.Pose(env.carrot.pose.sp.p, grasp_pose.q) * sapien.Pose([0.4, -0., -0.], q=euler2quat(0, 0, 0))
    goal_pose = sapien.Pose(env.carrot.pose.sp.p, grasp_pose.q) * sapien.Pose([0.3, -0., -0.], q=euler2quat(0, 0, 0))
    place_pose = sapien.Pose(left_init_pose.p, left_init_pose.q) * sapien.Pose([0., -0., 0.05], q=euler2quat(0, 0, -np.pi / 4))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(goal_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(place_pose, other_gripper_state=right_planner.gripper_state)


    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state, scale=0.01)
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
