import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPutCubeOnScaleEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: TwoRobotPutCubeOnScaleEnv, seed=None, debug=False, vis=False):
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
    # Negative -> move place target backward in world X.
    place_back_x_delta = -0.03
    
    # move knife
    cube_obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    initial_pose = env.left_agent.tcp.pose
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        cube_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.cube.pose.sp.p)
    
    # Set ordered direction for knife (pointing along X-axis)
    cube_goal_ori = euler2quat(0, 0, np.pi / 2)
    # Calculate TCP orientation at goal to achieve desired object orientation
    rel_pose = env.cube.pose.sp.inv() * grasp_pose
    # Cube should be at scale (y = 0)
    cube_goal_pos = env.scale.pose.sp.p + np.array([place_back_x_delta, 0, 0.1], dtype=np.float32)
    # cube_goal_pos[2] = env.cube.pose.sp.p[2]
    goal_pose = sapien.Pose(cube_goal_pos, cube_goal_ori) * rel_pose
    
    mid_goal_pose = goal_pose * sapien.Pose([0, 0, -0.1])

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose1 = grasp_pose * sapien.Pose([0, 0, -0.1])
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(mid_goal_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(goal_pose, other_gripper_state=right_planner.gripper_state)
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    
    left_planner.move_to_pose_with_screw(mid_goal_pose, other_gripper_state=right_planner.gripper_state)
    # # recover to init pose
    left_planner.move_to_pose_with_screw(initial_pose, other_gripper_state=right_planner.gripper_state)
    
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    
    return left_res, right_res
