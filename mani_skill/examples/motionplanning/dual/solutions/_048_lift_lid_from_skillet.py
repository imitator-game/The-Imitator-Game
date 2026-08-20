import numpy as np
import sapien

from mani_skill.envs.tasks import TwoRobotLiftLidFromSkilletEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import is_lr_mirror_enabled, mirror_xyz

def solve(env: TwoRobotLiftLidFromSkilletEnv, seed=None, debug=False, vis=False):
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
    lr_mirror = is_lr_mirror_enabled()
    
    lid_obb = get_actor_obb(env.lid)
    approaching = np.array([0, 0, -1])
    initial_pose = env.left_agent.tcp.pose
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        lid_obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]

    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.lid.pose.sp.p)
    grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.01])
    
    
    reach_pose1 = grasp_pose * sapien.Pose([0, 0, -0.05])
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)
    
    left_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    
    reach2_offset = np.array([0, 0.0, 0.1], dtype=np.float32)
    if lr_mirror:
        reach2_offset = mirror_xyz(reach2_offset)
    reach_pose2 = sapien.Pose(p=env.lid.pose.sp.p + reach2_offset, q=grasp_pose.q)
    left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_planner.gripper_state)
    
    reach3_offset = np.array([0, -0.25, 0.1], dtype=np.float32)
    if lr_mirror:
        reach3_offset = mirror_xyz(reach3_offset)
    reach_pose3 = sapien.Pose(p=env.skillet.pose.sp.p + reach3_offset, q=grasp_pose.q)
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    
    place_offset = np.array([0, -0.25, 0.01], dtype=np.float32)
    if lr_mirror:
        place_offset = mirror_xyz(place_offset)
    place_pose = sapien.Pose(p=env.skillet.pose.sp.p + place_offset, q=grasp_pose.q)
    left_planner.move_to_pose_with_screw(place_pose, other_gripper_state=right_planner.gripper_state)
    
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(initial_pose, other_gripper_state=right_planner.gripper_state)
    
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    # recover to init pose
    
    return left_res, right_res
