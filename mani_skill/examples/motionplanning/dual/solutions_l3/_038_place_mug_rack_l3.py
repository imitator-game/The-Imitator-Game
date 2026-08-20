import numpy as np
import sapien
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat

from mani_skill.envs.tasks import TwoRobotPlaceMugRackEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    mirror_offset_pose,
)


def _dbg_pose(tag: str, pose: sapien.Pose):
    p = np.array(pose.p, dtype=np.float32)
    q = np.array(pose.q, dtype=np.float32)
    e = quat2euler(q)
    r = quat2mat(q)

def solve(env: TwoRobotPlaceMugRackEnv, seed=None, debug=False, vis=False):
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

    obb = get_actor_obb(env.mug)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.mug.pose.sp.p)
    goal_pose = sapien.Pose(env.rack.pose.sp.p, grasp_pose.q) * sapien.Pose([0.03, -0.03, 0]) 
    reach_pose2 = grasp_pose * mirror_offset_pose(sapien.Pose([0.05, -0.05, -0.2], euler2quat(0, 0, -np.pi * 3/4)))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose1 = grasp_pose * mirror_offset_pose(sapien.Pose([0.05, -0.05, -0.15], euler2quat(0, 0, -np.pi * 3/4)))
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(grasp_pose * mirror_offset_pose(sapien.Pose([0.05, -0.05, -0.08], euler2quat(0, 0, -np.pi * 3/4))), other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_planner.gripper_state)
    reach_pose3 = goal_pose * sapien.Pose([0.0, 0.0, -0.2], euler2quat(0, 0, -np.pi * 3/4))
    if vis or debug:
        _dbg_pose("reach_pose3", reach_pose3)
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(goal_pose * mirror_offset_pose(sapien.Pose([0.0, 0.0, -0.1], euler2quat(0, 0, -np.pi * 3/4))), other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
