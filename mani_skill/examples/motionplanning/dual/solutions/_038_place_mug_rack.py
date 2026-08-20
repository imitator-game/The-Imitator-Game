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
    is_l3_enabled,
    is_lr_mirror_enabled,
)


def _dbg_pose(tag: str, pose: sapien.Pose):
    p = np.array(pose.p, dtype=np.float32)
    q = np.array(pose.q, dtype=np.float32)
    e = quat2euler(q)
    r = quat2mat(q)


def _manual_mirror_offset(
    p_xyz,
    euler_normal,
    euler_mirror,
):
    p = np.array(p_xyz, dtype=np.float32)
    if is_lr_mirror_enabled():
        p[1] = -p[1]  # y-only mirror for position
        q = euler2quat(*euler_mirror)
    else:
        q = euler2quat(*euler_normal)
    return sapien.Pose(p, q)

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
    # Mirror-only extra local rotation on grasp pose (tune this tuple).
    # Format: (rx, ry, rz) in radians, applied in local TCP frame.
    grasp_mirror_extra_euler = (0.0, 0.0, -np.pi / 2.8)
    if is_lr_mirror_enabled():
        grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(*grasp_mirror_extra_euler))
    goal_pose = sapien.Pose(env.rack.pose.sp.p, grasp_pose.q) * sapien.Pose([0, 0, -0.05])
    reach_pose2 = grasp_pose * _manual_mirror_offset(
        [0.05, -0.05, -0.2],
        (0, 0, -np.pi * 3 / 4),
        (0, 0, np.pi * 3 / 4 ),
    )

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose1 = grasp_pose * _manual_mirror_offset(
        [0.05, -0.05, -0.15],
        (0, 0, -np.pi * 3 / 4),
        (0, 0, np.pi * 3 / 4),
    )
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        grasp_pose
        * _manual_mirror_offset(
            [0.05, -0.05, -0.085],
            (0, 0, -np.pi * 3 / 4),
            (0, 0, np.pi * 3 / 4),
        ),
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_planner.gripper_state)
    reach_pose3 = goal_pose * _manual_mirror_offset(
        [0.17, 0.15, -0.2],
        (np.pi / 4, -np.pi / 8, -np.pi / 2),
        (-np.pi / 4, -np.pi / 8, np.pi / 2),
    )
    if vis or debug:
        _dbg_pose("reach_pose3", reach_pose3)
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(
        goal_pose
        * _manual_mirror_offset(
            [0.14, 0.07, -0.14],
            (np.pi / 4, -np.pi / 8, -np.pi / 2),
            (-np.pi / 4, -np.pi / 8, np.pi / 2),
        ),
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
