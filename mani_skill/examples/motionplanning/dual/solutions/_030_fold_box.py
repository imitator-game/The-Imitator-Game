import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotFoldBoxEnv
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import mirror_offset_pose, is_l1_enabled, is_l2_enabled
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver


def _transform_point(origin_base: sapien.Pose, new_base: sapien.Pose, origin_pos: sapien.Pose):
    from scipy.spatial.transform import Rotation
    # Convert quaternions to rotation objects
    R_O = Rotation.from_quat(
        [origin_base.q[1], origin_base.q[2], origin_base.q[3], origin_base.q[0]])  # scipy uses (x,y,z,w) format
    R_O_new = Rotation.from_quat([new_base.q[1], new_base.q[2], new_base.q[3], new_base.q[0]])

    # Calculate relative position of A with respect to O
    relative_pos = origin_pos.p - origin_base.p

    # Calculate relative rotation of A with respect to O
    R_A = Rotation.from_quat([origin_pos.q[1], origin_pos.q[2], origin_pos.q[3], origin_pos.q[0]])
    relative_rot = R_O.inv() * R_A

    # Calculate new position of A
    relative_pos_rotated = R_O_new.apply(R_O.inv().apply(relative_pos))
    p_A_new = new_base.p + relative_pos_rotated

    # Calculate new rotation of A
    R_A_new = R_O_new * relative_rot
    q_A_new = R_A_new.as_quat()  # Returns (x,y,z,w)

    # Convert back to (w,x,y,z) format
    q_A_new = np.array([q_A_new[3], q_A_new[0], q_A_new[1], q_A_new[2]])

    new_pos = sapien.Pose(
        p=p_A_new,
        q=q_A_new
    )
    return new_pos


def solve(env: TwoRobotFoldBoxEnv, seed=None, debug=False, vis=False):
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

    left_init_pose = env.left_init_pose
    right_init_pose = env.right_agent.tcp.pose

    # -------------------------------------------------------------------------- #
    # Close (dynamic waypoints from lid position to handle L2 models)
    # -------------------------------------------------------------------------- #
    lid_pos = env.partnet_link_positions()[0].cpu().numpy()
    robot_pos = env.left_agent.robot.pose.p[0].cpu().numpy()
    dir_xy = lid_pos[:2] - robot_pos[:2]
    norm = np.linalg.norm(dir_xy)
    if norm < 1e-6:
        dir_xy = np.array([0.0, 1.0])
    else:
        dir_xy = dir_xy / norm
    approach_pos = np.array([lid_pos[0], lid_pos[1], lid_pos[2] + 0.05])
    approach_pos[:2] -= dir_xy * 0.14
    push_pos = np.array([lid_pos[0], lid_pos[1], lid_pos[2] + 0.05])
    push_pos[:2] -= dir_xy * 0.02
    retreat_pos = approach_pos + np.array([0.0, 0.0, 0.10])

    waypoint1 = env.waypoint1
    waypoint2 = env.waypoint2
    waypoint3 = env.waypoint3
    if is_l2_enabled():
        left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
        current_pose = env.left_agent.tcp.pose
        mid_pos = (current_pose.p[0].cpu().numpy() + waypoint1.p) * 0.5
        waypoint1_mid = sapien.Pose(mid_pos, waypoint1.q)
        waypoint1_mid_rot = waypoint1_mid * mirror_offset_pose(sapien.Pose([-0.01, -0.01, -0.01], euler2quat(0, 0, 0)))
        left_planner.move_to_pose_with_screw(waypoint1_mid_rot, other_gripper_state=right_planner.gripper_state)
        # left_planner.move_to_pose_with_screw(waypoint1, other_gripper_state=right_planner.gripper_state)
        waypoint2_adjust = waypoint2 * mirror_offset_pose(sapien.Pose([-0.01, -0.01, -0.01], euler2quat(0, 0, 0)))
        left_planner.move_to_pose_with_screw(waypoint2_adjust, other_gripper_state=right_planner.gripper_state)
        left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
        waypoint2_3 = waypoint3
        left_planner.move_to_pose_with_screw(waypoint2_3, other_gripper_state=right_planner.gripper_state)
        # left_planner.move_to_pose_with_screw(waypoint3, other_gripper_state=right_planner.gripper_state)
    elif is_l1_enabled():
        left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
        current_pose = env.left_agent.tcp.pose
        mid_pos = (current_pose.p[0].cpu().numpy() + waypoint1.p) * 0.5
        waypoint1_mid = sapien.Pose(mid_pos, waypoint1.q)
        waypoint1_mid_rot = waypoint1_mid * sapien.Pose([0.15, -0.05, 0.08], euler2quat(0, 0, 0))
        waypoint1_mid_rot = _transform_point(env.origin_base, sapien.Pose(np.array(env.box.pose.p)[0], np.array(env.box.pose.q)[0]), waypoint1_mid_rot)
        left_planner.move_to_pose_with_screw(waypoint1_mid_rot, other_gripper_state=right_planner.gripper_state)
        # left_planner.move_to_pose_with_screw(waypoint1, other_gripper_state=right_planner.gripper_state)
        waypoint2_adjust = waypoint2 * sapien.Pose([0.08, -0.05, 0.08], euler2quat(0, 0, 0))
        waypoint2_adjust = _transform_point(env.origin_base, sapien.Pose(np.array(env.box.pose.p)[0], np.array(env.box.pose.q)[0]), waypoint2_adjust)
        left_planner.move_to_pose_with_screw(waypoint2_adjust, other_gripper_state=right_planner.gripper_state)
        left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
        waypoint2_3 = waypoint3 * sapien.Pose([0.08, 0.05, 0.08], euler2quat(0, 0, 0))
        waypoint2_3 = _transform_point(env.origin_base, sapien.Pose(np.array(env.box.pose.p)[0], np.array(env.box.pose.q)[0]), waypoint2_3)
        left_planner.move_to_pose_with_screw(waypoint2_3, other_gripper_state=right_planner.gripper_state)
        # left_planner.move_to_pose_with_screw(waypoint3, other_gripper_state=right_planner.gripper_state)
    else:
        left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
        current_pose = env.left_agent.tcp.pose
        mid_pos = (current_pose.p[0].cpu().numpy() + waypoint1.p) * 0.5
        waypoint1_mid = sapien.Pose(mid_pos, waypoint1.q)
        waypoint1_mid_rot = waypoint1_mid * mirror_offset_pose(sapien.Pose([0.0, 0.0, -0.00], euler2quat(0, 0, 0)))
        left_planner.move_to_pose_with_screw(waypoint1_mid_rot, other_gripper_state=right_planner.gripper_state)
        # left_planner.move_to_pose_with_screw(waypoint1, other_gripper_state=right_planner.gripper_state)
        waypoint2_adjust = waypoint2 * sapien.Pose([0.0, 0.0, -0.0], euler2quat(0, 0, 0))
        left_planner.move_to_pose_with_screw(waypoint2_adjust, other_gripper_state=right_planner.gripper_state)
        left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
        waypoint2_3 = waypoint3
        left_planner.move_to_pose_with_screw(waypoint2_3, other_gripper_state=right_planner.gripper_state)
        # left_planner.move_to_pose_with_screw(waypoint3, other_gripper_state=right_planner.gripper_state)
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
