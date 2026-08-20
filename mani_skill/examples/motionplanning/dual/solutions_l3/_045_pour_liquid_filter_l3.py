import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPourLiquidFilterEnvL3
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import mirror_sign
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver

def solve(env: TwoRobotPourLiquidFilterEnvL3, seed=None, debug=False, vis=False):
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
    right_init_pose = env.right_agent.tcp.pose
    liquid_initial_pos = env.liquid.pose.sp.p.copy()

    approaching = np.array([0, 0, -1])
    z_rotation_angle = -np.pi / 2
    closing = np.array([
        -np.sin(z_rotation_angle),
        np.cos(z_rotation_angle),
        0,
    ])

    grasp_pose_base = env.left_agent.build_grasp_pose(
        approaching,
        closing,
        env.liquid.pose.sp.p,
    )
    custom_offset = np.array([0.0, 0.0, 0.135], dtype=np.float32)
    grasp_pose = sapien.Pose(
        p=env.liquid.pose.sp.p + custom_offset,
        q=grasp_pose_base.q,
    )

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    left_planner.move_to_pose_with_screw(
        reach_pose,
        other_gripper_state=right_planner.gripper_state,
    )

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        grasp_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move above cup
    # -------------------------------------------------------------------------- #
    above_cup_pose_base = sapien.Pose(
        env.cup.pose.sp.p + np.array([0.0, 0.0, 0.25], dtype=np.float32),
        grasp_pose.q,
    )
    backward_offset = sapien.Pose([mirror_sign(-0.13), 0, 0])
    above_cup_pose = above_cup_pose_base * backward_offset
    left_planner.move_to_pose_with_screw(
        above_cup_pose,
        other_gripper_state=right_planner.gripper_state,
    )

    # -------------------------------------------------------------------------- #
    # Pour
    # -------------------------------------------------------------------------- #
    pour_rotation = euler2quat(0, mirror_sign(-np.pi / 6), 0)
    from scipy.spatial.transform import Rotation as R
    grasp_rot = R.from_quat([grasp_pose.q[1], grasp_pose.q[2], grasp_pose.q[3], grasp_pose.q[0]])
    pour_rot = R.from_quat([pour_rotation[1], pour_rotation[2], pour_rotation[3], pour_rotation[0]])
    combined_rot = grasp_rot * pour_rot
    combined_quat = combined_rot.as_quat()
    combined_quat_sapien = [combined_quat[3], combined_quat[0], combined_quat[1], combined_quat[2]]

    pour_pose = sapien.Pose(
        above_cup_pose.p,
        combined_quat_sapien,
    )
    left_planner.move_to_pose_with_screw(
        pour_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    for _ in range(3):
        left_planner.move_to_pose_with_screw(
            pour_pose,
            other_gripper_state=right_planner.gripper_state,
        )

    # -------------------------------------------------------------------------- #
    # Return and place down
    # -------------------------------------------------------------------------- #
    return_above_pose = sapien.Pose(
        liquid_initial_pos + np.array([0.0, 0.0, 0.30], dtype=np.float32),
        grasp_pose.q,
    )
    left_planner.move_to_pose_with_screw(
        return_above_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    place_pose = sapien.Pose(
        liquid_initial_pos + np.array([0.0, 0.0, 0.15], dtype=np.float32),
        grasp_pose.q,
    )
    left_planner.move_to_pose_with_screw(
        place_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    retract_pose = place_pose * sapien.Pose([0, 0, -0.1])
    left_planner.move_to_pose_with_screw(
        retract_pose,
        other_gripper_state=right_planner.gripper_state,
    )

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
