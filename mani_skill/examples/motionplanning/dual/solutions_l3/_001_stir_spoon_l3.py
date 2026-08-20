import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotStirSpoonEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

OPEN = 1
SETTLE_STEPS = 15


def _hold_action(env, left_gripper_state=OPEN, right_gripper_state=OPEN):
    env = env.unwrapped
    left_qpos = env.agent.agents[0].robot.qpos[0][:7].cpu().numpy()
    right_qpos = env.agent.agents[1].robot.qpos[0][:7].cpu().numpy()
    left_action = np.hstack([left_qpos, left_gripper_state])
    right_action = np.hstack([right_qpos, right_gripper_state])
    return {
        "panda_wristcam-0": left_action,
        "panda_wristcam-1": right_action,
    }


def _settle_env(env, steps=SETTLE_STEPS):
    hold_action = _hold_action(env, OPEN, OPEN)
    for _ in range(steps):
        env.step(hold_action)


def solve(env: TwoRobotStirSpoonEnvL3, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    _settle_env(env, steps=SETTLE_STEPS)

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
    tool_initial_position = env.spoon.pose.sp.p.copy()

    left_planner.open_gripper(other_gripper_state=right_state)
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    def _pose_delta_small(target_pose: sapien.Pose, pos_eps=1e-3, ang_eps=0.02):
        cur_pose = env.left_agent.tcp.pose.sp
        dp = np.linalg.norm(np.asarray(target_pose.p) - np.asarray(cur_pose.p))
        q1 = np.asarray(target_pose.q, dtype=np.float64)
        q2 = np.asarray(cur_pose.q, dtype=np.float64)
        q1 = q1 / (np.linalg.norm(q1) + 1e-12)
        q2 = q2 / (np.linalg.norm(q2) + 1e-12)
        dot = np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)
        dtheta = 2.0 * np.arccos(dot)
        return (dp < pos_eps) and (dtheta < ang_eps)

    def _safe_move(target_pose: sapien.Pose, refine_steps: int = 0):
        if _pose_delta_small(target_pose):
            return True
        res = left_planner.move_to_pose_with_screw(
            target_pose,
            refine_steps=refine_steps,
            other_gripper_state=right_state,
        )
        if res == -1:
            res = left_planner.move_to_pose_with_RRTConnect(
                target_pose,
                refine_steps=refine_steps,
                other_gripper_state=right_state,
            )
        return res != -1

    tool_pos = np.array(env.spoon.pose.sp.p, dtype=np.float32)
    grasp_roll = np.pi / 10
    grasp_pose_upright = sapien.Pose(
        p=tool_pos + np.array([-0.035, -0.02, 0.1], dtype=np.float32),
        q=euler2quat(np.pi, 0.0, 0.0),
    )
    # Pre-grasp roll around local x-axis.
    grasp_pose = grasp_pose_upright * sapien.Pose(q=euler2quat(grasp_roll, 0.0, 0.0))
    reach_pose = sapien.Pose(
        p=grasp_pose.p + np.array([0.0, 0.0, 0.18], dtype=np.float32),
        q=grasp_pose.q,
    )
    lift_pose_tilted = sapien.Pose(
        p=grasp_pose.p + np.array([0.0, 0.0, 0.18], dtype=np.float32),
        q=grasp_pose.q,
    )
    lift_pose_upright = sapien.Pose(
        p=lift_pose_tilted.p,
        q=grasp_pose_upright.q,
    )
    lift_pose_vertical = lift_pose_upright

    if not _safe_move(reach_pose):
        left_planner.close()
        right_planner.close()
        return -1, -1
    if not _safe_move(grasp_pose):
        left_planner.close()
        right_planner.close()
        return -1, -1
    left_planner.close_gripper(other_gripper_state=right_state, t=15)
    if not _safe_move(lift_pose_tilted):
        left_planner.close()
        right_planner.close()
        return -1, -1
    # Rotate back after lifting to a safer height.
    if not _safe_move(lift_pose_upright):
        left_planner.close()
        right_planner.close()
        return -1, -1
    if not _safe_move(lift_pose_vertical):
        left_planner.close()
        right_planner.close()
        return -1, -1

    bowl_center = env.bowl.pose.sp.p
    vertical_orientation = np.array(lift_pose_vertical.q, dtype=np.float32)
    above_bowl_pose = sapien.Pose(
        p=np.array([bowl_center[0], bowl_center[1], bowl_center[2] + 0.25]),
        q=vertical_orientation,
    )
    stir_height = bowl_center[2] + 0.16
    stir_radius = 0.02
    num_circles = 2
    points_per_circle = 12

    if not _safe_move(above_bowl_pose):
        left_planner.close()
        right_planner.close()
        return -1, -1

    center_pose = sapien.Pose(
        p=np.array([bowl_center[0], bowl_center[1], stir_height]),
        q=vertical_orientation,
    )
    if not _safe_move(center_pose):
        left_planner.close()
        right_planner.close()
        return -1, -1

    for _ in range(num_circles):
        for i in range(points_per_circle):
            angle = 2 * np.pi * i / points_per_circle
            stir_point = sapien.Pose(
                p=np.array(
                    [
                        bowl_center[0] + stir_radius * np.cos(angle),
                        bowl_center[1] + stir_radius * np.sin(angle),
                        stir_height,
                    ]
                ),
                q=vertical_orientation,
            )
            if not _safe_move(stir_point):
                left_planner.close()
                right_planner.close()
                return -1, -1

    if not _safe_move(center_pose):
        left_planner.close()
        right_planner.close()
        return -1, -1
    if not _safe_move(above_bowl_pose):
        left_planner.close()
        right_planner.close()
        return -1, -1

    mug_center = env.mug.pose.sp.p
    above_mug_pose = sapien.Pose(
        p=np.array([mug_center[0], mug_center[1], mug_center[2] + 0.4]),
        q=vertical_orientation,
    )
    if not _safe_move(above_mug_pose):
        left_planner.close()
        right_planner.close()
        return -1, -1

    return_pose = sapien.Pose(
        p=np.array([tool_initial_position[0], tool_initial_position[1], mug_center[2] + 0.19]),
        q=vertical_orientation,
    )
    if not _safe_move(return_pose):
        left_planner.close()
        right_planner.close()
        return -1, -1

    left_res = left_planner.open_gripper(other_gripper_state=right_state)

    if not _safe_move(above_mug_pose):
        left_planner.close()
        right_planner.close()
        return -1, -1
    if not _safe_move(left_init_pose):
        left_planner.close()
        right_planner.close()
        return -1, -1

    left_planner.close()
    right_planner.close()
    return left_res, left_res
