import numpy as np
import sapien
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat

from mani_skill.envs.tasks import TwoRobotPourLiquidMugEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    is_lr_mirror_enabled,
    mirror_offset_pose,
    mirror_pose,
)


def _dbg_pose(tag: str, pose: sapien.Pose):
    p = pose.p
    q = pose.q
    if hasattr(p, "detach"):
        p = p.detach().cpu().numpy()
    else:
        p = np.array(p, dtype=np.float32)
    if hasattr(q, "detach"):
        q = q.detach().cpu().numpy()
    else:
        q = np.array(q, dtype=np.float32)
    if p.ndim > 1:
        p = p[0]
    if q.ndim > 1:
        q = q[0]
    p = np.array(p, dtype=np.float32)
    q = np.array(q, dtype=np.float32)
    e = quat2euler(q)
    r = quat2mat(q)


def _dbg_move(label, planner, pose, other_gripper_state=None, do_debug=False):
    if do_debug:
        _dbg_pose(label, pose)
        dry_res = planner.move_to_pose_with_screw(
            pose, dry_run=True, other_gripper_state=other_gripper_state
        )
    res = planner.move_to_pose_with_screw(
        pose, other_gripper_state=other_gripper_state
    )
    return res

def solve(env: TwoRobotPourLiquidMugEnvL3, seed=None, debug=False, vis=False):
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
    do_debug = False

    def _maybe_local_flip_180(pose: sapien.Pose) -> sapien.Pose:
        if not lr_mirror:
            return pose
        return pose * sapien.Pose(q=euler2quat(0, 0, np.pi))

    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose
    obb = get_actor_obb(env.liquid)

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
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.liquid.pose.sp.p)
    if lr_mirror:
        # Scene positions are already mirrored in env; only mirror grasp orientation here.
        mirrored_grasp_pose = mirror_pose(grasp_pose, mode="full")
        grasp_pose = sapien.Pose(grasp_pose.p, mirrored_grasp_pose.q)
    reach_pose1 = grasp_pose * mirror_offset_pose(
        sapien.Pose([0, 0.07, -0.12], euler2quat(np.pi / 2, np.pi / 2, 0))
    )

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    _dbg_move(
        "reach_pose1",
        left_planner,
        reach_pose1,
        other_gripper_state=right_planner.gripper_state,
        do_debug=do_debug,
    )

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    grasp_pre_pose = grasp_pose * mirror_offset_pose(
        sapien.Pose([0, -0.03, -0.12], euler2quat(np.pi / 2, np.pi / 2, 0))
    )
    grasp_pre_pose = _maybe_local_flip_180(grasp_pre_pose)
    if lr_mirror:
        # Mirror mode only: add a pre-pre-grasp by retreating along grasp_pre_pose local z-axis.
        pre_pre_grasp_pose = grasp_pre_pose * sapien.Pose([0.0, 0.0, -0.1])
        _dbg_move(
            "pre_pre_grasp_pose",
            left_planner,
            pre_pre_grasp_pose,
            other_gripper_state=right_planner.gripper_state,
            do_debug=do_debug,
        )
    _dbg_move(
        "grasp_pre_pose",
        left_planner,
        grasp_pre_pose,
        other_gripper_state=right_planner.gripper_state,
        do_debug=do_debug,
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    lift_pose = grasp_pose * mirror_offset_pose(
        sapien.Pose([0, -0.03, -0.3], euler2quat(np.pi / 2, np.pi / 2, 0))
    )
    lift_pose = _maybe_local_flip_180(lift_pose)
    _dbg_move(
        "lift_pose",
        left_planner,
        lift_pose,
        other_gripper_state=right_planner.gripper_state,
        do_debug=do_debug,
    )

    # -------------------------------------------------------------------------- #
    # Pour
    # -------------------------------------------------------------------------- #
    reach_pose2 = sapien.Pose(env.cup.pose.sp.p, grasp_pose.q) * mirror_offset_pose(
        sapien.Pose(p=[-0.1, 0., -0.3], q=euler2quat(np.pi / 3, 0, 0))
    )
    reach_pose2 = _maybe_local_flip_180(reach_pose2)
    if do_debug:
        _dbg_pose("reach_pose2", reach_pose2)
    _dbg_move(
        "reach_pose2",
        left_planner,
        reach_pose2,
        other_gripper_state=right_planner.gripper_state,
        do_debug=do_debug,
    )
    # goal_pose = sapien.Pose(env.cup.pose.sp.p, grasp_pose.q) * mirror_offset_pose(
    #     sapien.Pose(p=[-0.1, 0., -0.15], q=euler2quat(np.pi / 2, 0, 0))
    # )
    # goal_pose = _maybe_local_flip_180(goal_pose)
    # if do_debug:
    #     _dbg_pose("goal_pose", goal_pose)
    # _dbg_move(
    #     "goal_pose",
    #     left_planner,
    #     goal_pose,
    #     other_gripper_state=right_planner.gripper_state,
    #     do_debug=do_debug,
    # )
    # _dbg_move(
    #     "reach_pose2_return",
    #     left_planner,
    #     reach_pose2,
    #     other_gripper_state=right_planner.gripper_state,
    #     do_debug=do_debug,
    # )
    _dbg_move(
        "lift_pose_return",
        left_planner,
        lift_pose,
        other_gripper_state=right_planner.gripper_state,
        do_debug=do_debug,
    )
    _dbg_move(
        "grasp_pre_pose_return",
        left_planner,
        grasp_pre_pose,
        other_gripper_state=right_planner.gripper_state,
        do_debug=do_debug,
    )
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    _dbg_move(
        "reach_pose1_return",
        left_planner,
        reach_pose1,
        other_gripper_state=right_planner.gripper_state,
        do_debug=do_debug,
    )

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
