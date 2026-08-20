import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotGrindFoodEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    is_lr_mirror_enabled,
    mirror_offset_pose,
    mirror_pose,
)

def solve(env: TwoRobotGrindFoodEnv, seed=None, debug=False, vis=False):
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

    obb = get_actor_obb(env.grind)

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
    lr_mirror = is_lr_mirror_enabled()

    def _maybe_mirror_wrist_flip_180(pose: sapien.Pose) -> sapien.Pose:
        if not lr_mirror:
            return pose
        return pose * sapien.Pose([0, 0, 0], euler2quat(0, 0, np.pi))

    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.grind.pose.sp.p)
    if lr_mirror:
        # Scene positions are already mirrored by env; only mirror grasp orientation here.
        mirrored_grasp_pose = mirror_pose(grasp_pose, mode="full")
        grasp_pose = sapien.Pose(grasp_pose.p, mirrored_grasp_pose.q)
    goal_pose = sapien.Pose(env.bowl.pose.sp.p, grasp_pose.q) * mirror_offset_pose(sapien.Pose([0, 0, -0.05]))

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    if lr_mirror:
        # Do not use mirror_offset_pose here for rotation, otherwise mirror_quat will flip
        # yaw sign again and cancel the explicit -pi/2 -> +pi/2 change.
        reach_offset = sapien.Pose([0, -0.15, -0.15], euler2quat(0, 0, np.pi / 2))
    else:
        reach_offset = sapien.Pose([0, 0.15, -0.15], euler2quat(0, 0, -np.pi / 2))
    reach_pose1 = _maybe_mirror_wrist_flip_180(grasp_pose * reach_offset)
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(
        _maybe_mirror_wrist_flip_180(
            grasp_pose * mirror_offset_pose(
                sapien.Pose([0.05, -0.05, 0.], q=euler2quat(np.pi / 2, np.pi / 2, 0))
            )
        ),
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(
        _maybe_mirror_wrist_flip_180(
            grasp_pose * mirror_offset_pose(
                sapien.Pose([0.05, -0.05, -0.15], q=euler2quat(np.pi / 2, np.pi / 2, 0))
            )
        ),
        other_gripper_state=right_planner.gripper_state,
    )

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    reach_pose3 = _maybe_mirror_wrist_flip_180(
        goal_pose * mirror_offset_pose(
            sapien.Pose([0.0, -0.02, -0.25], q=euler2quat(np.pi / 2, np.pi / 2, 0))
        )
    )
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(
        _maybe_mirror_wrist_flip_180(
            goal_pose * mirror_offset_pose(
                sapien.Pose([0.0, -0.02, -0.05], q=euler2quat(np.pi / 2, np.pi / 2, 0))
            )
        ),
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(
        _maybe_mirror_wrist_flip_180(
            goal_pose * mirror_offset_pose(
                sapien.Pose([0.0, -0.02, -0.05], q=euler2quat(np.pi / 2, np.pi / 2, 0))
            )
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
