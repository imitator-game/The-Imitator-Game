import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPourKetchupFriesEnv
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (is_lr_mirror_enabled, mirror_pose,
                                                              is_l2_enabled)
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

def solve(env: TwoRobotPourKetchupFriesEnv, seed=None, debug=False, vis=False):
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

    def _maybe_local_flip_180(pose: sapien.Pose) -> sapien.Pose:
        if not lr_mirror:
            return pose
        # Flip gripper in the pose's local frame.
        return pose * sapien.Pose(q=euler2quat(0, 0, np.pi))

    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose
    obb = get_actor_obb(env.ketchup)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # if is_lr_mirror_enabled():
    #     target_closing = mirror_xyz(target_closing)
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=None,
        depth=FINGER_LENGTH,
    )
    if is_l2_enabled():
        closing, center = -grasp_info["closing"], grasp_info["center"]
    else:
        closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.left_agent.build_grasp_pose(approaching, closing, env.ketchup.pose.sp.p)
    if lr_mirror:
        mirrored_grasp_pose = mirror_pose(grasp_pose, mode="full")
        grasp_pose = sapien.Pose(grasp_pose.p, mirrored_grasp_pose.q)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose1 = grasp_pose * sapien.Pose([-0.1, 0., -0.1], q=euler2quat(0, np.pi / 2, 0))
    reach_pose1 = _maybe_local_flip_180(reach_pose1)
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    grasp_rot_y = np.pi / 2 if lr_mirror else np.pi / 2
    grasp_target_pose = grasp_pose * sapien.Pose(q=euler2quat(0, grasp_rot_y, 0))
    grasp_target_pose = _maybe_local_flip_180(grasp_target_pose)
    # if is_lr_mirror_enabled():
    #     # Mirror mode only: add a local pre-grasp by retreating along the grasp target's local z-axis.
    #     pre_grasp_pose = grasp_target_pose * sapien.Pose([-0.2, 0, 0])
    #     left_planner.move_to_pose_with_screw(
    #         pre_grasp_pose,
    #         other_gripper_state=right_planner.gripper_state,
    #     )
    left_planner.move_to_pose_with_screw(
        grasp_target_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Pour
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.fries.pose.sp.p, grasp_pose.q) * sapien.Pose(
        p=[-0.1, 0, -0.18], q=euler2quat(0, np.pi / 8, 0)
    )
    goal_pose = _maybe_local_flip_180(goal_pose)
    left_planner.move_to_pose_with_screw(goal_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(
        grasp_target_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
