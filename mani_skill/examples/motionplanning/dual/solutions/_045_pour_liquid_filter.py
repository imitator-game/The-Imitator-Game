import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPourLiquidCupEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    is_lr_mirror_enabled,
    mirror_offset_pose,
    mirror_pose,
)

def solve(env: TwoRobotPourLiquidCupEnv, seed=None, debug=False, vis=False):
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
    if is_lr_mirror_enabled():
        # Scene positions are already mirrored by env; only mirror grasp orientation here.
        mirrored_grasp_pose = mirror_pose(grasp_pose, mode="full")
        grasp_pose = sapien.Pose(grasp_pose.p, mirrored_grasp_pose.q)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose1 = grasp_pose * mirror_offset_pose(
        sapien.Pose([-0.15, 0.1, -0.1], euler2quat(0, 0, np.pi / 4))
    )
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    grasp_target_pose = grasp_pose * mirror_offset_pose(
        sapien.Pose([0, 0.02, -0.1], euler2quat(np.pi / 4, np.pi / 2, 0))
    )
    # Pre-grasp in world frame: move 5 cm along Y from the final grasp target.
    # In mirrored L2/L3, flip the sign so the pre-grasp offset mirrors with the scene.
    pre_grasp_world_y = 0.05 if is_lr_mirror_enabled() else -0.05
    grasp_pre_pose = sapien.Pose(
        p=np.array(grasp_target_pose.p) + np.array([0.0, pre_grasp_world_y, 0.0], dtype=np.float32),
        q=grasp_target_pose.q,
    )
    left_planner.move_to_pose_with_screw(
        grasp_pre_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.move_to_pose_with_screw(
        grasp_target_pose,
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Pour
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.cup.pose.sp.p, grasp_pose.q) * mirror_offset_pose(
        sapien.Pose(p=[-0.11, 0., -0.2], q=euler2quat(np.pi / 2, 0, 0))
    )
    left_planner.move_to_pose_with_screw(goal_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(
        grasp_pose * mirror_offset_pose(
            sapien.Pose([0, 0.02, -0.11], euler2quat(np.pi / 4, np.pi / 2, 0))
        ),
        other_gripper_state=right_planner.gripper_state,
    )
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
