import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPourLiquidCupEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    get_actor_obb)
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
    lr_mirror = is_lr_mirror_enabled()

    left_init_pose = env.left_agent.tcp.pose
    right_init_pose = env.right_agent.tcp.pose
    obb = get_actor_obb(env.liquid)

    approaching = np.array([0, 0, -1])
    # Extract current TCP closing direction and make it orthogonal to approaching
    target_closing = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # Gram-Schmidt orthogonalization
    target_closing = target_closing - (target_closing @ approaching) * approaching
    target_closing = target_closing / np.linalg.norm(target_closing)

    grasp_pose = env.left_agent.build_grasp_pose(approaching, target_closing, env.liquid.pose.sp.p)
    if lr_mirror:
        mirrored_grasp_pose = mirror_pose(grasp_pose, mode="full")
        grasp_pose = sapien.Pose(grasp_pose.p, mirrored_grasp_pose.q)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose1 = grasp_pose * mirror_offset_pose(sapien.Pose([0, 0.07, -0.15], euler2quat(0, np.pi/4, -np.pi / 4)))
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    pre_grasp_offset = mirror_offset_pose(sapien.Pose([0, -0.02, -0.08], euler2quat(0, np.pi/4, -np.pi/4)))
    pre_grasp_pose = grasp_pose * pre_grasp_offset
    left_planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Pour
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.cup.pose.sp.p, pre_grasp_pose.q) * mirror_offset_pose(sapien.Pose(p=[0.05, 0., -0.15], q=euler2quat(0, -np.pi/3, 0)))
    left_planner.move_to_pose_with_screw(goal_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    retreat_pose = grasp_pose * mirror_offset_pose(sapien.Pose([0, 0.05, -0.15], euler2quat(0, np.pi/4, -np.pi/4)))
    left_planner.move_to_pose_with_screw(retreat_pose, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
