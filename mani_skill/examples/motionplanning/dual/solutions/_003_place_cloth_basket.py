import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPlaceClothBasketEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import mirror_offset_pose
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import is_l2_enabled

def solve(env: TwoRobotPlaceClothBasketEnv, seed=None, debug=False, vis=False):
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

    base_cloth = sapien.Pose(env.cloth.pose.sp.p, np.array(left_init_pose.q[0]))
    base_basket = sapien.Pose(env.basket.pose.sp.p, np.array(left_init_pose.q[0]))

    reach_pose1 = base_cloth * mirror_offset_pose(
        sapien.Pose(p=[0.05, 0.2, -0.15], q=euler2quat(0, 0, 0))
    )
    grasp_pose = base_cloth * mirror_offset_pose(
        sapien.Pose(p=[-0.05, 0.07, 0.0], q=euler2quat(np.pi / 2, 0, 0))
    )
    reach_pose2 = grasp_pose * mirror_offset_pose(sapien.Pose([0, -0.15, 0]))
    reach_pose3 = base_basket * mirror_offset_pose(
        sapien.Pose(p=[0, 0.15, -0.2], q=euler2quat(np.pi / 2, 0, 0))
    )
    goal_pose = base_basket * mirror_offset_pose(
        sapien.Pose(p=[0, -0.02, -0.2], q=euler2quat(np.pi / 3, 0, 0))
    )
    reach_pose4 = base_basket * mirror_offset_pose(
        sapien.Pose(p=[0, 0.15, -0.3], q=euler2quat(np.pi / 4, 0, 0))
    )

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose1, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    
    if is_l2_enabled():
        grasp_target_pose = grasp_pose * sapien.Pose([0.0, 0.0, 0.0])
    else: 
        grasp_target_pose = grasp_pose
    left_planner.move_to_pose_with_screw(grasp_target_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    left_planner.move_to_pose_with_screw(reach_pose2, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose3, other_gripper_state=right_planner.gripper_state)

    if is_l2_enabled(): 
        goal_target_pose = goal_pose * sapien.Pose([0, 0.0, -0.05])
    else: 
        goal_target_pose = goal_pose * sapien.Pose([0, 0.02, -0.05])
    left_planner.move_to_pose_with_screw(goal_target_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose4, other_gripper_state=right_planner.gripper_state)
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
