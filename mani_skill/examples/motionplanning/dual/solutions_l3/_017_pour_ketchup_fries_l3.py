import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import TwoRobotPourKetchupFriesEnvL3
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import is_lr_mirror_enabled
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver


def solve(env: TwoRobotPourKetchupFriesEnvL3, seed=None, debug=False, vis=False):
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

    grasp_pose = sapien.Pose(env.dispenser.pose.sp.p, env.left_agent.tcp.pose.sp.q) * sapien.Pose([0., 0., -0.1])
    reach_pose = sapien.Pose(env.dispenser.pose.sp.p, env.left_agent.tcp.pose.sp.q) * sapien.Pose([0., 0., -0.2])

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    left_planner.close_gripper(other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(reach_pose, other_gripper_state=right_planner.gripper_state)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    left_res = left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    return left_res, right_res
