import numpy as np
import sapien

from mani_skill.envs.tasks import TwoRobotPutCubeOnScaleEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb


def _pick_and_place(
    env: TwoRobotPutCubeOnScaleEnvL3,
    planner: PandaArmMotionPlanningSolver,
    other_planner: PandaArmMotionPlanningSolver,
    agent_id: int,
    obj,
    place_center_xyz: np.ndarray,
    place_offset_xyz: np.ndarray,
):
    env = env.unwrapped
    agent = env.left_agent if agent_id == 0 else env.right_agent

    obb = get_actor_obb(obj)
    approaching = np.array([0, 0, -1])
    target_closing = agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.025,
    )
    grasp_pose = agent.build_grasp_pose(approaching, grasp_info["closing"], grasp_info["center"])

    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.08])
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
    above_offset_xyz = place_offset_xyz + np.array([0.0, 0.0, 0.1], dtype=np.float32)
    above_scale = sapien.Pose(
        p=place_center_xyz + above_offset_xyz,
        q=grasp_pose.q,
    )
    place_pose = sapien.Pose(
        p=place_center_xyz + place_offset_xyz,
        q=grasp_pose.q,
    )

    planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=other_planner.gripper_state)
    planner.close_gripper(other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(lift_pose, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(above_scale, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(place_pose, other_gripper_state=other_planner.gripper_state)
    planner.open_gripper(other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(above_scale, other_gripper_state=other_planner.gripper_state)


def solve(env: TwoRobotPutCubeOnScaleEnvL3, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
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
    right_init_pose = env.right_agent.tcp.pose
    scale_center_xyz = env.scale.pose.sp.p

    # Place tuning knobs in world frame (meters), aligned with the env's
    # left/right pan goal regions after accounting for grasp geometry.
    left_place_offset_xyz = np.array([0.02, -0.06, 0.22], dtype=np.float32)
    right_place_offset_xyz = np.array([0.02, 0.06, 0.22], dtype=np.float32)
    # Fine-tune deltas (edit these for quick adjustment).
    left_place_delta = np.array([0.0, -0.04, 0.0], dtype=np.float32)
    right_place_delta = np.array([0.0, 0.05, 0.0], dtype=np.float32)

    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    _pick_and_place(
        env,
        left_planner,
        right_planner,
        agent_id=0,
        obj=env.cube,
        place_center_xyz=scale_center_xyz,
        place_offset_xyz=left_place_offset_xyz + left_place_delta,
    )
    left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)

    _pick_and_place(
        env,
        right_planner,
        left_planner,
        agent_id=1,
        obj=env.rubik,
        place_center_xyz=scale_center_xyz,
        place_offset_xyz=right_place_offset_xyz + right_place_delta,
    )
    right_planner.move_to_pose_with_screw(right_init_pose, other_gripper_state=left_planner.gripper_state)

    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    return left_res, right_res
