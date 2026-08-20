import numpy as np
import sapien
import sapien.physx as physx
import trimesh
from gymnasium import spaces

OPEN = 1

from mani_skill.envs.tasks import TwoRobotPickAppleToScaleEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

SETTLE_STEPS = 10

PLATE_ON_SCALE_OFFSET_Y = 0.1
PLATE_ON_SCALE_OFFSET_Z = 0.07

def _get_obb(obj):
    if hasattr(obj, "_objs"):
        return get_actor_obb(obj)
    component = obj.find_component_by_type(physx.PhysxRigidDynamicComponent)
    mesh = get_component_mesh(component, to_world_frame=True)
    if mesh is None:
        raise RuntimeError(f"can not get actor mesh for {obj}")
    obb: trimesh.primitives.Box = mesh.bounding_box_oriented
    return obb


# def _zero_action(space):
#     if isinstance(space, spaces.Box):
#         return np.zeros(space.shape, dtype=space.dtype)
#     if isinstance(space, spaces.Dict):
#         return {k: _zero_action(v) for k, v in space.spaces.items()}
#     if isinstance(space, spaces.Tuple):
#         return tuple(_zero_action(s) for s in space.spaces)
#     return 0


def _hold_action(env, gripper_state=OPEN):
    env = env.unwrapped
    left_qpos = env.agent.agents[0].robot.qpos[0][:7].cpu().numpy()
    right_qpos = env.agent.agents[1].robot.qpos[0][:7].cpu().numpy()
    left_action = np.hstack([left_qpos, gripper_state])
    right_action = np.hstack([right_qpos, gripper_state])
    return {
        "panda_wristcam-0": left_action,
        "panda_wristcam-1": right_action,
    }


def _settle_env(env, steps=SETTLE_STEPS):
    hold_action = _hold_action(env, gripper_state=OPEN)
    for _ in range(steps):
        env.step(hold_action)


def _pick_and_place(
    env: TwoRobotPickAppleToScaleEnvL3,
    planner: PandaArmMotionPlanningSolver,
    other_planner: PandaArmMotionPlanningSolver,
    agent_id: int,
    obj,
    place_center_xyz: np.ndarray,
    vis: bool, 
    plate_on_scale_offset_y, 
    plate_on_scale_offset_z, 
):
    env = env.unwrapped
    agent = env.left_agent if agent_id == 0 else env.right_agent

    obb = _get_obb(obj)
    approaching = np.array([0, 0, -1])
    target_closing = agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=0.025,
    )
    closing = grasp_info["closing"]
    grasp_pose = agent.build_grasp_pose(approaching, closing, grasp_info["center"])
    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.08])

    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.3])
    above_scale = sapien.Pose(p=place_center_xyz + np.array([0.02, plate_on_scale_offset_y, 0.28+plate_on_scale_offset_z]), q=grasp_pose.q)
    place_pose = sapien.Pose(p=place_center_xyz + np.array([0.02, plate_on_scale_offset_y, 0.16+plate_on_scale_offset_z]), q=grasp_pose.q)
    pre_place_pose = sapien.Pose(p=place_pose.p + np.array([0.0, 0.0, 0.1]), q = place_pose.q)

    planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=other_planner.gripper_state)
    planner.close_gripper(other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(lift_pose, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(pre_place_pose, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(place_pose, other_gripper_state=other_planner.gripper_state)
    planner.open_gripper(other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(above_scale, other_gripper_state=other_planner.gripper_state)


def solve(env: TwoRobotPickAppleToScaleEnvL3, seed=None, debug: bool = False, vis: bool = False):
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
    right_init_pose = env.right_agent.tcp.pose
    scale_center_xyz = env.scale.pose.sp.p

    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    _pick_and_place(env, left_planner, right_planner, agent_id=0, obj=env.apple, place_center_xyz=scale_center_xyz, vis=vis, plate_on_scale_offset_y=-PLATE_ON_SCALE_OFFSET_Y, plate_on_scale_offset_z=PLATE_ON_SCALE_OFFSET_Z)
    left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)

    _pick_and_place(env, right_planner, left_planner, agent_id=1, obj=env.weight, place_center_xyz=scale_center_xyz, vis=vis, plate_on_scale_offset_y=PLATE_ON_SCALE_OFFSET_Y, plate_on_scale_offset_z=PLATE_ON_SCALE_OFFSET_Z)
    right_planner.move_to_pose_with_screw(right_init_pose, other_gripper_state=left_planner.gripper_state)

    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    return left_res, right_res
