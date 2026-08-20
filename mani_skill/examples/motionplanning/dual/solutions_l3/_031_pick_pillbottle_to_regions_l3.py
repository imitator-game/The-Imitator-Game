import numpy as np
import sapien
import sapien.physx as physx
import trimesh
from gymnasium import spaces

from mani_skill.envs.tasks import TwoRobotPickPillToRegionsEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb
from mani_skill.examples.motionplanning.panda.utils import get_actor_obb
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh

OPEN = 1

SETTLE_STEPS = 10

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
    env: TwoRobotPickPillToRegionsEnv,
    planner: PandaArmMotionPlanningSolver,
    other_planner: PandaArmMotionPlanningSolver,
    agent_id: int,
    obj,
    place_center_xyz: np.ndarray,
):
    env = env.unwrapped
    agent = env.left_agent if agent_id == 0 else env.right_agent

    approaching = np.array([0, 0, -1])
    # Extract current TCP closing direction and make it orthogonal to approaching
    target_closing = env.right_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # Gram-Schmidt orthogonalization
    target_closing = target_closing - (target_closing @ approaching) * approaching
    target_closing = target_closing / np.linalg.norm(target_closing)

    grasp_pose = agent.build_grasp_pose(approaching, target_closing, obj.pose.sp.p) * sapien.Pose(p=[0, 0, -0.13])
    pre_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.08])
    above_grasp_pose = grasp_pose * sapien.Pose([0, 0, -0.12])

    place_pose = sapien.Pose(p=place_center_xyz + np.array([0.0, 0.0, 0.22]), q=grasp_pose.q)
    above_place_pose = sapien.Pose(p=place_pose.p + np.array([0.0, 0.0, 0.1]), q=grasp_pose.q)
    end_pose = sapien.Pose(p=place_pose.p + np.array([0.0, 0.0, 0.1]), q=place_pose.q)

    planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(grasp_pose, other_gripper_state=other_planner.gripper_state)
    planner.close_gripper(other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(above_grasp_pose, other_gripper_state=other_planner.gripper_state)

    planner.move_to_pose_with_screw(above_place_pose, other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(place_pose, other_gripper_state=other_planner.gripper_state)
    planner.open_gripper(other_gripper_state=other_planner.gripper_state)
    planner.move_to_pose_with_screw(end_pose, other_gripper_state=other_planner.gripper_state)


def solve(env: TwoRobotPickPillToRegionsEnv, seed=None, debug: bool = False, vis: bool = False):
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
    region_a_center = (sapien.Pose(env.displaystand_top.pose.sp.p) * sapien.Pose([0.0, -0.1, 0.0])).p
    region_b_center = (sapien.Pose(env.displaystand_top.pose.sp.p) * sapien.Pose([0.0, 0.1, 0.0])).p

    left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)

    _pick_and_place(env, left_planner, right_planner, agent_id=0, obj=env.pill_a, place_center_xyz=region_a_center + np.array([0.0, 0.0, 0.002]))
    left_planner.move_to_pose_with_screw(left_init_pose * sapien.Pose([0.0, 0.0, -0.2]), other_gripper_state=right_planner.gripper_state)
    left_planner.move_to_pose_with_screw(left_init_pose, other_gripper_state=right_planner.gripper_state)
    _pick_and_place(env, right_planner, left_planner, agent_id=1, obj=env.pill_b, place_center_xyz=region_b_center)
    right_planner.move_to_pose_with_screw(right_init_pose * sapien.Pose([0.0, 0.0, -0.2]), other_gripper_state=left_planner.gripper_state)
    right_planner.move_to_pose_with_screw(right_init_pose, other_gripper_state=left_planner.gripper_state)

    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    left_res = left_planner.open_gripper(other_gripper_state=right_planner.gripper_state)
    return left_res, right_res
