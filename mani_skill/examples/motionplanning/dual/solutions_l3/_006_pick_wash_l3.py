import numpy as np
import sapien
import sapien.physx as physx
import trimesh

from mani_skill.envs.tasks import TwoRobotPickWashEnvL3
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh


def _move_to_pose_with_rrt_dual(
    planner: PandaArmMotionPlanningSolver,
    pose: sapien.Pose,
    other_gripper_state,
):
    result = planner.move_to_pose_with_RRTConnect(pose, dry_run=True)
    if result == -1:
        return -1
    return planner.follow_path(result, other_gripper_state=other_gripper_state)


def _get_obb(obj):
    # Actor path.
    try:
        return get_actor_obb(obj)
    except Exception:
        pass

    # Articulation path: aggregate meshes from all rigid links.
    meshes = []
    if hasattr(obj, "links"):
        for link in obj.links:
            if not hasattr(link, "_objs") or len(link._objs) == 0:
                continue
            for comp in link._objs[0].entity.components:
                if isinstance(comp, physx.PhysxRigidBodyComponent):
                    mesh = get_component_mesh(comp, to_world_frame=True)
                    if mesh is not None:
                        meshes.append(mesh)

    if len(meshes) == 0:
        raise RuntimeError(f"can not get mesh for object {obj}")

    combined_mesh = trimesh.util.concatenate(meshes)
    return combined_mesh.bounding_box_oriented


def solve(env: TwoRobotPickWashEnvL3, seed=None, debug=False, vis=False):
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
    right_state = right_planner.gripper_state

    source_obj = env.bottle

    # Use fixed grasp orientation to avoid 180-degree yaw flip before grasping.
    source_p = np.array(source_obj.pose.sp.p).reshape(-1, 3)[0]
    fixed_q = np.array(left_init_pose.q[0])
    grasp_pose = sapien.Pose(p=source_p, q=fixed_q) * sapien.Pose([0, 0, -0.05])
    grasp_target_pose = grasp_pose * sapien.Pose([0, 0, -0.1])

    place_center = env.box.pose.sp.p
    goal_pose = sapien.Pose(place_center + np.array([0, 0, 0.06], dtype=np.float32), q=grasp_pose.q)

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.2])
    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.3])
    pre_place_pose = goal_pose * sapien.Pose([0, 0, -0.3])
    place_pose = goal_pose * sapien.Pose([0, 0, -0.15])

    left_res = _move_to_pose_with_rrt_dual(left_planner, reach_pose, right_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, grasp_target_pose, right_state)
    if left_res == -1:
        return -1, -1
    left_planner.close_gripper(other_gripper_state=right_state)

    left_res = _move_to_pose_with_rrt_dual(left_planner, lift_pose, right_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, pre_place_pose, right_state)
    if left_res == -1:
        return -1, -1
    left_res = _move_to_pose_with_rrt_dual(left_planner, place_pose, right_state)
    if left_res == -1:
        return -1, -1
    left_res = left_planner.open_gripper(other_gripper_state=right_state)

    left_res = _move_to_pose_with_rrt_dual(left_planner, pre_place_pose, right_state)
    if left_res == -1:
        return -1, -1
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    left_res = _move_to_pose_with_rrt_dual(left_planner, left_init_pose, right_state)
    if left_res == -1:
        return -1, -1

    left_planner.close()
    right_planner.close()
    return left_res, right_res
