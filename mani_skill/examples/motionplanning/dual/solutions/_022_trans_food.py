import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.dual_tasks._022_trans_food import TwoRobotTransFoodEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.envs.tasks.tabletop.utils.L0_L3_utils import (
    mirror_offset_pose,
    mirror_pose,
    mirror_xyz,
)


def solve(env:TwoRobotTransFoodEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    
    # -------------------------------------------------------------------------- #
    # 1. Initialize the planner (only use left arm, multi_robot_id=0)
    # -------------------------------------------------------------------------- #
    left_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[0].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=0
    )
    # Right arm as a placeholder to avoid errors
    right_planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.agents[1].robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        multi_robot_id=1
    )

    env = env.unwrapped
    # Use the BaseEnv-applied marker so solution mirroring only runs when scene mirroring
    # actually executed in this reset.
    lr_mirror = bool(getattr(env, "_lr_mirror_applied_this_reset", False))

    def _maybe_local_flip_180(pose: sapien.Pose) -> sapien.Pose:
        if not lr_mirror:
            return pose
        # Flip the gripper in each pose's own local frame.
        return pose * sapien.Pose([0, 0, 0], euler2quat(0, 0, np.pi))

    def _maybe_mirror_xyz(vec):
        arr = np.array(vec, dtype=np.float32)
        return mirror_xyz(arr) if lr_mirror else arr
    
    # -------------------------------------------------------------------------- #
    # 2. Determine the objects
    # -------------------------------------------------------------------------- #
    spoon = env.spoon       # the spoon to grasp
    bowl_src = env.bowl_src # the source bowl (right)
    bowl_dst = env.bowl_dst # the target bowl (left)


    # -------------------------------------------------------------------------- #
    # 3. Compute the grasp pose (Pick Spoon)
    # -------------------------------------------------------------------------- #
   
    obb = get_actor_obb(env.spoon, to_world_frame=True, vis=False)

    # Set the grasp approach direction
    # approaching: [0, 0, -1] grasp from top down
    approaching = np.array([0, 0, -1])
    # target_closing: desired closing direction (computed from the left arm's current pose)
    target_closing = env.left_agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=None,
        depth=0.0, # the spoon handle is very thin, set depth to 0 and pinch with fingertips
    )
    closing = grasp_info["closing"]

    # Generate the grasp pose
    # Note: for build_grasp_pose, the center is best set to obj_pos (the spoon center)
    grasp_pose = env.left_agent.build_grasp_pose(
        approaching=grasp_info["approaching"],
        closing=grasp_info["closing"],
        center=grasp_info["center"],
    )*sapien.Pose([-0.01, 0, 0])
    
    # Correct the grasp rotation:
    # The spoon's initial rotation is complex, so to be safe we force the gripper
    # to point straight down (rotate 180 degrees about the Y axis).
    # This maximizes the chance of avoiding the bowl walls.
    #rot_q = euler2quat(-np.pi/3, np.pi, 0) # pointing straight down
    rot_q = euler2quat(0, np.pi, -np.pi/2)
    # Keep the position unchanged, replace the rotation
    grasp_pose = sapien.Pose(p=grasp_pose.p, q=rot_q)
    if lr_mirror:
        # Scene positions are mirrored by env; only mirror grasp orientation here.
        mirrored_grasp_pose = mirror_pose(grasp_pose, mode="full")
        grasp_pose = sapien.Pose(p=grasp_pose.p, q=mirrored_grasp_pose.q)
    
    # Fine-tune the position again:
    # Adjust the grasp point along the spoon handle direction, or raise/lower it.
    # Here we place the grasp point 1cm above the spoon center.
    grasp_pose.set_p(grasp_pose.p + _maybe_mirror_xyz([0.0, -0.05, 0.0]))
    grasp_pose = grasp_pose * mirror_offset_pose(sapien.Pose([0, 0.0, 0]))
    # Pre-grasp point (10cm above)
    #pre_grasp_pose = grasp_pose * sapien.Pose([0, 0.1, 0])
    pre_grasp_pose = sapien.Pose(
        p=grasp_pose.p + _maybe_mirror_xyz([0.0, 0.0, 0.15]),
        q=grasp_pose.q,
    )
    pre_grasp_pose = _maybe_local_flip_180(pre_grasp_pose)
    #pre_grasp_pose.set_p(grasp_pose.p + np.array([0.0, 0.0, 0.1]))
    current_pose = env.left_agent.tcp.pose
    # -------------------------------------------------------------------------- #
    # 4. Compute the transport and dump poses (Transport & Dump)
    # -------------------------------------------------------------------------- #
    # Target position: above the target bowl
    dst_pos = bowl_dst.pose.p.cpu().numpy()[0]
    dst_pos[2] += 0.15 # raise 15cm to avoid hitting the bowl rim
    
    # Transport pose (keep the grasp rotation)
    transport_pose = sapien.Pose(dst_pos, grasp_pose.q)
    transport_pose.set_p(
        transport_pose.p + _maybe_mirror_xyz([0, -0.15, 0.0])
    )
    
    # Dump pose (Dump)
    # Based on the transport pose, rotate the wrist by 45~90 degrees
    # Assuming grasp_pose points straight down, rotate around the gripper's Y axis to simulate pouring
    dump_rot = euler2quat(0, -np.pi/7, 0) # pouring action
    dump_pose = transport_pose * mirror_offset_pose(sapien.Pose([0, 0, 0], dump_rot))
    dump_pose = _maybe_local_flip_180(dump_pose)

    # -------------------------------------------------------------------------- #
    # 5. Compute the return poses (Return)
    # -------------------------------------------------------------------------- #
    # Source bowl position
    src_pos = bowl_src.pose.p.cpu().numpy()[0]
    
    # Hover point for returning (15cm above Source)
    return_hover_pose = sapien.Pose(
        src_pos + _maybe_mirror_xyz([0, -0.15, 0.15]),
        grasp_pose.q,
    )
    return_hover_pose = _maybe_local_flip_180(return_hover_pose)
    
    # Drop point for returning (inside/5cm above Source)
    # Release slightly higher than the grasp point so it falls in
    return_drop_pose = sapien.Pose(
        src_pos + _maybe_mirror_xyz([0, -0.12, 0.08]),
        grasp_pose.q,
    )
    return_drop_pose = _maybe_local_flip_180(return_drop_pose)

    return_last_pose = sapien.Pose(
        src_pos + _maybe_mirror_xyz([0, -0.15, 0.2]),
        grasp_pose.q,
    )
    return_last_pose = _maybe_local_flip_180(return_last_pose)

    # -------------------------------------------------------------------------- #
    # 6. Execute the action sequence
    # -------------------------------------------------------------------------- #
    right_state = right_planner.gripper_state

    # === Phase 1: Grasp the spoon ===
    # 1.1 Move to the pre-grasp point
    left_planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=right_state)
    
    # 1.2 Descend to the grasp point
    left_planner.move_to_pose_with_screw(
        _maybe_local_flip_180(grasp_pose), other_gripper_state=right_state
    )
    
    # 1.3 Close the gripper
    left_planner.close_gripper(other_gripper_state=right_state)
    
    # 1.4 Lift up (back to the pre-grasp height)
    left_planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=right_state)

    # === Phase 2: Transport and dump ===
    # 2.1 Translate to above the target bowl
    # If the distance is large, use RRT planning to avoid hitting obstacles in between
    transport_exec_pose = _maybe_local_flip_180(transport_pose)
    res = left_planner.move_to_pose_with_screw(transport_exec_pose, other_gripper_state=right_state)
    if res == -1:
        left_planner.move_to_pose(transport_exec_pose, other_gripper_state=right_state)
    
    # 2.2 Perform the dump action (rotate the wrist)
    # This is an in-place rotation action
    left_planner.move_to_pose_with_screw(dump_pose, other_gripper_state=right_state)
    
    # [Optional] Shake or pause briefly to make sure the food falls out
    # Here we simply rotate back
    left_planner.move_to_pose_with_screw(transport_exec_pose, other_gripper_state=right_state)

    # === Phase 3: Return the spoon ===
    # 3.1 Move back above the source bowl
    res = left_planner.move_to_pose_with_screw(pre_grasp_pose, other_gripper_state=right_state)
    if res == -1:
        left_planner.move_to_pose(pre_grasp_pose, other_gripper_state=right_state)
    
    # 3.2 Descend and prepare to place
    left_planner.move_to_pose_with_screw(
        _maybe_local_flip_180(grasp_pose), other_gripper_state=right_state
    )
    
    # 3.3 Open the gripper
    left_planner.open_gripper(other_gripper_state=right_state)
    
    # Keep the return format consistent
    right_res = right_planner.open_gripper(other_gripper_state=left_planner.gripper_state)
    
    # 3.4 Lift up and leave (task complete)
    left_planner.move_to_pose_with_screw(return_last_pose, other_gripper_state=right_state)

    # Return to the initial pose (optional)
    left_res = left_planner.move_to_pose_with_screw(current_pose, other_gripper_state=right_state)

    left_planner.close()
    right_planner.close()
    
    return left_res, right_res
