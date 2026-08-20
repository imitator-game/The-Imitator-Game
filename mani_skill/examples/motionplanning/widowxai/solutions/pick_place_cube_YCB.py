import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.utils import common
from mani_skill.examples.motionplanning.widowxai.motionplanner import WidowXAIArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb


def solve(env, seed=None, debug=False, vis=False):
    """
    Solution for PickCubeYCB task using WidowXAI robot and motion planning.
    
    Task: Pick up a cube and place it inside a YCB plate.
    """
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel", 
    ], f"Unsupported control mode: {env.unwrapped.control_mode}"
    
    # Initialize motion planner for WidowXAI
    planner = WidowXAIArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.8,  # Slightly faster than before (was 0.7)
        joint_acc_limits=0.6,  # Moderate acceleration (was 0.4)
    )
    
    # Get environment objects
    env = env.unwrapped
    cube = env.cube  # The cube to pick up
    plate = env.container  # The YCB plate to place cube in
    
    # WidowXAI gripper parameters
    FINGER_LENGTH = 0.02  # WidowXAI has smaller fingers than Panda
    
    # -------------------------------------------------------------------------- #
    # Phase 1: Simple grasp pose for WidowXAI
    # -------------------------------------------------------------------------- #
    # Use a simple top-down grasp instead of complex OBB computation
    cube_pos = cube.pose.p.cpu().numpy()[0]  # Get cube position
    
    # Simple top-down grasp pose with proper wrist orientation
    from transforms3d.euler import euler2quat
    # Use euler angles to define a more controlled orientation
    # Roll=0, Pitch=90deg (pointing down), Yaw=0 for straight wrist
    grasp_quat = euler2quat(0, np.pi/2, 0, 'rxyz')  # More controlled rotation
    grasp_pose = sapien.Pose(
        p=cube_pos + np.array([0, 0, 0.1]),  # 10cm above cube
        q=grasp_quat  # Controlled downward orientation
    )

    # Like Panda - just use the computed grasp pose directly
    
    # -------------------------------------------------------------------------- #
    # Reach (go to grasp position)
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)

    # -------------------------------------------------------------------------- #
    # Lower to actually contact the cube
    # -------------------------------------------------------------------------- #
    contact_pose = sapien.Pose(
        p=cube_pos + np.array([0, 0, -0.03]),  # Go deeper into cube for secure grasp
        q=grasp_quat  # Same controlled orientation
    )
    planner.move_to_pose_with_screw(contact_pose)
    
    # -------------------------------------------------------------------------- #
    # Close gripper to grasp
    # -------------------------------------------------------------------------- #
    planner.close_gripper()
    
    # -------------------------------------------------------------------------- #
    # Lift the cube
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose(
        p=cube_pos + np.array([0, 0, 0.1]),  # Lift back up
        q=grasp_quat  # Same controlled orientation
    )
    planner.move_to_pose_with_screw(lift_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose (place in plate)
    # -------------------------------------------------------------------------- #
    # Get plate center position and place cube slightly above it
    plate_pos = plate.pose.p.cpu().numpy()[0]  # Convert tensor to numpy
    
    # Move above plate first
    above_plate_pose = sapien.Pose(
        p=plate_pos + np.array([0, 0, 0.1]),  # 10cm above plate center
        q=grasp_quat  # Same controlled orientation
    )
    planner.move_to_pose_with_screw(above_plate_pose)
    
    # Lower to place cube in plate
    place_pose = sapien.Pose(
        p=plate_pos + np.array([0, 0, 0.03]),  # Just above plate surface
        q=grasp_quat  # Same controlled orientation
    )
    res = planner.move_to_pose_with_screw(place_pose)
    
    # Release cube
    planner.open_gripper()
    
    planner.close()
    return res
