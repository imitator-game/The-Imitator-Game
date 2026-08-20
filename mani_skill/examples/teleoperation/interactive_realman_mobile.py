import gymnasium as gym
import numpy as np
import sapien.core as sapien
import sapien.utils.viewer

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import Pose

import tyro
from dataclasses import dataclass
import time


@dataclass
class Args:
    env_id: str = "RoboCasaKitchen-v1"
    """Environment ID."""
    obs_mode: str = "none"
    """Observation mode."""
    control_mode: str = "pd_joint_pos"
    """Control mode for the agent."""
    robot_uid: str = "realman_mobile_base"
    """Robot UID to use realman with mobile base."""
    render_mode: str = "human"
    """Render mode."""
    shader: str = "rt-fast"
    """Shader for rendering."""
    seed: int = 42
    """Random seed."""


def main(args: Args):
    np.set_printoptions(suppress=True, precision=3)

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode="none",
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        shader_dir=args.shader,
        robot_uids=args.robot_uid,
    )

    print("Environment created.")
    print("Control mode:", env.unwrapped.control_mode)
    print("Action space:", env.action_space)

    # Get joint information
    active_joints = env.unwrapped.agent.robot.get_active_joints()
    joint_names = [j.get_name() for j in active_joints]
    action_dim = env.action_space.shape[0]

    print(f"\nTotal joints: {len(joint_names)}")
    print(f"Action dimension: {action_dim}")

    # Print all joint names with indices for debugging
    print("\nAll joints:")
    for i, name in enumerate(joint_names):
        print(f"  {i}: {name}")

    # Parse joint mapping based on the actual joint names
    base_x_idx = None
    base_y_idx = None
    base_rz_idx = None
    head_joint_indices = []
    left_arm_indices = []
    right_arm_indices = []
    left_gripper_index = None
    right_gripper_index = None

    # Map joints based on their names
    for i, name in enumerate(joint_names):
        if name == "root_x_axis_joint":
            base_x_idx = i
        elif name == "root_y_axis_joint":
            base_y_idx = i
        elif name == "root_z_rotation_joint":
            base_rz_idx = i
        elif "head_joint" in name:
            head_joint_indices.append(i)
        elif name.startswith("l_joint") and len(name) == 8:  # l_joint1-7
            joint_num = int(name[-1])
            left_arm_indices.append((joint_num, i))
        elif name.startswith("r_joint") and len(name) == 8:  # r_joint1-7
            joint_num = int(name[-1])
            right_arm_indices.append((joint_num, i))
        elif name == "l_gripper_joint1":
            left_gripper_index = i
        elif name == "r_gripper_joint1":
            right_gripper_index = i

    # Sort arm indices by joint number
    left_arm_indices.sort(key=lambda x: x[0])
    right_arm_indices.sort(key=lambda x: x[0])
    left_arm_indices = [idx for _, idx in left_arm_indices]
    right_arm_indices = [idx for _, idx in right_arm_indices]

    print(f"\nParsed Joint Mapping:")
    print(f"Base: X={base_x_idx}, Y={base_y_idx}, Rz={base_rz_idx}")
    print(f"Head: {head_joint_indices}")
    print(f"Left arm: {left_arm_indices}")
    print(f"Right arm: {right_arm_indices}")
    print(f"Grippers: L={left_gripper_index}, R={right_gripper_index}")

    # Check actual controller configuration
    if hasattr(env.unwrapped.agent.controller, 'configs'):
        print("\nController configs:")
        for name, config in env.unwrapped.agent.controller.configs.items():
            if hasattr(config, 'joint_names'):
                print(f"  {name}: {config.joint_names}")

    # Build action index mapping based on the controller's expected order
    # The controller expects: [left_arm_joints, right_arm_joints, left_gripper, right_gripper]
    # WITHOUT head joints (based on the realman.py configuration)

    action_to_joint_name = []
    action_to_qpos = []

    # Map action indices to qpos indices
    # Actions 0-6: Left arm joints
    for i in range(7):
        if i < len(left_arm_indices):
            action_to_qpos.append(left_arm_indices[i])
            action_to_joint_name.append(f"l_joint{i + 1}")

    # Actions 7-13: Right arm joints
    for i in range(7):
        if i < len(right_arm_indices):
            action_to_qpos.append(right_arm_indices[i])
            action_to_joint_name.append(f"r_joint{i + 1}")

    # Actions 14-15: Grippers
    action_to_qpos.append(left_gripper_index)
    action_to_joint_name.append("l_gripper_joint1")
    action_to_qpos.append(right_gripper_index)
    action_to_joint_name.append("r_gripper_joint1")

    print(f"\nAction to joint mapping:")
    for i, (qpos_idx, name) in enumerate(zip(action_to_qpos, action_to_joint_name)):
        print(f"  Action {i}: {name} (qpos index {qpos_idx})")

    obs, _ = env.reset(seed=args.seed)
    viewer = env.render()

    if viewer is None:
        print("Error: Human rendering mode requires a graphical display.")
        env.close()
        return

    viewer.paused = False

    # Fix head joints at initialization
    fix_head_joints(env)

    # Control parameters
    base_move_speed = 0.02  # Delta position per step for base X/Y
    base_rot_speed = 0.05  # Delta angle per step for base Rz
    arm_joint_delta_speed = 0.05  # Delta angle per step for arm joints
    gripper_open_pos = 1.0  # Open position
    gripper_close_pos = 0.0  # Closed position

    # Initial states
    left_gripper_open = True
    right_gripper_open = True
    current_arm = "right"  # Start controlling right arm

    # Get initial joint positions to use as target positions
    initial_qpos = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
    print(f"\nInitial qpos shape: {initial_qpos.shape}")

    # Create initial target positions array
    # This will be our reference that we modify based on user input
    target_positions = np.zeros(action_dim, dtype=np.float32)
    for i, qpos_idx in enumerate(action_to_qpos):
        if qpos_idx is not None:
            target_positions[i] = initial_qpos[qpos_idx]

    # Set initial gripper positions
    target_positions[14] = gripper_open_pos  # Left gripper
    target_positions[15] = gripper_open_pos  # Right gripper

    print("\n--- Control Instructions ---")
    print("==== Base Movement (Direct qpos control) ====")
    print(" J/L: Move Forward/Backward")
    print(" U/O: Rotate Left/Right")
    print(" I/K: Move Right/Left")
    print("==== Arm Control ====")
    print(" Tab: Switch between left/right arm control")
    print(" 1-7: Increase arm joint angles")
    print(" Q/W/E/R/T/Y/U: Decrease arm joint angles (joint 1-7)")
    print(" G: Toggle gripper open/close")
    print("==== System ====")
    print(" Esc: Quit")
    print("-----------------------\n")
    print(f"Currently controlling: {current_arm.upper()} arm")

    step_counter = 0

    # Map keys to negative joint movements
    neg_joint_keys = {'q': 0, 'w': 1, 'e': 2, 'r': 3, 't': 4, 'y': 5, 'u': 6}

    while True:
        if viewer.closed:
            break

        # Get current joint positions
        current_qpos = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()

        # Check keyboard inputs
        should_quit = viewer.window.key_press('escape')
        switch_arm = viewer.window.key_press('tab')

        # Base movement
        move_forward = viewer.window.key_down('j')
        move_backward = viewer.window.key_down('l')
        rotate_left = viewer.window.key_down('u')
        rotate_right = viewer.window.key_down('o')
        move_left = viewer.window.key_down('k')
        move_right = viewer.window.key_down('i')

        # Gripper control
        toggle_gripper = viewer.window.key_press('g')

        # System controls
        if should_quit:
            break

        if switch_arm:
            current_arm = "left" if current_arm == "right" else "right"
            print(f"Now controlling: {current_arm.upper()} arm")

        # Handle base movement by directly modifying qpos
        if move_forward or move_backward or rotate_left or rotate_right or move_left or move_right:
            new_qpos = current_qpos.copy()

            if base_x_idx is not None:
                if move_forward:
                    new_qpos[base_x_idx] += base_move_speed
                elif move_backward:
                    new_qpos[base_x_idx] -= base_move_speed

            if base_y_idx is not None:
                if move_left:
                    new_qpos[base_y_idx] += base_move_speed
                elif move_right:
                    new_qpos[base_y_idx] -= base_move_speed

            if base_rz_idx is not None:
                if rotate_left:
                    new_qpos[base_rz_idx] += base_rot_speed
                elif rotate_right:
                    new_qpos[base_rz_idx] -= base_rot_speed

            # Set the new qpos directly
            env.unwrapped.agent.robot.set_qpos(new_qpos)
            # Re-fix head joints after base movement
            fix_head_joints(env)

        # Arm joint control
        if current_arm == "left":
            # Left arm is actions 0-6
            arm_action_indices = list(range(7))
        else:
            # Right arm is actions 7-13
            arm_action_indices = list(range(7, 14))

        # Positive movements (keys 1-7)
        for i in range(7):
            key = str(i + 1)
            if viewer.window.key_down(key):
                target_positions[arm_action_indices[i]] += arm_joint_delta_speed

        # Negative movements (keys q-u)
        for key, joint_idx in neg_joint_keys.items():
            if joint_idx < 7 and viewer.window.key_down(key):
                target_positions[arm_action_indices[joint_idx]] -= arm_joint_delta_speed

        # Gripper control
        if toggle_gripper:
            if current_arm == "left":
                left_gripper_open = not left_gripper_open
                target_positions[14] = gripper_open_pos if left_gripper_open else gripper_close_pos
                print(f"Left gripper: {'OPEN' if left_gripper_open else 'CLOSED'}")
            else:
                right_gripper_open = not right_gripper_open
                target_positions[15] = gripper_open_pos if right_gripper_open else gripper_close_pos
                print(f"Right gripper: {'OPEN' if right_gripper_open else 'CLOSED'}")

        # Use target_positions as action
        action = target_positions.copy()

        # Step the environment
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            step_counter += 1

            # Fix head joints after each step to prevent drift
            fix_head_joints(env)

        except Exception as e:
            print(f"Error during env.step: {e}")
            print(f"Action shape: {action.shape}, expected: {env.action_space.shape}")
            print(f"Action: {action}")
            break

        env.render()

    print("Closing environment.")
    env.close()


def fix_head_joints(env):
    """Fix head joints to prevent drift"""
    # Get current qpos
    current_qpos = env.agent.robot.get_qpos()[0].cpu().numpy()

    # Find head joint indices
    active_joints = env.agent.robot.get_active_joints()
    head_indices = []
    for i, joint in enumerate(active_joints):
        if "head_joint" in joint.get_name():
            head_indices.append(i)

    # Set head joints to fixed position (0.0)
    new_qpos = current_qpos.copy()
    for idx in head_indices:
        new_qpos[idx] = 0.0

    # Apply the new qpos
    env.agent.robot.set_qpos(new_qpos)

    # Set high stiffness and damping for head joints
    for idx in head_indices:
        active_joints[idx].set_drive_properties(
            stiffness=10000,  # High stiffness
            damping=1000,  # High damping
            force_limit=1000
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)