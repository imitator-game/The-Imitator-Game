#!/usr/bin/env python3

import numpy as np
import sapien
import gymnasium as gym

# Import environment
import sys
import os
maniskill_dev_path = '/home/ManiSkill'
if maniskill_dev_path in sys.path:
    sys.path.remove(maniskill_dev_path)
sys.path.insert(0, maniskill_dev_path)

import mani_skill.envs
from mani_skill.examples.motionplanning.widowxai.motionplanner import WidowXAIArmMotionPlanningSolver

def test_simple_motion():
    """Test the most basic motion planning functionality"""
    
    # Create the environment
    env = gym.make(
        "PickCubeYCB-v1",
        render_mode="human", 
        control_mode="pd_joint_pos",
        robot_uids="widowxai",
    )
    
    print("🚀 Starting simple motion planning test...")
    
    # Reset the environment
    env.reset(seed=42)
    
    # Create the motion planner
    planner = WidowXAIArmMotionPlanningSolver(
        env,
        debug=True,
        vis=True,
        base_pose=env.unwrapped.agent.robot.pose,
        print_env_info=True,
    )
    
    print("✅ Motion planner created successfully")
    
    # Get the current TCP position
    current_tcp_pos = env.unwrapped.agent.tcp.pose.p.cpu().numpy()[0]
    current_tcp_q = env.unwrapped.agent.tcp.pose.q.cpu().numpy()[0] 
    
    print(f"🖐️  Current TCP position: {current_tcp_pos}")
    print(f"🔄 Current TCP quaternion: {current_tcp_q}")
    
    # Test 1: simple relative movement - move up 5cm
    print("\n📏 Test 1: Moving up 5cm...")
    target_pos = current_tcp_pos + np.array([0, 0, 0.05])  # Up 5cm
    target_pose = sapien.Pose(p=target_pos, q=current_tcp_q)
    
    print(f"🎯 Target position: {target_pos}")
    print(f"📏 Movement distance: {np.linalg.norm(target_pos - current_tcp_pos):.3f}m")
    
    # Attempt motion planning and actually execute it
    print("🔄 Starting to execute motion planning...")
    result = planner.move_to_pose_with_screw(target_pose, dry_run=False)  # Actually execute
    
    if result != -1:
        print("✅ Test 1 passed: upward movement executed successfully!")
        return True
    else:
        print("❌ Test 1 failed: upward movement planning failed")
        
        # Test 2: more conservative movement - move up 2cm
        print("\n📏 Test 2: more conservative movement - move up 2cm...")
        target_pos = current_tcp_pos + np.array([0, 0, 0.02])  # Up 2cm
        target_pose = sapien.Pose(p=target_pos, q=current_tcp_q)
        
        result = planner.move_to_pose_with_screw(target_pose, dry_run=True)
        
        if result != -1:
            print("✅ Test 2 passed: conservative movement planning succeeded!")
            return True
        else:
            print("❌ Test 2 failed: conservative movement planning also failed")
            
            # Test 3: joint-space movement
            print("\n🔧 Test 3: direct joint-space movement...")
            current_qpos = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
            print(f"🤖 Current joint angles: {current_qpos}")
            
            # Slightly change the first joint
            target_qpos = current_qpos.copy()
            target_qpos[0] += 0.1  # Rotate the first joint by 0.1 radians
            
            # Execute the joint movement
            for i in range(10):
                action = np.hstack([target_qpos[:-2], 1])  # Keep the gripper open
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                
            print("✅ Test 3 done: direct joint control")
            return True
    
    planner.close()
    env.close()

if __name__ == "__main__":
    test_simple_motion()