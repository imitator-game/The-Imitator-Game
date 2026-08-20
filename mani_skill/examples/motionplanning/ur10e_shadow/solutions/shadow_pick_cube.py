import numpy as np
import sapien
from transforms3d import quaternions
from transforms3d.euler import euler2quat
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.ur10e_shadow.motionplanner import UR10eArmMotionPlanningSolver


def solve(env: BaseEnv, seed=None, debug=False, vis=False):
    """Test the motion planning of the UR10e arm."""

    # Reset the environment
    env.reset(seed=seed)

    # Create the UR10e motion planner
    planner = UR10eArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_pose=vis,
        print_env_info=True,
    )

    env = env.unwrapped

    # Define a series of test poses
    test_poses = [
        # Pose 1: extend forward
        sapien.Pose(
            p=[0.2, 0.05, 0.1],
            q=euler2quat(np.pi/2, 0, np.pi/2)
        ),
    ]

    # Run the motion planning test
    for i, target_pose in enumerate(test_poses):

        # Use screw-motion-based planning
        result = planner.move_to_pose_with_screw(target_pose)

        if result != -1:
            # Hold at the pose for a while
            for _ in range(20):
                if vis:
                    env.render_human()
        else:
            print(f"Could not reach pose {i + 1}")


    planner.close()
    return result