import numpy as np
import sapien
import trimesh

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.examples.motionplanning.planner import Planner

OPEN = 1
CLOSED = 0


class RealmanArmMotionPlanningSolver:
    def __init__(
            self,
            env: BaseEnv,
            arm_side: str = "left",  # "left" or "right"
            debug: bool = False,
            vis: bool = True,
            base_pose: sapien.Pose = None,
            visualize_target_grasp_pose: bool = True,
            print_env_info: bool = False,
            joint_vel_limits=0.9,
            joint_acc_limits=0.9,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.arm_side = arm_side
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        self.base_pose = to_sapien_pose(base_pose)

        # Setup arm-specific attributes
        if arm_side == "left":
            self.arm_joint_names = self.env_agent.left_arm_joint_names
            self.gripper_joint_names = self.env_agent.left_gripper_joint_names
            self.tcp_link_name = self.env_agent.left_ee_link_name
            self.move_group = "l_link7"
        else:  # right
            self.arm_joint_names = self.env_agent.right_arm_joint_names
            self.gripper_joint_names = self.env_agent.right_gripper_joint_names
            self.tcp_link_name = self.env_agent.right_ee_link_name
            self.move_group = "r_link7"

        self.move_group_joint_num = 7  # Always 7 for a single arm

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = None

        self.target_arm_qpos = None
        self.last_commanded_qpos = None

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        # Create mapping from planner joints to robot qpos indices
        self._create_joint_mappings()

        if self.vis and self.visualize_target_grasp_pose:
            visual_name = f"grasp_pose_visual_{arm_side}"
            if visual_name not in self.base_env.scene.actors:
                self.grasp_pose_visual = build_realman_gripper_grasp_pose_visual(
                    self.base_env.scene, arm_side
                )
            else:
                self.grasp_pose_visual = self.base_env.scene.actors[visual_name]

            # Set initial pose to TCP
            if arm_side == "left":
                self.grasp_pose_visual.set_pose(self.base_env.agent.left_tcp.pose)
            else:
                self.grasp_pose_visual.set_pose(self.base_env.agent.right_tcp.pose)

        self.elapsed_steps = 0
        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    def _create_joint_mappings(self):
        """Create correct joint mappings - including the head version"""
        # Get all active joints
        active_joints = self.robot.get_active_joints()
        active_joint_names = [j.get_name() for j in active_joints]

        # Create mapping from joint names to qpos indices
        self.joint_name_to_qpos_idx = {
            name: i for i, name in enumerate(active_joint_names)
        }

        # Get qpos indices for the current arm joints
        self.arm_qpos_indices = [
            self.joint_name_to_qpos_idx[name] for name in self.arm_joint_names
        ]

        # Update the action index mapping - based on the actual controller configuration
        # Head (2) + interleaved arms (14) + grippers (2) = 18
        if self.arm_side == "left":
            # Left arm positions in the interleaved sequence: 3, 5, 7, 9, 11, 13, 15
            self.arm_action_indices = [3, 5, 7, 9, 11, 13, 15]
            self.gripper_action_index = 16  # left gripper
        else:  # right arm
            # Right arm positions in the interleaved sequence: 2, 4, 6, 8, 10, 12, 14
            self.arm_action_indices = [2, 4, 6, 8, 10, 12, 14]
            self.gripper_action_index = 17  # right gripper


    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        # Get all robot links and joints for the planner
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        # Create planner with move_group specific configuration
        planner = Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.move_group,
            ee_link=self.tcp_link_name,
            joint_vel_limits=np.ones(self.move_group_joint_num) * self.joint_vel_limits,
            joint_acc_limits=np.ones(self.move_group_joint_num) * self.joint_acc_limits,
        )

        # Set the base pose
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))

        # Important: Get the correct joint indices for this arm from the planner
        self.planner_joint_indices = planner.move_group_joint_indices

        return planner

    def get_current_arm_qpos(self):
        """Get current joint positions for the specific arm (7 values)"""
        full_qpos = self.robot.get_qpos()[0].cpu().numpy()

        # Use our pre-computed indices
        arm_qpos = full_qpos[self.arm_qpos_indices]

        return arm_qpos

    def build_action(self, arm_qpos, arm_qvel=None):
        """Build an action containing all joints - fix drift of the non-controlled arm"""

        # Get the current full qpos
        current_qpos = self.robot.get_qpos()[0].cpu().numpy()

        # If this is the first call or the target is not initialized, save the current position as the target
        if self.last_commanded_qpos is None:
            self.last_commanded_qpos = current_qpos.copy()

        # Create the target qpos - use the last commanded position instead of the current position
        target_qpos = self.last_commanded_qpos.copy()

        # Only update the joints of the currently controlled arm
        for i, idx in enumerate(self.arm_qpos_indices):
            target_qpos[idx] = arm_qpos[i]

        # Save this commanded position
        self.last_commanded_qpos = target_qpos.copy()

        # Build the action (contains all joints except the grippers)
        action = target_qpos[:16].copy()

        # The gripper control logic remains unchanged
        left_gripper_current = current_qpos[18]
        right_gripper_current = current_qpos[16]

        if self.arm_side == "left":
            left_gripper_target = 0.0 if self.gripper_state == OPEN else 1.0
            right_gripper_target = self.last_commanded_qpos[17] if hasattr(self, 'last_commanded_qpos') and len(
                self.last_commanded_qpos) > 17 else right_gripper_current
        else:
            left_gripper_target = self.last_commanded_qpos[16] if hasattr(self, 'last_commanded_qpos') and len(
                self.last_commanded_qpos) > 16 else left_gripper_current
            right_gripper_target = 0.0 if self.gripper_state == OPEN else 1.0

        action = np.concatenate([
            action,
            [left_gripper_target],
            [right_gripper_target],
        ])

        # Update last_commanded_qpos to include the grippers
        if len(self.last_commanded_qpos) == 16:
            self.last_commanded_qpos = np.concatenate([
                self.last_commanded_qpos,
                [left_gripper_target],
                [right_gripper_target],
            ])
        else:
            self.last_commanded_qpos[16] = left_gripper_target
            self.last_commanded_qpos[17] = right_gripper_target

        # Handle velocity (if needed)
        if self.control_mode == "pd_joint_pos_vel" and arm_qvel is not None:
            vel_action = np.zeros(16)
            for i, idx in enumerate(self.arm_qpos_indices):
                vel_action[idx] = arm_qvel[i]
            action = np.hstack([action, vel_action])

        return action

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]

        for i in range(n_step + refine_steps):
            arm_qpos = result["position"][min(i, n_step - 1)]

            if self.control_mode == "pd_joint_pos_vel":
                arm_qvel = result["velocity"][min(i, n_step - 1)]
                action = self.build_action(arm_qpos, arm_qvel)
            else:
                action = self.build_action(arm_qpos, None)

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1

            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTConnect(self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0):
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)

        # Get current arm joint positions (7 values)
        current_arm_qpos = self.robot.get_qpos()[0]

        # Plan with correct dimensions
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            current_arm_qpos,  # Pass only 7 joint values
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )

        if result["status"] != "Success":
            print(f"{self.arm_side} arm planning failed: {result['status']}")
            self.render_wait()
            return -1

        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0):
        """Improved screw motion planning - adds fault tolerance and fallback strategies"""
        pose = to_sapien_pose(pose)

        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)

        # Get current arm joint positions
        current_arm_qpos = self.get_current_arm_qpos()

        # Get the full robot qpos for collision detection
        full_qpos = self.robot.get_qpos()[0].cpu().numpy()
        self.planner.robot.set_qpos(full_qpos, True)

        # Strategy 1: try screw motion planning (with more conservative parameters)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            current_arm_qpos,
            qpos_step=0.05,  # smaller step size
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )

        if result["status"] == "Success":
            if dry_run:
                return result
            return self.follow_path(result, refine_steps=refine_steps)

        # Strategy 2: try segmented screw planning
        segmented_result = self.segmented_screw_planning(pose, dry_run)
        if segmented_result != -1:
            return segmented_result

        # Strategy 3: try RRT planning
        rrt_result = self.rrt_planning_fallback(pose, current_arm_qpos, dry_run, refine_steps)
        if rrt_result != -1:
            return rrt_result

        return -1

    def segmented_screw_planning(self, target_pose: sapien.Pose, dry_run: bool = False):
        """Segmented screw planning - reach the target via intermediate waypoints"""

        current_tcp_pose = self.get_current_tcp_pose()

        # Generate intermediate poses
        intermediate_poses = self.generate_intermediate_poses(current_tcp_pose, target_pose, num_segments=3)

        for i, intermediate_pose in enumerate(intermediate_poses):

            current_arm_qpos = self.get_current_arm_qpos()

            result = self.planner.plan_screw(
                np.concatenate([intermediate_pose.p, intermediate_pose.q]),
                current_arm_qpos,
                qpos_step=0.03,  # smaller step size
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
                wrt_world=True,
            )

            if result["status"] != "Success":
                return -1

            if not dry_run:
                path_result = self.follow_path(result, refine_steps=0)
                if path_result == -1:
                    return -1

        return result if dry_run else path_result

    def rrt_planning_fallback(self, pose: sapien.Pose, current_arm_qpos, dry_run: bool, refine_steps: int):
        """Fixed RRT planning fallback"""

        # Use IK to find the target joint configuration
        target_pose_array = np.concatenate([pose.p, pose.q])

        # Get the full robot qpos
        full_current_qpos = self.robot.get_qpos()[0].cpu().numpy()

        # Use the full qpos for the IK computation (fix the error)
        ik_status, goal_qpos_list = self.planner.IK(target_pose_array, full_current_qpos)

        if ik_status != "Success" or not goal_qpos_list:
            return -1

        # Try planning to the best IK solution
        for i, goal_qpos in enumerate(goal_qpos_list[:3]):  # try at most 3 solutions

            # Only take the joints corresponding to the move_group as the target
            goal_arm_qpos = goal_qpos[self.planner.move_group_joint_indices]

            result = self.planner.plan_qpos_to_qpos(
                [goal_arm_qpos],
                current_arm_qpos,
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
                planning_time=3.0,
                rrt_range=0.15,
            )

            if result["status"] == "Success":
                if dry_run:
                    return result
                return self.follow_path(result, refine_steps=refine_steps)

        return -1

    def get_current_tcp_pose(self):
        """Get current TCP pose"""
        if self.arm_side == "left":
            return self.base_env.agent.left_tcp.pose
        else:
            return self.base_env.agent.right_tcp.pose

    def generate_intermediate_poses(self, start_pose, end_pose, num_segments=3):
        """Generate intermediate poses"""
        intermediate_poses = []

        for i in range(1, num_segments):
            alpha = i / num_segments

            # Position interpolation
            pos = start_pose.p[0].cpu().numpy() * (1 - alpha) + end_pose.p * alpha

            # Quaternion interpolation
            start_q = start_pose.q[0].cpu().numpy()
            end_q = end_pose.q

            # Ensure the quaternions are in the same hemisphere
            if np.dot(start_q, end_q) < 0:
                end_q = -end_q

            q = start_q * (1 - alpha) + end_q * alpha
            q = q / np.linalg.norm(q)

            intermediate_poses.append(sapien.Pose(pos, q))

        return intermediate_poses

    def open_gripper(self):
        self.gripper_state = OPEN
        for i in range(6):
            action = self.build_action(self.get_current_arm_qpos())
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1

            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6):
        self.gripper_state = CLOSED
        for i in range(t):
            action = self.build_action(self.get_current_arm_qpos())
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1

            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=to_sapien_pose(pose).to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass

def build_realman_gripper_grasp_pose_visual(scene: ManiSkillScene, arm_side: str):
    """Build visual marker for Realman gripper grasp pose"""
    from transforms3d import quaternions

    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.0  # Slightly smaller for Realman gripper

    # Different colors for left and right arm
    base_color = [0.8, 0.3, 0.3, 0.7] if arm_side == "left" else [0.3, 0.3, 0.8, 0.7]

    # Center sphere
    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=base_color)
    )

    # Approach direction (green)
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.06]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )

    # Closing direction base
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.04]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )

    # Gripper finger indicators
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0.02, grasp_width + grasp_pose_visual_width, 0.02 - 0.04],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.03, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )

    builder.add_box_visual(
        pose=sapien.Pose(
            p=[0.02, -grasp_width - grasp_pose_visual_width, 0.02 - 0.04],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.03, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )

    grasp_pose_visual = builder.build_kinematic(name=f"grasp_pose_visual_{arm_side}")
    return grasp_pose_visual