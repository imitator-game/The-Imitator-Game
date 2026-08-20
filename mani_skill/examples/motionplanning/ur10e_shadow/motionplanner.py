import numpy as np
import sapien
import trimesh
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.structs.pose import to_sapien_pose
from mani_skill.examples.motionplanning.planner import Planner


class UR10eArmMotionPlanningSolver:
    def __init__(
            self,
            env: BaseEnv,
            debug: bool = False,
            vis: bool = True,
            base_pose: sapien.Pose = None,
            visualize_target_pose: bool = True,
            print_env_info: bool = True,
            joint_vel_limits=0.9,
            joint_acc_limits=0.9,
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent
        self.robot = self.env_agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits

        # UR10e specific settings
        self.base_pose = to_sapien_pose(base_pose) if base_pose else sapien.Pose()

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_pose = visualize_target_pose

        # Create target pose visualization
        self.target_pose_visual = None

        self.elapsed_steps = 0
        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

    def setup_planner(self):
        # Get all link names for the robot
        link_names = [link.get_name() for link in self.robot.get_links()]

        # Get ALL joint names from the robot (mplib needs all of them)
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        # Store UR10e joint names for later use
        self.ur10e_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "rh_WRJ2",
            "rh_WRJ1"
        ]

        # Check which UR10e joints are actually present
        ur10e_joints_present = [j for j in self.ur10e_joint_names if j in joint_names]

        # Find the indices of UR10e joints
        self.ur10e_joint_indices = [joint_names.index(j) for j in ur10e_joints_present]

        planner = Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,  # Pass ALL joints
            move_group="rh_palm",  # This group in SRDF defines which joints to plan
            joint_vel_limits=np.ones(8) * self.joint_vel_limits,  # Only 6 for UR10e arm
            joint_acc_limits=np.ones(8) * self.joint_acc_limits,  # Only 6 for UR10e arm
        )

        # Set base pose
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))

        return planner

    def build_target_pose_visual(self, scene: ManiSkillScene):
        """Create visualization marker for target pose"""
        builder = scene.create_actor_builder()

        # Create coordinate axes visualization
        axis_length = 0.1
        axis_radius = 0.002

        # X-axis - red
        builder.add_box_visual(
            pose=sapien.Pose(p=[axis_length / 2, 0, 0]),
            half_size=[axis_length / 2, axis_radius, axis_radius],
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7])
        )

        # Y-axis - green
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, axis_length / 2, 0]),
            half_size=[axis_radius, axis_length / 2, axis_radius],
            material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7])
        )

        # Z-axis - blue
        builder.add_box_visual(
            pose=sapien.Pose(p=[0, 0, axis_length / 2]),
            half_size=[axis_radius, axis_radius, axis_length / 2],
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7])
        )

        # Center sphere
        builder.add_sphere_visual(
            pose=sapien.Pose(p=[0, 0, 0]),
            radius=0.01,
            material=sapien.render.RenderMaterial(base_color=[1, 1, 0, 0.8])
        )

        return builder.build_kinematic(name="target_pose_visual")

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def follow_path(self, result, refine_steps: int = 0):
        """Execute the planned path"""
        n_step = result["position"].shape[0]

        # Get the indices of UR10e joints in the full robot joint list
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        ur10e_indices = []
        for joint_name in self.ur10e_joint_names:
            if joint_name in joint_names:
                ur10e_indices.append(joint_names.index(joint_name))


        for i in range(n_step + refine_steps):
            # Get the planned position for this step
            planned_qpos = result["position"][min(i, n_step - 1)]

            # Get current full robot state
            current_full_qpos = self.robot.get_qpos()[0].cpu().numpy()

            # Update only the UR10e joints with planned values
            new_full_qpos = current_full_qpos.copy()
            for idx, ur_idx in enumerate(ur10e_indices):
                if idx < len(planned_qpos):
                    new_full_qpos[ur_idx] = planned_qpos[idx]

            # Prepare action based on control mode
            if self.control_mode == "pd_joint_pos_vel":
                # Get planned velocities
                planned_qvel = result["velocity"][min(i, n_step - 1)]
                current_full_qvel = np.zeros_like(current_full_qpos)

                # Update only UR10e joint velocities
                for idx, ur_idx in enumerate(ur10e_indices):
                    if idx < len(planned_qvel):
                        current_full_qvel[ur_idx] = planned_qvel[idx]

                action = np.concatenate([new_full_qpos, current_full_qvel])
            elif self.control_mode == "pd_joint_pos":
                action = new_full_qpos
            else:
                # Default to position control
                action = new_full_qpos

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1

            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTConnect(
            self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """Move to target pose using RRT-Connect algorithm"""
        pose = to_sapien_pose(pose)
        if self.target_pose_visual is not None:
            self.target_pose_visual.set_pose(pose)

        # Get current robot joint positions
        current_full_qpos = self.robot.get_qpos().cpu().numpy()[0]

        # Extract UR10e joint positions using stored indices
        if hasattr(self, 'ur10e_joint_indices'):
            current_ur10e_qpos = current_full_qpos[self.ur10e_joint_indices]
        else:
            # Fallback to first 6 joints if indices not set
            current_ur10e_qpos = current_full_qpos[:8]


        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            current_ur10e_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )

        if result["status"] != "Success":
            print(f"Planning failed: {result['status']}")
            self.render_wait()
            return -1

        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
            self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """Move to target pose using screw motion planning"""
        pose = to_sapien_pose(pose)
        if self.target_pose_visual is not None:
            self.target_pose_visual.set_pose(pose)

        # Get current robot joint positions
        current_full_qpos = self.robot.get_qpos().cpu().numpy()[0]

        # Extract UR10e joint positions using stored indices
        if hasattr(self, 'ur10e_joint_indices'):
            current_ur10e_qpos = current_full_qpos[self.ur10e_joint_indices]
        else:
            # Fallback to first 6 joints if indices not set
            current_ur10e_qpos = current_full_qpos[:8]


        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            current_ur10e_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )

        if result["status"] != "Success":
            # Retry once
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                current_ur10e_qpos,
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(f"Planning failed: {result['status']}")
                self.render_wait()
                return -1

        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def open_hand(self):
        """Open ShadowHand - placeholder for now"""
        # This would require implementing ShadowHand specific control
        # For now, just maintain current position
        current_qpos = self.robot.get_qpos()[0].cpu().numpy()
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = current_qpos
            else:
                action = np.concatenate([current_qpos, np.zeros_like(current_qpos)])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_hand(self, t=6):
        """Close ShadowHand - placeholder for now"""
        # This would require implementing ShadowHand specific control
        # For now, just maintain current position
        current_qpos = self.robot.get_qpos()[0].cpu().numpy()
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = current_qpos
            else:
                action = np.concatenate([current_qpos, np.zeros_like(current_qpos)])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        """Add box collision object"""
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
        pts, _ = trimesh.sample.sample_surface(box, 256)
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def add_collision_pts(self, pts: np.ndarray):
        """Add collision points"""
        self.use_point_cloud = True
        if self.all_collision_pts is None:
            self.all_collision_pts = pts
        else:
            self.all_collision_pts = np.vstack([self.all_collision_pts, pts])
        self.planner.update_point_cloud(self.all_collision_pts)

    def clear_collisions(self):
        """Clear all collision points"""
        self.all_collision_pts = None
        self.use_point_cloud = False

    def close(self):
        pass