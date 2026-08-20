import numpy as np
import sapien
import trimesh
from mani_skill.examples.motionplanning.planner import Planner
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose


class PandaStickMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
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

        self.base_pose = to_sapien_pose(base_pose)

        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.elapsed_steps = 0

        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

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
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_vel_limits,
            joint_acc_limits=np.ones(7) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

    def follow_path(self, result, refine_steps: int = 0):
        n_step = result["position"].shape[0]
        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel])
            else:
                action = np.hstack([qpos])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTConnect(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_qpos_to_pose(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            wrt_world=True,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def move_to_pose_with_screw(
        self, pose: sapien.Pose, dry_run: bool = False, refine_steps: int = 0
    ):
        pose = to_sapien_pose(pose)
        # try screw two times before giving up
        pose = sapien.Pose(p=pose.p, q=pose.q)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos().cpu().numpy()[0],
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
        )
        if result["status"] != "Success":
            result = self.planner.plan_screw(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        if dry_run:
            return result
        return self.follow_path(result, refine_steps=refine_steps)

    def add_box_collision(self, extents: np.ndarray, pose: sapien.Pose):
        self.use_point_cloud = True
        box = trimesh.creation.box(extents, transform=pose.to_transformation_matrix())
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

    def move_through_waypoints(
            self,
            waypoints,
            waypoint_types=None,
            smooth_factor=0.3,  # Reduced default for more precision
            critical_point_pause_time=0.05,  # Shorter default pause
            use_cartesian_interpolation=True,  # NEW: Default to Cartesian interpolation
            cartesian_density=50,  # NEW: Points per meter
            dry_run=False,
            other_gripper_state=None,
            verbose=False
    ):
        """
        Move through multiple waypoints with smooth transitions

        Args:
            waypoints: List of sapien.Pose objects representing target poses
            waypoint_types: List of 'smooth' or 'critical' for each waypoint
                           'smooth': robot will blend through this point
                           'critical': robot will stop precisely at this point
            smooth_factor: How smooth the transitions should be (0-1)
            critical_point_pause_time: How long to pause at critical points
            dry_run: If True, only return the plan without executing
            other_gripper_state: For multi-robot scenarios
            verbose: Print debug information

        Returns:
            Result from following the planned path or the plan itself if dry_run
        """
        # Convert sapien poses to numpy arrays
        waypoint_arrays = []
        for pose in waypoints:
            pose = to_sapien_pose(pose)
            waypoint_arrays.append(np.concatenate([pose.p, pose.q]))

        # Plan the trajectory through all waypoints
        result = self.planner.plan_waypoints(
            waypoint_arrays,
            self.robot.get_qpos().cpu().numpy()[0],
            waypoint_types=waypoint_types,
            time_step=self.base_env.control_timestep,
            use_point_cloud=self.use_point_cloud,
            smooth_factor=smooth_factor,
            critical_point_pause_time=critical_point_pause_time,
            use_cartesian_interpolation=use_cartesian_interpolation,
            cartesian_density=cartesian_density,
            wrt_world=True,
            verbose=verbose
        )

        if result["status"] != "Success":
            print(f"Waypoint planning failed: {result['status']}")
            if 'failed_waypoints' in result:
                print(f"Failed waypoints: {result['failed_waypoints']}")
            self.render_wait()
            return -1

        self.render_wait()

        if dry_run:
            return result

        # Execute the planned trajectory
        return self.follow_path(result, refine_steps=0)

    # Also add a convenience method for common use cases
    def move_to_poses_with_stops(
            self,
            poses,
            stop_at_indices=None,
            dry_run=False,
            other_gripper_state=None
    ):
        """
        Convenience method: Move through poses, stopping at specified indices

        Args:
            poses: List of target poses
            stop_at_indices: List of indices where robot should stop precisely
                            If None, robot will smoothly blend through all points
        """
        if stop_at_indices is None:
            stop_at_indices = []

        waypoint_types = []
        for i in range(len(poses)):
            if i in stop_at_indices:
                waypoint_types.append('critical')
            else:
                waypoint_types.append('smooth')

        return self.move_through_waypoints(
            poses,
            waypoint_types=waypoint_types,
            dry_run=dry_run,
            other_gripper_state=other_gripper_state
        )

    def preprocess_waypoints(self, waypoints, min_distance=0.01):
        """
        Remove redundant waypoints that are too close together

        Args:
            waypoints: List of sapien.Pose objects
            min_distance: Minimum distance between consecutive waypoints

        Returns:
            Filtered list of waypoints
        """
        if len(waypoints) <= 1:
            return waypoints

        filtered = [waypoints[0]]

        for i in range(1, len(waypoints)):
            current_pose = waypoints[i]
            last_pose = filtered[-1]

            # Calculate distance
            pos_diff = np.linalg.norm(
                np.array(current_pose.p) - np.array(last_pose.p)
            )

            # Calculate rotation difference
            rot_diff = np.arccos(np.clip(
                np.abs(np.dot(current_pose.q, last_pose.q)),
                -1.0, 1.0
            ))

            # Keep waypoint if it's sufficiently different
            if pos_diff > min_distance or rot_diff > 0.05:  # 0.05 rad ~= 3 degrees
                filtered.append(current_pose)

        # Always keep the last waypoint
        if filtered[-1] != waypoints[-1]:
            filtered.append(waypoints[-1])

        return filtered

    def adaptive_smooth_factor(self, waypoint_index, total_waypoints, task_phase="manipulation"):
        """
        Calculate adaptive smoothing factor based on waypoint position and task phase

        Args:
            waypoint_index: Current waypoint index
            total_waypoints: Total number of waypoints
            task_phase: "approach", "manipulation", or "retract"

        Returns:
            Smoothing factor for this waypoint
        """
        if task_phase == "approach":
            # More smoothing during approach
            return 0.5 + 0.2 * (waypoint_index / max(1, total_waypoints - 1))
        elif task_phase == "manipulation":
            # Less smoothing during precise manipulation
            return 0.2
        elif task_phase == "retract":
            # More smoothing during retraction
            return 0.6
        else:
            return 0.3  # Default
