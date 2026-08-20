from __future__ import annotations

import os
from typing import Optional, Sequence, List

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
from transforms3d.quaternions import mat2quat, quat2mat
import sapien

from mplib.pymp import ArticulatedModel, PlanningWorld
from mplib.pymp.planning import ompl


class Planner:
    """Enhanced Motion Planner with smooth waypoint planning capabilities"""

    def __init__(
            self,
            urdf: str,
            move_group: str,
            srdf: str = "",
            package_keyword_replacement: str = "",
            user_link_names: Sequence[str] = [],
            user_joint_names: Sequence[str] = [],
            joint_vel_limits: Optional[Sequence[float] | np.ndarray] = None,
            joint_acc_limits: Optional[Sequence[float] | np.ndarray] = None,
            **kwargs,
    ):
        """Motion planner for robots with fixed indexing issues and smooth waypoint planning.

        Args:
            urdf: Unified Robot Description Format file.
            user_link_names: names of links, the order matters.
                If empty, all links will be used.
            user_joint_names: names of the joints to plan.
                If empty, all active joints will be used.
            move_group: target link to move, usually the end-effector.
            joint_vel_limits: maximum joint velocities for time parameterization
            joint_acc_limits: maximum joint accelerations for time parameterization
            srdf: Semantic Robot Description Format file.
        """
        if joint_vel_limits is None:
            joint_vel_limits = []
        if joint_acc_limits is None:
            joint_acc_limits = []
        self.urdf = urdf
        self.srdf = srdf

        if srdf == "" and os.path.exists(urdf.replace(".urdf", ".srdf")):
            self.srdf = urdf.replace(".urdf", ".srdf")
            print(f"No SRDF file provided but found {self.srdf}")

        # replace package:// keyword if exists
        urdf = self.replace_package_keyword(package_keyword_replacement)

        self.robot = ArticulatedModel(
            urdf,
            self.srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            convex=True,
            verbose=False,
        )
        self.pinocchio_model = self.robot.get_pinocchio_model()
        self.user_link_names = self.pinocchio_model.get_link_names()
        self.user_joint_names = self.pinocchio_model.get_joint_names()

        self.planning_world = PlanningWorld(
            [self.robot],
            ["robot"],
            kwargs.get("normal_objects", []),
            kwargs.get("normal_object_names", []),
        )

        if self.srdf == "":
            self.generate_collision_pair()
            self.robot.update_SRDF(self.srdf)

        self.joint_name_2_idx = {}
        for i, joint in enumerate(self.user_joint_names):
            self.joint_name_2_idx[joint] = i
        self.link_name_2_idx = {}
        for i, link in enumerate(self.user_link_names):
            self.link_name_2_idx[link] = i

        assert (
                move_group in self.user_link_names
        ), f"end-effector '{move_group}' not found as one of the links in {self.user_link_names}"

        self.move_group = move_group
        self.robot.set_move_group(self.move_group)
        self.move_group_joint_indices = self.robot.get_move_group_joint_indices()

        self.joint_types = self.pinocchio_model.get_joint_types()
        self.joint_limits = np.concatenate(self.pinocchio_model.get_joint_limits())

        # Handle joint velocity and acceleration limits properly
        if len(joint_vel_limits) > 0:
            # If limits are provided, ensure they match move_group size
            if len(joint_vel_limits) != len(self.move_group_joint_indices):
                print(
                    f"Warning: joint_vel_limits size ({len(joint_vel_limits)}) doesn't match move_group size ({len(self.move_group_joint_indices)})")
                # Use the minimum to avoid index errors
                limit_size = min(len(joint_vel_limits), len(self.move_group_joint_indices))
                self.joint_vel_limits = np.array(joint_vel_limits[:limit_size])
            else:
                self.joint_vel_limits = np.array(joint_vel_limits)
        else:
            self.joint_vel_limits = np.ones(len(self.move_group_joint_indices))

        if len(joint_acc_limits) > 0:
            # If limits are provided, ensure they match move_group size
            if len(joint_acc_limits) != len(self.move_group_joint_indices):
                print(
                    f"Warning: joint_acc_limits size ({len(joint_acc_limits)}) doesn't match move_group size ({len(self.move_group_joint_indices)})")
                # Use the minimum to avoid index errors
                limit_size = min(len(joint_acc_limits), len(self.move_group_joint_indices))
                self.joint_acc_limits = np.array(joint_acc_limits[:limit_size])
            else:
                self.joint_acc_limits = np.array(joint_acc_limits)
        else:
            self.joint_acc_limits = np.ones(len(self.move_group_joint_indices))

        self.move_group_link_id = self.link_name_2_idx[self.move_group]

        # Validation with better error messages
        assert len(self.joint_vel_limits) == len(self.joint_acc_limits), (
            f"length of joint_vel_limits ({len(self.joint_vel_limits)}) =/= "
            f"length of joint_acc_limits ({len(self.joint_acc_limits)})"
        )

        # More flexible validation - allow different sizes but warn
        if len(self.joint_vel_limits) != len(self.move_group_joint_indices):
            print(f"Warning: Adjusting joint limits to match move_group size")
            self.joint_vel_limits = np.ones(len(self.move_group_joint_indices))
            self.joint_acc_limits = np.ones(len(self.move_group_joint_indices))

        self.planning_world = PlanningWorld([self.robot], ["robot"], [], [])
        self.planner = ompl.OMPLPlanner(world=self.planning_world)

        # Cache for performance optimization
        self._fixed_joint_indices_cache = None
        self._detect_fixed_joints()

    def _detect_fixed_joints(self):
        """Auto-detect joints that should be fixed (like head joints)"""
        self._fixed_joint_indices_cache = []
        for i, joint_name in enumerate(self.user_joint_names):
            # Auto-detect head joints or other non-arm joints
            if 'head' in joint_name.lower() or 'camera' in joint_name.lower():
                self._fixed_joint_indices_cache.append(i)

    def replace_package_keyword(self, package_keyword_replacement):
        """Replace package:// keyword with the given string"""
        rtn_urdf = self.urdf
        with open(self.urdf) as in_f:
            content = in_f.read()
            if "package://" in content:
                rtn_urdf = self.urdf.replace(".urdf", "_package_keyword_replaced.urdf")
                content = content.replace("package://", package_keyword_replacement)
                if not os.path.exists(rtn_urdf):
                    with open(rtn_urdf, "w") as out_f:
                        out_f.write(content)
        return rtn_urdf

    def generate_collision_pair(self, sample_time=1000000, echo_freq=100000):
        """Generate SRDF file with collision pairs"""
        print(
            "Since no SRDF file is provided. We will first detect link pairs that will"
            " always collide. This may take several minutes."
        )
        n_link = len(self.user_link_names)
        cnt = np.zeros((n_link, n_link), dtype=np.int32)
        for i in range(sample_time):
            qpos = self.pinocchio_model.get_random_configuration()
            self.robot.set_qpos(qpos, True)
            collisions = self.planning_world.collide_full()
            for collision in collisions:
                u = self.link_name_2_idx[collision.link_name1]
                v = self.link_name_2_idx[collision.link_name2]
                cnt[u][v] += 1
            if i % echo_freq == 0:
                print("Finish %.1f%%!" % (i * 100 / sample_time))

        import xml.etree.ElementTree as ET
        from xml.dom import minidom

        root = ET.Element("robot")
        robot_name = self.urdf.split("/")[-1].split(".")[0]
        root.set("name", robot_name)
        self.srdf = self.urdf.replace(".urdf", ".srdf")

        for i in range(n_link):
            for j in range(n_link):
                if cnt[i][j] == sample_time:
                    link1 = self.user_link_names[i]
                    link2 = self.user_link_names[j]
                    print(
                        f"Ignore collision pair: ({link1}, {link2}), "
                        "reason: always collide"
                    )
                    collision = ET.SubElement(root, "disable_collisions")
                    collision.set("link1", link1)
                    collision.set("link2", link2)
                    collision.set("reason", "Default")
        with open(self.srdf, "w") as srdf_file:
            srdf_file.write(
                minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
            )
            srdf_file.close()
        print("Saving the SRDF file to %s" % self.srdf)

    def distance_6D(self, p1, q1, p2, q2):
        """Compute the distance between two poses"""
        return np.linalg.norm(p1 - p2) + min(
            np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2)
        )

    def wrap_joint_limit(self, q) -> bool:
        """Check and wrap joint configuration within limits"""
        n = len(q)
        flag = True
        for i in range(n):
            if self.joint_types[i].startswith("JointModelR"):
                if -1e-3 <= q[i] - self.joint_limits[i][0] < 0:
                    continue
                q[i] -= (
                        2 * np.pi * np.floor((q[i] - self.joint_limits[i][0]) / (2 * np.pi))
                )
                if q[i] > self.joint_limits[i][1] + 1e-3:
                    flag = False
            else:
                if (
                        q[i] < self.joint_limits[i][0] - 1e-3
                        or q[i] > self.joint_limits[i][1] + 1e-3
                ):
                    flag = False
        return flag

    def pad_qpos(self, qpos, articulation=None):
        """
        FIXED VERSION: Pad qpos to full joint configuration if needed
        This version handles both sequential and non-sequential joint indices safely
        """
        if len(qpos) == len(self.move_group_joint_indices):
            tmp = (
                articulation.get_qpos()
                if articulation is not None
                else self.robot.get_qpos()
            )

            # Check if move_group indices are sequential starting from 0
            expected_sequential = list(range(len(self.move_group_joint_indices)))
            if list(self.move_group_joint_indices) == expected_sequential:
                # Sequential case - use simple assignment
                tmp[: len(qpos)] = qpos
            else:
                # Non-sequential case - use index-by-index assignment with bounds checking
                for i, idx in enumerate(self.move_group_joint_indices):
                    if idx < len(tmp):  # Bounds check to prevent index error
                        tmp[idx] = qpos[i]
                    else:
                        print(f"Warning: joint index {idx} exceeds qpos size {len(tmp)}")
            qpos = tmp

        # Final validation
        if len(qpos) != len(self.joint_limits):
            print(f"Warning: qpos length ({len(qpos)}) doesn't match joint_limits ({len(self.joint_limits)})")
            # Try to pad or truncate as needed
            if len(qpos) < len(self.joint_limits):
                # Pad with zeros or current positions
                padded = np.zeros(len(self.joint_limits))
                padded[:len(qpos)] = qpos
                qpos = padded
            else:
                # Truncate
                qpos = qpos[:len(self.joint_limits)]

        return qpos

    def check_for_collision(
            self,
            collision_function,
            articulation: Optional[ArticulatedModel] = None,
            qpos: Optional[np.ndarray] = None,
    ) -> list:
        """Helper function to check for collision"""
        if articulation is None:
            articulation = self.robot
        if qpos is None:
            qpos = articulation.get_qpos()
        qpos = self.pad_qpos(qpos, articulation)

        old_qpos = articulation.get_qpos()
        articulation.set_qpos(qpos, True)
        idx = self.planning_world.get_articulations().index(articulation)
        collisions = collision_function(idx)
        articulation.set_qpos(old_qpos, True)
        return collisions

    def check_for_self_collision(
            self,
            articulation: Optional[ArticulatedModel] = None,
            qpos: Optional[np.ndarray] = None,
    ) -> list:
        """Check if the robot is in self-collision"""
        return self.check_for_collision(
            self.planning_world.self_collide, articulation, qpos
        )

    def check_for_env_collision(
            self,
            articulation: Optional[ArticulatedModel] = None,
            qpos: Optional[np.ndarray] = None,
            with_point_cloud=False,
            use_attach=False,
    ) -> list:
        """Check if the robot is in collision with the environment"""
        prev_use_point_cloud = self.planning_world.use_point_cloud
        prev_use_attach = self.planning_world.use_attach
        self.planning_world.set_use_point_cloud(with_point_cloud)
        self.planning_world.set_use_attach(use_attach)

        results = self.check_for_collision(
            self.planning_world.collide_with_others, articulation, qpos
        )

        self.planning_world.set_use_point_cloud(prev_use_point_cloud)
        self.planning_world.set_use_attach(prev_use_attach)
        return results

    def IK(self, goal_pose, start_qpos, mask=None, n_init_qpos=20, threshold=1e-3):
        """Inverse kinematics with multiple initial configurations"""
        if mask is None:
            mask = []
        index = self.link_name_2_idx[self.move_group]
        min_dis = 1e9
        idx = self.move_group_joint_indices
        qpos0 = np.copy(start_qpos)
        results = []
        self.robot.set_qpos(start_qpos, True)

        for _ in range(n_init_qpos):
            ik_results = self.pinocchio_model.compute_IK_CLIK(
                index, goal_pose, start_qpos, mask
            )
            flag = self.wrap_joint_limit(ik_results[0])

            # Check collision with bounds checking
            if len(ik_results[0]) > max(idx) if idx else False:
                self.planning_world.set_qpos_all(ik_results[0][idx])
                if len(self.planning_world.collide_full()) != 0:
                    flag = False
            else:
                flag = False

            if flag:
                self.pinocchio_model.compute_forward_kinematics(ik_results[0])
                new_pose = self.pinocchio_model.get_link_pose(index)
                tmp_dis = self.distance_6D(
                    goal_pose[:3], goal_pose[3:], new_pose[:3], new_pose[3:]
                )
                if tmp_dis < min_dis:
                    min_dis = tmp_dis
                if tmp_dis < threshold:
                    result = ik_results[0]
                    unique = True
                    for j in range(len(results)):
                        if np.linalg.norm(results[j][idx] - result[idx]) < 0.1:
                            unique = False
                    if unique:
                        results.append(result)
            start_qpos = self.pinocchio_model.get_random_configuration()
            mask_len = len(mask)
            if mask_len > 0:
                for j in range(mask_len):
                    if mask[j]:
                        start_qpos[j] = qpos0[j]

        if len(results) != 0:
            status = "Success"
        elif min_dis != 1e9:
            status = "IK Failed! Distance {:f} is greater than threshold {:f}.".format(
                min_dis,
                threshold,
            )
        else:
            status = "IK Failed! Cannot find valid solution."
        return status, results

    def TOPP(self, path, step=0.1, verbose=False):
        """Time-Optimal Path Parameterization"""
        N_samples = path.shape[0]
        dof = path.shape[1]

        # Adjust limits if needed
        if dof != len(self.joint_vel_limits):
            print(f"Warning: path DOF ({dof}) doesn't match joint_vel_limits ({len(self.joint_vel_limits)})")
            # Use minimum to avoid errors
            dof = min(dof, len(self.joint_vel_limits))
            path = path[:, :dof]

        joint_vel_limits = self.joint_vel_limits[:dof]
        joint_acc_limits = self.joint_acc_limits[:dof]

        ss = np.linspace(0, 1, N_samples)
        path = ta.SplineInterpolator(ss, path)
        pc_vel = constraint.JointVelocityConstraint(joint_vel_limits)
        pc_acc = constraint.JointAccelerationConstraint(joint_acc_limits)
        instance = algo.TOPPRA(
            [pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel"
        )
        jnt_traj = instance.compute_trajectory()
        if jnt_traj is None:
            print("Fail to parameterize path")
            return None, None, None, None, None
        ts_sample = np.linspace(0, jnt_traj.duration, int(jnt_traj.duration / step))
        qs_sample = jnt_traj(ts_sample)
        qds_sample = jnt_traj(ts_sample, 1)
        qdds_sample = jnt_traj(ts_sample, 2)
        return ts_sample, qs_sample, qds_sample, qdds_sample, jnt_traj.duration

    def plan_waypoints_smooth(
            self,
            waypoint_poses: List[np.ndarray],
            current_qpos: np.ndarray,
            time_step: float = 0.1,
            mask: List = None,
            points_per_segment: int = 10,
            use_straight_line: bool = False,
            wrt_world: bool = True,
            verbose: bool = False
    ):
        """
        Plan smooth trajectory through multiple waypoints

        Args:
            waypoint_poses: List of 7D poses [x,y,z,qw,qx,qy,qz] to visit
            current_qpos: Starting joint configuration
            time_step: Time step for trajectory
            mask: IK mask for constraints
            points_per_segment: Number of intermediate points between waypoints
            use_straight_line: If True, use straight line interpolation between poses
            wrt_world: If True, poses are in world frame
            verbose: Print debug info
        """
        if mask is None:
            mask = []

        # Transform poses if needed
        if wrt_world:
            transformed_poses = []
            for pose in waypoint_poses:
                transformed_poses.append(self.transform_goal_to_wrt_base(pose))
            waypoint_poses = transformed_poses

        # Solve IK for all waypoints
        all_waypoint_configs = []
        working_qpos = current_qpos.copy()

        for i, pose in enumerate(waypoint_poses):
            ik_status, goal_configs = self.IK(pose, working_qpos, mask)
            if ik_status != "Success":
                return {"status": f"IK failed for waypoint {i}: {ik_status}"}

            # Choose the closest configuration to current pose
            best_config = None
            min_dist = float('inf')
            idx = self.move_group_joint_indices

            for config in goal_configs:
                if len(config) > max(idx) if idx else False:
                    config_joints = config[idx]
                    dist = np.linalg.norm(config_joints - working_qpos[idx])
                    if dist < min_dist:
                        min_dist = dist
                        best_config = config

            if best_config is None:
                return {"status": f"No valid IK solution for waypoint {i}"}

            all_waypoint_configs.append(best_config)
            working_qpos = best_config  # Update for next IK solve

        if verbose:
            print(f"Found IK solutions for {len(all_waypoint_configs)} waypoints")

        # Create smooth path through waypoints
        idx = self.move_group_joint_indices
        path_points = []

        # Add starting configuration
        path_points.append(current_qpos[idx])

        # Interpolate between waypoints
        for i in range(len(all_waypoint_configs)):
            target_config = all_waypoint_configs[i][idx]

            if use_straight_line and i > 0:
                # For straight line motion, use more points per segment
                actual_points_per_segment = max(points_per_segment, 15)
            else:
                actual_points_per_segment = points_per_segment

            # Create interpolation from previous to current waypoint
            if i == 0:
                start_config = current_qpos[idx]
            else:
                start_config = all_waypoint_configs[i - 1][idx]

            # Linear interpolation between configurations
            for j in range(1, actual_points_per_segment + 1):
                alpha = j / actual_points_per_segment
                interpolated = (1 - alpha) * start_config + alpha * target_config
                path_points.append(interpolated)

        # Convert to numpy array for TOPP
        path_array = np.array(path_points)

        if verbose:
            print(f"Generated path with {len(path_points)} points")

        # Apply time-optimal path parameterization
        if verbose:
            ta.setup_logging("INFO")
        else:
            ta.setup_logging("WARNING")

        times, pos, vel, acc, duration = self.TOPP(path_array, time_step)
        if times is None:
            return {"status": "TOPP failed for waypoint trajectory"}

        return {
            "status": "Success",
            "time": times,
            "position": pos,
            "velocity": vel,
            "acceleration": acc,
            "duration": duration,
            "waypoint_configs": all_waypoint_configs
        }

    # ... (keeping all existing methods the same)
    def plan_qpos_to_qpos(
            self,
            goal_qposes: list,
            current_qpos,
            time_step=0.1,
            rrt_range=0.1,
            planning_time=1,
            fix_joint_limits=True,
            use_point_cloud=False,
            use_attach=False,
            planner_name="RRTConnect",
            no_simplification=False,
            constraint_function=None,
            constraint_jacobian=None,
            constraint_tolerance=1e-3,
            fixed_joint_indices=None,
            verbose=False,
    ):
        """Planning from current to goal joint configurations"""
        if fixed_joint_indices is None:
            fixed_joint_indices = []

        self.planning_world.set_use_point_cloud(use_point_cloud)
        self.planning_world.set_use_attach(use_attach)

        n = current_qpos.shape[0]
        if fix_joint_limits:
            for i in range(n):
                if i < len(self.joint_limits):  # Bounds check
                    if current_qpos[i] < self.joint_limits[i][0]:
                        current_qpos[i] = self.joint_limits[i][0] + 1e-3
                    if current_qpos[i] > self.joint_limits[i][1]:
                        current_qpos[i] = self.joint_limits[i][1] - 1e-3

        current_qpos = self.pad_qpos(current_qpos)

        self.robot.set_qpos(current_qpos, True)
        collisions = self.planning_world.collide_full()
        if len(collisions) != 0:
            print("Invalid start state!")
            for collision in collisions:
                print(f"{collision.link_name1} and {collision.link_name2} collide!")

        idx = self.move_group_joint_indices

        # Safe extraction of goal positions
        goal_qpos_ = []
        for i in range(len(goal_qposes)):
            if len(goal_qposes[i]) > max(idx) if idx else False:
                goal_qpos_.append(goal_qposes[i][idx])
            else:
                print(f"Warning: goal_qpos {i} has insufficient joints")
                # Try to use what we have
                goal_qpos_.append(goal_qposes[i][:len(idx)])

        fixed_joints = set()
        for joint_idx in fixed_joint_indices:
            if joint_idx < len(current_qpos):  # Bounds check
                fixed_joints.add(ompl.FixedJoint(0, joint_idx, current_qpos[joint_idx]))

        if len(current_qpos[idx]) != len(goal_qpos_[0]):
            print(f"Warning: size mismatch - current: {len(current_qpos[idx])}, goal: {len(goal_qpos_[0])}")

        status, path = self.planner.plan(
            current_qpos[idx],
            goal_qpos_,
            range=rrt_range,
            time=planning_time,
            fixed_joints=fixed_joints,
            planner_name=planner_name,
            no_simplification=no_simplification,
            constraint_function=constraint_function,
            constraint_jacobian=constraint_jacobian,
            constraint_tolerance=constraint_tolerance,
            verbose=verbose,
        )

        if status == "Exact solution":
            if verbose:
                ta.setup_logging("INFO")
            else:
                ta.setup_logging("WARNING")
            times, pos, vel, acc, duration = self.TOPP(path, time_step)
            if times is None:
                return {"status": "TOPP Failed"}
            return {
                "status": "Success",
                "time": times,
                "position": pos,
                "velocity": vel,
                "acceleration": acc,
                "duration": duration,
            }
        else:
            return {"status": "RRT Failed. %s" % status}

    def transform_goal_to_wrt_base(self, goal_pose):
        """Transform goal pose from world frame to base frame"""
        base_pose = self.robot.get_base_pose()
        base_tf = np.eye(4)
        base_tf[0:3, 3] = base_pose[:3]
        base_tf[0:3, 0:3] = quat2mat(base_pose[3:])
        goal_tf = np.eye(4)
        goal_tf[0:3, 3] = goal_pose[:3]
        goal_tf[0:3, 0:3] = quat2mat(goal_pose[3:])
        goal_tf = np.linalg.inv(base_tf).dot(goal_tf)
        new_goal_pose = np.zeros(7)
        new_goal_pose[:3] = goal_tf[0:3, 3]
        new_goal_pose[3:] = mat2quat(goal_tf[0:3, 0:3])
        return new_goal_pose

    def plan_qpos_to_pose(
            self,
            goal_pose,
            current_qpos,
            mask=None,
            time_step=0.1,
            rrt_range=0.1,
            planning_time=1,
            fix_joint_limits=True,
            use_point_cloud=False,
            use_attach=False,
            wrt_world=True,
            planner_name="RRTConnect",
            no_simplification=False,
            constraint_function=None,
            constraint_jacobian=None,
            constraint_tolerance=1e-3,
            verbose=False,
    ):
        """Plan from start configuration to goal pose of end-effector"""
        if mask is None:
            mask = []

        # Fix joint limits if requested
        n = current_qpos.shape[0]
        if fix_joint_limits:
            for i in range(n):
                if i < len(self.joint_limits):  # Bounds check
                    if current_qpos[i] < self.joint_limits[i][0]:
                        current_qpos[i] = self.joint_limits[i][0] + 1e-3
                    if current_qpos[i] > self.joint_limits[i][1]:
                        current_qpos[i] = self.joint_limits[i][1] - 1e-3

        if wrt_world:
            goal_pose = self.transform_goal_to_wrt_base(goal_pose)

        ik_status, goal_qpos = self.IK(goal_pose, current_qpos, mask)
        if ik_status != "Success":
            return {"status": ik_status}

        if verbose:
            print("IK results:")
            for i in range(len(goal_qpos)):
                print(goal_qpos[i])

        return self.plan_qpos_to_qpos(
            goal_qpos,
            current_qpos,
            time_step,
            rrt_range,
            planning_time,
            fix_joint_limits,
            use_point_cloud,
            use_attach,
            planner_name,
            no_simplification,
            constraint_function,
            constraint_jacobian,
            constraint_tolerance,
            verbose=verbose,
        )

    def plan_screw(
            self,
            target_pose,
            qpos,
            qpos_step=0.1,
            time_step=0.1,
            use_point_cloud=False,
            use_attach=False,
            wrt_world=True,
            verbose=False,
    ):
        """Plan screw motion - using simpler version that works"""
        self.planning_world.set_use_point_cloud(use_point_cloud)
        self.planning_world.set_use_attach(use_attach)
        qpos = self.pad_qpos(qpos.copy())
        self.robot.set_qpos(qpos, True)

        if wrt_world:
            target_pose = self.transform_goal_to_wrt_base(target_pose)

        def pose7D2mat(pose):
            mat = np.eye(4)
            mat[0:3, 3] = pose[:3]
            mat[0:3, 0:3] = quat2mat(pose[3:])
            return mat

        def skew(vec):
            return np.array([
                [0, -vec[2], vec[1]],
                [vec[2], 0, -vec[0]],
                [-vec[1], vec[0], 0],
            ])

        def pose2exp_coordinate(pose: np.ndarray) -> tuple[np.ndarray, float]:
            def rot2so3(rotation: np.ndarray):
                assert rotation.shape == (3, 3)
                if np.isclose(rotation.trace(), 3):
                    return np.zeros(3), 1
                if np.isclose(rotation.trace(), -1):
                    return np.zeros(3), -1e6
                theta = np.arccos((rotation.trace() - 1) / 2)
                omega = (
                        1
                        / 2
                        / np.sin(theta)
                        * np.array([
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ]).T
                )
                return omega, theta

            omega, theta = rot2so3(pose[:3, :3])
            if theta < -1e5:
                return omega, theta
            ss = skew(omega)
            inv_left_jacobian = (
                    np.eye(3) / theta
                    - 0.5 * ss
                    + (1.0 / theta - 0.5 / np.tan(theta / 2)) * ss @ ss
            )
            v = inv_left_jacobian @ pose[:3, 3]
            return np.concatenate([v, omega]), theta

        self.pinocchio_model.compute_forward_kinematics(qpos)
        ee_index = self.link_name_2_idx[self.move_group]
        current_p = pose7D2mat(self.pinocchio_model.get_link_pose(ee_index))
        target_p = pose7D2mat(target_pose)
        relative_transform = target_p @ np.linalg.inv(current_p)

        omega, theta = pose2exp_coordinate(relative_transform)

        if theta < -1e4:
            return {"status": "screw plan failed. - theta < -1e4"}
        omega = omega.reshape((-1, 1)) * theta

        index = self.move_group_joint_indices
        path = [np.copy(qpos[index])]

        max_iterations = 10000
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            self.pinocchio_model.compute_full_jacobian(qpos)
            J = self.pinocchio_model.get_link_jacobian(ee_index, local=False)
            delta_q = np.linalg.pinv(J) @ omega
            delta_q *= qpos_step / (np.linalg.norm(delta_q) + 1e-6)  # Avoid division by zero
            delta_twist = J @ delta_q

            flag = False
            if np.linalg.norm(delta_twist) > np.linalg.norm(omega):
                ratio = np.linalg.norm(omega) / np.linalg.norm(delta_twist)
                delta_q = delta_q * ratio
                delta_twist = delta_twist * ratio
                flag = True

            qpos += delta_q.reshape(-1)
            omega -= delta_twist

            def check_joint_limit(q):
                n = min(len(q), len(self.joint_limits))  # Use minimum to avoid index error
                for i in range(n):
                    if (
                            q[i] < self.joint_limits[i][0] - 1e-3
                            or q[i] > self.joint_limits[i][1] + 1e-3
                    ):
                        return False
                return True

            within_joint_limit = check_joint_limit(qpos)

            # Safe index extraction
            if len(qpos) > max(index) if index else False:
                self.planning_world.set_qpos_all(qpos[index])
            else:
                print(f"Warning: qpos size issue in plan_screw")
                return {"status": "screw plan failed - index error"}

            collide = self.planning_world.collide()

            if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                if np.linalg.norm(delta_twist) < 1e-4:
                    flag = True  # Converged successfully
                else:
                    return {"status": "screw plan failed - np.linalg.norm(delta_twist) < 1e-4"}

            path.append(np.copy(qpos[index]))

            if flag:
                if verbose:
                    ta.setup_logging("INFO")
                else:
                    ta.setup_logging("WARNING")
                times, pos, vel, acc, duration = self.TOPP(np.vstack(path), time_step)
                if times is None:
                    return {"status": "TOPP failed in screw planning"}
                return {
                    "status": "Success",
                    "time": times,
                    "position": pos,
                    "velocity": vel,
                    "acceleration": acc,
                    "duration": duration,
                }

        return {"status": f"screw plan failed - max iterations ({max_iterations}) reached"}

    # Additional utility methods
    def update_point_cloud(self, pc, radius=1e-3):
        """Update point cloud for collision checking"""
        self.planning_world.update_point_cloud(pc, radius)

    def update_attached_tool(self, fcl_collision_geometry, pose, link_id=-1):
        """Update attached tool geometry"""
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_tool(fcl_collision_geometry, link_id, pose)

    def update_attached_sphere(self, radius, pose, link_id=-1):
        """Attach a sphere to a link"""
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_sphere(radius, link_id, pose)

    def update_attached_box(self, size, pose, link_id=-1):
        """Attach a box to a link"""
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_box(size, link_id, pose)

    def update_attached_mesh(self, mesh_path, pose, link_id=-1):
        """Attach a mesh to a link"""
        if link_id == -1:
            link_id = self.move_group_link_id
        self.planning_world.update_attached_mesh(mesh_path, link_id, pose)

    def set_base_pose(self, pose):
        """Set the base pose of the robot"""
        self.robot.set_base_pose(pose)

    def set_normal_object(self, name, collision_object):
        """Add or update a collision object in the scene"""
        self.planning_world.set_normal_object(name, collision_object)

    def remove_normal_object(self, name):
        """Remove a collision object from the scene"""
        return self.planning_world.remove_normal_object(name)

    def plan_waypoints(
            self,
            waypoints,
            current_qpos,
            waypoint_types=None,
            time_step=0.1,
            use_point_cloud=False,
            use_attach=False,
            wrt_world=True,
            smooth_factor=0.5,
            critical_point_pause_time=0.1,
            use_cartesian_interpolation=True,  # NEW: Enable Cartesian space interpolation
            cartesian_density=50,  # NEW: Points per meter for Cartesian interpolation
            verbose=False,
    ):
        """
        Plan smooth trajectory through multiple waypoints with optional critical points

        Args:
            waypoints: List of target poses (7D: position + quaternion)
            current_qpos: Current joint configuration
            waypoint_types: List of waypoint types ('smooth' or 'critical')
            smooth_factor: Blending radius for smooth waypoints (0=sharp, 1=very smooth)
            critical_point_pause_time: Pause duration at critical points
            use_cartesian_interpolation: If True, interpolate in Cartesian space for straight lines
            cartesian_density: Number of interpolation points per meter in Cartesian space
        """
        import scipy.interpolate as si
        from scipy.signal import savgol_filter
        from scipy.spatial.transform import Rotation, Slerp

        if waypoint_types is None:
            waypoint_types = ['smooth'] * len(waypoints)

        self.planning_world.set_use_point_cloud(use_point_cloud)
        self.planning_world.set_use_attach(use_attach)

        current_qpos = self.pad_qpos(current_qpos.copy())
        idx = self.move_group_joint_indices

        # Step 1: Cartesian space interpolation if requested
        if use_cartesian_interpolation:
            interpolated_waypoints = []
            interpolated_types = []

            # Get current TCP pose
            self.robot.set_qpos(current_qpos, True)
            self.pinocchio_model.compute_forward_kinematics(current_qpos)
            ee_index = self.link_name_2_idx[self.move_group]
            current_pose = self.pinocchio_model.get_link_pose(ee_index)

            # Add current position as start
            all_poses = [current_pose]
            all_types = ['start']

            # Convert waypoints to base frame if needed
            for i, wp in enumerate(waypoints):
                if wrt_world:
                    wp = self.transform_goal_to_wrt_base(wp)
                all_poses.append(wp)
                all_types.append(waypoint_types[i])

            # Interpolate between consecutive waypoints
            for i in range(len(all_poses) - 1):
                start_pose = all_poses[i]
                end_pose = all_poses[i + 1]
                start_type = all_types[i]
                end_type = all_types[i + 1]

                # Calculate distance for interpolation density
                pos_distance = np.linalg.norm(end_pose[:3] - start_pose[:3])

                # Determine interpolation method based on waypoint types
                if end_type == 'critical' or (use_cartesian_interpolation and pos_distance > 0.001):
                    # Use Cartesian interpolation for critical points or when requested
                    n_points = max(3, int(pos_distance * cartesian_density))

                    # Position interpolation (linear in Cartesian space)
                    pos_interp = np.linspace(start_pose[:3], end_pose[:3], n_points)

                    # Orientation interpolation (SLERP)
                    start_rot = Rotation.from_quat(start_pose[3:])
                    end_rot = Rotation.from_quat(end_pose[3:])

                    # Use SLERP for smooth rotation interpolation
                    key_times = [0, 1]
                    key_rots = Rotation.from_quat([start_pose[3:], end_pose[3:]])
                    slerp = Slerp(key_times, key_rots)
                    times = np.linspace(0, 1, n_points)
                    rot_interp = slerp(times)

                    # Combine position and orientation
                    for j in range(n_points):
                        if j == 0 and i > 0:  # Skip duplicate start point
                            continue
                        pose = np.concatenate([pos_interp[j], rot_interp[j].as_quat()])
                        interpolated_waypoints.append(pose)

                        # Determine type for interpolated point
                        if j == n_points - 1:
                            interpolated_types.append(end_type)
                        else:
                            interpolated_types.append('smooth')
                else:
                    # For non-critical smooth points, use fewer interpolation points
                    if i > 0:  # Skip start point
                        interpolated_waypoints.append(end_pose)
                        interpolated_types.append(end_type)

            # Add pause at critical points
            final_waypoints = []
            final_types = []
            for i, (wp, wtype) in enumerate(zip(interpolated_waypoints, interpolated_types)):
                final_waypoints.append(wp)
                final_types.append(wtype)

                if wtype == 'critical' and critical_point_pause_time > 0:
                    # Add duplicate points for pause
                    n_pause = max(2, int(critical_point_pause_time / time_step))
                    for _ in range(n_pause - 1):
                        final_waypoints.append(wp)
                        final_types.append('critical')

            waypoints = final_waypoints
            waypoint_types = final_types

        # Step 2: Convert all waypoints to joint space
        all_qpos_waypoints = []
        failed_waypoints = []
        temp_qpos = current_qpos.copy()

        for i, waypoint in enumerate(waypoints):
            if not use_cartesian_interpolation and wrt_world:
                waypoint = self.transform_goal_to_wrt_base(waypoint)

            # Use previous solution as seed for continuity
            ik_status, goal_qpos_list = self.IK(waypoint, temp_qpos, n_init_qpos=5)

            if ik_status != "Success":
                # Try harder for critical points
                if waypoint_types[i] == 'critical':
                    ik_status, goal_qpos_list = self.IK(waypoint, temp_qpos, n_init_qpos=20)

                if ik_status != "Success":
                    failed_waypoints.append(i)
                    if verbose:
                        print(f"IK failed for waypoint {i} (type: {waypoint_types[i]})")

                    # For critical points, this is a failure
                    if waypoint_types[i] == 'critical':
                        return {"status": f"IK failed for critical waypoint {i}"}
                    continue

            # Select IK solution with minimum joint movement
            best_qpos = None
            min_dist = float('inf')

            for qpos_candidate in goal_qpos_list:
                # Check collision
                self.robot.set_qpos(qpos_candidate, True)
                has_collision = len(self.planning_world.collide_full()) > 0

                # Calculate joint movement
                dist = np.linalg.norm(qpos_candidate[idx] - temp_qpos[idx])

                # Prefer collision-free solutions
                if not has_collision:
                    dist *= 0.1  # Heavily prefer collision-free

                if dist < min_dist:
                    min_dist = dist
                    best_qpos = qpos_candidate

            if best_qpos is None:
                best_qpos = goal_qpos_list[0]

            all_qpos_waypoints.append({
                'qpos': best_qpos[idx],
                'type': waypoint_types[i],
                'index': i
            })
            temp_qpos = best_qpos

        if len(all_qpos_waypoints) == 0:
            return {"status": "All waypoints failed IK"}

        # Step 3: Build path in joint space
        combined_path = []

        # For Cartesian interpolation, we already have dense waypoints
        if use_cartesian_interpolation:
            # Simply extract joint positions
            for wp in all_qpos_waypoints:
                combined_path.append(wp['qpos'])
        else:
            # Original joint space interpolation with smoothing
            current_pos = current_qpos[idx].copy()
            combined_path.append(current_pos)

            for i, wp in enumerate(all_qpos_waypoints):
                target_qpos = wp['qpos']

                # Simple linear interpolation in joint space
                distance = np.linalg.norm(target_qpos - current_pos)
                n_points = max(2, int(distance * 20))

                for j in range(1, n_points):
                    t = j / (n_points - 1)
                    interp_pos = current_pos * (1 - t) + target_qpos * t
                    combined_path.append(interp_pos)

                current_pos = target_qpos

        combined_path = np.array(combined_path)

        # Step 4: Post-processing for smoothness (optional for joint-space paths)
        if not use_cartesian_interpolation and len(combined_path) > 5:
            # Apply mild smoothing only for joint-space interpolation
            window_length = min(5, len(combined_path))
            if window_length % 2 == 0:
                window_length -= 1

            smoothed_path = np.zeros_like(combined_path)
            for j in range(combined_path.shape[1]):
                try:
                    smoothed_path[:, j] = savgol_filter(
                        combined_path[:, j],
                        window_length,
                        2,
                        mode='nearest'
                    )
                except:
                    smoothed_path[:, j] = combined_path[:, j]

            # Preserve exact critical points
            for i, wp in enumerate(all_qpos_waypoints):
                if wp['type'] == 'critical':
                    # Find closest point in path and restore it
                    distances = np.linalg.norm(smoothed_path - wp['qpos'], axis=1)
                    closest_idx = np.argmin(distances)
                    smoothed_path[closest_idx] = wp['qpos']

            combined_path = smoothed_path

        # Step 5: Remove redundant points
        unique_path = [combined_path[0]]
        for i in range(1, len(combined_path)):
            if np.linalg.norm(combined_path[i] - unique_path[-1]) > 1e-6:
                unique_path.append(combined_path[i])
        combined_path = np.array(unique_path)

        # Step 6: Validate path continuity
        combined_path = self.validate_path_continuity(combined_path, max_joint_delta=0.2)

        # Step 7: Apply time-optimal parameterization
        if verbose:
            ta.setup_logging("INFO")
        else:
            ta.setup_logging("WARNING")

        times, pos, vel, acc, duration = self.TOPP(combined_path, time_step)

        if times is None:
            # Try with reduced limits
            if verbose:
                print("TOPP failed, trying with reduced limits...")

            original_vel_limits = self.joint_vel_limits.copy()
            original_acc_limits = self.joint_acc_limits.copy()

            self.joint_vel_limits *= 0.6
            self.joint_acc_limits *= 0.6

            times, pos, vel, acc, duration = self.TOPP(combined_path, time_step)

            self.joint_vel_limits = original_vel_limits
            self.joint_acc_limits = original_acc_limits

            if times is None:
                return {"status": "TOPP Failed for waypoint trajectory"}

        return {
            "status": "Success",
            "time": times,
            "position": pos,
            "velocity": vel,
            "acceleration": acc,
            "duration": duration,
            "failed_waypoints": failed_waypoints,
            "num_waypoints": len(all_qpos_waypoints),
            "interpolation_mode": "cartesian" if use_cartesian_interpolation else "joint"
        }

    def validate_path_continuity(self, path, max_joint_delta=0.2):
        """
        Validate path continuity and fix discontinuities

        Args:
            path: numpy array of joint configurations
            max_joint_delta: maximum allowed joint angle change between consecutive points

        Returns:
            Fixed path with ensured continuity
        """
        if len(path) <= 1:
            return path

        fixed_path = [path[0]]

        for i in range(1, len(path)):
            delta = path[i] - fixed_path[-1]

            # Check for discontinuities
            if np.max(np.abs(delta)) > max_joint_delta:
                # Insert intermediate points
                n_intermediate = int(np.max(np.abs(delta)) / max_joint_delta) + 1
                for j in range(1, n_intermediate + 1):
                    alpha = j / n_intermediate
                    intermediate = fixed_path[-1] + alpha * delta
                    fixed_path.append(intermediate)
            else:
                fixed_path.append(path[i])

        return np.array(fixed_path)

    def smooth_path_with_constraints(self, path, smoothness=0.5):
        """
        Apply constrained smoothing to maintain path shape while reducing noise

        Args:
            path: numpy array of joint configurations
            smoothness: smoothing factor (0=no smoothing, 1=maximum smoothing)

        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path

        # Use exponential moving average for smoothing
        alpha = 1.0 - smoothness
        smoothed = np.zeros_like(path)
        smoothed[0] = path[0]

        for i in range(1, len(path)):
            smoothed[i] = alpha * path[i] + (1 - alpha) * smoothed[i - 1]

        # Ensure endpoints are preserved
        smoothed[0] = path[0]
        smoothed[-1] = path[-1]

        return smoothed

    def smooth_ik_solutions(self, ik_solutions, window_size=3):
        """
        Smooth IK solutions to reduce jitter

        Args:
            ik_solutions: List of joint configurations from IK
            window_size: Size of smoothing window

        Returns:
            Smoothed IK solutions
        """
        if len(ik_solutions) <= window_size:
            return ik_solutions

        smoothed = []
        for i in range(len(ik_solutions)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(ik_solutions), i + window_size // 2 + 1)

            # Average nearby solutions
            window = ik_solutions[start_idx:end_idx]
            avg_solution = np.mean(window, axis=0)
            smoothed.append(avg_solution)

        return smoothed

    def adaptive_cartesian_density(self, start_pose, end_pose, task_type="general"):
        """
        Calculate adaptive density based on task requirements

        Args:
            start_pose: Starting pose
            end_pose: Ending pose
            task_type: Type of task ("grasp", "draw", "push", "transport")

        Returns:
            Appropriate cartesian density
        """
        distance = np.linalg.norm(end_pose[:3] - start_pose[:3])

        if task_type == "grasp":
            # Lower density for grasping to reduce jitter
            return min(30, max(5, int(20 * distance)))
        elif task_type == "draw":
            # High density for drawing accuracy
            return min(150, max(10, int(100 * distance)))
        elif task_type == "push":
            # Medium-high density for pushing
            return min(80, max(10, int(60 * distance)))
        elif task_type == "transport":
            # Low density for transport
            return min(20, max(3, int(15 * distance)))
        else:
            return min(50, max(5, int(30 * distance)))