"""
ManiSkill GraspNet Integration Script
Author: Assistant
This script provides functions to integrate GraspNet with ManiSkill environments.
It allows prediction of grasp poses from point clouds and conversion to ManiSkill format.
"""

import os
import sys
import numpy as np
import torch
import sapien
from typing import Optional, Tuple, List
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import euler2quat

# Add GraspNet paths
GRASPNET_ROOT = os.path.join(os.path.dirname(__file__), '../../../graspnet-baseline')
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'dataset'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))

try:
    from graspnet import GraspNet, pred_decode
    from graspnetAPI import GraspGroup
    from collision_detector import ModelFreeCollisionDetector
except ImportError as e:
    print(f"Warning: GraspNet modules not found. Make sure graspnet-baseline is properly installed. Error: {e}")


class ManiSkillGraspNetPredictor:
    """
    GraspNet predictor for ManiSkill environments.
    Provides grasp pose prediction from point clouds.
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 num_view: int = 300,
                 num_angle: int = 12,
                 num_depth: int = 4,
                 collision_thresh: float = 0.01,
                 voxel_size: float = 0.01):
        """
        Initialize GraspNet predictor.
        
        Args:
            checkpoint_path: Path to GraspNet model checkpoint
            num_view: Number of viewing angles (default: 300)
            num_angle: Number of grasp angles (default: 12)  
            num_depth: Number of grasp depths (default: 4)
            collision_thresh: Collision detection threshold (default: 0.01)
            voxel_size: Voxel size for collision detection (default: 0.01)
        """
        self.checkpoint_path = checkpoint_path
        self.num_view = num_view
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self._load_model()
        
    def _load_model(self) -> GraspNet:
        """Load and initialize GraspNet model."""
        # Initialize model
        net = GraspNet(
            input_feature_dim=0,
            num_view=self.num_view,
            num_angle=self.num_angle,
            num_depth=self.num_depth,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False
        )
        
        net.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"-> Loaded GraspNet checkpoint from {self.checkpoint_path}")
        
        # Set to evaluation mode
        net.eval()
        return net
        
    def preprocess_point_cloud(self, 
                              point_cloud: np.ndarray,
                              colors: Optional[np.ndarray] = None,
                              num_points: int = 20000) -> dict:
        """
        Preprocess point cloud for GraspNet input.
        
        Args:
            point_cloud: Point cloud array [N, 3]
            colors: Color array [N, 3] (optional)
            num_points: Number of points to sample
            
        Returns:
            Dictionary containing preprocessed data
        """
        # Handle point cloud sampling
        if len(point_cloud) >= num_points:
            idxs = np.random.choice(len(point_cloud), num_points, replace=False)
        else:
            idxs1 = np.arange(len(point_cloud))
            idxs2 = np.random.choice(len(point_cloud), num_points - len(point_cloud), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
            
        cloud_sampled = point_cloud[idxs]
        
        # Handle colors
        if colors is not None:
            color_sampled = colors[idxs]
        else:
            # Default to white if no colors provided
            color_sampled = np.ones_like(cloud_sampled)
            
        # Convert to torch tensors
        cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        cloud_tensor = cloud_tensor.to(self.device)
        
        # Prepare end_points dictionary
        end_points = {
            'point_clouds': cloud_tensor,
            'cloud_colors': color_sampled
        }
        
        return end_points, point_cloud
        
    def predict_grasps(self, point_cloud: np.ndarray, 
                      colors: Optional[np.ndarray] = None,
                      num_points: int = 20000,
                      use_collision_detection: bool = True) -> GraspGroup:
        """
        Predict grasp poses from point cloud.
        
        Args:
            point_cloud: Input point cloud [N, 3]
            colors: Point colors [N, 3] (optional)
            num_points: Number of points to sample for inference
            use_collision_detection: Whether to filter colliding grasps
            
        Returns:
            GraspGroup containing predicted grasps
        """
        # Preprocess point cloud
        end_points, original_cloud = self.preprocess_point_cloud(
            point_cloud, colors, num_points
        )
        
        # Forward pass
        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
            
        # Convert to GraspGroup
        gg_array = grasp_preds[0].detach().cpu().numpy()
        grasp_group = GraspGroup(gg_array)
        
        # Apply collision detection if requested
        if use_collision_detection and self.collision_thresh > 0:
            grasp_group = self._filter_collisions(grasp_group, original_cloud)
            
        return grasp_group
        
    def _filter_collisions(self, grasp_group: GraspGroup, point_cloud: np.ndarray) -> GraspGroup:
        """Filter out colliding grasps using collision detection."""
        try:
            detector = ModelFreeCollisionDetector(point_cloud, voxel_size=self.voxel_size)
            collision_mask = detector.detect(
                grasp_group, 
                approach_dist=0.05, 
                collision_thresh=self.collision_thresh
            )
            return grasp_group[~collision_mask]
        except Exception as e:
            print(f"Warning: Collision detection failed: {e}")
            return grasp_group
            
    def get_best_grasp_pose(self, point_cloud: np.ndarray,
                           colors: Optional[np.ndarray] = None,
                           num_points: int = 20000,
                           use_collision_detection: bool = True) -> sapien.Pose:
        """
        Get the best grasp pose in ManiSkill/SAPIEN format.
        
        Args:
            point_cloud: Input point cloud [N, 3]
            colors: Point colors [N, 3] (optional)
            num_points: Number of points for inference
            use_collision_detection: Whether to filter colliding grasps
            
        Returns:
            Best grasp pose as sapien.Pose
        """
        # Predict grasps
        grasp_group = self.predict_grasps(
            point_cloud, colors, num_points, use_collision_detection
        )
        
        if len(grasp_group) == 0:
            raise RuntimeError("No valid grasps found")
            
        # Skip NMS, just sort by score to get the best one
        grasp_group.sort_by_score()
        
        # Get the best grasp
        best_grasp = grasp_group[0]
        
        # Convert to SAPIEN pose
        return self._graspnet_to_sapien_pose(best_grasp)
        
    def get_multiple_grasp_poses(self, point_cloud: np.ndarray,
                                colors: Optional[np.ndarray] = None,
                                num_points: int = 20000,
                                use_collision_detection: bool = True,
                                max_grasps: int = 5) -> List[sapien.Pose]:
        """
        Get multiple grasp poses ranked by score.
        
        Args:
            point_cloud: Input point cloud [N, 3]  
            colors: Point colors [N, 3] (optional)
            num_points: Number of points for inference
            use_collision_detection: Whether to filter colliding grasps
            max_grasps: Maximum number of grasps to return
            
        Returns:
            List of grasp poses as sapien.Pose objects
        """
        # Predict grasps
        grasp_group = self.predict_grasps(
            point_cloud, colors, num_points, use_collision_detection
        )
        
        if len(grasp_group) == 0:
            raise RuntimeError("No valid grasps found")
            
        # Skip NMS, just sort by score
        grasp_group.sort_by_score()
        
        # Get top grasps
        num_grasps = min(max_grasps, len(grasp_group))
        poses = []
        
        for i in range(num_grasps):
            pose = self._graspnet_to_sapien_pose(grasp_group[i])
            poses.append(pose)
            
        return poses
        
    def _graspnet_to_sapien_pose(self, grasp) -> sapien.Pose:
        """
        Convert GraspNet grasp to SAPIEN pose.
        
        Args:
            grasp: Single grasp from GraspGroup
            
        Returns:
            Corresponding sapien.Pose
        """
        # Extract grasp parameters
        center = grasp.translation  # Grasp center [3,]
        rotation_matrix = grasp.rotation_matrix  # Rotation matrix [3,3]
        
        # Convert rotation matrix to quaternion (w,x,y,z format for SAPIEN)
        quat_xyzw = mat2quat(rotation_matrix)  # Returns [w,x,y,z]
        quat_wxyz = [quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]]  # SAPIEN uses [w,x,y,z]
        
        # Create SAPIEN pose
        return sapien.Pose(p=center, q=quat_wxyz)


def predict_grasp_from_point_cloud(point_cloud: np.ndarray,
                                  checkpoint_path: str,
                                  colors: Optional[np.ndarray] = None,
                                  predictor: Optional[ManiSkillGraspNetPredictor] = None) -> sapien.Pose:
    """
    Convenience function to predict grasp pose from point cloud.
    
    Args:
        point_cloud: Input point cloud [N, 3]
        checkpoint_path: Path to GraspNet checkpoint
        colors: Point colors [N, 3] (optional)
        predictor: Existing predictor instance (optional, for efficiency)
        
    Returns:
        Best grasp pose as sapien.Pose
    """
    if predictor is None:
        predictor = ManiSkillGraspNetPredictor(checkpoint_path)
        
    return predictor.get_best_grasp_pose(point_cloud, colors)


# Example usage function
def example_usage():
    """Example of how to use the GraspNet predictor."""
    # Initialize predictor
    checkpoint_path = "/path/to/graspnet/checkpoint.tar"
    predictor = ManiSkillGraspNetPredictor(checkpoint_path)
    
    # Generate sample point cloud (replace with your actual point cloud)
    sample_points = np.random.rand(1000, 3)
    
    try:
        # Get best grasp pose
        best_pose = predictor.get_best_grasp_pose(sample_points)
        print(f"Best grasp pose: {best_pose}")
        
        # Get multiple grasp poses
        multiple_poses = predictor.get_multiple_grasp_poses(sample_points, max_grasps=3)
        print(f"Found {len(multiple_poses)} grasp poses")
        
    except RuntimeError as e:
        print(f"Grasp prediction failed: {e}")


if __name__ == "__main__":
    example_usage()