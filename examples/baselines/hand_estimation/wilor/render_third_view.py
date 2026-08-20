#!/usr/bin/env python3
"""
Render Third View with Cross-Camera Hand Projection

Features:
1. Play the four camera views together in a video grid
2. Optional: compute the relative transform between cam1 and zed2i, and project cam1's hand models onto zed2i

Usage:
    python render_third_view.py \
        --dataset_root /path/to/dataset \
        --repo_id repo_name \
        --output_dir /path/to/output \
        --model_cfg configs/model_config.yaml \
        --enable_cross_view  # Enable cross-view projection
"""

import numpy as np
import torch
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# WiLoR imports
from wilor.models import MANO
from wilor.utils.geometry import aa_to_rotmat
from wilor.utils.renderer import Renderer
from wilor.configs import get_config

# LeRobot imports
from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_episodes, load_info

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
LIGHT_GREEN = (0.2, 0.8, 0.2)
LIGHT_ORANGE = (1.0, 0.5, 0.0)  # color for hands projected from another view
LIGHT_CYAN = (0.0, 0.8, 0.8)    # color for hands projected from another view
BIAS = np.array([0.09566993, 0.00638343, 0.00618631])

# ============ Video Frame Reader ============

class VideoFrameReader:
    """Efficient video frame reader with multiple backend support."""

    def __init__(self, backend: str = "torchcodec"):
        self.backend = backend
        self._init_backend()

    def _init_backend(self):
        """Initialize the appropriate backend."""
        if self.backend == "torchcodec":
            try:
                from torchcodec.decoders import VideoDecoder
                self.VideoDecoder = VideoDecoder
            except ImportError:
                print("⚠️  torchcodec not available, falling back to decord")
                self.backend = "decord"
                self._init_backend()

        if self.backend == "decord":
            try:
                from decord import VideoReader, cpu
                self.VideoReader = VideoReader
                self.decord_cpu = cpu
            except ImportError:
                print("⚠️  decord not available, falling back to opencv")
                self.backend = "opencv"

    def read_frames(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Read specific frames from video."""
        if self.backend == "torchcodec":
            return self._read_torchcodec(video_path, frame_indices)
        elif self.backend == "decord":
            return self._read_decord(video_path, frame_indices)
        else:
            return self._read_opencv(video_path, frame_indices)

    def _read_torchcodec(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Read using torchcodec."""
        decoder = self.VideoDecoder(video_path)
        max_frame = decoder.metadata.num_frames - 1
        valid_indices = [min(idx, max_frame) for idx in frame_indices]

        try:
            frame_batch = decoder.get_frames_at(indices=valid_indices)
            frames_tensor = frame_batch.data
            frames_np = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()
            if frames_np.dtype != np.uint8:
                frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)
            return frames_np
        except Exception as e:
            print(f"⚠️  Torchcodec failed: {e}")
            frames = []
            for idx in valid_indices:
                try:
                    frame_tensor = decoder[idx]
                    frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
                    if frame_np.dtype != np.uint8:
                        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
                    frames.append(frame_np)
                except:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            return np.array(frames)

    def _read_decord(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Read using decord."""
        vr = self.VideoReader(video_path, ctx=self.decord_cpu(0))
        max_frame = len(vr) - 1
        valid_indices = [min(idx, max_frame) for idx in frame_indices]
        frames = vr.get_batch(valid_indices).asnumpy()
        return frames

    def _read_opencv(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Read using OpenCV."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        valid_indices = [min(idx, max_frame) for idx in frame_indices]
        sorted_indices = sorted(enumerate(valid_indices), key=lambda x: x[1])
        result_frames = [None] * len(valid_indices)

        current_pos = 0
        for original_idx, frame_idx in sorted_indices:
            if frame_idx != current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_pos = frame_idx

            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_frames[original_idx] = frame_rgb
                current_pos += 1
            else:
                if original_idx > 0 and result_frames[original_idx - 1] is not None:
                    result_frames[original_idx] = result_frames[original_idx - 1]
                else:
                    result_frames[original_idx] = np.zeros((480, 640, 3), dtype=np.uint8)

        cap.release()
        return np.array(result_frames)


def is_all_zero(x: np.ndarray, eps: float = 1e-6) -> bool:
    return np.all(np.abs(x) < eps)


def cubic_interpolate_cam_t(
    valid_indices: List[int],
    valid_cam_t: List[np.ndarray],
    target_indices: List[int]
) -> Dict[int, np.ndarray]:
    """
    Use cubic spline interpolation to estimate cam_t for missing frames.
    
    Args:
        valid_indices: Frame indices where cam_t is available
        valid_cam_t: cam_t values at those indices (list of (3,) arrays)
        target_indices: Frame indices where we need interpolated cam_t
    
    Returns:
        Dict mapping target_index -> interpolated cam_t
    """
    from scipy.interpolate import CubicSpline
    
    if len(valid_indices) < 2:
        # Not enough points for interpolation, return None for all targets
        return {idx: None for idx in target_indices}
    
    valid_indices = np.array(valid_indices)
    valid_cam_t = np.array(valid_cam_t)  # (N, 3)
    
    # Sort by index
    sort_order = np.argsort(valid_indices)
    valid_indices = valid_indices[sort_order]
    valid_cam_t = valid_cam_t[sort_order]
    
    result = {}
    
    # Use cubic spline for each dimension
    try:
        if len(valid_indices) >= 4:
            # Cubic spline needs at least 4 points for good behavior
            spline_x = CubicSpline(valid_indices, valid_cam_t[:, 0], bc_type='natural')
            spline_y = CubicSpline(valid_indices, valid_cam_t[:, 1], bc_type='natural')
            spline_z = CubicSpline(valid_indices, valid_cam_t[:, 2], bc_type='natural')
            
            for idx in target_indices:
                # Clamp to valid range for extrapolation safety
                clamped_idx = np.clip(idx, valid_indices[0], valid_indices[-1])
                result[idx] = np.array([
                    spline_x(clamped_idx),
                    spline_y(clamped_idx),
                    spline_z(clamped_idx)
                ])
        else:
            # Fall back to linear interpolation if not enough points
            for idx in target_indices:
                result[idx] = np.interp(
                    idx,
                    valid_indices,
                    valid_cam_t,
                    left=valid_cam_t[0],
                    right=valid_cam_t[-1]
                ) if len(valid_cam_t.shape) == 1 else np.array([
                    np.interp(idx, valid_indices, valid_cam_t[:, i]) 
                    for i in range(3)
                ])
    except Exception as e:
        print(f"⚠️  Cubic interpolation failed: {e}, falling back to linear")
        for idx in target_indices:
            result[idx] = np.array([
                np.interp(idx, valid_indices, valid_cam_t[:, i]) 
                for i in range(3)
            ])
    
    return result


class EMAFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.state = None

    def reset(self):
        self.state = None

    def update(self, x: np.ndarray) -> np.ndarray:
        if is_all_zero(x):
            self.reset()
            return x
        if self.state is None:
            self.state = x.copy()
        else:
            self.state = self.alpha * x + (1 - self.alpha) * self.state
        return self.state


def load_mano_model(model_cfg_path: str, device: torch.device) -> MANO:
    """Load MANO model from config."""
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    
    if 'DATA_DIR' in model_cfg.MANO:
        model_cfg.defrost()
        model_cfg.MANO.DATA_DIR = './mano_data/'
        model_cfg.MANO.MODEL_PATH = './mano_data/'
        model_cfg.MANO.MEAN_PARAMS = './mano_data/mano_mean_params.npz'
        model_cfg.freeze()
    
    mano_cfg = {k.lower(): v for k, v in dict(model_cfg.MANO).items()}
    mano = MANO(**mano_cfg)
    mano = mano.to(device)
    mano.eval()
    
    return mano, model_cfg


def axis_angle_to_rotmat(axis_angle: np.ndarray) -> torch.Tensor:
    """Convert axis-angle to rotation matrix."""
    if isinstance(axis_angle, np.ndarray):
        axis_angle = torch.from_numpy(axis_angle).float()
    
    original_shape = axis_angle.shape
    if axis_angle.ndim == 1:
        axis_angle = axis_angle.unsqueeze(0)
    
    rotmat = aa_to_rotmat(axis_angle)
    
    if len(original_shape) == 1:
        rotmat = rotmat.squeeze(0)
    elif len(original_shape) > 2:
        rotmat = rotmat.reshape(*original_shape[:-1], 3, 3)
    
    return rotmat


def mano_params_to_vertices(
    mano: MANO,
    global_orient_aa: np.ndarray,
    hand_pose_aa: np.ndarray,
    betas: np.ndarray,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert MANO parameters to vertices and joints."""
    global_orient_rotmat = axis_angle_to_rotmat(global_orient_aa)
    hand_pose_rotmat = axis_angle_to_rotmat(hand_pose_aa.reshape(-1, 3))
    
    batch_size = 1
    global_orient = global_orient_rotmat.unsqueeze(0).to(device)
    hand_pose = hand_pose_rotmat.unsqueeze(0).to(device)
    betas_tensor = torch.from_numpy(betas).float().unsqueeze(0).to(device)
    
    global_orient = global_orient.reshape(batch_size, -1, 3, 3)
    hand_pose = hand_pose.reshape(batch_size, -1, 3, 3)
    
    with torch.no_grad():
        mano_output = mano(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas_tensor,
            pose2rot=False
        )
    
    vertices = mano_output.vertices[0].cpu()
    joints = mano_output.joints[0].cpu()
    
    return vertices, joints


# MANO joint indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12


def compute_finger_distances(joints: torch.Tensor) -> Dict[str, float]:
    """
    Compute distances between thumb tip and other finger tips.
    
    Args:
        joints: MANO joints tensor of shape (21, 3)
    
    Returns:
        Dict with distances in meters
    """
    joints_np = joints.numpy() if isinstance(joints, torch.Tensor) else joints
    
    thumb_tip = joints_np[THUMB_TIP]
    index_tip = joints_np[INDEX_TIP]
    middle_tip = joints_np[MIDDLE_TIP]
    
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
    thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
    avg_dist = (thumb_index_dist + thumb_middle_dist) / 2
    
    return {
        'thumb_index': float(thumb_index_dist),
        'thumb_middle': float(thumb_middle_dist),
        'avg': float(avg_dist),
    }


def draw_distance_info(
    frame: np.ndarray,
    distances: Dict[str, Dict[str, float]],
    position: Tuple[int, int] = None
) -> np.ndarray:
    """
    Draw finger distance info on frame.
    
    Args:
        frame: RGB frame (H, W, 3)
        distances: Dict mapping hand_id (e.g., 'left', 'right') to distance dict
        position: Top-right corner position (x, y), auto-calculated if None
    """
    frame = frame.copy()
    H, W = frame.shape[:2]
    
    if position is None:
        x_start = W - 280
        y_start = 30
    else:
        x_start, y_start = position
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22
    
    y = y_start
    
    for hand_id, dist in distances.items():
        # Hand label
        color = (0, 255, 0) if 'right' in hand_id.lower() else (255, 0, 255)
        cv2.putText(frame, f"{hand_id}:", (x_start, y), font, font_scale, color, thickness)
        y += line_height
        
        # Distances (convert to cm for readability)
        thumb_idx_cm = dist['thumb_index'] * 100
        thumb_mid_cm = dist['thumb_middle'] * 100
        avg_cm = dist['avg'] * 100
        
        cv2.putText(frame, f"  Thumb-Index: {thumb_idx_cm:.1f}cm", (x_start, y), font, font_scale, (255, 255, 255), thickness)
        y += line_height
        cv2.putText(frame, f"  Thumb-Middle: {thumb_mid_cm:.1f}cm", (x_start, y), font, font_scale, (255, 255, 255), thickness)
        y += line_height
        cv2.putText(frame, f"  Average: {avg_cm:.1f}cm", (x_start, y), font, font_scale, (255, 255, 0), thickness)
        y += line_height + 5
    
    return frame


def render_hands(
    renderer: Renderer,
    all_vertices: List[torch.Tensor],
    all_cam_t: List[np.ndarray],
    all_is_right: List[bool],
    focal_length: np.ndarray,
    img_size: Tuple[int, int],
    mesh_base_colors: Optional[List[Tuple[float, float, float]]] = None,
    mark_origins: Optional[List[bool]] = None
) -> np.ndarray:
    """
    Render multiple hands together in one image.
    
    Args:
        mark_origins: If provided, render a red sphere at origin (0,0,0) for hands where mark_origins[i] is True
    """
    if len(all_vertices) == 0:
        H, W = img_size
        return np.zeros((H, W, 4), dtype=np.float32)
    
    H, W = img_size
    
    if isinstance(focal_length, np.ndarray):
        if focal_length.ndim > 0:
            focal_length = focal_length[0] if len(focal_length) > 0 else 5000.0
        else:
            focal_length = float(focal_length)
    
    vertices_list = []
    is_right_list = []
    hand_colors = []
    
    for i, (vertices, is_right) in enumerate(zip(all_vertices, all_is_right)):
        vertices_np = vertices.numpy() if isinstance(vertices, torch.Tensor) else vertices
        if not is_right:
            vertices_np = vertices_np.copy()
            vertices_np[:, 0] = -vertices_np[:, 0]
        
        vertices_list.append(vertices_np)
        is_right_list.append(1 if is_right else 0)
        
        if mesh_base_colors and i < len(mesh_base_colors):
            hand_colors.append(mesh_base_colors[i])
        else:
            hand_colors.append(LIGHT_GREEN if is_right else LIGHT_PURPLE)
    
    cam_view = renderer.render_rgba_multiple(
        vertices=vertices_list,
        cam_t=all_cam_t,
        render_res=[W, H],
        focal_length=focal_length,
        is_right=is_right_list,
        mesh_base_color=hand_colors,
        scene_bg_color=(0, 0, 0),
        origin_markers=mark_origins,
    )
    
    return cam_view


def overlay_mask_on_image(img_rgb: np.ndarray, mask_rgba: np.ndarray) -> np.ndarray:
    """Overlay a RGBA mask on an RGB image."""
    H, W = img_rgb.shape[:2]
    
    if mask_rgba.shape[:2] != (H, W):
        mask_rgba = cv2.resize(mask_rgba, (W, H))
    
    img = img_rgb.astype(np.float32) / 255.0
    mask_rgb = mask_rgba[:, :, :3]
    alpha = mask_rgba[:, :, 3:4]
    out = img * (1.0 - alpha) + mask_rgb * alpha
    
    return (255 * out).clip(0, 255).astype(np.uint8)


def create_grid_video(frames_dict: Dict[str, np.ndarray], target_size: Tuple[int, int] = (480, 640)) -> np.ndarray:
    """
    Create a 2x2 grid from 4 camera views.
    
    Args:
        frames_dict: Dict mapping camera name to frame (H, W, 3)
        target_size: Target size for each cell (H, W)
    
    Returns:
        Grid frame (2*H, 2*W, 3)
    """
    H, W = target_size
    grid = np.zeros((2 * H, 2 * W, 3), dtype=np.uint8)
    
    # Sort camera names for consistent ordering
    cam_names = sorted(frames_dict.keys())
    
    positions = [(0, 0), (0, W), (H, 0), (H, W)]  # top-left, top-right, bottom-left, bottom-right
    
    for i, cam_name in enumerate(cam_names[:4]):  # Max 4 cameras
        frame = frames_dict[cam_name]
        if frame is None:
            continue
        
        # Resize if needed
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        
        y, x = positions[i]
        grid[y:y+H, x:x+W] = frame
        
        # Add camera name label
        cv2.putText(grid, cam_name, (x + 10, y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return grid

def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def compute_relative_transform(
    global_orient_src: np.ndarray,
    cam_t_src: np.ndarray,
    global_orient_dst: np.ndarray,
    cam_t_dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the relative transform from the src camera to the dst camera.
    
    Args:
        global_orient_src: global orientation of the hand in the source camera (axis-angle, 3)
        cam_t_src: camera translation in the source camera (3,)
        global_orient_dst: global orientation of the hand in the target camera (axis-angle, 3)
        cam_t_dst: camera translation in the target camera (3,)
    
    Returns:
        relative_transform: relative transformation matrix (4, 4)
    """
    # Convert axis-angle to a rotation matrix
    R_src = axis_angle_to_rotmat(global_orient_src).numpy()  # (3, 3)
    R_dst = axis_angle_to_rotmat(global_orient_dst).numpy()  # (3, 3)
    
    R_rel = R_dst @ R_src.T  # Relative rotation: from the src hand orientation to the dst hand orientation
    
    T_rel = np.zeros((4, 4), dtype=np.float32)
    T_rel[:3, :3] = R_rel
    T_rel[:3, 3] = (cam_t_dst - BIAS) - R_rel @ (cam_t_src - BIAS)
    T_rel[3, 3] = 1.0

    return T_rel


def apply_transform_to_vertices(
    vertices: torch.Tensor,
    cam_t: np.ndarray,
    relative_transform: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the relative transform to the vertices and the camera translation.
    
    Args:
        vertices: original vertices (778, 3)
        cam_t: original camera translation (3,)
        relative_transform: relative transformation matrix (4, 4)
    
    Returns:
        transformed_vertices: transformed vertices (778, 3)
        transformed_cam_t: transformed camera translation (3,)
    """
    V = vertices.detach().cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
    V = V.copy()  # ⚠️ Must copy, otherwise the original cached vertices would be modified!
    cam_t = cam_t.copy()
    
    R = relative_transform[:3, :3]
    t = relative_transform[:3, 3]
    V -= BIAS
    V_dst = (R @ V.T).T
    V_dst += BIAS
    cam_t_dst = R @ (cam_t - BIAS) + t + BIAS
    return V_dst, cam_t_dst


def process_dataset(
    dataset_root: str,
    repo_id: str,
    output_dir: str,
    model_cfg_path: str,
    episodes: Optional[List[int]] = None,
    cameras: Optional[List[str]] = None,
    save_videos: bool = True,
    save_images: bool = False,
    fps: int = 30,
    ema: bool = False,
    video_backend: str = "torchcodec",
    enable_cross_view: bool = False,
    source_cameras: Optional[List[str]] = None,
    target_cameras: Optional[List[str]] = None,
    grid_size: Tuple[int, int] = (480, 640),
    use_mean_betas: bool = False,
):
    """
    Process LeRobot dataset and render hand masks with grid view and cross-camera projection.
    
    Args:
        dataset_root: Root directory of the dataset
        repo_id: Repository ID of the dataset
        output_dir: Output directory for rendered videos
        model_cfg_path: Path to model config file
        episodes: List of episode indices to process (None = all)
        cameras: List of camera names to process (None = all)
        save_videos: Whether to save videos
        fps: FPS for video output
        ema: Whether to use EMA smoothing
        video_backend: Video decoding backend
        enable_cross_view: Enable cross-camera hand projection
        source_cameras: List of source cameras for cross-view projection
        target_cameras: List of target cameras for cross-view projection (paired with source_cameras)
        grid_size: Size of each cell in the grid (H, W)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MANO model
    print("Loading MANO model...")
    mano, model_cfg = load_mano_model(model_cfg_path, device)
    
    if hasattr(model_cfg, 'MODEL') and hasattr(model_cfg.MODEL, 'IMAGE_SIZE'):
        actual_model_image_size = model_cfg.MODEL.IMAGE_SIZE
    else:
        raise ValueError(f"Model IMAGE_SIZE not found in config")
    
    # Load renderer
    print("Loading renderer...")
    renderer = Renderer(model_cfg, mano.faces)
    
    # Load dataset
    print(f"Loading dataset: {repo_id} from {dataset_root}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=episodes,
        download_videos=False,
    )
    
    print(f"Dataset loaded: {len(dataset)} frames, {dataset.meta.total_episodes} episodes")
    
    dataset_path = Path(dataset_root)
    info = load_info(dataset_path)
    episodes_dataset = load_episodes(dataset_path)
    
    video_reader = VideoFrameReader(backend=video_backend)
    video_path_format = info.get('video_path')
    dataset_fps = info.get('fps', fps)
    
    # Get camera names
    if cameras is None:
        cameras = []
        for key in dataset.meta.features.keys():
            if 'observation.hand' in key and 'left' in key:
                parts = key.split('.')
                if len(parts) >= 4:
                    cam_name = parts[3]
                    if cam_name not in cameras:
                        cameras.append(cam_name)
        cameras = sorted(cameras)
        print(f"Found cameras: {cameras}")
    
    # Validate cross-view cameras
    if enable_cross_view:
        if not source_cameras or not target_cameras:
            print(f"⚠️  source_cameras and target_cameras must be provided, disabling cross-view")
            enable_cross_view = False
        else:
            # Filter out invalid camera pairs
            valid_pairs = []
            for src, dst in zip(source_cameras, target_cameras):
                if src not in cameras:
                    print(f"⚠️  Source camera {src} not found, skipping this pair")
                elif dst not in cameras:
                    print(f"⚠️  Target camera {dst} not found, skipping this pair")
                else:
                    valid_pairs.append((src, dst))
            
            if not valid_pairs:
                print(f"⚠️  No valid camera pairs, disabling cross-view")
                enable_cross_view = False
            else:
                # Update source/target cameras to only valid ones
                source_cameras = [p[0] for p in valid_pairs]
                target_cameras = [p[1] for p in valid_pairs]
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_videos:
        video_output_dir = output_path / "videos"
        video_output_dir.mkdir(exist_ok=True)
    
    if save_images:
        img_output_dir = output_path / "frames"
        img_output_dir.mkdir(exist_ok=True)
    
    # Process each episode
    episode_indices = np.unique(dataset.hf_dataset['episode_index'])
    if episodes is not None:
        episode_indices = [ep for ep in episode_indices if ep in episodes]
    
    for ep_idx in tqdm(episode_indices, desc="Processing episodes"):
        episode_row = episodes_dataset[ep_idx]
        episode_length = int(episode_row['length'])
        
        ep_mask = np.array(dataset.hf_dataset['episode_index']) == ep_idx
        frame_indices = np.where(ep_mask)[0]
        
        if len(frame_indices) == 0:
            continue
        
        # Read video frames for all cameras
        camera_frames = {}
        for cam_name in cameras:
            video_key = f"observation.images.{cam_name}"
            chunk_key = f"videos/{video_key}/chunk_index"
            file_key = f"videos/{video_key}/file_index"
            
            if chunk_key in episode_row:
                chunk_idx = int(episode_row[chunk_key])
                file_idx = int(episode_row[file_key])
                
                rgb_video_path = dataset_path / video_path_format.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                
                from_ts_key = f"videos/{video_key}/from_timestamp"
                from_ts = float(episode_row.get(from_ts_key, 0.0))
                video_start_frame = int(from_ts * dataset_fps)
                relative_indices = list(range(episode_length))
                absolute_indices = [video_start_frame + i for i in relative_indices]
                
                try:
                    camera_frames[cam_name] = video_reader.read_frames(str(rgb_video_path), absolute_indices)
                except Exception as e:
                    print(f"⚠️  Failed to read video for episode {ep_idx}, camera {cam_name}: {e}")
                    camera_frames[cam_name] = None
            else:
                camera_frames[cam_name] = None
        
        # Initialize EMA filters for each camera
        ema_filters = {}
        if ema:
            for cam_name in cameras:
                ema_filters[cam_name] = {
                    'global_orient': {'left': EMAFilter(0.8), 'right': EMAFilter(0.8)},
                    'hand_pose': {'left': EMAFilter(0.25), 'right': EMAFilter(0.25)},
                    'cam_t': {'left': EMAFilter(0.4), 'right': EMAFilter(0.4)},
                }
        
        # Compute mean betas for each camera and hand type if enabled
        # mean_betas[cam_name][hand_type] = (10,) array
        mean_betas = {}
        if use_mean_betas:
            print(f"   📊 Computing mean betas for episode {ep_idx}...")
            betas_collection = {cam_name: {'left': [], 'right': []} for cam_name in cameras}
            
            for frame_idx in frame_indices:
                frame_data = dataset[frame_idx]
                for cam_name in cameras:
                    for hand_type in ['left', 'right']:
                        betas_key = f"observation.hand.{hand_type}.{cam_name}.mano_betas"
                        is_right_key = f"observation.hand.{hand_type}.{cam_name}.is_right"
                        
                        if betas_key not in frame_data:
                            continue
                        
                        is_right_flag = float(frame_data[is_right_key])
                        if is_right_flag < 0:  # No valid hand detection
                            continue
                        
                        betas = np.array(frame_data[betas_key]).flatten()
                        if not is_all_zero(betas):
                            betas_collection[cam_name][hand_type].append(betas)
            
            # Compute mean for each camera and hand type
            for cam_name in cameras:
                mean_betas[cam_name] = {}
                for hand_type in ['left', 'right']:
                    if len(betas_collection[cam_name][hand_type]) > 0:
                        mean_betas[cam_name][hand_type] = np.mean(
                            np.array(betas_collection[cam_name][hand_type]), axis=0
                        )
                        print(f"      ✓ {cam_name}/{hand_type}: mean from {len(betas_collection[cam_name][hand_type])} frames")
                    else:
                        mean_betas[cam_name][hand_type] = None
        
        # Build cross-view projection pairs
        cross_view_pairs = []  # List of (source_cam, target_cam) tuples
        if enable_cross_view and source_cameras and target_cameras:
            if len(source_cameras) != len(target_cameras):
                print(f"   ⚠️  source_cameras and target_cameras must have same length, disabling cross-view")
            else:
                cross_view_pairs = list(zip(source_cameras, target_cameras))
        
        # Find reference frame for cross-view calibration (separately for each pair and hand type)
        # relative_transforms[(src_cam, dst_cam)][hand_type] = transform matrix
        relative_transforms = {}
        
        for src_cam, dst_cam in cross_view_pairs:
            print(f"   🔄 Computing cross-view calibration from {src_cam} to {dst_cam}...")
            relative_transforms[(src_cam, dst_cam)] = {'left': None, 'right': None}
            
            for hand_type in ['left', 'right']:
                for frame_idx in frame_indices:
                    frame_data = dataset[frame_idx]
                    
                    # Source camera
                    src_orient_key = f"observation.hand.{hand_type}.{src_cam}.mano_global_orient"
                    src_cam_t_key = f"observation.hand.{hand_type}.{src_cam}.pred_cam_t_full"
                    src_is_right_key = f"observation.hand.{hand_type}.{src_cam}.is_right"
                    
                    # Target camera
                    dst_orient_key = f"observation.hand.{hand_type}.{dst_cam}.mano_global_orient"
                    dst_cam_t_key = f"observation.hand.{hand_type}.{dst_cam}.pred_cam_t_full"
                    dst_is_right_key = f"observation.hand.{hand_type}.{dst_cam}.is_right"
                    
                    if src_orient_key not in frame_data or dst_orient_key not in frame_data:
                        continue
                    
                    src_orient = np.array(frame_data[src_orient_key]).flatten()
                    src_t = np.array(frame_data[src_cam_t_key]).flatten()
                    src_is_right = float(frame_data[src_is_right_key])
                    
                    dst_orient = np.array(frame_data[dst_orient_key]).flatten()
                    dst_t = np.array(frame_data[dst_cam_t_key]).flatten()
                    dst_is_right = float(frame_data[dst_is_right_key])
                    
                    # Check both hands are valid (non-zero and detected)
                    if (not is_all_zero(src_orient) and src_is_right >= 0 and
                        not is_all_zero(dst_orient) and dst_is_right >= 0):
                        relative_transforms[(src_cam, dst_cam)][hand_type] = compute_relative_transform(
                            src_orient, src_t,
                            dst_orient, dst_t
                        )
                        print(f"   ✓ Calibration for {hand_type} hand found at frame {frame_idx}")
                        break
                
                if relative_transforms[(src_cam, dst_cam)][hand_type] is None:
                    print(f"   ⚠️  No valid calibration frame found for {hand_type} hand")
        
        # Collect hand data for all frames
        # First pass: compute native hand vertices for all cameras
        # native_hand_cache[local_idx][cam_name][hand_type] = {'vertices': ..., 'cam_t': ..., 'is_right': ..., 'focal_length': ...}
        native_hand_cache = []
        episode_hand_data = {cam_name: [] for cam_name in cameras}
        
        for local_idx, frame_idx in enumerate(tqdm(frame_indices, desc=f"Episode {ep_idx} (collecting)", leave=False)):
            frame_data = dataset[frame_idx]
            frame_cache = {cam_name: {} for cam_name in cameras}
            
            # Collect native hands for ALL cameras first
            for cam_name in cameras:
                for hand_type in ['left', 'right']:
                    global_orient_key = f"observation.hand.{hand_type}.{cam_name}.mano_global_orient"
                    hand_pose_key = f"observation.hand.{hand_type}.{cam_name}.mano_hand_pose"
                    betas_key = f"observation.hand.{hand_type}.{cam_name}.mano_betas"
                    cam_t_key = f"observation.hand.{hand_type}.{cam_name}.pred_cam_t_full"
                    focal_key = f"observation.hand.{hand_type}.{cam_name}.focal_length"
                    is_right_key = f"observation.hand.{hand_type}.{cam_name}.is_right"
                    
                    if global_orient_key not in frame_data:
                        continue
                    
                    global_orient_aa = np.array(frame_data[global_orient_key]).flatten()
                    hand_pose_aa = np.array(frame_data[hand_pose_key]).flatten()
                    betas = np.array(frame_data[betas_key]).flatten()
                    cam_t = np.array(frame_data[cam_t_key]).flatten()
                    
                    # Use mean betas if enabled and available
                    if use_mean_betas and cam_name in mean_betas:
                        if mean_betas[cam_name].get(hand_type) is not None:
                            betas = mean_betas[cam_name][hand_type]
                    
                    if ema and cam_name in ema_filters:
                        global_orient_aa = ema_filters[cam_name]['global_orient'][hand_type].update(global_orient_aa)
                        hand_pose_aa = ema_filters[cam_name]['hand_pose'][hand_type].update(hand_pose_aa)
                        cam_t = ema_filters[cam_name]['cam_t'][hand_type].update(cam_t)
                    
                    hand_focal_length = np.array(frame_data[focal_key]).flatten()
                    is_right_flag = float(frame_data[is_right_key])
                    
                    if is_right_flag < 0:
                        continue
                    
                    is_right_hand = bool(is_right_flag > 0.5)
                    
                    try:
                        vertices, joints = mano_params_to_vertices(
                            mano, global_orient_aa, hand_pose_aa, betas, device
                        )
                        frame_cache[cam_name][hand_type] = {
                            'vertices': vertices,
                            'joints': joints,
                            'cam_t': cam_t,
                            'is_right': is_right_hand,
                            'focal_length': hand_focal_length,
                        }
                    except Exception as e:
                        continue
            
            native_hand_cache.append(frame_cache)
        
        # ============ Interpolate cam_t for target cameras with missing detections ============
        # For each target camera in cross-view pairs, compute interpolated cam_t where native detection is missing
        interpolated_cam_t = {}  # {(cam_name, hand_type): {local_idx: cam_t}}
        
        for src_cam, dst_cam in cross_view_pairs:
            for hand_type in ['left', 'right']:
                # Collect valid cam_t from target camera
                valid_indices = []
                valid_cam_t_list = []
                
                for local_idx, frame_cache in enumerate(native_hand_cache):
                    if hand_type in frame_cache.get(dst_cam, {}):
                        valid_indices.append(local_idx)
                        valid_cam_t_list.append(frame_cache[dst_cam][hand_type]['cam_t'])
                
                # Find missing indices
                all_indices = list(range(len(native_hand_cache)))
                missing_indices = [i for i in all_indices if i not in valid_indices]
                
                if missing_indices and len(valid_indices) >= 2:
                    # Use cubic interpolation
                    interpolated = cubic_interpolate_cam_t(valid_indices, valid_cam_t_list, missing_indices)
                    interpolated_cam_t[(dst_cam, hand_type)] = interpolated
                    print(f"   📈 Interpolated {len(missing_indices)} cam_t values for {dst_cam}/{hand_type}")
        
        # ============ Build episode_hand_data for each frame ============
        for local_idx, frame_cache in enumerate(native_hand_cache):
            frame_idx = frame_indices[local_idx]
            
            for cam_name in cameras:
                original_frame = None
                if camera_frames[cam_name] is not None and local_idx < len(camera_frames[cam_name]):
                    original_frame = camera_frames[cam_name][local_idx]
                
                all_vertices = []
                all_cam_t = []
                all_is_right = []
                all_colors = []
                all_mark_origins = []  # Track which hands need origin markers
                focal_length = None
                finger_distances = {}  # Store finger distances for display
                
                # Add native hands for this camera
                for hand_type in ['left', 'right']:
                    if hand_type in frame_cache[cam_name]:
                        hand_data = frame_cache[cam_name][hand_type]
                        all_vertices.append(hand_data['vertices'])
                        all_cam_t.append(hand_data['cam_t'])
                        all_is_right.append(hand_data['is_right'])
                        all_colors.append(LIGHT_GREEN if hand_data['is_right'] else LIGHT_PURPLE)
                        all_mark_origins.append(False)  # Native hands don't need origin markers
                        if focal_length is None:
                            focal_length = hand_data['focal_length']
                        
                        # Compute finger distances for native hands
                        if 'joints' in hand_data:
                            dist = compute_finger_distances(hand_data['joints'])
                            hand_label = 'Right' if hand_data['is_right'] else 'Left'
                            finger_distances[hand_label] = dist
                
                # Add cross-view projected hands (reuse cached vertices from source cameras)
                for src_cam, dst_cam in cross_view_pairs:
                    if cam_name != dst_cam:
                        continue
                    
                    pair_transforms = relative_transforms.get((src_cam, dst_cam), {})
                    
                    for hand_type in ['left', 'right']:
                        transform = pair_transforms.get(hand_type)
                        if transform is None:
                            continue
                        
                        # Check if target camera has native detection for this hand
                        target_has_native = hand_type in frame_cache.get(dst_cam, {})
                        
                        # If target has no native detection, use source vertices + interpolated cam_t
                        if not target_has_native:
                            # Use the specified src_cam directly (not any arbitrary camera)
                            if hand_type in frame_cache.get(src_cam, {}):
                                source_hand = frame_cache[src_cam][hand_type]

                                interp_key = (dst_cam, hand_type)
                                interp_cam_t = interpolated_cam_t.get(interp_key, {}).get(local_idx)

                                if interp_cam_t is not None:
                                    try:
                                        src_vertices = source_hand['vertices']
                                        src_cam_t = source_hand['cam_t']
                                        is_right_hand = source_hand['is_right']

                                        # Use transform already fetched for (src_cam, dst_cam)
                                        transformed_vertices, _ = apply_transform_to_vertices(
                                            src_vertices, src_cam_t, transform
                                        )

                                        all_vertices.append(torch.from_numpy(transformed_vertices).float())
                                        all_cam_t.append(interp_cam_t)
                                        all_is_right.append(is_right_hand)
                                        all_colors.append(LIGHT_ORANGE if is_right_hand else LIGHT_CYAN)
                                        all_mark_origins.append(True)

                                        if focal_length is None and 'focal_length' in source_hand:
                                            focal_length = source_hand['focal_length']
                                    except Exception as e:
                                        print(f"⚠️  Failed to fill missing detection at frame {frame_idx}: {e}")
                            continue
                        
                        # # Normal case: target has native detection, add projected hand
                        # if hand_type not in frame_cache.get(src_cam, {}):
                        #     continue
                        
                        # src_hand = frame_cache[src_cam][hand_type]
                        # src_vertices = src_hand['vertices']
                        # src_cam_t = src_hand['cam_t']
                        # is_right_hand = src_hand['is_right']
                        
                        # try:
                        #     # Apply transform to cached vertices
                        #     transformed_vertices, transformed_cam_t = apply_transform_to_vertices(
                        #         src_vertices, src_cam_t, transform
                        #     )
                            
                        #     all_vertices.append(torch.from_numpy(transformed_vertices).float())
                        #     all_cam_t.append(transformed_cam_t)
                        #     all_is_right.append(is_right_hand)
                        #     all_colors.append(LIGHT_ORANGE if is_right_hand else LIGHT_CYAN)
                        #     all_mark_origins.append(True)  # Projected hands get origin markers (red sphere)
                        # except Exception as e:
                        #     print(f"⚠️  Failed to project hand from {src_cam} to {dst_cam}: {e}")
                        #     continue
                
                # Count native vs projected hands
                native_count = sum(1 for c in all_colors if c in [LIGHT_GREEN, LIGHT_PURPLE])
                projected_count = len(all_colors) - native_count
                
                episode_hand_data[cam_name].append({
                    'frame_idx': frame_idx,
                    'local_idx': local_idx,
                    'original_frame': original_frame,
                    'all_vertices': all_vertices,
                    'all_cam_t': all_cam_t,
                    'all_is_right': all_is_right,
                    'all_colors': all_colors,
                    'all_mark_origins': all_mark_origins,
                    'focal_length': focal_length,
                    'native_count': native_count,
                    'projected_count': projected_count,
                    'finger_distances': finger_distances,
                })
        
        # Render and create grid videos
        grid_frames = []
        
        for local_idx in tqdm(range(len(frame_indices)), desc=f"Episode {ep_idx} (rendering)", leave=False):
            camera_overlaid = {}
            
            for cam_name in cameras:
                if local_idx >= len(episode_hand_data[cam_name]):
                    continue
                
                hand_data = episode_hand_data[cam_name][local_idx]
                original_frame = hand_data['original_frame']
                all_vertices = hand_data['all_vertices']
                all_cam_t = hand_data['all_cam_t']
                all_is_right = hand_data['all_is_right']
                all_colors = hand_data['all_colors']
                all_mark_origins = hand_data['all_mark_origins']
                focal_length = hand_data['focal_length']
                native_count = hand_data['native_count']
                projected_count = hand_data['projected_count']
                finger_distances = hand_data.get('finger_distances', {})
                
                if original_frame is None:
                    img_size = grid_size
                    camera_overlaid[cam_name] = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
                    continue
                
                img_size = original_frame.shape[:2]
                H, W = img_size
                
                if len(all_vertices) > 0:
                    if focal_length is None:
                        focal_length = np.array([5000.0, 5000.0])
                    
                    img_max_size = max(H, W)
                    if isinstance(focal_length, np.ndarray) and len(focal_length) > 0:
                        base_focal = float(focal_length[0]) if focal_length.ndim > 0 else float(focal_length)
                    else:
                        base_focal = float(focal_length) if focal_length is not None else 5000.0
                    
                    scaled_focal_length = base_focal / actual_model_image_size * img_max_size
                    
                    # Render all hands together
                    mask = render_hands(
                        renderer, all_vertices, all_cam_t, all_is_right,
                        scaled_focal_length, img_size,
                        mesh_base_colors=all_colors,
                        mark_origins=all_mark_origins
                    )
                    
                    overlaid_frame = overlay_mask_on_image(original_frame, mask)
                    
                    # Draw finger distances on the frame
                    if finger_distances:
                        overlaid_frame = draw_distance_info(overlaid_frame, finger_distances)
                    
                    camera_overlaid[cam_name] = overlaid_frame
                    
                    # Save mask images if requested
                    if save_images:
                        frame_idx = hand_data['frame_idx']
                        cam_img_dir = img_output_dir / f"ep{ep_idx:04d}" / cam_name
                        cam_img_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save native hands mask (first native_count items)
                        if native_count > 0:
                            native_vertices = all_vertices[:native_count]
                            native_cam_t = all_cam_t[:native_count]
                            native_is_right = all_is_right[:native_count]
                            native_colors = all_colors[:native_count]
                            
                            native_mask = render_hands(
                                renderer, native_vertices, native_cam_t, native_is_right,
                                scaled_focal_length, img_size,
                                mesh_base_colors=native_colors
                            )
                            native_mask_rgb = (native_mask[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
                            native_mask_bgr = cv2.cvtColor(native_mask_rgb, cv2.COLOR_RGB2BGR)
                            native_img_path = cam_img_dir / f"frame_{frame_idx:06d}_native.png"
                            cv2.imwrite(str(native_img_path), native_mask_bgr)
                        else:
                            # Save empty native mask
                            empty_mask = np.zeros((H, W, 3), dtype=np.uint8)
                            native_img_path = cam_img_dir / f"frame_{frame_idx:06d}_native.png"
                            cv2.imwrite(str(native_img_path), empty_mask)
                        
                        # Save projected hands mask (items after native_count)
                        if projected_count > 0:
                            projected_vertices = all_vertices[native_count:]
                            projected_cam_t = all_cam_t[native_count:]
                            projected_is_right = all_is_right[native_count:]
                            projected_colors = all_colors[native_count:]
                            projected_mark_origins = [True] * projected_count  # All projected hands get markers
                            
                            projected_mask = render_hands(
                                renderer, projected_vertices, projected_cam_t, projected_is_right,
                                scaled_focal_length, img_size,
                                mesh_base_colors=projected_colors,
                                mark_origins=projected_mark_origins
                            )
                            projected_mask_rgb = (projected_mask[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
                            projected_mask_bgr = cv2.cvtColor(projected_mask_rgb, cv2.COLOR_RGB2BGR)
                            projected_img_path = cam_img_dir / f"frame_{frame_idx:06d}_projected.png"
                            cv2.imwrite(str(projected_img_path), projected_mask_bgr)
                        else:
                            # Save empty projected mask
                            empty_mask = np.zeros((H, W, 3), dtype=np.uint8)
                            projected_img_path = cam_img_dir / f"frame_{frame_idx:06d}_projected.png"
                            cv2.imwrite(str(projected_img_path), empty_mask)
                else:
                    camera_overlaid[cam_name] = original_frame
                    
                    # Save empty masks if no hands detected
                    if save_images:
                        frame_idx = hand_data['frame_idx']
                        cam_img_dir = img_output_dir / f"ep{ep_idx:04d}" / cam_name
                        cam_img_dir.mkdir(parents=True, exist_ok=True)
                        empty_mask = np.zeros((H, W, 3), dtype=np.uint8)
                        cv2.imwrite(str(cam_img_dir / f"frame_{frame_idx:06d}_native.png"), empty_mask)
                        cv2.imwrite(str(cam_img_dir / f"frame_{frame_idx:06d}_projected.png"), empty_mask)
            
            # Create grid
            grid_frame = create_grid_video(camera_overlaid, grid_size)
            grid_frames.append(grid_frame)
        
        # Save grid video
        if save_videos and len(grid_frames) > 0:
            video_name = f"ep{ep_idx:04d}_grid"
            if enable_cross_view and len(cross_view_pairs) > 0:
                # Check if any transform was found
                has_any_transform = any(
                    relative_transforms.get(pair, {}).get('left') is not None or
                    relative_transforms.get(pair, {}).get('right') is not None
                    for pair in cross_view_pairs
                )
                if has_any_transform:
                    pairs_str = "_".join(f"{s}to{t}" for s, t in cross_view_pairs)
                    video_name += f"_crossview_{pairs_str}"
            video_path = video_output_dir / f"{video_name}.mp4"
            
            H, W = grid_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))
            
            for frame in grid_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            
            video_writer.release()
            print(f"Saved grid video: {video_path}")
    
    print(f"Processing complete! Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render third view with grid and cross-camera projection")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Repository ID of the dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for rendered videos")
    parser.add_argument("--model_cfg", type=str, required=True,
                        help="Path to model config file")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Episode indices to process (default: all)")
    parser.add_argument("--cameras", type=str, nargs="+", default=None,
                        help="Camera names to process (default: all)")
    parser.add_argument("--save_videos", action="store_true", default=True,
                        help="Save grid videos")
    parser.add_argument("--save_images", action="store_true", default=False,
                        help="Save individual overlay frames")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for video output")
    parser.add_argument("--ema", action="store_true", default=False,
                        help="Using EMA smoothing")
    parser.add_argument("--video_backend", type=str, default="torchcodec",
                        choices=["torchcodec", "decord", "opencv"],
                        help="Video decoding backend")
    
    # Cross-view projection options
    parser.add_argument("--enable_cross_view", action="store_true", default=False,
                        help="Enable cross-camera hand projection")
    parser.add_argument("--source_cameras", type=str, nargs="+", default=None,
                        help="Source cameras for cross-view projection (list)")
    parser.add_argument("--target_cameras", type=str, nargs="+", default=None,
                        help="Target cameras for cross-view projection (paired with source_cameras)")
    parser.add_argument("--grid_height", type=int, default=480,
                        help="Height of each grid cell")
    parser.add_argument("--grid_width", type=int, default=640,
                        help="Width of each grid cell")
    parser.add_argument("--use_mean_betas", action="store_true", default=False,
                        help="Use per-episode mean betas as shape parameter instead of per-frame betas")
    
    args = parser.parse_args()
    
    process_dataset(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        model_cfg_path=args.model_cfg,
        episodes=args.episodes,
        cameras=args.cameras,
        save_videos=args.save_videos,
        save_images=args.save_images,
        fps=args.fps,
        ema=args.ema,
        video_backend=args.video_backend,
        enable_cross_view=args.enable_cross_view,
        source_cameras=args.source_cameras,
        target_cameras=args.target_cameras,
        grid_size=(args.grid_height, args.grid_width),
        use_mean_betas=args.use_mean_betas,
    )


if __name__ == "__main__":
    main()
