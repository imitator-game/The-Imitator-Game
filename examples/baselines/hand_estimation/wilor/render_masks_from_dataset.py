#!/usr/bin/env python3
"""
Render Hand Masks from LeRobot Dataset

This script reads hand pose data from a LeRobot dataset and re-renders hand masks
using the MANO model. It can be used to regenerate visualization masks or validate
the stored hand pose data.

Usage:
    python render_masks_from_dataset.py \
        --dataset_root /path/to/dataset \
        --repo_id repo_name \
        --output_dir /path/to/output \
        --model_cfg configs/model_config.yaml
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
        """
        Read specific frames from video.

        Args:
            video_path: Path to video file
            frame_indices: List of frame indices to read

        Returns:
            np.ndarray: [N, H, W, C] RGB frames in uint8 format
        """
        if self.backend == "torchcodec":
            return self._read_torchcodec(video_path, frame_indices)
        elif self.backend == "decord":
            return self._read_decord(video_path, frame_indices)
        else:
            return self._read_opencv(video_path, frame_indices)

    def _read_torchcodec(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Read using torchcodec (fastest, best for LeRobot)."""
        decoder = self.VideoDecoder(video_path)
        max_frame = decoder.metadata.num_frames - 1

        # Clip indices to valid range
        valid_indices = [min(idx, max_frame) for idx in frame_indices]

        try:
            # Batch read frames
            frame_batch = decoder.get_frames_at(indices=valid_indices)
            frames_tensor = frame_batch.data  # [N, C, H, W]

            # Convert to numpy: [N, C, H, W] -> [N, H, W, C]
            frames_np = frames_tensor.permute(0, 2, 3, 1).cpu().numpy()

            # Ensure uint8
            if frames_np.dtype != np.uint8:
                frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)

            return frames_np

        except Exception as e:
            print(f"⚠️  Torchcodec failed for {video_path}: {e}")
            # Fallback to frame-by-frame
            frames = []
            for idx in valid_indices:
                try:
                    frame_tensor = decoder[idx]  # [C, H, W]
                    frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()
                    if frame_np.dtype != np.uint8:
                        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
                    frames.append(frame_np)
                except:
                    # Use black frame on error
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

            return np.array(frames)

    def _read_decord(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Read using decord (very fast)."""
        vr = self.VideoReader(video_path, ctx=self.decord_cpu(0))
        max_frame = len(vr) - 1

        # Clip indices
        valid_indices = [min(idx, max_frame) for idx in frame_indices]

        # Batch read
        frames = vr.get_batch(valid_indices).asnumpy()  # [N, H, W, C] RGB uint8
        return frames

    def _read_opencv(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """Read using OpenCV (slowest but most compatible)."""
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
                # Use last valid frame or black frame
                if original_idx > 0 and result_frames[original_idx - 1] is not None:
                    result_frames[original_idx] = result_frames[original_idx - 1]
                else:
                    result_frames[original_idx] = np.zeros((480, 640, 3), dtype=np.uint8)

        cap.release()

        return np.array(result_frames)


def is_all_zero(x: np.ndarray, eps: float = 1e-6) -> bool:
            return np.all(np.abs(x) < eps)

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
    
class AdaptiveEMAFilter:
    def __init__(self, alpha: float, eps: float):
        self.alpha = alpha
        self.eps = eps
        self.state = None

    def reset(self):
        self.state = None

    def update(self, x: np.ndarray) -> np.ndarray:
        if is_all_zero(x):
            return x

        if self.state is None:
            self.state = x.copy()
            return x

        delta = np.linalg.norm(x - self.state)
        print(delta)

        if delta >= self.eps:
            self.state = x.copy()
            return x

        # 5️⃣ Small jitter → EMA
        self.state = self.alpha * x + (1 - self.alpha) * self.state
        return self.state

def load_mano_model(model_cfg_path: str, device: torch.device) -> MANO:
    """Load MANO model from config."""
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    
    # Update MANO config paths
    if 'DATA_DIR' in model_cfg.MANO:
        model_cfg.defrost()
        model_cfg.MANO.DATA_DIR = './mano_data/'
        model_cfg.MANO.MODEL_PATH = './mano_data/'
        model_cfg.MANO.MEAN_PARAMS = './mano_data/mano_mean_params.npz'
        model_cfg.freeze()
    
    # Create MANO model
    mano_cfg = {k.lower(): v for k, v in dict(model_cfg.MANO).items()}
    mano = MANO(**mano_cfg)
    mano = mano.to(device)
    mano.eval()
    
    return mano, model_cfg


def axis_angle_to_rotmat(axis_angle: np.ndarray) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix.
    
    Args:
        axis_angle: Array of shape (..., 3) containing axis-angle vectors
    
    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    if isinstance(axis_angle, np.ndarray):
        axis_angle = torch.from_numpy(axis_angle).float()
    
    original_shape = axis_angle.shape
    if axis_angle.ndim == 1:
        axis_angle = axis_angle.unsqueeze(0)
    
    rotmat = aa_to_rotmat(axis_angle)
    
    # Reshape to original shape (excluding last dimension)
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
    """
    Convert MANO parameters (axis-angle format) to vertices and joints.
    
    Args:
        mano: MANO model
        global_orient_aa: Global orientation in axis-angle format (3,)
        hand_pose_aa: Hand pose in axis-angle format (45,) - 15 joints * 3
        betas: Shape parameters (10,)
        device: torch device
    
    Returns:
        vertices: Mesh vertices (778, 3)
        joints: Hand joints (21, 3)
    """
    # Convert axis-angle to rotation matrices
    global_orient_rotmat = axis_angle_to_rotmat(global_orient_aa)  # (3, 3)
    hand_pose_rotmat = axis_angle_to_rotmat(hand_pose_aa.reshape(-1, 3))  # (15, 3, 3)
    
    # Prepare inputs for MANO
    batch_size = 1
    global_orient = global_orient_rotmat.unsqueeze(0).to(device)  # (1, 1, 3, 3)
    hand_pose = hand_pose_rotmat.unsqueeze(0).to(device)  # (1, 15, 3, 3)
    betas_tensor = torch.from_numpy(betas).float().unsqueeze(0).to(device)  # (1, 10)
    
    # Reshape for MANO
    global_orient = global_orient.reshape(batch_size, -1, 3, 3)
    hand_pose = hand_pose.reshape(batch_size, -1, 3, 3)
    
    # Forward through MANO
    with torch.no_grad():
        mano_output = mano(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas_tensor,
            pose2rot=False
        )
    
    vertices = mano_output.vertices[0].cpu()  # (778, 3)
    joints = mano_output.joints[0].cpu()  # (21, 3)
    
    return vertices, joints


def render_hands(
    renderer: Renderer,
    all_vertices: List[torch.Tensor],
    all_cam_t: List[np.ndarray],
    all_is_right: List[bool],
    focal_length: np.ndarray,
    img_size: Tuple[int, int],
    mesh_base_colors: Optional[List[Tuple[float, float, float]]] = None
) -> np.ndarray:
    """
    Render multiple hands together in one image.
    
    Args:
        renderer: Renderer instance
        all_vertices: List of mesh vertices, each (778, 3)
        all_cam_t: List of camera translations, each (3,)
        all_is_right: List of boolean flags indicating right hand
        focal_length: Focal length (2,) or scalar
        img_size: Image size (H, W)
        mesh_base_colors: Optional list of colors for each hand
    
    Returns:
        Rendered mask image (H, W, 4) RGBA
    """
    if len(all_vertices) == 0:
        H, W = img_size
        return np.zeros((H, W, 4), dtype=np.float32)
    
    H, W = img_size
    
    # Convert focal length to scalar if needed
    if isinstance(focal_length, np.ndarray):
        if focal_length.ndim > 0:
            focal_length = focal_length[0] if len(focal_length) > 0 else 5000.0
        else:
            focal_length = float(focal_length)
    
    # Prepare vertices and colors
    vertices_list = []
    is_right_list = []
    hand_colors = []
    
    for i, (vertices, is_right) in enumerate(zip(all_vertices, all_is_right)):
        vertices_np = vertices.numpy()
        # Adjust vertices for left/right hand
        if not is_right:
            vertices_np[:, 0] = -vertices_np[:, 0]
        
        vertices_list.append(vertices_np)
        is_right_list.append(1 if is_right else 0)
        
        # Set color
        if mesh_base_colors and i < len(mesh_base_colors):
            hand_colors.append(mesh_base_colors[i])
        else:
            # Default: red for right hand, purple for left hand
            hand_colors.append(LIGHT_GREEN if is_right else LIGHT_PURPLE)
    
    # Render all hands together
    cam_view = renderer.render_rgba_multiple(
        vertices=vertices_list,
        cam_t=all_cam_t,
        render_res=[W, H],  # [width, height]
        focal_length=focal_length,
        is_right=is_right_list,
        mesh_base_color=hand_colors,
        scene_bg_color=(0, 0, 0),
    )
    
    return cam_view


def overlay_mask_on_image(img_rgb: np.ndarray, mask_rgba: np.ndarray) -> np.ndarray:
    """
    Overlay a RGBA mask on an RGB image.
    
    Args:
        img_rgb: Original RGB image (H, W, 3) uint8
        mask_rgba: Mask image (H, W, 4) float32, values in [0, 1]
    
    Returns:
        Overlaid RGB image (H, W, 3) uint8
    """
    H, W = img_rgb.shape[:2]
    
    # Ensure mask matches image size
    if mask_rgba.shape[:2] != (H, W):
        mask_rgba = cv2.resize(mask_rgba, (W, H))
    
    # Convert image to float
    img = img_rgb.astype(np.float32) / 255.0  # (H, W, 3)
    
    # Extract mask RGB and alpha
    mask_rgb = mask_rgba[:, :, :3]  # (H, W, 3)
    alpha = mask_rgba[:, :, 3:4]    # (H, W, 1)
    
    # Blend: img * (1 - alpha) + mask_rgb * alpha
    out = img * (1.0 - alpha) + mask_rgb * alpha
    
    return (255 * out).clip(0, 255).astype(np.uint8)


def process_dataset(
    dataset_root: str,
    repo_id: str,
    output_dir: str,
    model_cfg_path: str,
    episodes: Optional[List[int]] = None,
    cameras: Optional[List[str]] = None,
    save_images: bool = True,
    save_videos: bool = False,
    fps: int = 30,
    ema: bool = False,
    video_backend: str = "torchcodec",
):
    """
    Process LeRobot dataset and render hand masks.
    
    Args:
        dataset_root: Root directory of the dataset
        repo_id: Repository ID of the dataset
        output_dir: Output directory for rendered masks
        model_cfg_path: Path to model config file
        episodes: List of episode indices to process (None = all)
        cameras: List of camera names to process (None = all)
        save_images: Whether to save individual mask images
        save_videos: Whether to save mask videos
        fps: FPS for video output
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MANO model
    print("Loading MANO model...")
    mano, model_cfg = load_mano_model(model_cfg_path, device)
    
    # Get model image size from config
    if hasattr(model_cfg, 'MODEL') and hasattr(model_cfg.MODEL, 'IMAGE_SIZE'):
        actual_model_image_size = model_cfg.MODEL.IMAGE_SIZE
    else:
        raise ValueError(f"Model IMAGE_SIZE not found in config: {model_cfg_path}")
    
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
    
    # Load dataset info and episodes for video reading
    dataset_path = Path(dataset_root) # TODO: Hard code. But same as other implementations.
    info = load_info(dataset_path)
    episodes_dataset = load_episodes(dataset_path)
    
    # Initialize video reader
    video_reader = VideoFrameReader(backend=video_backend)
    
    video_path_format = info.get('video_path')
    dataset_fps = info.get('fps', fps)
    
    # Get camera names from dataset features
    if cameras is None:
        cameras = []
        for key in dataset.meta.features.keys():
            if 'observation.hand' in key and 'left' in key:
                # Extract camera name from key like "observation.hand.left.cam1.mano_global_orient"
                parts = key.split('.')
                if len(parts) >= 4:
                    cam_name = parts[3]
                    if cam_name not in cameras:
                        cameras.append(cam_name)
        cameras = sorted(cameras)
        print(f"Found cameras: {cameras}")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_images:
        img_output_dir = output_path / "masks"
        img_output_dir.mkdir(exist_ok=True)
    
    if save_videos:
        video_output_dir = output_path / "videos"
        video_output_dir.mkdir(exist_ok=True)
    
    # Process each episode
    episode_indices = np.unique(dataset.hf_dataset['episode_index'])
    if episodes is not None:
        episode_indices = [ep for ep in episode_indices if ep in episodes]
    
    for ep_idx in tqdm(episode_indices, desc="Processing episodes"):
        # Get episode row for video path
        episode_row = episodes_dataset[ep_idx]
        episode_length = int(episode_row['length'])
        
        # Get frame indices for this episode
        ep_mask = np.array(dataset.hf_dataset['episode_index']) == ep_idx
        frame_indices = np.where(ep_mask)[0]
        
        if len(frame_indices) == 0:
            continue
        
        # Process each camera
        for cam_name in cameras:
            # Read original video frames for this camera
            video_key = f"observation.images.{cam_name}"
            chunk_key = f"videos/{video_key}/chunk_index"
            file_key = f"videos/{video_key}/file_index"
            
            original_frames = None
            if chunk_key in episode_row:
                chunk_idx = int(episode_row[chunk_key])
                file_idx = int(episode_row[file_key])
                
                rgb_video_path = dataset_path / video_path_format.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                
                # Calculate absolute frame indices
                from_ts_key = f"videos/{video_key}/from_timestamp"
                from_ts = float(episode_row.get(from_ts_key, 0.0))
                video_start_frame = int(from_ts * dataset_fps)
                relative_indices = list(range(episode_length))
                absolute_indices = [video_start_frame + i for i in relative_indices]
                
                # Read frames
                try:
                    original_frames = video_reader.read_frames(str(rgb_video_path), absolute_indices)  # [T, H, W, 3]
                except Exception as e:
                    print(f"⚠️  Failed to read video for episode {ep_idx}, camera {cam_name}: {e}")
                    original_frames = None
            
            # Collect frames for this camera
            episode_masks = []
            episode_overlaid = []
            if ema:
                # EMA filters (per camera, per episode)
                ema_global_orient = {
                    "left": EMAFilter(alpha=0.8),
                    "right": EMAFilter(alpha=0.8),
                }
                ema_hand_pose = {
                    "left": EMAFilter(alpha=0.25),
                    "right": EMAFilter(alpha=0.25),
                }
                ema_cam_t = {
                    "left": EMAFilter(alpha=0.4),
                    "right": EMAFilter(alpha=0.4),
                }
            
            # Get image size from first original frame if available
            img_size = None
            if original_frames is not None and len(original_frames) > 0:
                img_size = original_frames[0].shape[:2]  # (H, W) from original video
            
            # If no original frames, try to get size from first frame_data
            if img_size is None:
                first_frame_data = dataset[frame_indices[0]]
                for img_key in first_frame_data.keys():
                    if f"observation.images.{cam_name}" in img_key:
                        img = first_frame_data[img_key]
                        if isinstance(img, np.ndarray):
                            img_size = img.shape[:2]  # (H, W)
                            break
                
                if img_size is None:
                    # Default image size
                    img_size = (480, 640)
            
            # Collect hand data for all frames in this episode
            episode_hand_data = []  # List of dicts, each containing frame hand data
            
            for frame_idx in tqdm(frame_indices, desc=f"Episode {ep_idx} - {cam_name} (collecting)", leave=False):
                frame_data = dataset[frame_idx]
                
                # Get original frame if available
                original_frame = None
                if original_frames is not None:
                    frame_idx_in_episode = int(frame_idx - frame_indices[0])
                    if 0 <= frame_idx_in_episode < len(original_frames):
                        original_frame = original_frames[frame_idx_in_episode]  # (H, W, 3)
                
                # Collect hand data for both left and right hands
                all_vertices = []
                all_cam_t = []
                all_is_right = []
                
                # Extract hand data for both hands
                focal_length = None  # Will be set from first detected hand
                
                for hand_type in ['left', 'right']:
                    # Get MANO parameters
                    global_orient_key = f"observation.hand.{hand_type}.{cam_name}.mano_global_orient"
                    hand_pose_key = f"observation.hand.{hand_type}.{cam_name}.mano_hand_pose"
                    betas_key = f"observation.hand.{hand_type}.{cam_name}.mano_betas"
                    cam_t_key = f"observation.hand.{hand_type}.{cam_name}.pred_cam_t_full"
                    focal_key = f"observation.hand.{hand_type}.{cam_name}.focal_length"
                    is_right_key = f"observation.hand.{hand_type}.{cam_name}.is_right"
                    
                    if global_orient_key not in frame_data:
                        continue
                    
                    # Extract data
                    global_orient_aa = np.array(frame_data[global_orient_key]).flatten()  # (3,)
                    hand_pose_aa = np.array(frame_data[hand_pose_key]).flatten()  # (45,)
                    betas = np.array(frame_data[betas_key]).flatten()  # (10,)
                    cam_t = np.array(frame_data[cam_t_key]).flatten()  # (3,)

                    hand_id = hand_type  

                    if ema:
                        global_orient_aa = ema_global_orient[hand_id].update(global_orient_aa)
                        hand_pose_aa = ema_hand_pose[hand_id].update(hand_pose_aa)
                        cam_t = ema_cam_t[hand_id].update(cam_t)

                    hand_focal_length = np.array(frame_data[focal_key]).flatten()  # (2,) or scalar
                    is_right_flag = float(frame_data[is_right_key])
                    
                    # Skip if no hand detected (is_right == -1)
                    if is_right_flag < 0:
                        continue
                    
                    # Set focal_length from first detected hand
                    if focal_length is None:
                        focal_length = hand_focal_length
                    
                    is_right_hand = bool(is_right_flag > 0.5)
                    import time
                    t0 = time.perf_counter()
                    # Convert to vertices
                    try:
                        vertices, joints = mano_params_to_vertices(
                            mano, global_orient_aa, hand_pose_aa, betas, device
                        )
                    except Exception as e:
                        print(f"Error converting MANO params for frame {frame_idx}, {hand_type}: {e}")
                        continue
                    t1 = time.perf_counter()
                    print(f"MANO forward time: {t1 - t0:.4f} seconds")
                    # Add to lists for batch rendering
                    all_vertices.append(vertices)
                    all_cam_t.append(cam_t)
                    all_is_right.append(is_right_hand)
                
                # Store frame data for later rendering
                episode_hand_data.append({
                    'frame_idx': frame_idx,
                    'original_frame': original_frame,
                    'all_vertices': all_vertices,
                    'all_cam_t': all_cam_t,
                    'all_is_right': all_is_right,
                    'focal_length': focal_length,
                })
            
            # Render all frames for this episode
            for frame_data in tqdm(episode_hand_data, desc=f"Episode {ep_idx} - {cam_name} (rendering)", leave=False):
                frame_idx = frame_data['frame_idx']
                original_frame = frame_data['original_frame']
                all_vertices = frame_data['all_vertices']
                all_cam_t = frame_data['all_cam_t']
                all_is_right = frame_data['all_is_right']
                focal_length = frame_data['focal_length']
                
                # Render all hands together if any detected
                if len(all_vertices) > 0:
                    # Use default focal_length if none was set
                    if focal_length is None:
                        focal_length = np.array([5000.0, 5000.0])  # Default focal length
                    
                    # Scale focal_length according to image size (same as process_lerobot.py)
                    # The stored focal_length is for model_image_size, need to scale to actual image size
                    H, W = img_size
                    img_max_size = max(H, W)
                    if isinstance(focal_length, np.ndarray) and len(focal_length) > 0:
                        # Use first value if it's an array
                        base_focal = float(focal_length[0]) if focal_length.ndim > 0 else float(focal_length)
                    else:
                        base_focal = float(focal_length) if focal_length is not None else 5000.0
                    
                    # Scale focal length: base_focal / actual_model_image_size * actual_image_size
                    # This matches the scaling in process_lerobot.py:
                    # scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                    scaled_focal_length = base_focal / actual_model_image_size * img_max_size
                    
                    mask = render_hands(
                        renderer, all_vertices, all_cam_t, all_is_right, 
                        scaled_focal_length, img_size
                    )
                    episode_masks.append(mask)
                    
                    # Overlay mask on original frame if available
                    if original_frame is not None:
                        overlaid_frame = overlay_mask_on_image(original_frame, mask)
                        episode_overlaid.append(overlaid_frame)
                    else:
                        # If no original frame, use mask RGB
                        mask_rgb = (mask[:, :, :3] * 255).astype(np.uint8)
                        episode_overlaid.append(mask_rgb)
                    
                    # Save individual mask if requested
                    if save_images:
                        mask_img_path = img_output_dir / f"ep{ep_idx:04d}_cam{cam_name}_frame{frame_idx:06d}.png"
                        if original_frame is not None:
                            # Save overlaid image
                            overlaid_bgr = cv2.cvtColor(episode_overlaid[-1], cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(mask_img_path), overlaid_bgr)
                        else:
                            # Save mask only
                            mask_rgb = (mask[:, :, :3] * 255).astype(np.uint8)
                            cv2.imwrite(str(mask_img_path), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
                else:
                    # No hands detected, create empty mask
                    H, W = img_size
                    empty_mask = np.zeros((H, W, 4), dtype=np.float32)
                    episode_masks.append(empty_mask)
                    
                    # Use original frame if available, otherwise black frame
                    if original_frame is not None:
                        episode_overlaid.append(original_frame)
                    else:
                        episode_overlaid.append(np.zeros((H, W, 3), dtype=np.uint8))
            
            # Save video if requested
            if save_videos and len(episode_overlaid) > 0:
                video_path = video_output_dir / f"ep{ep_idx:04d}_cam{cam_name}_overlaid.mp4"
                H, W = episode_overlaid[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))
                
                for frame in episode_overlaid:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                
                video_writer.release()
                print(f"Saved overlaid video: {video_path}")
    
    print(f"Processing complete! Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render hand masks from LeRobot dataset")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory of the dataset")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Repository ID of the dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for rendered masks")
    parser.add_argument("--model_cfg", type=str, required=True,
                        help="Path to model config file")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Episode indices to process (default: all)")
    parser.add_argument("--cameras", type=str, nargs="+", default=None,
                        help="Camera names to process (default: all)")
    parser.add_argument("--save_images", action="store_true", default=False,
                        help="Save individual mask images")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save mask videos")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for video output")
    parser.add_argument("--ema", action="store_true", default=False,
                        help="Using ema smoothing.")
    parser.add_argument("--video_backend", type=str, default="torchcodec",
                        choices=["torchcodec", "decord", "opencv"],
                        help="Video decoding backend")
    
    args = parser.parse_args()
    
    process_dataset(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        model_cfg_path=args.model_cfg,
        episodes=args.episodes,
        cameras=args.cameras,
        save_images=args.save_images,
        save_videos=args.save_videos,
        fps=args.fps,
        ema=args.ema,
        video_backend=args.video_backend,
    )


if __name__ == "__main__":
    main()

