#!/usr/bin/env python3
"""
WiLoR Hand Estimation Pipeline for LeRobot Datasets

Processes human demonstration videos from LeRobot format,
extracts hand pose estimation using WiLoR model,
and saves results back to LeRobot format.

Features:
- Extract hand pose data and save to LeRobot format
- Optional: Save visualization videos with hand overlay
"""

import os
import json
import numpy as np
import torch
import cv2
import time
import argparse
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader

# WiLoR imports
from wilor.models import WiLoR, load_wilor, MANO
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full
from wilor.utils.geometry import aa_to_rotmat
from wilor.configs import get_config
from ultralytics import YOLO

# LeRobot imports
from lerobot.datasets.utils import load_episodes, load_info

# LeRobot dataset creation
from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset

# ============ Video Writer Helper ============

class VideoWriter:
    """Helper class for writing visualization videos."""
    
    def __init__(self, output_path: str, fps: int, frame_size: Tuple[int, int]):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to save the video
            fps: Frames per second
            frame_size: (width, height) of frames
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.frames = []
        
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the video buffer. Frame should be RGB uint8."""
        self.frames.append(frame)
        
    def save(self):
        """Save all frames to video file."""
        if not self.frames:
            return
            
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use H264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            self.output_path, 
            fourcc, 
            self.fps, 
            self.frame_size
        )
        
        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            
        writer.release()
        self.frames = []
        
    def __len__(self):
        return len(self.frames)


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
        import cv2

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


# ============ Dataset Configuration ============

@dataclass
class HumanVideoDataConfig:
    """Configuration for human video dataset loading."""
    root: str = "demos"
    dataset_file: Optional[str] = None
    cameras: List[str] = field(default_factory=lambda: ["cam1", "cam2"])
    video_backend: str = "torchcodec"
    fps: int = 30


# ============ Human Video Dataset ============

class HumanVideoDataset(Dataset):
    """
    Dataset for loading human demonstration videos from LeRobot format.
    
    Each __getitem__ returns a full episode's video data for all cameras.
    """

    def __init__(self, config: HumanVideoDataConfig):
        self.config = config
        self.repo_episodes = {}
        self.repo_info = {}
        self.repo_paths = {}
        self.episode_list = []

        # Load dataset configuration
        with open(config.dataset_file, 'r') as f:
            dataset_configs = json.load(f)

        for ds_cfg in tqdm(dataset_configs, desc="Loading datasets"):
            repo_id = ds_cfg.get("repo_id")
            ds_root = os.path.join(config.root, ds_cfg.get("root", repo_id))
            repo_path = Path(ds_root)

            try:
                info = load_info(repo_path)
                episodes_dataset = load_episodes(repo_path)

                self.repo_info[repo_id] = info
                self.repo_episodes[repo_id] = episodes_dataset
                self.repo_paths[repo_id] = repo_path

                num_episodes = len(episodes_dataset)
                for ep_idx in range(num_episodes):
                    self.episode_list.append((repo_id, ep_idx))
                    
                print(f"  ✓ Loaded {repo_id}: {num_episodes} episodes")
            except Exception as e:
                print(f"  ⚠️  Failed to load {repo_id}: {e}")

        # Initialize video reader
        self.video_reader = VideoFrameReader(backend=config.video_backend)

        print(f"✓ HumanVideoDataset initialized")
        print(f"  - Total episodes: {len(self.episode_list)}")
        print(f"  - Cameras: {config.cameras}")

    def __len__(self) -> int:
        return len(self.episode_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a full episode's video data.
        
        Returns:
            dict: {
                'video': List[np.ndarray],  # List of [T, H, W, 3] arrays per camera
                'repo_id': str,
                'episode_idx': int,
                'frame_indices': List[int]
            }
        """
        repo_id, episode_idx = self.episode_list[idx]
        repo_path = self.repo_paths[repo_id]
        episode_row = self.repo_episodes[repo_id][episode_idx]
        info = self.repo_info[repo_id]

        episode_length = int(episode_row['length'])
        frame_indices = list(range(episode_length))

        video_path_format = info.get('video_path')
        all_view_frames = []
        fps = info['fps']

        for cam_name in self.config.cameras:
            video_key = f"observation.images.{cam_name}"
            
            # Get video chunk and file indices
            chunk_key = f"videos/{video_key}/chunk_index"
            file_key = f"videos/{video_key}/file_index"
            
            if chunk_key not in episode_row:
                print(f"  ⚠️  Camera {cam_name} not found in episode")
                continue
                
            chunk_idx = int(episode_row[chunk_key])
            file_idx = int(episode_row[file_key])

            rgb_video_path = repo_path / video_path_format.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )

            # Calculate absolute frame indices
            from_ts_key = f"videos/{video_key}/from_timestamp"
            from_ts = float(episode_row.get(from_ts_key, 0.0))
            video_start_frame = int(from_ts * fps)
            absolute_indices = [video_start_frame + i for i in frame_indices]

            # Read frames
            frames = self.video_reader.read_frames(str(rgb_video_path), absolute_indices)
            all_view_frames.append(frames)  # [T, H, W, 3] numpy RGB

        return {
            'video': all_view_frames,
            'repo_id': repo_id,
            'episode_idx': episode_idx,
            'frame_indices': frame_indices
        }


# ============ MANO Helper Functions ============

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
    
    return mano


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


def compute_joints_with_mano(
    mano: MANO,
    global_orient_aa: np.ndarray,
    hand_pose_aa: np.ndarray,
    betas: np.ndarray,
    is_right: bool,
    device: torch.device
) -> np.ndarray:
    """
    Compute 3D joints using MANO model with given parameters.
    
    Args:
        mano: MANO model
        global_orient_aa: Global orientation in axis-angle (3,)
        hand_pose_aa: Hand pose in axis-angle (45,)
        betas: Shape parameters (10,)
        is_right: Whether this is a right hand
        device: Torch device
    
    Returns:
        joints: 3D joint positions (21, 3)
    """
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
    
    joints = mano_output.joints[0].cpu().numpy()  # (21, 3)
    
    # Flip x-axis for left hand (same convention as WiLoR output)
    if not is_right:
        joints[:, 0] = -joints[:, 0]
    
    return joints


# ============ Hand Prediction ============

def get_default_hand_result() -> Dict[str, np.ndarray]:
    """Return default result when no hand is detected (axis-angle format)."""
    return {
        'mano_global_orient': np.zeros((1, 3)),      # Axis-angle: 3D vector
        'mano_hand_pose': np.zeros((1, 45)),          # 15 joints × 3D axis-angle = 45
        'mano_betas': np.zeros((1, 10)),              # Shape parameters
        'pred_cam_t_full': np.zeros((1, 3)),
        #'pred_cam_t': np.zeros((1, 3)),
        'focal_length': np.zeros((1, 2)),
        'pred_keypoints_3d': np.zeros((1, 63)),       # 21 joints × 3D = 63 (flattened)
        
        'is_right': np.array([-1.0])  # -1 indicates no detection
        # Temporarily disabled for storage efficiency:
        # 'pred_keypoints_2d': np.zeros((1, 21, 2)),
        # 'pred_vertices': np.zeros((1, 778, 3)),
    }


def predict_hand_single_view(detector, model, device, img_rgb: np.ndarray, model_cfg, args,
                              renderer: Optional[Renderer] = None) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """
    Predict hand pose for a single image and optionally render overlay.
    
    Args:
        detector: YOLO hand detector
        model: WiLoR model
        device: torch device
        img_rgb: RGB image [H, W, 3] in uint8 format
        model_cfg: Model configuration
        args: Additional arguments (rescale_factor, etc.)
        renderer: Optional Renderer for visualization
    
    Returns:
        Tuple of (hand_data dict, rendered_overlay or None)
    """
    img_bgr = img_rgb[:, :, ::-1]
    detections = detector(img_bgr, conf=0.3, verbose=False)[0]

    bboxes, is_right_list = [], []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        if bbox.ndim == 1 and len(bbox) >= 4:
            bboxes.append(bbox[:4].tolist())
            is_right_list.append(det.boxes.cls.cpu().detach().squeeze().item())

    # No hand detected - return default zeros for both hands
    if len(bboxes) == 0:
        default_hand = get_default_hand_result()
        return {'left': default_hand, 'right': default_hand}, img_rgb.copy() if renderer else None

    # Create dataset for ViTDet
    dataset = ViTDetDataset(
        model_cfg, img_rgb, 
        np.stack(bboxes), np.stack(is_right_list), 
        rescale_factor=args.rescale_factor
    )
    batch = next(iter(DataLoader(dataset, batch_size=len(bboxes))))
    batch = recursive_to(batch, device)

    with torch.no_grad():
        out = model(batch)

    # Process predictions for all detected hands
    multiplier = (2 * batch['right'] - 1)
    pred_cam = out['pred_cam']
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    img_size = batch["img_size"].float()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(
        pred_cam, batch["box_center"].float(), 
        batch["box_size"].float(), img_size, 
        scaled_focal_length
    ).detach().cpu().numpy()

    # Collect all hands for rendering and separate left/right hands
    all_verts = []
    all_cam_t = []
    all_right = []
    
    # Separate left and right hands
    left_hand_idx = None
    right_hand_idx = None
    
    batch_size = batch['img'].shape[0]
    for n in range(batch_size):
        verts = out['pred_vertices'][n].detach().cpu().numpy()
        is_r = batch['right'][n].cpu().numpy()
        verts[:, 0] = (2 * is_r - 1) * verts[:, 0]
        
        all_verts.append(verts)
        all_cam_t.append(pred_cam_t_full[n])
        all_right.append(is_r)
        
        # Track left and right hand indices
        if is_r == 1 and right_hand_idx is None:
            right_hand_idx = n
        elif is_r == 0 and left_hand_idx is None:
            left_hand_idx = n
    
    # Extract pred_mano_params (primary output)
    pred_mano_params = out.get('pred_mano_params', {})
    from scipy.spatial.transform import Rotation as R
    
    # Helper function to extract hand data
    def extract_hand_data(hand_idx):
        if hand_idx is None:
            return get_default_hand_result()
        
        hand_result = {}
        
        # Add MANO parameters (convert rotation matrices to axis-angle)
        if pred_mano_params:
            # global_orient: (1, 3, 3) rotation matrix -> axis-angle (3,)
            if 'global_orient' in pred_mano_params:
                global_orient_rotmat = pred_mano_params['global_orient'][hand_idx].detach().cpu().numpy()  # (1, 3, 3)
                # Remove batch dimension: (1, 3, 3) -> (3, 3)
                global_orient_rotmat = global_orient_rotmat.squeeze(0)  # (3, 3)
                global_orient_aa = R.from_matrix(global_orient_rotmat).as_rotvec()  # (3,)
                hand_result['mano_global_orient'] = global_orient_aa[None]  # (1, 3)
            
            # hand_pose: (15, 3, 3) rotation matrices -> axis-angle (15, 3) = (45,)
            if 'hand_pose' in pred_mano_params:
                hand_pose_rotmat = pred_mano_params['hand_pose'][hand_idx].detach().cpu().numpy()  # (15, 3, 3)
                hand_pose_aa = R.from_matrix(hand_pose_rotmat).as_rotvec()  # (15, 3)
                hand_result['mano_hand_pose'] = hand_pose_aa.flatten()[None]  # (1, 45)
            
            # betas: (10,) shape parameters
            if 'betas' in pred_mano_params:
                betas = pred_mano_params['betas'][hand_idx].detach().cpu().numpy()
                hand_result['mano_betas'] = betas.flatten()[None]  # (1, 10)
        
        # Add camera translation
        hand_result['pred_cam_t_full'] = pred_cam_t_full[hand_idx][None]
        
        # Add camera outputs
        if 'focal_length' in out:
            hand_result['focal_length'] = out['focal_length'][hand_idx].detach().cpu().numpy()[None]
        else:
            hand_result['focal_length'] = model_cfg.EXTRA.FOCAL_LENGTH * np.ones((1, 2))
        
        # Add 3D keypoints (joints)
        if 'pred_keypoints_3d' in out:
            keypoints_3d = out['pred_keypoints_3d'][hand_idx].detach().cpu().numpy()  # (21, 3)
            # Flip x-axis for left hand (same as vertices)
            is_r = batch['right'][hand_idx].cpu().numpy()
            keypoints_3d[:, 0] = (2 * is_r - 1) * keypoints_3d[:, 0]
            hand_result['pred_keypoints_3d'] = keypoints_3d.flatten()[None]  # (1, 63)
        else:
            hand_result['pred_keypoints_3d'] = np.zeros((1, 63))
        
        # Add is_right flag
        is_r = batch['right'][hand_idx].cpu().numpy()
        hand_result['is_right'] = np.array([is_r])
        
        return hand_result
    
    # Extract left and right hand data separately
    left_hand_data = extract_hand_data(left_hand_idx)
    right_hand_data = extract_hand_data(right_hand_idx)
    
    # Combine into result dict with 'left' and 'right' keys
    result = {
        'left': left_hand_data,
        'right': right_hand_data
    }
    

    # Render overlay if renderer is provided
    rendered_frame = None
    if renderer is not None and len(all_verts) > 0:
        def render_both_hands(
            renderer,
            all_verts,
            all_cam_t,
            all_right,
            img_rgb,
            scaled_focal_length,
        ):
            """
            Render left & right hands together with different colors.
            Returns an RGB uint8 image.
            """
            if len(all_verts) == 0:
                return None

            H, W = img_rgb.shape[:2]

            hand_colors = []
            for r in all_right:
                if r == 1:   # right hand
                    hand_colors.append((0.2, 0.8, 0.2))   
                else:        # left hand
                    hand_colors.append((0.25, 0.27, 0.66))  

            # render_res expects [width, height] format
            cam_view = renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=[W, H],  # [width, height] format
                is_right=all_right,
                mesh_base_color=hand_colors,
                scene_bg_color=(1.0, 1.0, 1.0),
                focal_length=scaled_focal_length,
            )  # Returns (H, W, 4)

            # Ensure cam_view has the correct shape (H, W, 4)
            if cam_view.shape[:2] != (H, W):
                # If shape is wrong, it might be (W, H, 4), so transpose
                if cam_view.shape[:2] == (W, H):
                    cam_view = cam_view.transpose(1, 0, 2)  # (W, H, 4) -> (H, W, 4)

            img = img_rgb.astype(np.float32) / 255.0          # (H, W, 3)
            mesh_rgb = cam_view[:, :, :3]                    # (H, W, 3)
            alpha = cam_view[:, :, 3:4]                      # (H, W, 1)

            out = img * (1.0 - alpha) + mesh_rgb * alpha
            return (255 * out).clip(0, 255).astype(np.uint8)

        rendered_frame = render_both_hands(
            renderer,
            all_verts,
            all_cam_t,
            all_right,
            img_rgb,
            scaled_focal_length,
        )


    return result, rendered_frame


def predict_hand(detector, model, device, imgs: List[np.ndarray], model_cfg, args,
                 renderer: Optional[Renderer] = None) -> Tuple[Dict[str, Any], List[Optional[np.ndarray]]]:
    """
    Predict hand pose for a list of images (one per camera view at the same timestep).
    
    Args:
        detector: YOLO hand detector
        model: WiLoR model
        device: torch device
        imgs: List of RGB images [H, W, 3] in uint8 format (one per view)
        model_cfg: Model configuration
        args: Additional arguments (rescale_factor, etc.)
        renderer: Optional Renderer for visualization
    
    Returns:
        Tuple of:
        - dict: Hand prediction results with 'left' and 'right' keys, each containing hand data dict
        - list: Rendered overlay frames (or None for each view if renderer not provided)
    """
    left_results = {}
    right_results = {}
    rendered_frames = []
    
    for view_idx, img_rgb in enumerate(imgs):
        view_result, rendered_frame = predict_hand_single_view(
            detector, model, device, img_rgb, model_cfg, args, renderer
        )
        
        rendered_frames.append(rendered_frame)

        # Separate left and right hand results
        if 'left' in view_result:
            for k, v in view_result['left'].items():
                left_results[k] = left_results.get(k, []) + [v]
        
        if 'right' in view_result:
            for k, v in view_result['right'].items():
                right_results[k] = right_results.get(k, []) + [v]

    # Concatenate results along view axis
    left_hand_data = {k: np.concatenate(v, axis=0) for k, v in left_results.items()} if left_results else {}
    right_hand_data = {k: np.concatenate(v, axis=0) for k, v in right_results.items()} if right_results else {}
    
    return {'left': left_hand_data, 'right': right_hand_data}, rendered_frames


# ============ Feature Building ============

def get_hand_feature_names(feature_key: str, dim: int) -> List[str]:
    """Generate proper feature names for hand estimation data."""
    # MANO parameters (primary outputs)
    if "mano_global_orient" in feature_key:
        # Axis-angle representation: 3D vector
        return ["global_orient_x", "global_orient_y", "global_orient_z"]
    elif "mano_hand_pose" in feature_key:
        # 15 joints × 3D axis-angle = 45 values
        names = []
        for joint_idx in range(15):
            names.extend([f"joint{joint_idx+1}_aa_x", f"joint{joint_idx+1}_aa_y", f"joint{joint_idx+1}_aa_z"])
        return names[:dim]
    elif "mano_betas" in feature_key:
        return [f"beta_{i}" for i in range(dim)]
    # Other outputs
    elif "pred_vertices" in feature_key:
        return [f"vertex_{i}" for i in range(dim)]
    elif "pred_keypoints_3d" in feature_key:
        # OpenPose hand format: 21 joints × 3D = 63 values
        # Joint mapping:
        #   0: wrist
        #   1-4: thumb (CMC, MCP, IP, TIP)
        #   5-8: index (MCP, PIP, DIP, TIP)
        #   9-12: middle (MCP, PIP, DIP, TIP)
        #   13-16: ring (MCP, PIP, DIP, TIP)
        #   17-20: pinky (MCP, PIP, DIP, TIP)
        joint_names = ['wrist']
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            for part in ['CMC', 'MCP', 'IP', 'TIP'] if finger == 'thumb' else ['MCP', 'PIP', 'DIP', 'TIP']:
                joint_names.append(f"{finger}_{part}")
        names = []
        for joint_name in joint_names:
            names.extend([f"{joint_name}_x", f"{joint_name}_y", f"{joint_name}_z"])
        return names[:dim]
    elif "pred_cam_t_full" in feature_key:
        return ["tx", "ty", "tz"]
    elif "is_right" in feature_key:
        return ["is_right"]
    else:
        return [f"dim_{i}" for i in range(dim)]


def build_lerobot_features(sample_hand_data: Dict[str, Any], 
                           num_cameras: int,
                           camera_names: List[str]) -> Dict:
    """
    Build LeRobot features based on sample hand estimation data.
    
    Supports dynamic feature addition for future model extensions.
    Now supports both left and right hands separately.
    """
    features = {}

    # Define features to exclude from the final LeRobot dataset
    # These are kept in the `hand_data` dict for rendering but not stored
    EXCLUDE_FEATURES_FROM_STORAGE = ['pred_vertices', 'pred_cam_t', 'pred_keypoints_2d']

    # Handle both old format (single hand) and new format (left/right hands)
    if 'left' in sample_hand_data or 'right' in sample_hand_data:
        # New format: separate left and right hands
        for hand_type in ['left', 'right']:
            if hand_type not in sample_hand_data:
                continue
            
            hand_data = sample_hand_data[hand_type]
            for cam_idx, cam_name in enumerate(camera_names):
                for key, value in hand_data.items():
                    if key in EXCLUDE_FEATURES_FROM_STORAGE:
                        continue  # Skip excluded features
                    if value is None or len(value) == 0:
                        continue
                    
                    # Determine shape (excluding batch dimension)
                    if value.ndim > 1:
                        shape = value.shape[1:]
                    else:
                        shape = (1,)
                    
                    # Flatten shape for LeRobot
                    flat_dim = int(np.prod(shape))
                    feature_key = f"observation.hand.{hand_type}.{cam_name}.{key}"
                    
                    features[feature_key] = {
                        "dtype": "float32",
                        "shape": (flat_dim,),
                        "names": get_hand_feature_names(key, flat_dim)
                    }
    else:
        # Old format: single hand (backward compatibility)
        for cam_idx, cam_name in enumerate(camera_names):
            for key, value in sample_hand_data.items():
                if key in EXCLUDE_FEATURES_FROM_STORAGE:
                    continue  # Skip excluded features
                if value is None or len(value) == 0:
                    continue
                    
                # Determine shape (excluding batch dimension)
                if value.ndim > 1:
                    shape = value.shape[1:]
                else:
                    shape = (1,)
                
                # Flatten shape for LeRobot
                flat_dim = int(np.prod(shape))
                feature_key = f"observation.hand.{cam_name}.{key}"
                
                features[feature_key] = {
                    "dtype": "float32",
                    "shape": (flat_dim,),
                    "names": get_hand_feature_names(key, flat_dim)
                }

    return features


# ============ Dataset Completeness Check ============

def check_dataset_complete(dataset_dir: Path) -> Tuple[bool, str]:
    """Check if a dataset has been fully processed."""
    if not dataset_dir.exists():
        return False, "Directory does not exist"

    complete_marker = dataset_dir / ".complete"
    if complete_marker.exists():
        return True, "Complete marker found"

    info_file = dataset_dir / "meta" / "info.json"
    if not info_file.exists():
        return False, "Missing meta/info.json"

    try:
        with open(info_file, 'r') as f:
            info = json.load(f)
            if info.get('total_episodes', 0) > 0:
                return True, f"Has {info['total_episodes']} episodes"
    except:
        pass

    return False, "Incomplete dataset"


# ============ Video Visualization Helper ============

def concatenate_camera_views(frames: List[np.ndarray], layout: str = "horizontal") -> np.ndarray:
    """
    Concatenate multiple camera view frames into a single frame.
    
    Args:
        frames: List of RGB frames [H, W, 3]
        layout: "horizontal" or "vertical" or "grid"
    
    Returns:
        Concatenated frame
    """
    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Filter out None frames
    valid_frames = [f for f in frames if f is not None]
    if not valid_frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Resize all frames to same size (use first frame as reference)
    target_h, target_w = valid_frames[0].shape[:2]
    resized_frames = []
    for frame in valid_frames:
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h))
        resized_frames.append(frame)
    
    if layout == "horizontal":
        return np.concatenate(resized_frames, axis=1)
    elif layout == "vertical":
        return np.concatenate(resized_frames, axis=0)
    else:  # grid
        n = len(resized_frames)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        
        # Pad with black frames if needed
        while len(resized_frames) < rows * cols:
            resized_frames.append(np.zeros_like(resized_frames[0]))
        
        grid_rows = []
        for r in range(rows):
            row_frames = resized_frames[r * cols:(r + 1) * cols]
            grid_rows.append(np.concatenate(row_frames, axis=1))
        
        return np.concatenate(grid_rows, axis=0)


# ============ Main Processing ============

def process_dataset(args):
    """Main processing function."""
    print("=" * 80)
    print("🖐️  WiLoR Hand Estimation for LeRobot Datasets")
    print("=" * 80)

    # Load WiLoR model and detector
    print("📦 Loading WiLoR model...")
    model, model_cfg = load_wilor(
        checkpoint_path=args.checkpoint,
        cfg_path=args.model_cfg
    )
    detector = YOLO(args.detector)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    detector = detector.to(device)
    model.eval()
    print(f"  ✓ Model loaded on {device}")
    
    # Setup renderer and video output directory if saving videos
    renderer = None
    video_output_dir = None
    if args.save_video:
        print("🎬 Video saving enabled, initializing renderer...")
        renderer = Renderer(model_cfg, faces=model.mano.faces)
        video_output_dir = Path(args.video_output_dir) if args.video_output_dir else Path(args.output_dir) / "videos"
        video_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Renderer initialized, videos will be saved to: {video_output_dir}")
    
    # Load MANO model if using mean betas for 3D joints
    mano_model = None
    if args.use_mean_betas:
        print("📊 Mean betas mode enabled, loading MANO model...")
        mano_model = load_mano_model(args.model_cfg, device)
        print(f"  ✓ MANO model loaded for 3D joint computation")

    # Load dataset
    print("📂 Loading dataset...")
    config = HumanVideoDataConfig(
        root=args.root,
        dataset_file=args.config_file,
        cameras=args.cameras,
        video_backend=args.video_backend,
        fps=args.fps
    )
    dataset = HumanVideoDataset(config)

    # Check output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group episodes by repo_id for creating separate LeRobot datasets
    repo_episodes = {}
    for idx in range(len(dataset)):
        repo_id, ep_idx = dataset.episode_list[idx]
        if repo_id not in repo_episodes:
            repo_episodes[repo_id] = []
        repo_episodes[repo_id].append((idx, ep_idx))

    print(f"\n📊 Processing {len(dataset)} episodes from {len(repo_episodes)} repositories")

    # Process each repository
    for repo_id, episodes in repo_episodes.items():
        print(f"\n{'='*60}")
        print(f"📁 Processing repository: {repo_id}")
        print(f"   Episodes: {len(episodes)}")
        
        repo_output_dir = output_dir / repo_id
        
        # Check if already processed
        if args.continue_incomplete:
            is_complete, reason = check_dataset_complete(repo_output_dir)
            if is_complete:
                print(f"   ⏭️  Skipping: {reason}")
                continue

        # Process first episode to get feature structure
        first_data = dataset[episodes[0][0]]
        num_cameras = len(first_data['video'])
        
        # Get sample hand prediction for feature building
        sample_imgs = [first_data['video'][v][0] for v in range(num_cameras)]
        sample_hand_data, _ = predict_hand(detector, model, device, sample_imgs, model_cfg, args)
        
        if not sample_hand_data or ('left' not in sample_hand_data and 'right' not in sample_hand_data):
            print(f"   ⚠️  No hands detected in first frame, using default features")
            # Create default features with MANO params (axis-angle format)
            default_hand = get_default_hand_result()
            sample_hand_data = {'left': default_hand, 'right': default_hand}

        # Build features
        features = build_lerobot_features(sample_hand_data, num_cameras, args.cameras[:num_cameras])
        
        print(f"   📝 Features: {list(features.keys())}")

        # Create LeRobot dataset
        try:
            lerobot_dataset = LeRobotDataset.create(
                repo_id=repo_id + "_hand",
                fps=args.fps,
                root=str(repo_output_dir),
                features=features,
                use_videos=False,  # We're storing numerical data, not videos
                video_backend='torchcodec'
            )
        except Exception as e:
            print(f"   ❌ Failed to create dataset: {e}")
            continue
        
        # Process all episodes
        for dataset_idx, ep_idx in tqdm(episodes, desc=f"Processing {repo_id}"):
            try:
                data = dataset[dataset_idx]
                video_views = data['video']  # List of [T, H, W, 3]
                num_frames = len(data['frame_indices'])
                
                # Setup video writer for this episode if saving videos
                episode_video_writer = None
                video_path = None
                if args.save_video and renderer is not None and video_output_dir is not None:
                    # Determine frame size from first frame
                    first_frames = [video_views[v][0] for v in range(len(video_views))]
                    sample_concat = concatenate_camera_views(first_frames, args.video_layout)
                    frame_h, frame_w = sample_concat.shape[:2]
                    
                    video_path = video_output_dir / repo_id / f"episode_{ep_idx:04d}.mp4"
                    episode_video_writer = VideoWriter(
                        str(video_path), 
                        fps=args.fps, 
                        frame_size=(frame_w, frame_h)
                    )
                
                # ============ Two-pass processing if using mean betas ============
                if args.use_mean_betas and mano_model is not None:
                    # First pass: collect all hand data and betas
                    all_hand_data = []
                    all_rendered_frames = []
                    # betas_collection[cam_idx][hand_type] = list of betas arrays
                    betas_collection = {cam_idx: {'left': [], 'right': []} 
                                        for cam_idx in range(len(video_views))}
                    
                    for t in tqdm(range(num_frames), desc=f" Episode {ep_idx} (pass 1)", leave=False):
                        imgs = [video_views[v][t] for v in range(len(video_views))]
                        hand_data, rendered_frames = predict_hand(
                            detector, model, device, imgs, model_cfg, args, renderer
                        )
                        all_hand_data.append(hand_data)
                        all_rendered_frames.append(rendered_frames)
                        
                        # Collect betas for each camera and hand type
                        for hand_type in ['left', 'right']:
                            if hand_type not in hand_data:
                                continue
                            hand_type_data = hand_data[hand_type]
                            if 'mano_betas' in hand_type_data and hand_type_data['mano_betas'] is not None:
                                for cam_idx in range(len(video_views)):
                                    if cam_idx < len(hand_type_data['mano_betas']):
                                        betas = hand_type_data['mano_betas'][cam_idx].flatten()
                                        # Only collect non-zero betas (valid detections)
                                        if np.abs(betas).sum() > 1e-6:
                                            betas_collection[cam_idx][hand_type].append(betas)
                    
                    # Compute mean betas for each camera and hand type
                    mean_betas = {cam_idx: {'left': None, 'right': None} 
                                  for cam_idx in range(len(video_views))}
                    for cam_idx in range(len(video_views)):
                        for hand_type in ['left', 'right']:
                            if len(betas_collection[cam_idx][hand_type]) > 0:
                                mean_betas[cam_idx][hand_type] = np.mean(
                                    np.array(betas_collection[cam_idx][hand_type]), axis=0
                                )
                                if args.verbose:
                                    cam_name = args.cameras[cam_idx] if cam_idx < len(args.cameras) else f"cam{cam_idx}"
                                    print(f"      📊 {cam_name}/{hand_type}: mean betas from {len(betas_collection[cam_idx][hand_type])} frames")
                    
                    # Second pass: recompute 3D joints with mean betas
                    for t in tqdm(range(num_frames), desc=f" Episode {ep_idx} (pass 2)", leave=False):
                        hand_data = all_hand_data[t]
                        rendered_frames = all_rendered_frames[t]
                        
                        # Recompute 3D joints using mean betas
                        for hand_type in ['left', 'right']:
                            if hand_type not in hand_data:
                                continue
                            hand_type_data = hand_data[hand_type]
                            
                            # Check if we have valid MANO params
                            if ('mano_global_orient' not in hand_type_data or 
                                'mano_hand_pose' not in hand_type_data):
                                continue
                            
                            # Recompute joints for each camera
                            new_joints_list = []
                            for cam_idx in range(len(video_views)):
                                if cam_idx >= len(hand_type_data.get('mano_global_orient', [])):
                                    new_joints_list.append(np.zeros((1, 63)))
                                    continue
                                
                                global_orient = hand_type_data['mano_global_orient'][cam_idx].flatten()
                                hand_pose = hand_type_data['mano_hand_pose'][cam_idx].flatten()
                                is_right_arr = hand_type_data.get('is_right', [np.array([-1])])
                                is_right_val = is_right_arr[cam_idx].flatten()[0] if cam_idx < len(is_right_arr) else -1
                                
                                # Skip if no valid detection
                                if is_right_val < 0 or np.abs(global_orient).sum() < 1e-6:
                                    new_joints_list.append(np.zeros((1, 63)))
                                    continue
                                
                                # Use mean betas if available, otherwise use original
                                betas = mean_betas[cam_idx][hand_type]
                                if betas is None:
                                    # Fallback to original betas
                                    betas = hand_type_data['mano_betas'][cam_idx].flatten()
                                
                                is_right = bool(is_right_val > 0.5)
                                
                                try:
                                    joints = compute_joints_with_mano(
                                        mano_model, global_orient, hand_pose, betas, is_right, device
                                    )
                                    new_joints_list.append(joints.flatten()[None])  # (1, 63)
                                except Exception as e:
                                    if args.verbose:
                                        print(f"        ⚠️  Failed to compute joints: {e}")
                                    new_joints_list.append(np.zeros((1, 63)))
                            
                            # Update pred_keypoints_3d with new joints
                            hand_type_data['pred_keypoints_3d'] = np.concatenate(new_joints_list, axis=0)
                        
                        # Save rendered frame to video
                        if episode_video_writer is not None and rendered_frames:
                            imgs = [video_views[v][t] for v in range(len(video_views))]
                            frames_to_concat = []
                            for v_idx, rendered in enumerate(rendered_frames):
                                if rendered is not None:
                                    frames_to_concat.append(rendered)
                                else:
                                    frames_to_concat.append(imgs[v_idx])
                            
                            concat_frame = concatenate_camera_views(frames_to_concat, args.video_layout)
                            episode_video_writer.add_frame(concat_frame)
                        
                        # Build frame data for LeRobot dataset
                        frame = {"task": repo_id}
                        
                        # Handle both left and right hands
                        for hand_type in ['left', 'right']:
                            if hand_type not in hand_data:
                                continue
                            
                            hand_type_data = hand_data[hand_type]
                            for cam_idx, cam_name in enumerate(args.cameras[:len(video_views)]):
                                for key, value in hand_type_data.items():
                                    if value is None:
                                        continue
                                    # Handle per-view data (if applicable)
                                    if len(value) > cam_idx:
                                        frame_value = value[cam_idx].flatten().astype(np.float32)
                                    else:
                                        frame_value = value[0].flatten().astype(np.float32)
                                    feature_key = f"observation.hand.{hand_type}.{cam_name}.{key}"
                                    if feature_key in features:
                                        frame[feature_key] = frame_value

                        # Add frame to dataset
                        lerobot_dataset.add_frame(frame)
                
                else:
                    # ============ Single-pass processing (original behavior) ============
                    for t in tqdm(range(num_frames), desc=f" Episode {ep_idx}", leave=False):
                        imgs = [video_views[v][t] for v in range(len(video_views))]
                        hand_data, rendered_frames = predict_hand(
                            detector, model, device, imgs, model_cfg, args, renderer
                        )
                        # Save rendered frame to video
                        if episode_video_writer is not None and rendered_frames:
                            # Use rendered frames if available, otherwise use original
                            frames_to_concat = []
                            for v_idx, rendered in enumerate(rendered_frames):
                                if rendered is not None:
                                    frames_to_concat.append(rendered)
                                else:
                                    frames_to_concat.append(imgs[v_idx])
                            
                            concat_frame = concatenate_camera_views(frames_to_concat, args.video_layout)
                            episode_video_writer.add_frame(concat_frame)
                        # Build frame data for LeRobot dataset
                        frame = {"task": repo_id}
                        
                        # Handle both left and right hands
                        for hand_type in ['left', 'right']:
                            if hand_type not in hand_data:
                                continue
                            
                            hand_type_data = hand_data[hand_type]
                            for cam_idx, cam_name in enumerate(args.cameras[:len(video_views)]):
                                for key, value in hand_type_data.items():
                                    if value is None:
                                        continue
                                    # Handle per-view data (if applicable)
                                    if len(value) > cam_idx:
                                        frame_value = value[cam_idx].flatten().astype(np.float32)
                                    else:
                                        frame_value = value[0].flatten().astype(np.float32)
                                    feature_key = f"observation.hand.{hand_type}.{cam_name}.{key}"
                                    if feature_key in features:
                                        frame[feature_key] = frame_value

                        # Add frame to dataset
                        lerobot_dataset.add_frame(frame)
                
                # Save episode data
                lerobot_dataset.save_episode()
                
                # Save episode video
                if episode_video_writer is not None:
                    episode_video_writer.save()
                    if args.verbose:
                        print(f"      🎬 Saved video: {video_path}")

            except Exception as e:
                print(f"   ❌ Error processing episode {ep_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Finalize dataset
        try:
            total_episodes = len(episodes)
            lerobot_dataset.meta.splits = {
                "train": list(range(total_episodes)),
                repo_id: list(range(total_episodes))
            }
            lerobot_dataset.finalize()

            # Add joint mapping info to info.json
            info_file = repo_output_dir / "meta" / "info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info_data = json.load(f)
                
                # Add joint mapping documentation
                info_data["joint_mapping"] = {
                    "description": "OpenPose hand format: 21 joints × 3D coordinates",
                    "total_joints": 21,
                    "mapping": {
                        "0": "wrist",
                        "1-4": "thumb (CMC, MCP, IP, TIP)",
                        "5-8": "index (MCP, PIP, DIP, TIP)",
                        "9-12": "middle (MCP, PIP, DIP, TIP)",
                        "13-16": "ring (MCP, PIP, DIP, TIP)",
                        "17-20": "pinky (MCP, PIP, DIP, TIP)"
                    },
                    "tip_indices": {
                        "thumb_tip": 4,
                        "index_tip": 8,
                        "middle_tip": 12,
                        "ring_tip": 16,
                        "pinky_tip": 20
                    }
                }
                
                with open(info_file, 'w') as f:
                    json.dump(info_data, f, indent=2)
                print(f"   ✓ Added joint mapping info to info.json")

            # Create completion marker
            complete_marker = repo_output_dir / ".complete"
            completion_info = {
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "episodes": total_episodes,
                "data_type": "hand_estimation",
                "model": "wilor",
                "fps": args.fps
            }
            with open(complete_marker, 'w') as f:
                f.write(json.dumps(completion_info, indent=2))

            print(f"   ✅ Completed: {total_episodes} episodes")

        except Exception as e:
            print(f"   ❌ Failed to finalize dataset: {e}")

    print("\n" + "=" * 80)
    print("🎉 Processing complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="WiLoR Hand Estimation Pipeline for LeRobot Datasets"
    )
    
    # Dataset configuration
    parser.add_argument('--config_file', type=str, required=True,
                        help='JSON config file listing datasets to process')
    parser.add_argument('--root', type=str, default='demos',
                        help='Root directory containing human demo repos')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed datasets')
    
    # Model configuration
    parser.add_argument('--checkpoint', type=str, 
                        default='./pretrained_models/wilor_final.ckpt',
                        help='Path to WiLoR checkpoint')
    parser.add_argument('--model_cfg', type=str,
                        default='./pretrained_models/model_config.yaml',
                        help='Path to WiLoR model config')
    parser.add_argument('--detector', type=str,
                        default='./pretrained_models/detector.pt',
                        help='Path to YOLO hand detector')
    
    # Processing configuration
    parser.add_argument('--cameras', type=str, nargs='+', 
                        default=['cam1', 'cam2', 'cam3', 'zed2i'],
                        help='Camera names to process')
    parser.add_argument('--rescale_factor', type=float, default=2.0,
                        help='Factor for padding the bbox')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second of the dataset')
    parser.add_argument('--video_backend', type=str, default='torchcodec',
                        choices=['torchcodec', 'decord', 'opencv'],
                        help='Video decoding backend')
    
    # Video saving options
    parser.add_argument('--save_video', action='store_true',
                        help='Save visualization videos with hand overlay')
    parser.add_argument('--video_output_dir', type=str, default=None,
                        help='Output directory for videos (default: output_dir/videos)')
    parser.add_argument('--video_layout', type=str, default='grid',
                        choices=['horizontal', 'vertical', 'grid'],
                        help='Layout for multi-camera video concatenation')
    
    # Processing options
    parser.add_argument('--continue', dest='continue_incomplete', 
                        action='store_true',
                        help='Skip already processed datasets')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing of all datasets')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    parser.add_argument('--use_mean_betas', action='store_true',
                        help='Use per-episode mean betas to compute 3D joints via MANO model')

    args = parser.parse_args()

    process_dataset(args)


if __name__ == '__main__':
    main()
