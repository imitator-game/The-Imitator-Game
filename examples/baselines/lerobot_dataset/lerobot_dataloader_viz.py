"""
LeRobot Dataset Visualizer
===========================
Comprehensive visualization tool for LeRobot datasets (Robot/Sim/Human).

Features:
- Visualize multiple episodes and steps
- Random or manual episode selection
- Interactive browsing with keyboard controls
- DataLoader batch testing
- Normalization inspection
- State/Action trajectory plots
- Support for RGB + Depth cameras

Usage:
    # Robot dataset - random episodes
    python lerobot_dataloader_viz.py --dataset-type robot --num-episodes 5 --steps-per-episode 10

    # Simulation dataset - interactive mode
    python lerobot_dataloader_viz.py --dataset-type sim --mode interactive

    # Human video dataset - specific episodes
    python lerobot_dataloader_viz.py --dataset-type human --mode manual --episode-ids 0,5,10

    # Test DataLoader batching
    python lerobot_dataloader_viz.py --dataset-type robot --test-dataloader --batch-size 16
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import cv2
from tqdm import tqdm

# Import dataset classes
# Adjust these imports based on your project structure
try:
    from examples.baselines.lerobot_dataset.lerobot_robot_dataset import (
        LeRobotRobotDataset, LeRobotRobotDataConfig
    )
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False
    print("⚠️  Robot dataset not available")

try:
    from examples.baselines.lerobot_dataset.lerobot_sim_dataset import (
        LeRobotSimDataset, LeRobotSimDataConfig
    )
    SIM_AVAILABLE = True
except ImportError:
    SIM_AVAILABLE = False
    print("⚠️  Sim dataset not available")

try:
    from examples.baselines.lerobot_dataset.lerobot_human_dataset import (
        HumanVideoDataset, HumanVideoDataConfig
    )
    HUMAN_AVAILABLE = True
except ImportError:
    HUMAN_AVAILABLE = False
    print("⚠️  Human dataset not available")


class DatasetVisualizer:
    """Comprehensive dataset visualizer with multiple viewing modes."""

    def __init__(self, args):
        self.args = args
        self.dataset = None
        self.dataloader = None
        self.current_idx = 0
        self.episode_samples = []

        # Color maps
        self.cmap_depth = plt.cm.viridis

    def load_dataset(self):
        """Load the appropriate dataset based on args."""
        print(f"\n{'='*80}")
        print(f"Loading {self.args.dataset_type} dataset...")
        print(f"{'='*80}\n")

        if self.args.dataset_type == "robot":
            if not ROBOT_AVAILABLE:
                raise ImportError("Robot dataset not available")
            config = self._create_robot_config()
            self.dataset = LeRobotRobotDataset(config)

        elif self.args.dataset_type == "sim":
            if not SIM_AVAILABLE:
                raise ImportError("Sim dataset not available")
            config = self._create_sim_config()
            self.dataset = LeRobotSimDataset(config)

        elif self.args.dataset_type == "human":
            if not HUMAN_AVAILABLE:
                raise ImportError("Human dataset not available")
            config = self._create_human_config()
            self.dataset = HumanVideoDataset(config)
        else:
            raise ValueError(f"Unknown dataset type: {self.args.dataset_type}")

        print(f"✓ Dataset loaded: {len(self.dataset)} samples\n")
        return self.dataset

    def _create_robot_config(self):
        """Create config for robot dataset."""
        # Auto-detect cameras if not specified
        cameras = self.args.cameras
        if not cameras or cameras == ['cam1', 'cam2']:
            # Try to detect from dataset
            print("🔍 Auto-detecting cameras from dataset...")
            cameras = self._detect_cameras_from_dataset()

        config = LeRobotRobotDataConfig(
            root=self.args.root,
            split=self.args.split,
            dataset_file=self.args.dataset_file,
            task_description_file=self.args.task_description_file,
            image_size=tuple(self.args.image_size),
            state_type=self.args.state_type,
            include_depth=self.args.include_depth,
            cameras=cameras,
            horizon=self.args.horizon,
            single_arm=self.args.single_arm,
            fps=self.args.fps,
            video_backend=self.args.video_backend,
            normalization_method=self.args.normalization_method,
        )
        return config

    def _detect_cameras_from_dataset(self):
        """Detect available cameras from dataset files."""
        try:
            # Try to load first dataset to detect cameras
            if self.args.dataset_file:
                import json
                with open(self.args.dataset_file, 'r') as f:
                    dataset_configs = json.load(f)
                if dataset_configs:
                    first_config = dataset_configs[0]
                    ds_root = os.path.join(self.args.root, first_config.get("root"))

                    # Check for meta/info.json
                    info_file = Path(ds_root) / "meta" / "info.json"
                    if info_file.exists():
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                            # Extract camera names from features
                            cameras = []
                            for key in info.get('features', {}).keys():
                                if 'observation.images.' in key and '_depth' not in key:
                                    cam_name = key.replace('observation.images.', '')
                                    cameras.append(cam_name)
                            if cameras:
                                print(f"✓ Detected cameras: {cameras}")
                                return cameras
        except Exception as e:
            print(f"⚠️  Camera detection failed: {e}")

        # Fallback to common camera names
        common_cameras = ['cam1', 'cam2', 'cam3', 'zed2i', 'wristcam0', 'wristcam1']
        print(f"⚠️  Using default cameras: {common_cameras}")
        return common_cameras

    def _create_sim_config(self):
        """Create config for sim dataset."""
        config = LeRobotSimDataConfig(
            root=self.args.root,
            split=self.args.split,
            dataset_file=self.args.dataset_file,
            task_description_file=self.args.task_description_file,
            repo_id=self.args.repo_id,
            image_size=tuple(self.args.image_size),
            state_type=self.args.state_type,
            include_depth=self.args.include_depth,
            cameras=self.args.cameras,
            horizon=self.args.horizon,
            single_arm=self.args.single_arm,
            fps=self.args.fps,
            video_backend=self.args.video_backend,
            depth_mode=self.args.depth_mode,
            normalization_method=self.args.normalization_method,
        )
        return config

    def _create_human_config(self):
        """Create config for human video dataset."""
        config = HumanVideoDataConfig(
            root=self.args.root,
            split=self.args.split,
            dataset_file=self.args.dataset_file,
            task_description_file=self.args.task_description_file,
            cameras=self.args.cameras,
            include_depth=self.args.include_depth,
            num_frames=self.args.num_frames,
            image_size=tuple(self.args.image_size),
            sampling_strategy=self.args.sampling_strategy,
            video_backend=self.args.video_backend,
        )
        return config

    def sample_episodes(self):
        """Sample episodes to visualize."""
        if self.args.mode == "random":
            # Random sampling
            indices = random.sample(range(len(self.dataset)),
                                   min(self.args.num_episodes, len(self.dataset)))
            print(f"📊 Randomly sampled {len(indices)} episodes: {indices}\n")

        elif self.args.mode == "manual":
            # Manual episode selection
            if self.args.episode_ids:
                indices = [int(x) for x in self.args.episode_ids.split(',')]
                indices = [i for i in indices if i < len(self.dataset)]
                print(f"📊 Manually selected {len(indices)} episodes: {indices}\n")
            else:
                # Default to first N episodes
                indices = list(range(min(self.args.num_episodes, len(self.dataset))))
                print(f"📊 Using first {len(indices)} episodes\n")

        elif self.args.mode == "sequential":
            # Sequential from start
            indices = list(range(min(self.args.num_episodes, len(self.dataset))))
            print(f"📊 Sequential episodes: {indices}\n")

        else:  # interactive
            indices = [0]  # Start with first episode
            print(f"📊 Interactive mode - use arrow keys to navigate\n")

        self.episode_samples = indices
        return indices

    def visualize_sample(self, sample: Dict[str, Any], idx: int, save_path: Optional[str] = None):
        """Visualize a single sample with all information."""

        # Determine if this is human video data
        is_video = 'video' in sample

        if is_video:
            self._visualize_video_sample(sample, idx, save_path)
        else:
            self._visualize_robot_sample(sample, idx, save_path)

    def _visualize_robot_sample(self, sample: Dict[str, Any], idx: int, save_path: Optional[str] = None):
        """Visualize robot/sim sample with proper feature name detection."""
        # Extract data
        states = sample['states']  # [state_dim]
        actions = sample['actions']  # [horizon, action_dim]

        # Detect state type and get proper labels
        state_dim = states.shape[0]
        action_dim = actions.shape[1]

        # Determine data format based on dimensions
        if state_dim == 18:  # Sim format: qpos_gripper_states
            state_type = "qpos_gripper (18D)"
            state_labels = self._get_state_labels_18d()
        elif state_dim == 16:  # Robot format: qpos_gripper_states
            state_type = "qpos_gripper (16D)"
            state_labels = self._get_state_labels_16d()
        elif state_dim == 14:  # Robot format: eepos_gripper_states or joint_states
            if 'eepos' in self.args.state_type:
                state_type = "eepos_gripper (14D)"
                state_labels = self._get_state_labels_eepos_14d()
            else:
                state_type = "joint_states (14D)"
                state_labels = self._get_state_labels_joint_14d()
        elif state_dim == 12:  # ee_pose only
            state_type = "ee_pose (12D)"
            state_labels = self._get_state_labels_ee_12d()
        else:
            state_type = f"Unknown ({state_dim}D)"
            state_labels = [f"dim_{i}" for i in range(state_dim)]

        # Action labels
        if action_dim == 16:
            action_labels = self._get_action_labels_16d()
        elif action_dim == 14:
            action_labels = self._get_state_labels_joint_14d()
        elif action_dim == 12:
            action_labels = self._get_state_labels_ee_12d()
        else:
            action_labels = [f"dim_{i}" for i in range(action_dim)]

        # Count cameras
        view_keys = sorted([k for k in sample.keys() if k.startswith('view_')])
        num_cameras = len(view_keys)

        has_depth = self.args.include_depth and sample[view_keys[0]].shape[0] > 3

        # Create figure
        if has_depth:
            fig = plt.figure(figsize=(20, 14))
            gs = fig.add_gridspec(4, max(num_cameras, 2), hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, max(num_cameras, 2), hspace=0.3, wspace=0.3)

        # Title with data format info
        title = f"Sample {idx} | {state_type}"
        if 'task_descriptions' in sample and sample['task_descriptions']:
            title += f" | Task: {sample['task_descriptions']}"
        if 'repo_id' in sample and sample['repo_id']:
            title += f"\nRepo: {sample['repo_id']}"
        if 'episode_index' in sample and 'frame_index' in sample:
            try:
                ep_idx = sample['episode_index'].item() if hasattr(sample['episode_index'], 'item') else sample['episode_index']
                frame_idx = sample['frame_index'].item() if hasattr(sample['frame_index'], 'item') else sample['frame_index']
                title += f" | Episode: {ep_idx}, Frame: {frame_idx}"
            except:
                pass
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Plot RGB cameras
        for cam_idx, view_key in enumerate(view_keys):
            view = sample[view_key]  # [C, H, W]

            # Extract RGB
            if has_depth:
                rgb = view[:3]  # First 3 channels
            else:
                rgb = view

            # Convert to numpy [H, W, C]
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy()

            # Denormalize if needed
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)

            ax = fig.add_subplot(gs[0, cam_idx])
            ax.imshow(rgb_np)
            ax.set_title(f"Camera {cam_idx + 1} - RGB", fontweight='bold')
            ax.axis('off')

        # Plot Depth cameras
        if has_depth:
            for cam_idx, view_key in enumerate(view_keys):
                view = sample[view_key]
                depth = view[3:4]  # 4th channel

                depth_np = depth.squeeze(0).cpu().numpy()

                ax = fig.add_subplot(gs[1, cam_idx])
                im = ax.imshow(depth_np, cmap='viridis')
                ax.set_title(f"Camera {cam_idx + 1} - Depth", fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

        # Plot State with proper labels
        state_row = 2 if has_depth else 1
        ax_state = fig.add_subplot(gs[state_row, :])

        state_np = states.cpu().numpy()

        x = np.arange(state_dim)
        colors = self._get_bar_colors(state_dim)
        ax_state.bar(x, state_np, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax_state.set_xlabel('State Dimension', fontweight='bold')
        ax_state.set_ylabel('Value', fontweight='bold')
        ax_state.set_title(f'Current State: {state_type}', fontweight='bold')
        ax_state.grid(True, alpha=0.3, axis='y')
        ax_state.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Set x-axis labels
        if state_dim <= 20:  # Show all labels if not too many
            ax_state.set_xticks(x)
            ax_state.set_xticklabels(state_labels, rotation=45, ha='right', fontsize=8)
        else:
            # Show every nth label
            step = state_dim // 10
            ax_state.set_xticks(x[::step])
            ax_state.set_xticklabels([state_labels[i] for i in range(0, state_dim, step)],
                                    rotation=45, ha='right', fontsize=8)

        # Add value labels on bars (only if not too many)
        if state_dim <= 20:
            for i, v in enumerate(state_np):
                ax_state.text(i, v, f'{v:.2f}', ha='center',
                            va='bottom' if v >= 0 else 'top', fontsize=7, rotation=0)

        # Plot Actions with proper labels
        action_row = 3 if has_depth else 2
        ax_action = fig.add_subplot(gs[action_row, :])

        actions_np = actions.cpu().numpy()  # [horizon, action_dim]
        horizon, action_dim = actions_np.shape

        # Plot heatmap for actions
        if horizon > 1 and action_dim <= 20:
            im = ax_action.imshow(actions_np.T, aspect='auto', cmap='RdYlBu_r',
                                 interpolation='nearest', vmin=-1, vmax=1)
            ax_action.set_xlabel('Time Step (Horizon)', fontweight='bold')
            ax_action.set_ylabel('Action Dimension', fontweight='bold')
            ax_action.set_title(f'Action Sequence (horizon={horizon}, dim={action_dim})', fontweight='bold')

            # Set y-axis labels
            ax_action.set_yticks(range(action_dim))
            ax_action.set_yticklabels(action_labels, fontsize=8)

            plt.colorbar(im, ax=ax_action, label='Action Value')
        else:
            # Plot line chart if too many dimensions
            for dim in range(min(action_dim, 10)):  # Limit to first 10 dims
                ax_action.plot(range(horizon), actions_np[:, dim],
                             marker='o', label=action_labels[dim], alpha=0.7)

            ax_action.set_xlabel('Time Step (Horizon)', fontweight='bold')
            ax_action.set_ylabel('Action Value', fontweight='bold')
            ax_action.set_title(f'Action Sequence (horizon={horizon}, dim={action_dim}, showing first 10)',
                              fontweight='bold')
            ax_action.grid(True, alpha=0.3)
            ax_action.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax_action.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Add statistics text
        stats_text = f"State ({state_type}): min={state_np.min():.3f}, max={state_np.max():.3f}, mean={state_np.mean():.3f}\n"
        stats_text += f"Action: min={actions_np.min():.3f}, max={actions_np.max():.3f}, mean={actions_np.mean():.3f}"
        fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization to {save_path}")

        # Only show and close if not in interactive mode
        if self.args.mode != "interactive":
            plt.show()
        # Figure will be closed by caller if needed

        return fig

    def _get_bar_colors(self, dim):
        """Get color scheme for state bars to distinguish different joint groups."""
        colors = []
        if dim == 18:  # Sim: left(9) + right(9)
            colors = ['steelblue'] * 9 + ['coral'] * 9
        elif dim == 16:  # Robot: left(7) + left_gripper + right(7) + right_gripper
            colors = ['steelblue'] * 7 + ['navy'] + ['coral'] * 7 + ['darkred']
        elif dim == 14:  # Joint states or eepos_gripper
            colors = ['steelblue'] * 7 + ['coral'] * 7
        else:
            colors = ['steelblue'] * dim
        return colors

    def _get_state_labels_18d(self):
        """Labels for 18D sim qpos_gripper_states."""
        return (
            [f"L_j{i}" for i in range(7)] + ["L_g1", "L_g2"] +
            [f"R_j{i}" for i in range(7)] + ["R_g1", "R_g2"]
        )

    def _get_state_labels_16d(self):
        """Labels for 16D robot qpos_gripper_states."""
        return (
            [f"R_j{i}" for i in range(7)] + [f"L_j{i}" for i in range(7)] +
            ["R_grip", "L_grip"]
        )

    def _get_state_labels_eepos_14d(self):
        """Labels for 14D eepos_gripper_states."""
        return (
            ["R_x", "R_y", "R_z", "R_rx", "R_ry", "R_rz"] +
            ["L_x", "L_y", "L_z", "L_rx", "L_ry", "L_rz"] +
            ["R_grip", "L_grip"]
        )

    def _get_state_labels_joint_14d(self):
        """Labels for 14D joint_states."""
        return [f"R_j{i}" for i in range(7)] + [f"L_j{i}" for i in range(7)]

    def _get_state_labels_ee_12d(self):
        """Labels for 12D ee_pose."""
        return (
            ["R_x", "R_y", "R_z", "R_rx", "R_ry", "R_rz"] +
            ["L_x", "L_y", "L_z", "L_rx", "L_ry", "L_rz"]
        )

    def _get_action_labels_16d(self):
        """Labels for 16D actions."""
        return (
            [f"L_j{i}" for i in range(7)] + ["L_g"] +
            [f"R_j{i}" for i in range(7)] + ["R_g"]
        )

    def _visualize_video_sample(self, sample: Dict[str, Any], idx: int, save_path: Optional[str] = None):
        """Visualize human video sample."""
        video = sample['video']  # [T, H, W, C]
        T, H, W, C = video.shape

        task = sample.get('task', 'Unknown')
        episode_idx = sample.get('episode_idx', -1)

        # Create figure - show frames in grid
        num_frames_to_show = min(T, 16)  # Show up to 16 frames
        cols = 4
        rows = (num_frames_to_show + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

        title = f"Human Video Sample {idx} | Task: {task} | Episode: {episode_idx} | Frames: {T}"
        fig.suptitle(title, fontsize=14, fontweight='bold')

        frame_indices = np.linspace(0, T-1, num_frames_to_show, dtype=int)

        for i, frame_idx in enumerate(frame_indices):
            frame = video[frame_idx].cpu().numpy()

            # Denormalize if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).clip(0, 255).astype(np.uint8)

            axes[i].imshow(frame)
            axes[i].set_title(f'Frame {frame_idx}/{T-1}', fontweight='bold')
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(num_frames_to_show, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization to {save_path}")

        # Only show and close if not in interactive mode
        if self.args.mode != "interactive":
            plt.show()
        # Figure will be closed by caller if needed

        return fig

    def visualize_episodes(self):
        """Visualize multiple episodes."""
        print(f"\n{'='*80}")
        print(f"Visualizing Episodes")
        print(f"{'='*80}\n")

        for ep_idx, sample_idx in enumerate(self.episode_samples):
            print(f"\n--- Episode {ep_idx + 1}/{len(self.episode_samples)} (Sample {sample_idx}) ---")

            try:
                sample = self.dataset[sample_idx]

                # Print sample info
                print(f"  Keys: {list(sample.keys())}")
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}, dtype={value.dtype}, "
                              f"range=[{value.min():.3f}, {value.max():.3f}]")
                    else:
                        print(f"  {key}: {value}")

                # Visualize
                save_path = None
                if self.args.save_dir:
                    os.makedirs(self.args.save_dir + f'/{self.args.dataset_type}', exist_ok=True)
                    save_path = os.path.join(self.args.save_dir + f'/{self.args.dataset_type}', f"sample_{sample_idx:06d}.png")

                self.visualize_sample(sample, sample_idx, save_path)

            except Exception as e:
                print(f"  ⚠️  Error visualizing sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()

    def test_dataloader(self):
        """Test DataLoader batching."""
        print(f"\n{'='*80}")
        print(f"Testing DataLoader")
        print(f"{'='*80}\n")

        # Create DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=False,
        )

        print(f"DataLoader created:")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Num workers: {self.args.num_workers}")
        print(f"  Total batches: {len(self.dataloader)}\n")

        # Test loading batches
        num_test_batches = min(self.args.test_batches, len(self.dataloader))
        print(f"Testing {num_test_batches} batches...\n")

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Loading batches", total=num_test_batches)):
            if batch_idx >= num_test_batches:
                break

            if batch_idx == 0:
                # Print first batch details
                print(f"\nFirst batch contents:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {value.shape}, dtype={value.dtype}, "
                              f"range=[{value.min():.3f}, {value.max():.3f}]")
                    elif isinstance(value, list):
                        print(f"  {key}: list of {len(value)} items")
                        if len(value) > 0:
                            print(f"    First item: {value[0]}")
                    else:
                        print(f"  {key}: {type(value)}")

                # Visualize first sample from first batch
                if self.args.visualize_batch:
                    print("\nVisualizing first sample from first batch...")
                    first_sample = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            first_sample[key] = value[0]  # Get first item
                        elif isinstance(value, list):
                            first_sample[key] = value[0]
                        else:
                            first_sample[key] = value

                    save_path = None
                    if self.args.save_dir:
                        os.makedirs(self.args.save_dir + f'/{self.args.dataset_type}', exist_ok=True)
                        save_path = os.path.join(self.args.save_dir + f'/{self.args.dataset_type}', "batch_sample_0.png")

                    self.visualize_sample(first_sample, 0, save_path)

        print(f"\n✓ DataLoader test completed successfully!\n")

    def interactive_mode(self):
        """Interactive visualization with keyboard controls."""
        print(f"\n{'='*80}")
        print(f"Interactive Mode")
        print(f"{'='*80}\n")
        print("Controls:")
        print("  → / Right Arrow: Next sample")
        print("  ← / Left Arrow: Previous sample")
        print("  r: Random sample")
        print("  q / ESC: Quit")
        print("  s: Save current visualization")
        print(f"\n{'='*80}\n")

        self.current_idx = 0
        self.current_fig = None

        # Keyboard handler
        def on_key(event):
            if event.key in ['right', 'n']:
                self.current_idx = (self.current_idx + 1) % len(self.dataset)
                self._update_interactive_sample()
            elif event.key in ['left', 'p']:
                self.current_idx = (self.current_idx - 1) % len(self.dataset)
                self._update_interactive_sample()
            elif event.key == 'r':
                self.current_idx = random.randint(0, len(self.dataset) - 1)
                self._update_interactive_sample()
            elif event.key == 's':
                if self.current_fig is not None:
                    save_dir = self.args.save_dir or '.'
                    os.makedirs(save_dir + f'/{self.args.dataset_type}', exist_ok=True)
                    save_path = os.path.join(save_dir + f'/{self.args.dataset_type}', f"interactive_sample_{self.current_idx:06d}.png")
                    self.current_fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"💾 Saved to {save_path}")
            elif event.key in ['q', 'escape']:
                plt.close('all')
                return

        # Create initial visualization
        try:
            sample = self.dataset[self.current_idx]
            self.current_fig = self.visualize_sample(sample, self.current_idx)

            if self.current_fig is None:
                print(f"❌ Failed to create initial figure")
                return

            # Connect keyboard handler
            self.current_fig.canvas.mpl_connect('key_press_event', on_key)

            # Show non-blocking
            plt.show(block=True)

        except Exception as e:
            print(f"❌ Error in interactive mode: {e}")
            import traceback
            traceback.print_exc()

    def _update_interactive_sample(self):
        """Update interactive visualization."""
        print(f"\n{'='*40}")
        print(f"Loading sample {self.current_idx}...")
        print(f"{'='*40}")

        try:
            # Clear current figure
            if self.current_fig is not None:
                plt.clf()  # Clear figure content

            sample = self.dataset[self.current_idx]

            # Recreate visualization in same figure
            # This is a bit hacky but works for matplotlib
            plt.close(self.current_fig)
            self.current_fig = self.visualize_sample(sample, self.current_idx)

            if self.current_fig is not None:
                # Reconnect keyboard handler
                def on_key(event):
                    if event.key in ['right', 'n']:
                        self.current_idx = (self.current_idx + 1) % len(self.dataset)
                        self._update_interactive_sample()
                    elif event.key in ['left', 'p']:
                        self.current_idx = (self.current_idx - 1) % len(self.dataset)
                        self._update_interactive_sample()
                    elif event.key == 'r':
                        self.current_idx = random.randint(0, len(self.dataset) - 1)
                        self._update_interactive_sample()
                    elif event.key == 's':
                        if self.current_fig is not None:
                            save_dir = self.args.save_dir or '.'
                            os.makedirs(save_dir + f'/{self.args.dataset_type}', exist_ok=True)
                            save_path = os.path.join(save_dir + f'/{self.args.dataset_type}', f"interactive_sample_{self.current_idx:06d}.png")
                            self.current_fig.savefig(save_path, dpi=150, bbox_inches='tight')
                            print(f"💾 Saved to {save_path}")
                    elif event.key in ['q', 'escape']:
                        plt.close('all')
                        return

                self.current_fig.canvas.mpl_connect('key_press_event', on_key)
                plt.draw()
                plt.pause(0.001)
            else:
                print(f"⚠️  Failed to create figure for sample {self.current_idx}")

        except Exception as e:
            print(f"❌ Error updating sample: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main execution flow."""
        # Load dataset
        self.load_dataset()

        # Test DataLoader if requested
        if self.args.test_dataloader:
            self.test_dataloader()

        # Interactive mode
        if self.args.mode == "interactive":
            self.interactive_mode()

        # Sample episodes
        self.sample_episodes()

        # Visualize episodes
        self.visualize_episodes()

        print(f"\n{'='*80}")
        print(f"✓ Visualization completed!")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot Dataset Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize 5 random episodes from robot dataset
  python lerobot_dataloader_viz.py --dataset-type robot --num-episodes 5
  
  # Interactive mode for sim dataset
  python lerobot_dataloader_viz.py --dataset-type sim --mode interactive
  
  # Test dataloader with batching
  python lerobot_dataloader_viz.py --dataset-type robot --test-dataloader --batch-size 16
  
  # Visualize specific episodes
  python lerobot_dataloader_viz.py --dataset-type human --mode manual --episode-ids 0,5,10,15
        """
    )

    # Dataset selection
    parser.add_argument('--dataset-type', type=str, required=True,
                       choices=['robot', 'sim', 'human'],
                       help='Type of dataset to visualize')

    # Visualization mode
    parser.add_argument('--mode', type=str, default='random',
                       choices=['random', 'sequential', 'manual', 'interactive'],
                       help='Visualization mode')

    parser.add_argument('--num-episodes', type=int, default=5,
                       help='Number of episodes to visualize')

    parser.add_argument('--steps-per-episode', type=int, default=10,
                       help='Steps per episode to visualize (not used currently)')

    parser.add_argument('--episode-ids', type=str, default=None,
                       help='Comma-separated episode IDs for manual mode')

    # DataLoader testing
    parser.add_argument('--test-dataloader', action='store_true',
                       help='Test DataLoader batching')

    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for DataLoader testing')

    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of DataLoader workers')

    parser.add_argument('--test-batches', type=int, default=10,
                       help='Number of batches to test')

    parser.add_argument('--visualize-batch', action='store_true',
                       help='Visualize first sample from first batch')

    # Dataset configuration
    parser.add_argument('--root', type=str, default='demos',
                       help='Root directory for datasets')

    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split')

    parser.add_argument('--dataset-file', type=str, default=None,
                       help='JSON file with dataset configurations')

    parser.add_argument('--task-description-file', type=str, default=None,
                       help='JSON file with task descriptions')

    parser.add_argument('--repo-id', type=str, default=None,
                       help='LeRobot repo ID (for single dataset)')

    # Camera configuration
    parser.add_argument('--cameras', type=str, nargs='+',
                       default=['cam1', 'cam2'],
                       help='Camera names to load')

    parser.add_argument('--include-depth', action='store_true',
                       help='Include depth images')

    parser.add_argument('--image-size', type=int, nargs=2,
                       default=[224, 224],
                       help='Image size (H W)')

    # Robot/Sim specific
    parser.add_argument('--state-type', type=str, default='qpos',
                       choices=['eepos', 'qpos', 'mixpos'],
                       help='State type')

    parser.add_argument('--horizon', type=int, default=16,
                       help='Action horizon')

    parser.add_argument('--single-arm', action='store_true',
                       help='Use single arm data')

    parser.add_argument('--fps', type=int, default=30,
                       help='FPS for time calculations')

    parser.add_argument('--video-backend', type=str, default='torchcodec',
                       choices=['torchcodec', 'pyav', 'decord', 'opencv'],
                       help='Video backend')

    parser.add_argument('--normalization-method', type=str, default='bounds_q99',
                       choices=['bounds', 'bounds_q99'],
                       help='Normalization method')

    parser.add_argument('--depth-mode', type=str, default='sim',
                       choices=['sim', 'robot'],
                       help='Depth mode for sim dataset')

    # Human video specific
    parser.add_argument('--num-frames', type=int, default=10,
                       help='Number of frames for human video')

    parser.add_argument('--sampling-strategy', type=str, default='uniform_jitter',
                       choices=['uniform_jitter', 'random_interval'],
                       help='Frame sampling strategy')

    # Output
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save visualizations')

    args = parser.parse_args()

    # Create visualizer and run
    visualizer = DatasetVisualizer(args)
    visualizer.run()


if __name__ == "__main__":
    main()