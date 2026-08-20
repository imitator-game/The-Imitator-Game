"""
Unified testing utilities for LeRobot dataloaders.
Provides common visualization and validation functions.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_output_dir(base_name: str) -> Path:
    """Create output directory for test results."""
    output_dir = Path(f"test_lerobot_outputs/{base_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    return output_dir


def print_header(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 80}")
    print(title)
    print('=' * 80)


def print_dataset_info(dataset, dataset_name: str = "Dataset"):
    """Print dataset statistics."""
    print(f"\n📊 {dataset_name} Statistics:")
    print(f"  Total samples: {len(dataset)}")
    
    if hasattr(dataset, 'state_dim'):
        print(f"  State dimension: {dataset.state_dim}")
    if hasattr(dataset, 'action_dim'):
        print(f"  Action dimension: {dataset.action_dim}")
    if hasattr(dataset, 'horizon'):
        print(f"  Action horizon: {dataset.horizon}")
    if hasattr(dataset, 'cameras'):
        print(f"  Cameras: {dataset.cameras}")
    if hasattr(dataset, 'image_size'):
        print(f"  Image size: {dataset.image_size}")


def print_sample_info(sample: Dict[str, Any], sample_idx: int):
    """Print information about a sample."""
    print(f"\n📦 Sample {sample_idx} Data:")
    
    # Print metadata if available
    if 'repo_id' in sample:
        print(f"  📍 Data Location:")
        print(f"    - Repo ID: {sample.get('repo_id', 'N/A')}")
        if 'episode_index' in sample:
            ep_idx = sample['episode_index']
            if isinstance(ep_idx, torch.Tensor):
                ep_idx = ep_idx.item()
            print(f"    - Episode: {ep_idx}")
        if 'frame_index' in sample:
            frame_idx = sample['frame_index']
            if isinstance(frame_idx, torch.Tensor):
                frame_idx = frame_idx.item()
            print(f"    - Frame: {frame_idx}")
    
    # Print shapes
    print(f"  📊 Data Shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"    - {key}: {value.shape} [{value.dtype}]")
            
            # Check for zero dimensions in states/actions
            if key in ['states', 'actions']:
                if value.dim() == 1:
                    non_zero = (torch.abs(value) > 1e-6).sum().item()
                    print(f"      → Non-zero dims: {non_zero}/{value.shape[0]}")
                elif value.dim() == 2:
                    non_zero_per_timestep = (torch.abs(value) > 1e-6).sum(dim=1)
                    print(f"      → Non-zero dims per timestep: {non_zero_per_timestep.tolist()}")
        elif not key.startswith('view_'):
            print(f"    - {key}: {value}")


def visualize_images(sample: Dict[str, Any], 
                     output_dir: Path,
                     sample_idx: int = 0,
                     prefix: str = "sample"):
    """
    Visualize RGB and depth images from a sample.
    
    Args:
        sample: Sample dictionary containing view data
        output_dir: Directory to save visualizations
        sample_idx: Sample index for naming
        prefix: Prefix for output filenames
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metadata
    task_name = sample.get('task_descriptions', 'unknown')
    if isinstance(task_name, str):
        task_name = task_name.replace(' ', '_')[:20]
    
    repo_id = sample.get('repo_id', 'unknown')
    ep_idx = sample.get('episode_index', torch.tensor(0))
    if isinstance(ep_idx, torch.Tensor):
        ep_idx = ep_idx.item()
    
    frame_idx = sample.get('frame_index', torch.tensor(sample_idx))
    if isinstance(frame_idx, torch.Tensor):
        frame_idx = frame_idx.item()
    
    # Find all view keys
    view_keys = [k for k in sample.keys() if k.startswith('view_')]
    
    if not view_keys:
        print("  ⚠️  No view data found in sample")
        return
    
    # Visualize each view
    for view_key in view_keys:
        view_tensor = sample[view_key]  # [C+D, H, W]
        
        num_channels = view_tensor.shape[0]
        has_depth = num_channels > 3
        
        if has_depth:
            rgb_channels = view_tensor[:3]
            depth_channel = view_tensor[3:]
        else:
            rgb_channels = view_tensor
            depth_channel = None
        
        # Convert to numpy
        rgb_np = rgb_channels.permute(1, 2, 0).cpu().numpy()
        rgb_np = np.clip(rgb_np * 0.5 + 0.5, 0, 1)
        
        # Create figure
        if has_depth:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            axes[0].imshow(rgb_np)
            axes[0].set_title(f'RGB - {view_key}', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            depth_np = depth_channel[0].cpu().numpy()
            im = axes[1].imshow(depth_np, cmap='viridis')
            axes[1].set_title(f'Depth - {view_key}', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            ax.imshow(rgb_np)
            ax.set_title(f'RGB - {view_key}', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        fig.suptitle(
            f'📍 {repo_id} | Episode: {ep_idx} | Frame: {frame_idx} | Task: {task_name}',
            fontsize=14, fontweight='bold', y=0.98
        )
        plt.tight_layout()
        
        save_path = output_dir / f"{prefix}_{sample_idx}_{view_key}_ep{ep_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📷 Saved: {save_path.name}")
        plt.close()


def visualize_states(sample: Dict[str, Any],
                     output_dir: Path,
                     sample_idx: int = 0,
                     prefix: str = "sample",
                     filter_zeros: bool = True,
                     zero_threshold: float = 1e-6):
    """Visualize state vector."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'states' not in sample:
        print("  ⚠️  No states found in sample")
        return
    
    states = sample['states']
    state_dim = states.shape[0]
    states_np = states.cpu().numpy()
    
    # Extract metadata
    task_name = sample.get('task_descriptions', 'unknown')
    if isinstance(task_name, str):
        task_name = task_name.replace(' ', '_')[:20]
    
    repo_id = sample.get('repo_id', 'unknown')
    ep_idx = sample.get('episode_index', torch.tensor(0))
    if isinstance(ep_idx, torch.Tensor):
        ep_idx = ep_idx.item()
    
    frame_idx = sample.get('frame_index', torch.tensor(sample_idx))
    if isinstance(frame_idx, torch.Tensor):
        frame_idx = frame_idx.item()
    
    # Filter zero dimensions
    active_dims = []
    if filter_zeros:
        for dim_idx in range(state_dim):
            if not np.abs(states_np[dim_idx]) < zero_threshold:
                active_dims.append(dim_idx)
        
        if len(active_dims) == 0:
            print(f"  ⚠️  All state dimensions are zero")
            active_dims = list(range(state_dim))
    else:
        active_dims = list(range(state_dim))
    
    num_active = len(active_dims)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, num_active * 0.8), 5))
    
    filter_info = f"Showing {num_active}/{state_dim} non-zero dims" if filter_zeros else f"All {state_dim} dims"
    fig.suptitle(
        f'📊 State Vector - {repo_id}\n'
        f'Episode: {ep_idx} | Frame: {frame_idx} | Task: {task_name}\n'
        f'{filter_info}',
        fontsize=14, fontweight='bold'
    )
    
    active_values = states_np[active_dims]
    x_positions = np.arange(num_active)
    
    colors = ['steelblue' if v >= 0 else 'coral' for v in active_values]
    bars = ax.bar(x_positions, active_values, color=colors, alpha=0.7)
    
    ax.set_xlabel('State Dimension', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'Dim {i}' for i in active_dims], rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, active_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=8)
    
    plt.tight_layout()
    save_path = output_dir / f"{prefix}_{sample_idx}_states_ep{ep_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Saved: {save_path.name} (filtered {state_dim - num_active} zero dims)")
    plt.close()


def visualize_actions(sample: Dict[str, Any],
                      output_dir: Path,
                      sample_idx: int = 0,
                      prefix: str = "sample",
                      filter_zeros: bool = True,
                      zero_threshold: float = 1e-6):
    """Visualize action sequence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'actions' not in sample:
        print("  ⚠️  No actions found in sample")
        return
    
    actions = sample['actions']
    horizon, action_dim = actions.shape
    actions_np = actions.cpu().numpy()
    
    # Extract metadata
    task_name = sample.get('task_descriptions', 'unknown')
    if isinstance(task_name, str):
        task_name = task_name.replace(' ', '_')[:20]
    
    repo_id = sample.get('repo_id', 'unknown')
    ep_idx = sample.get('episode_index', torch.tensor(0))
    if isinstance(ep_idx, torch.Tensor):
        ep_idx = ep_idx.item()
    
    frame_idx = sample.get('frame_index', torch.tensor(sample_idx))
    if isinstance(frame_idx, torch.Tensor):
        frame_idx = frame_idx.item()
    
    # Filter zero dimensions
    active_dims = []
    if filter_zeros:
        for dim_idx in range(action_dim):
            dim_values = actions_np[:, dim_idx]
            if not np.all(np.abs(dim_values) < zero_threshold):
                active_dims.append(dim_idx)
        
        if len(active_dims) == 0:
            print(f"  ⚠️  All action dimensions are zero")
            active_dims = list(range(action_dim))
    else:
        active_dims = list(range(action_dim))
    
    num_active = len(active_dims)
    
    # Create figure
    ncols = min(4, num_active)
    nrows = (num_active + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    
    filter_info = f"Showing {num_active}/{action_dim} non-zero dims" if filter_zeros else f"All {action_dim} dims"
    fig.suptitle(
        f'🎯 Action Sequence - {repo_id}\n'
        f'Episode: {ep_idx} | Frame: {frame_idx} | Task: {task_name}\n'
        f'Horizon: {horizon} | {filter_info}',
        fontsize=14, fontweight='bold'
    )
    
    for plot_idx, dim_idx in enumerate(active_dims):
        row = plot_idx // ncols
        col = plot_idx % ncols
        ax = axes[row, col]
        
        dim_values = actions_np[:, dim_idx]
        ax.plot(dim_values, marker='o', linewidth=2, markersize=4)
        ax.set_title(f'Action Dim {dim_idx}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

        val_min, val_max = dim_values.min(), dim_values.max()
        ax.text(0.02, 0.98, f'Range: [{val_min:.3f}, {val_max:.3f}]',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(num_active, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    save_path = output_dir / f"{prefix}_{sample_idx}_actions_ep{ep_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  🎯 Saved: {save_path.name} (filtered {action_dim - num_active} zero dims)")
    plt.close()


def visualize_batch_grid(batch: Dict[str, Any],
                         output_dir: Path,
                         batch_idx: int = 0):
    """Visualize a grid of samples from a batch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = len(batch.get('task_descriptions', batch.get('task', [])))
    view_keys = [k for k in batch.keys() if k.startswith('view_')]
    
    if not view_keys:
        print("  ⚠️  No view data in batch")
        return
    
    view_key = view_keys[0]
    view_batch = batch[view_key]  # [B, C+D, H, W]
    
    num_show = min(8, batch_size)
    ncols = 4
    nrows = (num_show + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    
    fig.suptitle(f'Batch {batch_idx} - {view_key} (RGB only)', fontsize=16, fontweight='bold')
    
    for i in range(num_show):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        view_tensor = view_batch[i, :3]  # [3, H, W]
        rgb_np = view_tensor.permute(1, 2, 0).cpu().numpy()
        rgb_np = np.clip(rgb_np * 0.5 + 0.5, 0, 1)
        
        ax.imshow(rgb_np)
        
        task_key = 'task_descriptions' if 'task_descriptions' in batch else 'task'
        if task_key in batch:
            task_desc = str(batch[task_key][i])[:30]
            ax.set_title(f'Sample {i}\n{task_desc}...', fontsize=9)
        else:
            ax.set_title(f'Sample {i}', fontsize=9)
        ax.axis('off')
        
        rect = patches.Rectangle((0, 0), rgb_np.shape[1] - 1, rgb_np.shape[0] - 1,
                                linewidth=3, edgecolor=plt.cm.tab10(i % 10), facecolor='none')
        ax.add_patch(rect)
    
    for idx in range(num_show, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    save_path = output_dir / f"batch_{batch_idx}_grid.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📸 Saved: {save_path.name}")
    plt.close()


def print_summary(output_dir: Path):
    """Print test summary."""
    print_header("📊 TEST SUMMARY")
    
    print(f"\n💾 Output Directory: {output_dir}")
    
    png_files = list(output_dir.glob("*.png"))
    if png_files:
        print(f"\n📁 Generated {len(png_files)} visualizations:")
        for file in sorted(png_files):
            file_size = file.stat().st_size / 1024
            print(f"    - {file.name} ({file_size:.1f} KB)")
    else:
        print("\n  ⚠️  No visualization files generated")
    
    print(f"\n{'=' * 80}")
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print('=' * 80)