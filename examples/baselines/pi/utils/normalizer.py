"""
Action Normalizer for Pi Training
Compatible with OpenVLA's normalization format and methods.
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional, Dict


class ActionNormalizer:
    """
    Normalize/denormalize actions for training stability.
    Compatible with OpenVLA's format.
    """
    
    def __init__(
        self,
        demo_path: Optional[str] = None,
        action_dim: int = 8,
        dataset_name: str = "pi_maniskill",
        stats_dict: Optional[Dict] = None,
    ):
        """
        Args:
            demo_path: Path to demonstration file for computing statistics
            action_dim: Action dimension
            dataset_name: Name for saving/loading statistics
            stats_dict: Precomputed statistics dict (OpenVLA format)
                       Format: {dataset_name: {"action": {...}, "num_transitions": ..., ...}}
        """
        self.action_dim = action_dim
        self.dataset_name = dataset_name
        
        # Load statistics
        if stats_dict is not None:
            # OpenVLA format: stats_dict[dataset_name]['action']
            self.stats = stats_dict[dataset_name]['action']
        elif demo_path is not None:
            self.stats = self._compute_statistics(demo_path)
        else:
            raise ValueError("Must provide either demo_path or stats_dict")
    
    def _compute_statistics(self, demo_path: str) -> Dict:
        """Compute normalization statistics from demonstrations."""
        print(f"Computing action statistics from {demo_path}")
        
        # Load actions from HDF5
        from examples.baselines.pi.utils.utils import load_demo_dataset_with_lan
        
        trajectories = load_demo_dataset_with_lan(demo_path, concat=False)
        
        # Collect all actions
        all_actions = []
        for action_traj in trajectories["actions"]:
            if isinstance(action_traj, torch.Tensor):
                action_traj = action_traj.numpy()
            all_actions.append(action_traj)
        
        all_actions = np.concatenate(all_actions, axis=0)  # [N, action_dim]
        
        print(f"Loaded {all_actions.shape[0]} action samples")
        
        # Compute statistics (OpenVLA format)
        stats = {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
            "q01": np.percentile(all_actions, 1, axis=0).tolist(),
            "q99": np.percentile(all_actions, 99, axis=0).tolist(),
        }
        
        print("Statistics computed:")
        print(f"  Min: {stats['min']}")
        print(f"  Max: {stats['max']}")
        print(f"  Q01: {stats['q01']}")
        print(f"  Q99: {stats['q99']}")
        
        return stats
    
    def normalize(
        self,
        actions: Union[np.ndarray, torch.Tensor],
        method: str = "bounds_q99"
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize actions to [-1, 1] range.
        
        Args:
            actions: Actions to normalize, shape [..., action_dim]
            method: Normalization method
                - "bounds": Use min/max
                - "bounds_q99": Use 1st/99th percentile (robust to outliers)
                - "gaussian": Use mean/std (z-score normalization)
        
        Returns:
            Normalized actions in [-1, 1] range
        """
        is_torch = isinstance(actions, torch.Tensor)
        original_shape = actions.shape
        original_dtype = actions.dtype if is_torch else actions.dtype
        device = actions.device if is_torch else None
        
        # Convert to numpy for computation
        actions_np = actions.cpu().numpy() if is_torch else actions
        actions_np = actions_np.reshape(-1, self.action_dim)
        
        # Get bounds based on method
        if method == "bounds":
            low = np.array(self.stats["min"])
            high = np.array(self.stats["max"])
        elif method == "bounds_q99":
            low = np.array(self.stats["q01"])
            high = np.array(self.stats["q99"])
        elif method == "gaussian":
            # For gaussian, we'll use mean/std and scale to [-1, 1]
            mean = np.array(self.stats["mean"])
            std = np.array(self.stats["std"])
            # Z-score normalization then scale to approximately [-1, 1]
            normalized = (actions_np - mean) / (std + 1e-8)
            normalized = normalized / 3.0  # 3-sigma ≈ 99.7% data
            normalized = np.clip(normalized, -1.0, 1.0)
            normalized = normalized.reshape(original_shape)
            
            if is_torch:
                return torch.from_numpy(normalized).to(device=device, dtype=original_dtype)
            return normalized.astype(original_dtype)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Normalize to [-1, 1] (OpenVLA formula)
        normalized = 2.0 * (actions_np - low) / (high - low + 1e-8) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)
        
        # Reshape back
        normalized = normalized.reshape(original_shape)
        
        # Convert back to original type
        if is_torch:
            return torch.from_numpy(normalized).to(device=device, dtype=original_dtype)
        return normalized.astype(original_dtype)
    
    def denormalize(
        self,
        normalized_actions: Union[np.ndarray, torch.Tensor],
        method: str = "bounds_q99"
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize actions from [-1, 1] range back to original scale.
        
        Args:
            normalized_actions: Normalized actions in [-1, 1] range
            method: Denormalization method (must match normalization method)
        
        Returns:
            Denormalized actions in original scale
        """
        is_torch = isinstance(normalized_actions, torch.Tensor)
        original_shape = normalized_actions.shape
        original_dtype = normalized_actions.dtype if is_torch else normalized_actions.dtype
        device = normalized_actions.device if is_torch else None
        
        # Convert to numpy for computation
        normalized_np = normalized_actions.cpu().numpy() if is_torch else normalized_actions
        normalized_np = normalized_np.reshape(-1, self.action_dim)
        
        # Get bounds based on method
        if method == "bounds":
            low = np.array(self.stats["min"])
            high = np.array(self.stats["max"])
        elif method == "bounds_q99":
            low = np.array(self.stats["q01"])
            high = np.array(self.stats["q99"])
        elif method == "gaussian":
            # Reverse gaussian normalization
            mean = np.array(self.stats["mean"])
            std = np.array(self.stats["std"])
            denormalized = normalized_np * 3.0 * (std + 1e-8) + mean
            denormalized = denormalized.reshape(original_shape)
            
            if is_torch:
                return torch.from_numpy(denormalized).to(device=device, dtype=original_dtype)
            return denormalized.astype(original_dtype)
        else:
            raise ValueError(f"Unknown denormalization method: {method}")
        
        # Denormalize from [-1, 1] (OpenVLA formula)
        actions = 0.5 * (normalized_np + 1.0) * (high - low + 1e-8) + low
        
        # Reshape back
        actions = actions.reshape(original_shape)
        
        # Convert back to original type
        if is_torch:
            return torch.from_numpy(actions).to(device=device, dtype=original_dtype)
        return actions.astype(original_dtype)
    
    # Alias for compatibility with OpenVLA
    def unnormalize(
        self,
        normalized_actions: Union[np.ndarray, torch.Tensor],
        method: str = "bounds_q99"
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Alias for denormalize() to match OpenVLA's API.
        """
        return self.denormalize(normalized_actions, method=method)
    
    def save(self, save_path: str):
        """
        Save statistics to JSON file (OpenVLA format).
        
        Format:
        {
          "dataset_name": {
            "action": {
              "mean": [...],
              "std": [...],
              ...
            },
            "num_transitions": ...,
            "num_trajectories": ...
          }
        }
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # OpenVLA format: nested structure
        dataset_statistics = {
            self.dataset_name: {
                "action": self.stats,
                "num_transitions": -1,  # Optional, can be computed
                "num_trajectories": -1,  # Optional, can be computed
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(dataset_statistics, f, indent=2)
        
        print(f"✅ Saved normalization statistics to {save_path}")
    
    @classmethod
    def load(cls, stats_path: str, dataset_name: str = "pi_maniskill") -> 'ActionNormalizer':
        """
        Load statistics from JSON file (OpenVLA format).
        
        Args:
            stats_path: Path to statistics JSON file
            dataset_name: Dataset name key in the JSON file
        
        Returns:
            ActionNormalizer instance
        """
        with open(stats_path, 'r') as f:
            all_stats = json.load(f)
        
        if dataset_name not in all_stats:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in statistics file!\n"
                f"Available datasets: {list(all_stats.keys())}"
            )
        
        dataset_stats = all_stats[dataset_name]
        action_stats = dataset_stats["action"]
        action_dim = len(action_stats["q01"])
        
        # Create normalizer with stats_dict format
        stats_dict = {dataset_name: {"action": action_stats}}
        
        normalizer = cls(
            demo_path=None,
            action_dim=action_dim,
            dataset_name=dataset_name,
            stats_dict=stats_dict
        )
        
        print(f"✅ Loaded normalization statistics from {stats_path}")
        return normalizer


def save_complete_dataset_statistics(
    demo_path: str,
    save_path: str,
    dataset_name: str = "pi_maniskill",
    action_dim: int = 8,
):
    """
    Compute and save complete dataset statistics (OpenVLA format).
    
    This is a utility function to create normalization statistics file
    that can be used by both training and evaluation.
    
    Args:
        demo_path: Path to demonstration HDF5 file
        save_path: Path to save statistics JSON
        dataset_name: Name for the dataset
        action_dim: Action dimension
    """
    from examples.baselines.pi.utils.utils import load_demo_dataset_with_lan
    
    print(f"Computing statistics for dataset: {dataset_name}")
    trajectories = load_demo_dataset_with_lan(demo_path, concat=False)
    
    # Collect all actions
    all_actions = []
    for action_traj in trajectories["actions"]:
        if isinstance(action_traj, torch.Tensor):
            action_traj = action_traj.numpy()
        all_actions.append(action_traj)
    
    all_actions = np.concatenate(all_actions, axis=0)
    num_transitions = len(all_actions)
    num_trajectories = len(trajectories["actions"])
    
    # Compute statistics
    action_stats = {
        "mean": all_actions.mean(axis=0).tolist(),
        "std": all_actions.std(axis=0).tolist(),
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "q01": np.percentile(all_actions, 1, axis=0).tolist(),
        "q99": np.percentile(all_actions, 99, axis=0).tolist(),
    }
    
    # Build statistics dictionary (OpenVLA format)
    dataset_statistics = {
        dataset_name: {
            "action": action_stats,
            "num_transitions": int(num_transitions),
            "num_trajectories": int(num_trajectories),
        }
    }
    
    # Save to file
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(dataset_statistics, f, indent=2)
    
    print(f"\n✅ Saved complete dataset statistics:")
    print(f"   File: {save_path}")
    print(f"   Dataset: {dataset_name}")
    print(f"   Transitions: {num_transitions}")
    print(f"   Trajectories: {num_trajectories}")


def test_normalizer():
    """Test normalizer with synthetic data."""
    print("\n" + "="*60)
    print("Testing ActionNormalizer (OpenVLA-compatible)")
    print("="*60 + "\n")
    
    # Create synthetic actions
    np.random.seed(42)
    action_dim = 8
    n_samples = 1000
    
    actions = np.random.randn(n_samples, action_dim) * 2 + 1
    actions[0] = 10  # outlier
    actions[1] = -10  # outlier
    
    # Compute stats
    action_stats = {
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "q01": np.percentile(actions, 1, axis=0).tolist(),
        "q99": np.percentile(actions, 99, axis=0).tolist(),
    }
    
    # Create normalizer with OpenVLA format
    stats_dict = {
        "test_dataset": {
            "action": action_stats,
            "num_transitions": n_samples,
            "num_trajectories": 10,
        }
    }
    
    normalizer = ActionNormalizer(
        action_dim=action_dim,
        dataset_name="test_dataset",
        stats_dict=stats_dict
    )
    
    # Test different methods
    for method in ["bounds", "bounds_q99", "gaussian"]:
        print(f"\nTesting method: {method}")
        print("-" * 40)
        
        # Normalize
        normalized = normalizer.normalize(actions, method=method)
        print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        print(f"Normalized mean: {normalized.mean():.3f}")
        
        # Denormalize
        denormalized = normalizer.denormalize(normalized, method=method)
        print(f"Denormalized range: [{denormalized.min():.3f}, {denormalized.max():.3f}]")
        
        # Check reconstruction error
        error = np.abs(actions - denormalized).mean()
        print(f"Reconstruction error: {error:.6f}")
        
        # For bounds_q99 and gaussian, error might be larger due to clipping
        if method == "bounds":
            assert error < 1e-5, f"Large reconstruction error: {error}"
    
    # Test with torch tensors
    print("\n" + "-"*40)
    print("Testing with PyTorch tensors")
    print("-"*40)
    
    actions_torch = torch.from_numpy(actions).float()
    normalized_torch = normalizer.normalize(actions_torch, method="bounds_q99")
    denormalized_torch = normalizer.denormalize(normalized_torch, method="bounds_q99")
    
    print(f"Input type: {type(actions_torch)}")
    print(f"Normalized type: {type(normalized_torch)}")
    print(f"Denormalized type: {type(denormalized_torch)}")
    
    # Test save/load
    print("\n" + "-"*40)
    print("Testing save/load")
    print("-"*40)
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    normalizer.save(temp_path)
    loaded_normalizer = ActionNormalizer.load(temp_path, dataset_name="test_dataset")
    
    # Verify loaded normalizer works
    normalized_loaded = loaded_normalizer.normalize(actions, method="bounds_q99")
    error = np.abs(normalized - normalized_loaded).mean()
    print(f"Save/load error: {error:.6f}")
    assert error < 1e-10, f"Save/load inconsistency: {error}"
    
    # Cleanup
    os.remove(temp_path)
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_normalizer()