import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, Optional


class ActionNormalizer:

    def __init__(
            self,
            demo_path: Optional[str] = None,
            action_dim: int = 8,
            dataset_name: str = "maniskill_pickcubeycb",
            stats_dict: Optional[Dict] = None
    ):
        self.action_dim = action_dim
        self.dataset_name = dataset_name

        if stats_dict is not None:
            self.stats = stats_dict[dataset_name]['action']
        elif demo_path is not None:
            self.stats = self._compute_statistics(demo_path)
        else:
            raise ValueError("Either demo_path or stats_dict must be provided")

    def _compute_statistics(self, demo_path: str) -> Dict:
        from examples.baselines.openvla_oft.utils.utils import load_demo_dataset_with_lan

        trajectories = load_demo_dataset_with_lan(demo_path, concat=False)
        all_actions = []
        for action_traj in trajectories["actions"]:
            if isinstance(action_traj, torch.Tensor):
                action_traj = action_traj.numpy()
            all_actions.append(action_traj)

        all_actions = np.concatenate(all_actions, axis=0)
        stats = {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
            "q01": np.percentile(all_actions, 1, axis=0).tolist(),
            "q99": np.percentile(all_actions, 99, axis=0).tolist(),
        }

        return stats

    def normalize(self, actions: torch.Tensor, method: str = "bounds_q99") -> torch.Tensor:
        original_shape = actions.shape
        original_dtype = actions.dtype
        actions_np = actions.numpy() if isinstance(actions, torch.Tensor) else actions
        actions_np = actions_np.reshape(-1, self.action_dim)
        if method == "bounds":
            low = np.array(self.stats["min"])
            high = np.array(self.stats["max"])
        elif method == "bounds_q99":
            low = np.array(self.stats["q01"])
            high = np.array(self.stats["q99"])
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        normalized = 2.0 * (actions_np - low) / (high - low + 1e-8) - 1.0
        normalized = np.clip(normalized, -1.0, 1.0)

        normalized = normalized.reshape(original_shape)
        return torch.from_numpy(normalized).to(original_dtype)

    def unnormalize(
            self,
            normalized_actions: np.ndarray,
            method: str = "bounds_q99"
    ) -> np.ndarray:
        original_shape = normalized_actions.shape
        normalized_actions = normalized_actions.reshape(-1, self.action_dim)
        if method == "bounds":
            low = np.array(self.stats["min"])
            high = np.array(self.stats["max"])
        elif method == "bounds_q99":
            low = np.array(self.stats["q01"])
            high = np.array(self.stats["q99"])
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        actions = 0.5 * (normalized_actions + 1.0) * (high - low + 1e-8) + low

        actions = actions.reshape(original_shape)
        return actions

    def save(self, save_path: Path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_statistics = {
            self.dataset_name: {
                "action": self.stats,
                "num_transitions": -1,  # optional; can be computed during the compute step
                "num_trajectories": -1,
            }
        }

        with open(save_path, 'w') as f:
            json.dump(dataset_statistics, f, indent=2)

    @classmethod
    def load(cls, stats_path: Path, dataset_name: str = "maniskill_pickcubeycb") -> "ActionNormalizer":
        with open(stats_path, 'r') as f:
            all_stats = json.load(f)

        if dataset_name not in all_stats:
            raise KeyError(
                f"Dataset '{dataset_name}' is not in the statistics file!\n"
                f"Available datasets: {list(all_stats.keys())}"
            )
        dataset_stats = all_stats[dataset_name]
        action_stats = dataset_stats["action"]
        action_dim = len(action_stats["q01"])
        return cls(
            demo_path=None,
            action_dim=action_dim,
            dataset_name=dataset_name,
            stats_dict=action_stats
        )


def save_complete_dataset_statistics(
        demo_path: str,
        save_path: Path,
        dataset_name: str = "maniskill_pickcubeycb",
        action_dim: int = 8,
        include_proprio: bool = False
):
    from examples.baselines.openvla_oft.utils.utils import load_demo_dataset_with_lan

    trajectories = load_demo_dataset_with_lan(demo_path, concat=False)
    all_actions = []
    for action_traj in trajectories["actions"]:
        if isinstance(action_traj, torch.Tensor):
            action_traj = action_traj.numpy()
        all_actions.append(action_traj)

    all_actions = np.concatenate(all_actions, axis=0)
    num_transitions = len(all_actions)
    num_trajectories = len(trajectories["actions"])

    action_stats = {
        "mean": all_actions.mean(axis=0).tolist(),
        "std": all_actions.std(axis=0).tolist(),
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "q01": np.percentile(all_actions, 1, axis=0).tolist(),
        "q99": np.percentile(all_actions, 99, axis=0).tolist(),
    }

    # Build the statistics dictionary
    dataset_statistics = {
        dataset_name: {  # ✅ use "maniskill_pickcubeycb"
            "action": action_stats,
            "num_transitions": int(num_transitions),
            "num_trajectories": int(num_trajectories),
        }
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(dataset_statistics, f, indent=2)