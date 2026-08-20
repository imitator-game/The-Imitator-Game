import torch
from typing import Dict


class ActionNormalizer:
    """
    Wrapper for ActionNormalizer that works with multiple datasets.
    Each dataset has its own normalization statistics.
    """

    def __init__(self):
        """Initialize empty normalizer that will be populated with dataset stats."""
        self.dataset_stats = {}  # Maps dataset_idx -> stats_dict
        self.dataset_info = {}   # Maps dataset_idx -> {repo_id, state_key, action_key}

    def add_dataset_stats(
        self,
        dataset_idx: int,
        repo_id: str,
        stats: Dict,
        state_key: str,
        action_key: str,
        single_arm: bool = False
    ):
        """
        Add normalization statistics for a specific dataset.

        Args:
            dataset_idx: Index of the dataset in ConcatDataset
            repo_id: Repository ID
            stats: Statistics dictionary from LeRobotDataset.meta.stats
            state_key: Key for state in stats
            action_key: Key for action in stats
            single_arm: Whether to extract only right arm
        """
        self.dataset_stats[dataset_idx] = stats
        self.dataset_info[dataset_idx] = {
            'repo_id': repo_id,
            'state_key': state_key,
            'action_key': action_key,
            'single_arm': single_arm
        }

    def _extract_right_arm(self, x: torch.Tensor, single_arm: bool) -> torch.Tensor:
        """Extract right arm data from [right_arm, left_arm, right_gripper, left_gripper] layout."""
        if not single_arm:
            return x
        M = x.shape[-1] // 2
        # Right arm: indices 0 to M-2, and 2*M-2
        return torch.cat([x[..., :M-1], x[..., 2*M-2 : 2*M-1]], dim=-1)

    def normalize_state(
        self,
        state: torch.Tensor,
        dataset_idx: int,
        method: str = "bounds_q99"
    ) -> torch.Tensor:
        """
        Normalize state using dataset-specific statistics.

        Args:
            state: State tensor to normalize
            dataset_idx: Which dataset's stats to use
            method: Normalization method ("bounds" or "bounds_q99")

        Returns:
            Normalized state in [-1, 1] range
        """
        if dataset_idx not in self.dataset_stats:
            raise ValueError(f"No stats found for dataset {dataset_idx}")

        stats = self.dataset_stats[dataset_idx]
        info = self.dataset_info[dataset_idx]
        state_key = info['state_key']
        single_arm = info['single_arm']

        if state_key not in stats:
            return state

        # Get bounds
        if method == "bounds":
            s_min = stats[state_key].get("min")
            s_max = stats[state_key].get("max")
        elif method == "bounds_q99":
            s_min = stats[state_key].get("q01")
            s_max = stats[state_key].get("q99")
        else:
            raise ValueError(f"Unknown method: {method}")

        if s_min is None or s_max is None:
            return state

        # Convert to tensor
        s_min = torch.tensor(s_min, dtype=state.dtype, device=state.device)
        s_max = torch.tensor(s_max, dtype=state.dtype, device=state.device)

        # Extract right arm if needed
        s_min = self._extract_right_arm(s_min, single_arm)
        s_max = self._extract_right_arm(s_max, single_arm)

        # Normalize to [-1, 1]
        denom = s_max - s_min
        denom[denom == 0] = 1.0
        normalized = 2.0 * (state - s_min) / denom - 1.0
        normalized = torch.clamp(normalized, -1.0, 1.0)

        return normalized

    def normalize_action(
        self,
        action: torch.Tensor,
        dataset_idx: int,
        method: str = "bounds_q99"
    ) -> torch.Tensor:
        """
        Normalize action using dataset-specific statistics.

        Args:
            action: Action tensor to normalize
            dataset_idx: Which dataset's stats to use
            method: Normalization method ("bounds" or "bounds_q99")

        Returns:
            Normalized action in [-1, 1] range
        """
        if dataset_idx not in self.dataset_stats:
            raise ValueError(f"No stats found for dataset {dataset_idx}")

        stats = self.dataset_stats[dataset_idx]
        info = self.dataset_info[dataset_idx]
        action_key = info['action_key']
        single_arm = info['single_arm']

        if action_key not in stats:
            return action

        # Get bounds
        if method == "bounds":
            a_min = stats[action_key].get("min")
            a_max = stats[action_key].get("max")
        elif method == "bounds_q99":
            a_min = stats[action_key].get("q01")
            a_max = stats[action_key].get("q99")
        else:
            raise ValueError(f"Unknown method: {method}")

        if a_min is None or a_max is None:
            return action

        # Convert to tensor
        a_min = torch.tensor(a_min, dtype=action.dtype, device=action.device)
        a_max = torch.tensor(a_max, dtype=action.dtype, device=action.device)

        # Extract right arm if needed
        a_min = self._extract_right_arm(a_min, single_arm)
        a_max = self._extract_right_arm(a_max, single_arm)

        # Normalize to [-1, 1]
        denom = a_max - a_min
        denom[denom == 0] = 1.0
        normalized = 2.0 * (action - a_min) / denom - 1.0
        normalized = torch.clamp(normalized, -1.0, 1.0)

        return normalized

    def denormalize_action(
        self,
        normalized_action: torch.Tensor,
        dataset_idx: int,
        method: str = "bounds_q99"
    ) -> torch.Tensor:
        """
        Denormalize action from [-1, 1] range back to original scale.

        Args:
            normalized_action: Normalized action tensor
            dataset_idx: Which dataset's stats to use
            method: Denormalization method (must match normalization method)

        Returns:
            Denormalized action in original scale
        """
        if dataset_idx not in self.dataset_stats:
            raise ValueError(f"No stats found for dataset {dataset_idx}")

        stats = self.dataset_stats[dataset_idx]
        info = self.dataset_info[dataset_idx]
        action_key = info['action_key']
        single_arm = info['single_arm']

        if action_key not in stats:
            return normalized_action

        # Get bounds
        if method == "bounds":
            a_min = stats[action_key].get("min")
            a_max = stats[action_key].get("max")
        elif method == "bounds_q99":
            a_min = stats[action_key].get("q01")
            a_max = stats[action_key].get("q99")
        else:
            raise ValueError(f"Unknown method: {method}")

        if a_min is None or a_max is None:
            return normalized_action

        # Convert to tensor
        a_min = torch.tensor(a_min, dtype=normalized_action.dtype, device=normalized_action.device)
        a_max = torch.tensor(a_max, dtype=normalized_action.dtype, device=normalized_action.device)

        # Extract right arm if needed
        a_min = self._extract_right_arm(a_min, single_arm)
        a_max = self._extract_right_arm(a_max, single_arm)

        # Denormalize from [-1, 1]
        action = 0.5 * (normalized_action + 1.0) * (a_max - a_min) + a_min

        return action

    def denormalize_state(
        self,
        normalized_state: torch.Tensor,
        dataset_idx: int,
        method: str = "bounds_q99"
    ) -> torch.Tensor:
        """
        Denormalize state from [-1, 1] range back to original scale.

        Args:
            normalized_state: Normalized state tensor
            dataset_idx: Which dataset's stats to use
            method: Denormalization method (must match normalization method)

        Returns:
            Denormalized state in original scale
        """
        if dataset_idx not in self.dataset_stats:
            raise ValueError(f"No stats found for dataset {dataset_idx}")

        stats = self.dataset_stats[dataset_idx]
        info = self.dataset_info[dataset_idx]
        state_key = info['state_key']
        single_arm = info['single_arm']

        if state_key not in stats:
            return normalized_state

        # Get bounds
        if method == "bounds":
            s_min = stats[state_key].get("min")
            s_max = stats[state_key].get("max")
        elif method == "bounds_q99":
            s_min = stats[state_key].get("q01")
            s_max = stats[state_key].get("q99")
        else:
            raise ValueError(f"Unknown method: {method}")

        if s_min is None or s_max is None:
            return normalized_state

        # Convert to tensor
        s_min = torch.tensor(s_min, dtype=normalized_state.dtype, device=normalized_state.device)
        s_max = torch.tensor(s_max, dtype=normalized_state.dtype, device=normalized_state.device)

        # Extract right arm if needed
        s_min = self._extract_right_arm(s_min, single_arm)
        s_max = self._extract_right_arm(s_max, single_arm)

        # Denormalize from [-1, 1]
        state = 0.5 * (normalized_state + 1.0) * (s_max - s_min) + s_min

        return state