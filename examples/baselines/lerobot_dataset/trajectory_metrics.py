"""
Trajectory Similarity Score (TSS)
==================================

A single normalized metric in [0, 1] measuring the shape similarity between
a predicted action trajectory and a ground-truth demonstration, robust to
speed differences (temporal warping handled by DTW).

Formula
-------
Given predicted trajectory A_pred in R^{T_pred x D} and
ground-truth trajectory A_gt in R^{T_gt x D}:

    DTW  = min_{W} sum_{(i,j) in W} ||a_i^pred - a_j^gt||_2
    nDTW = DTW / |W*|          (normalize by warping path length)
    TSS  = 1 / (1 + nDTW)     (map to (0,1]; 1=perfect, 0=worst)

where W* is the optimal warping path and |W*| its length.
Sakoe-Chiba band: r = max(floor(0.15 * max(T_pred, T_gt)), |T_pred - T_gt|).

References
----------
* Berndt & Clifford, KDD 1994 - original DTW
* Toohey & Duckham, SIGSPATIAL 2014 - nDTW removes length bias
* Gong et al., Sensors 2022 - DTW for robot trajectory evaluation
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# DTW core
# ---------------------------------------------------------------------------

def _dtw_core(
    seq_a: np.ndarray,  # (T_a, D)
    seq_b: np.ndarray,  # (T_b, D)
    band:  int,
) -> Tuple[float, int]:
    """
    Multi-dimensional DTW with Sakoe-Chiba band.

    Returns
    -------
    dtw_cost : float - accumulated optimal warping cost
    path_len : int   - length of the optimal warping path
    """
    T_a = seq_a.shape[0]
    T_b = seq_b.shape[0]
    INF = np.inf

    # Vectorized local cost: L2 over action dimensions
    diff       = seq_a[:, None, :] - seq_b[None, :, :]   # (T_a, T_b, D)
    local_cost = np.linalg.norm(diff, axis=-1)             # (T_a, T_b)

    # DP with Sakoe-Chiba band
    acc = np.full((T_a, T_b), INF, dtype=np.float64)
    for i in range(T_a):
        j_lo = max(0, i - band)
        j_hi = min(T_b, i + band + 1)
        for j in range(j_lo, j_hi):
            c = local_cost[i, j]
            if i == 0 and j == 0:
                acc[i, j] = c
            elif i == 0:
                acc[i, j] = c + acc[i, j - 1]
            elif j == 0:
                acc[i, j] = c + acc[i - 1, j]
            else:
                acc[i, j] = c + min(acc[i-1, j-1], acc[i-1, j], acc[i, j-1])

    dtw_cost = float(acc[T_a - 1, T_b - 1])

    # Back-track to get path length
    path_len = 1
    i, j = T_a - 1, T_b - 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = min(
                (acc[i-1, j-1], i-1, j-1),
                (acc[i-1, j],   i-1, j  ),
                (acc[i,   j-1], i,   j-1),
            )
            i, j = best[1], best[2]
        path_len += 1

    return dtw_cost, path_len


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TrajectoryMetrics:
    """
    Compute Trajectory Similarity Score (TSS) between a predicted trajectory
    and a ground-truth demonstration trajectory.

    TSS in (0, 1]:
        TSS = 1   -> identical trajectories
        TSS -> 0  -> maximally different trajectories

    Parameters
    ----------
    band_ratio : float
        Sakoe-Chiba band as a fraction of max(T_pred, T_gt). Default 0.15.
        Enlarged to abs(T_pred - T_gt) when necessary so the DP can always
        reach the far corner (avoids DTW = inf for very different lengths).
    """

    def __init__(self, band_ratio: float = 0.15):
        self.band_ratio = band_ratio

    def compute(
        self,
        pred_traj: ArrayLike,  # (T_pred, D)
        gt_traj:   ArrayLike,  # (T_gt, D)
    ) -> Dict[str, float]:
        """
        Compute TSS and its underlying nDTW.

        Returns
        -------
        {
            "tss"  : float in (0, 1]  - primary metric (higher = more similar)
            "ndtw" : float >= 0       - normalized DTW (lower = more similar)
                                        kept for raw reporting / sanity check
        }
        """
        pred = _to_numpy(pred_traj).astype(np.float64)
        gt   = _to_numpy(gt_traj  ).astype(np.float64)

        if pred.ndim == 1:
            pred = pred[:, None]
        if gt.ndim == 1:
            gt = gt[:, None]

        T_pred, D_pred = pred.shape
        T_gt,   D_gt   = gt.shape

        if D_pred != D_gt:
            raise ValueError(
                f"Action dimension mismatch: pred={D_pred}, gt={D_gt}."
            )
        if T_pred < 2 or T_gt < 2:
            raise ValueError("Trajectories must have at least 2 steps.")

        # Sakoe-Chiba band: enlarge to abs(T_pred-T_gt) so DP always reaches
        # the far corner when sequence lengths differ significantly.
        band = max(
            max(1, int(self.band_ratio * max(T_pred, T_gt))),
            abs(T_pred - T_gt),
        )

        dtw_cost, path_len = _dtw_core(pred, gt, band)
        ndtw = dtw_cost / path_len
        tss  = 1.0 / (1.0 + ndtw)

        return {"tss": float(tss), "ndtw": float(ndtw)}

    def compute_batch(
        self,
        pred_trajs: List[ArrayLike],
        gt_traj:    ArrayLike,
    ) -> Dict[str, np.ndarray]:
        """Compute TSS/nDTW for a list of predicted trajectories (one GT)."""
        results = [self.compute(p, gt_traj) for p in pred_trajs]
        return {k: np.array([r[k] for r in results]) for k in results[0]}


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_tss(
    pred_traj:  ArrayLike,
    gt_traj:    ArrayLike,
    band_ratio: float = 0.15,
) -> float:
    """
    One-line TSS computation.

    Parameters
    ----------
    pred_traj  : (T_pred, D) executed action trajectory
    gt_traj    : (T_gt,   D) ground-truth demonstration trajectory
    band_ratio : Sakoe-Chiba band fraction (default 0.15)

    Returns
    -------
    TSS in (0, 1] - higher is better
    """
    return TrajectoryMetrics(band_ratio=band_ratio).compute(pred_traj, gt_traj)["tss"]


# ---------------------------------------------------------------------------
# GT trajectory provider
# ---------------------------------------------------------------------------

class GTTrajectoryProvider:
    """Loads ground-truth action trajectories from a sim LeRobot dataset."""

    def __init__(
        self,
        sim_dataset_file: str,
        sim_root:         str,
        action_key:       str = "action.qpos_gripper_actions",
        normalizer=None,
        normalization_method: str = "bounds_q99",
    ):
        import json
        import pyarrow.parquet as pq
        from pathlib import Path

        self.action_key           = action_key
        self.normalizer           = normalizer
        self.normalization_method = normalization_method
        self._episodes: Dict[str, List[np.ndarray]] = {}

        with open(sim_dataset_file) as f:
            dataset_configs = json.load(f)

        for ds_cfg in dataset_configs:
            repo_id   = ds_cfg.get("repo_id")
            data_path = Path(sim_root) / ds_cfg.get("root") / "data"
            if not data_path.exists():
                continue

            trajs: List[np.ndarray] = []
            for pf in sorted(data_path.glob("**/*.parquet")):
                try:
                    table = pq.read_table(pf)
                    if self.action_key not in table.column_names:
                        continue
                    actions = table.column(self.action_key).to_pylist()
                    ep_col  = (table.column("episode_index").to_pylist()
                               if "episode_index" in table.column_names else None)

                    if ep_col is not None:
                        from itertools import groupby
                        for _, grp in groupby(zip(ep_col, actions), key=lambda x: x[0]):
                            ep = np.array([a for _, a in grp], dtype=np.float32)
                            if len(ep) > 0:
                                trajs.append(ep)
                    else:
                        ep = np.array(actions, dtype=np.float32)
                        if len(ep) > 0:
                            trajs.append(ep)
                except Exception:
                    continue

            if trajs:
                self._episodes[repo_id] = trajs
                if "_" in repo_id:
                    self._episodes[repo_id.split("_", 1)[1]] = trajs

    def sample_gt_trajectory(
        self,
        task_id:     str,
        seed:        Optional[int] = None,
        normalize:   bool = False,
        dataset_idx: int  = 0,
    ) -> Optional[np.ndarray]:
        """Return a randomly sampled GT trajectory (T, D) for task_id."""
        if task_id not in self._episodes:
            for k in self._episodes:
                if k.endswith(task_id) or task_id.endswith(k):
                    task_id = k
                    break
            else:
                return None

        episodes = self._episodes[task_id]
        if not episodes:
            return None

        traj = episodes[int(np.random.default_rng(seed).integers(0, len(episodes)))]

        if normalize and self.normalizer is not None:
            t    = torch.tensor(traj)
            t    = self.normalizer.normalize_action(
                t, dataset_idx, method=self.normalization_method
            )
            traj = t.numpy()

        return traj


# ---------------------------------------------------------------------------
# Action buffer
# ---------------------------------------------------------------------------

class EpisodeActionBuffer:
    """Accumulates per-step actions during an evaluation episode."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._bufs: List[List[np.ndarray]] = [[] for _ in range(num_envs)]

    def append(self, action: ArrayLike) -> None:
        arr = _to_numpy(action)
        for i in range(self.num_envs):
            self._bufs[i].append(arr[i])

    def get_and_reset(self) -> List[np.ndarray]:
        trajs = [
            np.stack(self._bufs[i], axis=0) if self._bufs[i]
            else np.empty((0, 0), dtype=np.float32)
            for i in range(self.num_envs)
        ]
        self._bufs = [[] for _ in range(self.num_envs)]
        return trajs