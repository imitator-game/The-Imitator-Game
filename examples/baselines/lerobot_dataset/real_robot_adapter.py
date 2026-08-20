"""
Real-robot streaming adapter that matches LeRobotRobotDataset output format.

This module is ROS-agnostic.  Callers provide raw observations via
``update(...)`` then fetch normalised samples via ``get_latest_sample()``.

The ``observation`` dict passed to ``update()`` must already have depth in the
format produced by the training pipeline (i.e., after encode→decode), because
``_process_view`` does NOT re-encode raw metric depth.  The ROS control node
callback is responsible for that step:

    raw_16UC1 → ÷1000 → depth2rgb (encode) → decode_depth (decode)
    → store in observation["depth"][cam_name]

Output format of ``get_latest_sample()`` is identical to
``LeRobotRobotDataset.__getitem__()``:
    states   : (obs_horizon, state_dim)   normalised [-1,1]
    actions  : (action_dim,)              normalised [-1,1]  (optional)
    view_i   : (obs_horizon, C, H, W)    C=3 (RGB) or C=4 (RGBD)
    task_descriptions: str
    repo_id  : str
    dataset_idx: torch.long

To convert to agent.get_action() format use item_to_obs() from
eval_real_robot.py, which takes view_i[-1] and stacks cameras into
{"state": (1,D), "rgb": (1,H,W,C_total)}.

Bug fixes vs original:
  1. _state_buffer and _view_buffers pad with the first frame when the
     buffer has fewer than obs_horizon entries (episode start).
  2. View tensors are always (obs_horizon, C, H, W).
  3. Depth processing comments clarify the caller contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer


@dataclass
class RealRobotStreamConfig:
    # Visual input
    image_size: Tuple[int, int] = (224, 224)
    cameras: List[str] = field(default_factory=lambda: ["cam1", "cam2"])
    include_depth: bool = True
    depth_mode: str = "robot"          # "robot" | "sim"

    # State / action
    state_type: str = "qpos"           # "qpos" | "eepos" | "mixpos"
    single_arm: bool = False

    # Sequence horizons (must match training config)
    obs_horizon: int = 1
    horizon: int = 16

    # Normalization
    normalization_method: str = "bounds_q99"


class RealRobotStreamDataset:
    """
    Online adapter that emits samples compatible with LeRobotRobotDataset.

    Expected raw observation format for update():
    {
        "rgb":   {cam_name: np.ndarray[H, W, 3]  uint8 RGB},
        "depth": {cam_name: np.ndarray[H, W] float32 metres,
                             AFTER encode→decode (same quantisation as training)},
        "state": np.ndarray[dim] or torch.Tensor[dim],
        "action": np.ndarray[dim] or torch.Tensor[dim],   # optional
        "task_description": str,                           # optional
        "repo_id": str,                                    # optional
    }

    Depth contract:
        The caller MUST pass depth that has already been through the
        encode (depth2rgb) → decode (decode_depth) pipeline so that the
        quantisation artefacts match the training dataset.  In the ROS
        control node this is done in _camera_depth_callback before
        storing to self.current_depths[cam].
    """

    def __init__(
        self,
        config: RealRobotStreamConfig,
        *,
        normalizer: Optional[ActionNormalizer] = None,
        dataset_idx: int = 0,
        default_task_description: str = "",
        default_repo_id: str = "real_robot",
    ) -> None:
        self.config = config
        self.normalizer = normalizer
        self.dataset_idx = int(dataset_idx)
        self.default_task_description = default_task_description
        self.default_repo_id = default_repo_id

        obs_h = max(1, config.obs_horizon)
        self._state_buffer: deque[torch.Tensor] = deque(maxlen=obs_h)
        self._view_buffers: Dict[str, deque[torch.Tensor]] = {
            cam: deque(maxlen=obs_h) for cam in config.cameras
        }
        self._latest_sample: Optional[Dict[str, Any]] = None

        # Deterministic transforms (no colour-jitter, no spatial augmentation)
        self.geometric_transform = A.Compose(
            [A.Resize(height=config.image_size[0], width=config.image_size[1], p=1.0)],
            additional_targets={"depth": "image"},
        )
        self.rgb_transform = A.Compose(
            [
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
                            max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
        self.depth_transform = A.Compose([ToTensorV2()])

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _extract_right_arm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.single_arm:
            return x
        m = x.shape[-1] // 2
        return torch.cat([x[..., :m - 1], x[..., 2 * m - 2: 2 * m - 1]], dim=-1)

    def _get_depth_range(self, cam_name: str) -> Tuple[float, float]:
        """zrange used to clip depth; must match the encode zrange the caller used."""
        if self.config.depth_mode == "robot":
            return (0.0, 0.5) if "zed" in cam_name.lower() else (0.0, 4.0)
        # sim defaults
        if "wrist" in cam_name.lower() or "wristcam" in cam_name.lower():
            return (0.0, 1.0)
        if "cam2" in cam_name.lower():
            return (0.0, 2.0)
        return (0.0, 3.0)

    def _process_view(
        self,
        cam_name: str,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> torch.Tensor:
        """
        Process one camera frame into a (C, H, W) tensor.

        depth: float32 HW or HW1 in metres, ALREADY encode-decoded by caller.
               Values outside [zmin, zmax] are clipped.
        """
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(
                f"RGB for {cam_name} must be HWC with 3 channels, got {rgb.shape}"
            )

        if depth is not None:
            if depth.ndim == 2:
                depth = depth[..., None]
            depth = depth.astype(np.float32)
            zmin, zmax = self._get_depth_range(cam_name)
            depth = np.clip(depth, zmin, zmax)
            res = self.geometric_transform(image=rgb, depth=depth)
            aug_rgb = res["image"]
            aug_dep = res["depth"]
        else:
            res = self.geometric_transform(image=rgb)
            aug_rgb = res["image"]
            aug_dep = None

        t_rgb = self.rgb_transform(image=aug_rgb)["image"]       # (3, H, W)

        if aug_dep is not None:
            t_dep = self.depth_transform(image=aug_dep)["image"]
            if t_dep.dim() == 2:
                t_dep = t_dep.unsqueeze(0)
            return torch.cat([t_rgb, t_dep], dim=0)              # (4, H, W)

        return t_rgb                                              # (3, H, W)

    def _pad_buffer(
        self,
        buf: deque,
        target_len: int,
    ) -> List[torch.Tensor]:
        """Return exactly target_len frames, padding at the start with the first frame."""
        frames = list(buf)
        if len(frames) == 0:
            raise RuntimeError("Buffer is empty — call update() first.")
        if len(frames) < target_len:
            pad    = target_len - len(frames)
            frames = [frames[0]] * pad + frames
        return frames  # len == target_len

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, observation: Dict[str, Any]) -> None:
        """
        Ingest one observation step and (re-)build the latest sample dict.

        This is called once per control cycle in the ROS node with:
          observation["rgb"][cam]   = uint8 RGB (H,W,3) from ROS topic
          observation["depth"][cam] = float32 metric depth (H,W) after encode-decode
          observation["state"]      = current right-arm state vector (not normalised)
          observation["action"]     = optional GT action for offline debugging
        """
        rgb_dict:   Dict[str, np.ndarray] = observation.get("rgb",   {})
        depth_dict: Dict[str, np.ndarray] = observation.get("depth", {})

        # ── Process views ────────────────────────────────────────────────────
        for cam in self.config.cameras:
            if cam not in rgb_dict:
                continue
            view = self._process_view(cam, rgb_dict[cam], depth_dict.get(cam))
            self._view_buffers[cam].append(view)

        # ── Process state ────────────────────────────────────────────────────
        state = observation["state"]
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.dim() == 2:
            state = state[0]
        state = self._extract_right_arm(state.float())
        self._state_buffer.append(state)

        obs_h = max(1, self.config.obs_horizon)

        # ── Build state sequence (obs_horizon, state_dim) with padding ───────
        state_frames = self._pad_buffer(self._state_buffer, obs_h)
        states = torch.stack(state_frames, dim=0)   # (obs_h, state_dim)

        if self.normalizer is not None:
            states = self.normalizer.normalize_state(
                states, self.dataset_idx,
                method=self.config.normalization_method,
            )

        ret: Dict[str, Any] = {
            "states":            states.float(),
            "task_descriptions": str(
                observation.get("task_description", self.default_task_description)
            ),
            "repo_id":    str(observation.get("repo_id", self.default_repo_id)),
            "dataset_idx": torch.tensor(self.dataset_idx, dtype=torch.long),
        }

        # ── Build view sequences (obs_horizon, C, H, W) ─────────────────────
        for i, cam in enumerate(self.config.cameras):
            buf = self._view_buffers.get(cam)
            if not buf:
                continue
            view_frames = self._pad_buffer(buf, obs_h)
            ret[f"view_{i + 1}"] = torch.stack(view_frames, dim=0)  # (obs_h, C, H, W)

        # ── Optional action (for offline eval / debugging) ───────────────────
        action = observation.get("action")
        if action is not None:
            if not isinstance(action, torch.Tensor):
                action = torch.as_tensor(action, dtype=torch.float32)
            if action.dim() == 2:
                action = action[0]
            action = self._extract_right_arm(action.float())
            if self.normalizer is not None:
                action = self.normalizer.normalize_action(
                    action, self.dataset_idx,
                    method=self.config.normalization_method,
                )
            ret["actions"] = action

        self._latest_sample = ret

    def get_latest_sample(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recent sample dict (dataset item format).
        To convert to agent obs format use item_to_obs() from eval_real_robot.py.
        """
        return self._latest_sample

    def reset(self) -> None:
        """Clear all internal buffers (call at episode start)."""
        for buf in self._view_buffers.values():
            buf.clear()
        self._state_buffer.clear()
        self._latest_sample = None

    # ── Denormalization helpers ──────────────────────────────────────────────

    def denormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return normalized_action
        return self.normalizer.denormalize_action(
            normalized_action, self.dataset_idx,
            method=self.config.normalization_method,
        )

    def denormalize_state(self, normalized_state: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return normalized_state
        return self.normalizer.denormalize_state(
            normalized_state, self.dataset_idx,
            method=self.config.normalization_method,
        )