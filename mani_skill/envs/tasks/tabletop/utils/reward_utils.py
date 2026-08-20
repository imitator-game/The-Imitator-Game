"""
reward_utils.py — Universal Reward Utilities for ManiSkill Dual-Arm Tasks
==========================================================================

All sub-rewards output [0, 1].
Total reward = mean( peak(sub_reward_i) ) → [0, 1].

Verified interfaces used:
  - agent.tcp.pose.p              → (B, 3)
  - agent.is_grasping(actor)      → (B,) bool
  - agent.is_static(threshold)    → (B,) bool
  - scene.get_pairwise_contact_impulses(a, b) → (B, 3)
  - actor.pose.p                  → (B, 3)

Multi-robot note:
  In multi-robot (MultiAgent) setups the `action` passed to
  compute_dense_reward is a **dict** of tensors, NOT a single tensor.
  Never call action.shape directly — use get_batch_size(action, num_envs)
  or simply use self.num_envs.
"""

from __future__ import annotations
import torch
from typing import Dict, List, Union
from mani_skill.utils import common
from mani_skill.utils.geometry.geometry import transform_points


# ═══════════════════════════════════════════════════════════════════════════════
# §0  Multi-robot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_batch_size(
    action: Union[torch.Tensor, Dict[str, torch.Tensor]],
    fallback: int = 1,
) -> int:
    """Return the batch dimension from an action that may be a dict (multi-robot)
    or a plain tensor (single-robot)."""
    if isinstance(action, dict):
        for v in action.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
        return fallback
    if isinstance(action, torch.Tensor):
        return action.shape[0]
    return fallback


def get_action_device(
    action: Union[torch.Tensor, Dict[str, torch.Tensor]],
    fallback_device: torch.device = None,
) -> torch.device:
    """Return the device of the action tensor(s)."""
    if isinstance(action, dict):
        for v in action.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return fallback_device
    if isinstance(action, torch.Tensor):
        return action.device
    return fallback_device


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Distance → [0, 1] reward primitives
# ═══════════════════════════════════════════════════════════════════════════════

def tanh_reward(dist: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    """scale=5 → r ≈ 0.95 at d ≈ 0.06 m."""
    return 1.0 - torch.tanh(scale * dist)


def exp_reward(dist: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Sharper peak near zero."""
    return torch.exp(-scale * dist)


def normalized_progress(
    value: torch.Tensor,
    start: float,
    target: float,
) -> torch.Tensor:
    """
    Map scalar progress to [0, 1], supporting increasing or decreasing targets.
    """
    denom = target - start
    if abs(denom) < 1e-6:
        return torch.ones_like(value)
    progress = (value - start) / denom
    return torch.clamp(progress, 0.0, 1.0)


def geom_center_from_local_mesh(
    obj,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute batched geom center from the actor's local collision mesh and per-env pose.
    This avoids using `to_world_frame=True` on batched actors, which only reflects the
    first managed actor's pose.
    """
    collision_mesh = obj.get_first_collision_mesh(to_world_frame=False)
    bounds = common.to_tensor(collision_mesh.bounding_box.bounds, device=device)
    local_center = ((bounds[0] + bounds[1]) * 0.5).unsqueeze(0).repeat(len(obj.pose), 1)
    return transform_points(obj.pose.to_transformation_matrix().clone(), local_center)


def world_aabb_from_local_mesh(
    obj,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute a batched world-axis-aligned bounding box of shape (B, 2, 3) from the
    first local collision mesh and the per-env pose.
    """
    collision_mesh = obj.get_first_collision_mesh(to_world_frame=False)
    bounds = common.to_tensor(collision_mesh.bounding_box.bounds, device=device)
    lower, upper = bounds[0], bounds[1]
    corners = torch.tensor(
        [
            [lower[0], lower[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [lower[0], upper[1], lower[2]],
            [lower[0], upper[1], upper[2]],
            [upper[0], lower[1], lower[2]],
            [upper[0], lower[1], upper[2]],
            [upper[0], upper[1], lower[2]],
            [upper[0], upper[1], upper[2]],
        ],
        device=device,
        dtype=torch.float32,
    )
    batch_size = len(obj.pose)
    mats = obj.pose.to_transformation_matrix().clone()
    corners_world = (
        torch.bmm(
            corners.unsqueeze(0).repeat(batch_size, 1, 1),
            mats[:, :3, :3].transpose(2, 1),
        )
        + mats[:, None, :3, 3]
    )
    return torch.stack(
        [corners_world.min(dim=1).values, corners_world.max(dim=1).values], dim=1
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Reward building blocks (only verified APIs)
# ═══════════════════════════════════════════════════════════════════════════════

def reach_reward(
    tcp_pos: torch.Tensor,  # (B, 3)
    obj_pos: torch.Tensor,  # (B, 3)
    scale: float = 5.0,
) -> torch.Tensor:
    """TCP → object approach → [0, 1]."""
    dist = torch.linalg.norm(tcp_pos - obj_pos, dim=-1)
    return tanh_reward(dist, scale=scale)


def grasp_reward(
    tcp_pos: torch.Tensor,       # (B, 3)
    obj_pos: torch.Tensor,       # (B, 3)
    is_grasping: torch.Tensor,   # (B,) bool — from agent.is_grasping()
    proximity_scale: float = 5.0,
) -> torch.Tensor:
    """
    Grasp reward → [0, 1]:
      Not grasping → [0.0, 1.0)  proximity only (continuous approach signal)
      Grasping     → 1.0         confirmed, phase complete
    """
    dist = torch.linalg.norm(tcp_pos - obj_pos, dim=-1)
    proximity = tanh_reward(dist, scale=proximity_scale)
    return torch.where(is_grasping, torch.ones_like(proximity), proximity)


def transport_reward(
    obj_pos: torch.Tensor,       # (B, 3)
    goal_pos: torch.Tensor,      # (B, 3)
    is_grasping: torch.Tensor,   # (B,) bool
    scale: float = 5.0,
) -> torch.Tensor:
    """Object → goal, gated by grasp → [0, 1]."""
    dist = torch.linalg.norm(obj_pos - goal_pos, dim=-1)
    return tanh_reward(dist, scale=scale) * is_grasping.float()


def place_reward(
    obj_pos: torch.Tensor,       # (B, 3)
    goal_pos: torch.Tensor,      # (B, 3)
    is_grasping: torch.Tensor,   # (B,) bool
    scale: float = 15.0,
) -> torch.Tensor:
    """Precise placement after release → [0, 1]."""
    dist = torch.linalg.norm(obj_pos - goal_pos, dim=-1)
    return exp_reward(dist, scale=scale) * (~is_grasping).float()


def above_reward(
    obj_pos: torch.Tensor,       # (B, 3)
    target_pos: torch.Tensor,    # (B, 3)
    min_height: float,           # minimum height above target (metres)
    is_grasping: torch.Tensor,   # (B,) bool
    h_scale: float = 5.0,
) -> torch.Tensor:
    """
    Reward for positioning obj above target (XY close + height above),
    gated by grasp → [0, 1].

    Combines horizontal proximity and vertical progress as a product,
    so both conditions must be met simultaneously.
    """
    h_dist = torch.linalg.norm(obj_pos[..., :2] - target_pos[..., :2], dim=-1)
    height_diff = obj_pos[..., 2] - target_pos[..., 2]
    h_reward = tanh_reward(h_dist, scale=h_scale)
    v_reward = torch.clamp(height_diff / max(min_height, 1e-6), 0.0, 1.0)
    return h_reward * v_reward * is_grasping.float()


def wipe_progress_reward(
    steps: torch.Tensor,     # (B,) int — accumulated contact steps
    target_steps: int,
    dist: torch.Tensor,      # (B,) float — accumulated wipe distance
    target_dist: float,
) -> torch.Tensor:
    """
    Progress reward for cyclic contact tasks (wipe / clean / grind) → [0, 1].
    50 % from step count progress, 50 % from accumulated travel distance.
    Both are clamped to [0, 1] individually.
    """
    step_progress = torch.clamp(steps.float() / max(target_steps, 1), 0.0, 1.0)
    dist_progress = torch.clamp(dist / max(target_dist, 1e-6), 0.0, 1.0)
    return 0.5 * step_progress + 0.5 * dist_progress


def return_reward(
    obj_pos: torch.Tensor,  # (B, 3)
    home_pos: torch.Tensor, # (B, 3)
    gate: torch.Tensor,     # (B,) bool
    scale: float = 5.0,
) -> torch.Tensor:
    """Move object back to home, gated → [0, 1]."""
    dist = torch.linalg.norm(obj_pos - home_pos, dim=-1)
    return tanh_reward(dist, scale=scale) * gate.float()


def release_and_static_reward(
    is_grasping: torch.Tensor,      # (B,) bool
    is_robot_static: torch.Tensor,  # (B,) bool
    gate: torch.Tensor,             # (B,) bool
) -> torch.Tensor:
    """50 % release + 50 % static, gated → [0, 1]."""
    released = (~is_grasping) & gate
    static_ok = is_robot_static & gate
    return 0.5 * released.float() + 0.5 * static_ok.float()


def in_region_reward(
    obj_pos: torch.Tensor,       # (B, 3)
    region_center: torch.Tensor, # (B, 3)
    radius: float,
    height_threshold: float,
) -> torch.Tensor:
    """Binary: is the object inside a cylindrical region? → 0 or 1."""
    xy_dist = torch.linalg.norm(obj_pos[..., :2] - region_center[..., :2], dim=-1)
    in_xy = xy_dist <= radius
    in_z = obj_pos[..., 2] <= height_threshold
    return (in_xy & in_z).float()


# ═══════════════════════════════════════════════════════════════════════════════
# §3  RewardTracker — peak tracking + arithmetic mean
# ═══════════════════════════════════════════════════════════════════════════════

class RewardTracker:
    """
    Peak-tracking reward aggregator.
    total = mean( max_over_time(sub_i) for i in phases ) → [0, 1]
    """

    def __init__(
        self,
        phase_names: List[str],
        num_envs: int,
        device: torch.device,
    ):
        self.phase_names = list(phase_names)
        self.num_phases = len(self.phase_names)
        self.num_envs = num_envs
        self.device = device
        self._peaks: Dict[str, torch.Tensor] = {}
        self._current: Dict[str, torch.Tensor] = {}
        for name in self.phase_names:
            self._peaks[name] = torch.zeros(num_envs, device=device)
            self._current[name] = torch.zeros(num_envs, device=device)

    def reset(self, env_idx: torch.Tensor):
        for name in self.phase_names:
            self._peaks[name][env_idx] = 0.0
            self._current[name][env_idx] = 0.0

    def update(self, name: str, value: torch.Tensor):
        self._current[name] = value
        self._peaks[name] = torch.max(self._peaks[name], value)

    def total(self) -> torch.Tensor:
        """sum(peak_i) / N → [0, 1]."""
        result = torch.zeros(self.num_envs, device=self.device)
        for name in self.phase_names:
            result = result + self._peaks[name]
        return result / self.num_phases

    def get_peak_dict(self, prefix: str = "peak_r_") -> Dict[str, torch.Tensor]:
        return {
            f"{prefix}{name}": self._peaks[name].clone()
            for name in self.phase_names
        }

    def get_current_dict(self, prefix: str = "cur_r_") -> Dict[str, torch.Tensor]:
        return {
            f"{prefix}{name}": self._current[name].clone()
            for name in self.phase_names
        }

    def write_to_info(self, info: Dict):
        for i, name in enumerate(self.phase_names):
            info[f"R{i+1}_{name}"] = self._current[name]