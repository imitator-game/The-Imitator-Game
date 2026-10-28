# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch


def validate_mcp_settings(
    num_mcp_modules,
    mcp_blocks_per_group,
    mcp_hidden_collect_layers,
    mcp_loss_weights=None,
    num_layers=None,
):
    if num_mcp_modules <= 0:
        raise ValueError("num_mcp_modules must be positive")
    if mcp_blocks_per_group <= 0:
        raise ValueError("mcp_blocks_per_group must be positive")
    if len(mcp_hidden_collect_layers) == 0:
        raise ValueError("mcp_hidden_collect_layers cannot be empty")
    if len(set(mcp_hidden_collect_layers)) != len(mcp_hidden_collect_layers):
        raise ValueError("mcp_hidden_collect_layers must be unique")
    if any(layer < 0 for layer in mcp_hidden_collect_layers):
        raise ValueError("mcp_hidden_collect_layers must be non-negative")
    if num_layers is not None and max(mcp_hidden_collect_layers) >= num_layers:
        raise ValueError(
            "mcp_hidden_collect_layers must be smaller than the number of model layers"
        )
    if num_layers is not None and mcp_blocks_per_group > num_layers:
        raise ValueError(
            "mcp_blocks_per_group cannot exceed the number of model layers")
    if mcp_loss_weights is not None:
        if len(mcp_loss_weights) != num_mcp_modules:
            raise ValueError("mcp_loss_weights must match num_mcp_modules")
        if any(weight < 0 for weight in mcp_loss_weights):
            raise ValueError("mcp_loss_weights must be non-negative")


def shift_latents_for_mcp(latents, frame_shift):
    """Shift video latents into the future and mark targets backed by real frames."""
    if latents.ndim != 5:
        raise ValueError("latents must have shape [B, C, F, H, W]")
    if frame_shift <= 0:
        raise ValueError("frame_shift must be positive")

    batch_size, _, num_frames, _, _ = latents.shape
    if num_frames == 0:
        raise ValueError("latents must contain at least one frame")

    pad_frames = min(frame_shift, num_frames)
    shifted = latents[:, :, pad_frames:]
    last_frame = latents[:, :, -1:]
    shifted = torch.cat(
        [shifted, last_frame.expand(-1, -1, pad_frames, -1, -1)], dim=2
    )

    valid_frames = torch.arange(num_frames, device=latents.device) + frame_shift
    valid_frames = valid_frames < num_frames
    valid_mask = valid_frames.view(1, 1, num_frames, 1, 1).expand(
        batch_size, -1, -1, -1, -1
    )
    return shifted, valid_mask
