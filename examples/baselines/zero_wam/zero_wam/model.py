"""Observed-state adapter and native action channel mapping."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask

IG_CHANNELS = (14, 15, 16, 17, 18, 19, 20, 28, 21, 22, 23, 24, 25, 26, 27, 29)


class StateConditionedActionEmbedding(nn.Linear):
    def __init__(self, base):
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=torch.float32,
        )
        self.weight = nn.Parameter(base.weight.detach().float().clone())
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().float().clone())
        self.proprio_projector = nn.Sequential(nn.Linear(18, 512), nn.SiLU(), nn.Linear(512, base.out_features)).to(
            device=base.weight.device, dtype=torch.float32
        )
        nn.init.zeros_(self.proprio_projector[-1].weight)
        nn.init.zeros_(self.proprio_projector[-1].bias)
        self.current_state = None
        self.tokens_per_chunk = 32

    def forward(self, x):
        if self.current_state is None:
            raise RuntimeError("Observed state must be supplied before action embedding")
        state = self.current_state
        if state.ndim == 2:
            state = state[:, None]
        condition = self.proprio_projector(state.to(self.proprio_projector[0].weight.dtype))
        if condition.shape[1] == 1:
            condition = condition.expand(-1, x.shape[1], -1)
        else:
            condition = condition.repeat_interleave(self.tokens_per_chunk, dim=1)[:, : x.shape[1]]
        return F.linear(x.to(self.weight.dtype), self.weight, self.bias) + condition.to(x.dtype)


def pack_actions(actions, channel_offset=14):
    if actions.shape[-1] != 16:
        raise ValueError("Expected IG action16")
    packed = actions.new_zeros(*actions.shape[:-1], 30)
    from .action_space import channel_ids
    packed[..., list(channel_ids(channel_offset))] = actions
    return packed


def unpack_actions(actions, channel_offset=14):
    if actions.shape[-1] != 30:
        raise ValueError("Expected native action30")
    from .action_space import channel_ids
    return actions[..., list(channel_ids(channel_offset))]


def eager_block_mask(*args, **kwargs):
    kwargs["_compile"] = False
    return create_block_mask(*args, **kwargs)
