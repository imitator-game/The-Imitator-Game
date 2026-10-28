"""LoRA and observed-state adapters; no changes to native attention or losses."""

import torch
from torch import nn
from torch.nn import functional as F

IG_CHANNELS = (14, 15, 16, 17, 18, 19, 20, 28, 21, 22, 23, 24, 25, 26, 27, 29)


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.scale = alpha / rank

    def forward(self, x):
        delta = F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return self.base(x) + (delta * self.scale).to(x.dtype)


def target_names(model):
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.startswith(("mcp_blocks.", "mcp_mlp_hidden.", "mcp_projections.")) and not model.enable_mcp:
            continue
        if name.startswith("blocks."):
            if any(s in name for s in [".attn1.", ".attn2.", ".ffn.", ".action_ffn."]):
                if any(s in name for s in ["to_q", "to_out", ".ffn.", ".action_ffn."]):
                    targets.append(name)
                elif ".attn1." in name and any(s in name for s in ["to_k", "to_v"]):
                    targets.append(name)
        elif name.startswith("mcp_blocks."):
            # One-block MCP groups have no downstream consumer of their action
            # outputs. Adapt video projections/FFN and action K/V used by video.
            if ".action_ffn." in name or "action_to_q" in name or "action_to_out" in name:
                continue
            if any(s in name for s in ["to_q", "to_out", ".ffn."]):
                targets.append(name)
            elif ".attn1." in name and any(s in name for s in ["to_k", "to_v"]):
                targets.append(name)
        elif name.startswith(("mcp_mlp_hidden.", "mcp_projections.")):
            targets.append(name)
    return targets


def adapt_transformer(model, config=None, *, rank=None):
    from .model import StateConditionedActionEmbedding

    scope = config.get("train_scope", "joint") if isinstance(config, dict) else config.train_scope
    names = target_names(model)
    per_rank = sum(model.get_submodule(n).in_features + model.get_submodule(n).out_features for n in names)
    model.requires_grad_(False)
    model.action_embedder = StateConditionedActionEmbedding(model.action_embedder)
    model.action_proj_out.float().requires_grad_(True)
    fixed = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank is None:
        rank = config.get("ig_lora_rank", config.get("adapter_rank")) if isinstance(config, dict) else config.lora_rank
        rank = rank or 106
    for name in names:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, LoRALinear(model.get_submodule(name), rank, rank * 2))
    trainable = {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad}
    report = dict(
        rank=rank,
        alpha=rank * 2,
        lora_targets=names,
        trainable_parameters=fixed + rank * per_rank,
        lora_parameters=rank * per_rank,
        adapter_head_parameters=fixed,
        trainable_shapes=trainable,
    )
    report["trainable_scope"] = scope
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual != fixed + rank * per_rank:
        raise ValueError(f"Adapter parameter accounting mismatch: {actual}")
    report["enable_mcp"] = model.enable_mcp
    model.ig_adaptation_report = report
    return report
