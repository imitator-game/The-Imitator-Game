# IG modification: optional host-local sharding mesh for multi-node HSDP.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import os

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in enumerate(model.blocks):
        transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.blocks[layer_id] = transformer_block
    if getattr(model, 'enable_mcp', False):
        for group_id, mcp_group in enumerate(model.mcp_blocks):
            for block_id, transformer_block in enumerate(mcp_group):
                transformer_block = ptd_checkpoint_wrapper(
                    transformer_block, preserve_rng_state=False)
                model.mcp_blocks[group_id][block_id] = transformer_block


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}
    if os.environ.get("IG_LOCAL_SHARDING") == "1":
        # Keep foundation all-gathers on local NVLink; replicate over nodes.
        from torch.distributed.device_mesh import init_device_mesh
        local_world = int(os.environ["LOCAL_WORLD_SIZE"])
        world = int(os.environ["WORLD_SIZE"])
        assert world % local_world == 0
        fsdp_config["mesh"] = init_device_mesh(
            "cuda", (world // local_world, local_world),
            mesh_dim_names=("replicate", "shard"))

    for block in model.blocks:
        fully_shard(block.attn1, **fsdp_config)
        fully_shard(block.attn2, **fsdp_config)
        fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    if getattr(model, 'enable_mcp', False):
        for mcp_group in model.mcp_blocks:
            for block in mcp_group:
                fully_shard(block.attn1, **fsdp_config)
                fully_shard(block.attn2, **fsdp_config)
                fully_shard(block.ffn, **fsdp_config)
                fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
