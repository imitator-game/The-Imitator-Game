# Upstream provenance

Source: https://github.com/robbyant-research/Zero-WAM

Reference commit: `08e2c4ae41e2b63573a299825cebe6753481407c`.

This is a required-source subset, not a second complete upstream checkout. It contains no model assets, latent caches or dataset files. The Apache-2.0 license and existing source notices are retained.

Local changes relative to the released training path:

- `train.py`: retain native `_add_noise`, `_prepare_input_dict`, `compute_loss`, `_train_step` and the step loop; delegate data, adapters, logging and full-state checkpoints to the sibling `training` module. Remove unused dataset launchers, WandB startup and native dataset-specific path/statistics configuration. Activation checkpointing remains disabled.
- `distributed/fsdp.py`: support an optional host-local sharding mesh, replicated between hosts. Base compute uses BF16; gradient reductions use FP32.
- `modules/icl_model.py`: make padded per-example tokens invalid through the dataset mask helper, preserving the native causal/temporal predicates.
- `__init__.py`: avoid importing unused upstream launch configurations.
- `modules/model.py`: make FlashAttention an optional import; require it only when the `flashattn` backend is selected. The flex-attention path is unchanged.
- `utils/__init__.py`: expose the utilities used by this baseline without importing the unused websocket inference server.

The model/action architecture is reused from Zero-WAM. IG action-channel packing and observed-state conditioning live outside the vendored tree in `model.py`; LoRA scope/rank selection lives in `lora.py`. No custom replacement diffusion objective is introduced.

The current episode path restores the four native algorithmic methods directly from the reference commit. It calls native noise and loss without the historical per-sample batch-loss wrapper. The IG adapter selects episode data and proprio at chunk start, and supplies LoRA/checkpoint/logging hooks. Random training chunk1..4 and four MCP branches are enabled in the episode recipe; inference chunk2 is configured separately. See the parent README for cadence16 and IG pairing differences.
