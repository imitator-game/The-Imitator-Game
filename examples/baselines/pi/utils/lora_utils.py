"""
Manual LoRA implementation - full fix version
Problems fixed:
1. dtype mismatch
2. freezing non-LoRA parameters
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List


class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute LoRA adaptation."""
        if self.dropout is not None:
            x = self.dropout(x)

        # 🔥 FIX: also match device and dtype
        lora_A = self.lora_A.to(x.device, dtype=x.dtype)
        lora_B = self.lora_B.to(x.device, dtype=x.dtype)

        # LoRA: (B @ A @ x) * scaling
        result = (x @ lora_A.T @ lora_B.T) * self.scaling
        return result


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 32,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Original linear layer (frozen)
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA adaptation
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Move LoRA to same device/dtype as linear
        self.lora.to(linear.weight.device, dtype=linear.weight.dtype)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def in_features(self):
        return self.linear.in_features

    @property
    def out_features(self):
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original + LoRA."""
        return self.linear(x) + self.lora(x)


def freeze_all_except_lora(model: nn.Module, verbose: bool = False):
    """🔥 New: freeze all non-LoRA parameters"""
    frozen_count = 0
    lora_count = 0

    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
            lora_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"Parameter Freezing:")
        print(f"  LoRA parameters (trainable): {lora_count}")
        print(f"  Other parameters (frozen): {frozen_count}")
        print(f"{'='*60}\n")


def replace_linear_with_lora(
    module: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 32,
    alpha: float = 16.0,
    dropout: float = 0.0,
    verbose: bool = False,
    module_path: str = "",
    path_filter: Optional[List[str]] = None,
) -> int:
    """Recursively replace Linear layers with LoRALinear layers."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    replaced_count = 0

    for name, child in list(module.named_children()):
        full_path = f"{module_path}.{name}" if module_path else name

        # Path filter check
        if path_filter is not None:
            path_matches = any(pf in full_path for pf in path_filter)
            if not path_matches:
                replaced_count += replace_linear_with_lora(
                    module=child,
                    target_modules=target_modules,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    verbose=verbose,
                    module_path=full_path,
                    path_filter=path_filter,
                )
                continue

        # Check whether LoRA should be applied
        should_apply = any(target in name for target in target_modules)

        if isinstance(child, nn.Linear) and should_apply:
            # Replace with LoRALinear
            lora_linear = LoRALinear(
                linear=child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            setattr(module, name, lora_linear)
            replaced_count += 1

            if verbose:
                print(f"  ✓ Applied LoRA to: {full_path} "
                      f"(in={child.in_features}, out={child.out_features}, rank={rank})")
        else:
            replaced_count += replace_linear_with_lora(
                module=child,
                target_modules=target_modules,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                verbose=verbose,
                module_path=full_path,
                path_filter=path_filter,
            )

    return replaced_count


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 32,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    path_filter: Optional[List[str]] = None,
    verbose: bool = False,
) -> nn.Module:
    """Apply LoRA to the model and freeze the other parameters."""
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Applying LoRA to model:")
        print(f"  Rank: {rank}")
        print(f"  Alpha: {alpha}")
        print(f"  Dropout: {dropout}")
        print(f"  Target modules: {target_modules}")
        if path_filter:
            print(f"  🎯 Path filter: {path_filter}")
        print(f"{'='*60}\n")

    # 1. Apply LoRA
    replaced_count = replace_linear_with_lora(
        module=model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        verbose=verbose,
        module_path="",
        path_filter=path_filter,
    )

    # 🔥 2. Freeze all non-LoRA parameters
    freeze_all_except_lora(model, verbose=verbose)

    if verbose:
        print(f"\n{'='*60}")
        print(f"LoRA Application Summary:")
        print(f"  Replaced {replaced_count} Linear layers with LoRALinear")
        print(f"{'='*60}\n")

    # 3. Compute parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    if verbose:
        print(f"{'='*60}")
        print(f"Parameter Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable (LoRA): {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
        print(f"{'='*60}\n")

    if replaced_count == 0:
        raise ValueError(
            f"No layers were replaced with LoRA!\n"
            f"Target modules: {target_modules}\n"
            f"Path filter: {path_filter}\n"
            f"Please check that these match layers in your model."
        )

    # Warning: too many trainable parameters
    if trainable_params / total_params > 0.10:
        print(f"\n⚠️  WARNING: {100*trainable_params/total_params:.1f}% parameters trainable!")
        print(f"    For LoRA, this should typically be < 5%")
        print(f"    Check if path_filter is correct: {path_filter}\n")

    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only the state dict of the LoRA parameters."""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_state_dict[name] = param.data
    return lora_state_dict