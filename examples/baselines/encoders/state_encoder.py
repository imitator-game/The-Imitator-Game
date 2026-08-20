"""
State Encoder with Multiple Architecture Options
=================================================
Supports: MANO, Robot Joints, EE Pose, and various encoder architectures
"""

import math
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class StateType(Enum):
    MANO = "mano"  # Hand pose (MANO parameters)
    QPOS = "qpos"  # Joint positions
    EEPOS = "eepos"  # End-effector position
    QPOS_GRIPPER = "qpos_gripper"  # Joint + gripper
    EEPOS_GRIPPER = "eepos_gripper"  # EE + gripper
    FULL = "full"  # All available state


class StateEncoderType(Enum):
    MLP = "mlp"
    TRANSFORMER = "transformer"
    MAMBA = "mamba"
    TCN = "tcn"  # Temporal Convolutional Network
    LSTM = "lstm"
    GRU = "gru"
    S4 = "s4"  # Structured State Space


@dataclass
class StateEncoderConfig:
    # State specifications
    state_type: str = "mano"
    state_dim: int = 14  # Default for MANO (58*2)
    num_frames: int = 10
    
    # Architecture
    encoder_type: str = "transformer"
    hidden_dim: int = 256
    output_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    
    # MANO specific
    mano_global_orient_dim: int = 3
    mano_hand_pose_dim: int = 45
    mano_betas_dim: int = 10
    
    # Robot specific
    robot_dof: int = 8  # Per arm
    gripper_dim: int = 1
    dual_arm: bool = True
    
    # Advanced
    use_positional_encoding: bool = True
    use_state_embedding: bool = True
    normalize_input: bool = True
    
    # Fourier features
    use_fourier_features: bool = False
    fourier_dim: int = 64


# =============================================================================
# Positional Encodings
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 1000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class FourierFeatures(nn.Module):
    """Random Fourier Features for better high-frequency representation."""
    
    def __init__(self, input_dim: int, output_dim: int, sigma: float = 10.0):
        super().__init__()
        self.output_dim = output_dim
        B = torch.randn(input_dim, output_dim // 2) * sigma
        self.register_buffer('B', B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# =============================================================================
# State Preprocessors
# =============================================================================

class MANOPreprocessor(nn.Module):
    """Preprocess MANO hand parameters."""
    
    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config
        
        # Per-hand dimensions
        self.per_hand_dim = (config.mano_global_orient_dim + 
                            config.mano_hand_pose_dim + 
                            config.mano_betas_dim)  # 58
        
        # Separate embeddings for different MANO components
        self.orient_embed = nn.Linear(config.mano_global_orient_dim, config.hidden_dim // 4)
        self.pose_embed = nn.Linear(config.mano_hand_pose_dim, config.hidden_dim // 2)
        self.betas_embed = nn.Linear(config.mano_betas_dim, config.hidden_dim // 4)
        
        # Combine left and right
        self.hand_combine = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 116] = [B, T, 58*2]
        # [NEW] x: [B, T, 14] = [B, T, 58*2]
        B, T, D = x.shape
        
        # Split left and right hands
        left = x[..., :self.per_hand_dim]
        right = x[..., self.per_hand_dim:]
        
        def process_hand(h):
            orient = h[..., :3]
            pose = h[..., 3:48]
            betas = h[..., 48:]
            
            o = self.orient_embed(orient)
            p = self.pose_embed(pose)
            b = self.betas_embed(betas)
            
            return torch.cat([o, p, b], dim=-1)
        
        left_feat = process_hand(left)
        right_feat = process_hand(right)
        
        # Combine hands
        combined = left_feat + right_feat  # Simple sum, could use attention
        combined = self.hand_combine(combined)
        
        return self.norm(combined)


class RobotStatePreprocessor(nn.Module):
    """Preprocess robot state (qpos/eepos)."""
    
    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config
        
        input_dim = config.state_dim
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Optional: separate processing for different components
        if config.dual_arm:
            # Use state_dim // 2 so the linear layer matches the actual per-arm
            # slice width (D // 2) computed in forward(), rather than the
            # config defaults robot_dof + gripper_dim which may differ.
            arm_dim = config.state_dim // 2
            self.left_arm = nn.Linear(arm_dim, config.hidden_dim // 2)
            self.right_arm = nn.Linear(arm_dim, config.hidden_dim // 2)
            self.combine = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'left_arm') and self.config.dual_arm:
            B, T, D = x.shape
            half = D // 2
            left = self.left_arm(x[..., :half])
            right = self.right_arm(x[..., half:])
            x = self.combine(torch.cat([left, right], dim=-1))
        else:
            x = self.proj(x)
        return x


# =============================================================================
# MLP State Encoder
# =============================================================================

class MLPStateEncoder(nn.Module):
    """Simple MLP-based state encoder."""
    
    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        if config.state_type == "mano":
            self.preprocess = MANOPreprocessor(config)
        else:
            self.preprocess = RobotStatePreprocessor(config)
        
        # Temporal MLP
        self.temporal_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * config.num_frames, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        
        self.cls_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.seq_proj = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, state_dim]
        B, T, _ = x.shape
        
        x = self.preprocess(x)  # [B, T, hidden_dim]
        
        # Flatten temporal
        x_flat = rearrange(x, 'b t d -> b (t d)')
        cls_feat = self.temporal_mlp(x_flat)
        cls_feat = self.cls_proj(cls_feat)
        
        seq_feat = self.seq_proj(x)
        
        return cls_feat, seq_feat


# =============================================================================
# Transformer State Encoder
# =============================================================================

class TransformerStateEncoder(nn.Module):
    """Transformer-based state encoder with attention."""
    
    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        if config.state_type == "mano":
            self.preprocess = MANOPreprocessor(config)
        else:
            self.preprocess = RobotStatePreprocessor(config)
        
        # Positional encoding
        if config.use_positional_encoding:
            self.pos_enc = SinusoidalPositionalEncoding(config.hidden_dim)
        else:
            self.pos_enc = None
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.cls_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.seq_proj = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        
        x = self.preprocess(x)
        
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.transformer(x)
        x = self.norm(x)
        
        cls_feat = self.cls_proj(x[:, 0])
        seq_feat = self.seq_proj(x[:, 1:])
        
        return cls_feat, seq_feat


# =============================================================================
# TCN State Encoder
# =============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=self.padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        x = self.dropout(F.gelu(self.norm2(self.conv2(x))))
        return x + res


class TCNStateEncoder(nn.Module):
    """Temporal Convolutional Network for state encoding."""
    
    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        if config.state_type == "mano":
            self.preprocess = MANOPreprocessor(config)
        else:
            self.preprocess = RobotStatePreprocessor(config)
        
        # TCN blocks with increasing dilation
        channels = [config.hidden_dim] * config.num_layers
        self.tcn_blocks = nn.ModuleList()
        
        for i in range(config.num_layers):
            dilation = 2 ** i
            in_ch = config.hidden_dim if i == 0 else channels[i-1]
            self.tcn_blocks.append(
                TCNBlock(in_ch, channels[i], kernel_size=3, 
                        dilation=dilation, dropout=config.dropout)
            )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cls_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.seq_proj = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        
        x = self.preprocess(x)  # [B, T, hidden_dim]
        x = rearrange(x, 'b t d -> b d t')  # [B, hidden_dim, T]
        
        for block in self.tcn_blocks:
            x = block(x)
        
        cls_feat = self.pool(x).squeeze(-1)
        cls_feat = self.cls_proj(cls_feat)
        
        seq_feat = rearrange(x, 'b d t -> b t d')
        seq_feat = self.seq_proj(seq_feat)
        
        return cls_feat, seq_feat


# =============================================================================
# LSTM/GRU State Encoder
# =============================================================================

class RNNStateEncoder(nn.Module):
    """LSTM/GRU-based state encoder."""
    
    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        if config.state_type == "mano":
            self.preprocess = MANOPreprocessor(config)
        else:
            self.preprocess = RobotStatePreprocessor(config)
        
        # RNN
        rnn_cls = nn.LSTM if config.encoder_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.norm = nn.LayerNorm(config.hidden_dim * 2)
        self.cls_proj = nn.Linear(config.hidden_dim * 2, config.output_dim)
        self.seq_proj = nn.Linear(config.hidden_dim * 2, config.output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.preprocess(x)
        
        output, hidden = self.rnn(x)
        output = self.norm(output)
        
        # Use last hidden state for CLS
        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]
        
        # Concat forward and backward
        cls_feat = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        cls_feat = self.cls_proj(cls_feat)
        
        seq_feat = self.seq_proj(output)
        
        return cls_feat, seq_feat


# =============================================================================
# Mamba State Encoder (State Space Model)
# =============================================================================

class MambaBlock(nn.Module):
    """Simplified Mamba block."""
    
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        
        inner_dim = dim * expand
        
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(inner_dim, inner_dim, d_conv, 
                                padding=d_conv-1, groups=inner_dim)
        
        self.x_proj = nn.Linear(inner_dim, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(inner_dim, inner_dim, bias=True)
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(inner_dim, d_state))
        self.D = nn.Parameter(torch.ones(inner_dim))
        
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # Project and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv
        x = rearrange(x, 'b t d -> b d t')
        x = self.conv1d(x)[:, :, :T]
        x = rearrange(x, 'b d t -> b t d')
        x = F.silu(x)
        
        # SSM
        x_dbl = self.x_proj(x)
        dt = F.softplus(self.dt_proj(x))
        
        # Simplified selective scan
        A = -torch.exp(self.A)
        y = x * self.D + x  # Simplified
        
        # Gate and project
        y = y * F.silu(z)
        return self.norm(self.out_proj(y) + x)


class MambaStateEncoder(nn.Module):
    """Mamba (State Space Model) based encoder."""
    
    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        if config.state_type == "mano":
            self.preprocess = MANOPreprocessor(config)
        else:
            self.preprocess = RobotStatePreprocessor(config)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(config.hidden_dim) for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.cls_proj = nn.Linear(config.hidden_dim, config.output_dim)
        self.seq_proj = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.preprocess(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        cls_feat = self.cls_proj(x.mean(dim=1))
        seq_feat = self.seq_proj(x)
        
        return cls_feat, seq_feat


# =============================================================================
# Unified State Encoder Factory
# =============================================================================

class StateEncoder(nn.Module):
    """Unified state encoder supporting multiple architectures."""
    
    ENCODER_MAP = {
        "mlp": MLPStateEncoder,
        "transformer": TransformerStateEncoder,
        "tcn": TCNStateEncoder,
        "lstm": RNNStateEncoder,
        "gru": RNNStateEncoder,
        "mamba": MambaStateEncoder,
    }
    
    # Predefined state dimensions
    STATE_DIMS = {
        "mano": 14,  # 58 * 2 (both hands)
        "qpos": 14,   # 7 DoF * 2 arms
        "eepos": 14,  # xyz + quat * 2 arms
        "qpos_gripper": 16,  # qpos + 2 grippers
        "eepos_gripper": 16,  # eepos + 2 grippers
    }

    def __init__(self, config: StateEncoderConfig):
        super().__init__()
        self.config = config
        
        # Auto-set state dim if not specified
        if config.state_dim == 0:
            config.state_dim = self.STATE_DIMS.get(config.state_type, 14)
        
        encoder_cls = self.ENCODER_MAP.get(config.encoder_type)
        if encoder_cls is None:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")
        
        self.encoder = encoder_cls(config)
        
        # Optional Fourier features
        if config.use_fourier_features:
            self.fourier = FourierFeatures(config.state_dim, config.fourier_dim)
            self.fourier_proj = nn.Linear(config.fourier_dim, config.hidden_dim)
        else:
            self.fourier = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: State tensor [B, T, state_dim]
        Returns:
            cls_feat: [B, output_dim]
            seq_feat: [B, T, output_dim]
        """
        # Optional: add Fourier features
        if self.fourier is not None:
            fourier_feat = self.fourier(x)
            x = x + self.fourier_proj(fourier_feat)
        
        return self.encoder(x)

    def get_output_dim(self) -> int:
        return self.config.output_dim

    @classmethod
    def available_encoders(cls) -> List[str]:
        return list(cls.ENCODER_MAP.keys())


# =============================================================================
# Helper: Create encoder easily
# =============================================================================

def create_state_encoder(
    state_type: str = "mano",
    encoder_type: str = "transformer",
    state_dim: Optional[int] = None,
    output_dim: int = 512,
    num_frames: int = 10,
    **kwargs
) -> StateEncoder:
    """Factory function to create state encoder."""
    config = StateEncoderConfig(
        state_type=state_type,
        encoder_type=encoder_type,
        state_dim=state_dim or StateEncoder.STATE_DIMS.get(state_type, 14),
        output_dim=output_dim,
        num_frames=num_frames,
        **kwargs
    )
    return StateEncoder(config)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Available state encoders:", StateEncoder.available_encoders())
    print("State dimensions:", StateEncoder.STATE_DIMS)
    
    B, T = 4, 10
    
    # Test MANO state with Transformer
    print("\n--- MANO + Transformer ---")
    config = StateEncoderConfig(
        state_type="mano",
        state_dim=14,
        encoder_type="transformer",
        hidden_dim=256,
        output_dim=512,
        num_frames=T
    )
    encoder = StateEncoder(config)
    
    x = torch.randn(B, T, 14)
    cls_feat, seq_feat = encoder(x)
    print(f"Input: {x.shape}")
    print(f"CLS: {cls_feat.shape}")
    print(f"Seq: {seq_feat.shape}")
    
    params = sum(p.numel() for p in encoder.parameters())
    print(f"Params: {params:,}")
    
    # Test robot state with TCN
    print("\n--- Robot QPOS + TCN ---")
    encoder = create_state_encoder(
        state_type="qpos_gripper",
        encoder_type="tcn",
        state_dim=16,
        output_dim=512,
        num_frames=T
    )
    
    x = torch.randn(B, T, 16)
    cls_feat, seq_feat = encoder(x)
    print(f"Input: {x.shape}")
    print(f"CLS: {cls_feat.shape}")
    print(f"Seq: {seq_feat.shape}")
    
    # Test with Mamba
    print("\n--- MANO + Mamba ---")
    encoder = create_state_encoder(
        state_type="mano",
        encoder_type="mamba",
        output_dim=512,
        num_frames=T
    )
    
    x = torch.randn(B, T, 14)
    cls_feat, seq_feat = encoder(x)
    print(f"Input: {x.shape}")
    print(f"CLS: {cls_feat.shape}")
    print(f"Seq: {seq_feat.shape}")
    
    params = sum(p.numel() for p in encoder.parameters())
    print(f"Params: {params:,}")
