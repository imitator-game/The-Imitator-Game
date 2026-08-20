"""
Observation Encoder with Multiple Backbone Options
===================================================
Supports: SimpleCNN, ViT, ResNet, DINO, CLIP, DINOv2, SigLIP, etc.
"""

import math
from typing import Dict, Optional, Tuple, List, Literal
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class ObsEncoderType(Enum):
    SIMPLE_CNN = "simple_cnn"
    VIT_SCRATCH = "vit_scratch"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    DINO_VITB16 = "dino_vitb16"
    DINO_VITS16 = "dino_vits16"
    DINOV2_VITB14 = "dinov2_vitb14"
    DINOV2_VITL14 = "dinov2_vitl14"
    DINOV2_VITG14 = "dinov2_vitg14"
    CLIP_VITB32 = "clip_vitb32"
    CLIP_VITB16 = "clip_vitb16"
    CLIP_VITL14 = "clip_vitl14"
    SIGLIP_VITB16 = "siglip_vitb16"
    MAE_VITB16 = "mae_vitb16"
    EVA_VITG14 = "eva_vitg14"


@dataclass
class ObsEncoderConfig:
    encoder_type: str = "simple_cnn"
    image_size: int = 224
    output_dim: int = 512
    hidden_dim: int = 512
    
    # ViT specific
    patch_size: int = 16
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # Pretrained specific
    freeze_backbone: bool = False
    use_cls_token: bool = True
    use_patch_tokens: bool = True
    
    # Fine-tuning
    finetune_layers: int = 0  # Number of layers to unfreeze from top


# =============================================================================
# Simple CNN Encoder
# =============================================================================

class SimpleCNNEncoder(nn.Module):
    """Lightweight CNN encoder for fast training."""
    
    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            
            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(512, config.output_dim)
        
        # For patch tokens output
        self.patch_proj = nn.Conv2d(512, config.output_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, H, W]
        feat = self.features(x)  # [B, 512, 7, 7]
        
        # CLS token (global)
        cls_feat = self.pool(feat).flatten(1)  # [B, 512]
        cls_feat = self.proj(cls_feat)  # [B, output_dim]
        
        # Patch tokens
        patch_feat = self.patch_proj(feat)  # [B, output_dim, 7, 7]
        patch_feat = rearrange(patch_feat, 'b c h w -> b (h w) c')  # [B, 49, output_dim]
        
        return cls_feat, patch_feat


# =============================================================================
# ViT Encoder (From Scratch)
# =============================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTScratchEncoder(nn.Module):
    """Vision Transformer from scratch."""
    
    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        self.patch_embed = PatchEmbedding(
            config.image_size, config.patch_size, 3, config.hidden_dim
        )
        
        num_patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            ViTBlock(config.hidden_dim, config.num_heads, dropout=config.dropout)
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.proj = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        cls_feat = self.proj(x[:, 0])  # [B, output_dim]
        patch_feat = self.proj(x[:, 1:])  # [B, num_patches, output_dim]
        
        return cls_feat, patch_feat


# =============================================================================
# ResNet Encoder
# =============================================================================

class ResNetEncoder(nn.Module):
    """ResNet-based encoder with torchvision pretrained weights."""
    
    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        import torchvision.models as models
        
        if config.encoder_type == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
            feat_dim = 512
        else:  # resnet50
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
            feat_dim = 2048
        
        # Remove final FC
        self.backbone.fc = nn.Identity()
        
        # Feature projection
        self.proj = nn.Linear(feat_dim, config.output_dim)
        
        # For patch tokens, we need intermediate features
        self.patch_proj = nn.Conv2d(feat_dim, config.output_dim, 1)
        
        # Freeze if needed
        if config.freeze_backbone:
            self._freeze_backbone()
        elif config.finetune_layers > 0:
            self._freeze_except_top_layers(config.finetune_layers)

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _freeze_except_top_layers(self, num_layers: int):
        # Freeze all first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze top layers
        layers = [self.backbone.layer4, self.backbone.layer3, 
                  self.backbone.layer2, self.backbone.layer1]
        for layer in layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get intermediate features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        feat_map = self.backbone.layer4(x)  # [B, C, H, W]
        
        # Global feature
        cls_feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        cls_feat = self.proj(cls_feat)
        
        # Patch features
        patch_feat = self.patch_proj(feat_map)
        patch_feat = rearrange(patch_feat, 'b c h w -> b (h w) c')
        
        return cls_feat, patch_feat


# =============================================================================
# DINO Encoder
# =============================================================================

class DINOEncoder(nn.Module):
    """DINO (Self-distillation with no labels) encoder."""
    
    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        # Load DINO model
        if "vitb16" in config.encoder_type:
            self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            feat_dim = 768
        else:  # vits16
            self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            feat_dim = 384
        
        self.proj = nn.Linear(feat_dim, config.output_dim)
        self.patch_proj = nn.Linear(feat_dim, config.output_dim)
        
        if config.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get intermediate features
        x = self.backbone.prepare_tokens(x)
        
        for blk in self.backbone.blocks:
            x = blk(x)
        
        x = self.backbone.norm(x)
        
        cls_feat = self.proj(x[:, 0])
        patch_feat = self.patch_proj(x[:, 1:])
        
        return cls_feat, patch_feat


# =============================================================================
# DINOv2 Encoder
# =============================================================================

class DINOv2Encoder(nn.Module):
    """DINOv2 encoder with various model sizes."""
    
    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        # Model mapping
        model_map = {
            "dinov2_vitb14": ("dinov2_vitb14", 768),
            "dinov2_vitl14": ("dinov2_vitl14", 1024),
            "dinov2_vitg14": ("dinov2_vitg14", 1536),
        }
        
        model_name, feat_dim = model_map[config.encoder_type]
        self.backbone = torch.hub.load('facebookresearch/dinov2:main', model_name)
        
        self.proj = nn.Linear(feat_dim, config.output_dim)
        self.patch_proj = nn.Linear(feat_dim, config.output_dim)
        
        if config.freeze_backbone:
            self._freeze_backbone()
        elif config.finetune_layers > 0:
            self._freeze_except_top_layers(config.finetune_layers)

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _freeze_except_top_layers(self, num_layers: int):
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze top blocks
        for block in self.backbone.blocks[-num_layers:]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # DINOv2 forward with intermediate features
        features = self.backbone.forward_features(x)
        
        # features dict contains 'x_norm_clstoken' and 'x_norm_patchtokens'
        if isinstance(features, dict):
            cls_token = features['x_norm_clstoken']
            patch_tokens = features['x_norm_patchtokens']
        else:
            # Older API
            cls_token = features[:, 0]
            patch_tokens = features[:, 1:]
        
        cls_feat = self.proj(cls_token)
        patch_feat = self.patch_proj(patch_tokens)
        
        return cls_feat, patch_feat


# =============================================================================
# CLIP Encoder
# =============================================================================

class CLIPEncoder(nn.Module):
    """CLIP vision encoder."""
    
    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        try:
            import clip
        except ImportError:
            raise ImportError("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")
        
        model_map = {
            "clip_vitb32": ("ViT-B/32", 512),
            "clip_vitb16": ("ViT-B/16", 512),
            "clip_vitl14": ("ViT-L/14", 768),
        }
        
        model_name, feat_dim = model_map[config.encoder_type]
        self.clip_model, self.preprocess = clip.load(model_name, device='cpu')
        self.backbone = self.clip_model.visual
        
        self.proj = nn.Linear(feat_dim, config.output_dim)
        self.patch_proj = nn.Linear(feat_dim, config.output_dim)
        
        if config.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # CLIP expects normalized input
        x = x.type(self.backbone.conv1.weight.dtype)
        
        # Patch embedding
        x = self.backbone.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        
        # Add class token
        cls_token = self.backbone.class_embedding.expand(x.shape[0], 1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.backbone.positional_embedding
        
        x = self.backbone.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.backbone.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.backbone.ln_post(x)
        
        cls_feat = self.proj(x[:, 0].float())
        patch_feat = self.patch_proj(x[:, 1:].float())
        
        return cls_feat, patch_feat


# =============================================================================
# SigLIP Encoder
# =============================================================================

class SigLIPEncoder(nn.Module):
    """SigLIP vision encoder (requires transformers)."""
    
    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        from transformers import SiglipVisionModel, SiglipImageProcessor
        
        model_name = "google/siglip-base-patch16-224"
        self.backbone = SiglipVisionModel.from_pretrained(model_name)
        self.processor = SiglipImageProcessor.from_pretrained(model_name)
        
        feat_dim = self.backbone.config.hidden_size
        
        self.proj = nn.Linear(feat_dim, config.output_dim)
        self.patch_proj = nn.Linear(feat_dim, config.output_dim)
        
        if config.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        
        last_hidden = outputs.last_hidden_state
        
        # SigLIP uses mean pooling for CLS
        cls_feat = self.proj(last_hidden.mean(dim=1))
        patch_feat = self.patch_proj(last_hidden)
        
        return cls_feat, patch_feat


# =============================================================================
# MAE Encoder
# =============================================================================

class MAEEncoder(nn.Module):
    """Masked Autoencoder encoder."""
    
    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        # Load MAE from timm
        import timm
        
        self.backbone = timm.create_model('vit_base_patch16_224.mae', pretrained=True)
        feat_dim = 768
        
        self.proj = nn.Linear(feat_dim, config.output_dim)
        self.patch_proj = nn.Linear(feat_dim, config.output_dim)
        
        if config.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get features before head
        x = self.backbone.forward_features(x)
        
        cls_feat = self.proj(x[:, 0])
        patch_feat = self.patch_proj(x[:, 1:])
        
        return cls_feat, patch_feat


# =============================================================================
# EVA Encoder
# =============================================================================

class EVAEncoder(nn.Module):
    """EVA (Exploring the Limits of Masked Visual Representation Learning at Scale)."""
    
    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        import timm
        
        # EVA-CLIP-G/14
        self.backbone = timm.create_model('eva_giant_patch14_224.clip_ft_in1k', pretrained=True)
        feat_dim = 1408
        
        self.proj = nn.Linear(feat_dim, config.output_dim)
        self.patch_proj = nn.Linear(feat_dim, config.output_dim)
        
        if config.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.forward_features(x)
        
        cls_feat = self.proj(x[:, 0])
        patch_feat = self.patch_proj(x[:, 1:])
        
        return cls_feat, patch_feat


# =============================================================================
# R3M Encoder (Robotics-specific)
# =============================================================================

class R3MEncoder(nn.Module):
    """R3M: A Universal Visual Representation for Robot Manipulation."""
    
    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        from r3m import load_r3m
        
        self.backbone = load_r3m("resnet50")
        self.backbone.eval()
        feat_dim = 2048
        
        self.proj = nn.Linear(feat_dim, config.output_dim)
        
        if config.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # R3M expects [0, 255] input
        x_scaled = x * 255.0
        
        with torch.no_grad() if self.config.freeze_backbone else torch.enable_grad():
            features = self.backbone(x_scaled)
        
        cls_feat = self.proj(features)
        # R3M doesn't provide patch features directly
        patch_feat = cls_feat.unsqueeze(1).expand(-1, 49, -1)
        
        return cls_feat, patch_feat


# =============================================================================
# Unified Observation Encoder Factory
# =============================================================================

class ObservationEncoder(nn.Module):
    """Unified observation encoder supporting multiple backbones."""
    
    ENCODER_MAP = {
        "simple_cnn": SimpleCNNEncoder,
        "vit_scratch": ViTScratchEncoder,
        "resnet18": ResNetEncoder,
        "resnet50": ResNetEncoder,
        "dino_vitb16": DINOEncoder,
        "dino_vits16": DINOEncoder,
        "dinov2_vitb14": DINOv2Encoder,
        "dinov2_vitl14": DINOv2Encoder,
        "dinov2_vitg14": DINOv2Encoder,
        "clip_vitb32": CLIPEncoder,
        "clip_vitb16": CLIPEncoder,
        "clip_vitl14": CLIPEncoder,
        "siglip_vitb16": SigLIPEncoder,
        "mae_vitb16": MAEEncoder,
        "eva_vitg14": EVAEncoder,
    }

    def __init__(self, config: ObsEncoderConfig):
        super().__init__()
        self.config = config
        
        encoder_cls = self.ENCODER_MAP.get(config.encoder_type)
        if encoder_cls is None:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")
        
        self.encoder = encoder_cls(config)
        
        # Optional temporal modeling for video sequences
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Image tensor [B, C, H, W] or video tensor [B, T, C, H, W]
        Returns:
            cls_feat: [B, output_dim]
            patch_feat: [B, num_patches, output_dim]
        """
        if x.dim() == 5:  # Video input
            B, T, C, H, W = x.shape
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            cls_feat, patch_feat = self.encoder(x)
            
            # Temporal pooling
            cls_feat = rearrange(cls_feat, '(b t) d -> b d t', b=B, t=T)
            cls_feat = self.temporal_pool(cls_feat).squeeze(-1)
            
            patch_feat = rearrange(patch_feat, '(b t) n d -> b (t n) d', b=B, t=T)
        else:
            cls_feat, patch_feat = self.encoder(x)
        
        return cls_feat, patch_feat

    def get_output_dim(self) -> int:
        return self.config.output_dim

    @classmethod
    def available_encoders(cls) -> List[str]:
        return list(cls.ENCODER_MAP.keys())


# =============================================================================
# Helper: Create encoder with preprocessing
# =============================================================================

def create_obs_encoder(
    encoder_type: str = "dinov2_vitb14",
    output_dim: int = 512,
    freeze_backbone: bool = True,
    finetune_layers: int = 0,
    **kwargs
) -> ObservationEncoder:
    """Factory function to create observation encoder."""
    config = ObsEncoderConfig(
        encoder_type=encoder_type,
        output_dim=output_dim,
        freeze_backbone=freeze_backbone,
        finetune_layers=finetune_layers,
        **kwargs
    )
    return ObservationEncoder(config)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Available encoders:", ObservationEncoder.available_encoders())
    
    # Test simple CNN
    config = ObsEncoderConfig(encoder_type="simple_cnn", output_dim=512)
    encoder = ObservationEncoder(config)
    
    x = torch.randn(2, 3, 224, 224)
    cls_feat, patch_feat = encoder(x)
    print(f"\nSimple CNN:")
    print(f"  Input: {x.shape}")
    print(f"  CLS: {cls_feat.shape}")
    print(f"  Patches: {patch_feat.shape}")
    
    # Test ViT scratch
    config = ObsEncoderConfig(encoder_type="vit_scratch", output_dim=512)
    encoder = ObservationEncoder(config)
    
    cls_feat, patch_feat = encoder(x)
    print(f"\nViT Scratch:")
    print(f"  CLS: {cls_feat.shape}")
    print(f"  Patches: {patch_feat.shape}")
    
    # Count parameters
    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  Params: {total:,} (trainable: {trainable:,})")
