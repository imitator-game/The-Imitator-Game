import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from einops.layers.torch import Rearrange

from dynamics.st_transformer import STTransformerDecoder


class IDM(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        skill_dim: int,
        out_dim: int,
        idm_resolution: int,
        window_size: int = 2,
        patch_size: int = 16,
    ):
        super().__init__()
        patch_height = patch_width = patch_size
        patch_dim = patch_height * patch_width
        self.num_depth_tokens = (idm_resolution // patch_size) ** 2
        self.window_size = window_size
        
        self.depth_proj = nn.Sequential(
            Rearrange("b t (h p1) (w p2) -> b t (h w) (p1 p2)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim, eps=1e-05),
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-05),
        )
        
        resnet = torchvision.models.resnet18(pretrained=True)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        
        # get visual feature dimension
        with torch.no_grad():
            visual_features = self.visual_encoder(torch.randn(1, 3, idm_resolution, idm_resolution))
            visual_channel = visual_features.size(1)
        
        self.modal_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim+visual_channel, eps=1e-05),
            nn.Linear(hidden_dim+visual_channel, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-05),
        )
        
        self.transformer = STTransformerDecoder(
            num_layers=num_layers,
            num_heads=num_heads,
            d_model=hidden_dim,
        )
        self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, 1, self.num_depth_tokens, hidden_dim)
            )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, window_size, 1, hidden_dim)
        )
        
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.skill_norm = nn.LayerNorm(hidden_dim, eps=1e-05)
        self.skill_proj = nn.Linear(hidden_dim, skill_dim)

        self.latent_norm = nn.LayerNorm(skill_dim, eps=1e-05)
        self.latent_proj = nn.Linear(skill_dim, out_dim)
        self.dtype = torch.float16
        
        self.init_weights()
        
    def init_weights(self):
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, depth_outputs, visual_features, return_skill=False):
        B, T, _, _, _ = visual_features.size()
        pos_embed = self.pos_embed_spatial.repeat(
                1, self.window_size, 1, 1
            ) + self.pos_embed_temporal.repeat(
                1, 1, self.num_depth_tokens, 1
            )
        pos_embed = pos_embed.expand(B, -1, -1, -1)
        
        # normalize depth outputs
        depth_min, depth_max = depth_outputs.flatten(2).min(-1).values, depth_outputs.flatten(2).max(-1).values
        depth_outputs = (depth_outputs - depth_min[..., None, None]) / (depth_max[..., None, None] - depth_min[..., None, None])
        
        visual_features = self.forward_encoder(visual_features)
        depth_features = self.depth_proj(depth_outputs)
        
        x = torch.cat([depth_features, visual_features], dim=-1)
        x = self.modal_proj(x)
        
        x = x + pos_embed
        x = self.transformer(x)
        
        x = x[:, [-1]].mean(dim=2)
        x = self.proj(x)
        
        skill = self.skill_norm(x)
        skill = self.skill_proj(skill)
        
        if return_skill:
            return skill

        latent = self.latent_norm(skill)
        latent = self.latent_proj(latent)

        return latent
    
    def forward_encoder(self, x):
        B, T, _, _, _ = x.size()
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.normalize(x)
        
        out = self.visual_encoder(x)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = rearrange(out, "(b t) c h w -> b t (h w) c", b=B)
        
        return out
        