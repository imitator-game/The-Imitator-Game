"""
video_backbone.py — Frozen Mainstream Video/Image Backbone + Lightweight Adapter.
                    Now with optional LoRA fine-tuning of the backbone itself.

Two operation modes
───────────────────
  lora_rank == 0  (default)
      Backbone is 100% frozen. Only the thin adapter layers are trainable.
      Feature cache is active: ~0.15 ms/iter at training time.

  lora_rank > 0   (LoRA mode)
      LoRA adapters are injected into every attention projection of the backbone.
      Only LoRA params + adapter params are trainable; base backbone weights stay
      frozen.  Feature cache is DISABLED (features change every gradient step).
      Requires: pip install peft

  OOM fixes applied (all backbone types)
  ───────────────────────────────────────
  1. Removed output_hidden_states=True from _encode_dino / _encode_clip_hf /
     _encode_siglip — only last_hidden_state is ever used; storing all 24-27
     intermediate hidden states multiplied activation memory ~27×.
  2. Frame-level chunking in all image-backbone encoders (dino / clip / siglip):
     the B×T frames are split into sub-batches of `lora_frame_chunk` images and
     encoded sequentially.  The LoRA gradient path is preserved because the
     per-chunk last_hidden_states are cat-ed before rearrange.
  3. Gradient checkpointing enabled on the backbone when LoRA is active:
     trades ~30 % compute overhead for a large reduction in activation memory
     (only O(sqrt(layers)) activations held at any time).

SUPPORTED BACKBONES
───────────────────
  ┌──────────────────────────┬────────┬────────┬─────────────────────────────┐
  │ key                      │  dim   │ type   │ notes                       │
  ├──────────────────────────┼────────┼────────┼─────────────────────────────┤
  │ "dinov2_vitl14"          │ 1024   │ frame  │ Best frozen spatial quality │
  │ "dinov2_vitb14"          │  768   │ frame  │ Lighter DINOv2              │
  │ "dinov2_vitl14_reg"      │ 1024   │ frame  │ DINOv2 ViT-L with registers │
  │ "clip_vitl14"            │ 1024   │ frame  │ OpenAI CLIP via HuggingFace │
  │ "siglip2_so400m"         │  768   │ frame  │ Google SigLIP, strong V-L   │
  │ "videomae_large"         │ 1024   │ video  │ VideoMAE-large, temporal    │
  │ "videomae_base"          │  768   │ video  │ VideoMAE-base, lighter      │
  └──────────────────────────┴────────┴────────┴─────────────────────────────┘
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ── Backbone metadata ──────────────────────────────────────────────────────────

_BACKBONE_META: Dict[str, dict] = {
    "dinov2_vitb14": {
        "hf_model":      "facebook/dinov2-base",
        "dim":           768,
        "patch_size":    14,
        "type":          "dino",
        "has_registers": False,
    },
    "dinov2_vitl14": {
        "hf_model":      "facebook/dinov2-large",
        "dim":           1024,
        "patch_size":    14,
        "type":          "dino",
        "has_registers": False,
    },
    "dinov2_vitl14_reg": {
        "hf_model":      "facebook/dinov2-with-registers-large",
        "dim":           1024,
        "patch_size":    14,
        "type":          "dino",
        "has_registers": True,
    },
    "clip_vitl14": {
        "hf_model":   "openai/clip-vit-large-patch14",
        "dim":        1024,
        "patch_size": 14,
        "type":       "clip_hf",
    },
    "clip_vitb16": {
        "hf_model":   "openai/clip-vit-base-patch16",
        "dim":        768,
        "patch_size": 16,
        "type":       "clip_hf",
    },
    "siglip_so400m": {
        "hf_model":   "google/siglip2-base-patch16-224",
        "dim":        768,
        "patch_size": 16,
        "type":       "siglip",
        "image_size": 224,
    },
    "siglip2_so400m": {
        "hf_model":   "google/siglip2-base-patch16-224",
        "dim":        768,
        "patch_size": 16,
        "type":       "siglip",
        "image_size": 224,
    },
    "videomae_base": {
        "hf_model":   "MCG-NJU/videomae-base",
        "dim":        768,
        "type":       "videomae",
    },
    "videomae_large": {
        "hf_model":   "MCG-NJU/videomae-large",
        "dim":        1024,
        "type":       "videomae",
    },
}

# LoRA target module names by backbone type.
# These match the attention projection Linear layer names in HuggingFace
# implementations of each architecture.
_LORA_TARGET_MODULES: Dict[str, List[str]] = {
    # DINOv2: Dinov2SelfAttention has .query / .key / .value
    "dino":     ["query", "key", "value"],
    # CLIP: CLIPAttention has .q_proj / .k_proj / .v_proj
    "clip_hf":  ["q_proj", "k_proj", "v_proj"],
    # SigLIP: SiglipAttention has .q_proj / .k_proj / .v_proj
    "siglip":   ["q_proj", "k_proj", "v_proj"],
    # VideoMAE: VideoMAESelfAttention has .query / .key / .value
    "videomae": ["query", "key", "value"],
}


# ── Sinusoidal positional encoding ─────────────────────────────────────────────

class _SinPE(nn.Module):
    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        pe  = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ═══════════════════════════════════════════════════════════════════════════════
# FrozenVideoBackbone
# ═══════════════════════════════════════════════════════════════════════════════

class FrozenVideoBackbone(nn.Module):
    """Frozen/LoRA-tunable mainstream backbone + minimal trainable adapter.

    Parameters
    ──────────
    backbone_type : str
        One of the keys in _BACKBONE_META.
    latent_dim : int
        Target output dimension.
    lora_rank : int
        LoRA rank. 0 = fully frozen backbone (default).
        > 0 = inject LoRA into all attention projection layers.
        Requires `pip install peft` when > 0.
    lora_alpha : float
        LoRA alpha scaling factor. Effective scale = lora_alpha / lora_rank.
    lora_dropout : float
        Dropout applied to LoRA inputs (0 = no dropout).
    lora_frame_chunk : int
        Maximum number of frames (B×T images) to feed through the backbone in
        a single call when LoRA is active.  Larger = faster but more VRAM.
        Has no effect in frozen mode (cache is used).  Default: 64.
    lora_gradient_checkpointing : bool
        When True and LoRA is active, enable gradient checkpointing on the
        backbone to trade compute for activation memory.  Default: True.
    """

    _DEFAULT_DIM = 1024

    def __init__(
        self,
        backbone_type:               str            = "dinov2_vitl14",
        latent_dim:                  int            = 256,
        max_seq_patches:             int            = 32,
        adapter_layers:              int            = 1,
        adapter_add_layernorm:       bool           = True,
        num_sampled_frames:          int            = 10,
        hf_cache_dir:                Optional[str]  = None,
        image_size:                  int            = 224,
        # ── LoRA parameters ────────────────────────────────────────────────
        lora_rank:                   int            = 0,
        lora_alpha:                  float          = 16.0,
        lora_dropout:                float          = 0.0,
        # ── OOM-reduction parameters ───────────────────────────────────────
        lora_frame_chunk:            int            = 64,
        lora_gradient_checkpointing: bool           = True,
    ):
        super().__init__()

        if backbone_type not in _BACKBONE_META:
            raise ValueError(
                f"Unknown backbone_type '{backbone_type}'. "
                f"Choose one of: {list(_BACKBONE_META)}"
            )

        self.backbone_type               = backbone_type
        self.latent_dim                  = latent_dim
        self.max_seq_patches             = max_seq_patches
        self.adapter_layers              = adapter_layers
        self.adapter_add_layernorm       = adapter_add_layernorm
        self.num_sampled_frames          = num_sampled_frames
        self.hf_cache_dir                = hf_cache_dir
        self.image_size                  = image_size

        # ── LoRA config ────────────────────────────────────────────────────
        self._lora_rank                    = lora_rank
        self._lora_alpha                   = lora_alpha
        self._lora_dropout                 = lora_dropout
        self._lora_enabled                 = lora_rank > 0
        self._lora_frame_chunk             = lora_frame_chunk
        self._lora_gradient_checkpointing  = lora_gradient_checkpointing

        self._meta    = _BACKBONE_META[backbone_type]
        self._type    = self._meta["type"]
        self._loaded  = False
        self._backbone = None

        self._build_adapter(self._meta.get("dim", self._DEFAULT_DIM))
        self._adapter_final_dim = self._meta.get("dim", self._DEFAULT_DIM)

        self._pe = _SinPE(latent_dim, max_len=512)

    # ── Adapter construction ─────────────────────────────────────────────────

    def _build_adapter(self, backbone_dim: int) -> None:
        d_in  = backbone_dim
        d_out = self.latent_dim

        if self.adapter_layers == 1:
            layers_cls = [nn.Linear(d_in, d_out)]
            layers_seq = [nn.Linear(d_in, d_out)]
            if self.adapter_add_layernorm:
                layers_cls.append(nn.LayerNorm(d_out))
                layers_seq.append(nn.LayerNorm(d_out))
        else:
            mid = max(d_out, (d_in + d_out) // 2)
            layers_cls = [
                nn.Linear(d_in, mid), nn.LayerNorm(mid), nn.GELU(),
                nn.Linear(mid, d_out),
            ]
            layers_seq = [
                nn.Linear(d_in, mid), nn.LayerNorm(mid), nn.GELU(),
                nn.Linear(mid, d_out),
            ]

        self.cls_adapter = nn.Sequential(*layers_cls)
        self.seq_adapter = nn.Sequential(*layers_seq)

    # ── LoRA injection ───────────────────────────────────────────────────────

    def _inject_lora(self) -> None:
        """Inject LoRA adapters into the backbone's attention projection layers.

        Uses the `peft` library. Only LoRA parameters are made trainable;
        base backbone weights remain frozen.

        Also enables gradient checkpointing on the backbone when
        ``lora_gradient_checkpointing=True`` (default) to reduce activation
        memory at the cost of ~30 % extra compute.

        Called automatically inside each _load_*() method when lora_rank > 0.
        """
        if not self._lora_enabled or self._backbone is None:
            return

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "The `peft` library is required for LoRA fine-tuning of the video encoder.\n"
                "Install it with:  pip install peft"
            )

        target_modules = _LORA_TARGET_MODULES.get(self._type)
        if target_modules is None:
            raise ValueError(
                f"LoRA is not configured for backbone type '{self._type}'. "
                f"Add target module names to _LORA_TARGET_MODULES in video_backbone.py."
            )

        lora_config = LoraConfig(
            r=self._lora_rank,
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Wrap the backbone with LoRA. get_peft_model() marks LoRA params as
        # trainable and all base params as frozen.
        self._backbone = get_peft_model(self._backbone, lora_config)

        # Double-check: ensure only LoRA params require grad (base weights stay frozen).
        for name, param in self._backbone.named_parameters():
            if "lora_" not in name:
                param.requires_grad_(False)

        # ── Gradient checkpointing ─────────────────────────────────────────
        # Replaces storing all layer activations with recomputation on backward.
        # Cuts activation memory by ~sqrt(num_layers) at the cost of one extra
        # forward pass per backward.  Critical for LoRA where the full backbone
        # forward runs with grad enabled.
        if self._lora_gradient_checkpointing:
            try:
                self._backbone.enable_input_require_grads()
                self._backbone.gradient_checkpointing_enable()
                print("    Gradient checkpointing: ENABLED (saves activation memory)")
            except Exception as e:
                print(f"    Gradient checkpointing: could not enable — {e}")

        n_lora = sum(p.numel() for p in self._backbone.parameters()
                     if p.requires_grad) / 1e6
        n_base = sum(p.numel() for p in self._backbone.parameters()
                     if not p.requires_grad) / 1e6
        print(
            f"  ✓ LoRA injected into {self.backbone_type}: "
            f"rank={self._lora_rank}, alpha={self._lora_alpha}, "
            f"trainable LoRA: {n_lora:.2f}M / frozen base: {n_base:.1f}M params\n"
            f"    Target modules: {target_modules}\n"
            f"    Frame chunk size (LoRA): {self._lora_frame_chunk}"
        )

    # ── Lazy backbone loading ────────────────────────────────────────────────

    def _load_backbone(self) -> None:
        if self._loaded:
            return
        loader = {
            "dino":      self._load_dino,
            "clip_hf":   self._load_clip_hf,
            "siglip":    self._load_siglip,
            "videomae":  self._load_videomae,
        }[self._type]
        loader()
        self._loaded = True

    def _load_dino(self) -> None:
        from transformers import AutoModel, AutoImageProcessor
        meta = self._meta
        print(f"  [FrozenVideoBackbone] Loading {meta['hf_model']} (DINOv2 via HF) ...")
        kw = {}
        if self.hf_cache_dir:
            kw["cache_dir"] = self.hf_cache_dir
        model = AutoModel.from_pretrained(meta["hf_model"], **kw)
        proc  = AutoImageProcessor.from_pretrained(meta["hf_model"], **kw)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._backbone  = model
        self._processor = proc

        true_dim = model.config.hidden_size
        if true_dim != self._adapter_final_dim:
            dev = self.cls_adapter[0].weight.device
            self._build_adapter(true_dim)
            self.cls_adapter = self.cls_adapter.to(dev)
            self.seq_adapter = self.seq_adapter.to(dev)
            self._adapter_final_dim = true_dim

        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓ DINOv2 {meta['hf_model']}: {n:.0f}M params loaded")

        # Inject LoRA after base model is ready
        self._inject_lora()

    def _load_clip_hf(self) -> None:
        from transformers import CLIPVisionModel, CLIPImageProcessor
        meta = self._meta
        print(f"  [FrozenVideoBackbone] Loading {meta['hf_model']} (CLIP vision) ...")
        kw = {}
        if self.hf_cache_dir:
            kw["cache_dir"] = self.hf_cache_dir
        model = CLIPVisionModel.from_pretrained(meta["hf_model"], **kw)
        proc  = CLIPImageProcessor.from_pretrained(meta["hf_model"], **kw)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._backbone  = model
        self._processor = proc

        true_dim = model.config.hidden_size
        if true_dim != self._adapter_final_dim:
            dev = self.cls_adapter[0].weight.device
            self._build_adapter(true_dim)
            self.cls_adapter = self.cls_adapter.to(dev)
            self.seq_adapter = self.seq_adapter.to(dev)
            self._adapter_final_dim = true_dim

        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓ CLIP {meta['hf_model']}: {n:.0f}M params loaded")
        self._inject_lora()

    def _load_siglip(self) -> None:
        from transformers import SiglipVisionModel, SiglipImageProcessor
        meta = self._meta
        print(f"  [FrozenVideoBackbone] Loading {meta['hf_model']} (SigLIP vision) ...")
        kw = {}
        if self.hf_cache_dir:
            kw["cache_dir"] = self.hf_cache_dir
        model = SiglipVisionModel.from_pretrained(meta["hf_model"], **kw)
        proc  = SiglipImageProcessor.from_pretrained(meta["hf_model"], **kw)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._backbone  = model
        self._processor = proc

        true_dim = model.config.hidden_size
        if true_dim != self._adapter_final_dim:
            dev = self.cls_adapter[0].weight.device
            self._build_adapter(true_dim)
            self.cls_adapter = self.cls_adapter.to(dev)
            self.seq_adapter = self.seq_adapter.to(dev)
            self._adapter_final_dim = true_dim

        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓ SigLIP {meta['hf_model']}: {n:.0f}M params loaded")
        self._inject_lora()

    def _load_videomae(self) -> None:
        from transformers import VideoMAEModel, VideoMAEImageProcessor
        meta = self._meta
        print(f"  [FrozenVideoBackbone] Loading {meta['hf_model']} (VideoMAE) ...")
        kw = {}
        if self.hf_cache_dir:
            kw["cache_dir"] = self.hf_cache_dir
        model = VideoMAEModel.from_pretrained(meta["hf_model"], **kw)
        proc  = VideoMAEImageProcessor.from_pretrained(meta["hf_model"], **kw)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._backbone  = model
        self._processor = proc

        true_dim = model.config.hidden_size
        if true_dim != self._adapter_final_dim:
            dev = self.cls_adapter[0].weight.device
            self._build_adapter(true_dim)
            self.cls_adapter = self.cls_adapter.to(dev)
            self.seq_adapter = self.seq_adapter.to(dev)
            self._adapter_final_dim = true_dim

        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓ VideoMAE {meta['hf_model']}: {n:.0f}M params loaded")
        self._inject_lora()

    # ── Video preprocessing helpers ──────────────────────────────────────────

    @staticmethod
    def _to_bthwc_uint8(video: torch.Tensor) -> torch.Tensor:
        if video is None:
            raise TypeError(
                "_to_bthwc_uint8: received video=None.\n"
                "  The backbone was called without an actual video tensor.\n"
                "  If caching is enabled, this indicates a cache miss — "
                "check that sample_id keys in the DataLoader match those in the cache."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.shape[-1] not in (3, 4):
            video = video.permute(0, 1, 3, 4, 2)
        video = video[..., :3]
        if video.dtype != torch.uint8:
            video = (video.float().clamp(0, 1) * 255).to(torch.uint8)
        return video

    def _sample_frames(self, video_bthwc: torch.Tensor) -> torch.Tensor:
        T = video_bthwc.shape[1]
        n = min(self.num_sampled_frames, T)
        idx = torch.linspace(0, T - 1, n).long()
        return video_bthwc[:, idx]

    # ── Helper: chunked backbone forward for image-based models ─────────────
    #
    #  When LoRA is active the entire backbone forward runs under enable_grad,
    #  which means all intermediate activations are retained for backprop.
    #  Processing B*T frames at once (e.g. 256*10 = 2560) can easily OOM.
    #  This helper splits pixel_values into sub-batches of `chunk` frames,
    #  runs the backbone on each sub-batch, and concatenates the results.
    #  The LoRA gradient path is fully preserved through torch.cat.
    #
    #  In frozen mode (lora_rank == 0) chunking is NOT applied because the
    #  whole forward runs under no_grad and memory pressure is much lower.

    def _chunked_backbone_forward(
        self,
        pixel_values: torch.Tensor,
        *,
        call_fn,          # callable(pixel_values_chunk) -> last_hidden_state
    ) -> torch.Tensor:
        """Run backbone in chunks; return cat-ed last_hidden_state (N, P, D)."""
        if not self._lora_enabled:
            # Frozen mode: single call, no chunking needed.
            return call_fn(pixel_values)

        chunk = self._lora_frame_chunk
        N = pixel_values.shape[0]
        if N <= chunk:
            return call_fn(pixel_values)

        parts = []
        for i in range(0, N, chunk):
            parts.append(call_fn(pixel_values[i : i + chunk]))
        return torch.cat(parts, dim=0)

    # ── Per-type encoding ────────────────────────────────────────────────────
    # NOTE: no @torch.no_grad() here — gradient context is managed in forward().
    #
    # OOM FIX #1: output_hidden_states=True removed from all _encode_* methods.
    #   The original code passed output_hidden_states=True to the backbone but
    #   only ever used out.last_hidden_state.  Storing all 24-27 intermediate
    #   hidden states multiplied activation memory by ~27× for no benefit.
    #
    # OOM FIX #2: frame chunking via _chunked_backbone_forward().
    #   With batch_size=256 and num_sampled_frames=10 the original code sent
    #   2560 images through the backbone in a single call with grad enabled.
    #   Now they are split into sub-batches of `lora_frame_chunk` (default 64).

    def _encode_dino(
        self, video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_bthwc = self._to_bthwc_uint8(video)
        frames      = self._sample_frames(video_bthwc)
        B, n, H, W, C = frames.shape

        dev     = next(self._backbone.parameters()).device
        flat_np = frames.reshape(B * n, H, W, C).cpu().numpy()

        inputs = self._processor(images=list(flat_np), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(dev)

        # FIX #1: removed output_hidden_states=True (only last_hidden_state used)
        # FIX #2: chunked forward to avoid OOM with large B*n
        def _call(pv):
            with torch.amp.autocast("cuda", enabled=pv.is_cuda):
                return self._backbone(pixel_values=pv).last_hidden_state.float()

        last = self._chunked_backbone_forward(pixel_values, call_fn=_call)

        cls_tok   = last[:, 0,  :]
        patch_tok = last[:, 1:, :]

        cls_tok   = rearrange(cls_tok,   '(b t) d   -> b t d',   b=B, t=n)
        patch_tok = rearrange(patch_tok, '(b t) p d -> b t p d', b=B, t=n)

        raw_cls = cls_tok.mean(dim=1)
        all_patches = rearrange(patch_tok, 'b t p d -> b (t p) d')
        cap    = min(self.max_seq_patches, all_patches.shape[1])
        stride = max(1, all_patches.shape[1] // cap)
        raw_seq = all_patches[:, ::stride, :][:, :cap, :]

        return raw_cls, raw_seq

    def _encode_clip_hf(
        self, video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_bthwc = self._to_bthwc_uint8(video)
        frames      = self._sample_frames(video_bthwc)
        B, n, H, W, C = frames.shape

        dev = next(self._backbone.parameters()).device
        flat_np = frames.reshape(B * n, H, W, C).cpu().numpy()

        inputs = self._processor(images=list(flat_np), return_tensors="pt", do_rescale=True)
        pixel_values = inputs["pixel_values"].to(dev)

        # FIX #1: removed output_hidden_states=True (only last_hidden_state used)
        # FIX #2: chunked forward
        def _call(pv):
            with torch.amp.autocast("cuda", enabled=pv.is_cuda):
                return self._backbone(pixel_values=pv).last_hidden_state.float()

        last = self._chunked_backbone_forward(pixel_values, call_fn=_call)

        cls_tok   = last[:, 0, :]
        patch_tok = last[:, 1:, :]

        cls_tok   = rearrange(cls_tok,   '(b t) d   -> b t d',   b=B, t=n)
        patch_tok = rearrange(patch_tok, '(b t) p d -> b t p d', b=B, t=n)

        raw_cls = cls_tok.mean(dim=1)
        all_patches = rearrange(patch_tok, 'b t p d -> b (t p) d')
        cap    = min(self.max_seq_patches, all_patches.shape[1])
        stride = max(1, all_patches.shape[1] // cap)
        raw_seq = all_patches[:, ::stride, :][:, :cap, :]

        return raw_cls, raw_seq

    def _encode_siglip(
        self, video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_bthwc = self._to_bthwc_uint8(video)
        frames      = self._sample_frames(video_bthwc)
        B, n, H, W, C = frames.shape

        dev = next(self._backbone.parameters()).device
        flat_np = frames.reshape(B * n, H, W, C).cpu().numpy()

        inputs = self._processor(images=list(flat_np), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(dev)

        # FIX #1: removed output_hidden_states=True (only last_hidden_state used)
        # FIX #2: chunked forward
        def _call(pv):
            with torch.amp.autocast("cuda", enabled=pv.is_cuda):
                # NOTE: SigLIP has no dedicated CLS token; mean-pool over patches
                return self._backbone(pixel_values=pv).last_hidden_state.float()

        last = self._chunked_backbone_forward(pixel_values, call_fn=_call)

        # SigLIP: no CLS token — use mean of all patch tokens as CLS surrogate
        cls_tok   = last.mean(dim=1)
        patch_tok = last

        cls_tok   = rearrange(cls_tok,   '(b t) d   -> b t d',   b=B, t=n)
        patch_tok = rearrange(patch_tok, '(b t) p d -> b t p d', b=B, t=n)

        raw_cls = cls_tok.mean(dim=1)
        all_patches = rearrange(patch_tok, 'b t p d -> b (t p) d')
        cap    = min(self.max_seq_patches, all_patches.shape[1])
        stride = max(1, all_patches.shape[1] // cap)
        raw_seq = all_patches[:, ::stride, :][:, :cap, :]

        return raw_cls, raw_seq

    def _encode_videomae(
        self, video: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # VideoMAE takes the full video clip (B, T, C, H, W) — no per-frame
        # chunking.  B is the only dimension we could chunk over here, but
        # VideoMAE already uses far fewer tokens than image models (1568 vs
        # ~196*10=1960), and its sequence length is fixed.  If OOM still occurs
        # with VideoMAE+LoRA, reduce batch_size instead.
        video_bthwc = self._to_bthwc_uint8(video)
        B, T, H, W, C = video_bthwc.shape

        flat = rearrange(video_bthwc, 'b t h w c -> (b t) c h w').float() / 255.0
        if H != 224 or W != 224:
            flat = F.interpolate(flat, (224, 224), mode='bilinear', align_corners=False)
        frames_224 = rearrange(flat, '(b t) c h w -> b t c h w', b=B)

        target_T = 16
        if T < target_T:
            repeats = (target_T + T - 1) // T
            frames_224 = frames_224.repeat(1, repeats, 1, 1, 1)[:, :target_T]
        elif T > target_T:
            idx = torch.linspace(0, T - 1, target_T).long()
            frames_224 = frames_224[:, idx]

        dev = next(self._backbone.parameters()).device
        flat_np = (frames_224.permute(0, 1, 3, 4, 2).cpu().numpy() * 255).astype("uint8")

        inputs = self._processor(
            list([list(flat_np[i]) for i in range(B)]),
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(dev)

        # VideoMAE processes whole video clips; single call (no per-frame chunking).
        # output_hidden_states is NOT set here (was already correct in original).
        with torch.amp.autocast("cuda", enabled=pixel_values.is_cuda):
            out = self._backbone(pixel_values=pixel_values)

        seq_out = out.last_hidden_state.float()
        raw_cls = seq_out.mean(dim=1)

        cap    = min(self.max_seq_patches, seq_out.shape[1])
        stride = max(1, seq_out.shape[1] // cap)
        raw_seq = seq_out[:, ::stride, :][:, :cap, :]

        return raw_cls, raw_seq

    # ── Main forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        video: torch.Tensor,
        sample_ids=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode video through backbone + trainable adapter.

        Gradient context
        ─────────────────
        lora_rank == 0  (frozen mode):  backbone runs under torch.no_grad()
                                        for efficiency. Adapter receives grads.
        lora_rank > 0   (LoRA mode):    backbone runs with gradients enabled so
                                        LoRA parameter updates are computed.
                                        Adapter also receives grads.
                                        Frames are processed in chunks of
                                        `lora_frame_chunk` to limit activation
                                        memory.

        Parameters
        ──────────
        video : (B, T, H, W, C) float [0,1] or uint8 — or (B, T, C, H, W)

        Returns
        ───────
        cls : (B, latent_dim)
        seq : (B, N, latent_dim)
        """
        self._load_backbone()

        # Move backbone to adapter device if needed
        if self._backbone is not None:
            enc_dev = self.cls_adapter[0].weight.device
            try:
                bb_dev = next(self._backbone.parameters()).device
                if bb_dev != enc_dev:
                    self._backbone = self._backbone.to(enc_dev)
            except StopIteration:
                pass

        dispatch = {
            "dino":     self._encode_dino,
            "clip_hf":  self._encode_clip_hf,
            "siglip":   self._encode_siglip,
            "videomae": self._encode_videomae,
        }

        # ── Gradient context: no_grad for frozen, enable_grad for LoRA ────────
        if self._lora_enabled:
            # LoRA mode: LoRA params need gradients → run under enable_grad
            # (we may be inside an outer torch.no_grad() during eval, so be explicit)
            grad_ctx = torch.enable_grad()
        else:
            # Frozen mode: save memory and time
            grad_ctx = torch.no_grad()

        with grad_ctx:
            raw_cls, raw_seq = dispatch[self._type](video)

        # Adapter projection (always trainable)
        dev = self.cls_adapter[0].weight.device
        raw_cls = raw_cls.to(dev)
        raw_seq = raw_seq.to(dev)

        cls = self.cls_adapter(raw_cls)
        seq = self.seq_adapter(raw_seq)
        seq = self._pe(seq)

        return cls, seq

    def encode(
        self,
        human_video=None,
        human_desc=None,
        robot_first_frame=None,
        human_states=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if human_video is None:
            raise ValueError(
                "FrozenVideoBackbone.encode() requires human_video. "
                "Set input_mode='video_only' when using frozen_backbone."
            )
        cls, seq = self.forward(human_video)
        return {"z": cls, "z_seq": seq}

    # ── Trainable parameters ─────────────────────────────────────────────────

    def trainable_params(self) -> List[nn.Parameter]:
        """Adapter params + optional LoRA params (when lora_rank > 0)."""
        params = (
            list(self.cls_adapter.parameters()) +
            list(self.seq_adapter.parameters()) +
            list(self._pe.parameters())
        )
        if self._lora_enabled and self._backbone is not None:
            lora_params = [p for p in self._backbone.parameters() if p.requires_grad]
            params.extend(lora_params)
        return params

    def print_info(self) -> None:
        total_bb = 0
        total_lora = 0
        if self._backbone is not None and hasattr(self._backbone, 'parameters'):
            for p in self._backbone.parameters():
                if p.requires_grad:
                    total_lora += p.numel()
                else:
                    total_bb += p.numel()
        total_ad = (
            sum(p.numel() for p in self.cls_adapter.parameters()) +
            sum(p.numel() for p in self.seq_adapter.parameters()) +
            sum(p.numel() for p in self._pe.parameters())
        )
        mode = f"LoRA (rank={self._lora_rank}, alpha={self._lora_alpha})" \
               if self._lora_enabled else "Frozen"
        print(f"  FrozenVideoBackbone ({self.backbone_type})")
        print(f"    Backbone base:  {total_bb / 1e6:.1f}M params (frozen)")
        if self._lora_enabled:
            print(f"    Backbone LoRA:  {total_lora / 1e3:.1f}K params (trainable)")
            print(f"    Frame chunk:    {self._lora_frame_chunk} images/call")
            print(f"    Grad ckpt:      {'yes' if self._lora_gradient_checkpointing else 'no'}")
        print(f"    Adapter:        {total_ad / 1e3:.1f}K params (trainable)")
        print(f"    Mode:           {mode}")

    @property
    def lora_enabled(self) -> bool:
        return self._lora_enabled


# ── Convenience factory ───────────────────────────────────────────────────────

def build_video_backbone(
    backbone_type:               str,
    latent_dim:                  int,
    max_seq_patches:             int           = 32,
    adapter_layers:              int           = 1,
    hf_cache_dir:                Optional[str] = None,
    num_sampled_frames:          int           = 10,
    # LoRA parameters
    lora_rank:                   int           = 0,
    lora_alpha:                  float         = 16.0,
    lora_dropout:                float         = 0.0,
    # OOM-reduction parameters
    lora_frame_chunk:            int           = 64,
    lora_gradient_checkpointing: bool          = True,
) -> FrozenVideoBackbone:
    """Build and return a FrozenVideoBackbone (with optional LoRA).

    Does NOT trigger lazy loading — the backbone is loaded on first forward().
    Call .to(device) after this function before training.

    Notes
    ─────
    When lora_rank > 0:
      - `peft` library must be installed: pip install peft
      - Feature caching is NOT compatible with LoRA (features change each step).
        The caller (train_*_imitator.py) is responsible for skipping cache setup.
      - LoRA parameters + adapter parameters are trainable.
      - Frame chunking (lora_frame_chunk=64) and gradient checkpointing are
        enabled by default to avoid OOM with large training batches.
        Tune lora_frame_chunk up/down to trade speed vs VRAM.

    OOM tips
    ────────
      Reduce lora_frame_chunk (e.g. 32 or 16) if still OOM.
      Set lora_gradient_checkpointing=False only if you have abundant VRAM and
      want to maximise throughput.
    """
    return FrozenVideoBackbone(
        backbone_type=backbone_type,
        latent_dim=latent_dim,
        max_seq_patches=max_seq_patches,
        adapter_layers=adapter_layers,
        num_sampled_frames=num_sampled_frames,
        hf_cache_dir=hf_cache_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_frame_chunk=lora_frame_chunk,
        lora_gradient_checkpointing=lora_gradient_checkpointing,
    )