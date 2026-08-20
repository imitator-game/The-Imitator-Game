"""
lerobot_dataset.py  — OpenVLA adapter for HumanSimPairedDataset  (V2)

V2 Cache Changes vs V1
-----------------------
1. Sharded format support
   Reads from  <cache_dir>/shards/<id>/feat.bin  files produced by
   precompute_vla_features.py V2.  Each shard is 1–2 GB, so the OS
   page-cache works effectively on random reads without the whole
   dataset fitting in RAM (vs a single ~200 GB monolithic file in V1).

2. Worker-local lazy memmap initialization
   Memmap handles are NOT opened in __init__.  They are opened the first
   time each DataLoader worker calls __getitem__.  This avoids:
     - Forked-file-descriptor contention across 24+ workers reading the
       same underlying pages in the same kernel inode.
     - The parent process holding open all shards unnecessarily.
   Each worker gets its own fd set and its own page-fault pattern, which
   is more cache-friendly for the OS.

3. Backward-compatible monolithic fallback
   If  vla_cache_meta.json  contains  "format": "monolithic"  (V1 caches),
   the old single-file path is used.  No migration needed.

4. Mean-pooled feature support  (--pool_features from precompute)
   If the cache stores (N, llm_dim) instead of (N, num_tokens, llm_dim),
   _getitem_cached returns (1, llm_dim) so the action head always sees a
   3-D tensor (B, seq, llm_dim) without any changes to train_openvla.py.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset

from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    PairedDatasetConfig,
    HumanSimPairedDataset,
)
from examples.baselines.openvla_oft.prismatic.models.backbones.llm.prompting import PurePromptBuilder
from examples.baselines.openvla_oft.prismatic.vla.action_tokenizer import ActionTokenizer
from examples.baselines.openvla_oft.prismatic.vla.constants import IGNORE_INDEX


# ---------------------------------------------------------------------------
# Sharded cache descriptor  (populated once by the first worker that reads it)
# ---------------------------------------------------------------------------

class _ShardedCacheDescriptor:
    """
    Holds the logical layout of a sharded feature cache.
    Created once in __init__ (cheap — just reads JSON files).
    Each DataLoader worker opens its own np.memmap handles lazily.
    """

    def __init__(self, cache_dir: str, pred_horizon: int, action_dim: int):
        self.cache_dir   = cache_dir
        self.pred_horizon = pred_horizon
        self.action_dim  = action_dim

        top_meta = _read_json(os.path.join(cache_dir, "vla_cache_meta.json"))
        _assert_completed(top_meta, cache_dir)

        self.total_samples = top_meta["total_samples"]
        self.feat_shape    = tuple(top_meta["feat_shape"])   # (num_tokens, llm_dim) or (llm_dim,)
        self.pool_features = bool(top_meta.get("pool_features", False))
        self.llm_dim       = top_meta["llm_dim"]
        self.shard_size    = top_meta["shard_size"]
        self.num_shards    = top_meta["num_shards"]
        self.has_proprio   = bool(top_meta.get("has_proprio", False))

        _validate_dims(top_meta, pred_horizon, action_dim, self.total_samples,
                       len_dataset=None)  # dataset-len check done separately

        # Collect per-shard metadata (start, end, shard_n)
        self.shards: List[Dict] = []
        for si in range(self.num_shards):
            sm_path = os.path.join(cache_dir, "shards", f"{si:05d}", "shard_meta.json")
            sm = _read_json(sm_path)
            if not sm.get("completed", False):
                raise RuntimeError(
                    f"Shard {si:05d} in {cache_dir} is not marked complete. "
                    "Re-run precompute_vla_features.py."
                )
            self.shards.append({
                "shard_dir": os.path.join(cache_dir, "shards", f"{si:05d}"),
                "start":   sm["start"],
                "end":     sm["end"],
                "shard_n": sm["shard_n"],
            })

        print(
            f"[Dataset] Sharded VLA cache: {self.total_samples:,} samples, "
            f"{self.num_shards} shards, feat_shape={self.feat_shape}, "
            f"pool={self.pool_features}, proprio={'yes' if self.has_proprio else 'no'}"
        )

    def idx_to_shard(self, idx: int) -> Tuple[int, int]:
        """Return (shard_idx, within-shard offset) for global sample index."""
        si = idx // self.shard_size
        # Guard against edge cases at last shard
        si = min(si, self.num_shards - 1)
        offset = idx - self.shards[si]["start"]
        return si, offset


class _MonolithicCacheDescriptor:
    """Descriptor for legacy single-file caches produced by precompute V1."""

    def __init__(self, cache_dir: str, pred_horizon: int, action_dim: int):
        self.cache_dir    = cache_dir
        self.pred_horizon = pred_horizon
        self.action_dim   = action_dim

        top_meta = _read_json(os.path.join(cache_dir, "vla_cache_meta.json"))
        _assert_completed(top_meta, cache_dir)

        self.total_samples = top_meta["total_samples"]
        self.num_tokens    = top_meta["num_tokens"]
        self.llm_dim       = top_meta["llm_dim"]
        self.pool_features = bool(top_meta.get("pool_features", False))
        self.has_proprio   = bool(top_meta.get("has_proprio", False))

        if self.pool_features:
            self.feat_shape = (self.llm_dim,)
        else:
            self.feat_shape = (self.num_tokens, self.llm_dim)

        _validate_dims(top_meta, pred_horizon, action_dim, self.total_samples,
                       len_dataset=None)

        print(
            f"[Dataset] Monolithic VLA cache (V1): {self.total_samples:,} samples, "
            f"feat_shape={self.feat_shape}, proprio={'yes' if self.has_proprio else 'no'}"
        )


# ---------------------------------------------------------------------------
# Worker-local memmap store
# ---------------------------------------------------------------------------
# Each worker process keeps its own dict mapping shard_idx → opened np.memmap.
# With DataLoader forkserver/fork, each worker process has its own copy of this
# dict (initially empty), so there is no cross-worker contention.

class _WorkerMmapStore:
    """
    Per-process lazy memmap store.  Lives on the dataset instance.
    On fork/forkserver, each worker starts with _opened={}, opening
    handles on first access.  The parent process also has its own instance
    (used only for __len__, not __getitem__).
    """
    __slots__ = ("_opened",)

    def __init__(self):
        self._opened: Dict[str, np.memmap] = {}   # key: file path

    def get(self, path: str, dtype: str, shape: tuple) -> np.memmap:
        if path not in self._opened:
            self._opened[path] = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        return self._opened[path]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _assert_completed(meta: dict, cache_dir: str) -> None:
    if not meta.get("completed", False):
        raise RuntimeError(
            f"VLA feature cache in {cache_dir} is incomplete "
            f"(processed {meta.get('processed_count', 0)} / "
            f"{meta.get('total_samples', '?')} samples). "
            "Re-run precompute_vla_features.py to finish it."
        )


def _validate_dims(
    meta: dict,
    pred_horizon: int,
    action_dim: int,
    total_samples: int,
    len_dataset: Optional[int],
) -> None:
    cached_ph = meta.get("pred_horizon")
    cached_ad = meta.get("action_dim")
    if cached_ph is not None and cached_ph != pred_horizon:
        raise ValueError(
            f"Cache pred_horizon={cached_ph} != dataset pred_horizon={pred_horizon}. "
            "Rebuild the cache with matching --pred_horizon."
        )
    if cached_ad is not None and cached_ad != action_dim:
        raise ValueError(
            f"Cache action_dim={cached_ad} != dataset action_dim={action_dim}. "
            "Rebuild the cache with matching --action_dim."
        )
    if len_dataset is not None and len_dataset != total_samples:
        raise ValueError(
            f"Cache total_samples={total_samples:,} does not match "
            f"current dataset size={len_dataset:,}. "
            "The dataset files may have changed since the cache was built. "
            "Rebuild the cache with the same dataset config."
        )


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------

class OpenVLAPairedDataset(Dataset):
    """Adapter from HumanSimPairedDataset (vla=True) to OpenVLA training batches.

    When ``vla_cache_dir`` is provided, the dataset switches to cache mode:
    - ``pixel_values``, ``input_ids``, ``labels`` are NOT returned.
    - ``vla_feat``  is returned: float16 tensor shape (num_tokens, llm_dim)
      or (1, llm_dim) if the cache was built with --pool_features.
    - ``actions`` and optionally ``proprio`` are returned as usual.

    Cache formats supported
    -----------------------
    - V2 sharded  (``"format": "sharded"`` in meta):  produced by precompute V2.
      Small per-shard files; fast random reads; worker-local lazy opens.
    - V1 monolithic (``"format": "monolithic"`` or key absent):  legacy single
      file.  Still works but random-read IO may be slow for large datasets.
    """

    def __init__(
        self,
        paired_config: PairedDatasetConfig,
        processor,
        action_tokenizer: ActionTokenizer,
        default_language: str,
        action_dim: int,
        pred_horizon: int,
        image_size: Optional[Tuple[int, int]] = None,
        vla_cache_dir: Optional[str] = None,
    ) -> None:
        self.processor        = processor
        self.action_tokenizer = action_tokenizer
        self.default_language = default_language
        self.action_dim       = action_dim
        self.pred_horizon     = pred_horizon
        self.image_size       = image_size or (224, 224)

        paired_config.vla = True
        self.paired_dataset   = HumanSimPairedDataset(paired_config)
        self.norm_stats       = None

        # ── Cache descriptors (cheap: just JSON reads) ───────────────────────
        self._cache_mode:  Optional[str] = None          # "sharded" | "monolithic" | None
        self._sharded_desc:   Optional[_ShardedCacheDescriptor]    = None
        self._monolithic_desc: Optional[_MonolithicCacheDescriptor] = None

        # Worker-local memmap store: each process starts empty.
        # Do NOT open any memmap here (parent process must stay clean).
        self._mmap_store = _WorkerMmapStore()

        # Paths cached from monolithic descriptor for fast worker access
        self._mono_feat_path: Optional[str]    = None
        self._mono_actions_path: Optional[str] = None
        self._mono_states_path: Optional[str]  = None
        self._mono_feat_shape: Optional[tuple]  = None

        if vla_cache_dir is not None:
            self._init_cache(vla_cache_dir)

        # Pre-extract image processor params for the fast tensor path.
        ip = self.processor.image_processor
        self._ip_do_letterbox   = getattr(ip, "tvf_do_letterbox", False)
        self._ip_letterbox_fill = getattr(ip, "tvf_letterbox_fill", None)
        self._ip_resize_params  = ip.tvf_resize_params
        self._ip_crop_params    = ip.tvf_crop_params
        self._ip_normalize_params = ip.tvf_normalize_params
        self._ip_num_backbones  = len(ip.input_sizes)

    # ------------------------------------------------------------------
    # Cache initialisation (runs only in parent process during __init__)
    # ------------------------------------------------------------------

    def _init_cache(self, cache_dir: str) -> None:
        meta_path = os.path.join(cache_dir, "vla_cache_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"vla_cache_meta.json not found in {cache_dir}. "
                "Run precompute_vla_features.py first."
            )
        top_meta = _read_json(meta_path)
        fmt = top_meta.get("format", "monolithic")

        if fmt == "sharded":
            desc = _ShardedCacheDescriptor(cache_dir, self.pred_horizon, self.action_dim)
            # Validate dataset size
            _validate_dims(top_meta, self.pred_horizon, self.action_dim,
                           desc.total_samples, len(self.paired_dataset))
            self._sharded_desc  = desc
            self._cache_mode    = "sharded"

        else:  # "monolithic" or legacy cache without format key
            desc = _MonolithicCacheDescriptor(cache_dir, self.pred_horizon, self.action_dim)
            _validate_dims(top_meta, self.pred_horizon, self.action_dim,
                           desc.total_samples, len(self.paired_dataset))
            self._monolithic_desc    = desc
            self._cache_mode         = "monolithic"
            self._mono_feat_path     = os.path.join(cache_dir, "vla_features.bin")
            self._mono_actions_path  = os.path.join(cache_dir, "actions.bin")
            self._mono_states_path   = os.path.join(cache_dir, "states.bin")
            self._mono_feat_shape    = (desc.total_samples, *desc.feat_shape)

            for p, name in [
                (self._mono_feat_path,    "vla_features.bin"),
                (self._mono_actions_path, "actions.bin"),
            ]:
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"{name} not found in {cache_dir}. "
                        "Run precompute_vla_features.py first."
                    )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.paired_dataset)

    def __getitem__(self, idx: int) -> Dict:
        if self._cache_mode == "sharded":
            return self._getitem_sharded(idx)
        if self._cache_mode == "monolithic":
            return self._getitem_monolithic(idx)
        return self._getitem_live(idx)

    # ------------------------------------------------------------------
    # Cache mode: sharded (V2)
    # ------------------------------------------------------------------

    def _getitem_sharded(self, idx: int) -> Dict:
        desc = self._sharded_desc
        si, offset = desc.idx_to_shard(idx)
        shard_info = desc.shards[si]
        shard_n    = shard_info["shard_n"]
        sd         = shard_info["shard_dir"]

        # Lazily open this shard's memmaps in this worker process
        feat_path    = os.path.join(sd, "feat.bin")
        actions_path = os.path.join(sd, "actions.bin")
        states_path  = os.path.join(sd, "states.bin")

        feat_shape_full    = (shard_n, *desc.feat_shape)
        actions_shape_full = (shard_n, self.pred_horizon, self.action_dim)
        states_shape_full  = (shard_n, self.action_dim)

        feat_mmap    = self._mmap_store.get(feat_path,    "float16", feat_shape_full)
        action_mmap  = self._mmap_store.get(actions_path, "float32", actions_shape_full)

        # Read this sample (copy to avoid holding a page-fault reference)
        feat_np   = feat_mmap[offset].copy()    # (*feat_shape) float16
        action_np = action_mmap[offset].copy()  # (pred_horizon, action_dim) float32

        # If pooled: feat_shape = (llm_dim,) → unsqueeze to (1, llm_dim)
        if desc.pool_features:
            vla_feat = torch.from_numpy(feat_np).unsqueeze(0)  # (1, llm_dim) float16
        else:
            vla_feat = torch.from_numpy(feat_np)               # (num_tokens, llm_dim) float16

        item: Dict = {
            "vla_feat":     vla_feat,
            "actions":      torch.from_numpy(action_np),
            "dataset_name": "lerobot_paired",
        }

        if desc.has_proprio and os.path.exists(states_path):
            sta_mmap = self._mmap_store.get(states_path, "float32", states_shape_full)
            item["proprio"] = torch.from_numpy(sta_mmap[offset].copy())

        return item

    # ------------------------------------------------------------------
    # Cache mode: monolithic (V1 backward compat)
    # ------------------------------------------------------------------

    def _getitem_monolithic(self, idx: int) -> Dict:
        desc = self._monolithic_desc
        N    = desc.total_samples

        # Lazily open memmaps in this worker
        feat_mmap   = self._mmap_store.get(
            self._mono_feat_path, "float16",
            (N, *desc.feat_shape),
        )
        action_mmap = self._mmap_store.get(
            self._mono_actions_path, "float32",
            (N, self.pred_horizon, self.action_dim),
        )

        feat_np   = feat_mmap[idx].copy()
        action_np = action_mmap[idx].copy()

        if desc.pool_features:
            vla_feat = torch.from_numpy(feat_np).unsqueeze(0)  # (1, llm_dim)
        else:
            vla_feat = torch.from_numpy(feat_np)               # (num_tokens, llm_dim)

        item: Dict = {
            "vla_feat":     vla_feat,
            "actions":      torch.from_numpy(action_np).float(),
            "dataset_name": "lerobot_paired",
        }

        if desc.has_proprio and os.path.exists(self._mono_states_path):
            sta_mmap = self._mmap_store.get(
                self._mono_states_path, "float32",
                (N, self.action_dim),
            )
            item["proprio"] = torch.from_numpy(sta_mmap[idx].copy()).float()

        return item

    # ------------------------------------------------------------------
    # Live (no cache) mode — unchanged from V1
    # ------------------------------------------------------------------

    def _getitem_live(self, idx: int) -> Dict:
        sample = self.paired_dataset[idx]

        language = sample.get("language", "")
        if not language or language.strip() == "":
            raise ValueError("Missing language in paired sample (human desc required).")

        if getattr(self.paired_dataset.config, "debug", False):
            human_task_id = sample.get("human_task_id")
            sim_task_id   = sample.get("sim_task_id")
            human_descs   = self.paired_dataset.task_mapper.human_descriptions.get(human_task_id)
            sim_descs     = self.paired_dataset.task_mapper.sim_descriptions.get(sim_task_id)
            source = "unknown"
            chosen_idx = None
            list_len   = None
            if isinstance(human_descs, list):
                list_len = len(human_descs)
                if language in human_descs:
                    source = "human_desc"
                    chosen_idx = human_descs.index(language)
            elif isinstance(human_descs, str) and language == human_descs:
                source = "human_desc"
            if source == "unknown":
                if isinstance(sim_descs, list) and language in sim_descs:
                    source = "sim_desc"
                elif isinstance(sim_descs, str) and language == sim_descs:
                    source = "sim_desc"

        # Extract image
        robot_obs = sample.get("robot_obs", {})
        view_tensor = self._select_view_tensor(robot_obs)
        pixel_values = self._apply_image_transform(view_tensor)

        # Extract proprio
        state_seq = robot_obs.get("states")
        proprio = None
        if state_seq is not None:
            proprio = self._extract_proprio(state_seq)

        action_seq = sample.get("robot_actions")
        if not isinstance(action_seq, torch.Tensor):
            action_seq = torch.tensor(action_seq)
        if action_seq.dim() == 1:
            action_seq = action_seq.unsqueeze(0)
        if action_seq.shape[0] != self.pred_horizon:
            action_seq = self._pad_or_trim(action_seq, self.pred_horizon)
        if action_seq.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: expected {self.action_dim}, got {action_seq.shape[-1]}"
            )

        action_seq = action_seq.to(torch.float32)
        action_seq = torch.clamp(action_seq, -1.0, 1.0)

        _, input_ids, labels = self._build_prompt_and_labels(language, action_seq)

        item = {
            "pixel_values": pixel_values,
            "input_ids":    input_ids,
            "labels":       labels,
            "actions":      action_seq,
            "dataset_name": "lerobot_paired",
            "language":     language,
        }
        if proprio is not None:
            item["proprio"] = proprio
        return item

    # ------------------------------------------------------------------
    # Image transform: direct tensor path (no PIL round-trip)
    # ------------------------------------------------------------------

    def _apply_image_transform(self, view_tensor: torch.Tensor) -> torch.Tensor:
        """Apply processor image transforms directly on a tensor."""
        tensor = view_tensor
        if tensor.dim() == 4:
            tensor = tensor[0]

        if tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor[..., :3].permute(2, 0, 1)
        elif tensor.dim() == 3 and tensor.shape[0] > 3:
            tensor = tensor[:3]

        tensor = tensor.float().clamp_(0.0, 1.0)

        if self._ip_do_letterbox and self._ip_letterbox_fill is not None:
            _, h, w = tensor.shape
            max_wh  = max(h, w)
            hp = int((max_wh - h) / 2)
            wp = int((max_wh - w) / 2)
            if hp > 0 or wp > 0:
                fill_vals = [v / 255.0 for v in self._ip_letterbox_fill]
                if fill_vals[0] == fill_vals[1] == fill_vals[2]:
                    tensor = TVF.pad(tensor, [wp, hp, wp, hp],
                                     fill=fill_vals[0], padding_mode="constant")
                else:
                    import torch.nn.functional as F
                    channels = []
                    for c in range(3):
                        channels.append(
                            F.pad(tensor[c:c+1], (wp, wp, hp, hp),
                                  mode="constant", value=fill_vals[c])
                        )
                    tensor = torch.cat(channels, dim=0)

        imgs_t = []
        for i in range(self._ip_num_backbones):
            img = TVF.resize(tensor, **self._ip_resize_params[i])
            img = TVF.center_crop(img, **self._ip_crop_params[i])
            img = TVF.normalize(img, **self._ip_normalize_params[i])
            imgs_t.append(img)

        return torch.vstack(imgs_t)

    # ------------------------------------------------------------------
    # Prompt / label construction
    # ------------------------------------------------------------------

    def _build_prompt_and_labels(self, lang_text: str, actions: torch.Tensor):
        prompt_builder   = PurePromptBuilder("openvla")
        flat_actions     = actions.flatten().cpu().float().numpy()
        action_token_str = self.action_tokenizer(flat_actions)

        conversation = [
            {"from": "human",
             "value": f"What action should the robot take to {lang_text.lower()}?"},
            {"from": "gpt", "value": action_token_str},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        prompt    = prompt_builder.get_prompt()
        input_ids = self.processor.tokenizer(prompt, add_special_tokens=True).input_ids
        labels    = list(input_ids)

        input_ids = torch.tensor(input_ids)
        labels    = torch.tensor(labels)
        labels[: -(len(action_token_str) + 1)] = IGNORE_INDEX

        return prompt, input_ids, labels

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_or_trim(seq: torch.Tensor, target_len: int) -> torch.Tensor:
        if seq.shape[0] >= target_len:
            return seq[:target_len]
        pad = seq[-1:].repeat(target_len - seq.shape[0], 1)
        return torch.cat([seq, pad], dim=0)

    def _extract_proprio(self, state_seq: torch.Tensor) -> torch.Tensor:
        if not isinstance(state_seq, torch.Tensor):
            state_seq = torch.tensor(state_seq)
        state = state_seq[0] if state_seq.dim() >= 2 else state_seq
        state = state.to(torch.float32)
        if state.shape[-1] < self.action_dim:
            pad   = torch.zeros(self.action_dim - state.shape[-1],
                                device=state.device, dtype=state.dtype)
            state = torch.cat([state, pad], dim=-1)
        elif state.shape[-1] > self.action_dim:
            state = state[: self.action_dim]
        return state

    @staticmethod
    def _select_view_tensor(robot_obs: Dict) -> torch.Tensor:
        if "view_1" in robot_obs:
            return robot_obs["view_1"]
        view_keys = sorted([k for k in robot_obs.keys() if k.startswith("view_")])
        if not view_keys:
            raise KeyError("robot_obs missing view tensors")
        return robot_obs[view_keys[0]]
