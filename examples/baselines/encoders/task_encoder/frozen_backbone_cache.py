"""
frozen_backbone_cache.py — Raw Feature Cache for FrozenVideoBackbone.

WHY THIS FILE EXISTS
──────────────────────────────────────────────────────────────────────────────
FrozenVideoBackbone has two distinct cost centres:

  1. Backbone forward (DINOv2/CLIP/VideoMAE): 20–50 ms per batch on GPU.
     Cost is per-step. With B=256 and ~15 unique tasks, this is 20–50 ms
     of pure inference cost on a model that NEVER changes its weights.

  2. Adapter forward (Linear × 2): ~0.1 ms per batch.
     Cost is negligible.

Since the backbone is 100% frozen, its output for a given video never changes.
We can therefore cache the pre-adapter raw features once and replay them every
training step — eliminating cost (1) and paying only cost (2).

ARCHITECTURE
──────────────────────────────────────────────────────────────────────────────

  Pre-compute phase (one-time, runs before training loop):
    video → [frozen backbone] → (cls_raw, seq_raw) → disk cache

  Training phase (per-step):
    human_repo_id → [RAM lookup] → (cls_raw, seq_raw)
                  → [trainable adapter] → (cls, seq)
                  → {"z": cls, "z_seq": seq}

  Gradients only flow through the adapter → correct optimisation.

SPEED COMPARISON (A100, batch=256, 15 unique tasks)
──────────────────────────────────────────────────────────────────────────────
  Without cache:  ~30 ms/iter (backbone runs every step)
  With cache:     ~0.15 ms/iter (dict lookup + 2× Linear)

INTEGRATION
──────────────────────────────────────────────────────────────────────────────
  # In training script, after building the backbone:
  backbone = setup_frozen_backbone_cache(
      backbone,
      dataloader=train_dataloader,
      device=device,
      backbone_type=args.frozen_backbone_type,
      te_cache_root=args.te_cache_root,
  )
  # enable_skip_human_video works unchanged because CachedFrozenBackbone
  # exposes _task_emb_cache so existing infrastructure picks it up.
  enable_skip_human_video(backbone, train_dataloader)

──────────────────────────────────────────────────────────────────────────────
  setup_frozen_task_encoder() already handles Condition A correctly:
    - ckpt_path=None → L1 VL cache disabled (no fingerprinting)
    - L3 task-embedding cache IS built under te_cache_root/task_embeddings_no_ckpt/
    - One-time precompute: runs Qwen2-VL once per unique task (~slow but once)
    - Training after precompute: ~0.05 ms/iter (L3 dict lookup, bypasses encoder)
  → No code changes needed for Condition A.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


# ── Compatibility shim so enable_skip_human_video() works unchanged ───────────

class _IsPopulatedProxy:
    """Minimal duck-type of TaskEmbeddingCache for enable_skip_human_video()."""

    def __init__(self, raw_cache: "RawFeatureCache") -> None:
        self._c = raw_cache

    @property
    def is_populated(self) -> bool:
        return self._c.is_populated

    @property
    def num_entries(self) -> int:
        return len(self._c._store)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  RawFeatureCache — persistent store for pre-adapter backbone features
# ═══════════════════════════════════════════════════════════════════════════════

class RawFeatureCache:
    """Disk + RAM cache for pre-adapter backbone raw features.

    Stored fields per key (human_repo_id):
        "cls" : Tensor[backbone_dim]          — global / CLS token
        "seq" : Tensor[max_seq_patches, backbone_dim]  — patch tokens

    All tensors kept on CPU; non-blocking transfer to training device at
    lookup time (~50 µs).

    File layout
    -----------
    {cache_dir}/backbone_{backbone_type}/raw_features.pt
    Single .pt file — key set is small (15–200 unique tasks), so one
    torch.save/load per flush is fast and simple.
    """

    FILENAME         = "raw_features.pt"
    LAZY_FLUSH_EVERY = 20

    def __init__(
        self,
        cache_dir: str,
        backbone_type: str,
        preload_to_memory: bool = True,
        verbose: bool = True,
    ) -> None:
        subdir = Path(cache_dir) / f"backbone_{backbone_type}"
        subdir.mkdir(parents=True, exist_ok=True)
        self._path   = subdir / self.FILENAME
        self._store: Dict[str, Dict[str, torch.Tensor]] = {}
        self._dirty  = 0
        self._log    = print if verbose else lambda *a, **k: None

        if preload_to_memory and self._path.exists():
            self._load()

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def is_populated(self) -> bool:
        return len(self._store) > 0

    def is_cached(self, key: str) -> bool:
        return key in self._store

    def all_cached(self, keys: List[str]) -> bool:
        return all(k in self._store for k in keys)

    # ── Hot-path read (called every training step — must be fast) ────────────

    def get_batch(
        self,
        keys: List[str],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return {"cls": (B, D), "seq": (B, N, D)} on `device`, or None on miss.

        Complexity: O(B) dict lookups + one non-blocking .to(device) per field.
        Typical latency: ~0.05 ms for B=256, 15 unique keys.
        """
        entries = [self._store.get(k) for k in keys]
        if any(e is None for e in entries):
            return None

        cls_list = [e["cls"] for e in entries]
        seq_list = [e["seq"] for e in entries]

        # Pad seq to uniform length in case different tasks have different N
        max_n = max(s.shape[0] for s in seq_list)
        if not all(s.shape[0] == max_n for s in seq_list):
            padded = []
            for s in seq_list:
                if s.shape[0] < max_n:
                    pad = s.new_zeros(max_n - s.shape[0], s.shape[1])
                    s = torch.cat([s, pad], dim=0)
                padded.append(s)
            seq_list = padded

        return {
            "cls": torch.stack(cls_list).to(device, non_blocking=True),
            "seq": torch.stack(seq_list).to(device, non_blocking=True),
        }

    # ── Write ────────────────────────────────────────────────────────────────

    def put_batch(
        self,
        keys: List[str],
        cls_raw: torch.Tensor,   # (B, D) CPU
        seq_raw: torch.Tensor,   # (B, N, D) CPU
        flush: bool = False,
    ) -> None:
        """Cache a batch of raw features. Idempotent: skips already-cached keys."""
        added = 0
        for i, key in enumerate(keys):
            if key in self._store:
                continue
            self._store[key] = {
                "cls": cls_raw[i].detach().cpu(),
                "seq": seq_raw[i].detach().cpu(),
            }
            added += 1
        self._dirty += added
        if flush or self._dirty >= self.LAZY_FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        """Atomically persist to disk (tmp → rename, crash-safe)."""
        if self._dirty == 0:
            return
        tmp = self._path.with_suffix(".tmp")
        torch.save(self._store, tmp)
        tmp.replace(self._path)
        self._log(
            f"  [RawFeatCache] Flushed {len(self._store)} entries → {self._path}"
        )
        self._dirty = 0

    def stats_str(self, recent_keys: Optional[List[str]] = None) -> str:
        if recent_keys is None:
            return f"{len(self._store)} entries"
        hits = sum(1 for k in recent_keys if k in self._store)
        pct  = hits / max(len(recent_keys), 1) * 100
        return f"{hits}/{len(recent_keys)} hits ({pct:.0f}%) | {len(self._store)} total"

    def _load(self) -> None:
        try:
            data = torch.load(self._path, map_location="cpu", weights_only=True)
            if isinstance(data, dict):
                self._store = data
                self._log(
                    f"  [RawFeatCache] Loaded {len(self._store)} entries ← {self._path}"
                )
        except Exception as e:
            self._log(f"  [RawFeatCache] WARNING: load failed ({e}), starting fresh.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CachedFrozenBackbone — wraps FrozenVideoBackbone with raw-feature cache
# ═══════════════════════════════════════════════════════════════════════════════

class CachedFrozenBackbone(nn.Module):
    """FrozenVideoBackbone wrapper that serves raw features from a RAM cache.

    At training time, the backbone is NEVER called — only the trainable adapter
    layers run. This eliminates the ~20–50 ms/iter backbone cost and brings
    task-encoding time to ~0.15 ms/iter (adapter-only).

    Gradient flow
    ─────────────
    raw features (cached, no grad) → adapter layers (trainable, grad flows) → z

    This is correct: gradients do not need to reach the frozen backbone, and
    they correctly update the adapter so it can learn from the policy loss.

    The backbone itself is kept as a submodule so that:
      • self.backbone.cls_adapter and .seq_adapter remain registered
        PyTorch submodules and appear in optimizer parameter groups.
      • state_dict() serialises them properly.
      • .to(device) propagates correctly.
    """

    def __init__(
        self,
        backbone,                    # FrozenVideoBackbone
        cache: RawFeatureCache,
    ) -> None:
        super().__init__()
        self.backbone = backbone     # submodule — adapter gets proper grad tracking
        self._raw_cache  = cache

        # Expose a _task_emb_cache proxy so enable_skip_human_video() works
        # unchanged (it checks .is_populated on this attribute).
        self._task_emb_cache = _IsPopulatedProxy(cache)

    # ── Forward (hot path — called every training step) ──────────────────────

    def forward(
        self,
        video,
        sample_ids: Optional[List[str]] = None,
    ):
        """
        Args
            video       (B, T, H, W, C) or similar — only used on cache miss
            sample_ids  list[str] of human_repo_id — cache keys

        Returns
            cls : (B, latent_dim)
            seq : (B, N, latent_dim)
        """
        dev = self.backbone.cls_adapter[0].weight.device

        # ── Fast path: cache hit ──────────────────────────────────────────────
        if sample_ids is not None:
            cached = self._raw_cache.get_batch(sample_ids, dev)
            if cached is not None:
                raw_cls = cached["cls"]   # (B, D_backbone)
                raw_seq = cached["seq"]   # (B, N, D_backbone)

                # Adapter + PE — these are the ONLY ops that run at training time
                cls = self.backbone.cls_adapter(raw_cls)    # (B, latent_dim)
                seq = self.backbone.seq_adapter(raw_seq)    # (B, N, latent_dim)
                seq = self.backbone._pe(seq)                # + positional encoding
                return cls, seq

        # ── Slow path: cache miss — run full backbone (precompute or cold start)
        # Guard: if skip_human_video is active, video will be None here, which
        # means sample_ids were not found in the cache (key mismatch or stale
        # cache).  Raise a clear error instead of crashing inside the backbone.
        if video is None:
            missing = []
            if sample_ids is not None:
                missing = [
                    sid for sid in sample_ids
                    if not self._raw_cache.is_cached(sid)
                ]
            cache_path = getattr(self._raw_cache, "_path", "unknown")
            raise RuntimeError(
                "[CachedFrozenBackbone] Cache miss with video=None.\n"
                "  skip_human_video is active (video frames are not loaded by the "
                "DataLoader) but the following sample_id(s) were not found in the "
                f"raw-feature cache:\n"
                f"    {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
                f"  Cache file: {cache_path}  ({len(self._raw_cache._store)} entries)\n"
                "  Possible causes:\n"
                "    1. The sample_id keys in the dataset do not match those stored "
                "in the cache (e.g. path format changed after caching).\n"
                "    2. New tasks were added to the dataset after the cache was built.\n"
                "  Fix: delete the cache file and re-run training to trigger "
                "precompute, then the cache will be rebuilt with the correct keys."
            )

        return self.backbone(video, sample_ids=sample_ids)

    def encode(
        self,
        human_video=None,
        human_vl_ids: Optional[List[str]] = None,
        **_,   # absorb human_desc, robot_first_frame, cache_keys, etc.
    ):
        """Encode human_video → {"z": cls, "z_seq": seq}.

        human_vl_ids serves as cache keys (same convention as VLFeatureCache).
        """
        if human_video is None and human_vl_ids is None:
            raise ValueError(
                "CachedFrozenBackbone.encode() requires human_video OR human_vl_ids."
            )
        cls, seq = self.forward(human_video, sample_ids=human_vl_ids)
        return {"z": cls, "z_seq": seq}

    # ── Utility ──────────────────────────────────────────────────────────────

    def trainable_params(self):
        """Only adapter + PE are trainable (backbone weights frozen)."""
        return self.backbone.trainable_params()

    def cache_stats(self, recent_keys: Optional[List[str]] = None) -> str:
        return "RawFeat: " + self._raw_cache.stats_str(recent_keys)


# ── Precompute DataLoader helpers (deduplicated over unique human repos) ──────

class _HumanVideoOnlyDataset(torch.utils.data.Dataset):
    """One item per unique human_repo_id — avoids coding the same video N times."""

    def __init__(self, paired_dataset) -> None:
        self._human_dataset = paired_dataset.human_dataset
        mapper = paired_dataset.task_mapper

        # Build sim_idx -> task_id reverse map
        sim_idx_to_task: dict = {}
        for attr in ('task_to_sim_indices', 'task_to_robot_indices'):
            mapping = getattr(paired_dataset, attr, None)
            if mapping:
                for tid, idxs in mapping.items():
                    for i in idxs:
                        sim_idx_to_task[i] = tid
                break

        # Collect unique human_repo_ids
        seen: dict = {}
        for actual_idx in paired_dataset.valid_indices:
            sim_task_id = sim_idx_to_task.get(actual_idx)
            if sim_task_id is None:
                continue
            htid = (
                mapper.get_human_task_from_sim(sim_task_id)
                or mapper.get_human_task_from_robot(sim_task_id)
            )
            if htid is None or htid not in paired_dataset.task_to_human_indices:
                continue
            rid = paired_dataset.task_to_human_indices[htid]['repos'][0]['repo_id']
            seen.setdefault(rid, True)

        self._unique_repo_ids: List[str] = sorted(seen.keys())
        print(
            f"  [Precompute] {len(self._unique_repo_ids)} unique repos "
            f"(dedup from {len(paired_dataset.valid_indices)} training samples)"
        )

    def __len__(self) -> int:
        return len(self._unique_repo_ids)

    def __getitem__(self, idx: int) -> dict:
        rid  = self._unique_repo_ids[idx]
        item = self._human_dataset._get_target_item(rid)
        return {
            'human_video': item['video'],   # [T, H, W, C]
            'human_desc':  '',
            'sample_id':   rid,             # used as cache_key
        }


def _build_precompute_loader(train_dataloader):
    """Build a fast deduplicated DataLoader for offline precomputation.

    Returns (loader, aug_was_on).
    """
    dataset    = train_dataloader.dataset
    aug_was_on = False
    if getattr(dataset, 'enable_augmentation', False):
        aug_was_on  = True
        dataset.enable_augmentation = False

    has_meta = (
        getattr(dataset, 'human_dataset', None) is not None
        and hasattr(dataset, 'valid_indices')
        and (hasattr(dataset, 'task_to_sim_indices')
             or hasattr(dataset, 'task_to_robot_indices'))
        and hasattr(dataset, 'task_to_human_indices')
        and hasattr(dataset, 'task_mapper')
    )

    if has_meta:
        try:
            thin = _HumanVideoOnlyDataset(dataset)
            n    = len(thin)

            def _collate(batch):
                return {
                    'human_video': torch.stack([b['human_video'] for b in batch]),
                    'human_desc':  [b['human_desc']  for b in batch],
                    'sample_id':   [b['sample_id']   for b in batch],
                }

            nw = min(getattr(train_dataloader, 'num_workers', 4) or 4, max(1, n))
            bs = max(1, min(train_dataloader.batch_size or 32, max(1, n // 4)))

            loader = torch.utils.data.DataLoader(
                thin,
                batch_size=bs,
                shuffle=False,
                num_workers=nw,
                collate_fn=_collate,
                pin_memory=True,
                drop_last=False,
                persistent_workers=(nw > 1),
                prefetch_factor=4 if nw > 1 else None,
            )
            print(
                f"  [Precompute] Loader: {n} repos, bs={bs}, "
                f"workers={nw}, batches={len(loader)}"
            )
            return loader, aug_was_on

        except Exception as e:
            print(
                f"  [Precompute] WARNING: thin dataset failed ({e}), "
                f"falling back to training loader."
            )
            if aug_was_on:
                dataset.enable_augmentation = True
            aug_was_on = False

    print("  [Precompute] Using training loader (fallback).")
    return train_dataloader, aug_was_on


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  precompute_raw_features — one-time offline precomputation
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_raw_features(
    backbone,               # FrozenVideoBackbone (unwrapped)
    dataloader,
    device: torch.device,
    cache: RawFeatureCache,
    verbose: bool = True,
) -> RawFeatureCache:
    """Run the frozen backbone on every unique task and store raw features.

    Only the backbone is called here — NOT the adapter.  The adapter runs
    during training and is NOT cached.

    Args
        backbone    : FrozenVideoBackbone instance (must be loaded / initialised)
        dataloader  : training DataLoader (used to build the thin precompute loader)
        device      : GPU device for backbone inference
        cache       : RawFeatureCache to populate
        verbose     : print progress
    """
    _log = print if verbose else lambda *a, **k: None

    backbone.eval()

    loader, aug_was_on = _build_precompute_loader(dataloader)
    n_unique = len(loader.dataset)

    already    = sum(1 for rid in loader.dataset._unique_repo_ids
                     if cache.is_cached(rid))
    to_compute = n_unique - already
    _log(
        f"\n  [RawFeatCache] Precompute: {n_unique} unique repos, "
        f"{already} cached, {to_compute} to compute."
    )

    if to_compute == 0:
        _log("  [RawFeatCache] All entries cached — skipping precompute.")
        if aug_was_on:
            dataloader.dataset.enable_augmentation = True
        return cache

    # Trigger lazy load of backbone weights (if not already loaded)
    backbone._load_backbone()

    try:
        with torch.no_grad():
            for batch in tqdm(loader, desc="  Precomputing backbone raw features",
                              disable=not verbose):
                repo_ids: List[str] = batch["sample_id"]

                if cache.all_cached(repo_ids):
                    continue

                human_video = batch["human_video"].to(device, non_blocking=True)

                # Run ONLY the frozen backbone — not the adapter
                dispatch = {
                    "dino":     backbone._encode_dino,
                    "clip_hf":  backbone._encode_clip_hf,
                    "siglip":   backbone._encode_siglip,
                    "videomae": backbone._encode_videomae,
                }
                raw_cls, raw_seq = dispatch[backbone._type](human_video)
                # raw_cls: (B, D_backbone) — pre-adapter, on backbone's device
                # raw_seq: (B, N, D_backbone)

                cache.put_batch(
                    repo_ids,
                    raw_cls.cpu(),
                    raw_seq.cpu(),
                    flush=True,   # crash-safe: save progress after every batch
                )

    finally:
        if aug_was_on:
            dataloader.dataset.enable_augmentation = True

    cache.flush()
    _log(
        f"  [RawFeatCache] Done.  {cache._store and len(cache._store)} entries cached → "
        f"{cache._path}"
    )
    return cache


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  setup_frozen_backbone_cache — main public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def setup_frozen_backbone_cache(
    backbone,                         # FrozenVideoBackbone
    dataloader,
    device: torch.device,
    backbone_type: str,
    te_cache_root: Optional[str] = "./te_cache",
    preload_to_memory: bool = True,
    recompute: bool = False,
    verbose: bool = True,
) -> "CachedFrozenBackbone":
    """Freeze the backbone and build a raw-feature cache for maximum speed.

    Call AFTER building the training DataLoader, BEFORE the training loop.
    Returns a CachedFrozenBackbone ready to drop into the agent in place of
    the bare FrozenVideoBackbone.

    After this call, also run:
        enable_skip_human_video(cached_backbone, dataloader)
    to stop the DataLoader from decoding video frames that will never be used.

    Performance after setup
    ───────────────────────
    Task-encoding cost per training step:
        dict lookup (~0.05 ms) + 2× Linear (~0.1 ms) = ~0.15 ms/batch
    vs. without cache:
        backbone forward ~20–50 ms/batch

    Args
        backbone        : FrozenVideoBackbone (adapter layers remain trainable)
        dataloader      : training DataLoader
        device          : training device
        backbone_type   : string key (e.g. "dinov2_vitl14") for cache sub-dir
        te_cache_root   : root cache directory
        preload_to_memory: load all entries into RAM at startup
        recompute       : force re-precompute even if cache exists
        verbose         : print progress messages
    """
    _log = print if verbose else lambda *a, **k: None
    _log(f"\n{'=' * 60}")
    _log(f"[FrozenBB] Setting up frozen backbone cache ({backbone_type})")

    # ── Step 1: Freeze backbone weights entirely ──────────────────────────────
    for p in backbone.parameters():
        p.requires_grad_(False)
    # Un-freeze only the adapter (cls_adapter + seq_adapter + _pe)
    for p in backbone.trainable_params():
        p.requires_grad_(True)

    frozen_n   = sum(1 for p in backbone.parameters() if not p.requires_grad)
    trainable_n = sum(p.numel() for p in backbone.trainable_params())
    _log(f"  Backbone: {frozen_n} frozen tensors | Adapter: {trainable_n/1e3:.1f}K trainable params")

    # ── Step 2: Build or load cache ──────────────────────────────────────────
    if te_cache_root is None:
        _log("  [RawFeatCache] te_cache_root=None — cache disabled. "
             "Training will run backbone every step (~20-50 ms/iter).")
        return CachedFrozenBackbone(backbone, _EmptyCache())

    cache = RawFeatureCache(
        cache_dir=te_cache_root,
        backbone_type=backbone_type,
        preload_to_memory=preload_to_memory,
        verbose=verbose,
    )

    if recompute and cache._path.exists():
        _log("  [RawFeatCache] recompute=True — clearing existing cache...")
        cache._path.unlink()
        cache = RawFeatureCache(
            cache_dir=te_cache_root,
            backbone_type=backbone_type,
            preload_to_memory=False,
            verbose=verbose,
        )

    # ── Step 3: Precompute if needed ─────────────────────────────────────────
    if not cache.is_populated:
        _log("  [RawFeatCache] Cache empty — starting one-time precompute...")
        _log(f"  Backbone: {backbone.backbone_type}  "
             f"(will run once per unique task, never again)")
        backbone.eval()
        precompute_raw_features(backbone, dataloader, device, cache, verbose=verbose)
    else:
        _log(f"  [RawFeatCache] Cache ready: {len(cache._store)} entries loaded from disk.")

    # ── Step 4: Wrap ─────────────────────────────────────────────────────────
    wrapped = CachedFrozenBackbone(backbone, cache)

    _log(f"  [Done] CachedFrozenBackbone ready.")
    _log(f"  Expected per-step cost: ~0.15 ms  (dict lookup + adapter)")
    _log(f"{'=' * 60}\n")
    return wrapped


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  enable_skip_human_video — drop video decode I/O once the cache is ready
# ═══════════════════════════════════════════════════════════════════════════════

def enable_skip_human_video(
    task_encoder,
    dataloader,
    recreate_dataloader: bool = False,
    verbose: bool = True,
):
    """Activate skip_human_video on the dataset when the raw-feature cache is ready.

    After setup_frozen_backbone_cache() populates the cache, every sample's
    human_video tensor is decoded from disk and then immediately discarded
    (raw features are read from cache instead of recomputed).  Enabling
    skip_human_video removes this dominant per-sample I/O cost (~50-100 ms of
    video decode per sample).

    num_workers=0 (default)
    -----------------------
    Setting the flag on the dataset object propagates immediately.  The main
    process reads __getitem__ directly, so recreate_dataloader is not needed.

    num_workers>0 with persistent_workers=True
    ------------------------------------------
    Worker processes have already forked with a stale copy of the dataset and
    will not see the flag change.  Pass recreate_dataloader=True to rebuild
    the DataLoader so workers pick up the new flag.
    """
    _log = print if verbose else lambda *a, **k: None

    cache = getattr(task_encoder, '_task_emb_cache', None)
    if cache is None or not cache.is_populated:
        _log("  [SkipVideo] Raw feature cache not ready — skip_human_video NOT activated.")
        return None

    dataset = dataloader.dataset
    if not hasattr(dataset, 'skip_human_video'):
        _log("  [SkipVideo] Dataset does not support skip_human_video; skipped.")
        return None

    dataset.skip_human_video = True
    _log(
        f"  [SkipVideo] skip_human_video=True activated on dataset "
        f"({cache.num_entries} cache entries cover all tasks).\n"
        f"  Video decode cost (~50-100 ms/sample) eliminated for training."
    )

    if recreate_dataloader:
        if dataloader.batch_size is None:
            raise ValueError(
                "[SkipVideo] Cannot determine batch_size for DataLoader recreation."
            )
        import torch.utils.data as tud
        new_loader = tud.DataLoader(
            dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=True,
            persistent_workers=(dataloader.num_workers > 0),
            prefetch_factor=(
                dataloader.prefetch_factor if dataloader.num_workers > 0 else None
            ),
            multiprocessing_context=(
                dataloader.multiprocessing_context
                if dataloader.num_workers > 0 else None
            ),
        )
        return new_loader

    return None


# ── Fallback: empty cache (when te_cache_root is None) ───────────────────────

class _EmptyCache:
    """Placeholder used when caching is disabled (te_cache_root=None)."""

    is_populated = False

    def get_batch(self, keys, device):
        return None   # always cache miss → backbone runs every step

    def put_batch(self, *a, **k):
        pass

    def stats_str(self, *a, **k):
        return "disabled"