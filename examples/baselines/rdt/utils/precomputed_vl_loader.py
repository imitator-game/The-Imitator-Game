"""
Precomputed vision/language feature loader for RDT training.

Supported layouts:

1. Sharded format (recommended)
   <precomputed_dir>/
     metadata.json
     feature_index.json   # sample_id -> {"path": "shard_00000.pt", "offset": 123}
     shard_00000.pt
     shard_00001.pt

2. Mmap image-only format
   <precomputed_dir>/
     metadata.json
     feature_index.json   # sample_id -> {"array": "img_tokens.npy", "offset": 123}
     img_tokens.npy

3. Legacy per-sample format
   <precomputed_dir>/
     metadata.json
     feature_index.json   # optional, sample_id -> "<sample_id>.pt"
     <sample_id>.pt

Each full V/L sample provides:
    - img_tokens: Tensor[img_len, img_dim]
    - lang_embeds: Tensor[lang_len, lang_dim]
    - lang_mask: Tensor[lang_len] (bool)

Language-only caches provide only:
    - lang_embeds
    - lang_mask

Image-only caches provide only:
    - img_tokens
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class PrecomputedVLFeatures:
    """Loads precomputed image/language features keyed by stable sample ids."""

    def __init__(
        self,
        precomputed_dir: str,
        *,
        device: str = "cpu",
        preload_to_memory: bool = False,
        shard_cache_size: int = 8,
    ):
        self.precomputed_dir = Path(precomputed_dir)
        self.device = device
        self.preload_to_memory = preload_to_memory
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._memory_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._shard_cache: "OrderedDict[Path, dict[str, object]]" = OrderedDict()
        self._array_cache: dict[Path, np.memmap] = {}
        self._text_cache: Optional[dict[str, tuple[torch.Tensor, torch.Tensor]]] = None

        if not self.precomputed_dir.exists():
            raise FileNotFoundError(f"Precomputed VL feature dir not found: {self.precomputed_dir}")

        self._index = self._load_index()
        self.metadata = self._load_metadata()
        self._load_text_cache()
        if preload_to_memory:
            self._preload_all()

    def _load_index(self) -> dict[str, object]:
        index_paths = sorted(self.precomputed_dir.glob("feature_index*.json"))
        if not index_paths:
            return {}

        mapping: dict[str, object] = {}
        for index_path in index_paths:
            with index_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, dict):
                iterator = raw.items()
            elif isinstance(raw, list):
                iterator = (
                    (item["sample_id"], item)
                    for item in raw
                    if isinstance(item, dict) and "sample_id" in item
                )
            else:
                raise ValueError(
                    f"Unsupported feature index format in {index_path}: {type(raw).__name__}"
                )

            for sample_id, entry in iterator:
                sid = str(sample_id)
                if isinstance(entry, str):
                    mapping[sid] = (self.precomputed_dir / entry).resolve()
                elif isinstance(entry, dict):
                    array_rel_path = entry.get("array")
                    if array_rel_path:
                        offset = entry.get("offset")
                        if offset is None:
                            raise ValueError(f"feature_index mmap entry missing 'offset' for sample_id={sid}")
                        mapped_entry = {
                            "array": (self.precomputed_dir / str(array_rel_path)).resolve(),
                            "offset": int(offset),
                        }
                        if entry.get("quantization"):
                            mapped_entry["quantization"] = str(entry["quantization"])
                        if entry.get("scale_array"):
                            mapped_entry["scale_array"] = (
                                self.precomputed_dir / str(entry["scale_array"])
                            ).resolve()
                        mapping[sid] = mapped_entry
                        continue

                    rel_path = entry.get("path")
                    if not rel_path:
                        raise ValueError(f"feature_index entry missing 'path' for sample_id={sid}")
                    offset = entry.get("offset")
                    resolved = (self.precomputed_dir / rel_path).resolve()
                    if offset is None:
                        mapping[sid] = resolved
                    else:
                        mapping[sid] = (resolved, int(offset))
                else:
                    raise ValueError(
                        f"Unsupported feature_index entry for sample_id={sid}: {type(entry).__name__}"
                    )
        return mapping

    def _load_metadata(self) -> dict:
        metadata_path = self.precomputed_dir / "metadata.json"
        if not metadata_path.exists():
            return {}
        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"metadata.json must contain a JSON object: {metadata_path}")
        return data

    def _load_text_cache(self) -> None:
        path = self.precomputed_dir / "lang_features.pt"
        if not path.exists():
            return
        data = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f"lang_features.pt must contain a dict: {path}")
        texts = data.get("texts")
        embeds = data.get("lang_embeds")
        masks = data.get("lang_mask")
        if texts is None or embeds is None or masks is None:
            raise KeyError(f"lang_features.pt must contain texts/lang_embeds/lang_mask: {path}")
        self._text_cache = {
            str(text): (embeds[i], masks[i].bool())
            for i, text in enumerate(texts)
        }

    def _default_path(self, sample_id: str) -> Path:
        safe_name = sample_id.replace("/", "__")
        return self.precomputed_dir / f"{safe_name}.pt"

    def _resolve_entry(self, sample_id: str) -> object:
        return self._index.get(sample_id, self._default_path(sample_id))

    def _get_shard(self, path: Path) -> dict[str, object]:
        cached = self._shard_cache.get(path)
        if cached is not None:
            self._shard_cache.move_to_end(path)
            return cached

        data = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f"Shard file must contain a dict: {path}")

        feature_mode = self.metadata.get("feature_mode", "vl")
        required = ("img_tokens",) if feature_mode == "image_only" else ("lang_embeds", "lang_mask")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing keys in shard file {path}: {missing}")

        self._shard_cache[path] = data
        self._shard_cache.move_to_end(path)
        while len(self._shard_cache) > self.shard_cache_size and not self.preload_to_memory:
            self._shard_cache.popitem(last=False)
        return data

    def _get_array(self, path: Path) -> np.memmap:
        cached = self._array_cache.get(path)
        if cached is not None:
            return cached
        if not path.exists():
            raise FileNotFoundError(f"Missing precomputed mmap array: {path}")
        array = np.load(path, mmap_mode="r")
        if not isinstance(array, np.memmap):
            raise ValueError(f"Expected mmap-backed .npy array: {path}")
        self._array_cache[path] = array
        return array

    def has_sample(self, sample_id: str) -> bool:
        if sample_id in self._memory_cache:
            return True
        entry = self._resolve_entry(sample_id)
        if isinstance(entry, dict) and "array" in entry:
            if not entry["array"].exists():
                return False
            scale_array = entry.get("scale_array")
            return scale_array is None or scale_array.exists()
        if isinstance(entry, tuple):
            return entry[0].exists()
        return entry.exists()

    def _load_sample(self, sample_id: str) -> dict[str, torch.Tensor]:
        if sample_id in self._memory_cache:
            return self._memory_cache[sample_id]

        entry = self._resolve_entry(sample_id)
        if isinstance(entry, dict) and "array" in entry:
            array = self._get_array(entry["array"])
            offset = int(entry["offset"])
            if offset < 0 or offset >= array.shape[0]:
                raise IndexError(
                    f"Precomputed mmap offset out of bounds for sample_id={sample_id}: "
                    f"offset={offset} shape={array.shape}"
                )
            if entry.get("quantization") == "int8_per_channel":
                scale_array = self._get_array(entry["scale_array"])
                sample = {
                    "img_tokens_q": torch.from_numpy(np.asarray(array[offset]).copy()),
                    "img_tokens_scale": torch.from_numpy(np.asarray(scale_array[offset]).copy()),
                }
            else:
                sample = {"img_tokens": torch.from_numpy(np.asarray(array[offset]).copy())}
        elif isinstance(entry, tuple):
            path, offset = entry
            if not path.exists():
                raise FileNotFoundError(f"Missing precomputed VL shard for sample_id={sample_id}: {path}")
            shard = self._get_shard(path)
            sample = {}
            if "lang_embeds" in shard:
                sample["lang_embeds"] = shard["lang_embeds"][offset]
            if "lang_mask" in shard:
                sample["lang_mask"] = shard["lang_mask"][offset]
            if "img_tokens" in shard:
                sample["img_tokens"] = shard["img_tokens"][offset]
        else:
            path = entry
            if not path.exists():
                raise FileNotFoundError(f"Missing precomputed VL feature for sample_id={sample_id}: {path}")
            data = torch.load(path, map_location=self.device, weights_only=False)
            if not isinstance(data, dict):
                raise ValueError(f"Precomputed VL feature file must contain a dict: {path}")
            feature_mode = self.metadata.get("feature_mode", "vl")
            required = ("img_tokens",) if feature_mode == "image_only" else ("lang_embeds", "lang_mask")
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"Missing keys in precomputed VL feature file {path}: {missing}")
            sample = {}
            if "lang_embeds" in data:
                sample["lang_embeds"] = data["lang_embeds"]
            if "lang_mask" in data:
                sample["lang_mask"] = data["lang_mask"]
            if "img_tokens" in data:
                sample["img_tokens"] = data["img_tokens"]

        if self.preload_to_memory:
            self._memory_cache[sample_id] = sample
        return sample

    def _get_mmap_image_batch(self, sample_ids: list[str]) -> Optional[dict[str, torch.Tensor]]:
        entries = [self._resolve_entry(sample_id) for sample_id in sample_ids]
        if not entries or not all(isinstance(entry, dict) and "array" in entry for entry in entries):
            return None

        quantized = str(entries[0].get("quantization", "none"))
        if any(str(entry.get("quantization", "none")) != quantized for entry in entries):
            raise ValueError("Mixed quantized and non-quantized mmap image cache entries in one batch.")

        grouped: dict[Path, list[tuple[int, int]]] = {}
        for batch_pos, entry in enumerate(entries):
            grouped.setdefault(entry["array"], []).append((batch_pos, int(entry["offset"])))

        out = None
        scale_out = None
        for path, positions_offsets in grouped.items():
            array = self._get_array(path)
            offsets = [offset for _, offset in positions_offsets]
            if any(offset < 0 or offset >= array.shape[0] for offset in offsets):
                raise IndexError(f"Precomputed mmap offsets out of bounds for array {path}")
            rows = np.asarray(array[offsets])
            if out is None:
                out = np.empty((len(sample_ids), *rows.shape[1:]), dtype=rows.dtype)
            for local_idx, (batch_pos, _) in enumerate(positions_offsets):
                out[batch_pos] = rows[local_idx]

            if quantized == "int8_per_channel":
                first_entry = entries[positions_offsets[0][0]]
                scale_array = self._get_array(first_entry["scale_array"])
                scale_rows = np.asarray(scale_array[offsets])
                if scale_out is None:
                    scale_out = np.empty(
                        (len(sample_ids), *scale_rows.shape[1:]),
                        dtype=scale_rows.dtype,
                    )
                for local_idx, (batch_pos, _) in enumerate(positions_offsets):
                    scale_out[batch_pos] = scale_rows[local_idx]

        if out is None:
            return None
        if quantized == "int8_per_channel":
            if scale_out is None:
                raise KeyError("Quantized mmap image cache is missing scale rows.")
            return {
                "img_tokens_q": torch.from_numpy(out),
                "img_tokens_scale": torch.from_numpy(scale_out),
            }
        return {"img_tokens": torch.from_numpy(out)}

    def _preload_all(self) -> None:
        if any(isinstance(v, dict) and "array" in v for v in self._index.values()):
            raise ValueError("preload_to_memory is not supported for mmap precomputed features")
        if any(isinstance(v, tuple) for v in self._index.values()):
            shard_paths = sorted({entry[0] for entry in self._index.values() if isinstance(entry, tuple)})
            for path in shard_paths:
                try:
                    self._get_shard(path)
                except Exception as exc:
                    print(f"Warning: failed to preload VL shard {path}: {exc}")
        else:
            candidate_ids: set[str] = set(self._index.keys())
            if not candidate_ids:
                for path in self.precomputed_dir.glob("*.pt"):
                    if path.name.startswith("shard_"):
                        continue
                    candidate_ids.add(path.stem.replace("__", "/"))

            for sample_id in sorted(candidate_ids):
                try:
                    self._memory_cache[sample_id] = self._load_sample(sample_id)
                except Exception as exc:
                    print(f"Warning: failed to preload VL feature for {sample_id}: {exc}")

    def _get_text_batch(self, languages: list[str]) -> Optional[dict[str, torch.Tensor]]:
        if self._text_cache is None or not languages:
            return None
        if any(str(text) not in self._text_cache for text in languages):
            return None
        entries = [self._text_cache[str(text)] for text in languages]
        return {
            "lang_embeds": torch.stack([entry[0] for entry in entries]),
            "lang_mask": torch.stack([entry[1] for entry in entries]).bool(),
        }

    def get_batch(
        self,
        sample_ids: list[str],
        languages: Optional[list[str]] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        text_batch = self._get_text_batch(languages or [])
        if text_batch is not None:
            if sample_ids and all(self.has_sample(sample_id) for sample_id in sample_ids):
                mmap_img_batch = self._get_mmap_image_batch(sample_ids)
                if mmap_img_batch is not None:
                    text_batch.update(mmap_img_batch)
                else:
                    samples = [self._load_sample(sample_id) for sample_id in sample_ids]
                    if all("img_tokens" in sample for sample in samples):
                        text_batch["img_tokens"] = torch.stack([sample["img_tokens"] for sample in samples])
                    if all("img_tokens_q" in sample and "img_tokens_scale" in sample for sample in samples):
                        text_batch["img_tokens_q"] = torch.stack([sample["img_tokens_q"] for sample in samples])
                        text_batch["img_tokens_scale"] = torch.stack(
                            [sample["img_tokens_scale"] for sample in samples]
                        )
            return text_batch

        if not sample_ids or any(not self.has_sample(sample_id) for sample_id in sample_ids):
            return None

        mmap_img_batch = self._get_mmap_image_batch(sample_ids)
        if mmap_img_batch is not None:
            return mmap_img_batch

        samples = [self._load_sample(sample_id) for sample_id in sample_ids]
        batch = {}
        if all("lang_embeds" in sample for sample in samples):
            batch["lang_embeds"] = torch.stack([sample["lang_embeds"] for sample in samples])
        if all("lang_mask" in sample for sample in samples):
            batch["lang_mask"] = torch.stack([sample["lang_mask"] for sample in samples]).bool()
        if all("img_tokens" in sample for sample in samples):
            batch["img_tokens"] = torch.stack([sample["img_tokens"] for sample in samples])
        if all("img_tokens_q" in sample and "img_tokens_scale" in sample for sample in samples):
            batch["img_tokens_q"] = torch.stack([sample["img_tokens_q"] for sample in samples])
            batch["img_tokens_scale"] = torch.stack([sample["img_tokens_scale"] for sample in samples])
        return batch or None

    def get_stats(self) -> dict:
        mmap_files = sorted(self.precomputed_dir.glob("*img_tokens*.npy"))
        if mmap_files:
            feature_files = len(mmap_files)
            index_entries = len(self._index)
            storage_format = "mmap_npy"
        else:
            storage_format = self.metadata.get("storage_format", "unknown")
        shard_files = sorted(self.precomputed_dir.glob("*shard_*.pt"))
        if not mmap_files and shard_files:
            feature_files = len(shard_files)
            index_entries = len(self._index)
        elif not mmap_files:
            feature_files = len(self._index) if self._index else len(list(self.precomputed_dir.glob("*.pt")))
            index_entries = len(self._index) if self._index else feature_files
        return {
            "storage_format": storage_format,
            "feature_files": feature_files,
            "index_entries": index_entries,
            "cached_in_memory": len(self._memory_cache),
            "cached_shards": len(self._shard_cache),
            "cached_arrays": len(self._array_cache),
            "text_cache_entries": len(self._text_cache or {}),
            "preload_to_memory": self.preload_to_memory,
        }

    def validate_metadata(self, expected: dict) -> None:
        if not self.metadata:
            raise ValueError(
                f"Missing metadata.json in precomputed VL directory: {self.precomputed_dir}"
            )

        mismatches = []
        for key, expected_value in expected.items():
            actual_value = self.metadata.get(key)
            if actual_value != expected_value:
                mismatches.append((key, actual_value, expected_value))

        if mismatches:
            lines = [
                "Precomputed VL metadata mismatch detected.",
                f"Directory: {self.precomputed_dir}",
            ]
            for key, actual_value, expected_value in mismatches:
                lines.append(f"  {key}: actual={actual_value!r} expected={expected_value!r}")
            raise ValueError("\n".join(lines))
