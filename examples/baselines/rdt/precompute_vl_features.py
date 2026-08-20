"""
Offline precompute script for RDT vision/language features.

This script mirrors the current RDT training input pipeline and exports
sharded feature files containing one of:
  - img_tokens
  - lang_embeds
  - lang_mask

`--feature_mode image_only` exports only img_tokens and is intended to be
combined with the existing language-only cache during training.

It also writes:
  - feature_index.json
  - metadata.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Subset
from tqdm import tqdm
import tyro

from examples.baselines.rdt.utils.precomputed_vl_metadata import build_precomputed_vl_expected_metadata


class FeatureEncoder:
    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self._lang_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.vision_encoder = None
        if args.feature_mode in ("vl", "image_only"):
            from examples.baselines.rdt.models.multimodal_encoder.siglip_encoder import SiglipVisionTower

            self.vision_encoder = SiglipVisionTower(vision_tower=args.vision_encoder, args=None)
            self.vision_encoder.vision_tower.to(device)
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        self.text_embedder = None
        if args.feature_mode in ("vl", "language_only"):
            from examples.baselines.rdt.models.multimodal_encoder.t5_encoder import T5Embedder

            self.text_embedder = T5Embedder(
                from_pretrained=args.text_encoder,
                model_max_length=args.max_lang_len,
                device=device,
                local_files_only=True,
            )
            self.text_embedder.model.eval()
            for param in self.text_embedder.model.parameters():
                param.requires_grad = False

    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def encode_images_batch(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.vision_encoder is None:
            raise RuntimeError("Vision encoder is not initialized in language_only feature mode.")
        bsz = rgb.shape[0]
        rgb_flat = rgb.flatten(end_dim=1).to(memory_format=torch.channels_last)
        if rgb_flat.shape[-2:] != (384, 384):
            rgb_flat = F.interpolate(
                rgb_flat, size=(384, 384), mode="bicubic", align_corners=False, antialias=False
            )
        img_tokens = self.vision_encoder(rgb_flat)
        return img_tokens.reshape(bsz, -1, img_tokens.shape[-1])

    def encode_language(self, lang_texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_embedder is None:
            raise RuntimeError("Text encoder is not initialized in image_only feature mode.")
        miss_texts: list[str] = []
        miss_positions: list[int] = []
        embeds_out: list[Optional[torch.Tensor]] = [None] * len(lang_texts)
        masks_out: list[Optional[torch.Tensor]] = [None] * len(lang_texts)

        for i, text in enumerate(lang_texts):
            cached = self._lang_cache.get(text)
            if cached is not None:
                embeds_out[i], masks_out[i] = cached
            else:
                miss_texts.append(text)
                miss_positions.append(i)

        if miss_texts:
            unique_texts: list[str] = []
            unique_order: dict[str, int] = {}
            for text in miss_texts:
                if text not in unique_order:
                    unique_order[text] = len(unique_texts)
                    unique_texts.append(text)

            with torch.no_grad():
                unique_embeds, unique_masks = self.text_embedder.get_text_embeddings(unique_texts)
            unique_embeds = unique_embeds.float()
            unique_masks = unique_masks.bool()

            for idx, text in enumerate(unique_texts):
                cached = (unique_embeds[idx].detach().cpu(), unique_masks[idx].detach().cpu())
                self._lang_cache[text] = cached

            for pos, text in zip(miss_positions, miss_texts):
                embeds_out[pos], masks_out[pos] = self._lang_cache[text]

        return (
            torch.stack([x for x in embeds_out if x is not None]).to(self.device),
            torch.stack([x for x in masks_out if x is not None]).to(self.device),
        )


def _resolve_save_dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "int8": torch.int8,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported save_dtype: {name}")
    return mapping[value]


def _cast_feature_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == torch.bool:
        return tensor
    return tensor.to(dtype=dtype)


def _pad_or_truncate_lang_feature(
    embedding: torch.Tensor,
    mask: torch.Tensor,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = embedding.shape[0]
    if seq_len > max_length:
        return embedding[:max_length], mask[:max_length]
    if seq_len < max_length:
        pad_len = max_length - seq_len
        embedding = torch.cat(
            [
                embedding,
                torch.zeros(pad_len, embedding.shape[1], dtype=embedding.dtype, device=embedding.device),
            ],
            dim=0,
        )
        mask = torch.cat(
            [
                mask,
                torch.zeros(pad_len, dtype=torch.bool, device=mask.device),
            ],
            dim=0,
        )
    return embedding, mask.bool()


def _paired_language_records(paired_dataset) -> list[tuple[str, str]]:
    sim_idx_to_task: dict[int, str] = {}
    for sim_task_id, indices in paired_dataset.task_to_sim_indices.items():
        for actual_idx in indices:
            sim_idx_to_task[int(actual_idx)] = sim_task_id

    records: list[tuple[str, str]] = []
    for actual_idx in paired_dataset.valid_indices:
        actual_idx = int(actual_idx)
        sim_task_id = sim_idx_to_task.get(actual_idx)
        if sim_task_id is None:
            continue

        human_task_id = paired_dataset.task_mapper.get_human_task_from_sim(sim_task_id)
        if human_task_id is None:
            continue

        language = paired_dataset.task_mapper.get_description("human", human_task_id)
        if language is None:
            language = paired_dataset.task_mapper.get_description("sim", sim_task_id)
        if language is None:
            language = f"Task {human_task_id}"

        records.append((f"{sim_task_id}::{actual_idx}", language))

    return records


def _parse_episode_selection(selection: str | None) -> Optional[set[int]]:
    if selection is None or str(selection).strip() == "":
        return None

    selected: set[int] = set()
    for part in str(selection).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_s, end_s = part.split(":", 1)
            start = int(start_s) if start_s.strip() else 0
            end = int(end_s) if end_s.strip() else start
            selected.update(range(start, end))
        else:
            selected.add(int(part))
    return selected


def _load_episode_lengths(repo_dir: Path, selected_episodes: Optional[set[int]]) -> int:
    candidates = [
        repo_dir / "meta" / "episodes.jsonl",
        repo_dir / "meta" / "episodes.json",
    ]

    episodes: list[dict] = []
    if candidates[0].exists():
        with candidates[0].open("r", encoding="utf-8") as f:
            episodes = [json.loads(line) for line in f if line.strip()]
    elif candidates[1].exists():
        with candidates[1].open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            episodes = raw
        elif isinstance(raw, dict):
            episodes = raw.get("episodes", [])

    if not episodes:
        raise FileNotFoundError(
            f"Could not find LeRobot episode metadata under {repo_dir}/meta. "
            "language_only fast export needs meta/episodes.jsonl or meta/episodes.json "
            "to avoid constructing the full dataset."
        )

    total = 0
    for default_idx, episode in enumerate(episodes):
        ep_idx = int(episode.get("episode_index", episode.get("index", default_idx)))
        if selected_episodes is not None and ep_idx not in selected_episodes:
            continue
        length = (
            episode.get("length")
            or episode.get("num_frames")
            or episode.get("episode_length")
        )
        if length is None:
            raise KeyError(f"Episode metadata missing length field in {repo_dir}: {episode}")
        total += int(length)
    return total


def _first_description(desc_data: dict, task_id: str) -> Optional[str]:
    desc = desc_data.get(task_id)
    if isinstance(desc, list):
        for item in desc:
            if isinstance(item, str) and item.strip():
                return item
        return None
    if isinstance(desc, str) and desc.strip():
        return desc
    return None


def _all_descriptions(desc_data: dict, task_id: str) -> list[str]:
    desc = desc_data.get(task_id)
    if isinstance(desc, list):
        return [item for item in desc if isinstance(item, str) and item.strip()]
    if isinstance(desc, str) and desc.strip():
        return [desc]
    return []


def _fast_paired_language_records_from_configs(args) -> list[tuple[str, str]]:
    with open(args.lerobot_sim_dataset_file, "r", encoding="utf-8") as f:
        sim_cfgs = json.load(f)
    with open(args.lerobot_human_dataset_file, "r", encoding="utf-8") as f:
        human_cfgs = json.load(f)
    with open(args.lerobot_task_mapping_file, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    with open(args.lerobot_human_task_description_file, "r", encoding="utf-8") as f:
        human_desc = json.load(f)

    sim_to_human: dict[str, str] = {}
    for mapping in mapping_data.get("task_mappings", []):
        human_task_id = mapping.get("human_task_id")
        for sim_task_id in mapping.get("sim_task_id", []):
            sim_to_human[str(sim_task_id)] = str(human_task_id)

    loaded_human_tasks = {str(cfg.get("repo_id", "")) for cfg in human_cfgs}
    records: list[tuple[str, str]] = []
    cumulative_idx = 0
    sim_root = Path(args.lerobot_sim_root or args.lerobot_root or "demos")

    for cfg in sim_cfgs:
        sim_task_id = str(cfg.get("repo_id", ""))
        root_name = str(cfg.get("root", sim_task_id))
        repo_dir = sim_root / root_name
        selected_episodes = _parse_episode_selection(cfg.get("train"))
        num_transitions = _load_episode_lengths(repo_dir, selected_episodes)

        human_task_id = sim_to_human.get(sim_task_id)
        if human_task_id is not None and human_task_id in loaded_human_tasks:
            language = _first_description(human_desc, human_task_id)
            if language is None:
                language = f"Task {human_task_id}"
            for offset in range(num_transitions):
                records.append((f"{sim_task_id}::{cumulative_idx + offset}", language))

        cumulative_idx += num_transitions

    print(
        f"Fast language-only export generated {len(records)} sample records "
        f"from {len(sim_cfgs)} sim repos without constructing LeRobot datasets."
    )
    return records


def _fast_paired_language_texts_from_configs(args) -> list[str]:
    with open(args.lerobot_human_dataset_file, "r", encoding="utf-8") as f:
        human_cfgs = json.load(f)
    with open(args.lerobot_human_task_description_file, "r", encoding="utf-8") as f:
        human_desc = json.load(f)

    loaded_human_tasks = [str(cfg.get("repo_id", "")) for cfg in human_cfgs]
    texts: list[str] = []
    seen: set[str] = set()
    for human_task_id in loaded_human_tasks:
        for text in _all_descriptions(human_desc, human_task_id):
            if text not in seen:
                seen.add(text)
                texts.append(text)
    print(f"Fast language-only export collected {len(texts)} unique descriptions.")
    return texts


def _export_language_text_cache(
    texts: list[str],
    encoder: FeatureEncoder,
    output_dir: Path,
    batch_size: int,
    save_dtype: torch.dtype,
    max_length: int,
) -> None:
    all_embeds: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Precomputing language descriptions"):
        chunk = texts[start:start + batch_size]
        with torch.no_grad():
            lang_embeds, lang_mask = encoder.encode_language(chunk)
        for embed, mask in zip(lang_embeds, lang_mask):
            embed, mask = _pad_or_truncate_lang_feature(
                embed.detach().cpu(),
                mask.detach().cpu().bool(),
                max_length,
            )
            all_embeds.append(_cast_feature_tensor(embed, save_dtype))
            all_masks.append(mask)

    payload = {
        "texts": texts,
        "lang_embeds": torch.stack(all_embeds) if all_embeds else torch.empty(0),
        "lang_mask": torch.stack(all_masks).bool() if all_masks else torch.empty(0, dtype=torch.bool),
    }
    torch.save(payload, output_dir / "lang_features.pt")


def _export_language_records(
    records: list[tuple[str, str]],
    encoder: FeatureEncoder,
    writer: "ShardedFeatureWriter",
    batch_size: int,
) -> None:
    for start in tqdm(range(0, len(records), batch_size), desc="Precomputing language features"):
        chunk = records[start:start + batch_size]
        sample_ids = [sample_id for sample_id, _ in chunk]
        lang_texts = [language for _, language in chunk]
        with torch.no_grad():
            lang_embeds, lang_mask = encoder.encode_language(lang_texts)
        for i, sample_id in enumerate(sample_ids):
            writer.add(
                sample_id=sample_id,
                lang_embeds=lang_embeds[i],
                lang_mask=lang_mask[i],
            )


def _sample_ids_from_dataset(dataset) -> list[str]:
    if isinstance(dataset, Subset):
        base_sample_ids = _sample_ids_from_dataset(dataset.dataset)
        return [base_sample_ids[int(idx)] for idx in dataset.indices]

    if hasattr(dataset, "valid_indices") and hasattr(dataset, "task_to_sim_indices"):
        sim_idx_to_task: dict[int, str] = {}
        for sim_task_id, indices in dataset.task_to_sim_indices.items():
            for actual_idx in indices:
                sim_idx_to_task[int(actual_idx)] = str(sim_task_id)

        sample_ids: list[str] = []
        for actual_idx in dataset.valid_indices:
            actual_idx = int(actual_idx)
            sim_task_id = sim_idx_to_task.get(actual_idx)
            if sim_task_id is None:
                raise KeyError(f"Could not resolve sim task for dataset index {actual_idx}")
            sample_ids.append(f"{sim_task_id}::{actual_idx}")
        return sample_ids

    raise ValueError(
        "Cannot derive sample_ids without iterating dataset items for this dataset type. "
        "mmap index rebuild currently supports paired LeRobot datasets."
    )


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


class ShardedFeatureWriter:
    def __init__(
        self,
        output_dir: Path,
        shard_size: int,
        save_dtype: torch.dtype,
        *,
        filename_prefix: str = "",
    ):
        self.output_dir = output_dir
        self.shard_size = max(1, int(shard_size))
        self.save_dtype = save_dtype
        self.filename_prefix = filename_prefix
        self.feature_index: dict[str, dict[str, object]] = {}
        self._buffer: list[dict[str, object]] = []
        self._shard_id = 0
        self._total = 0

    def add(
        self,
        sample_id: str,
        lang_embeds: Optional[torch.Tensor] = None,
        lang_mask: Optional[torch.Tensor] = None,
        img_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        item = {
            "sample_id": sample_id,
        }
        if lang_embeds is not None:
            item["lang_embeds"] = _cast_feature_tensor(lang_embeds.detach().cpu(), self.save_dtype)
        if lang_mask is not None:
            item["lang_mask"] = lang_mask.detach().cpu().bool()
        if img_tokens is not None:
            item["img_tokens"] = _cast_feature_tensor(img_tokens.detach().cpu(), self.save_dtype)
        self._buffer.append(item)
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        shard_name = f"{self.filename_prefix}shard_{self._shard_id:05d}.pt"
        shard_path = self.output_dir / shard_name
        payload = {
            "sample_ids": [item["sample_id"] for item in self._buffer],
        }
        if all("lang_embeds" in item for item in self._buffer):
            payload["lang_embeds"] = torch.stack([item["lang_embeds"] for item in self._buffer])
        if all("lang_mask" in item for item in self._buffer):
            payload["lang_mask"] = torch.stack([item["lang_mask"] for item in self._buffer]).bool()
        if all("img_tokens" in item for item in self._buffer):
            payload["img_tokens"] = torch.stack([item["img_tokens"] for item in self._buffer])
        torch.save(payload, shard_path)
        for offset, item in enumerate(self._buffer):
            self.feature_index[str(item["sample_id"])] = {"path": shard_name, "offset": offset}
        self._total += len(self._buffer)
        self._buffer.clear()
        self._shard_id += 1

    def finalize(self) -> tuple[int, int]:
        self.flush()
        return self._total, self._shard_id


class MmapImageFeatureWriter:
    """Write image-only features into one row-addressable .npy memmap file."""

    def __init__(
        self,
        output_dir: Path,
        save_dtype: torch.dtype,
        *,
        capacity: int,
        filename_prefix: str = "",
    ):
        self.output_dir = output_dir
        self.save_dtype = save_dtype
        self.capacity = int(capacity)
        self.filename_prefix = filename_prefix
        self.quantized = save_dtype == torch.int8
        self.array_name = (
            f"{filename_prefix}img_tokens_int8.npy"
            if self.quantized
            else f"{filename_prefix}img_tokens.npy"
        )
        self.scale_array_name = f"{filename_prefix}img_tokens_scale.npy" if self.quantized else None
        self.feature_index: dict[str, dict[str, object]] = {}
        self._array: Optional[np.memmap] = None
        self._scale_array: Optional[np.memmap] = None
        self._total = 0

    def _np_dtype(self) -> np.dtype:
        if self.save_dtype == torch.int8:
            return np.dtype("int8")
        if self.save_dtype == torch.float16:
            return np.dtype("float16")
        if self.save_dtype == torch.float32:
            return np.dtype("float32")
        raise ValueError(
            "mmap image cache supports int8, float16/fp16, or float32/fp32 save_dtype only; "
            f"got {self.save_dtype}"
        )

    def _ensure_array(self, sample_shape: tuple[int, ...]) -> None:
        if self._array is not None:
            return
        if self.capacity <= 0:
            raise ValueError("mmap writer capacity must be positive")
        array_path = self.output_dir / self.array_name
        self._array = np.lib.format.open_memmap(
            array_path,
            mode="w+",
            dtype=self._np_dtype(),
            shape=(self.capacity, *sample_shape),
        )
        if self.quantized:
            scale_path = self.output_dir / str(self.scale_array_name)
            self._scale_array = np.lib.format.open_memmap(
                scale_path,
                mode="w+",
                dtype=np.dtype("float16"),
                shape=(self.capacity, sample_shape[-1]),
            )

    def add(
        self,
        sample_id: str,
        lang_embeds: Optional[torch.Tensor] = None,
        lang_mask: Optional[torch.Tensor] = None,
        img_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        if lang_embeds is not None or lang_mask is not None:
            raise ValueError("MmapImageFeatureWriter only supports image_only features")
        if img_tokens is None:
            raise ValueError("MmapImageFeatureWriter requires img_tokens")
        if self._total >= self.capacity:
            raise RuntimeError(
                f"mmap writer capacity exceeded: capacity={self.capacity} sample_id={sample_id}"
            )

        tensor = img_tokens.detach().cpu().contiguous()
        self._ensure_array(tuple(tensor.shape))
        assert self._array is not None
        if self.quantized:
            assert self._scale_array is not None
            tensor_f = tensor.float()
            scale = (tensor_f.abs().amax(dim=0) / 127.0).clamp_min(1e-8)
            q = torch.round(tensor_f / scale).clamp(-127, 127).to(torch.int8)
            self._array[self._total] = q.numpy()
            self._scale_array[self._total] = scale.to(torch.float16).numpy()
            self.feature_index[str(sample_id)] = {
                "array": self.array_name,
                "offset": self._total,
                "quantization": "int8_per_channel",
                "scale_array": self.scale_array_name,
            }
        else:
            tensor = _cast_feature_tensor(tensor, self.save_dtype)
            self._array[self._total] = tensor.numpy()
            self.feature_index[str(sample_id)] = {"array": self.array_name, "offset": self._total}
        self._total += 1

    def finalize(self) -> tuple[int, int]:
        if self._array is not None:
            self._array.flush()
        if self._scale_array is not None:
            self._scale_array.flush()
        return self._total, 1 if self._array is not None else 0


@dataclass
class Args:
    output_dir: str
    batch_size: int = 64
    num_dataload_workers: int = 8
    device: str = "cuda"
    feature_mode: str = "vl"
    cache_format: str = "sharded_pt"
    shard_size: int = 2048
    save_dtype: str = "float16"
    dataset_num_shards: int = 1
    dataset_shard_id: int = 0
    rebuild_mmap_index_only: bool = False

    env_id: str = "PickCubeYCB-v1"
    demo_path: str = "demos/PickCubeYCB-v1/motionplanning/multi_task_4.rgbd.pd_joint_delta_pos.physx_cpu.h5"
    num_demos: Optional[int] = None
    obs_mode: str = "rgb"
    control_mode: str = "pd_joint_pos"
    reward_mode: str = "dense"
    max_episode_steps: int = 600
    sim_backend: str = "physx_cpu"
    shader: str = "rt-fast"
    obs_horizon: int = 2
    pred_horizon: int = 16

    vision_encoder: str = "/path/to/google--siglip-so400m-patch14-384"
    text_encoder: str = "/path/to/google--t5-v1_1-xxl"
    t5_version: str = "t5-v1_1-xxl"
    max_lang_len: int = 1024

    use_lerobot: bool = False
    lerobot_repo_id: Optional[str] = None
    lerobot_root: Optional[str] = None
    lerobot_cameras: Tuple[str, ...] = ("zed2i",)
    lerobot_image_size: Tuple[int, int] = (224, 224)
    lerobot_state_type: str = "qpos"
    lerobot_include_depth: bool = False
    lerobot_depth_mode: str = "sim"
    lerobot_video_backend: str = "torchcodec"
    lerobot_tolerance_s: float = 0.05
    lerobot_task_description_file: Optional[str] = None
    lerobot_use_paired_dataset: bool = False
    lerobot_human_root: Optional[str] = "/path/to/human_root"
    lerobot_sim_root: Optional[str] = "/path/to/sim_root"
    lerobot_task_mapping_file: str = "examples/baselines/lerobot_dataset/task_mapping.json"
    lerobot_human_dataset_file: str = "examples/baselines/lerobot_dataset/config/human_config.json"
    lerobot_sim_dataset_file: str = "examples/baselines/lerobot_dataset/config/sim_config.json"
    lerobot_human_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    lerobot_sim_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"


def main():
    args = tyro.cli(Args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_num_shards < 1:
        raise ValueError(f"dataset_num_shards must be >= 1, got {args.dataset_num_shards}")
    if not 0 <= args.dataset_shard_id < args.dataset_num_shards:
        raise ValueError(
            f"dataset_shard_id must be in [0, {args.dataset_num_shards}), got {args.dataset_shard_id}"
        )
    if args.feature_mode not in ("vl", "language_only", "image_only"):
        raise ValueError(
            f"feature_mode must be 'vl', 'language_only', or 'image_only', got {args.feature_mode}"
        )
    if args.cache_format not in ("sharded_pt", "mmap"):
        raise ValueError(
            f"cache_format must be 'sharded_pt' or 'mmap', got {args.cache_format}"
        )
    if args.cache_format == "mmap" and args.feature_mode != "image_only":
        raise ValueError("cache_format='mmap' is currently supported only with feature_mode='image_only'")
    if args.rebuild_mmap_index_only and args.cache_format != "mmap":
        raise ValueError("rebuild_mmap_index_only requires cache_format='mmap'")

    fast_paired_language_only = (
        args.feature_mode == "language_only"
        and args.use_lerobot
        and args.lerobot_use_paired_dataset
    )
    if not fast_paired_language_only:
        from examples.baselines.rdt.train_rdt_scratch import (
            RDTSimplifiedDataset,
            _extract_base_env_id,
            _extract_l_level_from_task_id,
            _set_l_level_flags,
            collate_fn_cpu,
            collate_fn_lerobot,
        )
        from examples.baselines.rdt.utils.utils import (
            build_state_obs_extractor,
            convert_obs,
            worker_init_fn,
        )

        requested_level = _extract_l_level_from_task_id(args.env_id)
        _set_l_level_flags(requested_level)
        base_env_id = _extract_base_env_id(args.env_id)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    include_rgb = True
    include_depth = False

    if args.use_lerobot:
        if args.feature_mode == "language_only" and args.lerobot_use_paired_dataset:
            language_texts = _fast_paired_language_texts_from_configs(args)
            language_records = None
            dataset = None
            collate = None
        elif args.lerobot_use_paired_dataset:
            language_texts = None
            from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
                HumanSimPairedDataset,
                PairedDatasetConfig,
            )

            paired_config = PairedDatasetConfig(
                human_root=args.lerobot_human_root or args.lerobot_root or "demos",
                sim_root=args.lerobot_sim_root or args.lerobot_root or "demos",
                task_mapping_file=args.lerobot_task_mapping_file,
                human_dataset_file=args.lerobot_human_dataset_file,
                sim_dataset_file=args.lerobot_sim_dataset_file,
                human_task_description_file=args.lerobot_human_task_description_file,
                sim_task_description_file=args.lerobot_sim_task_description_file,
                split="train",
                cameras=["zed2i"],
                include_depth=include_depth,
                image_size=tuple(args.lerobot_image_size),
                horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                state_type=args.lerobot_state_type,
                fps=30,
                video_backend=args.lerobot_video_backend,
                input_mode="language_only",
                include_first_frame=False,
            )
            dataset = HumanSimPairedDataset(paired_config)
            sim_repo_ids = tuple(ds.repo_id for ds in dataset.sim_dataset.lerobot_dataset.datasets)
            sim_task_descriptions = dict(dataset.sim_dataset.task_descriptions)
            language_records = None
        else:
            language_texts = None
            from examples.baselines.lerobot_dataset.lerobot_dataloader import LeRobotDataConfig, build_lerobot_dataset

            lerobot_config = LeRobotDataConfig(
                source_type="sim",
                root=args.lerobot_root or "./data/lerobot",
                repo_id=args.lerobot_repo_id,
                split="train",
                image_size=tuple(args.lerobot_image_size),
                state_type=args.lerobot_state_type,
                include_depth=include_depth,
                cameras=["zed2i"],
                horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                tolerance_s=args.lerobot_tolerance_s,
                depth_mode=args.lerobot_depth_mode,
                video_backend=args.lerobot_video_backend,
                task_description_file=args.lerobot_task_description_file,
            )
            dataset = build_lerobot_dataset(lerobot_config)
            sim_repo_ids = None
            sim_task_descriptions = dict(getattr(dataset, "task_descriptions", {}))
            language_records = None

        if args.feature_mode != "language_only" or not args.lerobot_use_paired_dataset:
            collate = partial(
                collate_fn_lerobot,
                use_paired_dataset=args.lerobot_use_paired_dataset,
                sim_repo_ids=sim_repo_ids,
                sim_task_descriptions=sim_task_descriptions,
                precomputed_vl=None,
            )
    else:
        language_texts = None
        language_records = None
        env_kwargs = dict(
            control_mode=args.control_mode,
            reward_mode=args.reward_mode,
            obs_mode=args.obs_mode,
            render_mode="rgb_array",
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            max_episode_steps=args.max_episode_steps,
        )
        tmp_env = gym.make(base_env_id, **env_kwargs)
        original_obs_space = tmp_env.observation_space
        include_rgb = "rgb" in args.obs_mode
        include_depth = "depth" in args.obs_mode
        tmp_env.close()

        obs_process_fn = partial(
            convert_obs,
            concat_fn=partial(build_state_obs_extractor(base_env_id)),
        )
        dataset = RDTSimplifiedDataset(
            demo_path=args.demo_path,
            obs_process_fn=obs_process_fn,
            obs_space=original_obs_space,
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
            include_rgb=include_rgb,
            include_depth=include_depth,
            num_traj=args.num_demos,
        )
        collate = partial(collate_fn_cpu, precomputed_vl=None)

    if language_texts is None and language_records is None and args.dataset_num_shards > 1:
        shard_indices = list(range(args.dataset_shard_id, len(dataset), args.dataset_num_shards))
        print(
            f"Dataset export shard {args.dataset_shard_id}/{args.dataset_num_shards}: "
            f"{len(shard_indices)} / {len(dataset)} samples"
        )
        dataset = Subset(dataset, shard_indices)

    if args.rebuild_mmap_index_only:
        sample_ids = _sample_ids_from_dataset(dataset)
        writer_prefix = (
            f"part{args.dataset_shard_id:03d}_"
            if args.dataset_num_shards > 1 else ""
        )
        int8_array_name = f"{writer_prefix}img_tokens_int8.npy"
        fp_array_name = f"{writer_prefix}img_tokens.npy"
        int8_array_path = out_dir / int8_array_name
        fp_array_path = out_dir / fp_array_name
        if int8_array_path.exists():
            array_name = int8_array_name
            array_path = int8_array_path
            quantized = True
            scale_array_name = f"{writer_prefix}img_tokens_scale.npy"
            scale_array_path = out_dir / scale_array_name
            if not scale_array_path.exists():
                raise FileNotFoundError(f"Missing int8 scale array: {scale_array_path}")
        elif fp_array_path.exists():
            array_name = fp_array_name
            array_path = fp_array_path
            quantized = False
            scale_array_name = None
        else:
            raise FileNotFoundError(
                f"Could not find mmap image array {int8_array_path} or {fp_array_path}"
            )

        array = np.load(array_path, mmap_mode="r")
        if array.shape[0] != len(sample_ids):
            raise ValueError(
                "mmap array row count does not match dataset sample count: "
                f"array_rows={array.shape[0]} sample_ids={len(sample_ids)} path={array_path}"
            )

        feature_index: dict[str, dict[str, object]] = {}
        for offset, sample_id in enumerate(sample_ids):
            entry: dict[str, object] = {"array": array_name, "offset": offset}
            if quantized:
                entry["quantization"] = "int8_per_channel"
                entry["scale_array"] = scale_array_name
            feature_index[str(sample_id)] = entry

        index_name = (
            f"feature_index_part{args.dataset_shard_id:03d}.json"
            if args.dataset_num_shards > 1 else "feature_index.json"
        )
        _atomic_write_json(out_dir / index_name, feature_index)
        print(
            f"Rebuilt mmap feature index with {len(feature_index)} entries: {out_dir / index_name}",
            flush=True,
        )
        return

    dataloader = None
    if language_texts is None and language_records is None:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_dataload_workers,
            worker_init_fn=partial(worker_init_fn, base_seed=1),
            persistent_workers=(args.num_dataload_workers > 0),
            collate_fn=collate,
            pin_memory=True,
            multiprocessing_context="forkserver" if args.num_dataload_workers > 0 else None,
        )

    encoder = FeatureEncoder(args, device)
    writer_prefix = (
        f"part{args.dataset_shard_id:03d}_"
        if args.dataset_num_shards > 1 else ""
    )
    save_dtype = _resolve_save_dtype(args.save_dtype)
    if args.cache_format == "mmap":
        if dataloader is None or dataset is None:
            raise ValueError("mmap cache export requires a concrete dataset/dataloader")
        writer = MmapImageFeatureWriter(
            out_dir,
            save_dtype=save_dtype,
            capacity=len(dataset),
            filename_prefix=writer_prefix,
        )
    else:
        writer = ShardedFeatureWriter(
            out_dir,
            shard_size=args.shard_size,
            save_dtype=save_dtype,
            filename_prefix=writer_prefix,
        )

    metadata = build_precomputed_vl_expected_metadata(args)
    metadata["generator"] = "examples.baselines.rdt.precompute_vl_features"
    metadata["feature_mode"] = args.feature_mode
    metadata["storage_format"] = "mmap_npy" if args.cache_format == "mmap" else "sharded_pt"
    metadata["shard_size"] = args.shard_size
    metadata["save_dtype"] = args.save_dtype
    metadata["dataset_num_shards"] = args.dataset_num_shards
    metadata["language_description_policy"] = (
        "all_human_descriptions_text_cache" if args.feature_mode == "language_only" else "dataset_runtime"
    )
    metadata["augmentation_note"] = (
        "Exporter matches current training preprocessing, but random augmentation during training "
        "can still make online features differ from precomputed features."
    )
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if language_texts is not None:
        _export_language_text_cache(
            language_texts,
            encoder,
            out_dir,
            args.batch_size,
            save_dtype,
            args.max_lang_len,
        )
    elif language_records is not None:
        _export_language_records(language_records, encoder, writer, args.batch_size)
    else:
        for batch in tqdm(dataloader, desc="Precomputing V/L features"):
            sample_ids = [str(x) for x in batch.get("sample_id", [])]
            observations = batch["observations"]
            lang_texts = batch["language"]

            with torch.no_grad():
                lang_embeds = None
                lang_mask = None
                if args.feature_mode != "image_only":
                    lang_embeds, lang_mask = encoder.encode_language(lang_texts)
                img_tokens = None
                if args.feature_mode in ("vl", "image_only") and include_rgb and "rgb" in observations:
                    rgb = observations["rgb"].float()
                    if not args.use_lerobot:
                        rgb = rgb / 255.0
                    img_tokens = encoder.encode_images_batch(rgb.to(device))
                elif args.feature_mode in ("vl", "image_only"):
                    batch_size = len(sample_ids)
                    img_tokens = torch.zeros(
                        batch_size,
                        args.obs_horizon * encoder.vision_encoder.num_patches,
                        encoder.vision_encoder.hidden_size,
                        device=device,
                    )

            for i, sample_id in enumerate(sample_ids):
                writer.add(
                    sample_id=str(sample_id),
                    lang_embeds=lang_embeds[i] if lang_embeds is not None else None,
                    lang_mask=lang_mask[i] if lang_mask is not None else None,
                    img_tokens=img_tokens[i] if img_tokens is not None else None,
                )

    print("Finished feature encode loop; finalizing feature cache...", flush=True)
    total_samples, total_shards = writer.finalize()
    index_name = (
        f"feature_index_part{args.dataset_shard_id:03d}.json"
        if args.dataset_num_shards > 1 else "feature_index.json"
    )
    _atomic_write_json(out_dir / index_name, writer.feature_index)
    print(f"Saved feature index: {out_dir / index_name}", flush=True)

    if language_texts is not None:
        print(
            f"Saved {len(language_texts)} unique language feature entries to "
            f"{out_dir / 'lang_features.pt'}"
        )
    elif args.cache_format == "mmap":
        print(
            f"Saved {total_samples} mmap image feature entries into {total_shards} array(s) at {out_dir}"
        )
    else:
        print(
            f"Saved {total_samples} precomputed feature entries into {total_shards} shard(s) at {out_dir}"
        )


if __name__ == "__main__":
    main()
