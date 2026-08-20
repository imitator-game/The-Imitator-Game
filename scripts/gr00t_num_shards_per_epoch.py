#!/usr/bin/env python3
"""Compute a GR00T epoch shard count from LeRobot experiment configs.

GR00T's sharded dataset defines an internal epoch by num_shards_per_epoch.
This helper resolves the same sim/robot roots selected by launch_finetune.py
and returns the total number of generated shards, so one epoch is close to
one pass over the resolved training datasets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_training_entries(config_path: str | Path) -> list[dict[str, Any]]:
    with open(config_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {config_path}")
    return data


def parse_episode_indices(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [int(idx) for idx in raw]
    value = str(raw).strip()
    if not value:
        return None
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"Episode list must decode to a list: {raw}")
        return [int(idx) for idx in parsed]
    if ":" in value:
        start_s, end_s = value.split(":", 1)
        if not end_s:
            return None
        start = int(start_s) if start_s else 0
        end = int(end_s)
        return list(range(start, end))
    return [int(value)]


def resolve_human_task_id(
    task_id: str,
    task_to_human_map: dict[str, str],
    config_kind: str,
) -> str | None:
    if config_kind == "human":
        return task_id if task_id.startswith("human_") else task_to_human_map.get(task_id)
    if config_kind in {"sim", "robot"}:
        return task_to_human_map.get(task_id)
    raise ValueError(f"Unsupported config kind: {config_kind}")


def resolve_dataset_roots(
    dataset_parent: str | Path,
    human_config_path: str | Path,
    other_config_path: str | Path,
    other_kind: str,
    task_mapping_path: str | Path,
) -> list[dict[str, Any]]:
    with open(task_mapping_path, "r") as f:
        mapping_data = json.load(f)

    task_to_human_map: dict[str, str] = {}
    for mapping in mapping_data.get("task_mappings", []):
        human_task_id = mapping.get("human_task_id")
        if not human_task_id:
            continue
        task_to_human_map[str(human_task_id)] = str(human_task_id)
        for sim_task_id in mapping.get("sim_task_id", []):
            task_to_human_map[str(sim_task_id)] = str(human_task_id)
        for robot_task_id in mapping.get("robot_task_id", []):
            task_to_human_map[str(robot_task_id)] = str(human_task_id)

    human_entries = load_training_entries(human_config_path)
    other_entries = load_training_entries(other_config_path)

    human_task_ids = {
        resolved
        for entry in human_entries
        if (
            resolved := resolve_human_task_id(
                str(entry.get("repo_id") or entry.get("root")),
                task_to_human_map,
                "human",
            )
        )
    }
    other_task_ids = {
        resolved
        for entry in other_entries
        if (
            resolved := resolve_human_task_id(
                str(entry.get("repo_id") or entry.get("root")),
                task_to_human_map,
                other_kind,
            )
        )
    }
    intersected_human_task_ids = human_task_ids & other_task_ids
    if not intersected_human_task_ids:
        raise ValueError("No intersected tasks found between the provided config files")

    parent_dir = Path(dataset_parent)
    selected_roots: list[dict[str, Any]] = []
    for entry in other_entries:
        task_id = str(entry.get("repo_id") or entry.get("root"))
        resolved_human_task = resolve_human_task_id(task_id, task_to_human_map, other_kind)
        if resolved_human_task not in intersected_human_task_ids:
            continue

        root = str(entry.get("root") or entry.get("repo_id"))
        dataset_root = parent_dir / root
        if not (dataset_root / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"Resolved dataset path does not look like a LeRobot dataset: {dataset_root}"
            )
        selected_roots.append(
            {
                "path": dataset_root,
                "episode_indices": parse_episode_indices(entry.get("train")),
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for root_spec in selected_roots:
        key = json.dumps(
            {
                "path": str(root_spec["path"]),
                "episode_indices": root_spec["episode_indices"],
            },
            sort_keys=True,
        )
        deduped[key] = root_spec
    return list(deduped.values())


def read_info_meta(dataset_root: Path) -> dict[str, Any]:
    meta_dir = dataset_root / "meta"
    with open(meta_dir / "info.json", "r") as f:
        return json.load(f)


def resolve_lerobot_version(dataset_root: Path, info_meta: dict[str, Any], lerobot_version: str) -> str:
    meta_dir = dataset_root / "meta"

    resolved_version = lerobot_version.lower()
    if resolved_version == "auto":
        codebase_version = str(info_meta.get("codebase_version", "")).lower()
        if codebase_version.startswith("v3"):
            resolved_version = "v3"
        elif codebase_version.startswith("v2"):
            resolved_version = "v2"
        elif (meta_dir / "episodes.jsonl").exists():
            resolved_version = "v2"
        elif (meta_dir / "episodes").exists():
            resolved_version = "v3"
        else:
            resolved_version = "v2"
    return resolved_version


def read_episode_lengths(dataset_root: Path, lerobot_version: str) -> list[int]:
    meta_dir = dataset_root / "meta"
    info_meta = read_info_meta(dataset_root)
    resolved_version = resolve_lerobot_version(dataset_root, info_meta, lerobot_version)
    if resolved_version == "v2":
        episodes_path = meta_dir / "episodes.jsonl"
        with open(episodes_path, "r") as f:
            return [int(json.loads(line)["length"]) for line in f]

    if resolved_version == "v3":
        import pandas as pd

        episode_tables = sorted((meta_dir / "episodes").glob("chunk-*/file-*.parquet"))
        if not episode_tables:
            raise FileNotFoundError(f"No v3 episode parquet files found under {meta_dir / 'episodes'}")

        lengths: list[int] = []
        for table_path in episode_tables:
            table = pd.read_parquet(table_path, columns=["length"])
            lengths.extend(int(length) for length in table["length"].tolist())
        return lengths

    raise ValueError(f"Unsupported lerobot version: {lerobot_version}")


def read_effective_step_count(dataset_root: Path, lerobot_version: str, action_horizon: int) -> int:
    info_meta = read_info_meta(dataset_root)
    resolved_version = resolve_lerobot_version(dataset_root, info_meta, lerobot_version)

    if resolved_version == "v3":
        total_frames = info_meta.get("total_frames")
        total_episodes = info_meta.get("total_episodes")
        if total_frames is not None and total_episodes is not None:
            return max(0, int(total_frames) - int(total_episodes) * (action_horizon - 1))

    episode_lengths = read_episode_lengths(dataset_root, lerobot_version)
    return sum(max(0, length - action_horizon + 1) for length in episode_lengths)


def compute_total_shards(
    dataset_roots: list[dict[str, Any]],
    lerobot_version: str,
    shard_size: int,
    action_horizon: int,
    min_shards_per_dataset: int,
) -> int:
    total_shards = 0
    for root_spec in dataset_roots:
        dataset_root = Path(root_spec["path"])
        episode_indices = root_spec.get("episode_indices")
        if episode_indices is None:
            total_steps = read_effective_step_count(dataset_root, lerobot_version, action_horizon)
        else:
            episode_lengths = read_episode_lengths(dataset_root, lerobot_version)
            total_steps = sum(
                max(0, episode_lengths[int(idx)] - action_horizon + 1)
                for idx in episode_indices
            )
        if total_steps <= 0:
            raise ValueError(f"No valid effective steps found for {dataset_root}")
        total_shards += max(min_shards_per_dataset, math.ceil(total_steps / shard_size))
    return total_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-config-path", required=True)
    parser.add_argument("--sim-config-path")
    parser.add_argument("--robot-config-path")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--task-mapping-path",
        default="examples/baselines/lerobot_dataset/task_mapping.json",
    )
    parser.add_argument("--lerobot-version", default="v3", choices=["auto", "v2", "v3"])
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--min-shards-per-dataset", type=int, default=1)
    args = parser.parse_args()
    active_configs = [args.sim_config_path is not None, args.robot_config_path is not None]
    if sum(active_configs) != 1:
        parser.error("Exactly one of --sim-config-path or --robot-config-path must be provided")
    return args


if __name__ == "__main__":
    args = parse_args()
    roots = resolve_dataset_roots(
        dataset_parent=args.dataset_path,
        human_config_path=args.human_config_path,
        other_config_path=args.sim_config_path or args.robot_config_path,
        other_kind="sim" if args.sim_config_path is not None else "robot",
        task_mapping_path=args.task_mapping_path,
    )
    print(
        compute_total_shards(
            dataset_roots=roots,
            lerobot_version=args.lerobot_version,
            shard_size=args.shard_size,
            action_horizon=args.action_horizon,
            min_shards_per_dataset=args.min_shards_per_dataset,
        )
    )
