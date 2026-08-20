#!/usr/bin/env python3
"""Prepare GR00T LeRobot stats before launching parallel training jobs."""

from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.stats import LE_ROBOT_INFO_FILENAME, LE_ROBOT_REL_STATS_FILENAME, LE_ROBOT_STATS_FILENAME
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.data.types import ActionRepresentation
from gr00t_num_shards_per_epoch import resolve_dataset_roots


STAT_KEYS = ("mean", "std", "min", "max", "q01", "q99")
REL_STAT_KEYS = ("max", "min", "q01", "q99", "mean", "std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-config-path", action="append", required=True)
    parser.add_argument("--sim-config-path", action="append")
    parser.add_argument("--robot-config-path", action="append")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--task-mapping-path",
        default="examples/baselines/lerobot_dataset/task_mapping.json",
    )
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--lerobot-version", default="v3", choices=["auto", "v2", "v3"])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    sim_paths = args.sim_config_path or []
    robot_paths = args.robot_config_path or []
    if bool(sim_paths) == bool(robot_paths):
        parser.error("Exactly one of --sim-config-path or --robot-config-path must be provided")
    other_paths = sim_paths or robot_paths
    if len(args.human_config_path) != len(other_paths):
        parser.error(
            "--human-config-path and the selected non-human config path must be provided in pairs"
        )
    return args


def resolve_all_dataset_roots(args: argparse.Namespace) -> list[Path]:
    dataset_roots: list[Path] = []
    other_kind = "sim" if args.sim_config_path else "robot"
    other_config_paths = args.sim_config_path or args.robot_config_path
    assert other_config_paths is not None
    for human_config_path, other_config_path in zip(args.human_config_path, other_config_paths):
        dataset_roots.extend(
            Path(root_spec["path"])
            for root_spec in resolve_dataset_roots(
                dataset_parent=args.dataset_path,
                human_config_path=human_config_path,
                other_config_path=other_config_path,
                other_kind=other_kind,
                task_mapping_path=args.task_mapping_path,
            )
        )
    return list(dict.fromkeys(dataset_roots))


def load_json_object(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    if not isinstance(data, dict):
        return None, "json root is not an object"
    return data, None


def expected_lowdim_features(dataset_root: Path) -> tuple[list[str], list[str]]:
    info_path = dataset_root / LE_ROBOT_INFO_FILENAME
    info, error = load_json_object(info_path)
    if error is not None:
        return [], [f"{info_path}: {error}"]
    features = info.get("features", {})
    if not isinstance(features, dict):
        return [], [f"{info_path}: features is not an object"]
    return [
        feature
        for feature, feature_info in features.items()
        if isinstance(feature_info, dict) and "float" in str(feature_info.get("dtype", ""))
    ], []


def expected_relative_action_keys(embodiment_tag: EmbodimentTag) -> list[str]:
    action_config = MODALITY_CONFIGS[embodiment_tag.value]["action"]
    if action_config.action_configs is None:
        return []
    return [
        key
        for key, action_config in zip(action_config.modality_keys, action_config.action_configs)
        if action_config.rep == ActionRepresentation.RELATIVE
    ]


def validate_stats(dataset_root: Path, embodiment_tag: EmbodimentTag) -> list[str]:
    errors: list[str] = []
    lowdim_features, feature_errors = expected_lowdim_features(dataset_root)
    errors.extend(feature_errors)

    stats_path = dataset_root / LE_ROBOT_STATS_FILENAME
    stats, error = load_json_object(stats_path)
    if error is not None:
        errors.append(f"{stats_path}: {error}")
    elif stats is not None:
        for feature in lowdim_features:
            feature_stats = stats.get(feature)
            if not isinstance(feature_stats, dict):
                errors.append(f"{stats_path}: missing or invalid feature stats for {feature}")
                continue
            for stat_key in STAT_KEYS:
                stat_value = feature_stats.get(stat_key)
                if not isinstance(stat_value, list) or len(stat_value) == 0:
                    errors.append(f"{stats_path}: {feature}.{stat_key} is missing or empty")

    rel_action_keys = expected_relative_action_keys(embodiment_tag)
    if rel_action_keys:
        rel_stats_path = dataset_root / LE_ROBOT_REL_STATS_FILENAME
        rel_stats, rel_error = load_json_object(rel_stats_path)
        if rel_error is not None:
            errors.append(f"{rel_stats_path}: {rel_error}")
        elif rel_stats is not None:
            for action_key in rel_action_keys:
                action_stats = rel_stats.get(action_key)
                if not isinstance(action_stats, dict):
                    errors.append(f"{rel_stats_path}: missing or invalid relative stats for {action_key}")
                    continue
                for stat_key in REL_STAT_KEYS:
                    stat_value = action_stats.get(stat_key)
                    if not isinstance(stat_value, list) or len(stat_value) == 0:
                        errors.append(f"{rel_stats_path}: {action_key}.{stat_key} is missing or empty")

    return errors


def main() -> None:
    args = parse_args()
    dataset_roots = resolve_all_dataset_roots(args)
    embodiment_tag = EmbodimentTag[args.embodiment_tag]

    if not args.check_only:
        for dataset_root in dataset_roots:
            print(f"Preparing GR00T stats for {dataset_root}")
            generate_stats(dataset_root)
            generate_rel_stats(dataset_root, embodiment_tag, lerobot_version=args.lerobot_version)

    failed = False
    print(f"Checking GR00T stats for {len(dataset_roots)} dataset roots")
    for dataset_root in dataset_roots:
        errors = validate_stats(dataset_root, embodiment_tag)
        if errors:
            failed = True
            print(f"[FAIL] {dataset_root}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK] {dataset_root}")

    if failed:
        raise SystemExit("GR00T stats check failed")
    print("GR00T stats check passed")


if __name__ == "__main__":
    main()
