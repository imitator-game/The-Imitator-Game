#!/usr/bin/env python3

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


# ==================== Utility functions ====================

def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file"""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a JSON file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file"""
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a Parquet file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def get_hand_features_from_info(hand_info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract hand-related feature definitions from the hand dataset's info.json"""
    hand_features: Dict[str, Any] = {}
    for key, value in hand_info.get("features", {}).items():
        if "observation.hand" in key:
            hand_features[key] = value
    if not hand_features:
        raise ValueError("No hand features found in hand dataset info.json "
                         "(no key contains 'observation.hand').")
    return hand_features


def aggregate_stats(stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple stats dictionaries

    stats structure: {feature_key: {mean: [...], std: [...], min: [...], max: [...], ...}}
    """
    if not stats_list:
        return {}
    
    if len(stats_list) == 1:
        return stats_list[0]
    
    # Get all feature keys
    all_keys = set()
    for stats in stats_list:
        all_keys.update(stats.keys())
    
    merged_stats = {}
    for key in all_keys:
        # Collect the data for this key across all stats
        key_stats = [s.get(key, {}) for s in stats_list if key in s]
        if not key_stats:
            continue
        
        # Merge statistics (average the means, merge min/max, etc.)
        merged_key_stats = {}
        stat_types = ["mean", "std", "min", "max", "q01", "q99"]
        
        for stat_type in stat_types:
            if stat_type in key_stats[0]:
                # For mean/std: take the average (assuming each dataset has equal weight)
                # For min/max: take the global min/max
                values = [s[stat_type] for s in key_stats if stat_type in s]
                if values:
                    if stat_type in ["min"]:
                        merged_key_stats[stat_type] = np.minimum.reduce(values).tolist()
                    elif stat_type in ["max"]:
                        merged_key_stats[stat_type] = np.maximum.reduce(values).tolist()
                    else:
                        # Average mean, std, etc.
                        merged_key_stats[stat_type] = np.mean(values, axis=0).tolist()
        
        merged_stats[key] = merged_key_stats
    
    return merged_stats


def build_hand_index(hand_data_dir: Path) -> Dict[Tuple[int, int], int]:
    """
    Build an index for the hand dataset: (episode_index, frame_index) -> (chunk_idx, file_idx, row_idx)
    
    Returns: {(ep_idx, fr_idx): (chunk_idx, file_idx, row_idx)}
    """
    index: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    
    # Iterate over all data parquet files
    data_dir = hand_data_dir / "data"
    if not data_dir.exists():
        raise ValueError(f"Hand dataset data directory not found: {data_dir}")
    
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    
    for parquet_file in parquet_files:
        df = load_parquet(parquet_file)
        
        if "episode_index" not in df.columns or "frame_index" not in df.columns:
            continue
        
        # Extract chunk and file indices from the path
        parts = parquet_file.parts
        chunk_idx = None
        file_idx = None
        for i, part in enumerate(parts):
            if part.startswith("chunk-"):
                chunk_idx = int(part.split("-")[1])
            elif part.startswith("file-"):
                file_idx = int(part.split(".")[0].split("-")[1])
        
        if chunk_idx is None or file_idx is None:
            continue
        
        # Build the index
        for row_idx, (ep_idx, fr_idx) in enumerate(zip(df["episode_index"], df["frame_index"])):
            key = (int(ep_idx), int(fr_idx))
            if key in index:
                print(f"⚠️  Warning: Duplicate hand frame key {key}")
            index[key] = (chunk_idx, file_idx, row_idx)
    
    return index


def get_all_data_files(data_dir: Path) -> List[Tuple[int, int, Path]]:
    """Get all data parquet files, returning a list of (chunk_idx, file_idx, path)"""
    files = []
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    
    for parquet_file in parquet_files:
        parts = parquet_file.parts
        chunk_idx = None
        file_idx = None
        for part in parts:
            if part.startswith("chunk-"):
                chunk_idx = int(part.split("-")[1])
            elif part.startswith("file-"):
                file_idx = int(part.split(".")[0].split("-")[1])
        
        if chunk_idx is not None and file_idx is not None:
            files.append((chunk_idx, file_idx, parquet_file))
    
    return sorted(files)


# ==================== Main logic ====================

def concat_datasets(
    original_dataset_root: str,
    original_repo_id: str,
    hand_dataset_root: str,
    hand_repo_id: str,
    output_root: str,
    output_repo_id: str,
    episodes: Optional[List[int]] = None,
) -> None:
    print("=" * 80)
    print("🔄 Direct Concat Datasets (Frame-by-Frame)")
    print("=" * 80)
    
    original_dir = Path(original_dataset_root) / original_repo_id
    hand_dir = Path(hand_dataset_root) / hand_repo_id
    output_dir = Path(output_root) / output_repo_id
    
    # ---------- 1. Load info.json ----------
    print(f"\n📝 Loading info.json files...")
    original_info = load_json(original_dir / "meta" / "info.json")
    hand_info = load_json(hand_dir / "meta" / "info.json")
    
    hand_features = get_hand_features_from_info(hand_info)
    print(f"   ✓ Found {len(hand_features)} hand features to merge")
    
    # Merge features
    merged_features = original_info["features"].copy()
    for k in hand_features:
        if k in merged_features:
            raise ValueError(f"Hand feature '{k}' already exists in original features")
        merged_features[k] = hand_features[k]
    
    # Create the output info.json
    output_info = original_info.copy()
    output_info["features"] = merged_features
    output_info["total_frames"] = 0  # will be updated during processing
    output_info["total_episodes"] = 0
    
    # ---------- 2. Load stats.json ----------
    print(f"\n📊 Loading stats.json files...")
    original_stats_path = original_dir / "meta" / "stats.json"
    hand_stats_path = hand_dir / "meta" / "stats.json"
    
    original_stats = load_json(original_stats_path) if original_stats_path.exists() else {}
    hand_stats = load_json(hand_stats_path) if hand_stats_path.exists() else {}
    
    # Merge stats (only merge the stats of hand features)
    merged_stats = original_stats.copy()
    for key in hand_features.keys():
        if key in hand_stats:
            merged_stats[key] = hand_stats[key]
    
    # ---------- 3. Copy tasks.parquet ----------
    print(f"\n📋 Copying tasks.parquet...")
    original_tasks_path = original_dir / "meta" / "tasks.parquet"
    output_tasks_path = output_dir / "meta" / "tasks.parquet"
    if original_tasks_path.exists():
        output_tasks_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_tasks_path, output_tasks_path)
        print("   ✓ Tasks copied")
    else:
        print("   ⚠️  No tasks.parquet found")
    
    # ---------- 4. Build the hand index ----------
    print(f"\n🔍 Building hand frame index...")
    hand_index = build_hand_index(hand_dir)
    print(f"   ✓ Indexed {len(hand_index)} hand frames")
    
    # ---------- 5. Load episodes ----------
    print(f"\n📦 Loading episodes...")
    original_episodes_dir = original_dir / "meta" / "episodes"
    original_episode_files = sorted(original_episodes_dir.rglob("*.parquet"))
    
    # Load all original episodes
    original_episodes_list = []
    for ep_file in original_episode_files:
        df = load_parquet(ep_file)
        original_episodes_list.append(df)
    
    if original_episodes_list:
        original_episodes_df = pd.concat(original_episodes_list, ignore_index=True)
    else:
        raise ValueError("No episode files found in original dataset")
    
    # Filter episodes (if specified)
    if episodes is not None:
        episodes_set = set(episodes)
        original_episodes_df = original_episodes_df[original_episodes_df["episode_index"].isin(episodes_set)]
    
    print(f"   ✓ Loaded {len(original_episodes_df)} episodes")
    
    # ---------- 6. Process data files: concat by frame ----------
    print(f"\n🔄 Concatenating data files frame by frame...")
    
    original_data_dir = original_dir / "data"
    hand_data_dir = hand_dir / "data"
    output_data_dir = output_dir / "data"
    
    # Get all original data files (sorted by chunk and file)
    original_data_files = get_all_data_files(original_data_dir)
    
    # Preload all original data files (sorted by index)
    print("   📂 Loading all original data files...")
    all_original_frames = []
    for chunk_idx, file_idx, file_path in tqdm(original_data_files, desc="Loading files", leave=False):
        df = load_parquet(file_path)
        if "index" in df.columns:
            all_original_frames.append(df)
    
    if not all_original_frames:
        raise ValueError("No data files found in original dataset")
    
    # Concatenate all original frames and sort by index
    print("   🔗 Concatenating original frames...")
    all_original_df = pd.concat(all_original_frames, ignore_index=True)
    if "index" in all_original_df.columns:
        all_original_df = all_original_df.sort_values("index").reset_index(drop=True)
    
    # Filter the frames of the selected episodes
    if episodes is not None:
        episodes_set = set(episodes)
        all_original_df = all_original_df[all_original_df["episode_index"].isin(episodes_set)]
    
    print(f"   ✓ Loaded {len(all_original_df)} frames from original dataset")
    
    # Add hand features to each frame
    print("   🖐️  Adding hand features to frames...")
    hand_data_dict = {feat_key: [] for feat_key in hand_features.keys()}
    
    # Preload hand data files (cached)
    hand_file_cache = {}
    
    for _, row in tqdm(all_original_df.iterrows(), total=len(all_original_df), desc="Processing frames"):
        ep_val = int(row["episode_index"])
        fr_val = int(row["frame_index"])
        
        key = (ep_val, fr_val)
        if key in hand_index:
            # Read hand data
            chunk_idx, file_idx, row_idx = hand_index[key]
            cache_key = (chunk_idx, file_idx)
            
            if cache_key not in hand_file_cache:
                hand_file_path = hand_data_dir / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
                hand_file_cache[cache_key] = load_parquet(hand_file_path)
            
            hand_df = hand_file_cache[cache_key]
            hand_row = hand_df.iloc[row_idx]
            
            # Extract hand features
            for feat_key in hand_features.keys():
                if feat_key in hand_row:
                    hand_data_dict[feat_key].append(hand_row[feat_key])
                else:
                    # Fill with zeros
                    feat_info = hand_features[feat_key]
                    shape = tuple(feat_info["shape"])
                    dtype_str = feat_info.get("dtype", "float32")
                    dtype = np.float32 if dtype_str == "float32" else np.float64
                    hand_data_dict[feat_key].append(np.zeros(shape, dtype=dtype))
        else:
            # No hand data, fill with zeros
            for feat_key, feat_info in hand_features.items():
                shape = tuple(feat_info["shape"])
                dtype_str = feat_info.get("dtype", "float32")
                dtype = np.float32 if dtype_str == "float32" else np.float64
                hand_data_dict[feat_key].append(np.zeros(shape, dtype=dtype))
    
    # Add hand data to the dataframe
    for feat_key in hand_features.keys():
        all_original_df[feat_key] = hand_data_dict[feat_key]
    
    # Update the frame index (re-number starting from 0)
    all_original_df["index"] = range(len(all_original_df))
    
    # Save to output files (organized by chunk/file)
    print("   💾 Saving concatenated data files...")
    chunks_size = output_info.get("chunks_size", 100)
    data_files_size_mb = output_info.get("data_files_size_in_mb", 500)
    
    current_chunk_idx = 0
    current_file_idx = 0
    current_global_frame_idx = 0
    current_file_frames = []
    
    # Record the frame range of each file
    file_ranges = []  # [(chunk_idx, file_idx, start_idx, end_idx), ...]
    episode_start_indices = {}
    
    for idx, (_, row) in enumerate(tqdm(all_original_df.iterrows(), total=len(all_original_df), desc="Saving frames")):
        ep_idx = int(row["episode_index"])
        
        # Record the starting index of each episode
        if ep_idx not in episode_start_indices:
            episode_start_indices[ep_idx] = current_global_frame_idx + len(current_file_frames)
        
        current_file_frames.append(row.to_dict())
        
        # Check whether the current file should be saved
        frames_per_file = 10000  # number of frames stored per file
        if len(current_file_frames) >= frames_per_file:
            # Save the current file
            output_chunk_dir = output_data_dir / f"chunk-{current_chunk_idx:03d}"
            output_file_path = output_chunk_dir / f"file-{current_file_idx:03d}.parquet"
            
            file_df = pd.DataFrame(current_file_frames)
            save_parquet(file_df, output_file_path)
            
# Record the file range
            file_start_idx = current_global_frame_idx
            file_end_idx = current_global_frame_idx + len(current_file_frames)
            file_ranges.append((current_chunk_idx, current_file_idx, file_start_idx, file_end_idx))
            
            # Move to the next file
            current_global_frame_idx += len(current_file_frames)
            current_file_frames = []
            current_file_idx += 1
            
            if current_file_idx >= chunks_size:
                current_file_idx = 0
                current_chunk_idx += 1
    
    # Save the last file
    if current_file_frames:
        output_chunk_dir = output_data_dir / f"chunk-{current_chunk_idx:03d}"
        output_file_path = output_chunk_dir / f"file-{current_file_idx:03d}.parquet"
        file_df = pd.DataFrame(current_file_frames)
        save_parquet(file_df, output_file_path)
        
        # Record the file range
        file_start_idx = current_global_frame_idx
        file_end_idx = current_global_frame_idx + len(current_file_frames)
        file_ranges.append((current_chunk_idx, current_file_idx, file_start_idx, file_end_idx))
        current_global_frame_idx += len(current_file_frames)
    
    # Create output episode metadata
    print("   📝 Creating episode metadata...")
    output_episodes_list = []
    for _, ep_row in original_episodes_df.iterrows():
        ep_idx = int(ep_row["episode_index"])
        if ep_idx not in episode_start_indices:
            continue
        
        episode_length = int(ep_row["length"])
        dataset_from_idx = episode_start_indices[ep_idx]
        dataset_to_idx = dataset_from_idx + episode_length
        
        # Find the file that contains dataset_from_idx
        chunk_idx = 0
        file_idx = 0
        for c_idx, f_idx, start_idx, end_idx in file_ranges:
            if start_idx <= dataset_from_idx < end_idx:
                chunk_idx = c_idx
                file_idx = f_idx
                break
        
        output_ep_row = ep_row.copy()
        output_ep_row["data/chunk_index"] = chunk_idx
        output_ep_row["data/file_index"] = file_idx
        output_ep_row["dataset_from_index"] = dataset_from_idx
        output_ep_row["dataset_to_index"] = dataset_to_idx
        
        # Keep video information (if present)
        for col in ep_row.index:
            if col.startswith("videos/"):
                output_ep_row[col] = ep_row[col]
        
        output_episodes_list.append(output_ep_row.to_dict())
    
    # ---------- 7. Save episodes ----------
    print(f"\n💾 Saving episodes...")
    if output_episodes_list:
        output_episodes_df = pd.DataFrame(output_episodes_list)
        output_episodes_dir = output_dir / "meta" / "episodes" / "chunk-000"
        output_episodes_dir.mkdir(parents=True, exist_ok=True)
        save_parquet(output_episodes_df, output_episodes_dir / "file-000.parquet")
        print(f"   ✓ Saved {len(output_episodes_df)} episodes")
    
    # ---------- 8. Update and save info.json and stats.json ----------
    print(f"\n💾 Saving info.json and stats.json...")
    output_info["total_frames"] = len(all_original_df)
    output_info["total_episodes"] = len(output_episodes_list)
    
    # Add joint mapping documentation (OpenPose hand format)
    output_info["joint_mapping"] = {
        "description": "OpenPose hand format: 21 joints × 3D coordinates",
        "total_joints": 21,
        "mapping": {
            "0": "wrist",
            "1-4": "thumb (CMC, MCP, IP, TIP)",
            "5-8": "index (MCP, PIP, DIP, TIP)",
            "9-12": "middle (MCP, PIP, DIP, TIP)",
            "13-16": "ring (MCP, PIP, DIP, TIP)",
            "17-20": "pinky (MCP, PIP, DIP, TIP)"
        },
        "tip_indices": {
            "thumb_tip": 4,
            "index_tip": 8,
            "middle_tip": 12,
            "ring_tip": 16,
            "pinky_tip": 20
        }
    }
    
    save_json(output_info, output_dir / "meta" / "info.json")
    save_json(merged_stats, output_dir / "meta" / "stats.json")
    print("   ✓ Saved metadata files (with joint mapping info)")
    
    # ---------- 9. Copy the videos directory (if it exists) ----------
    print(f"\n🎥 Copying videos directory...")
    original_videos_dir = original_dir / "videos"
    output_videos_dir = output_dir / "videos"
    if original_videos_dir.exists():
        if output_videos_dir.exists():
            shutil.rmtree(output_videos_dir)
        shutil.copytree(original_videos_dir, output_videos_dir)
        print("   ✓ Videos copied")
    else:
        print("   ⚠️  No videos directory found")
    
    print(f"\n✅ Concat complete!")
    print(f"   - Total frames:      {len(all_original_df)}")
    print(f"   - Total episodes:    {len(output_episodes_list)}")
    print(f"   - Output dataset dir: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Direct concat datasets frame by frame (without LeRobot package)"
    )
    parser.add_argument(
        "--original_dataset_root",
        type=str,
        required=True,
        help="Root directory that contains the original repo",
    )
    parser.add_argument(
        "--original_repo_id",
        type=str,
        required=True,
        help="Repository ID of original dataset",
    )
    parser.add_argument(
        "--hand_dataset_root",
        type=str,
        required=True,
        help="Root directory that contains the hand repo",
    )
    parser.add_argument(
        "--hand_repo_id",
        type=str,
        required=True,
        help="Repository ID of hand dataset",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for output dataset",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        required=True,
        help="Repository ID for output dataset",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of episode indices to process",
    )
    
    args = parser.parse_args()
    
    concat_datasets(
        original_dataset_root=args.original_dataset_root,
        original_repo_id=args.original_repo_id,
        hand_dataset_root=args.hand_dataset_root,
        hand_repo_id=args.hand_repo_id,
        output_root=args.output_root,
        output_repo_id=args.output_repo_id,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()

