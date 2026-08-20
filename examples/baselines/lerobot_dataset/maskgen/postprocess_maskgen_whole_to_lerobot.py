#!/usr/bin/env python3
"""Postprocess mask_gen_whole outputs into additional LeRobot video features.

This script creates a new LeRobot dataset root and keeps original data untouched:
1) Copies the full source dataset tree to a new root.
2) Adds new mask video features under `videos/observation.images.*_mask/`.
3) Updates `meta/info.json` and `meta/episodes/*.parquet` with new feature columns.

Supported mask files produced by mask_gen_whole:
- .npy (raw uint8 mask)
- .npz (contains key "mask")
- .npz packbits format (keys: encoding=packbits_little, shape, packed)
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
import re
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

THIS_DIR = Path(__file__).resolve().parent
LEROBOT_DIR = THIS_DIR.parent
if str(LEROBOT_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(LEROBOT_DIR))


INSTANCE_RE = re.compile(r"instance_(\d+)_mask\.(npy|npz)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mask_gen_whole masks into additional LeRobot video features."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Single-task root (human_H1) or dataset root (imitator_human_v1)")
    parser.add_argument("--mask-root", type=Path, required=True, help="Single-task mask root (human_H1) or mask dataset root containing task dirs")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New LeRobot root to create. Source root is never modified.",
    )
    parser.add_argument(
        "--feature-suffix",
        type=str,
        default="_mask",
        help="Suffix appended to each RGB video key, e.g. observation.images.cam1 -> observation.images.cam1_mask",
    )
    parser.add_argument(
        "--video-keys",
        type=str,
        default="",
        help="Optional comma-separated mask video keys to include, e.g. observation.images.cam1,observation.images.cam2",
    )
    parser.add_argument(
        "--episode-ids",
        type=str,
        default="",
        help="Optional comma-separated episode ids to process, e.g. 0,1,2. Default processes all.",
    )
    parser.add_argument(
        "--missing-policy",
        type=str,
        default="error",
        choices=["error", "zero"],
        help="When mask jsonl is missing for an episode/key: error or fill zero masks.",
    )
    parser.add_argument(
        "--save-label-map",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save label maps under meta/mask_labels/.",
    )
    parser.add_argument(
        "--save-episode-label-map",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-episode label map json files. Default is off (global map only).",
    )
    parser.add_argument(
        "--mask-vcodec",
        type=str,
        default="libx264rgb",
        help="Codec used for encoded mask videos.",
    )
    parser.add_argument(
        "--mask-pix-fmt",
        type=str,
        default="rgb24",
        help="Pixel format used for encoded mask videos.",
    )
    parser.add_argument(
        "--mask-crf",
        type=int,
        default=0,
        help="CRF used for encoded mask videos (0 is lossless for libx264rgb).",
    )
    parser.add_argument(
        "--mask-preset",
        type=str,
        default="ultrafast",
        help="Encoder preset for mask videos.",
    )
    parser.add_argument(
        "--skip-existing-mask-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If output mask video already exists, skip re-encoding and only update metadata.",
    )
    parser.add_argument(
        "--save-vis-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save a colorized visualization mp4 for each mask video.",
    )
    parser.add_argument(
        "--vis-video-suffix",
        type=str,
        default=".vis",
        help="Suffix inserted before .mp4 for visualization videos. Example: file-000.vis.mp4",
    )
    parser.add_argument(
        "--vis-vcodec",
        type=str,
        default="libx264",
        help="Codec for visualization video.",
    )
    parser.add_argument(
        "--vis-pix-fmt",
        type=str,
        default="yuv420p",
        help="Pixel format for visualization video.",
    )
    parser.add_argument(
        "--vis-crf",
        type=int,
        default=18,
        help="CRF for visualization video.",
    )
    parser.add_argument(
        "--vis-preset",
        type=str,
        default="veryfast",
        help="Encoder preset for visualization video.",
    )
    parser.add_argument(
        "--vis-legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay color legend (id->label) on visualization video.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow existing output-root only if empty; otherwise raises.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output-root by copying missing files and continuing processing.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "single", "batch"],
        help="auto: detect from source-root; single: one task; batch: all task subdirs under source-root.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default="",
        help="Batch mode only. Optional comma-separated task dirs to process, e.g. human_H1,human_H2",
    )
    parser.add_argument(
        "--task-list-file",
        type=Path,
        default=None,
        help="Batch mode only. Optional text file containing one task id per line.",
    )
    parser.add_argument(
        "--select-tasks-from-mask-root",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Batch mode only. Restrict processing to task directory names present "
            "under mask-root. Useful with a curated/symlink-based staging root."
        ),
    )
    parser.add_argument(
        "--copy-selected-tasks-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Batch mode only. Copy only selected task roots into output-root "
            "instead of copying the entire source-root dataset tree."
        ),
    )
    parser.add_argument(
        "--copy-source-task-prefix",
        type=str,
        default="",
        help=(
            "Batch mode only. Copy every source task directory whose name starts "
            "with this prefix, while processing masks only for the selected tasks. "
            "Example: robot_."
        ),
    )
    parser.add_argument(
        "--require-complete-mask-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Batch mode only. Add mask features only to tasks for which every "
            "source episode/RGB-camera pair has a non-empty maskgen JSONL."
        ),
    )
    parser.add_argument(
        "--skip-missing-mask-task",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Batch mode only. Skip task if mask task dir is missing.",
    )
    parser.add_argument(
        "--on-error",
        type=str,
        default="skip",
        choices=["skip", "raise"],
        help="Error handling strategy during processing. skip records and continues; raise aborts.",
    )
    parser.add_argument(
        "--error-report",
        type=Path,
        default=None,
        help="Optional path to save skipped-item report jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Batch mode only. Audit task selection and mask completeness, print "
            "the copy/process plan, and exit without writing output files."
        ),
    )
    return parser.parse_args()


def _read_info(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_info(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _is_depth_feature(key: str, ft: dict) -> bool:
    key_lower = key.lower()
    return bool(ft.get("info", {}).get("video.is_depth_map", False)) or (
        "depth" in key_lower or key_lower.endswith("_depth")
    )


def _discover_rgb_keys(info: dict) -> list[str]:
    keys: list[str] = []
    for key, ft in info.get("features", {}).items():
        if not key.startswith("observation.images."):
            continue
        if key.endswith("_mask"):
            continue
        if ft.get("dtype") != "video":
            continue
        if _is_depth_feature(key, ft):
            continue
        keys.append(key)
    return sorted(keys)


def _discover_mask_keys(mask_root: Path) -> list[str]:
    episode_dirs = sorted([p for p in mask_root.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_* dirs found under {mask_root}")
    keys = sorted(
        {
            path.name
            for episode_dir in episode_dirs
            for path in episode_dir.iterdir()
            if path.is_dir()
        }
    )
    if not keys:
        raise FileNotFoundError(f"No video_key dirs found under {mask_root}")
    return keys


def _prepare_output_tree(source_root: Path, output_root: Path, overwrite: bool, resume: bool) -> None:
    if output_root.exists():
        if overwrite:
            if any(output_root.iterdir()):
                raise FileExistsError(f"Output root is not empty: {output_root}")
            _copy_tree_with_progress(source_root, output_root, resume=False)
            return
        if not resume:
            raise FileExistsError(f"Output root already exists: {output_root}")
        print(f"[resume] output root exists, continue from: {output_root}")
        _copy_tree_with_progress(source_root, output_root, resume=True)
        return
    _copy_tree_with_progress(source_root, output_root, resume=False)


def _needs_copy(src: Path, dst: Path) -> bool:
    if src.is_symlink():
        if not dst.exists() and not dst.is_symlink():
            return True
        if not dst.is_symlink():
            return True
        try:
            return os.readlink(src) != os.readlink(dst)
        except OSError:
            return True
    if not dst.exists():
        return True
    if dst.is_symlink():
        return True
    try:
        return int(src.stat().st_size) != int(dst.stat().st_size)
    except FileNotFoundError:
        return True


def _count_tree(src_root: Path, dst_root: Path | None = None, resume: bool = False) -> tuple[int, int]:
    total_files = 0
    total_bytes = 0
    for root, _, files in os.walk(src_root):
        root_p = Path(root)
        rel = root_p.relative_to(src_root)
        for name in files:
            p = root_p / name
            if resume and dst_root is not None:
                dst = dst_root / rel / name
                if not _needs_copy(p, dst):
                    continue
            try:
                if p.is_symlink():
                    total_files += 1
                    continue
                st = p.stat()
                total_files += 1
                total_bytes += int(st.st_size)
            except FileNotFoundError:
                continue
    return total_files, total_bytes


def _copy_tree_with_progress(src_root: Path, dst_root: Path, resume: bool) -> None:
    total_files, total_bytes = _count_tree(src_root, dst_root=dst_root, resume=resume)
    dst_root.mkdir(parents=True, exist_ok=resume)

    bytes_bar = None
    file_bar = None
    if tqdm is not None:
        bytes_bar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Copy bytes")
        file_bar = tqdm(total=total_files, desc="Copy files", leave=False)

    try:
        for root, dirs, files in os.walk(src_root):
            root_p = Path(root)
            rel = root_p.relative_to(src_root)
            dst_dir = dst_root / rel
            dst_dir.mkdir(parents=True, exist_ok=True)
            for d in dirs:
                (dst_dir / d).mkdir(parents=True, exist_ok=True)

            for name in files:
                src_path = root_p / name
                dst_path = dst_dir / name
                if resume and not _needs_copy(src_path, dst_path):
                    continue
                if src_path.is_symlink():
                    if dst_path.exists() or dst_path.is_symlink():
                        dst_path.unlink()
                    target = os.readlink(src_path)
                    dst_path.symlink_to(target)
                    if file_bar is not None:
                        file_bar.update(1)
                    continue

                size = int(src_path.stat().st_size)
                shutil.copy2(src_path, dst_path)
                if bytes_bar is not None:
                    bytes_bar.update(size)
                if file_bar is not None:
                    file_bar.update(1)
    finally:
        if bytes_bar is not None:
            bytes_bar.close()
        if file_bar is not None:
            file_bar.close()


def _decode_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        if "encoding" in data and str(data["encoding"][0]) == "packbits_little":
            shape = tuple(int(x) for x in data["shape"].tolist())
            packed = data["packed"]
            total = int(np.prod(shape))
            flat = np.unpackbits(packed, bitorder="little", count=total)
            arr = flat.reshape(shape)
        elif "mask" in data:
            arr = data["mask"]
        else:
            first = list(data.keys())[0]
            arr = data[first]
    else:
        raise ValueError(f"Unsupported mask suffix: {path.suffix} ({path})")

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeeze, got {arr.shape} for {path}")
    return (arr > 0).astype(np.uint8)


def _parse_instance_id(mask_rel_path: str) -> int:
    name = Path(mask_rel_path).name
    m = INSTANCE_RE.match(name)
    if m is None:
        raise ValueError(f"Invalid mask filename format: {mask_rel_path}")
    return int(m.group(1))


def _load_jsonl_records(jsonl_path: Path) -> list[dict]:
    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _normalize_label(label: str) -> str:
    return " ".join(str(label).strip().split())


def _record_label(rec: dict) -> str:
    label = rec.get("label", "")
    return _normalize_label(label)


def _build_frame_index(records: list[dict]) -> tuple[dict[int, list[dict]], dict[int, str]]:
    by_frame: dict[int, list[dict]] = {}
    label_map: dict[int, str] = {}
    for rec in records:
        frame_idx = int(rec["frame_idx"])
        inst_id = _parse_instance_id(rec["mask_path"])
        by_frame.setdefault(frame_idx, []).append(rec)
        label_map.setdefault(inst_id, str(rec.get("label", f"obj_{inst_id}")))

    for frame_idx in by_frame:
        by_frame[frame_idx].sort(key=lambda x: _parse_instance_id(x["mask_path"]))
    return by_frame, label_map


def _infer_hw(records: list[dict], key_dir: Path) -> tuple[int, int]:
    if not records:
        raise ValueError("Cannot infer mask shape from empty records.")
    sample = key_dir / records[0]["mask_path"]
    mask = _decode_mask(sample)
    return int(mask.shape[0]), int(mask.shape[1])


def _render_episode_mask_frames(
    key_dir: Path,
    records: list[dict],
    episode_len: int,
    missing_policy: str,
) -> tuple[dict[int, str], int, int, dict[int, list[dict]]]:
    by_frame, label_map = _build_frame_index(records)
    h, w = _infer_hw(records, key_dir) if records else (0, 0)

    if not records and missing_policy == "error":
        raise FileNotFoundError(f"No mask records found for {key_dir}")
    if not records and missing_policy == "zero":
        raise ValueError("zero fill requires at least one frame to infer H/W; mask records are empty")
    return label_map, h, w, by_frame


def _encode_instance_id_video(
    key_dir: Path,
    by_frame: dict[int, list[dict]],
    episode_len: int,
    h: int,
    w: int,
    out_video_abs: Path,
    fps: float,
    vcodec: str,
    pix_fmt: str,
    crf: int,
    preset: str,
    missing_policy: str,
    frame_desc: str,
    vis_video_abs: Path | None = None,
    vis_vcodec: str = "libx264",
    vis_pix_fmt: str = "yuv420p",
    vis_crf: int = 18,
    vis_preset: str = "veryfast",
) -> None:
    out_video_abs.parent.mkdir(parents=True, exist_ok=True)
    options = {"preset": preset}
    if crf is not None:
        options["crf"] = str(crf)
    with av.open(str(out_video_abs), mode="w") as output:
        stream = output.add_stream(vcodec, rate=int(round(fps)), options=options)
        stream.width = int(w)
        stream.height = int(h)
        stream.pix_fmt = pix_fmt

        vis_output = None
        vis_stream = None
        if vis_video_abs is not None:
            vis_video_abs.parent.mkdir(parents=True, exist_ok=True)
            vis_options = {"preset": vis_preset}
            if vis_crf is not None:
                vis_options["crf"] = str(vis_crf)
            vis_output = av.open(str(vis_video_abs), mode="w")
            vis_stream = vis_output.add_stream(
                vis_vcodec, rate=int(round(fps)), options=vis_options
            )
            vis_stream.width = int(w)
            vis_stream.height = int(h)
            vis_stream.pix_fmt = vis_pix_fmt

        frame_iter = range(episode_len)
        if tqdm is not None:
            frame_iter = tqdm(frame_iter, desc=frame_desc, leave=False)

        for frame_idx in frame_iter:
            frame_records = by_frame.get(frame_idx, [])
            if not frame_records and missing_policy == "error":
                raise ValueError(f"Missing frame {frame_idx} in {key_dir}")

            canvas = np.zeros((h, w), dtype=np.uint8)
            for rec in frame_records:
                inst_id = _parse_instance_id(rec["mask_path"])
                if inst_id > 255:
                    raise ValueError(f"Instance id {inst_id} exceeds uint8 range in {key_dir}")
                mask_path = key_dir / rec["mask_path"]
                mask = _decode_mask(mask_path)
                if mask.shape != (h, w):
                    raise ValueError(f"Mask shape mismatch in {mask_path}: {mask.shape} vs {(h, w)}")
                canvas[mask > 0] = inst_id

            rgb = np.stack([canvas, canvas, canvas], axis=-1)
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(video_frame):
                output.mux(packet)
            if vis_stream is not None:
                vis_rgb = _colorize_instance_map(canvas)
                vis_frame = av.VideoFrame.from_ndarray(vis_rgb, format="rgb24")
                for packet in vis_stream.encode(vis_frame):
                    vis_output.mux(packet)

        for packet in stream.encode():
            output.mux(packet)
        if vis_stream is not None:
            for packet in vis_stream.encode():
                vis_output.mux(packet)
            vis_output.close()


def _render_canvas_from_records(
    key_dir: Path,
    frame_records: list[dict],
    h: int,
    w: int,
    label_to_global_id: dict[str, int],
) -> np.ndarray:
    canvas = np.zeros((h, w), dtype=np.uint8)
    for rec in frame_records:
        label = _record_label(rec)
        if not label:
            raise ValueError(f"Missing label in record for {key_dir}: {rec}")
        if label not in label_to_global_id:
            raise ValueError(f"Label not found in global map for {key_dir}: '{label}'")
        global_id = int(label_to_global_id[label])
        if global_id > 255:
            raise ValueError(f"Global id {global_id} exceeds uint8 range in {key_dir}")
        mask_path = key_dir / rec["mask_path"]
        mask = _decode_mask(mask_path)
        if mask.shape != (h, w):
            raise ValueError(f"Mask shape mismatch in {mask_path}: {mask.shape} vs {(h, w)}")
        canvas[mask > 0] = global_id
    return canvas


def _encode_group_video(
    frame_records_by_global_idx: dict[int, tuple[Path, list[dict], bool]],
    total_frames: int,
    h: int,
    w: int,
    out_video_abs: Path,
    fps: float,
    vcodec: str,
    pix_fmt: str,
    crf: int,
    preset: str,
    frame_desc: str,
    vis_video_abs: Path | None = None,
    vis_vcodec: str = "libx264",
    vis_pix_fmt: str = "yuv420p",
    vis_crf: int = 18,
    vis_preset: str = "veryfast",
    label_to_global_id: dict[str, int] | None = None,
    global_id_to_label: dict[int, str] | None = None,
    vis_legend: bool = True,
) -> None:
    out_video_abs.parent.mkdir(parents=True, exist_ok=True)
    out_video_tmp = out_video_abs.with_name(
        f".{out_video_abs.stem}.{os.getpid()}.tmp{out_video_abs.suffix}"
    )
    temp_paths = [out_video_tmp]
    vis_video_tmp = None
    if vis_video_abs is not None:
        vis_video_abs.parent.mkdir(parents=True, exist_ok=True)
        vis_video_tmp = vis_video_abs.with_name(
            f".{vis_video_abs.stem}.{os.getpid()}.tmp{vis_video_abs.suffix}"
        )
        temp_paths.append(vis_video_tmp)
    for path in temp_paths:
        path.unlink(missing_ok=True)

    options = {"preset": preset}
    if crf is not None:
        options["crf"] = str(crf)
    try:
        with ExitStack() as stack:
            output = stack.enter_context(av.open(str(out_video_tmp), mode="w"))
            stream = output.add_stream(vcodec, rate=int(round(fps)), options=options)
            stream.width = int(w)
            stream.height = int(h)
            stream.pix_fmt = pix_fmt

            vis_output = None
            vis_stream = None
            if vis_video_tmp is not None:
                vis_options = {"preset": vis_preset}
                if vis_crf is not None:
                    vis_options["crf"] = str(vis_crf)
                vis_output = stack.enter_context(av.open(str(vis_video_tmp), mode="w"))
                vis_stream = vis_output.add_stream(
                    vis_vcodec, rate=int(round(fps)), options=vis_options
                )
                vis_stream.width = int(w)
                vis_stream.height = int(h)
                vis_stream.pix_fmt = vis_pix_fmt

            frame_iter = range(total_frames)
            if tqdm is not None:
                frame_iter = tqdm(frame_iter, desc=frame_desc, leave=False)

            for frame_idx in frame_iter:
                if frame_idx in frame_records_by_global_idx:
                    key_dir, frame_records, must_exist = frame_records_by_global_idx[frame_idx]
                    if must_exist and not frame_records:
                        raise ValueError(
                            f"Missing frame records at global frame {frame_idx} in {key_dir}"
                        )
                    if label_to_global_id is None:
                        raise ValueError("label_to_global_id is required for encoding mask videos")
                    canvas = _render_canvas_from_records(
                        key_dir, frame_records, h, w, label_to_global_id
                    )
                else:
                    canvas = np.zeros((h, w), dtype=np.uint8)

                rgb = np.stack([canvas, canvas, canvas], axis=-1)
                video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(video_frame):
                    output.mux(packet)
                if vis_stream is not None:
                    vis_rgb = _colorize_instance_map(canvas)
                    if vis_legend and global_id_to_label is not None:
                        present_ids = sorted(int(i) for i in np.unique(canvas) if int(i) != 0)
                        vis_rgb = _overlay_legend(vis_rgb, global_id_to_label, present_ids)
                    vis_frame = av.VideoFrame.from_ndarray(vis_rgb, format="rgb24")
                    for packet in vis_stream.encode(vis_frame):
                        vis_output.mux(packet)

            for packet in stream.encode():
                output.mux(packet)
            if vis_stream is not None:
                for packet in vis_stream.encode():
                    vis_output.mux(packet)

        os.replace(out_video_tmp, out_video_abs)
        if vis_video_tmp is not None and vis_video_abs is not None:
            os.replace(vis_video_tmp, vis_video_abs)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)


def _color_for_instance(instance_id: int) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 128, 255],
        ],
        dtype=np.uint8,
    )
    return palette[instance_id % len(palette)]


def _colorize_instance_map(canvas: np.ndarray) -> np.ndarray:
    out = np.zeros((canvas.shape[0], canvas.shape[1], 3), dtype=np.uint8)
    ids = np.unique(canvas)
    for instance_id in ids:
        if instance_id == 0:
            continue
        out[canvas == instance_id] = _color_for_instance(int(instance_id))
    return out


def _overlay_legend(
    vis_rgb: np.ndarray,
    id_to_label: dict[int, str],
    present_ids: list[int],
) -> np.ndarray:
    if not present_ids:
        return vis_rgb
    img = Image.fromarray(vis_rgb)
    draw = ImageDraw.Draw(img, mode="RGBA")
    font = ImageFont.load_default()

    x0 = 8
    y0 = 8
    line_h = 16
    box = 10
    pad = 6
    max_text = 0
    texts = []
    for gid in present_ids:
        label = id_to_label.get(gid, f"id_{gid}")
        text = f"{gid}: {label}"
        texts.append((gid, text))
        max_text = max(max_text, len(text))
    panel_w = pad * 3 + box + max_text * 7
    panel_h = pad * 2 + line_h * len(texts)
    draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], fill=(0, 0, 0, 140))

    y = y0 + pad
    for gid, text in texts:
        c = tuple(int(v) for v in _color_for_instance(gid))
        draw.rectangle([x0 + pad, y + 3, x0 + pad + box, y + 3 + box], fill=(*c, 255))
        draw.text((x0 + pad * 2 + box, y), text, font=font, fill=(255, 255, 255, 255))
        y += line_h
    return np.asarray(img)


def _update_info_with_mask_features(
    info: dict,
    src_keys: list[str],
    suffix: str,
    vcodec: str,
    pix_fmt: str,
) -> dict[str, str]:
    key_map: dict[str, str] = {}
    for src_key in src_keys:
        dst_key = f"{src_key}{suffix}"
        if dst_key in info["features"]:
            key_map[src_key] = dst_key
            continue
        src_ft = info["features"][src_key]
        src_info = dict(src_ft.get("info", {}))
        src_info["video.is_depth_map"] = False
        src_info["video.channels"] = 3
        src_info["video.codec"] = vcodec
        src_info["video.pix_fmt"] = pix_fmt
        info["features"][dst_key] = {
            "dtype": "video",
            "shape": list(src_ft["shape"]),
            "names": list(src_ft.get("names", ["channels", "height", "width"])),
            "info": src_info,
        }
        key_map[src_key] = dst_key
    return key_map


def _episode_file_indices(episode_idx: int, chunks_size: int) -> tuple[int, int]:
    chunk_idx = episode_idx // chunks_size
    file_idx = episode_idx % chunks_size
    return chunk_idx, file_idx


def _timestamp_to_frame_index(ts: float, fps: float) -> int:
    # Use round (not floor) to avoid boundary collisions when timestamps are
    # represented as decimal strings like 8.333333 at 30fps.
    return int(round(float(ts) * float(fps)))


def _write_label_map(
    output_root: Path,
    task_name: str,
    episode_idx: int,
    key: str,
    label_map: dict[int, str],
) -> None:
    path = output_root / "meta" / "mask_labels" / task_name / key / f"episode_{episode_idx:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {str(k): label_map[k] for k in sorted(label_map)}
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


def _write_global_label_map(
    output_root: Path,
    task_name: str,
    key: str,
    id_to_label: dict[int, str],
) -> None:
    path = output_root / "meta" / "mask_labels" / task_name / key / "global.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {str(k): id_to_label[k] for k in sorted(id_to_label)}
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


def _has_task_signature(task_root: Path) -> bool:
    return (task_root / "meta" / "info.json").exists() and (task_root / "meta" / "episodes").exists()


def _process_single_task(
    args: argparse.Namespace, output_root: Path, mask_root: Path, task_name: str
) -> list[dict[str, str]]:
    skipped_items: list[dict[str, str]] = []
    info_path = output_root / "meta" / "info.json"
    info = _read_info(info_path)
    fps = float(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))

    available_mask_keys = _discover_mask_keys(mask_root)
    source_rgb_keys = _discover_rgb_keys(info)
    if args.video_keys:
        target_keys = [k.strip() for k in args.video_keys.split(",") if k.strip()]
    elif args.missing_policy == "zero":
        # A partially completed maskgen task may have no directory at all for
        # one camera. The source metadata is authoritative for release layout;
        # missing episode-camera pairs are represented by all-zero videos.
        target_keys = source_rgb_keys
    else:
        target_keys = available_mask_keys

    for key in target_keys:
        if key not in source_rgb_keys:
            raise ValueError(f"mask key is not a source RGB key in info.json: {key}")
        if key not in available_mask_keys and args.missing_policy == "error":
            raise ValueError(f"mask key not found under mask-root: {key}")

    key_map = _update_info_with_mask_features(
        info,
        target_keys,
        args.feature_suffix,
        vcodec=args.mask_vcodec,
        pix_fmt=args.mask_pix_fmt,
    )

    episodes_files = sorted((output_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episodes_files:
        raise FileNotFoundError(f"No episodes parquet files under {output_root / 'meta' / 'episodes'}")
    selected_eps: set[int] | None = None
    if args.episode_ids:
        selected_eps = {int(x.strip()) for x in args.episode_ids.split(",") if x.strip()}

    # Load all episode tables once. We need global grouping by source chunk/file.
    episode_tables: list[tuple[Path, pd.DataFrame]] = []
    ep_file_iter = episodes_files if tqdm is None else tqdm(
        episodes_files, desc=f"{task_name}: load-episode-files", leave=False
    )
    for ep_file in ep_file_iter:
        episode_tables.append((ep_file, pd.read_parquet(ep_file)))

    # Prepare metadata columns for new mask keys by copying source camera mappings.
    for src_key in target_keys:
        dst_key = key_map[src_key]
        src_cols = {
            "chunk": f"videos/{src_key}/chunk_index",
            "file": f"videos/{src_key}/file_index",
            "from": f"videos/{src_key}/from_timestamp",
            "to": f"videos/{src_key}/to_timestamp",
        }
        dst_cols = {
            "chunk": f"videos/{dst_key}/chunk_index",
            "file": f"videos/{dst_key}/file_index",
            "from": f"videos/{dst_key}/from_timestamp",
            "to": f"videos/{dst_key}/to_timestamp",
        }
        for _, df in episode_tables:
            df[dst_cols["chunk"]] = df[src_cols["chunk"]]
            df[dst_cols["file"]] = df[src_cols["file"]]
            df[dst_cols["from"]] = df[src_cols["from"]]
            df[dst_cols["to"]] = df[src_cols["to"]]

    key_iter = target_keys if tqdm is None else tqdm(
        target_keys, desc=f"{task_name}: cameras", leave=False
    )
    for src_key in key_iter:
        try:
            dst_key = key_map[src_key]
            src_h = int(info["features"][src_key]["shape"][1])
            src_w = int(info["features"][src_key]["shape"][2])
            group_to_episodes: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
            # (chunk,file) -> list of (episode_idx, episode_len, start_frame)
            for _, df in episode_tables:
                for _, row in df.iterrows():
                    ep_idx = int(row["episode_index"])
                    if selected_eps is not None and ep_idx not in selected_eps:
                        continue
                    ep_len = int(row["length"])
                    chunk_idx = int(row[f"videos/{src_key}/chunk_index"])
                    file_idx = int(row[f"videos/{src_key}/file_index"])
                    start_frame = _timestamp_to_frame_index(
                        float(row[f"videos/{src_key}/from_timestamp"]), fps
                    )
                    group_to_episodes.setdefault((chunk_idx, file_idx), []).append(
                        (ep_idx, ep_len, start_frame)
                    )

            # Build task-level global label map for this camera key.
            unique_eps = sorted({ep for eps in group_to_episodes.values() for ep, _, _ in eps})
            episode_cache: dict[int, tuple[Path, list[dict], dict[int, list[dict]]]] = {}
            invalid_eps: set[int] = set()
            labels_set: set[str] = set()
            eps_iter = unique_eps if tqdm is None else tqdm(
                unique_eps, desc=f"{task_name}/{src_key}: build-global-map", leave=False
            )
            for ep_idx in eps_iter:
                try:
                    episode_dir = mask_root / f"episode_{ep_idx:03d}" / src_key
                    jsonl_path = episode_dir / f"episode_{ep_idx:03d}.jsonl"
                    has_records = jsonl_path.exists()
                    if not has_records and args.missing_policy == "error":
                        raise FileNotFoundError(f"Missing mask jsonl: {jsonl_path}")
                    records = _load_jsonl_records(jsonl_path) if has_records else []
                    if not records and args.missing_policy == "error":
                        raise ValueError(f"Empty mask jsonl: {jsonl_path}")
                    by_frame: dict[int, list[dict]] = {}
                    if records:
                        for rec in records[:8]:
                            if int(rec["episode_id"]) != ep_idx:
                                raise ValueError(f"episode_id mismatch in {jsonl_path}")
                            if rec["video_key"] != src_key:
                                raise ValueError(f"video_key mismatch in {jsonl_path}")
                        by_frame, _ = _build_frame_index(records)
                        for rec in records:
                            label = _record_label(rec)
                            if label:
                                labels_set.add(label)
                    episode_cache[ep_idx] = (episode_dir, records, by_frame)
                except Exception as exc:
                    invalid_eps.add(ep_idx)
                    skipped_items.append(
                        {
                            "task_id": task_name,
                            "video_key": src_key,
                            "episode_id": str(ep_idx),
                            "reason": repr(exc),
                        }
                    )
                    if args.on_error == "raise":
                        raise

            # Filter out invalid episodes from groups
            if invalid_eps:
                filtered: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
                for k, eps in group_to_episodes.items():
                    kept = [e for e in eps if e[0] not in invalid_eps]
                    if kept:
                        filtered[k] = kept
                group_to_episodes = filtered

            sorted_labels = sorted(labels_set)
            if len(sorted_labels) > 255:
                raise ValueError(
                    f"Too many labels for uint8 id space in {task_name}/{src_key}: {len(sorted_labels)}"
                )
            label_to_global_id = {label: i + 1 for i, label in enumerate(sorted_labels)}
            global_id_to_label = {i + 1: label for i, label in enumerate(sorted_labels)}
            if args.save_label_map and global_id_to_label:
                _write_global_label_map(output_root, task_name, dst_key, global_id_to_label)

            groups_iter = sorted(group_to_episodes.items())
            if tqdm is not None:
                groups_iter = tqdm(groups_iter, desc=f"{task_name}/{src_key}: files", leave=False)

            for (chunk_idx, file_idx), ep_list in groups_iter:
                try:
                    out_video_rel = Path(
                        info["video_path"].format(
                            video_key=dst_key, chunk_index=chunk_idx, file_index=file_idx
                        )
                    )
                    out_video_abs = output_root / out_video_rel
                    vis_video_abs = None
                    if args.save_vis_video:
                        vis_video_abs = out_video_abs.with_name(
                            f"{out_video_abs.stem}{args.vis_video_suffix}{out_video_abs.suffix}"
                        )
                    if out_video_abs.exists() and args.skip_existing_mask_videos and (
                        not args.save_vis_video
                        or (vis_video_abs is not None and vis_video_abs.exists())
                    ):
                        continue

                    frame_records_by_global_idx: dict[int, tuple[Path, list[dict], bool]] = {}
                    max_global_frame = -1
                    prev_end = -1
                    # Resolve boundary collisions robustly: if two consecutive episodes map
                    # to the same frame due to floating-point timestamp rounding, shift the
                    # later one to prev_end + 1.
                    for ep_idx, ep_len, raw_start_frame in sorted(ep_list, key=lambda x: x[2]):
                        start_frame = raw_start_frame
                        if start_frame <= prev_end:
                            if raw_start_frame <= prev_end + 1:
                                start_frame = prev_end + 1
                            else:
                                raise ValueError(
                                    f"Non-boundary overlap in {task_name}/{src_key} "
                                    f"(chunk={chunk_idx}, file={file_idx}): "
                                    f"episode {ep_idx} raw_start={raw_start_frame}, prev_end={prev_end}"
                                )
                        episode_dir, records, by_frame = episode_cache[ep_idx]
                        if records and args.save_label_map and args.save_episode_label_map:
                            episode_global_ids = sorted(
                                {
                                    label_to_global_id[_record_label(rec)]
                                    for rec in records
                                    if _record_label(rec) in label_to_global_id
                                }
                            )
                            if episode_global_ids:
                                ep_map = {gid: global_id_to_label[gid] for gid in episode_global_ids}
                                _write_label_map(output_root, task_name, ep_idx, dst_key, ep_map)

                        for local_frame in range(ep_len):
                            global_frame = start_frame + local_frame
                            if global_frame in frame_records_by_global_idx:
                                raise ValueError(
                                    f"Overlapping episodes in {task_name}/{src_key} at global frame {global_frame} "
                                    f"(chunk={chunk_idx}, file={file_idx})"
                                )
                            frame_records_by_global_idx[global_frame] = (
                                episode_dir,
                                by_frame.get(local_frame, []),
                                args.missing_policy == "error",
                            )
                            if global_frame > max_global_frame:
                                max_global_frame = global_frame
                        prev_end = max(prev_end, start_frame + ep_len - 1)

                    total_frames = max_global_frame + 1
                    if total_frames <= 0:
                        continue

                    _encode_group_video(
                        frame_records_by_global_idx=frame_records_by_global_idx,
                        total_frames=total_frames,
                        h=src_h,
                        w=src_w,
                        out_video_abs=out_video_abs,
                        fps=fps,
                        vcodec=args.mask_vcodec,
                        pix_fmt=args.mask_pix_fmt,
                        crf=args.mask_crf,
                        preset=args.mask_preset,
                        frame_desc=f"{task_name}/{src_key}/chunk{chunk_idx:03d}/file{file_idx:03d}",
                        vis_video_abs=vis_video_abs,
                        vis_vcodec=args.vis_vcodec,
                        vis_pix_fmt=args.vis_pix_fmt,
                        vis_crf=args.vis_crf,
                        vis_preset=args.vis_preset,
                        label_to_global_id=label_to_global_id,
                        global_id_to_label=global_id_to_label,
                        vis_legend=args.vis_legend,
                    )
                except Exception as exc:
                    skipped_items.append(
                        {
                            "task_id": task_name,
                            "video_key": src_key,
                            "chunk": str(chunk_idx),
                            "file": str(file_idx),
                            "reason": repr(exc),
                        }
                    )
                    if args.on_error == "raise":
                        raise
                    continue
        except Exception as exc:
            skipped_items.append(
                {
                    "task_id": task_name,
                    "video_key": src_key,
                    "reason": repr(exc),
                }
            )
            if args.on_error == "raise":
                raise
            continue

    for ep_file, df in episode_tables:
        df.to_parquet(ep_file, index=False)

    _write_info(info_path, info)
    print(f"[done] wrote updated dataset to: {output_root}")
    print(f"[done] added mask features: {', '.join(key_map.values())}")
    if skipped_items:
        print(f"[postprocess] skipped items in {task_name}: {len(skipped_items)}")
        for item in skipped_items[:20]:
            print(f"  - {item}")
    return skipped_items


def _discover_tasks(dataset_root: Path) -> list[str]:
    tasks: list[str] = []
    for p in sorted(dataset_root.iterdir()):
        if not p.is_dir():
            continue
        if _has_task_signature(p):
            tasks.append(p.name)
    return tasks


def _read_task_list_file(path: Path) -> set[str]:
    selected: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value and not value.startswith("#"):
                selected.add(value)
    return selected


def _task_dir_names(root: Path) -> set[str]:
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir()
    }


def _prepare_selected_output_root(
    output_root: Path,
    overwrite: bool,
    resume: bool,
) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise NotADirectoryError(f"Output root is not a directory: {output_root}")
        if overwrite:
            if any(output_root.iterdir()):
                raise FileExistsError(f"Output root is not empty: {output_root}")
            return
        if not resume:
            raise FileExistsError(f"Output root already exists: {output_root}")
        print(f"[resume] selected-task output root exists: {output_root}")
        return
    output_root.mkdir(parents=True, exist_ok=False)


def _copy_source_tasks_by_prefix(
    source_root: Path,
    output_root: Path,
    prefix: str,
    overwrite: bool,
    resume: bool,
) -> list[str]:
    task_names = sorted(
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )
    if not task_names:
        raise FileNotFoundError(
            f"No source task directories start with {prefix!r} under {source_root}"
        )

    task_iter = task_names
    if tqdm is not None:
        task_iter = tqdm(task_names, desc=f"Copy source tasks [{prefix}*]", unit="task")
    for task_name in task_iter:
        _prepare_output_tree(
            source_root / task_name,
            output_root / task_name,
            overwrite=overwrite,
            resume=resume,
        )
    return task_names


def _mask_coverage_for_task(source_task: Path, mask_task: Path) -> tuple[int, int]:
    info = _read_info(source_task / "meta" / "info.json")
    rgb_keys = _discover_rgb_keys(info)
    if not rgb_keys:
        raise ValueError(f"No RGB video keys found in {source_task}")

    episode_files = sorted(
        (source_task / "meta" / "episodes").glob("chunk-*/file-*.parquet")
    )
    if not episode_files:
        raise FileNotFoundError(
            f"No episode parquet files under {source_task / 'meta' / 'episodes'}"
        )

    expected = 0
    available = 0
    for episode_file in episode_files:
        episodes = pd.read_parquet(
            episode_file,
            columns=["episode_index"],
        )
        for episode_id_raw in episodes["episode_index"]:
            episode_id = int(episode_id_raw)
            for video_key in rgb_keys:
                expected += 1
                jsonl_path = (
                    mask_task
                    / f"episode_{episode_id:03d}"
                    / video_key
                    / f"episode_{episode_id:03d}.jsonl"
                )
                try:
                    if jsonl_path.is_file() and jsonl_path.stat().st_size > 0:
                        available += 1
                except OSError:
                    continue
    return available, expected


def _process() -> None:
    args = _parse_args()
    source_root = args.source_root.resolve()
    mask_root = args.mask_root.resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"source-root not found: {source_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"mask-root not found: {mask_root}")
    if source_root == output_root:
        raise ValueError("output-root must differ from source-root")

    detected_single = _has_task_signature(source_root)
    mode = args.mode
    if mode == "auto":
        mode = "single" if detected_single else "batch"

    if mode == "single":
        if args.dry_run:
            raise ValueError("--dry-run is supported only in batch mode")
        _prepare_output_tree(
            source_root, output_root, overwrite=args.overwrite, resume=args.resume
        )
        skipped_all = _process_single_task(
            args, output_root=output_root, mask_root=mask_root, task_name=source_root.name
        )
        if args.error_report is not None and skipped_all:
            args.error_report.parent.mkdir(parents=True, exist_ok=True)
            with args.error_report.open("w", encoding="utf-8") as f:
                for item in skipped_all:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[postprocess] wrote skip report: {args.error_report}")
        return

    if mode != "batch":
        raise ValueError(f"Unsupported mode: {mode}")

    tasks = _discover_tasks(source_root)
    if not tasks:
        raise FileNotFoundError(f"No task dirs with meta/info.json found under {source_root}")

    if args.task_ids:
        selected = {x.strip() for x in args.task_ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t in selected]
    if args.task_list_file is not None:
        selected = _read_task_list_file(args.task_list_file)
        tasks = [t for t in tasks if t in selected]
    if args.select_tasks_from_mask_root:
        mask_task_names = _task_dir_names(mask_root)
        tasks = [t for t in tasks if t in mask_task_names]
    if args.require_complete_mask_coverage:
        complete_tasks: list[str] = []
        incomplete_tasks: list[tuple[str, int, int]] = []
        coverage_iter = tasks
        if tqdm is not None:
            coverage_iter = tqdm(tasks, desc="Check complete mask coverage", unit="task")
        for task_name in coverage_iter:
            available, expected = _mask_coverage_for_task(
                source_root / task_name,
                mask_root / task_name,
            )
            if available == expected:
                complete_tasks.append(task_name)
            else:
                incomplete_tasks.append((task_name, available, expected))
        tasks = complete_tasks
        print(
            "[batch] complete mask tasks: "
            f"{len(complete_tasks)}; tasks left without mask features: {len(incomplete_tasks)}"
        )
        for task_name, available, expected in incomplete_tasks:
            print(f"  [no-mask] {task_name}: {available}/{expected} episode-camera pairs")
    if not tasks:
        raise ValueError("No tasks selected to process.")

    if args.dry_run:
        if args.copy_source_task_prefix:
            copy_count = sum(
                1
                for path in source_root.iterdir()
                if path.is_dir()
                and path.name.startswith(args.copy_source_task_prefix)
            )
            print(
                f"[dry-run] source task directories to copy: {copy_count} "
                f"(prefix={args.copy_source_task_prefix!r})"
            )
        elif args.copy_selected_tasks_only:
            print(f"[dry-run] selected source task directories to copy: {len(tasks)}")
        else:
            print("[dry-run] the complete source root would be copied")
        print(f"[dry-run] task directories that would receive mask features: {len(tasks)}")
        print("[dry-run] no files were written")
        return

    if args.copy_source_task_prefix:
        if args.copy_selected_tasks_only:
            raise ValueError(
                "--copy-source-task-prefix and --copy-selected-tasks-only are mutually exclusive"
            )
        _prepare_selected_output_root(
            output_root, overwrite=args.overwrite, resume=args.resume
        )
        copied_tasks = _copy_source_tasks_by_prefix(
            source_root,
            output_root,
            prefix=args.copy_source_task_prefix,
            overwrite=args.overwrite,
            resume=args.resume,
        )
        print(
            f"[batch] copied {len(copied_tasks)} source task directories matching "
            f"{args.copy_source_task_prefix!r}; adding masks to {len(tasks)} tasks"
        )
    elif args.copy_selected_tasks_only:
        _prepare_selected_output_root(
            output_root, overwrite=args.overwrite, resume=args.resume
        )
        print(f"[batch] copying/processing selected tasks only: {len(tasks)}")
    else:
        _prepare_output_tree(
            source_root, output_root, overwrite=args.overwrite, resume=args.resume
        )

    skipped_all: list[dict[str, str]] = []
    task_iter = tasks
    if tqdm is not None:
        task_iter = tqdm(tasks, desc="Tasks")
    for task_name in task_iter:
        src_task = source_root / task_name
        out_task = output_root / task_name
        mask_task = mask_root / task_name
        if not mask_task.exists():
            if args.skip_missing_mask_task:
                print(f"[skip] missing mask task: {mask_task}")
                continue
            raise FileNotFoundError(f"Missing mask task dir: {mask_task}")
        if not _has_task_signature(src_task):
            print(f"[skip] invalid task structure: {src_task}")
            continue
        if args.copy_selected_tasks_only:
            _prepare_output_tree(
                src_task,
                out_task,
                overwrite=args.overwrite,
                resume=args.resume,
            )
        if not _has_task_signature(out_task):
            raise FileNotFoundError(f"Output task dir missing expected metadata: {out_task}")

        if tqdm is None:
            print(f"[task] processing {task_name}")
        skipped_all.extend(
            _process_single_task(args, output_root=out_task, mask_root=mask_task, task_name=task_name)
        )

    if skipped_all:
        print(f"[postprocess] total skipped items: {len(skipped_all)}")
        for item in skipped_all[:50]:
            print(f"  - {item}")
    if args.error_report is not None and skipped_all:
        args.error_report.parent.mkdir(parents=True, exist_ok=True)
        with args.error_report.open("w", encoding="utf-8") as f:
            for item in skipped_all:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[postprocess] wrote skip report: {args.error_report}")


if __name__ == "__main__":
    _process()
