#!/usr/bin/env python3
"""Batch overlay withmask videos and sample frames into position folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Batch overlay existing *_mask videos and save uniformly sampled frames"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Task root, e.g. .../human_H1, or dataset root containing task dirs.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "single", "batch"],
        help="auto: detect task root by meta/info.json; batch scans subdirs.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default="",
        help="Batch mode only. Optional comma-separated task ids.",
    )
    parser.add_argument(
        "--video-keys",
        type=str,
        default="",
        help="Optional comma-separated RGB video keys, e.g. observation.images.zed2i.",
    )
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--legend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image-ext", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Debug limit. 0 means process all videos.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_task_root(root: Path) -> bool:
    return (root / "meta" / "info.json").exists() and (root / "videos").exists()


def _discover_tasks(dataset_root: Path) -> list[Path]:
    return sorted(p for p in dataset_root.iterdir() if p.is_dir() and _is_task_root(p))


def _discover_video_keys(task_root: Path) -> list[str]:
    info = _read_json(task_root / "meta" / "info.json")
    keys = []
    for key, feature in info.get("features", {}).items():
        if not key.startswith("observation.images."):
            continue
        if key.endswith("_mask") or key.endswith("_depth"):
            continue
        if feature.get("dtype") != "video":
            continue
        if (task_root / "videos" / f"{key}_mask").exists():
            keys.append(key)
    return sorted(keys)


def _load_global_map(task_root: Path, mask_key: str) -> dict[int, str]:
    path = task_root / "meta" / "mask_labels" / task_root.name / mask_key / "global.json"
    if not path.exists():
        return {}
    data = _read_json(path)
    return {int(k): str(v) for k, v in data.items()}


def _iter_video_pairs(task_root: Path, video_key: str) -> list[tuple[Path, Path]]:
    rgb_root = task_root / "videos" / video_key
    mask_root = task_root / "videos" / f"{video_key}_mask"
    if not rgb_root.exists() or not mask_root.exists():
        return []

    pairs = []
    for rgb_video in sorted(rgb_root.glob("chunk-*/file-*.mp4")):
        rel = rgb_video.relative_to(rgb_root)
        mask_video = mask_root / rel
        if mask_video.exists():
            pairs.append((rgb_video, mask_video))
    return pairs


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


def _colorize_instance_map(mask_ids: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask_ids.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(mask_ids):
        iid = int(instance_id)
        if iid == 0:
            continue
        out[mask_ids == iid] = _color_for_instance(iid)
    return out


def _overlay_legend(vis_rgb: np.ndarray, id_to_label: dict[int, str], present_ids: list[int]) -> np.ndarray:
    if not present_ids:
        return vis_rgb

    image = Image.fromarray(vis_rgb)
    draw = ImageDraw.Draw(image, mode="RGBA")
    font = ImageFont.load_default()

    x0 = 8
    y0 = 8
    line_h = 16
    box = 10
    pad = 6
    texts: list[tuple[int, str]] = []
    max_text = 0
    for gid in present_ids:
        text = f"{gid}: {id_to_label.get(gid, f'id_{gid}')}"
        texts.append((gid, text))
        max_text = max(max_text, len(text))

    panel_w = pad * 3 + box + max_text * 7
    panel_h = pad * 2 + line_h * len(texts)
    draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], fill=(0, 0, 0, 140))

    y = y0 + pad
    for gid, text in texts:
        color = tuple(int(v) for v in _color_for_instance(gid))
        draw.rectangle([x0 + pad, y + 3, x0 + pad + box, y + 3 + box], fill=(*color, 255))
        draw.text((x0 + pad * 2 + box, y), text, font=font, fill=(255, 255, 255, 255))
        y += line_h
    return np.asarray(image)


def _mask_ids_from_frame(frame: av.VideoFrame, target_size: tuple[int, int]) -> np.ndarray:
    rgb = frame.to_ndarray(format="rgb24")
    mask_ids = rgb[:, :, 0].astype(np.uint8)
    target_w, target_h = target_size
    if mask_ids.shape != (target_h, target_w):
        mask_ids = np.asarray(
            Image.fromarray(mask_ids).resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        )
    return mask_ids


def _overlay_frame(rgb: np.ndarray, mask_ids: np.ndarray, alpha: float) -> np.ndarray:
    color = _colorize_instance_map(mask_ids)
    active = mask_ids > 0
    out = rgb.copy()
    out[active] = ((1.0 - alpha) * out[active] + alpha * color[active]).astype(np.uint8)
    return out


def _sample_indices(total: int, num_frames: int) -> list[int]:
    if total <= 0:
        raise ValueError("Cannot sample frames from a video with unknown/non-positive frame count")
    n = min(num_frames, total)
    return sorted({int(round(x)) for x in np.linspace(0, total - 1, n)})


def _save_image(path: Path, image: np.ndarray, image_ext: str, jpeg_quality: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(image)
    if image_ext == "jpg":
        pil.save(path, quality=jpeg_quality)
    else:
        pil.save(path)


def _safe_name(video_key: str, rgb_video: Path, frame_idx: int) -> str:
    chunk = rgb_video.parent.name
    file_stem = rgb_video.stem
    key = video_key.replace(".", "-")
    return f"{key}__{chunk}__{file_stem}__frame-{frame_idx:06d}"


def _process_video_pair(
    task_root: Path,
    video_key: str,
    rgb_video: Path,
    mask_video: Path,
    output_root: Path,
    num_frames: int,
    alpha: float,
    legend: bool,
    image_ext: str,
    jpeg_quality: int,
    overwrite: bool,
) -> int:
    mask_key = f"{video_key}_mask"
    id_to_label = _load_global_map(task_root, mask_key)

    saved = 0
    with av.open(str(rgb_video), mode="r") as rgb_container, av.open(str(mask_video), mode="r") as mask_container:
        rgb_stream = rgb_container.streams.video[0]
        mask_stream = mask_container.streams.video[0]
        total = int(rgb_stream.frames or mask_stream.frames or 0)
        sample_indices = _sample_indices(total, num_frames)
        sample_set = set(sample_indices)
        sample_pos = {idx: pos for pos, idx in enumerate(sample_indices, start=1)}

        width = int(rgb_stream.width)
        height = int(rgb_stream.height)
        rgb_frames = rgb_container.decode(rgb_stream)
        mask_frames = mask_container.decode(mask_stream)
        for frame_idx, (rgb_frame, mask_frame) in enumerate(zip(rgb_frames, mask_frames)):
            if frame_idx not in sample_set:
                continue

            rgb = rgb_frame.to_ndarray(format="rgb24")
            mask_ids = _mask_ids_from_frame(mask_frame, target_size=(width, height))
            vis = _overlay_frame(rgb, mask_ids, alpha)
            if legend:
                present_ids = sorted(int(i) for i in np.unique(mask_ids) if int(i) != 0)
                vis = _overlay_legend(vis, id_to_label, present_ids)

            pos = sample_pos[frame_idx]
            out_name = f"{_safe_name(video_key, rgb_video, frame_idx)}.{image_ext}"
            out_path = output_root / task_root.name / f"sample_{pos:02d}" / out_name
            _save_image(out_path, vis, image_ext, jpeg_quality, overwrite)
            saved += 1

            if saved >= len(sample_indices):
                break
    return saved


def _collect_pairs(tasks: list[Path], selected_video_keys: set[str]) -> list[tuple[Path, str, Path, Path]]:
    pairs: list[tuple[Path, str, Path, Path]] = []
    for task_root in tasks:
        keys = _discover_video_keys(task_root)
        if selected_video_keys:
            keys = [k for k in keys if k in selected_video_keys]
        for key in keys:
            for rgb_video, mask_video in _iter_video_pairs(task_root, key):
                pairs.append((task_root, key, rgb_video, mask_video))
    return pairs


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")

    mode = args.mode
    if mode == "auto":
        mode = "single" if _is_task_root(root) else "batch"

    if mode == "single":
        if not _is_task_root(root):
            raise ValueError(f"Not a task root: {root}")
        tasks = [root]
    elif mode == "batch":
        tasks = _discover_tasks(root)
        if args.task_ids:
            selected = {x.strip() for x in args.task_ids.split(",") if x.strip()}
            tasks = [t for t in tasks if t.name in selected]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if not tasks:
        raise ValueError(f"No tasks found under {root}")

    selected_video_keys = {x.strip() for x in args.video_keys.split(",") if x.strip()}
    pairs = _collect_pairs(tasks, selected_video_keys)
    if args.max_videos > 0:
        pairs = pairs[: args.max_videos]
    if not pairs:
        raise ValueError("No RGB/mask video pairs found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    iterator = pairs
    if tqdm is not None:
        iterator = tqdm(pairs, desc="overlay sample videos")

    total_saved = 0
    manifest_path = args.output_root / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for task_root, video_key, rgb_video, mask_video in iterator:
            saved = _process_video_pair(
                task_root=task_root,
                video_key=video_key,
                rgb_video=rgb_video,
                mask_video=mask_video,
                output_root=args.output_root,
                num_frames=args.num_frames,
                alpha=args.alpha,
                legend=args.legend,
                image_ext=args.image_ext,
                jpeg_quality=args.jpeg_quality,
                overwrite=args.overwrite,
            )
            total_saved += saved
            manifest.write(
                json.dumps(
                    {
                        "task": task_root.name,
                        "video_key": video_key,
                        "rgb_video": str(rgb_video),
                        "mask_video": str(mask_video),
                        "saved_frames": saved,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"saved_frames={total_saved}")
    print(f"output_root={args.output_root}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
