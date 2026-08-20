#!/usr/bin/env python3
"""Generate *.vis.mp4 from existing *_mask.mp4 videos.

This script does not modify mask videos. It only writes visualization videos
next to them, e.g.:
  file-000.mp4 -> file-000.vis.mp4
"""

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
        description="Generate visualization videos from existing *_mask.mp4 files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Task root (e.g., .../human_H1) or dataset root containing task dirs.",
    )
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
        "--mask-keys",
        type=str,
        default="",
        help="Optional comma-separated mask keys, e.g. observation.images.cam1_mask",
    )
    parser.add_argument(
        "--vis-video-suffix",
        type=str,
        default=".vis",
        help="Suffix inserted before .mp4, e.g. file-000.vis.mp4",
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
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip if *.vis.mp4 already exists.",
    )
    parser.add_argument(
        "--legend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay legend (id->label).",
    )
    return parser.parse_args()


def _is_task_root(root: Path) -> bool:
    return (root / "meta" / "info.json").exists() and (root / "videos").exists()


def _discover_tasks(dataset_root: Path) -> list[Path]:
    tasks: list[Path] = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and _is_task_root(p):
            tasks.append(p)
    return tasks


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_mask_keys(task_root: Path) -> list[str]:
    info = _read_json(task_root / "meta" / "info.json")
    keys: list[str] = []
    for k, ft in info.get("features", {}).items():
        if not k.startswith("observation.images."):
            continue
        if not k.endswith("_mask"):
            continue
        if ft.get("dtype") != "video":
            continue
        keys.append(k)
    return sorted(keys)


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
        iid = int(instance_id)
        if iid == 0:
            continue
        out[canvas == iid] = _color_for_instance(iid)
    return out


def _overlay_legend(vis_rgb: np.ndarray, id_to_label: dict[int, str], present_ids: list[int]) -> np.ndarray:
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
    texts: list[tuple[int, str]] = []
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


def _load_global_map(task_root: Path, mask_key: str) -> dict[int, str]:
    p = task_root / "meta" / "mask_labels" / task_root.name / mask_key / "global.json"
    if not p.exists():
        return {}
    d = _read_json(p)
    return {int(k): str(v) for k, v in d.items()}


def _iter_mask_videos(task_root: Path, mask_key: str) -> list[Path]:
    vdir = task_root / "videos" / mask_key
    if not vdir.exists():
        return []
    return sorted(
        p for p in vdir.glob("chunk-*/file-*.mp4") if not p.name.endswith(".vis.mp4")
    )


def _encode_vis_from_mask(
    in_video: Path,
    out_video: Path,
    id_to_label: dict[int, str],
    vcodec: str,
    pix_fmt: str,
    crf: int,
    preset: str,
    legend: bool,
) -> None:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(in_video), mode="r") as in_container:
        in_stream = in_container.streams.video[0]
        fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0
        width = int(in_stream.width)
        height = int(in_stream.height)

        options = {"preset": preset}
        if crf is not None:
            options["crf"] = str(crf)

        with av.open(str(out_video), mode="w") as out_container:
            out_stream = out_container.add_stream(vcodec, rate=int(round(fps)), options=options)
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = pix_fmt

            frame_iter = in_container.decode(in_stream)
            if tqdm is not None:
                frame_iter = tqdm(frame_iter, desc=f"vis {in_video.name}", leave=False)

            for frame in frame_iter:
                rgb = frame.to_ndarray(format="rgb24")
                canvas = rgb[:, :, 0].astype(np.uint8)
                vis = _colorize_instance_map(canvas)
                if legend:
                    present_ids = sorted(int(i) for i in np.unique(canvas) if int(i) != 0)
                    vis = _overlay_legend(vis, id_to_label, present_ids)
                out_frame = av.VideoFrame.from_ndarray(vis, format="rgb24")
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)

            for packet in out_stream.encode():
                out_container.mux(packet)


def _process_task(task_root: Path, args: argparse.Namespace) -> None:
    keys = _discover_mask_keys(task_root)
    if args.mask_keys:
        selected = {k.strip() for k in args.mask_keys.split(",") if k.strip()}
        keys = [k for k in keys if k in selected]
    if not keys:
        print(f"[skip] no mask keys in {task_root}")
        return

    key_iter = keys if tqdm is None else tqdm(keys, desc=f"{task_root.name}: keys", leave=False)
    for mask_key in key_iter:
        id_to_label = _load_global_map(task_root, mask_key)
        videos = _iter_mask_videos(task_root, mask_key)
        v_iter = videos if tqdm is None else tqdm(videos, desc=f"{task_root.name}/{mask_key}", leave=False)
        for in_video in v_iter:
            out_video = in_video.with_name(f"{in_video.stem}{args.vis_video_suffix}{in_video.suffix}")
            if out_video.exists() and args.skip_existing:
                continue
            _encode_vis_from_mask(
                in_video=in_video,
                out_video=out_video,
                id_to_label=id_to_label,
                vcodec=args.vis_vcodec,
                pix_fmt=args.vis_pix_fmt,
                crf=args.vis_crf,
                preset=args.vis_preset,
                legend=args.legend,
            )


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    mode = args.mode
    if mode == "auto":
        mode = "single" if _is_task_root(root) else "batch"

    if mode == "single":
        if not _is_task_root(root):
            raise ValueError(f"Not a task root: {root}")
        _process_task(root, args)
        return

    if mode != "batch":
        raise ValueError(f"Unsupported mode: {mode}")

    tasks = _discover_tasks(root)
    if args.task_ids:
        selected = {x.strip() for x in args.task_ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t.name in selected]
    if not tasks:
        raise ValueError(f"No tasks to process under {root}")

    task_iter = tasks if tqdm is None else tqdm(tasks, desc="Tasks")
    for t in task_iter:
        _process_task(t, args)


if __name__ == "__main__":
    main()

