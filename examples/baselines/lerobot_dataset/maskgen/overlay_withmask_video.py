#!/usr/bin/env python3
"""Overlay an existing *_mask.mp4 video onto the corresponding RGB video."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Overlay a withmask video onto its RGB video")
    parser.add_argument("--task-root", type=Path, required=True, help="Task root, e.g. .../human_H1")
    parser.add_argument("--video-key", type=str, required=True, help="RGB video key, e.g. observation.images.zed2i")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--file-index", type=int, default=0)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--suffix", type=str, default=".overlay")
    parser.add_argument("--legend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vcodec", type=str, default="libx264")
    parser.add_argument("--pix-fmt", type=str, default="yuv420p")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", type=str, default="veryfast")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _video_path(task_root: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    return (
        task_root
        / "videos"
        / video_key
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def _load_global_map(task_root: Path, mask_key: str) -> dict[int, str]:
    path = task_root / "meta" / "mask_labels" / task_root.name / mask_key / "global.json"
    if not path.exists():
        return {}
    data = _read_json(path)
    return {int(k): str(v) for k, v in data.items()}


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


def main() -> None:
    args = _parse_args()
    task_root = args.task_root.resolve()
    mask_key = f"{args.video_key}_mask"
    rgb_video = _video_path(task_root, args.video_key, args.chunk_index, args.file_index)
    mask_video = _video_path(task_root, mask_key, args.chunk_index, args.file_index)

    if not rgb_video.exists():
        raise FileNotFoundError(f"RGB video not found: {rgb_video}")
    if not mask_video.exists():
        raise FileNotFoundError(f"Mask video not found: {mask_video}")

    output_video = args.output_video
    if output_video is None:
        output_video = mask_video.with_name(f"{mask_video.stem}{args.suffix}{mask_video.suffix}")
    output_video.parent.mkdir(parents=True, exist_ok=True)

    id_to_label = _load_global_map(task_root, mask_key)

    with av.open(str(rgb_video), mode="r") as rgb_container, av.open(str(mask_video), mode="r") as mask_container:
        rgb_stream = rgb_container.streams.video[0]
        mask_stream = mask_container.streams.video[0]
        fps = rgb_stream.average_rate or Fraction(30, 1)
        width = int(rgb_stream.width)
        height = int(rgb_stream.height)

        options = {"preset": args.preset, "crf": str(args.crf)}
        with av.open(str(output_video), mode="w") as out_container:
            out_stream = out_container.add_stream(args.vcodec, rate=fps, options=options)
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = args.pix_fmt

            rgb_frames = rgb_container.decode(rgb_stream)
            mask_frames = mask_container.decode(mask_stream)
            total = rgb_stream.frames or mask_stream.frames or None
            frame_iter = zip(rgb_frames, mask_frames)
            if tqdm is not None:
                frame_iter = tqdm(frame_iter, total=total, desc=f"overlay {rgb_video.name}")
            for rgb_frame, mask_frame in frame_iter:
                rgb = rgb_frame.to_ndarray(format="rgb24")
                mask_ids = _mask_ids_from_frame(mask_frame, target_size=(width, height))
                vis = _overlay_frame(rgb, mask_ids, args.alpha)
                if args.legend:
                    present_ids = sorted(int(i) for i in np.unique(mask_ids) if int(i) != 0)
                    vis = _overlay_legend(vis, id_to_label, present_ids)
                out_frame = av.VideoFrame.from_ndarray(vis, format="rgb24")
                for packet in out_stream.encode(out_frame):
                    out_container.mux(packet)

            for packet in out_stream.encode():
                out_container.mux(packet)

    print(output_video)


if __name__ == "__main__":
    main()
