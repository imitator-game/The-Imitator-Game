#!/usr/bin/env python3
"""Episode-level Grounded-SAM-2 mask generation for full videos."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import sys

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

THIS_DIR = Path(__file__).resolve().parent
LEROBOT_DIR = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[4]
GS2_ROOT_DEFAULT = Path("../Grounded-SAM-2")
if str(LEROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(LEROBOT_DIR))


def _normalize_task_id(task_id: str) -> str:
    tid = task_id.strip()
    if tid.startswith("human_"):
        tid = tid[len("human_") :]
        return tid.split("_")[0]
    if tid.startswith("robot_"):
        tid = tid[len("robot_") :]
        if tid.endswith("_e_d"):
            tid = tid[: -len("_e_d")]
        elif tid.endswith("_e"):
            tid = tid[: -len("_e")]
        return tid
    return tid


def _task_key_candidates(task_id: str) -> list[str]:
    keys = [task_id.strip(), _normalize_task_id(task_id)]
    # deduplicate while preserving order
    out: list[str] = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _load_object_list(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Object list not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip())
    if "task_id" not in df.columns or "item" not in df.columns:
        raise ValueError("CSV must contain columns: task_id, item")

    df["task_id"] = df["task_id"].astype(str).str.strip()
    df["item"] = df["item"].astype(str).str.strip()

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        tid = row["task_id"]
        item = row["item"]
        if tid and item and item.lower() != "nan":
            mapping[tid] = item
            mapping[_normalize_task_id(tid)] = item
    return mapping


def _select_object_map(task_id: str, obj: dict[str, Any], group: str) -> dict[str, str] | None:
    if all(k in obj for k in ("human", "robot", "sim")):
        if group != "auto":
            return obj.get(group, {})
        candidates = _task_key_candidates(task_id)
        for key in ("human", "robot", "sim"):
            m = obj.get(key, {})
            if any(c in m for c in candidates):
                return m
        return None
    if group not in ("auto", "human"):
        print(f"[mask_gen_whole] warning: object list is flat; ignoring group={group}")
    return obj


def _get_prompt(task_id: str, obj: dict[str, Any], group: str) -> str | None:
    mapping = _select_object_map(task_id, obj, group)
    if not mapping:
        return None
    for k in _task_key_candidates(task_id):
        if k in mapping:
            return mapping[k]
    return None


def _match_task_domain(task_name: str, domain: str) -> bool:
    if domain == "all":
        return True
    if domain == "human":
        return task_name.startswith("human_")
    if domain == "robot":
        return task_name.startswith("robot_")
    if domain == "sim":
        return not task_name.startswith(("human_", "robot_"))
    raise ValueError(f"Unsupported task domain: {domain}")


def _load_info_json(task_dir: Path) -> dict[str, Any]:
    info_path = task_dir / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_episodes(task_dir: Path) -> pd.DataFrame:
    epi_root = task_dir / "meta" / "episodes"
    files = sorted(epi_root.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet under {epi_root}")
    dfs = [pd.read_parquet(p) for p in files]
    return pd.concat(dfs, ignore_index=True)


def _is_depth_key(key: str, info: dict[str, Any]) -> bool:
    feat = info.get("features", {}).get(key, {})
    is_depth_flag = bool(feat.get("info", {}).get("video.is_depth_map", False))
    name_lower = key.lower()
    is_depth_name = "depth" in name_lower or name_lower.endswith("_depth")
    return is_depth_flag or is_depth_name


def _list_video_keys(info: dict[str, Any], episodes: pd.DataFrame, include_depth: bool) -> list[str]:
    keys = set()
    for k, v in info.get("features", {}).items():
        if v.get("dtype") != "video":
            continue
        if not k.startswith("observation.images."):
            continue
        keys.add(k)

    for col in episodes.columns:
        if not col.startswith("videos/") or not col.endswith("/chunk_index"):
            continue
        key = col[len("videos/") : -len("/chunk_index")]
        if key.startswith("observation.images."):
            keys.add(key)

    filtered = []
    for k in keys:
        if _is_depth_key(k, info) and not include_depth:
            continue
        filtered.append(k)
    return sorted(filtered)


def _render_video_path(task_dir: Path, info: dict[str, Any], video_key: str, row: pd.Series) -> Path:
    template = info["video_path"]
    chunk_idx = int(row[f"videos/{video_key}/chunk_index"])
    file_idx = int(row[f"videos/{video_key}/file_index"])
    rel = template.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx)
    return task_dir / rel


def _timestamps_for_episode(row: pd.Series, video_key: str, fps: float) -> list[float]:
    start = float(row[f"videos/{video_key}/from_timestamp"])
    length = int(row["length"])
    return [start + i / fps for i in range(length)]


def _decode_frames(video_path: Path, timestamps: list[float], backend: str) -> np.ndarray:
    from video_utils import decode_video_frames

    try:
        frames = decode_video_frames(video_path, timestamps, tolerance_s=0.2, backend=backend)
    except Exception as exc:
        if backend == "torchcodec":
            print(f"[mask_gen_whole] torchcodec failed ({exc}); falling back to pyav.")
            frames = decode_video_frames(video_path, timestamps, tolerance_s=0.2, backend="pyav")
        else:
            raise
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
    frames = (frames * 255.0).clip(0, 255).astype(np.uint8)
    return frames


def _save_frame_jpg(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, quality=95)


def _safe_path_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _prepare_frames_dir(
    out_root: Path,
    frame_cache_root: Path | None,
    task_id: str,
    episode_id: int,
    video_key: str,
) -> tuple[Path, bool]:
    if frame_cache_root is None:
        frames_dir = out_root / "_frames_jpg"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        return frames_dir, False

    frame_cache_root.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"{_safe_path_token(task_id)}_"
        f"episode_{episode_id:03d}_"
        f"{_safe_path_token(video_key)}_"
    )
    frames_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(frame_cache_root)))
    return frames_dir, True


def _save_mask(mask: np.ndarray, path_stem: Path, storage: str) -> Path:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = mask.astype(np.uint8)
    if storage == "npy":
        out = path_stem.with_suffix(".npy")
        np.save(out, mask_u8)
        return out
    if storage == "npz":
        out = path_stem.with_suffix(".npz")
        np.savez_compressed(out, mask=mask_u8)
        return out
    if storage == "packbits":
        out = path_stem.with_suffix(".npz")
        flat = np.packbits(mask_u8.reshape(-1), bitorder="little")
        np.savez_compressed(
            out,
            encoding=np.array(["packbits_little"]),
            shape=np.array(mask_u8.shape, dtype=np.int32),
            packed=flat,
        )
        return out
    raise ValueError(f"Unsupported mask storage: {storage}")


def _overlay(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}"
        )
    color = np.array([255, 0, 0], dtype=np.uint8)
    mask_bool = mask.astype(bool)
    overlay = image.copy()
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * color
    ).astype(np.uint8)
    return overlay


def _save_vis_grid(image: np.ndarray, masks: list[np.ndarray], path: Path, cols: int, alpha: float) -> None:
    if not masks:
        return
    overlays = [_overlay(image, m, alpha) for m in masks]
    h, w, _ = overlays[0].shape
    cols = max(1, cols)
    rows = (len(overlays) + cols - 1) // cols
    canvas = Image.new("RGB", (w * cols, h * rows), color=(0, 0, 0))
    for idx, overlay in enumerate(overlays):
        r = idx // cols
        c = idx % cols
        canvas.paste(Image.fromarray(overlay), (c * w, r * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _build_models(gs2_root: Path, sam2_ckpt: Path, sam2_cfg: str, gdino_cfg: Path, gdino_ckpt: Path, device: str):
    import sys
    if str(gs2_root) not in sys.path:
        sys.path.insert(0, str(gs2_root))

    import torch
    from sam2.build_sam import build_sam2
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from grounding_dino.groundingdino.util.inference import load_model

    sam2_img_model = build_sam2(sam2_cfg, str(sam2_ckpt), device=device)
    image_predictor = SAM2ImagePredictor(sam2_img_model)
    video_predictor = build_sam2_video_predictor(sam2_cfg, str(sam2_ckpt), device=device)

    grounding_model = load_model(
        model_config_path=str(gdino_cfg),
        model_checkpoint_path=str(gdino_ckpt),
        device=device,
    )

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return grounding_model, image_predictor, video_predictor


_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def _split_prompt_phrases(prompt: str) -> list[str]:
    phrases = []
    for part in re.split(r"[,;]", prompt):
        phrase = re.sub(r"\s+", " ", part).strip(" .")
        if phrase:
            phrases.append(phrase)
    return phrases


def _singularize_word(word: str) -> str:
    word = word.lower()
    if "'" in word or word.endswith("is"):
        return word
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("xes"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def _normalize_label_text(text: str) -> str:
    text = text.lower().replace("-", " ")
    words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)
    return " ".join(_singularize_word(word) for word in words)


def _phrase_count_and_text(phrase: str) -> tuple[int, str]:
    norm = _normalize_label_text(phrase)
    if not norm:
        return 1, norm
    words = norm.split()
    first = words[0]
    if first in _NUMBER_WORDS and len(words) > 1:
        return _NUMBER_WORDS[first], " ".join(words[1:])
    if first.isdigit() and len(words) > 1:
        return max(1, int(first)), " ".join(words[1:])
    if first == "pair" and len(words) > 1:
        if len(words) > 2 and words[1] == "of":
            return 2, " ".join(words[2:])
        return 2, " ".join(words[1:])
    return 1, norm


def _prompt_label_limits(prompt: str) -> tuple[dict[str, int], dict[str, int]]:
    exact_counts: dict[str, int] = {}
    head_counts: dict[str, int] = {}
    for raw_phrase in _split_prompt_phrases(prompt):
        count, phrase = _phrase_count_and_text(raw_phrase)
        if not phrase:
            continue
        exact_counts[phrase] = exact_counts.get(phrase, 0) + count
        head = phrase.split()[-1]
        head_counts[head] = head_counts.get(head, 0) + count
    return exact_counts, head_counts


def _format_prompt_phrases(prompt: str) -> str:
    phrases = _split_prompt_phrases(prompt)
    if not phrases:
        return prompt
    return ". ".join(phrases) + "."


def _limit_for_grounding_label(
    label: str,
    exact_counts: dict[str, int],
    head_counts: dict[str, int],
    max_boxes_per_label: int,
) -> int:
    label_norm = _normalize_label_text(label)
    if not label_norm:
        return max(1, max_boxes_per_label)
    label_words = label_norm.split()
    if label_norm in exact_counts:
        return min(max_boxes_per_label, max(1, exact_counts[label_norm]))
    if len(label_words) >= 2:
        suffix_matches = [
            count
            for phrase, count in exact_counts.items()
            if phrase.endswith(f" {label_norm}")
        ]
        if suffix_matches:
            return min(max_boxes_per_label, max(1, sum(suffix_matches)))
    head = label_words[-1]
    if head in head_counts:
        return min(max_boxes_per_label, max(1, head_counts[head]))
    return 1


def _ground_first_frame(
    grounding_model,
    image_predictor,
    image: np.ndarray,
    text_prompt: str,
    box_th: float,
    text_th: float,
    max_boxes_per_label: int,
    infer_boxes_per_label_from_prompt: bool,
    device: str,
):
    from torchvision.ops import box_convert
    import torch
    from grounding_dino.groundingdino.util.inference import load_image, predict

    # GroundingDINO expects RGB image array in load_image, but we already have numpy.
    # We mimic their load_image by saving to temp array via PIL.
    # Use their transforms by converting to PIL and applying transforms in load_image.
    # Simpler: write image to temp PIL and re-use load_image with path is not available.
    # Instead we reimplement minimal: convert to PIL and apply same transforms.
    # Use predict(model, image_tensor, caption, ...)
    import grounding_dino.groundingdino.datasets.transforms as T
    from PIL import Image as PILImage

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_pil = PILImage.fromarray(image).convert("RGB")
    image_transformed, _ = transform(image_pil, None)

    grounding_prompt = _format_prompt_phrases(text_prompt) if infer_boxes_per_label_from_prompt else text_prompt
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image_transformed,
        caption=grounding_prompt,
        box_threshold=box_th,
        text_threshold=text_th,
        device=device,
        remove_combined=infer_boxes_per_label_from_prompt,
    )

    # Keep the top-K boxes per label. This is important for robot tasks whose
    # prompts include multiple same-class objects, e.g. "two tennis balls".
    conf_list = confidences.cpu().numpy().tolist()
    exact_counts, head_counts = (
        _prompt_label_limits(text_prompt) if infer_boxes_per_label_from_prompt else ({}, {})
    )
    # Prefer specific hand labels: drop generic "hand" if left/right exists.
    label_set = set(labels)
    boxes_by_label: dict[str, list[tuple[int, float]]] = {}
    for idx, (label, score) in enumerate(zip(labels, conf_list)):
        if label == "hand" and ("left hand" in label_set or "right hand" in label_set):
            continue
        boxes_by_label.setdefault(label, []).append((idx, score))
    keep_indices = []
    for label, candidates in boxes_by_label.items():
        candidates.sort(key=lambda item: item[1], reverse=True)
        label_limit = (
            _limit_for_grounding_label(label, exact_counts, head_counts, max_boxes_per_label)
            if infer_boxes_per_label_from_prompt
            else max(1, max_boxes_per_label)
        )
        keep_indices.extend(idx for idx, _ in candidates[:label_limit])
    if not keep_indices:
        return None, [], np.empty((0, image.shape[0], image.shape[1]), dtype=np.uint8)
    keep_indices.sort()

    h, w, _ = image.shape
    boxes = boxes[keep_indices] * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    labels = [labels[i] for i in keep_indices]

    image_predictor.set_image(image)
    masks, scores, _ = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    return input_boxes, labels, masks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Episode-level mask generation")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory containing human_H*/")
    parser.add_argument("--task-id", type=str, default=None, help="Run only this task_id (e.g., human_H1).")
    parser.add_argument(
        "--task-ids",
        type=str,
        default="",
        help="Run only these comma-separated task ids. Useful for sharding robot runs.",
    )
    parser.add_argument(
        "--task-list-file",
        type=Path,
        default=None,
        help="Run only task ids listed in this text file, one task id per line.",
    )
    parser.add_argument(
        "--task-domain",
        type=str,
        default="human",
        choices=["human", "robot", "sim", "all"],
        help="Which task namespace to scan under dataset-root. Defaults to human.",
    )
    parser.add_argument("--episode-id", type=int, default=None, help="Run only this episode_index.")
    parser.add_argument("--video-key", type=str, default=None, help="Run only this camera key (e.g., observation.images.cam1).")
    parser.add_argument(
        "--object-list",
        type=Path,
        default=Path("examples/baselines/lerobot_dataset/maskgen/object_list_all.json"),
        help="Object list JSON (grouped by human/robot/sim) or CSV with task_id->item.",
    )
    parser.add_argument(
        "--object-group",
        type=str,
        default="auto",
        choices=["auto", "human", "robot", "sim"],
        help="Which group to use when object list is grouped. auto tries to match task_id.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gs2-root", type=Path, default=GS2_ROOT_DEFAULT)
    parser.add_argument("--sam2-checkpoint", type=Path, default=GS2_ROOT_DEFAULT / "checkpoints" / "sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--gdino-config", type=Path, default=GS2_ROOT_DEFAULT / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gdino-checkpoint", type=Path, default=GS2_ROOT_DEFAULT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth")
    parser.add_argument("--bert-base-path", type=Path, default=None, help="Local path to bert-base-uncased directory (contains config.json).")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--max-boxes-per-label",
        type=int,
        default=1,
        help="Keep up to this many GroundingDINO boxes per label on the first frame.",
    )
    parser.add_argument(
        "--infer-boxes-per-label-from-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Infer per-label box limits from comma-separated object prompts. "
            "This also queries GroundingDINO with phrase-separated captions to "
            "preserve labels such as 'grey plate' and 'pink plate'."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video-backend", type=str, default="torchcodec", choices=["torchcodec", "pyav", "video_reader"])
    parser.add_argument("--include-depth", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-frame-images", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--autocast-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-hands", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hf-cache-dir", type=Path, default=None, help="Local HF cache dir for offline bert-base-uncased.")
    parser.add_argument(
        "--frame-cache-root",
        type=Path,
        default=None,
        help="Optional fast local directory for temporary SAM2 jpg frames, e.g. /dev/shm/maskgen_frames.",
    )
    parser.add_argument("--save-vis-grid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mask-storage",
        type=str,
        default="packbits",
        choices=["packbits", "npz", "npy"],
        help="Mask file storage format. packbits is smallest and default.",
    )
    parser.add_argument("--vis-grid-cols", type=int, default=4)
    parser.add_argument("--vis-alpha", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--debug-video-keys",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print resolved video keys per task and exit.",
    )
    parser.add_argument(
        "--debug-episode-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print (task, episode, video_key) as they are processed.",
    )
    parser.add_argument(
        "--empty-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call torch.cuda.empty_cache() after each video_key to release cached GPU memory.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip episode/camera if output jsonl already exists.",
    )
    parser.add_argument(
        "--save-run-manifest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save CLI args and resolved task list under output-root/_run_manifests/.",
    )
    return parser.parse_args()


def _parse_task_filter(args: argparse.Namespace) -> set[str] | None:
    selected: set[str] = set()
    if args.task_id:
        selected.add(args.task_id)
    if args.task_ids:
        selected.update(t.strip() for t in args.task_ids.split(",") if t.strip())
    if args.task_list_file is not None:
        with args.task_list_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    selected.add(line)
    return selected or None


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def _write_run_manifest(args: argparse.Namespace, tasks: list[Path]) -> None:
    manifest_dir = args.output_root / "_run_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = manifest_dir / f"mask_gen_whole_{stamp}.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "args": _jsonable_args(args),
        "tasks": [p.name for p in tasks],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[mask_gen_whole] wrote run manifest: {manifest_path}")


def main() -> None:
    if tqdm is None:
        raise ImportError("tqdm is required. Install with `pip install tqdm`.")

    args = _parse_args()
    if args.hf_cache_dir is not None:
        hf_root = args.hf_cache_dir
        os.environ["HF_HOME"] = str(hf_root)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_root / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(hf_root / "transformers")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    object_list = _load_object_list(args.object_list)

    gdino_config = args.gdino_config
    if args.bert_base_path is not None:
        cfg_text = gdino_config.read_text(encoding="utf-8")
        local_path = str(args.bert_base_path)
        cfg_text = cfg_text.replace('text_encoder_type = "bert-base-uncased"', f'text_encoder_type = "{local_path}"')
        cfg_text = cfg_text.replace("text_encoder_type='bert-base-uncased'", f"text_encoder_type='{local_path}'")
        tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        tmp.write(cfg_text)
        tmp.close()
        gdino_config = Path(tmp.name)

    grounding_model, image_predictor, video_predictor = _build_models(
        gs2_root=args.gs2_root,
        sam2_ckpt=args.sam2_checkpoint,
        sam2_cfg=args.sam2_config,
        gdino_cfg=gdino_config,
        gdino_ckpt=args.gdino_checkpoint,
        device=args.device,
    )

    tasks = sorted(
        [
            p
            for p in args.dataset_root.iterdir()
            if p.is_dir() and _match_task_domain(p.name, args.task_domain)
        ]
    )
    task_filter = _parse_task_filter(args)
    if task_filter is not None:
        tasks = [p for p in tasks if p.name in task_filter]
    if args.limit > 0:
        tasks = tasks[: args.limit]
    if args.save_run_manifest:
        _write_run_manifest(args, tasks)

    failed_tasks: list[dict[str, str]] = []
    skipped_cams: list[dict[str, str]] = []

    for task_dir in tqdm(tasks, desc="Tasks"):
        task_id = task_dir.name
        try:
            prompt = _get_prompt(task_id, object_list, args.object_group)
            if not prompt:
                print(f"[mask_gen_whole] skip {task_id}: no prompt")
                continue
            if args.add_hands:
                prompt = f"{prompt}. left hand. right hand."

            info = _load_info_json(task_dir)
            fps = float(info.get("fps", 30))
            episodes = _load_episodes(task_dir)
            if args.episode_id is not None:
                episodes = episodes[episodes["episode_index"] == args.episode_id]
            video_keys = _list_video_keys(info, episodes, include_depth=args.include_depth)
            if args.video_key:
                video_keys = [k for k in video_keys if k == args.video_key]
            if not video_keys:
                print(f"[mask_gen_whole] {task_id}: no video keys found after filtering.")
                continue
            if args.debug_video_keys:
                print(f"[mask_gen_whole] {task_id} video_keys: {video_keys}")
                continue

            for _, row in tqdm(episodes.iterrows(), total=len(episodes), desc=f"{task_id}: episodes", leave=False):
                episode_id = int(row["episode_index"])
                for video_key in tqdm(video_keys, desc=f"{task_id}/episode_{episode_id:03d}: cameras", leave=False):
                    out_root = args.output_root / task_id / f"episode_{episode_id:03d}" / video_key
                    index_path = out_root / f"episode_{episode_id:03d}.jsonl"
                    if args.resume and index_path.exists() and index_path.stat().st_size > 0:
                        if args.debug_episode_video:
                            print(
                                f"[mask_gen_whole] resume skip {task_id} episode_{episode_id:03d} {video_key}: "
                                f"existing {index_path}"
                            )
                        continue
                    if args.debug_episode_video:
                        print(f"[mask_gen_whole] processing {task_id} episode_{episode_id:03d} {video_key}")
                    video_path = _render_video_path(task_dir, info, video_key, row)
                    if args.debug_episode_video:
                        print(f"[mask_gen_whole] video path {video_path} exists={video_path.exists()}")
                    if not video_path.exists():
                        skipped_cams.append(
                            {
                                "task_id": task_id,
                                "episode_id": str(episode_id),
                                "video_key": video_key,
                                "reason": f"missing video: {video_path}",
                            }
                        )
                        print(f"[mask_gen_whole] missing video: {video_path}")
                        continue

                    frames_dir: Path | None = None
                    frames_dir_is_external_cache = False
                    try:
                        timestamps = _timestamps_for_episode(row, video_key, fps)
                        frames = _decode_frames(video_path, timestamps, args.video_backend)

                        # prepare output dirs
                        frames_dir, frames_dir_is_external_cache = _prepare_frames_dir(
                            out_root=out_root,
                            frame_cache_root=args.frame_cache_root,
                            task_id=task_id,
                            episode_id=episode_id,
                            video_key=video_key,
                        )

                        for idx, img in enumerate(
                            tqdm(frames, desc=f"{task_id}/episode_{episode_id:03d}/{video_key}: save frames", leave=False)
                        ):
                            _save_frame_jpg(img, frames_dir / f"{idx:06d}.jpg")

                        # ground + init masks on first frame
                        # GroundingDINO deformable attention does not support BF16; run in FP32.
                        input_boxes, labels, init_masks = _ground_first_frame(
                            grounding_model,
                            image_predictor,
                            frames[0],
                            prompt,
                            args.box_threshold,
                            args.text_threshold,
                            args.max_boxes_per_label,
                            args.infer_boxes_per_label_from_prompt,
                            args.device,
                        )
                        if input_boxes is None or len(labels) == 0:
                            skipped_cams.append(
                                {
                                    "task_id": task_id,
                                    "episode_id": str(episode_id),
                                    "video_key": video_key,
                                    "reason": "no boxes found",
                                }
                            )
                            print(
                                f"[mask_gen_whole] {task_id} episode_{episode_id:03d} {video_key}: no boxes found, skipping."
                            )
                            continue

                        # init tracking
                        inference_state = video_predictor.init_state(
                            video_path=str(frames_dir),
                            offload_video_to_cpu=args.offload_video_to_cpu,
                            offload_state_to_cpu=args.offload_state_to_cpu,
                        )
                        for obj_id, (label, mask) in enumerate(zip(labels, init_masks), start=1):
                            video_predictor.add_new_mask(
                                inference_state=inference_state,
                                frame_idx=0,
                                obj_id=obj_id,
                                mask=mask,
                            )

                        # propagate
                        index_records = []
                        propagate_iter = video_predictor.propagate_in_video(inference_state)
                        if args.autocast_bf16:
                            import torch

                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(
                                    propagate_iter,
                                    total=len(frames),
                                    desc=f"{task_id}/episode_{episode_id:03d}/{video_key}: track",
                                    leave=False,
                                ):
                                    frame_dir = out_root / f"frame_{out_frame_idx:06d}"
                                    frame_dir.mkdir(parents=True, exist_ok=True)
                                    if args.save_frame_images:
                                        _save_frame_jpg(frames[out_frame_idx], frame_dir / "frame.jpg")

                                    masks = []
                                    for i, obj_id in enumerate(out_obj_ids):
                                        mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)
                                        mask_path = _save_mask(
                                            mask,
                                            frame_dir / f"instance_{obj_id:02d}_mask",
                                            args.mask_storage,
                                        )
                                        masks.append(mask)
                                        label = labels[obj_id - 1] if obj_id - 1 < len(labels) else f"obj_{obj_id}"
                                        index_records.append(
                                            {
                                                "task_id": task_id,
                                                "episode_id": episode_id,
                                                "video_key": video_key,
                                                "frame_idx": out_frame_idx,
                                                "label": label,
                                                "mask_path": str(mask_path.relative_to(out_root)),
                                            }
                                        )

                                    if args.save_vis_grid:
                                        _save_vis_grid(
                                            frames[out_frame_idx],
                                            masks,
                                            frame_dir / "vis_grid.png",
                                            args.vis_grid_cols,
                                            args.vis_alpha,
                                        )
                        else:
                            for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(
                                propagate_iter,
                                total=len(frames),
                                desc=f"{task_id}/episode_{episode_id:03d}/{video_key}: track",
                                leave=False,
                            ):
                                frame_dir = out_root / f"frame_{out_frame_idx:06d}"
                                frame_dir.mkdir(parents=True, exist_ok=True)
                                if args.save_frame_images:
                                    _save_frame_jpg(frames[out_frame_idx], frame_dir / "frame.jpg")

                                masks = []
                                for i, obj_id in enumerate(out_obj_ids):
                                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)
                                    mask_path = _save_mask(
                                        mask,
                                        frame_dir / f"instance_{obj_id:02d}_mask",
                                        args.mask_storage,
                                    )
                                    masks.append(mask)
                                    label = labels[obj_id - 1] if obj_id - 1 < len(labels) else f"obj_{obj_id}"
                                    index_records.append(
                                        {
                                            "task_id": task_id,
                                            "episode_id": episode_id,
                                            "video_key": video_key,
                                            "frame_idx": out_frame_idx,
                                            "label": label,
                                            "mask_path": str(mask_path.relative_to(out_root)),
                                        }
                                    )

                                if args.save_vis_grid:
                                    _save_vis_grid(
                                        frames[out_frame_idx],
                                        masks,
                                        frame_dir / "vis_grid.png",
                                        args.vis_grid_cols,
                                        args.vis_alpha,
                                    )

                        # write episode index
                        index_path = out_root / f"episode_{episode_id:03d}.jsonl"
                        with index_path.open("w", encoding="utf-8") as f:
                            for rec in index_records:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    except Exception as exc:
                        skipped_cams.append(
                            {
                                "task_id": task_id,
                                "episode_id": str(episode_id),
                                "video_key": video_key,
                                "reason": repr(exc),
                            }
                        )
                        print(
                            f"[mask_gen_whole] error in {task_id} episode_{episode_id:03d} {video_key}: {exc}. Skipping camera."
                        )
                        continue
                    finally:
                        should_remove_frames_dir = frames_dir_is_external_cache or (not args.save_frame_images)
                        if frames_dir is not None and should_remove_frames_dir and frames_dir.exists():
                            shutil.rmtree(frames_dir)
                        if args.empty_cache:
                            import gc
                            import torch

                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
        except Exception as exc:
            failed_tasks.append({"task_id": task_id, "error": repr(exc)})
            print(f"[mask_gen_whole] error in {task_id}: {exc}. Skipping task.")
            continue

    if failed_tasks:
        print("[mask_gen_whole] tasks skipped due to errors:")
        for item in failed_tasks:
            print(f"  - {item['task_id']}: {item['error']}")
    if skipped_cams:
        print("[mask_gen_whole] cameras skipped due to errors:")
        for item in skipped_cams:
            print(
                f"  - {item['task_id']} episode_{item['episode_id']} {item['video_key']}: {item['reason']}"
            )


if __name__ == "__main__":
    main()
