import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlretrieve

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder


def _get_registry_path() -> Path:
    return Path(
        os.getenv(
            "MANI_SKILL_SKETCHFAB_REGISTRY",
            str(PACKAGE_ASSET_DIR / "sketchfab_registry.json"),
        )
    )


def _get_assets_root() -> Path:
    return Path(
        os.getenv(
            "MANI_SKILL_SKETCHFAB_DIR",
            str(ASSET_DIR / "sketchfab" / "objects"),
        )
    )


def _load_registry() -> Dict[str, Dict[str, Any]]:
    path = _get_registry_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Sketchfab registry not found: {path}. "
            "Create mani_skill/assets/sketchfab_registry.json first."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Sketchfab registry must be a JSON object: {path}")
    return data


def _resolve_record(object_key: str) -> Dict[str, Any]:
    registry = _load_registry()
    if object_key not in registry:
        available = ", ".join(sorted(registry.keys())[:20])
        raise KeyError(
            f"Unknown sketchfab object_key={object_key}. "
            f"Available (first 20): {available}"
        )
    record = dict(registry[object_key])
    record.setdefault("object_key", object_key)
    return record


def _resolve_glb_path(record: Dict[str, Any]) -> Path:
    object_key = record.get("object_key")
    if not object_key:
        raise ValueError("Sketchfab record is missing required field: object_key")

    # Canonical layout:
    #   ~/.maniskill/data/sketchfab/objects/{object_key}/model.glb
    filename = record.get("glb_filename", "model.glb")
    canonical_path = _get_assets_root() / object_key / filename
    if canonical_path.exists():
        return canonical_path

    # Backward-compatible fallback for older records.
    legacy_path = record.get("glb_path")
    if legacy_path:
        p = Path(legacy_path)
        if p.is_absolute():
            return p
        return _get_assets_root() / p

    return canonical_path


def _find_any_glb_under_object_dir(object_key: str) -> Optional[Path]:
    object_dir = _get_assets_root() / object_key
    if not object_dir.exists():
        return None
    glb_candidates = sorted(object_dir.glob("*.glb"))
    if not glb_candidates:
        return None
    return glb_candidates[0]


def _normalize_source_url(url: str) -> str:
    u = url.strip()
    if not u:
        return u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return f"https://{u}"


def _is_valid_glb(file_path: Path) -> bool:
    if not file_path.exists() or file_path.stat().st_size < 12:
        return False
    with open(file_path, "rb") as f:
        magic = f.read(4)
    return magic == b"glTF"


def _ensure_glb_file(record: Dict[str, Any]) -> Path:
    object_key = record.get("object_key")
    if not object_key:
        raise ValueError("Sketchfab record is missing required field: object_key")

    glb_path = _resolve_glb_path(record)
    if glb_path.exists():
        if not _is_valid_glb(glb_path):
            raise ValueError(f"Found file but it is not a valid .glb: {glb_path}")
        return glb_path

    fallback_glb = _find_any_glb_under_object_dir(object_key)
    if fallback_glb is not None:
        if not _is_valid_glb(fallback_glb):
            raise ValueError(
                f"Found fallback .glb but it is invalid: {fallback_glb}"
            )
        return fallback_glb

    source_url = record.get("source_url", "")
    if not source_url:
        raise FileNotFoundError(
            f"Sketchfab GLB not found at {glb_path}, and source_url is empty."
        )
    source_url = _normalize_source_url(source_url)

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(source_url, str(glb_path))
    if not glb_path.exists():
        raise FileNotFoundError(f"Download finished but file not found: {glb_path}")
    if not _is_valid_glb(glb_path):
        raise ValueError(
            f"Downloaded file is not a valid .glb: {glb_path}. "
            f"source_url={source_url}"
        )
    return glb_path


def _resolve_scale(record: Dict[str, Any], scales: Optional[List[float]]) -> List[float]:
    if scales is not None:
        if len(scales) == 1:
            s = float(scales[0])
            return [s, s, s]
        if len(scales) == 3:
            return [float(scales[0]), float(scales[1]), float(scales[2])]
        raise ValueError(f"Invalid scales length={len(scales)} for sketchfab actor")

    scale = record.get("scale", [1.0, 1.0, 1.0])
    if isinstance(scale, (int, float)):
        s = float(scale)
        return [s, s, s]
    if isinstance(scale, list) and len(scale) == 3:
        return [float(scale[0]), float(scale[1]), float(scale[2])]
    raise ValueError("Invalid scale in sketchfab registry; expected number or 3-list")


def get_sketchfab_builder(
    scene: ManiSkillScene,
    object_key: str,
    add_collision: bool = True,
    add_visual: bool = True,
    scales: Optional[List[float]] = None,
) -> ActorBuilder:
    record = _resolve_record(object_key)
    glb_file = _ensure_glb_file(record)
    scale = _resolve_scale(record, scales)
    collision_mode = record.get("collision_mode", "convex")

    builder = scene.create_actor_builder()
    if add_collision:
        if collision_mode == "nonconvex":
            builder.add_nonconvex_collision_from_file(filename=str(glb_file), scale=scale)
        else:
            builder.add_multiple_convex_collisions_from_file(
                filename=str(glb_file), scale=scale
            )
    if add_visual:
        builder.add_visual_from_file(filename=str(glb_file), scale=scale)
    return builder


def create_sketchfab_actor(
    scene: ManiSkillScene,
    object_key: str,
    pose,
    name: Optional[str] = None,
    is_static: bool = False,
    scales: Optional[List[float]] = None,
):
    builder = get_sketchfab_builder(scene, object_key=object_key, scales=scales)
    builder.set_physx_body_type("static" if is_static else "dynamic")
    builder.initial_pose = pose
    actor = builder.build(name=name or object_key)
    actor.set_pose(pose)
    return actor
