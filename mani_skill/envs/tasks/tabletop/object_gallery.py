from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import sapien
import torch
from gymnasium.vector.utils import batch_space
from transforms3d.euler import euler2quat

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.sim_assets_catigory import ROBOTWIN_MODEL_IDS
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.actors.sketchfab import create_sketchfab_actor
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.utils import create_actor


def _load_gallery_catalog() -> Dict[str, Any]:
    catalog_path = Path(__file__).with_name("object_gallery_catalog.json")
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]+", "_", value).strip("_")


def _as_xyz_scale(scale: Any) -> Optional[Tuple[float, float, float]]:
    if scale is None:
        return None
    if isinstance(scale, dict):
        scale_factor = scale.get("scale_factor")
        if scale_factor is None:
            return None
        s = float(scale_factor)
        return (s, s, s)
    if isinstance(scale, (int, float)):
        s = float(scale)
        return (s, s, s)
    if isinstance(scale, Sequence) and len(scale) == 3:
        return (float(scale[0]), float(scale[1]), float(scale[2]))
    raise ValueError(f"Unsupported gallery scale: {scale}")


def _first_numeric_half_size(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Sequence) and len(value) == 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return None


def _as_rgba_color(
    value: Any, default: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    if value is None:
        return default
    if isinstance(value, Sequence) and len(value) in {3, 4}:
        rgba = [float(v) for v in value]
        if len(rgba) == 3:
            rgba.append(1.0)
        return tuple(rgba)
    raise ValueError(f"Unsupported RGBA color: {value}")


@register_env("ObjectGallery-v1", max_episode_steps=1)
class ObjectGalleryEnv(BaseEnv):
    """Static gallery of all objects used by the tabletop dual-task environments."""

    SUPPORTED_ROBOTS = ["none", "panda", ("panda", "panda")]
    SUPPORTED_REWARD_MODES = ("none",)

    group_order = ("robotwin", "ycb", "sketchfab", "partnet", "primitive")
    hdri_presets = {
        "default": Path(__file__).resolve().parents[3]
        / "assets/environment_maps/default.hdr",
        "autumn": Path(__file__).resolve().parents[3]
        / "utils/scene_builder/replicacad/autumn_field_puresky_4k.hdr",
        "misty": Path(__file__).resolve().parents[3]
        / "examples/benchmarking/envs/maniskill/kloofendal_28d_misty_puresky_1k.hdr",
        "overcast": Path(__file__).resolve().parents[3]
        / "examples/benchmarking/envs/maniskill/overcast.exr",
    }

    def __init__(
        self,
        *args,
        robot_uids: str | Tuple[str, str] = "none",
        obs_mode: str = "none",
        reward_mode: str = "none",
        robot_init_qpos_noise: float = 0.0,
        grid_cols: int = 12,
        grid_spacing: float = 0.2,
        footprint_padding: float = 0.0,
        max_slot_span: int = 7,
        robotwin_variant_mode: str = "used_ids",
        hdri_background: Optional[str] = "misty",
        backdrop_enabled: bool = False,
        backdrop_color: Optional[Sequence[float]] = None,
        stage_front_y: float = 1.5,
        stage_back_y: float = -1,
        dual_robot_base_y: float = 1.7,
        dual_robot_x_offset: float = 0.45,
        robot_stage_margin: float = 0.12,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.grid_cols = grid_cols
        self.grid_spacing = grid_spacing
        self.footprint_padding = footprint_padding
        self.max_slot_span = max_slot_span
        self.robotwin_variant_mode = robotwin_variant_mode
        self.hdri_background = hdri_background
        self.backdrop_enabled = backdrop_enabled
        self.backdrop_color = _as_rgba_color(
            backdrop_color, default=(0.95, 0.95, 0.96, 1.0)
        )
        self.stage_front_y = float(stage_front_y)
        self.stage_back_y = float(stage_back_y)
        if self.stage_front_y <= self.stage_back_y:
            raise ValueError(
                f"stage_front_y({self.stage_front_y}) must be greater than stage_back_y({self.stage_back_y})"
            )
        self.dual_robot_base_y = dual_robot_base_y
        self.dual_robot_x_offset = dual_robot_x_offset
        self.robot_stage_margin = robot_stage_margin
        self.catalog = self._prepare_gallery_catalog(_load_gallery_catalog())
        self.gallery_items: List[Dict[str, Any]] = []
        self.gallery_objects: List[Any] = []
        self._object_initial_poses: List[sapien.Pose] = []
        self._object_layout_poses: List[sapien.Pose] = []
        self._object_z_offsets: List[float] = []
        super().__init__(
            *args,
            robot_uids=robot_uids,
            obs_mode=obs_mode,
            reward_mode=reward_mode,
            **kwargs,
        )
        if self.agent is None:
            self.single_action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(0,), dtype=np.float32
            )
            self.action_space = batch_space(self.single_action_space, n=self.num_envs)
            self._orig_single_action_space = self.single_action_space

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[3.2, -4.4, 3.0], target=[0.0, 0.0, 0.15])
        return [CameraConfig("base_camera", pose, 32, 32, 0.95, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[3.2, -4.4, 3.0], target=[0.0, 0.0, 0.15])
        return CameraConfig("render_camera", pose, 64, 64, 0.95, 0.01, 100)

    @property
    def _default_viewer_camera_configs(self):
        pose = sapien_utils.look_at(eye=[3.2, -4.4, 3.0], target=[0.0, 0.0, 0.15])
        return CameraConfig("viewer", pose, 64, 64, 0.95, 0.01, 100)

    def _load_agent(self, options: dict):
        if self.robot_uids == "none":
            super()._load_agent(options)
        elif self.robot_uids == ("panda", "panda"):
            super()._load_agent(options, self._dual_panda_base_poses())
        else:
            super()._load_agent(options, sapien.Pose(p=[-2.0, 0.0, 0.0]))

    def _resolve_hdri_background(self) -> Optional[Path]:
        if self.hdri_background in {None, "", "none"}:
            return None
        if self.hdri_background in self.hdri_presets:
            return self.hdri_presets[self.hdri_background]
        hdri_path = Path(self.hdri_background).expanduser()
        if not hdri_path.is_absolute():
            hdri_path = (Path.cwd() / hdri_path).resolve()
        return hdri_path

    def _load_lighting(self, options: dict):
        super()._load_lighting(options)
        hdri_path = self._resolve_hdri_background()
        if hdri_path is None:
            return
        if not hdri_path.exists():
            raise FileNotFoundError(f"HDRI background not found: {hdri_path}")
        for sub_scene in self.scene.sub_scenes:
            sub_scene.set_environment_map(str(hdri_path))

    def _dual_panda_base_poses(self) -> List[sapien.Pose]:
        half_x, stage_back_y, stage_front_y = self._gallery_stage_bounds()
        # Keep the dual-arm bases on top of the stage even if the table size changes.
        y = min(
            self.dual_robot_base_y,
            max(0.0, stage_front_y - self.robot_stage_margin),
        )
        x = min(self.dual_robot_x_offset, max(0.0, half_x - self.robot_stage_margin))
        q = euler2quat(0, 0, -np.pi / 2)
        return [
            sapien.Pose(p=[-x, y, 0.0], q=q),
            sapien.Pose(p=[x, y, 0.0], q=q),
        ]

    def _load_scene(self, options: dict):
        self._build_gallery_stage()
        self._build_gallery_objects()

    def _gallery_stage_bounds(self) -> Tuple[float, float, float]:
        half_x = self.grid_cols * self.grid_spacing * 0.5 + 0.25
        return half_x, self.stage_back_y, self.stage_front_y

    def _gallery_stage_half_extents(self) -> Tuple[float, float]:
        half_x, stage_back_y, stage_front_y = self._gallery_stage_bounds()
        half_y = (stage_front_y - stage_back_y) * 0.5
        return half_x, half_y

    def _build_gallery_stage(self):
        half_x, stage_back_y, stage_front_y = self._gallery_stage_bounds()
        half_y = (stage_front_y - stage_back_y) * 0.5
        center_y = (stage_front_y + stage_back_y) * 0.5
        self.gallery_stage = actors.build_box(
            self.scene,
            half_sizes=[half_x, half_y, 0.02],
            color=[0.92, 0.92, 0.88, 1.0],
            name="gallery_stage",
            body_type="static",
            add_collision=True,
            # Allow asymmetrically shortening only the far side of the stage by
            # shifting the box center while keeping the front edge fixed.
            initial_pose=sapien.Pose(p=[0.0, center_y, -0.02]),
        )
        if self.backdrop_enabled:
            self._build_gallery_backdrop(half_x=half_x, stage_front_y=stage_front_y)

    def _build_gallery_backdrop(self, half_x: float, stage_front_y: float):
        self.gallery_backdrop = actors.build_box(
            self.scene,
            half_sizes=[half_x + 1.0, 0.02, 1.8],
            color=self.backdrop_color,
            name="gallery_backdrop",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.0, stage_front_y + 0.45, 1.8]),
        )

    def _prepare_gallery_catalog(self, catalog: Dict[str, Any]) -> Dict[str, Any]:
        prepared = {key: value for key, value in catalog.items()}
        prepared["robotwin"] = self._expand_robotwin_entries(catalog.get("robotwin", []))
        return prepared

    def _expand_robotwin_entries(
        self, entries: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if self.robotwin_variant_mode == "single":
            return [dict(entry) for entry in entries]
        if self.robotwin_variant_mode not in {"used_ids", "all_ids"}:
            raise ValueError(
                f"Unsupported robotwin_variant_mode: {self.robotwin_variant_mode}"
            )

        expanded: List[Dict[str, Any]] = []
        for entry in entries:
            name = entry["name"]
            current_id = entry.get("model_id", 0)
            available_ids = ROBOTWIN_MODEL_IDS.get(name, {}).get("model_ids", [])
            default_id = ROBOTWIN_MODEL_IDS.get(name, {}).get("default", current_id)
            if self.robotwin_variant_mode == "used_ids":
                candidate_ids = entry.get("used_model_ids", [])
            else:
                candidate_ids = available_ids

            base_id = current_id
            if available_ids and base_id not in available_ids:
                base_id = default_id

            model_ids = [base_id]
            model_ids.extend(
                model_id
                for model_id in candidate_ids
                if (not available_ids or model_id in available_ids)
                and model_id not in model_ids
            )
            if not model_ids:
                model_ids = [current_id]

            for model_id in model_ids:
                expanded_entry = dict(entry)
                expanded_entry["model_id"] = model_id
                expanded_entry["gallery_name"] = f"{name}_id{model_id}"
                expanded.append(expanded_entry)
        return expanded

    def _preliminary_xy(self, index: int) -> Tuple[float, float]:
        row = index // self.grid_cols
        col = index % self.grid_cols
        x = (col - (self.grid_cols - 1) * 0.5) * self.grid_spacing
        y = (
            (math.ceil(self._gallery_count / self.grid_cols) - 1) * 0.5 - row
        ) * self.grid_spacing
        return x, y

    def _build_gallery_objects(self):
        self.gallery_items = []
        self.gallery_objects = []
        self._object_initial_poses = []
        self._object_layout_poses = []

        self._gallery_count = sum(
            len(self.catalog[group]) for group in self.group_order
        )
        index = 0
        for group in self.group_order:
            for entry in self.catalog[group]:
                x, y = self._preliminary_xy(index)
                q = entry.get("gallery_pose", {}).get("q", [1.0, 0.0, 0.0, 0.0])
                pose = sapien.Pose(p=[x, y, 0.0], q=q)
                item_name = self._entry_name(group, entry)
                actor_name = f"gallery_{index:03d}_{group}_{_clean_name(item_name)}"
                obj = self._build_entry(group, entry, pose, actor_name)
                self.gallery_items.append(
                    {
                        "index": index,
                        "group": group,
                        "name": item_name,
                        "actor_name": actor_name,
                        "catalog_entry": entry,
                    }
                )
                self.gallery_objects.append(obj)
                self._object_initial_poses.append(pose)
                self._object_layout_poses.append(pose)
                index += 1

    def _entry_name(self, group: str, entry: Dict[str, Any]) -> str:
        if group == "robotwin":
            return entry.get("gallery_name", entry["name"])
        if group == "ycb":
            return entry["id"]
        if group == "sketchfab":
            return entry["object_key"]
        if group == "partnet":
            return f"{entry['mode']}_{entry['model_id']}"
        if group == "primitive":
            return entry["kind"]
        raise ValueError(f"Unknown gallery group: {group}")

    def _build_entry(
        self,
        group: str,
        entry: Dict[str, Any],
        pose: sapien.Pose,
        actor_name: str,
    ):
        if group == "robotwin":
            scale = _as_xyz_scale(entry.get("gallery_scale"))
            if entry["name"] == "104_board" and scale is not None:
                # 104_board is rotated 90 deg about x in the gallery, so local y
                # becomes the vertical thickness in world space. Shrink only that
                # axis to make the board look flatter without changing its footprint.
                scale = (scale[0], 0.2, scale[2])
            robotwin_obj = create_actor(
                scene=self.scene,
                pose=pose,
                modelname=entry["name"],
                scale=scale,
                replace_scale=bool(entry.get("gallery_replace_scale", False)),
                convex=True,
                is_static=True,
                model_id=entry.get("model_id", 0),
                name=actor_name,
            )
            return robotwin_obj.actor

        if group == "ycb":
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{entry['id']}",
                scales=entry.get("gallery_scales"),
            )
            builder.initial_pose = pose
            return builder.build_static(name=actor_name)

        if group == "sketchfab":
            return create_sketchfab_actor(
                scene=self.scene,
                object_key=entry["object_key"],
                pose=pose,
                name=actor_name,
                is_static=True,
                scales=entry.get("gallery_scales"),
            )

        if group == "partnet":
            builder = articulations.get_articulation_builder(
                self.scene,
                id=f"partnet-mobility:{entry['model_id']}",
                mode=entry.get("mode"),
                scale=entry.get("gallery_scale"),
            )
            builder.initial_pose = pose
            return builder.build(name=actor_name, fix_root_link=True)

        if group == "primitive":
            half_size = self._primitive_half_size(entry)
            if entry["kind"] == "cube":
                return actors.build_cube(
                    self.scene,
                    half_size=float(half_size),
                    color=[0.1, 0.45, 0.95, 1.0],
                    name=actor_name,
                    body_type="static",
                    initial_pose=pose,
                )
            return actors.build_box(
                self.scene,
                half_sizes=half_size,
                color=[0.9, 0.25, 0.25, 0.55],
                name=actor_name,
                body_type="static",
                add_collision=False,
                initial_pose=pose,
            )

        raise ValueError(f"Unknown gallery group: {group}")

    def _primitive_half_size(self, entry: Dict[str, Any]):
        half_size = _first_numeric_half_size(entry.get("gallery_half_size"))
        if half_size is not None:
            return half_size
        for usage in entry.get("size_usages", []):
            half_size = _first_numeric_half_size(usage.get("half_size"))
            if half_size is not None:
                return half_size
        if entry["kind"] == "box":
            return [0.08, 0.08, 0.005]
        if entry["kind"] == "cube":
            return 0.02
        raise ValueError(f"Unknown primitive kind: {entry['kind']}")

    def _after_reconfigure(self, options: dict):
        footprints = []
        self._object_z_offsets = []
        for obj, item in zip(self.gallery_objects, self.gallery_items):
            footprint_x, footprint_y, z_offset = self._object_footprint(obj, item)
            footprints.append((footprint_x, footprint_y))
            self._object_z_offsets.append(z_offset)
        self._object_layout_poses = self._pack_gallery_layout(footprints)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_gallery_robots(env_idx)
        for obj, layout_pose, z_offset in zip(
            self.gallery_objects, self._object_layout_poses, self._object_z_offsets
        ):
            pose = sapien.Pose(
                p=[layout_pose.p[0], layout_pose.p[1], z_offset],
                q=layout_pose.q,
            )
            if hasattr(obj, "set_root_pose"):
                obj.set_root_pose(pose)
            else:
                obj.set_pose(pose)

    def _object_footprint(self, obj, item: Dict[str, Any]) -> Tuple[float, float, float]:
        entry = item["catalog_entry"]
        if item["group"] == "primitive":
            half_size = self._primitive_half_size(entry)
            if isinstance(half_size, (int, float)):
                return float(half_size) * 2, float(half_size) * 2, float(half_size)
            return float(half_size[0]) * 2, float(half_size[1]) * 2, float(half_size[2])

        mesh = obj.get_first_collision_mesh()
        if mesh is None and hasattr(obj, "get_first_visual_mesh"):
            mesh = obj.get_first_visual_mesh()
        if mesh is None:
            return self.grid_spacing, self.grid_spacing, 0.0
        bounds = mesh.bounding_box.bounds
        extents = bounds[1] - bounds[0]
        return float(extents[0]), float(extents[1]), float(-bounds[0, 2])

    def _pack_gallery_layout(
        self, footprints: List[Tuple[float, float]]
    ) -> List[sapien.Pose]:
        stage_half_x, _ = self._gallery_stage_half_extents()
        width_limit = stage_half_x * 2
        # Use continuous rectangle packing so the placed objects can touch their
        # footprint boxes exactly instead of being expanded to discrete grid cells.
        placements: List[Tuple[float, float, float, float]] = []
        for footprint_x, footprint_y in footprints:
            padded_x = footprint_x + self.footprint_padding
            padded_y = footprint_y + self.footprint_padding
            candidate_xs = {0.0}
            candidate_ys = {0.0}
            for placed_x, placed_y, placed_w, placed_h in placements:
                candidate_xs.add(placed_x + placed_w)
                candidate_ys.add(placed_y + placed_h)

            best_placement = None
            for candidate_y in sorted(candidate_ys):
                for candidate_x in sorted(candidate_xs):
                    if candidate_x + padded_x > width_limit + 1e-9:
                        continue
                    overlaps = any(
                        not (
                            candidate_x + padded_x <= placed_x + 1e-9
                            or placed_x + placed_w <= candidate_x + 1e-9
                            or candidate_y + padded_y <= placed_y + 1e-9
                            or placed_y + placed_h <= candidate_y + 1e-9
                        )
                        for placed_x, placed_y, placed_w, placed_h in placements
                    )
                    if overlaps:
                        continue
                    best_placement = (
                        candidate_x,
                        candidate_y,
                        padded_x,
                        padded_y,
                    )
                    break
                if best_placement is not None:
                    break
            if best_placement is None:
                raise RuntimeError(
                    f"Failed to pack gallery object footprint {footprint_x} x {footprint_y} into width {width_limit}"
                )
            placements.append(best_placement)

        total_depth = max(
            (placed_y + placed_h for _, placed_y, _, placed_h in placements),
            default=0.0,
        )

        poses = []
        for item, initial_pose, placement in zip(
            self.gallery_items, self._object_initial_poses, placements
        ):
            placed_x, placed_y, padded_x, padded_y = placement
            footprint_x = padded_x - self.footprint_padding
            footprint_y = padded_y - self.footprint_padding
            box_x = placed_x + self.footprint_padding * 0.5
            box_y = placed_y + self.footprint_padding * 0.5
            x = box_x + footprint_x * 0.5 - width_limit * 0.5
            y = total_depth * 0.5 - (box_y + footprint_y * 0.5)
            item["layout"] = {
                "x0": box_x,
                "y0": box_y,
                "width": footprint_x,
                "depth": footprint_y,
            }
            poses.append(sapien.Pose(p=[x, y, 0.0], q=initial_pose.q))
        return poses

    def _initialize_gallery_robots(self, env_idx: torch.Tensor):
        if self.agent is None:
            return

        b = len(env_idx)
        qpos = np.array(
            [
                0.0,
                np.pi / 8,
                0,
                -np.pi * 5 / 8,
                0,
                np.pi * 3 / 4,
                np.pi / 4,
                0.04,
                0.04,
            ]
        )
        qpos = np.repeat(qpos[None, :], b, axis=0)
        if self.robot_init_qpos_noise > 0:
            qpos += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, size=qpos.shape
            )
            qpos[:, -2:] = 0.04

        if self.robot_uids == ("panda", "panda"):
            base_poses = self._dual_panda_base_poses()
            self.agent.agents[0].reset(qpos)
            self.agent.agents[0].robot.set_pose(base_poses[0])
            self.agent.agents[1].reset(qpos)
            self.agent.agents[1].robot.set_pose(base_poses[1])
        elif self.robot_uids == "panda":
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([-2.0, 0.0, 0.0]))

    def evaluate(self):
        return {
            "success": torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        }

    def get_state_dict(self):
        if self.agent is None:
            return self.scene.get_sim_state()
        return super().get_state_dict()

    def _step_action(self, action):
        if self.agent is not None:
            return super()._step_action(action)

        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self._before_simulation_step()
            self.scene.step()
            self._after_simulation_step()
        self._after_control_step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        return action


@register_env("ObjectGalleryTwoPanda-v1", max_episode_steps=1)
class ObjectGalleryTwoPandaEnv(ObjectGalleryEnv):
    def __init__(self, *args, robot_uids=("panda", "panda"), **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
