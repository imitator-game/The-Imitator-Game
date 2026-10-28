"""Strict full-episode causal VAE preparation with PTS-checked HEVC fallback.

Encoding and temporal alignment are retained from the validated streaming run;
selection is delegated to the public protocol configuration.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from .config import REPO_ROOT
from .config import outside_repository
from .config import selected_episodes

IMAGE_KEY = "observation.images.zed2i"
ACTION_KEY = "action.qpos_gripper_actions"
STATE_KEY = "observation.qpos_gripper_states"


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def file_identity(path, content_hash=False):
    path = Path(path).resolve()
    st = path.stat()
    result = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if content_hash:
        result["sha256"] = digest_bytes(path.read_bytes())
    return result


def select_episodes(spec):
    if ":" not in spec:
        raise ValueError(f"Expected start:stop episode selection, got {spec!r}")
    start, stop = map(int, spec.split(":"))
    if start < 0 or stop <= start:
        raise ValueError(spec)
    return list(range(start, stop))


def base_task(repo_id):
    return repo_id.split("TwoRobot", 1)[1].rsplit("-v", 1)[0]


def sample_plan(length, fps, kind, contract):
    """Keep physical time. Pad the last frame explicitly to Wan's 1+4*k format."""
    import numpy as np

    if length < 1 or fps <= 0:
        raise ValueError(f"Invalid length/fps {length}/{fps}")
    cfg = contract["video"][kind]
    if kind == "sim":
        stride = cfg["source_frame_stride"]
        # Each latent after the initial history group supervises stride*4
        # actions. Observation endpoints alone miss action D*k when T=D*k+1.
        # New episode contracts cover ALL action rows, repeating only the RGB
        # tail where needed; historical contracts retain their original plan.
        covered_rows = length if cfg.get("action_tail_policy") == "all_actions_v1" else length - 1
        latent_intervals = math.ceil(covered_rows / (stride * 4))
        nominal = np.arange(1 + 4 * latent_intervals, dtype=np.int64) * stride
        valid = nominal < length
        indices = np.minimum(nominal, length - 1)
    else:
        target_fps = cfg["target_fps"]
        n = max(1, int(round(length / fps * target_fps)))
        raw = np.unique(
            np.clip(np.floor(np.arange(n, dtype=np.float64) * fps / target_fps).astype(np.int64), 0, length - 1)
        )
        target_n = 1 + 4 * math.ceil((len(raw) - 1) / 4)
        indices = np.pad(raw, (0, target_n - len(raw)), mode="edge")
        valid = np.arange(target_n) < len(raw)
        nominal = indices.copy()
    return {
        "sampled_frame_indices": indices,
        "sample_valid_mask": valid,
        "latent_anchor_indices": indices[::4],
        "latent_nominal_anchor_indices": nominal[::4],
    }


@functools.lru_cache(maxsize=4)
def _read_data_table(paths, kind):
    import pyarrow.parquet as pq

    cols = ["episode_index", "timestamp", "frame_index"]
    if kind == "sim":
        cols += [ACTION_KEY, STATE_KEY]
    return pq.read_table(list(paths), columns=cols)


def read_episode(record):
    import numpy as np
    import pyarrow.compute as pc

    table = _read_data_table(tuple(record["source"]["data_paths"]), record["kind"])
    table = table.filter(pc.equal(table["episode_index"], record["episode_index"]))
    if len(table) != record["source"]["episode_length"]:
        raise ValueError(
            f"{record['repo_id']} episode {record['episode_index']}: "
            f"{len(table)} rows != metadata length "
            f"{record['source']['episode_length']}"
        )
    frame_indices = table["frame_index"].to_numpy().astype(np.int64)
    timestamps = table["timestamp"].to_numpy().astype(np.float64)
    order = np.argsort(frame_indices)
    frame_indices, timestamps = frame_indices[order], timestamps[order]
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError(f"Non-contiguous episode frame_index: {record['repo_id']}")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Invalid/non-monotonic timestamps: {record['repo_id']}")
    fps = record["source"]["fps"]
    # Stored timestamps are float32; permit sub-millisecond roundoff, never a frame shift.
    if np.max(np.abs(timestamps - frame_indices / fps)) > max(1e-3, 0.05 / fps):
        raise ValueError(f"Timestamp/frame/fps mismatch: {record['repo_id']}")
    result = {"frame_indices": frame_indices, "timestamps": timestamps}
    if record["kind"] == "sim":
        actions = np.asarray(table[ACTION_KEY].to_pylist(), dtype=np.float32)[order]
        states = np.asarray(table[STATE_KEY].to_pylist(), dtype=np.float32)[order]
        if actions.shape != (len(table), 16) or states.shape != (len(table), 18):
            raise ValueError(f"Unexpected action/state shape: {actions.shape}/{states.shape}")
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError(f"Non-finite actions/states: {record['repo_id']}")
        result.update(actions=actions, states=states)
    return result


def load_task_mapping(contract):
    """Use the selected mapping, never an implicit repository default."""
    path = contract["paths"].get("task_mapping")
    expected = contract.get("task_mapping_identity")
    if not path or not expected:
        raise ValueError("Cache contract lacks task mapping identity; rebuild the manifest")
    identity = file_identity(path, True)
    if identity["path"] != expected["path"] or identity["sha256"] != expected["sha256"]:
        raise ValueError("Task mapping changed; rebuild the manifest")
    return read_json(path)["task_mappings"], identity


def episode_encoding_identity(record):
    """Identify encoded episode data independently of task pairing/cache location."""
    return digest_json({key: record[key] for key in ("kind", "repo_id", "episode_index", "source")})


def validate_cached_payload(payload, record, contract_hash, encoder_hash):
    # Legacy payloads carry all source fields, so their encoding identity can
    # also be checked without trusting the pairing-dependent record hash.
    try:
        expected = episode_encoding_identity(record)
        actual = episode_encoding_identity(payload)
        compatible = (
            actual == expected
            and payload.get("encoding_sha256", actual) == expected
            and payload.get("data_contract_sha256") == contract_hash
            and payload.get("encoder_fingerprint") == encoder_hash
        )
    except (KeyError, TypeError, ValueError):
        compatible = False
    if not compatible:
        raise RuntimeError(f"Refusing stale/incompatible cache: {record['cache_path']}")


def make_records(contract, cache_root):
    import pyarrow.parquet as pq

    config_root = Path(contract["paths"]["imitator_repo"]) / "examples/baselines/lerobot_dataset"
    mappings, mapping_identity = load_task_mapping(contract)
    sim_to_human = {s: m["human_task_id"] for m in mappings for s in m["sim_task_id"]}
    expected = set(contract["tasks"]["c15"])
    records = {"sim": [], "human": []}
    config_identities = {"task_mapping": mapping_identity}
    human_to_task = {}
    for m in mappings:
        for env in m["sim_task_id"]:
            if base_task(env) in expected:
                human_to_task[m["human_task_id"]] = base_task(env)

    for kind in ("sim", "human"):
        config_path = Path(
            contract["paths"].get(f"{kind}_config", config_root / f"config/exp_configs/{kind}_train_config_15.json")
        )
        entries = read_json(config_path)
        config_identities[kind] = file_identity(config_path, True)
        if kind == "sim" and {base_task(x["repo_id"]) for x in entries} != expected:
            raise ValueError("C15 config membership differs from paper contract")
        source_root = Path(contract["paths"][f"{kind}_root"])
        for entry in entries:
            ids = select_episodes(entry["train"])
            repo_id = entry["repo_id"]
            repo_path = (source_root / entry["root"]).resolve()
            info_path = repo_path / "meta/info.json"
            info = read_json(info_path)
            if IMAGE_KEY not in info["features"]:
                raise ValueError(f"Missing ZED2i RGB in {repo_path}")
            fps = float(info["fps"])
            video_info = info["features"][IMAGE_KEY]["info"]
            video_fps = float(video_info["video.fps"])
            if abs(video_fps - fps) > 1e-6:
                raise ValueError(f"Video/data fps mismatch in {repo_id}")
            if kind == "sim" and info["features"][ACTION_KEY]["shape"] != [16]:
                raise ValueError(f"Action schema mismatch: {repo_id}")
            meta_paths = sorted((repo_path / "meta/episodes").glob("*/*.parquet"))
            if not meta_paths:
                raise FileNotFoundError(f"No LeRobot v3 episode metadata: {repo_path}")
            meta = pq.read_table(meta_paths)
            required = [
                "episode_index",
                "length",
                "data/chunk_index",
                "data/file_index",
                f"videos/{IMAGE_KEY}/chunk_index",
                f"videos/{IMAGE_KEY}/file_index",
                f"videos/{IMAGE_KEY}/from_timestamp",
                f"videos/{IMAGE_KEY}/to_timestamp",
            ]
            rows = {int(x["episode_index"]): x for x in meta.select(required).to_pylist()}
            if len(rows) != len(meta):
                raise ValueError(f"Duplicate episode metadata: {repo_id}")
            meta_identity = [file_identity(p, True) for p in meta_paths]
            for ep in ids:
                row = rows[ep]
                video_path = repo_path / info["video_path"].format(
                    video_key=IMAGE_KEY,
                    chunk_index=int(row[f"videos/{IMAGE_KEY}/chunk_index"]),
                    file_index=int(row[f"videos/{IMAGE_KEY}/file_index"]),
                )
                # Include all declared files; read_episode filters strictly by episode_index.
                # This also supports episodes spanning a data shard boundary.
                data_paths = sorted((repo_path / "data").glob("*/*.parquet"))
                source = {
                    "repo_path": str(repo_path),
                    "info": file_identity(info_path, True),
                    "episode_metadata": meta_identity,
                    "fps": fps,
                    "video_fps": video_fps,
                    "source_image_shape": info["features"][IMAGE_KEY]["shape"],
                    "episode_length": int(row["length"]),
                    "video_path": str(video_path),
                    "video_identity": file_identity(video_path),
                    "video_from_timestamp": float(row[f"videos/{IMAGE_KEY}/from_timestamp"]),
                    "video_to_timestamp": float(row[f"videos/{IMAGE_KEY}/to_timestamp"]),
                    "data_paths": [str(p) for p in data_paths],
                    "data_identities": [file_identity(p) for p in data_paths],
                }
                if not data_paths:
                    raise FileNotFoundError(f"No data parquet: {repo_path}")
                if kind == "sim":
                    source["normalization_identity"] = file_identity(repo_path / "meta/stats.json", True)
                task = base_task(repo_id) if kind == "sim" else human_to_task.get(repo_id, repo_id)
                record = {
                    "kind": kind,
                    "repo_id": repo_id,
                    "episode_index": ep,
                    "base_task": task,
                    "level": repo_id[:2] if kind == "sim" else None,
                    "human_repo_id": sim_to_human[repo_id] if kind == "sim" else repo_id,
                    "cache_path": str((cache_root / kind / repo_id / f"episode_{ep:06d}.pt").resolve()),
                    "source": source,
                }
                plan = sample_plan(source["episode_length"], fps, kind, contract)
                record["expected_latent_frames"] = len(plan["latent_anchor_indices"])
                record["expected_sampled_frames"] = len(plan["sampled_frame_indices"])
                record["encoding_sha256"] = episode_encoding_identity(record)
                record["record_sha256"] = digest_json(record)
                records[kind].append(record)
    expected_sim = int(contract["tasks"].get("sim_training_episodes", len(records["sim"])))
    expected_human = int(contract["tasks"].get("human_training_episodes", len(records["human"])))
    if len(records["sim"]) != expected_sim or len(records["human"]) != expected_human:
        raise ValueError(f"Unexpected single-task episode budget: {len(records['sim'])}/{len(records['human'])}")
    selected_humans = {record["repo_id"] for record in records["human"]}
    if any(record["human_repo_id"] not in selected_humans for record in records["sim"]):
        raise ValueError("Task mapping refers to an unselected human repository")
    return records, config_identities


def validate_sources(record, verified=None):
    source = record["source"]
    extra = [source["normalization_identity"]] if "normalization_identity" in source else []
    identities = [
        source["video_identity"],
        *source["data_identities"],
        source["info"],
        *source["episode_metadata"],
        *extra,
    ]
    for identity in identities:
        if verified is not None and verified.get(identity["path"]) == identity:
            continue
        if file_identity(identity["path"], "sha256" in identity) != identity:
            raise RuntimeError(f"Source changed since manifest: {identity['path']}")
        if verified is not None:
            verified[identity["path"]] = identity


class VideoReader:
    """Decode requested episode frames only, with real PTS checks and bounded RAM."""

    def __init__(self, path, num_threads=1):
        import av

        self.path, self.num_threads = path, num_threads
        with av.open(path) as container:
            stream = container.streams.video[0]
            self.fps = float(stream.average_rate)
            self.length = int(stream.frames or round(float(stream.duration * stream.time_base) * self.fps))
        self.ids = None
        self.frames = self.times = None

    def get_avg_fps(self):
        return self.fps

    def __len__(self):
        return self.length

    def _read(self, ids):
        import numpy as np
        import av

        ids = np.asarray(ids, dtype=np.int64)
        if self.ids is not None and np.array_equal(ids, self.ids):
            return
        wanted = set(ids.tolist())
        found, times = {}, {}

        def decode(seek):
            found, times = {}, {}
            with av.open(self.path) as container:
                stream = container.streams.video[0]
                stream.codec_context.thread_count = self.num_threads
                if seek:
                    start = max(0.0, (int(ids.min()) - 2) / self.fps)
                    container.seek(int(start / float(stream.time_base)), stream=stream, backward=True)
                for frame in container.decode(stream):
                    if frame.time is None:
                        raise RuntimeError("Missing video PTS: " + self.path)
                    timestamp = float(frame.time)
                    index = int(round(timestamp * self.fps))
                    if index in wanted:
                        found[index] = frame.to_ndarray(format="rgb24")
                        times[index] = timestamp
                    if index >= int(ids.max()):
                        break
            return found, times

        found, times = decode(seek=True)
        if wanted - found.keys():
            # Some concatenated HEVC episodes have unsafe random-access points.
            # Reopen and decode from the beginning; never substitute other frames.
            print(
                json.dumps(
                    {
                        "phase": "decoder_sequential_fallback",
                        "video": self.path,
                        "missing_after_seek": sorted(wanted - found.keys()),
                    }
                ),
                flush=True,
            )
            found, times = decode(seek=False)
        missing = wanted - found.keys()
        if missing:
            raise RuntimeError(
                f"Missing requested video frames after sequential fallback in {self.path}: {sorted(missing)[:8]}"
            )
        self.ids = ids.copy()
        self.frames = np.stack([found[int(i)] for i in ids])
        self.times = np.asarray([times[int(i)] for i in ids])

    def get_frame_timestamp(self, ids):
        import numpy as np

        self._read(ids)
        return np.stack([self.times, self.times + 1.0 / self.fps], axis=1)

    def get_batch(self, ids):
        import numpy as np

        ids = np.asarray(ids, dtype=np.int64)
        positions = {int(v): i for i, v in enumerate(self.ids)}
        return self.frames[[positions[int(i)] for i in ids]]

def encode(args):
    import av
    import numpy as np
    import torch
    import torch.nn.functional as F


    contract = read_json(args.contract)
    from .wan_va.modules.utils import WanVAEStreamingWrapper
    from .wan_va.modules.utils import load_vae
    from .wan_va.modules.utils import resolve_model_component

    contract_hash = digest_json(contract["video"])
    manifest = read_json(args.manifest)
    if manifest["data_contract_sha256"] != contract_hash:
        raise RuntimeError("Manifest was built with a different data contract")
    model_root = Path(args.model_path).resolve()
    vae_path = Path(resolve_model_component(model_root, "vae"))
    model_identity = {
        "config": file_identity(vae_path / "config.json", True),
        "weights": [file_identity(p, True) for p in sorted(vae_path.glob("*.safetensors"))],
    }
    if not model_identity["weights"]:
        raise FileNotFoundError(f"No VAE safetensors under {vae_path}")
    encoder_hash = digest_json(model_identity)
    records = []
    for kind in ("human", "sim"):
        if args.kind in ("all", kind):
            records += manifest[f"{kind}_episodes"]
    if args.repo_id:
        records = [r for r in records if r["repo_id"] == args.repo_id]
    records = records[args.shard_index :: args.num_shards]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("No episodes selected")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    vae = load_vae(str(vae_path), torch_dtype=dtype, torch_device=args.device).eval()
    vae.requires_grad_(False)
    if int(vae.config.scale_factor_spatial) != contract["video"]["vae_spatial_stride"]:
        raise ValueError("VAE spatial stride differs from data contract")
    if int(vae.config.scale_factor_temporal) != contract["video"]["vae_temporal_stride"]:
        raise ValueError("VAE temporal stride differs from data contract")
    wrapper = WanVAEStreamingWrapper(vae)
    mean = torch.tensor(vae.config.latents_mean, device=args.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=args.device).view(1, -1, 1, 1, 1)
    readers = {}

    completed, reused = 0, 0

    def progress(complete=False):
        atomic_json(
            Path(args.manifest).parent / f"{args.kind}_rank{args.shard_index}" / "progress.json",
            {
                "complete": complete,
                "completed": completed,
                "reused": reused,
                "total": len(records),
                "rank": args.shard_index,
                "updated_unix_time": time.time(),
            },
        )

    for n, record in enumerate(records):
        destination = Path(record["cache_path"])
        if destination.exists() and not args.overwrite:
            old = torch.load(destination, map_location="cpu", weights_only=True)
            validate_cached_payload(old, record, contract_hash, encoder_hash)
            reused += 1
            progress()
            print(json.dumps({"phase": "skip", "cache": str(destination)}), flush=True)
            continue
        validate_sources(record)
        data = read_episode(record)
        source = record["source"]
        plan = sample_plan(len(data["timestamps"]), source["fps"], record["kind"], contract)
        sampled_ids = plan["sampled_frame_indices"]
        requested_times = source["video_from_timestamp"] + data["timestamps"][sampled_ids]
        video_path = source["video_path"]
        if video_path not in readers:
            readers.clear()
            readers[video_path] = VideoReader(video_path, num_threads=args.decode_threads)
        reader = readers[video_path]
        decode_fps = float(reader.get_avg_fps())
        if abs(decode_fps - source["video_fps"]) > 0.01:
            raise RuntimeError(f"Decoder fps mismatch in {video_path}: {decode_fps}")
        absolute_ids = np.rint(requested_times * decode_fps).astype(np.int64)
        if absolute_ids.min() < 0 or absolute_ids.max() >= len(reader):
            raise RuntimeError(f"Episode video index out of range: {record['repo_id']}")
        decoded_times = np.asarray(reader.get_frame_timestamp(absolute_ids.tolist()))[:, 0]
        if np.max(np.abs(decoded_times - requested_times)) > 0.55 / decode_fps:
            raise RuntimeError(f"Video/parquet timing misalignment: {record['repo_id']}")
        wrapper.clear_cache()
        encoded = []
        resize_hw = tuple(contract["video"][record["kind"]]["resize_hw"])
        # First causal frame is separate; subsequent chunks are multiples of temporal stride 4.
        ranges = [(0, 1)]
        ranges += [
            (start, min(start + args.encode_frames, len(absolute_ids)))
            for start in range(1, len(absolute_ids), args.encode_frames)
        ]
        with torch.inference_mode():
            for lo, hi in ranges:
                frames = reader.get_batch(absolute_ids[lo:hi])
                x = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float()
                x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False)
                x = (x / 255.0 * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)
                result = wrapper.encode_chunk(x.to(device=args.device, dtype=dtype))
                mu, _ = result.chunk(2, dim=1)
                latent = ((mu.float() - mean) / std).to(dtype)
                if not torch.isfinite(latent).all():
                    raise RuntimeError(f"Non-finite VAE output: {record['repo_id']}")
                encoded.append(latent[0].cpu())
        wrapper.clear_cache()
        latent = torch.cat(encoded, dim=1).contiguous()
        if latent.shape[1] != len(plan["latent_anchor_indices"]):
            raise RuntimeError(f"Unexpected VAE temporal shape {latent.shape}")
        item = {
            "schema_version": contract["schema_version"],
            "kind": record["kind"],
            "repo_id": record["repo_id"],
            "episode_index": record["episode_index"],
            "latent": latent,
            "latent_layout": "C,F,H,W",
            "frame_indices": torch.from_numpy(data["frame_indices"]),
            "timestamps": torch.from_numpy(data["timestamps"]),
            "sampled_frame_indices": torch.from_numpy(sampled_ids),
            "sampled_timestamps": torch.from_numpy(data["timestamps"][sampled_ids]),
            "sample_valid_mask": torch.from_numpy(plan["sample_valid_mask"]),
            "decoder_frame_indices": torch.from_numpy(absolute_ids),
            "decoder_timestamps": torch.from_numpy(decoded_times.copy()),
            "latent_anchor_indices": torch.from_numpy(plan["latent_anchor_indices"]),
            "latent_nominal_anchor_indices": torch.from_numpy(plan["latent_nominal_anchor_indices"]),
            "latent_timestamps": torch.from_numpy(data["timestamps"][plan["latent_anchor_indices"]]),
            "action_chunk_size": contract["video"]["sim"]["source_frame_stride"] * 4 if record["kind"] == "sim" else None,
            "source": source,
            "record_sha256": record["record_sha256"],
            "encoding_sha256": episode_encoding_identity(record),
            "data_contract_sha256": contract_hash,
            "encoder_fingerprint": encoder_hash,
            "encoder_identity": model_identity,
            "resize_hw": list(resize_hw),
            "text_condition": "empty",
            "frame_padding": "repeat_last_with_explicit_mask",
        }
        if record["kind"] == "sim":
            item["actions"] = torch.from_numpy(data["actions"])
            item["states"] = torch.from_numpy(data["states"])
            item["action_valid_mask"] = torch.ones(len(data["actions"]), dtype=torch.bool)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        torch.save(item, tmp)
        os.replace(tmp, destination)
        completed += 1
        progress()
        print(
            json.dumps(
                {
                    "phase": "encoded",
                    "index": n + 1,
                    "total": len(records),
                    "cache": str(destination),
                    "latent_shape": list(latent.shape),
                    "max_video_time_error_s": float(np.max(np.abs(decoded_times - requested_times))),
                }
            ),
            flush=True,
        )

    progress(complete=True)


def encoder_identity(model_path):
    root = Path(model_path) / "vae"
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"No VAE weights under {root}")
    return {"config": file_identity(root / "config.json", True), "weights": [file_identity(p, True) for p in weights]}


def build_manifest(config):
    from filelock import FileLock

    config.validate()
    manifest_path = config.manifest_path().resolve()
    outside_repository(str(manifest_path), "manifest")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(manifest_path) + ".lock"):
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("selection_identity") != config.selection_identity():
                raise ValueError("Manifest selection differs from requested protocol")
            if manifest.get("cache_request_identity") != config.cache_request_identity():
                raise ValueError("Manifest source roots or encoding settings differ from the requested configuration")
            if manifest["encoder_identity"] != encoder_identity(config.model_path):
                raise ValueError("VAE weights changed; use a new manifest/cache namespace")
            load_task_mapping(read_json(manifest["contract_path"]))
            verified = {}
            for kind in ("sim", "human"):
                for record in manifest[f"{kind}_episodes"]:
                    validate_sources(record, verified)
            return manifest_path
        humans, robots = config.selections()
        spec = manifest_path.parent / (manifest_path.stem + "_spec")
        spec.mkdir(exist_ok=True)
        human_file, sim_file = spec / "human.json", spec / "sim.json"
        atomic_json(human_file, humans)
        atomic_json(sim_file, robots)
        contract = {
            "schema_version": 4,
            "paths": {
                "imitator_repo": str(REPO_ROOT),
                "human_config": str(human_file),
                "sim_config": str(sim_file),
                "human_root": str(Path(config.human_root).resolve()),
                "sim_root": str(Path(config.sim_root).resolve()),
                "task_mapping": str(Path(config.task_mapping_file).expanduser().resolve()),
            },
            "task_mapping_identity": file_identity(config.task_mapping_file, True),
            "tasks": {
                "c15": sorted({base_task(r["repo_id"]) for r in robots}),
                "sim_training_episodes": sum(len(selected_episodes(r["train"])) for r in robots),
                "human_training_episodes": sum(len(selected_episodes(r["train"])) for r in humans),
            },
            "video": {
                "sim": {"source_frame_stride": config.robot_frame_stride, "resize_hw": [config.image_size, config.robot_width],
                    "action_tail_policy": "all_actions_v1"},
                "human": {"target_fps": config.human_fps, "resize_hw": [config.human_height, config.human_width]},
                "vae_spatial_stride": 16,
                "vae_temporal_stride": 4,
                "transformer_spatial_patch": 2,
            },
        }
        model = encoder_identity(config.model_path)
        recipe_key = digest_json(
            {
                "encoder": model,
                "video": contract["video"],
                "source_roots": {k: contract["paths"][k] for k in ("human_root", "sim_root")},
            }
        )
        cache_root = Path(config.cache_dir).resolve() / "episodes" / recipe_key[:24]
        records, identities = make_records(contract, cache_root)
        contract_path = spec / "contract.json"
        atomic_json(contract_path, contract)
        summaries = {}
        for kind, rows in records.items():
            lengths = [r["source"]["episode_length"] for r in rows]
            latent = [r["expected_latent_frames"] for r in rows]
            summaries[kind] = {
                "episodes": len(rows),
                "source_frames": sum(lengths),
                "source_frames_min_median_max": [min(lengths), statistics.median(lengths), max(lengths)],
                "latent_frames_min_median_max": [min(latent), statistics.median(latent), max(latent)],
            }
        manifest = {
            "schema_version": 4,
            "selection_identity": config.selection_identity(),
            "cache_request_identity": config.cache_request_identity(),
            "protocol": config.protocol,
            "contract_path": str(contract_path),
            "data_contract_sha256": digest_json(contract["video"]),
            "encoder_identity": model,
            "config_identities": identities,
            "sim_episodes": records["sim"],
            "human_episodes": records["human"],
            "records_sha256": digest_json(records),
            "summary": summaries,
        }
        atomic_json(manifest_path, manifest)
    return manifest_path


def require_complete(config, manifest_path=None):
    import numpy as np
    import torch

    path = Path(manifest_path or config.manifest_path())
    manifest = read_json(path)
    if manifest.get("selection_identity") != config.selection_identity():
        raise ValueError("Cache manifest does not match the configured task/episode selection")
    if manifest.get("cache_request_identity") != config.cache_request_identity():
        raise ValueError("Cache manifest does not match the requested source roots or encoding settings")
    contract = read_json(manifest["contract_path"])
    load_task_mapping(contract)
    if digest_json(contract["video"]) != manifest["data_contract_sha256"]:
        raise ValueError("Cache contract changed")
    if encoder_identity(config.model_path) != manifest["encoder_identity"]:
        raise ValueError("VAE model identity changed")
    verified = {}
    for kind in ["sim", "human"]:
        for record in manifest[f"{kind}_episodes"]:
            validate_sources(record, verified)
            payload = torch.load(record["cache_path"], map_location="cpu", weights_only=True, mmap=True)
            validate_cached_payload(
                payload, record, manifest["data_contract_sha256"], digest_json(manifest["encoder_identity"])
            )
            plan = sample_plan(record["source"]["episode_length"], record["source"]["fps"], kind, contract)
            latent = payload["latent"]
            height, width = contract["video"][kind]["resize_hw"]
            spatial_stride = contract["video"]["vae_spatial_stride"]
            if tuple(latent.shape) != (48, record["expected_latent_frames"], height // spatial_stride, width // spatial_stride):
                raise ValueError(f"Invalid VAE layout: {record['cache_path']}")
            if not np.array_equal(payload["sampled_frame_indices"].numpy(), plan["sampled_frame_indices"]):
                raise ValueError("VAE frame selection mismatch")
    return manifest


def prepare(config, *, kind="all", num_shards=1, shard_index=0, build_only=False, verify_only=False, encode_frames=16):
    if not 0 <= shard_index < num_shards or encode_frames < 4 or encode_frames % 4:
        raise ValueError("Invalid sharding or VAE chunk size")
    manifest = build_manifest(config)
    if verify_only:
        require_complete(config, manifest)
    elif not build_only:
        contract = read_json(manifest)["contract_path"]
        args = SimpleNamespace(
            contract=contract,
            manifest=str(manifest),
            model_path=config.model_path,
            kind=kind,
            repo_id=None,
            limit=None,
            num_shards=num_shards,
            shard_index=shard_index,
            dtype="bfloat16",
            device=config.device,
            encode_frames=encode_frames,
            decode_threads=2,
            overwrite=False,
        )
        encode(args)
    return manifest
