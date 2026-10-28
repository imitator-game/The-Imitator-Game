"""Shared IG rollout with native incremental history and flat comparison videos."""

from dataclasses import dataclass
from dataclasses import fields
import hashlib
import os
from pathlib import Path
import random

import numpy as np
import torch

from .config import CONFIG_ROOT
from .config import ZeroWAMConfig
from .config import outside_repository
from .runtime import atomic_json_save


def rgb_frame(value, *, normalized):
    x = torch.as_tensor(value).detach()
    if x.ndim == 5:
        x = x[:, -1]
    if x.ndim != 4 or x.shape[0] != 1 or x.shape[-1] != 3:
        raise ValueError(f"Diagnostic camera expects B=1 HWC RGB, got {x.shape}")
    x = x[0]
    if normalized:
        x = x.float().clamp(0, 1).mul(255).round()
    return x.to(device="cpu", dtype=torch.uint8).numpy().copy()


class PreviewDiagnostics:
    def __init__(self, output, config):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.actual = []
        self.history = {}
        self.predictions = []
        self.initial = None

    def begin(self, robot_obs):
        if self.actual:
            raise ValueError("Diagnostic output supports exactly one episode")
        self.actual.append(rgb_frame(robot_obs.get("view_1", robot_obs.get("rgb")), normalized=True))

    def actual_step(self, result):
        obs, _, _, truncated, info = result
        terminal = info.get("final_observation") if isinstance(info, dict) else None
        if np.asarray(truncated).any() and isinstance(terminal, dict) and ("rgb" in terminal):
            obs = terminal
        self.actual.append(rgb_frame(obs["rgb"], normalized=False))

    def history_latent(self, value, initial, chunk_idx):
        cpu = value.detach().cpu().clone()
        if initial:
            self.initial = cpu
        else:
            self.history[int(chunk_idx)] = cpu

    def prediction(self, value, chunk_idx):
        predicted_steps = self.config.frame_chunk_size * self.config.action_per_frame - (
            self.config.action_per_frame if chunk_idx == 0 else 0
        )
        self.predictions.append(
            {
                "chunk": int(chunk_idx),
                "start_step": len(self.actual) - 1,
                "action_steps": min(
                    predicted_steps, getattr(self.config, "execution_horizon", None) or predicted_steps
                ),
                "predicted_action_steps": predicted_steps,
                "latent": value.detach().cpu().clone(),
            }
        )

    @torch.no_grad()
    def finish(self, vae):
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

        if self.initial is None or not self.predictions:
            raise ValueError("No real policy predictions were recorded")
        if [v["chunk"] for v in self.predictions] != list(range(len(self.predictions))):
            raise ValueError("Prediction chunks contain gaps or duplicates")
        torch.save(
            {
                "initial": self.initial,
                "history": self.history,
                "predictions": self.predictions,
                "frame_stride": self.config.robot_frame_stride,
                "action_per_frame": self.config.action_per_frame,
                "frame_chunk_size": self.config.frame_chunk_size,
            },
            self.output / "prediction_latents.pt",
        )
        np.savez_compressed(self.output / "actual_rgb.npz", frames=np.stack(self.actual))
        device = torch.device(self.config.device)
        dtype = next(vae.parameters()).dtype
        mean = torch.tensor(vae.config.latents_mean, device=device)[None, :, None, None, None]
        std = torch.tensor(vae.config.latents_std, device=device)[None, :, None, None, None]
        history_cache = [None] * vae._cached_conv_counts["decoder"]

        def decode_frame(latent, cache, first=False):
            z = (latent.to(device=device).float() * std + mean).to(dtype)
            x = vae.post_quant_conv(z)
            y = vae.decoder(x, feat_cache=cache, feat_idx=[0], first_chunk=first)
            if vae.config.patch_size is not None:
                y = unpatchify(y, patch_size=vae.config.patch_size)
            return ((y.clamp(-1, 1).float() + 1) * 127.5).round().to(torch.uint8).cpu()[0].permute(1, 2, 3, 0).numpy()

        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            decode_frame(self.initial, history_cache, first=True)
            metadata = []
            for item in self.predictions:
                idx = item["chunk"]
                if idx:
                    observed = self.history[idx - 1]
                    for t in range(observed.shape[2]):
                        decode_frame(observed[:, :, t : t + 1], history_cache)
                cache = [v.clone() if isinstance(v, torch.Tensor) else v for v in history_cache]
                latent = item["latent"]
                start = 1 if idx == 0 else 0
                frames = np.concatenate(
                    [decode_frame(latent[:, :, t : t + 1], cache) for t in range(start, latent.shape[2])]
                )
                expected = item["predicted_action_steps"] // self.config.robot_frame_stride
                if len(frames) != expected:
                    raise ValueError(f"Decoded prediction length {len(frames)} != {expected}")
                np.savez_compressed(self.output / f"predicted_chunk_{idx:03d}.npz", frames=frames)
                metadata.append({k: v for k, v in item.items() if k != "latent"})
                del cache
        manifest = {
            "version": 2,
            "actual_frames": len(self.actual),
            "executed_steps": len(self.actual) - 1,
            "chunks": metadata,
            "fps": self.config.fps,
            "robot_frame_stride": self.config.robot_frame_stride,
            "panels": "Actual and predicted use the policy RGB camera; original renderer videos are preserved separately",
            "prediction_source": "Exact final normalized video latent used to condition this rollout's action prediction",
            "decoding": "Causal VAE decoder branches from executed observation history for each chunk",
            "temporal_alignment": "Predicted RGB frames correspond to actions at stride4; held for4 executed steps, no interpolated motion",
            "last_chunk": "Full prediction retained; comparison clips unexecuted future tail",
        }
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[DIAGNOSTICS] {self.output}: {len(metadata)} chunks, {len(self.actual) - 1} executed steps", flush=True)


"""Four-panel comparison of two independent rollouts and their own predictions."""
import json

import imageio_ffmpeg
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def writer(path, size, fps):
    out = imageio_ffmpeg.write_frames(
        str(path),
        size,
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=7,
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    out.send(None)
    return out


def load_lane(path):
    path = Path(path)
    meta = json.loads((path / "manifest.json").read_text())
    actual = np.load(path / "actual_rgb.npz")["frames"]
    predicted = []
    for chunk in meta["chunks"]:
        frames = np.load(path / f"predicted_chunk_{chunk['chunk']:03d}.npz")["frames"]
        predicted.append(np.repeat(frames, meta["robot_frame_stride"], axis=0))
    if len(actual) != meta["executed_steps"] + 1:
        raise ValueError("Actual camera frames do not cover every executed step")
    return meta, actual, predicted


def compose(left, right, output, pause_seconds=1.0, panel_size=320):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lanes = [load_lane(left), load_lane(right)]
    a, b = [x[0] for x in lanes]
    key = lambda m: [(x["chunk"], x["start_step"], x["action_steps"]) for x in m["chunks"]]
    if key(a) != key(b) or a["executed_steps"] != b["executed_steps"] or a["fps"] != b["fps"]:
        raise ValueError("Independent rollouts have different chunk boundaries; refusing false alignment")
    fps = a["fps"]
    pause_frames = round(pause_seconds * fps)
    font = ImageFont.load_default(size=18)
    width, height = panel_size * 4, panel_size + 80
    movie = writer(output, (width, height), fps)
    singles = []
    total = 0
    pauses = []

    def image(panels, chunk, start, end, paused=False):
        canvas = Image.new("RGB", (width, height), "#101620")
        draw = ImageDraw.Draw(canvas)
        labels = ["Rollout 1 | actual", "Rollout 1 | predicted", "Rollout 2 | actual", "Rollout 2 | predicted"]
        for i, (frame, label) in enumerate(zip(panels, labels)):
            canvas.paste(Image.fromarray(frame).resize((panel_size, panel_size)), (i * panel_size, 36))
            draw.text((i * panel_size + 8, 9), label, font=font, fill="white")
        caption = f"Chunk {chunk:02d} | executed steps {start + 1}-{end} | "
        caption += f"boundary pause {pause_seconds:g}s" if paused else "predicted RGB held for 4 action steps"
        draw.text((8, panel_size + 47), caption, font=font, fill="#f4cb68" if paused else "white")
        return np.asarray(canvas).copy()

    try:
        for j, chunk in enumerate(a["chunks"]):
            start = chunk["start_step"]
            end = min(start + chunk["action_steps"], a["executed_steps"])
            for step in range(start, end):
                panels = []
                for lane in lanes:
                    panels.extend([lane[1][step + 1], lane[2][j][step - start]])
                for stream, frame in zip(singles, panels):
                    stream.send(np.ascontiguousarray(frame))
                movie.send(image(panels, chunk["chunk"], start, end))
                total += 1
            if j + 1 < len(a["chunks"]):
                pause = image(panels, chunk["chunk"], start, end, paused=True)
                pauses.append(dict(after_chunk=chunk["chunk"], start_video_frame=total, frames=pause_frames))
                for _ in range(pause_frames):
                    movie.send(pause)
                    total += 1
    finally:
        movie.close()
        for stream in singles:
            stream.close()
    result = dict(
        status="completed",
        video=str(output),
        inputs=[str(left), str(right)],
        panels=["rollout1_actual", "rollout1_predicted", "rollout2_actual", "rollout2_predicted"],
        fps=fps,
        frames=total,
        duration_seconds=total / fps,
        pause_seconds=pause_seconds,
        pauses=pauses,
        executed_steps=a["executed_steps"],
        chunks=len(a["chunks"]),
        resolution=[width, height],
        alignment="Each prediction belongs to its own rollout; RGB stride4 displayed by holding each frame for4 action steps. No cross-rollout sharing.",
        camera="Policy RGB camera, matching video prediction coordinates; native renderer videos retained in each evaluation output.",
        pause_scope="Encoded playback only; simulator and training are never paused.",
    )
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result


class TaskScopedGT:
    """Load only current task's action columns via IG's unchanged GT provider."""

    def __init__(self, config_file, sim_root, output, normalizer):
        self.configs = json.loads(Path(config_file).read_text())
        self.sim_root = sim_root
        self.output = Path(output) / "dtw_reference_configs"
        self.output.mkdir(exist_ok=True)
        self.normalizer = normalizer
        self.provider = None
        self.current = None

    def select(self, env_id):
        from examples.baselines.lerobot_dataset.trajectory_metrics import GTTrajectoryProvider

        def matches(item):
            repo = item["repo_id"]
            keys = [repo, repo.split("_", 1)[1] if "_" in repo else repo]
            return any(k == env_id or k.endswith(env_id) or env_id.endswith(k) for k in keys)

        configs = [item for item in self.configs if item["repo_id"] == env_id]
        if not configs:
            configs = [item for item in self.configs if matches(item)]
        if not configs:
            raise ValueError(f"No official GT task configuration for {env_id}")
        path = self.output / (env_id + ".json")
        path.write_text(json.dumps(configs, indent=2))
        self.provider = GTTrajectoryProvider(
            str(path),
            self.sim_root,
            action_key="action.qpos_gripper_actions",
            normalizer=self.normalizer,
            normalization_method="bounds_q99",
        )
        if not self.provider._episodes:
            raise ValueError(f"No GT action trajectories loaded for {env_id}")
        self.current = env_id
        print(f"[METRICS] Native IG episode metrics + TSS/nDTW/wTSS; task-only GT={env_id}", flush=True)

    def sample_gt_trajectory(self, task_id, seed=None, normalize=False, dataset_idx=0):
        if task_id != self.current:
            raise ValueError("Wrong GT task requested")
        result = self.provider.sample_gt_trajectory(task_id, seed=seed, normalize=normalize, dataset_idx=dataset_idx)
        if result is None or len(result) < 2 or result.shape[1] != 16:
            raise ValueError(f"Invalid GT action trajectory for {task_id}")
        digest = hashlib.sha256(result.tobytes()).hexdigest()
        path = self.output / (digest + ".npy")
        if not path.exists():
            np.save(path, result)
        print(f"[METRICS] GT reference={path} steps={len(result)}", flush=True)
        return result


def compose_single(diagnostics, output, panel_size=320):
    """Export the unpaired rollout honestly as actual/predicted two-panel video."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata, actual, predicted = load_lane(diagnostics)
    movie = writer(output, (panel_size * 2, panel_size + 36), metadata["fps"])
    font = ImageFont.load_default(size=18)
    frames_written = 0
    try:
        for index, chunk in enumerate(metadata["chunks"]):
            start = chunk["start_step"]
            stop = min(start + chunk["action_steps"], metadata["executed_steps"])
            for step in range(start, stop):
                canvas = Image.new("RGB", (panel_size * 2, panel_size + 36), "#101620")
                draw = ImageDraw.Draw(canvas)
                panels = [actual[step + 1], predicted[index][step - start]]
                for column, (frame, label) in enumerate(zip(panels, ["Actual", "Predicted"], strict=True)):
                    canvas.paste(Image.fromarray(frame).resize((panel_size, panel_size)), (column * panel_size, 36))
                    draw.text((column * panel_size + 8, 9), label, font=font, fill="white")
                movie.send(np.ascontiguousarray(canvas))
                frames_written += 1
    finally:
        movie.close()
    result = {
        "status": "completed",
        "video": str(output),
        "inputs": [str(diagnostics)],
        "panels": ["actual", "predicted"],
        "frames": frames_written,
        "fps": metadata["fps"],
    }
    atomic_json_save(result, output.with_suffix(".json"))
    return result


@dataclass
class EvalArgs:
    checkpoint: str
    output_dir: str
    video_dir: str | None = None
    eval_config: str | None = None
    human_config: str | None = None
    sim_config: str | None = None
    human_root: str | None = None
    sim_root: str | None = None
    cache_dir: str | None = None
    model_path: str | None = None
    empty_text_embedding: str | None = None
    task_mapping_file: str | None = None
    human_task_description_file: str | None = None
    sim_task_description_file: str | None = None
    num_episodes: int = 10
    episode_start: int = 0  # Global episode index for disjoint parallel rollout shards.
    max_episode_steps: int = 500
    seed: int = 100000
    capture_video: bool = False
    trajectory_metrics: bool = True
    icl_guidance_scale: float = 1.0
    dispatch_url: str | None = None
    case_root: str | None = None


class RecordedEnvironment:
    def __init__(self, env, seed, recorder):
        self.env, self.seed, self.recorder = env, seed, recorder

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(seed=self.seed, **kwargs)

    def step(self, action):
        result = self.env.step(action)
        if self.recorder is not None:
            self.recorder.actual_step(result)
        return result


def resolve_code_resources(config, saved, args, output):
    """Relocate repository defaults; preserve custom content without old checkouts."""
    from .config import CODE_RESOURCES

    report = {}
    for name, relative in CODE_RESOURCES.items():
        default = (CONFIG_ROOT / relative).resolve()
        descriptor = saved.get(name, {})
        explicit = getattr(args, name, None)
        stored = Path(getattr(config, name)).expanduser()
        if explicit is not None:
            chosen = Path(explicit).expanduser().resolve()
        elif descriptor.get("kind") == "custom":
            raw = descriptor["text"].encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
                raise ValueError(f"Checkpoint custom resource identity is invalid: {name}")
            chosen = output / "resolved_resources" / f"{name}_{descriptor['sha256'][:16]}.json"
            chosen.parent.mkdir(parents=True, exist_ok=True)
            chosen.write_bytes(raw)
        elif descriptor.get("kind") == "repository" or stored.as_posix().endswith(
            "/examples/baselines/lerobot_dataset/" + relative
        ):
            chosen = default
        else:
            # Legacy checkpoints cannot recover custom content they never saved.
            # Preserve an existing custom path, otherwise require an explicit override.
            chosen = stored.resolve()
        if not chosen.is_file():
            raise FileNotFoundError(f"Missing {name}: {chosen}; supply --{name.replace('_', '-')}")
        fingerprint = hashlib.sha256(chosen.read_bytes()).hexdigest()
        if explicit is None and descriptor.get("sha256") and fingerprint != descriptor["sha256"]:
            raise ValueError(
                f"Resource identity changed: {name}; supply an explicit --{name.replace('_', '-')} override"
            )
        setattr(config, name, str(chosen))
        report[name] = {"path": str(chosen), "sha256": fingerprint, "explicit_override": explicit is not None}
    atomic_json_save(report, output / "resolved_resources.json")


def run(args):
    from .runtime import setup_runtime
    from .runtime import start_log

    output = outside_repository(args.output_dir, "output_dir")
    output.mkdir(parents=True, exist_ok=True)
    videos = None
    if args.capture_video:
        videos = outside_repository(args.video_dir or str(output / "videos"), "video_dir")
        videos.mkdir(parents=True, exist_ok=True)
    start_log(output, "evaluation")
    if args.num_episodes <= 0 or args.max_episode_steps <= 0 or args.episode_start < 0:
        raise ValueError("Episode count and length must be positive, and start must be nonnegative")
    if args.icl_guidance_scale < 1:
        raise ValueError("ICL guidance scale must be at least1")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_stat = checkpoint_path.stat()
    protocol = {
        "max_episode_steps": args.max_episode_steps,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
    }
    if args.icl_guidance_scale != 1:
        protocol["icl_guidance_scale"] = args.icl_guidance_scale
    if args.episode_start:
        protocol["episode_start"] = args.episode_start
    protocol_file = output / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError(f"Evaluation protocol changed in existing output: {protocol_file}")
    atomic_json_save(protocol, protocol_file)
    print(f"[PROTOCOL] max_control_steps={args.max_episode_steps} episodes={args.num_episodes} seed={args.seed}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not checkpoint.get("native_trainer") or checkpoint.get("format_version") != 3:
        raise ValueError("Expected a full-state checkpoint from this baseline")
    fields_allowed = {f.name for f in fields(ZeroWAMConfig)}
    stored = checkpoint["native_config"].get("ig_app_config")
    if stored is None:
        raise ValueError("Checkpoint is missing portable baseline configuration")
    config = ZeroWAMConfig(**{k: v for k, v in stored.items() if k in fields_allowed})
    if "ig_normalization" in checkpoint["native_config"]:
        stats_path = output / "normalization.json"
        atomic_json_save(checkpoint["native_config"]["ig_normalization"], stats_path)
        config.normalization_path = str(stats_path)
    # Restore the architecture recorded by the final episode checkpoints.
    config.adapter_rank = checkpoint["adaptation"].get("rank", checkpoint["native_config"].get("ig_lora_rank"))
    config.sampling_mode = checkpoint["native_config"].get("ig_sampling_mode", "episode")
    if "enable_mcp" not in stored:
        config.enable_mcp = checkpoint["native_config"].get("enable_mcp", False)
    for name in ["human_root", "sim_root", "cache_dir", "model_path", "empty_text_embedding"]:
        if getattr(args, name) is not None:
            setattr(config, name, getattr(args, name))
    resolve_code_resources(config, checkpoint["native_config"].get("ig_code_resources", {}), args, output)
    setup_runtime(config.cache_dir)
    # Evaluation datasets may differ from training. Build their own selection
    # manifest while sharing only content-compatible complete-clip VAE caches.
    split = "unseen" if config.protocol == "scratch" else "seen_unseen"
    eval_config = Path(
        args.eval_config
        or CONFIG_ROOT
        / "eval/exp_list"
        / ("unseen_5tasks_env_list.txt" if split == "unseen" else "seen_plus_unseen_10tasks_env_list.txt")
    )
    human_config = args.human_config or str(
        CONFIG_ROOT
        / "config/exp_configs"
        / ("human_test_config_unseen.json" if split == "unseen" else "human_eval_config_seen_plus_unseen_10tasks.json")
    )
    sim_config = args.sim_config or str(
        CONFIG_ROOT
        / "config/exp_configs"
        / ("sim_test_config_unseen.json" if split == "unseen" else "sim_eval_config_seen_plus_unseen_10tasks.json")
    )
    config.human_dataset_file = human_config
    config.sim_dataset_file = sim_config
    config.task = None
    config.episode_count = None
    config.manifest = None
    from .cache import prepare
    from .cache import read_json

    # Each evaluation worker owns its manifest/spec and real FileLock locally.
    # Episode payloads remain in cache_dir with unchanged compatibility checks.
    runtime_root = os.environ.get("IG_RUNTIME_CACHE_ROOT")
    if runtime_root:
        config.manifest = str(
            Path(runtime_root) / "vae_manifests" / f"{config.protocol}_{config.cache_request_identity()[:16]}.json"
        )
    manifest_path = prepare(config, kind="human")
    manifest = read_json(manifest_path)
    humans = {}
    for record in manifest["human_episodes"]:
        humans.setdefault(record["repo_id"], []).append(record)
    mappings = json.loads(Path(config.task_mapping_file).read_text())["task_mappings"]
    pairing = {task: row["human_task_id"] for row in mappings for task in row["sim_task_id"]}
    from .policy import ZeroWAMAgent

    agent = ZeroWAMAgent(config, icl_guidance_scale=args.icl_guidance_scale)
    print(f"[ICL_CFG] scale={args.icl_guidance_scale}; native video sequence0/1 guidance", flush=True)
    saved_shapes = checkpoint["adaptation"]["trainable_shapes"]
    backbone_shapes = {n: shape for n, shape in saved_shapes.items() if not n.startswith(("mcp_blocks.", "mcp_mlp_hidden.", "mcp_projections."))}
    if agent.adaptation_report["trainable_shapes"] != backbone_shapes:
        raise ValueError("Training/evaluation backbone adapter architecture mismatch")
    if set(checkpoint["adapter_state_dict"]) != set(saved_shapes):
        raise ValueError("Checkpoint adapter coverage mismatch")
    agent.load_adapter({n: checkpoint["adapter_state_dict"][n] for n in backbone_shapes})
    agent.eval()
    agent.requires_grad_(False)
    from examples.baselines.diffusion_policy.eval_dp_imitator import clear_l_level
    from .rollout import evaluate_with_task_encoder
    from examples.baselines.diffusion_policy.eval_dp_imitator import make_eval_envs_with_level
    from examples.baselines.lerobot_dataset.evaluate_processor import HumanVideoSimEvaluateProcessor
    from examples.baselines.lerobot_dataset.evaluate_processor import HumanVideoSimEvaluateProcessorConfig
    from examples.baselines.lerobot_dataset.lerobot_paired_dataset import InputMode
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    processor = HumanVideoSimEvaluateProcessor(
        HumanVideoSimEvaluateProcessorConfig(
            human_root=config.human_root,
            human_split="train",
            human_dataset_file=human_config,
            human_task_description_file=config.human_task_description_file,
            human_cameras=["zed2i"],
            human_include_depth=False,
            human_num_frames=33,
            human_image_size=(224, 224),
            human_max_jitter=0,
            human_video_backend="torchcodec",
            human_fps=30,
            sim_root=config.sim_root,
            sim_split="train",
            sim_dataset_file=sim_config,
            sim_task_description_file=config.sim_task_description_file,
            sim_state_type="qpos",
            sim_single_arm=False,
            normalization_method="bounds_q99",
            task_mapping_file=config.task_mapping_file,
        )
    )
    # The actual condition is the matched complete-clip latent below. This
    # placeholder only bypasses the shared evaluator's redundant RGB decoding.
    processor.get_video = lambda env_id, count: torch.zeros(count, 1, 3, 224, 224)
    if agent.action_space:
        from .action_space import EpisodeEvaluationNormalizer
        processor.normalizer = EpisodeEvaluationNormalizer(processor.normalizer, agent)
    gt = TaskScopedGT(sim_config, config.sim_root, output, processor.normalizer) if args.trajectory_metrics else None
    environments = [
        s.strip() for s in eval_config.read_text().splitlines() if s.strip() and not s.lstrip().startswith("#")
    ]
    if not environments or len(environments) != len(set(environments)):
        raise ValueError("Environment list must be nonempty and contain no duplicates")
    rows = []
    from .dispatch import evaluation_cases
    for env_id, selected_episode in evaluation_cases(environments, args):
        if env_id not in pairing or pairing[env_id] not in humans:
            raise ValueError(f"Missing evaluation human pairing: {env_id}")
        level, base = env_id.split("_", 1)
        if level not in ["L0", "L1", "L2", "L3"]:
            raise ValueError(f"Unsupported level: {env_id}")
        if gt is not None:
            gt.select(env_id)
        diagnostics = []
        episodes = ([selected_episode] if selected_episode is not None else
                    range(args.episode_start, args.episode_start + args.num_episodes))
        for episode in episodes:
            seed = args.seed + episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            human = humans[pairing[env_id]][episode % len(humans[pairing[env_id]])]
            agent.preview_human_latent = torch.load(
                human["cache_path"], map_location="cpu", weights_only=True, mmap=True
            )["latent"][None]
            lane = Path(args.case_root or output) / env_id / f"episode_{episode:03d}"
            lane.mkdir(parents=True, exist_ok=True)
            # A completed episode is immutable across a worker-count change.
            existing_metrics = lane / "metrics.json"
            if existing_metrics.exists():
                saved = json.loads(existing_metrics.read_text())
                if saved["env_id"] != env_id or saved["seed"] != seed or saved["human_cache"] != human["cache_path"]:
                    raise ValueError(f"Existing episode identity mismatch: {existing_metrics}")
                if saved.get("max_episode_steps") != args.max_episode_steps:
                    raise ValueError(f"Existing episode control-step limit mismatch: {existing_metrics}")
                rows.append(saved)
                atomic_json_save({"checkpoint": args.checkpoint, "results": rows}, output / "metrics.json")
                print(f"[RESUME] Reused completed episode {env_id} seed={seed}", flush=True)
                continue
            recorder = PreviewDiagnostics(lane / "diagnostics", config) if args.capture_video else None
            if recorder is not None:
                agent.preview_diagnostics = recorder
            env = make_eval_envs_with_level(
                base,
                level,
                1,
                "physx_cpu",
                {
                    "control_mode": "pd_joint_pos",
                    "reward_mode": "dense",
                    "obs_mode": "rgb",
                    "render_mode": "rgb_array",
                    "max_episode_steps": args.max_episode_steps,
                    "sensor_configs": {"shader_pack": "rt-fast"},
                    "human_render_camera_configs": {"shader_pack": "rt-fast"},
                },
                {"obs_horizon": 1},
                None,
                [FlattenRGBDObservationWrapper],
            )
            env = RecordedEnvironment(env, seed, recorder)
            agent.history_action_denormalizer = lambda value, task=env_id: processor.denormalize_action(value, task)
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    metrics = evaluate_with_task_encoder(
                        1,
                        agent,
                        env,
                        {
                            "env_id": env_id,
                            "delta_control": False,
                            "pred_horizon": 32,
                            "temporal_agg": False,
                            "light_temporal_agg": False,
                            "dynamic_action_chunks": True,
                            "max_timesteps": args.max_episode_steps,
                            "device": "cuda",
                            "sim_backend": "physx_cpu",
                        },
                        processor,
                        InputMode.VIDEO_ONLY,
                        dtw_provider=gt,
                    )
                if recorder is not None:
                    recorder.finish(agent.vae)
                    diagnostics.append(lane / "diagnostics")
                row = {
                    "env_id": env_id,
                    "seed": seed,
                    "human_cache": human["cache_path"],
                    "max_episode_steps": args.max_episode_steps,
                    "metrics": {k: np.asarray(v).reshape(-1).tolist() for k, v in metrics.items()},
                    "gripper_history": {
                        "limited_commands_per_gripper": agent.gripper_history_corrections.cpu().tolist(),
                        "max_raw_correction_per_gripper": agent.gripper_history_max_correction.cpu().tolist(),
                        "executed_control_steps": agent.total_executed_steps,
                    },
                }
                rows.append(row)
                atomic_json_save(row, lane / "metrics.json")
                atomic_json_save({"checkpoint": args.checkpoint, "results": rows}, output / "metrics.json")
                if recorder is not None and len(diagnostics) % 2 == 0:
                    compose(
                        diagnostics[-2],
                        diagnostics[-1],
                        videos / f"{env_id}_pair_{episode // 2:03d}_comparison_4panel.mp4",
                    )
            finally:
                env.close()
                clear_l_level()
                agent.clear_cache()
        if args.capture_video and len(diagnostics) % 2:
            compose_single(
                diagnostics[-1],
                videos / f"{env_id}_episode_{args.episode_start + args.num_episodes - 1:03d}_comparison_2panel.mp4",
            )
    summary = {}
    for env_id in environments:
        selected = [r for r in rows if r["env_id"] == env_id]
        if not selected:
            continue
        summary[env_id] = {
            k: float(np.mean([v for r in selected for v in r["metrics"].get(k, [])]))
            for k in ["success_once", "success_at_end", "return"]
        }
    atomic_json_save(
        {"checkpoint": args.checkpoint, "max_episode_steps": args.max_episode_steps, "episodes_per_environment": args.num_episodes, "summary": summary},
        output / "summary.json",
    )
