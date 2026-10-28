"""Data, LoRA, batching, logging and resumable checkpoints for native Trainer.

The released Trainer._prepare_input_dict, compute_loss and _train_step remain
the source of noise, chunk/window sampling, objectives and optimizer behavior.
"""

from dataclasses import asdict
from datetime import datetime
from datetime import timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.checkpoint.state_dict import set_model_state_dict
from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict
from torch.utils.data import DataLoader

from .dataset import EpisodeBatchSampler
from .dataset import IGDataset
from .dataset import collate
from .dataset import ACTION_HISTORY_MODE
from .lora import adapt_transformer as inject_adapters


def cosine_lr_multiplier(step, config):
    """Decay on absolute optimizer updates so segmented resumes retain one horizon."""
    warmup_steps = config.get("warmup_steps", 0)
    if step < warmup_steps and config.ig_lr_decay_start_step >= warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    elapsed = min(max(step - config.ig_lr_decay_start_step, 0), config.ig_lr_decay_steps)
    minimum = config.ig_min_learning_rate / config.learning_rate
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * elapsed / config.ig_lr_decay_steps))


def build_lr_scheduler(optimizer, config):
    if config.get("ig_lr_schedule", "constant") == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: cosine_lr_multiplier(step, config))
    from .wan_va.utils import warmup_constant_lambda

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))




def atomic_json(path, value):
    from .runtime import atomic_json_save

    atomic_json_save(value, path)


def atomic_torch(path, value):
    from .runtime import atomic_torch_save

    atomic_torch_save(value, path)


def setup_data(trainer):
    config = trainer.config
    run = Path(config.save_root)
    run.mkdir(parents=True, exist_ok=True)
    dataset = IGDataset(
        config.ig_manifest,
        empty_text_path=config.empty_text_emb_path,
        seed=config.seed,
        sampling_mode=config.ig_sampling_mode,
        action_history_mode=config.get("ig_action_history_mode", ACTION_HISTORY_MODE),
        action_per_frame=config.get("ig_action_per_frame", 16),
        normalization_path=config.ig_app_config.get("normalization_path"),
        action_representation=config.ig_app_config.get("action_representation", "relative_episode"),
        action_channel_offset=config.ig_app_config.get("action_channel_offset", 14),
    )
    sampler = EpisodeBatchSampler(len(dataset), config.world_size, config.rank, seed=config.seed)
    trainer.train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=config.load_worker,
        persistent_workers=config.load_worker > 0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config.seed + 1233),
        **({"prefetch_factor": 2} if config.load_worker else {}),
    )
    trainer.ig_sampler = sampler
    trainer.ig_progress = {"epoch": 0, "next_batch": 0}
    trainer.ig_adaptation = trainer.transformer.ig_adaptation_report
    trainer.ig_trainable_names = list(trainer.ig_adaptation["trainable_shapes"])
    trainer.ig_checks = None
    trainer.ig_last_saved_epoch = 0
    trainer.ig_last_update_time = time.monotonic()
    trainer.ig_last_save_time = time.monotonic()
    trainer.ig_stopping = False
    import signal

    def stop_at_update(signum, frame):
        trainer.ig_stopping = True
        print(f"Signal {signum}: saving at the next complete optimizer boundary", flush=True)

    signal.signal(signal.SIGTERM, stop_at_update)
    signal.signal(signal.SIGINT, stop_at_update)
    random.seed(config.seed + config.rank)
    np.random.seed(config.seed + config.rank)
    torch.manual_seed(config.seed + config.rank)
    torch.cuda.manual_seed(config.seed + config.rank)
    if config.rank == 0:
        atomic_json(run / "config.json", dict(config))
        atomic_json(run / "trainable_parameters.json", trainer.ig_adaptation)
        atomic_json(
            run / "sampling_contract.json",
            dict(
                epoch="one shuffled presentation per episode plus native DistributedSampler padding",
                samples_per_epoch=len(dataset), presentations_per_epoch=sampler.total_size,
                updates_per_epoch=len(sampler), robot_episodes=len(dataset.records), human_episodes=len(dataset.humans),
                sampling_mode="episode", action_history_mode=dataset.action_history_mode,
                action_per_frame=config.ig_action_per_frame, robot_frame_stride=config.ig_action_per_frame // 4,
                start="fixed episode frame0; first RGB separate, then causal groups of4",
                window="entire robot episode; native random chunk1..4 / attention window4..64",
                human_pairing="uniform same-task random complete episode on each read; cadence-independent reproducible draws",
                targets="official video/action/MCP losses; initial normalized-zero action group supervised; tail actions masked",
                mcp_source="within current complete episode only; native shift/mask",
                state="observed state at chunk start, before supervised actions",
                action_representation=config.ig_app_config.get("action_representation", "relative_episode"),
                action_channel_offset=config.ig_app_config.get("action_channel_offset", 14),
                normalization="unified train-only" if dataset.action_space else "per-repository",
                padding="native sampler may repeat episodes; no temporal padding between different samples",
            ),
        )


def rng_state(device):
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device),
    )


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def resume_sample_progress(checkpoint, settings, samples_per_epoch):
    """Translate optimizer-boundary cursors without replaying committed samples."""
    old = checkpoint["native_config"]
    saved = checkpoint["progress"]
    epoch = saved["epoch"]
    if old.get("ig_sampling_mode") != "episode":
        raise ValueError("Episode training cannot resume a legacy window checkpoint; start fresh")
    world = settings["world_size"]
    batches = math.ceil(samples_per_epoch / world)
    next_batch = saved["next_batch"]
    if not 0 <= next_batch <= batches:
        raise ValueError("Episode checkpoint cursor is outside the native padded epoch")
    if next_batch == batches:
        epoch, next_batch = epoch + 1, 0
    remaining = max(0, settings["ig_total_epochs"] - epoch) * batches - next_batch
    return dict(epoch=epoch, next_batch=next_batch, sample_offset=next_batch*world), checkpoint["global_step"] + max(0, remaining)


def resume(trainer):
    path = trainer.config.get("ig_resume")
    if not path:
        return
    checkpoint = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
    current = dict(trainer.config)
    old = checkpoint["native_config"]
    if old.get("ig_sampling_mode") != "episode":
        raise ValueError("Resume requires a complete-episode checkpoint")
    mutable = {
        "ig_total_epochs",
        "ig_resume",
        "ig_max_steps",
        "num_steps",
        "rank",
        "local_rank",
        "save_root",
        "save_seconds",
        "save_step_freq",
        "save_epoch_freq",
        "ig_app_config",
        "ig_initialization",
    }
    if old["world_size"] != current["world_size"]:
        raise ValueError("Resume requires the original distributed topology")

    for key in old:
        if key not in mutable and old[key] != current.get(key):
            raise ValueError(f"Resume configuration changed: {key}")
    options = StateDictOptions(full_state_dict=True, strict=False)
    set_model_state_dict(trainer.transformer, checkpoint["adapter_state_dict"], options=options)
    set_optimizer_state_dict(
        trainer.transformer, trainer.optimizer, checkpoint["optimizer_state_dict"], options=options
    )
    trainer.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    assert equal(trainer.lr_scheduler.state_dict(), checkpoint["scheduler_state_dict"])
    trainer.step = checkpoint["global_step"]
    trainer.ig_progress, target_steps = resume_sample_progress(checkpoint, current, len(trainer.train_loader.dataset))
    trainer.config.num_steps = target_steps
    trainer.ig_sampler.set_epoch(trainer.ig_progress["epoch"])
    trainer.ig_sampler.start_batch = trainer.ig_progress["next_batch"]
    trainer.ig_last_saved_epoch = checkpoint.get("last_saved_epoch", 0)
    # Initialization is the lineage of this run, not another action on resume.
    # Restore lineage as metadata only; never initialize the model again on resume.
    trainer.config.ig_initialization = old.get("ig_initialization")
    if "initialized_from" in checkpoint.get("adaptation", {}):
        trainer.ig_adaptation["initialized_from"] = checkpoint["adaptation"]["initialized_from"]
    if trainer.config.rank == 0:
        reconcile_run_outputs(Path(trainer.config.save_root), trainer.step)
    if trainer.config.world_size > 1:
        dist.barrier()
    # Bookkeeping/report rendering must not consume the restored training RNG.
    restore_rng(checkpoint["rank_rng_states"][trainer.config.rank], trainer.device)
    # User requested immediate training without another restore verification run.
    trainer.ig_checks = None
    atomic_json(
        Path(trainer.config.save_root) / f"resume_loaded_rank{trainer.config.rank}.json",
        dict(
            checkpoint_step=trainer.step,
            adapter_loaded=True,
            optimizer_loaded=True,
            scheduler_loaded=True,
            rng_loaded=True,
            verification_run=False,
            previous_progress=checkpoint["progress"],
            current_progress=trainer.ig_progress,
        ),
    )


def reconcile_run_outputs(root, restored_step):
    """Archive abandoned future outputs before replaying updates from a checkpoint.

    Only rank zero calls this while the run is exclusively owned. A malformed
    unterminated final line is an interrupted append; corruption elsewhere is
    rejected before any file is changed. Epoch/final/latest links newer than
    the restored update leave the canonical namespace and remain in recovery.
    """
    root = Path(root)
    loss = root / "loss.jsonl"
    original = loss.read_bytes() if loss.exists() else None
    kept, rows = [], []
    discarded, incomplete_tail = 0, False
    lines = original.splitlines(keepends=True) if original is not None else []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as error:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                incomplete_tail = True
                continue
            raise ValueError(f"Invalid loss.jsonl line {index + 1}; no outputs were changed") from error
        if not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] < 0:
            raise ValueError(f"Invalid loss.jsonl step at line {index + 1}; no outputs were changed")
        if row["step"] > restored_step:
            discarded += 1
            continue
        if rows and row["step"] <= rows[-1]["step"]:
            raise ValueError(f"Nonmonotonic committed loss.jsonl line {index + 1}; no outputs were changed")
        rows.append(row)
        kept.append(line.rstrip(b"\r\n") + b"\n")
    canonical = b"".join(kept)
    loss_changed = original is not None and canonical != original
    future_checkpoints = []
    folder = root / "checkpoints"
    candidates = [folder / "latest.pt", folder / "final.pt", *sorted(folder.glob("epoch_*.pt"))]
    for path in candidates:
        if path.is_file():
            metadata = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
            if metadata["global_step"] > restored_step:
                future_checkpoints.append(path)
            del metadata
    metrics_path = root / "latest_metrics.json"
    stale_metrics = metrics_path.exists() and json.loads(metrics_path.read_text())["step"] > restored_step
    status_path = root / "checkpoint_status.json"
    stale_status = status_path.exists() and json.loads(status_path.read_text())["step"] > restored_step
    if not (loss_changed or future_checkpoints or stale_metrics or stale_status):
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    archive = root / "recovery" / f"resume_{stamp}_step{restored_step}"
    archive.mkdir(parents=True)
    # Preserve exact bytes (including any interrupted append) before rewriting.
    reports = ["loss.jsonl", "latest_metrics.json", "loss.csv", "loss.png", "checkpoint_status.json"]
    for name in reports:
        path = root / name
        if path.exists():
            shutil.copy2(path, archive / name)
    atomic_json(
        archive / "reconciliation.json",
        dict(
            restored_step=restored_step,
            discarded_future_rows=discarded,
            incomplete_final_line=incomplete_tail,
            retained_rows=len(rows),
            archived_checkpoints=[p.name for p in future_checkpoints],
        ),
    )
    for path in future_checkpoints:
        target = archive / "checkpoints" / path.name
        target.parent.mkdir(exist_ok=True)
        os.replace(path, target)
    if stale_status:
        status_path.unlink()
    if loss_changed:
        temporary = loss.with_name(f".loss.jsonl.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(canonical)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, loss)
    if rows:
        atomic_json(metrics_path, rows[-1])
        # Replace stale derived reports only when they were already requested.
        if (root / "loss.csv").exists() or (root / "loss.png").exists():
            export_loss(root)
    else:
        for name in ["latest_metrics.json", "loss.csv", "loss.png"]:
            (root / name).unlink(missing_ok=True)


def next_batch(trainer):
    if trainer.train_loader_iter is None:
        trainer.train_loader_iter = iter(trainer.train_loader)
    try:
        return next(trainer.train_loader_iter)
    except StopIteration:
        trainer.ig_sampler.set_epoch(trainer.ig_sampler.epoch + 1)
        trainer.train_loader_iter = iter(trainer.train_loader)
        return next(trainer.train_loader_iter)






def prepare_episode(trainer, batch, original):
    inputs = original(trainer, batch)
    chunk = inputs["chunk_size"]
    trainer.transformer.action_embedder.current_state = batch["ig_states"][:, ::chunk]
    trainer.transformer.action_embedder.tokens_per_chunk = chunk * trainer.config.ig_action_per_frame
    epoch, batch_index = int(batch["ig_epoch"][0]), int(batch["ig_batch_index"][0])
    trainer.ig_progress = dict(epoch=epoch, next_batch=batch_index + 1,
        sample_offset=trainer.ig_sampler.sample_offset_after(batch_index))
    if batch_index + 1 == len(trainer.ig_sampler):
        trainer.ig_progress = dict(epoch=epoch+1, next_batch=0, sample_offset=0)
    trainer.ig_group_count = trainer.config.world_size
    trainer.ig_latest_shape = dict(video_frames=batch["ig_video_length"].tolist(),
        human_frames=batch["ig_human_length"].tolist(), chunk_size=chunk,
        window=inputs["window_size"], human_indices=batch["ig_human_index"].tolist(),
        mcp_valid_frames=[stream["valid_mask"].flatten(1).sum(1).cpu().tolist()
            for stream in inputs.get("mcp_latent_dicts", [])])
    return inputs




def replace_checkpoint_link(source, target):
    """Atomically refresh a named pointer without mutating older hardlinks."""
    target = Path(target)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        os.link(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(trainer):
    state = get_model_state_dict(trainer.transformer, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    optimizer = get_optimizer_state_dict(
        trainer.transformer, trainer.optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    local_rng = rng_state(trainer.device)
    states = [None] * trainer.config.world_size
    dist.all_gather_object(states, local_rng)
    run = Path(trainer.config.save_root)
    if trainer.config.rank == 0:
        adapter = {name: state[name].float() for name in trainer.ig_trainable_names}
        checkpoint = dict(
            format_version=3,
            native_trainer=True,
            native_config=dict(trainer.config),
            adaptation=trainer.ig_adaptation,
            adapter_state_dict=adapter,
            optimizer_state_dict=optimizer,
            scheduler_state_dict=trainer.lr_scheduler.state_dict(),
            rank_rng_states=states,
            global_step=trainer.step,
            progress=trainer.ig_progress,
            last_saved_epoch=trainer.ig_last_saved_epoch,
        )
        atomic_torch(run / "checkpoints/latest.pt", checkpoint)
        epoch = trainer.ig_progress["epoch"]
        if epoch and epoch % trainer.config.save_epoch_freq == 0 and trainer.ig_progress["next_batch"] == 0:
            target = run / "checkpoints" / f"epoch_{epoch:03d}.pt"
            if not target.exists():
                os.link(run / "checkpoints/latest.pt", target)
        if trainer.step >= trainer.config.num_steps and not trainer.config.get("ig_max_steps"):
            target = run / "checkpoints/final.pt"
            replace_checkpoint_link(run / "checkpoints/latest.pt", target)
        atomic_json(
            run / "checkpoint_status.json",
            dict(step=trainer.step, progress=trainer.ig_progress, updated_at=datetime.now(timezone.utc).isoformat()),
        )
    del state, optimizer
    dist.barrier()


def checkpoint_due(trainer):
    epoch = trainer.ig_progress["epoch"]
    epoch_due = (
        epoch >= trainer.ig_last_saved_epoch + trainer.config.save_epoch_freq and trainer.ig_progress["next_batch"] == 0
    )
    if epoch_due:
        trainer.ig_last_saved_epoch = epoch
    decision = 0
    if trainer.config.rank == 0:
        decision = int(
            trainer.ig_stopping
            or trainer.step == 1
            or epoch_due
            or trainer.step % trainer.config.save_step_freq == 0
            or time.monotonic() - trainer.ig_last_save_time >= trainer.config.save_seconds
            or trainer.step >= trainer.config.num_steps
        )
    flag = torch.tensor(decision, device=trainer.device, dtype=torch.int32)
    dist.broadcast(flag, src=0)
    if flag.item():
        trainer.ig_last_save_time = time.monotonic()
    return bool(flag.item())


def log_update(trainer, values):
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated(trainer.device) / 2**30
    reserved = torch.cuda.max_memory_reserved(trainer.device) / 2**30
    memory = torch.tensor([allocated, reserved], device=trainer.device)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    now = time.monotonic()
    elapsed = now - trainer.ig_last_update_time
    trainer.ig_last_update_time = now
    duration = torch.tensor(elapsed, device=trainer.device)
    dist.all_reduce(duration, op=dist.ReduceOp.MAX)
    elapsed = float(duration)
    if not all(math.isfinite(float(values[k])) for k in ["video", "action", "mcp"]):
        trainer.ig_stopping = True
    memory_stats = torch.cuda.memory_stats(trainer.device)
    stop = torch.tensor(int(trainer.ig_stopping), device=trainer.device, dtype=torch.int32)
    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
    trainer.ig_stopping = bool(stop.item())
    if trainer.config.rank == 0:
        row = dict(
            event="optimizer_update",
            step=trainer.step,
            update_seconds=elapsed,
            epoch=trainer.ig_progress["epoch"],
            next_batch=trainer.ig_progress["next_batch"],
            sample_offset=trainer.ig_progress.get("sample_offset"),
            video_loss=values["video"],
            action_loss=values["action"],
            mcp_loss=values["mcp"],
            mcp_depth_losses=values.get("mcp_depths"),
            allocator_retries=memory_stats.get("num_alloc_retries", 0),
            allocator_ooms=memory_stats.get("num_ooms", 0),
            loss=values["video"] + values["action"] + values["mcp"],
            lr=values["lr"],
            grad_norm=values["grad_norm"],
            optimizer_skipped=values["skipped"],
            samples_per_epoch=len(trainer.train_loader.dataset),
            sampling_mode=trainer.config.get("ig_sampling_mode", "episode"),
            samples=trainer.ig_group_count,
            effective_batch=trainer.config.ig_effective_batch,
            action_history_mode=trainer.config.get("ig_action_history_mode", ACTION_HISTORY_MODE),
            batch_per_gpu=trainer.config.batch_size,
            world_size=trainer.config.world_size,
            gpu_peak_gib=float(memory[0]),
            gpu_reserved_gib=float(memory[1]),
            shapes=trainer.ig_latest_shape,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        run = Path(trainer.config.save_root)
        with (run / "loss.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        atomic_json(run / "latest_metrics.json", row)
        if trainer.ig_checks and trainer.step > trainer.ig_checks["checkpoint_step"]:
            atomic_json(
                run / "resume_verified.json",
                dict(passed=True, checks=trainer.ig_checks, four_rank_restore=True, resumed_update=trainer.step),
            )


def install(Trainer):
    native_prepare = Trainer._prepare_input_dict
    Trainer._ig_native_compute_loss = Trainer.compute_loss
    Trainer._prepare_input_dict = lambda self, batch: prepare_episode(self, batch, native_prepare)
    Trainer._get_next_batch = next_batch
    Trainer.save_checkpoint = save_checkpoint


def native_config(config, manifest):
    """Translate the public recipe into the native Trainer contract."""
    from easydict import EasyDict

    world = int(os.environ.get("WORLD_SIZE", "4"))
    stride = config.batch_size * world
    warmup = config.warmup_steps if config.warmup_steps is not None else (10 if config.protocol == "scratch" else 100)
    samples = len(manifest["sim_episodes"])
    if (config.batch_size != 1 or config.effective_batch_size != world):
        raise ValueError("Native episode loader requires one sample per rank and no accumulation")
    updates = math.ceil(samples / world)
    settings = EasyDict(
        model_path=config.model_path,
        save_root=str(Path(config.run_dir).resolve()),
        train_scope=config.train_scope,
        empty_text_emb_path=config.empty_text_embedding,
        ig_manifest=str(config.manifest_path().resolve()),
        ig_robot_frames=config.robot_frames,
        ig_action_per_frame=config.action_per_frame,
        ig_lora_rank=config.lora_rank,
        ig_sampling_mode=config.sampling_mode,
        ig_action_history_mode=config.action_history_mode,
        ig_effective_batch=config.effective_batch_size,
        ig_total_epochs=config.total_epochs,
        ig_resume=config.resume_from,
        ig_initialization=config.initialize_from,
        ig_manifest_identity=manifest["records_sha256"],
        ig_app_config=asdict(config),
        ig_code_resources=config.code_resources(),
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=world,
        seed=config.seed,
        batch_size=config.batch_size,
        load_worker=config.num_workers,
        num_steps=updates * config.total_epochs,
        gradient_accumulation_steps=math.ceil(config.effective_batch_size / stride),
        param_dtype=torch.bfloat16,
        patch_size=(1, 2, 2),
        frame_chunk_size=0,
        max_frame_chunk_size=4,
        attn_window=0,
        max_attn_window=64,
        enable_wandb=False,
        enable_mcp=config.enable_mcp,
        ig_activation_checkpointing=config.activation_checkpointing,
        num_mcp_modules=4,
        mcp_blocks_per_group=1,
        mcp_hidden_collect_layers=[3, 11, 19, 29],
        mcp_snr_shift=10.0,
        mcp_loss_weights=[0.5, 0.25, 0.15, 0.1],
        future_chunk_stride=2,
        snr_shift=config.video_snr_shift,
        action_snr_shift=config.action_snr_shift,
        video_loss_reweight=True,
        action_loss_reweight=False,
        noisy_img_prob=0.5,
        noisy_cond_min_timestep_bd=0.0,
        noisy_cond_max_timestep_bd=1.0,
        drop_icl=0.1,
        icl_rope_h=config.icl_rope_h,
        learning_rate=config.learning_rate,
        ig_lr_schedule=config.lr_schedule,
        ig_min_learning_rate=config.min_learning_rate,
        ig_lr_decay_start_step=config.lr_decay_start_step or warmup,
        ig_lr_decay_steps=config.lr_decay_steps or (updates * config.total_epochs - warmup),
        beta1=0.9,
        beta2=0.95,
        weight_decay=0.01,
        warmup_steps=warmup,
        max_norm=1.0,
        skip_step_grad_norm_multiplier=20.0,
        gc_interval=50,
        save_epoch_freq=config.save_epoch_freq,
        save_step_freq=config.save_step_freq,
        save_seconds=config.save_seconds,
    )
    if config.normalization_path:
        settings.ig_normalization = json.loads(Path(config.normalization_path).read_text())
        settings.ig_action_representation = config.action_representation
        settings.ig_action_channel_offset = config.action_channel_offset
    return settings


def run(config):
    from filelock import Timeout

    from .cache import require_complete
    from .runtime import atomic_json_save
    from .runtime import setup_runtime
    from .runtime import start_log
    from .runtime import training_rank_lock

    config.validate(training=True)
    manifest = require_complete(config)
    settings = native_config(config, manifest)
    if config.dry_run:
        print(json.dumps(dict(settings), default=str, indent=2))
        return
    if "RANK" not in os.environ:
        raise ValueError("Launch training with torchrun; --dry-run is CPU-only")
    root = Path(config.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "checkpoints/latest.pt").exists() and not config.resume_from:
        raise ValueError("Existing checkpoint found: use a new run directory or explicit --resume-from")
    lock = training_rank_lock(root, settings.rank)
    try:
        lock.acquire(timeout=0)
    except Timeout as error:
        raise RuntimeError(f"This run rank {settings.rank} is already active") from error
    setup_runtime(config.cache_dir)
    start_log(root, f"train_rank{settings.rank}")
    os.environ["IG_LOCAL_SHARDING"] = "1" if config.local_sharding else "0"
    from .wan_va.distributed.util import init_distributed
    from .wan_va.utils import init_logger

    torch._dynamo.config.recompile_limit = max(64, torch._dynamo.config.recompile_limit)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    init_logger()
    state_path = root / f"rank{settings.rank}_status.json"
    try:
        init_distributed(settings.world_size, settings.local_rank, settings.rank)
        atomic_json_save(
            {
                "status": "running",
                "pid": os.getpid(),
                "rank": settings.rank,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            state_path,
        )
        from .wan_va.train import Trainer

        trainer = Trainer(settings)
        trainer.train()
        dist.barrier()
        complete = trainer.step >= settings.num_steps
        atomic_json_save(
            {
                "status": "completed" if complete else "paused",
                "pid": os.getpid(),
                "step": trainer.step,
                "progress": trainer.ig_progress,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            state_path,
        )
        if settings.rank == 0:
            export_loss(root)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        atomic_json_save(
            {
                "status": "failed",
                "pid": os.getpid(),
                "error": repr(error),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            state_path,
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        lock.release()


def export_loss(root):
    import csv

    rows = [json.loads(s) for s in (root / "loss.jsonl").read_text().splitlines() if s.strip()]
    columns = ["step", "epoch", "samples", "loss", "video_loss", "action_loss", "lr", "grad_norm", "update_seconds"]
    with (root / "loss.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 4))
    for key in ["loss", "video_loss", "action_loss"]:
        axis.plot([r["step"] for r in rows], [r[key] for r in rows], label=key)
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("Training loss")
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "loss.png", dpi=160)
    plt.close(fig)


def adapt_transformer(model, config):
    report = inject_adapters(model, config)
    path = config.get("ig_initialization")
    if path:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if report["trainable_shapes"] != state["adaptation"]["trainable_shapes"]:
            raise ValueError("Initialization checkpoint adapter scope/rank differs")
        expected = set(report["trainable_shapes"])
        if set(state["adapter_state_dict"]) != expected:
            raise ValueError("Initialization checkpoint adapter coverage differs")
        model.load_state_dict(state["adapter_state_dict"], strict=False)
        report["initialized_from"] = str(Path(path).resolve())
        report["optimizer_reset"] = True
    return report
