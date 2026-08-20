import os
import pickle
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from omegaconf import DictConfig, OmegaConf
from xskill.model.diffusion_model import get_resnet, replace_bn_with_gn
from xskill.model.encoder import ResnetConv
import random
import copy
from pathlib import Path
from torchvision import transforms
import torchvision.transforms as Tr
from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanSimPairedDataset,
    HumanRobotPairedDataset,
    PairedDatasetConfig,
)
from typing import Dict

_PROTO_EXTRACT_LOGGED = False
_PROTO_SNAP_LOGGED = False


def _fit_horizon(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] > target_len:
        return arr[:target_len]
    pad = np.repeat(arr[-1:], target_len - arr.shape[0], axis=0)
    return np.concatenate([arr, pad], axis=0)


def _extract_right_arm_np(x: np.ndarray, single_arm: bool) -> np.ndarray:
    if not single_arm:
        return x
    m = x.shape[-1] // 2
    return np.concatenate([x[..., : m - 1], x[..., 2 * m - 2 : 2 * m - 1]], axis=-1)


def _get_traj_dataset_from_paired(paired_dataset):
    """
    HumanSimPairedDataset exposes sim_dataset;
    HumanRobotPairedDataset exposes robot_dataset.
    This helper normalizes access to the trajectory dataset used for BC training.
    """
    if hasattr(paired_dataset, "sim_dataset"):
        return paired_dataset.sim_dataset, "sim"
    if hasattr(paired_dataset, "robot_dataset"):
        return paired_dataset.robot_dataset, "robot"
    raise TypeError(
        f"Unsupported paired dataset type: {type(paired_dataset)}. "
        "Expected HumanSimPairedDataset or HumanRobotPairedDataset."
    )


def _compute_stats_from_paired_dataset(paired_dataset) -> Dict[str, Dict[str, np.ndarray]]:
    traj_dataset, traj_type = _get_traj_dataset_from_paired(paired_dataset)
    print(f"[stage2] Computing normalization stats from paired {traj_type} metadata")

    normalizer = getattr(traj_dataset, "normalizer", None)

    obs_min = None
    obs_max = None
    act_min = None
    act_max = None

    if normalizer is not None:
        norm_method = getattr(traj_dataset.config, "normalization_method", "bounds_q99")
        lo_key, hi_key = ("q01", "q99") if norm_method == "bounds_q99" else ("min", "max")

        for ds_idx, ds_stats in normalizer.dataset_stats.items():
            info = normalizer.dataset_info.get(ds_idx, {})
            state_key = info.get("state_key")
            action_key = info.get("action_key")
            single_arm = bool(info.get("single_arm", False))

            if state_key in ds_stats:
                s_lo = ds_stats[state_key].get(lo_key)
                s_hi = ds_stats[state_key].get(hi_key)
                if s_lo is not None and s_hi is not None:
                    s_lo = _extract_right_arm_np(np.asarray(s_lo, dtype=np.float32), single_arm)
                    s_hi = _extract_right_arm_np(np.asarray(s_hi, dtype=np.float32), single_arm)
                    obs_min = s_lo if obs_min is None else np.minimum(obs_min, s_lo)
                    obs_max = s_hi if obs_max is None else np.maximum(obs_max, s_hi)

            if action_key in ds_stats:
                a_lo = ds_stats[action_key].get(lo_key)
                a_hi = ds_stats[action_key].get(hi_key)
                if a_lo is not None and a_hi is not None:
                    a_lo = _extract_right_arm_np(np.asarray(a_lo, dtype=np.float32), single_arm)
                    a_hi = _extract_right_arm_np(np.asarray(a_hi, dtype=np.float32), single_arm)
                    act_min = a_lo if act_min is None else np.minimum(act_min, a_lo)
                    act_max = a_hi if act_max is None else np.maximum(act_max, a_hi)

    if obs_min is None or obs_max is None or act_min is None or act_max is None:
        raise RuntimeError(f"Failed to compute stats from paired {traj_type} metadata")

    print(
        f"[stage2] Stats ready from {traj_type}: "
        f"obs_dim={obs_min.shape[0]}, action_dim={act_min.shape[0]}"
    )

    return {
        "obs": {"min": obs_min.astype(np.float32), "max": obs_max.astype(np.float32)},
        "actions": {"min": act_min.astype(np.float32), "max": act_max.astype(np.float32)},
    }


class PairedToStage2Adapter(torch.utils.data.Dataset):
    def __init__(self, paired_dataset, pred_horizon: int, obs_horizon: int, snap_frames: int):
        self.paired_dataset = paired_dataset
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.snap_frames = snap_frames
        print(
            f"[stage2] Adapter initialized: samples={len(self.paired_dataset)}, "
            f"pred_horizon={self.pred_horizon}, obs_horizon={self.obs_horizon}"
        )

    def __len__(self):
        return len(self.paired_dataset)

    @staticmethod
    def _to_tchw_float01(clip) -> np.ndarray:
        clip_np = clip.detach().cpu().numpy() if isinstance(clip, torch.Tensor) else np.asarray(clip)
        if clip_np.ndim != 4:
            raise ValueError(f"Expected clip shape (T,H,W,C) or (T,C,H,W), got {clip_np.shape}")
        if clip_np.shape[-1] not in (1, 3, 4) and clip_np.shape[1] in (1, 3, 4):
            clip_np = np.transpose(clip_np, (0, 2, 3, 1))
        if clip_np.dtype != np.uint8:
            if clip_np.max() <= 1.0:
                clip_np = (clip_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                clip_np = clip_np.clip(0, 255).astype(np.uint8)
        return np.transpose(clip_np, (0, 3, 1, 2)).astype(np.float32) / 255.0

    def __getitem__(self, idx):
        sample = self.paired_dataset[idx]
        robot_obs = sample["robot_obs"]
        # view_keys = [k for k in robot_obs.keys() if k != "states"]
        # if not view_keys:
        #     raise ValueError("No robot view found in paired sample")
        # robot_clip = robot_obs[sorted(view_keys)[0]]
        robot_clip = sample["skill_frames"]["view_1"]
        human_clip = sample["human_video"]
        obs = sample["robot_obs"]["states"]
        actions = sample["robot_actions"]
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        obs = _fit_horizon(np.asarray(obs, dtype=np.float32), self.obs_horizon)
        actions = _fit_horizon(np.asarray(actions, dtype=np.float32), self.pred_horizon)
        robot_images = _fit_horizon(self._to_tchw_float01(robot_clip), self.obs_horizon)
        human_images = _fit_horizon(
            self._to_tchw_float01(human_clip),
            self.snap_frames
        )
        return {
            "obs": torch.from_numpy(
                np.ascontiguousarray(obs[: self.obs_horizon])
            ).float().clone(),

            "actions": torch.from_numpy(
                np.ascontiguousarray(actions[: self.pred_horizon])
            ).float().clone(),

            "images": torch.from_numpy(
                np.ascontiguousarray(robot_images[: self.obs_horizon])
            ).float().clone(),

            "human_images": torch.from_numpy(
                np.ascontiguousarray(human_images)
            ).float().clone(),
        }


def _repeat_last_proto_batch(rep: torch.Tensor, target_len: int) -> torch.Tensor:
    if rep.shape[1] >= target_len:
        return rep[:, :target_len]
    last = rep[:, -1:, :].repeat(1, target_len - rep.shape[1], 1)
    return torch.cat([rep, last], dim=1)


def _load_xskill_model(cfg: DictConfig, device: torch.device):
    config_path = Path(cfg.pretrain_path) / ".hydra" / "config.yaml"
    print(f"[stage2] Loading stage1 config from: {config_path}")
    exp_cfg = OmegaConf.load(str(config_path))
    model = hydra.utils.instantiate(exp_cfg.Model).to(device)
    ckpt_id = int(cfg.pretrain_ckpt)
    candidates = [
        Path(cfg.pretrain_path) / f"{ckpt_id}.ckpt",
        Path(cfg.pretrain_path) / f"{ckpt_id:02d}.ckpt",
        Path(cfg.pretrain_path) / f"{ckpt_id:04d}.ckpt",
        Path(cfg.pretrain_path) / f"epoch{ckpt_id}.ckpt",
        Path(cfg.pretrain_path) / f"epoch{ckpt_id:02d}.ckpt",
        Path(cfg.pretrain_path) / f"epoch{ckpt_id:04d}.ckpt",
        Path(cfg.pretrain_path) / f"epoch={ckpt_id}.ckpt",
        Path(cfg.pretrain_path) / f"epoch={ckpt_id:02d}.ckpt",
        Path(cfg.pretrain_path) / f"epoch={ckpt_id:04d}.ckpt",
    ]
    ckpt_path = None
    for c in candidates:
        if c.exists():
            ckpt_path = c
            break
    if ckpt_path is None:
        all_ckpts = sorted(Path(cfg.pretrain_path).glob("*.ckpt"))
        if len(all_ckpts) == 0:
            raise FileNotFoundError(f"No checkpoint found under {cfg.pretrain_path}")
        ckpt_path = all_ckpts[-1]
        print(f"Checkpoint {cfg.pretrain_ckpt} not found, using latest: {ckpt_path}")
    print(f"[stage2] Loading stage1 checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[stage2] Stage1 model loaded and frozen")
    return model


def _extract_traj_representation(model, images: torch.Tensor, pipeline: nn.Module) -> torch.Tensor:
    global _PROTO_EXTRACT_LOGGED
    images = images.float()
    if images.max() > 1.0:
        images = images / 255.0
    b, t, c, h, w = images.shape
    images = pipeline(images.reshape(b * t, c, h, w)).reshape(b, t, c, 112, 112)
    if t <= model.slide:
        raise ValueError(f"Need T > slide for proto extraction, got T={t}, slide={model.slide}")
    windows = [images[:, j : j + model.slide + 1] for j in range(t - model.slide)]
    im_q_processed = torch.cat(windows, dim=0)
    state_rep = model.encoder_q.get_state_representation(im_q_processed, None)
    traj_rep = model.encoder_q.get_traj_representation(state_rep)
    traj_rep = traj_rep.reshape(t - model.slide, b, -1).permute(1, 0, 2).contiguous()
    traj_rep = _repeat_last_proto_batch(traj_rep, t)
    if not _PROTO_EXTRACT_LOGGED:
        print(
            f"[stage2] Online proto extraction: input={tuple(images.shape)}, "
            f"slide={model.slide}, output={tuple(traj_rep.shape)}"
        )
        _PROTO_EXTRACT_LOGGED = True
    return traj_rep


def _sample_proto_snap_batch(proto_seq: torch.Tensor, snap_frames: int) -> torch.Tensor:
    global _PROTO_SNAP_LOGGED
    b, t, d = proto_seq.shape
    if t >= snap_frames:
        idx = torch.linspace(0, t - 1, steps=snap_frames, device=proto_seq.device).long()
        sampled = proto_seq[:, idx, :]
    else:
        pad = proto_seq[:, -1:, :].repeat(1, snap_frames - t, 1)
        sampled = torch.cat([proto_seq, pad], dim=1)
    if not _PROTO_SNAP_LOGGED:
        print(
            f"[stage2] Proto snap sampling: src_len={t}, snap_frames={snap_frames}, "
            f"result={tuple(sampled.shape)}"
        )
        _PROTO_SNAP_LOGGED = True
    return sampled

def _atomic_torch_save(obj, path: str):
    """
    Prevent partial writes from corrupting `last.pt` if the process is killed.
    Write to a temp file first, then replace atomically.
    """
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _save_training_checkpoint(
    path: str,
    nets: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    ema: EMAModel,
    epoch_idx: int,
    batch_idx: int,
    global_step: int,
    cfg: DictConfig,
):
    ckpt = {
        "type": "stage2_training_checkpoint",
        "nets": nets.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "ema": ema.state_dict(),
        "epoch_idx": int(epoch_idx),
        "batch_idx": int(batch_idx),
        "global_step": int(global_step),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "np_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    _atomic_torch_save(ckpt, path)
    print(
        f"[stage2] Saved training checkpoint: {path} "
        f"epoch={epoch_idx} batch={batch_idx} global_step={global_step}",
        flush=True,
    )


def _load_training_checkpoint(
    path: str,
    nets: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    ema: EMAModel,
    device: torch.device,
):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Backward compatibility: if this is just a plain state_dict, restore nets only.
    if not isinstance(ckpt, dict) or "nets" not in ckpt:
        print(f"[stage2] Loading legacy weights-only checkpoint: {path}")
        nets.load_state_dict(ckpt)
        return 0, -1, 0

    print(f"[stage2] Loading full training checkpoint: {path}")

    nets.load_state_dict(ckpt["nets"])
    optimizer.load_state_dict(ckpt["optimizer"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    if ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
        ema.to(device)

    if ckpt.get("rng_state") is not None:
        torch.set_rng_state(ckpt["rng_state"])

    if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])

    if ckpt.get("np_rng_state") is not None:
        np.random.set_state(ckpt["np_rng_state"])

    if ckpt.get("python_rng_state") is not None:
        random.setstate(ckpt["python_rng_state"])

    epoch_idx = int(ckpt.get("epoch_idx", 0))
    batch_idx = int(ckpt.get("batch_idx", -1))
    global_step = int(ckpt.get("global_step", 0))

    print(
        f"[stage2] Resumed from epoch={epoch_idx}, "
        f"batch={batch_idx}, global_step={global_step}",
        flush=True,
    )

    return epoch_idx, batch_idx, global_step

def _load_finetune_weights(
    path: str,
    nets: nn.Module,
    device: torch.device,
    strict: bool = False,
):
    """
    Finetuning helper:
    load network weights only, without restoring optimizer / scheduler / ema /
    epoch / step / rng state.

    Compatible with three .pt formats:
    1. full training checkpoint: {"nets": ..., "optimizer": ..., ...}
    2. lightning-like checkpoint: {"state_dict": ...}
    3. pure state_dict: the file is just `nets.state_dict()`
    """
    print(f"[stage2] Loading finetune weights from: {path}", flush=True)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    
    if isinstance(ckpt, dict) and "nets" in ckpt:
        state_dict = ckpt["nets"]
        print("[stage2] Detected full checkpoint format, using ckpt['nets']")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("[stage2] Detected state_dict checkpoint format, using ckpt['state_dict']")
    else:
        state_dict = ckpt
        print("[stage2] Detected raw state_dict checkpoint format")

    missing, unexpected = nets.load_state_dict(state_dict, strict=strict)

    print(
        f"[stage2] Finetune weights loaded. strict={strict}, "
        f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}",
        flush=True,
    )

    if len(missing) > 0:
        print("[stage2] Missing keys preview:", missing[:20], flush=True)
    if len(unexpected) > 0:
        print("[stage2] Unexpected keys preview:", unexpected[:20], flush=True)

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="stage2_skill_transfer",
)
def train_xskill_bc(cfg: DictConfig):
    # create save dir
    # unique_id = str(uuid.uuid4())
    # save_dir = os.path.join(cfg.save_dir, unique_id)
    save_dir = cfg.save_dir
    cfg.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(save_dir, "hydra_config.yaml"))
    print(f"output_dir: {save_dir}")
    print(
        f"[stage2] Run config: epochs={cfg.num_epochs}, batch_size={cfg.batch_size}, "
        f"use_proto={cfg.use_proto}, proto_snap={cfg.prototype_snap}"
    )
    
    use_wandb = os.environ.get("WANDB_DISABLED", "").lower() not in {"1", "true", "yes"}
    # Set up logger
    if use_wandb:
        wandb.init(project=cfg.project_name)
        wandb.config.update(OmegaConf.to_container(cfg))

    # set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # parameters
    pred_horizon = cfg.pred_horizon
    obs_horizon = cfg.obs_horizon
    snap_frames = int(cfg.get("snap_frames", 10))
    prototype_snap = bool(cfg.get("prototype_snap", True))

    need_proto = bool(cfg.get("use_proto", False))

    paired_source = str(cfg.paired_source).lower()
    if paired_source not in {"sim", "robot"}:
        raise ValueError(f"Unknown paired_source={paired_source!r}, expected 'sim' or 'robot'")

    paired_cfg = PairedDatasetConfig(
        human_root=cfg.human_root,
        sim_root=cfg.sim_root,
        robot_root=cfg.robot_root,

        task_mapping_file=cfg.task_mapping_file,

        human_dataset_file=cfg.human_dataset_file,
        sim_dataset_file=cfg.sim_dataset_file,
        robot_dataset_file=cfg.robot_dataset_file,

        human_task_description_file=cfg.human_task_description_file,
        sim_task_description_file=cfg.sim_task_description_file,
        robot_task_description_file=cfg.robot_task_description_file,

        split=cfg.split,
        cameras=list(cfg.cameras),
        include_depth=cfg.include_depth,
        image_size=tuple(cfg.image_size),

        num_frames=max(int(cfg.obs_horizon), snap_frames),
        sampling_strategy="uniform_jitter",
        video_backend=cfg.video_backend,

        horizon=int(cfg.pred_horizon),
        obs_horizon=int(cfg.obs_horizon),
        state_type=cfg.state_type,
        single_arm=cfg.single_arm,
        fps=int(cfg.fps),

        input_mode="video_only",
        include_first_frame=False,
        enable_augmentation=cfg.enable_augmentation,

        pre_decode=cfg.pre_decode,
        pre_decode_cache_dir=cfg.pre_decode_cache_dir,
        pre_decode_num_workers=cfg.pre_decode_num_workers,

        skill=True,
        xskill=True,
        robot_frame_gap=int(cfg.robot_frame_gap),
    )
    if paired_source == "sim":
        paired_dataset = HumanSimPairedDataset(paired_cfg)
    else:
        paired_dataset = HumanRobotPairedDataset(paired_cfg)

    print(
        f"[stage2] Paired dataset ready: source={paired_source}, "
        f"samples={len(paired_dataset)}"
    )
    
    dataset = PairedToStage2Adapter(
        paired_dataset=paired_dataset,
        pred_horizon=int(cfg.pred_horizon),
        obs_horizon=int(cfg.obs_horizon),
        snap_frames=snap_frames,
    )

    stats = _compute_stats_from_paired_dataset(paired_dataset)
    with open(os.path.join(save_dir, "stats.pickle"), "wb") as f:
        pickle.dump(stats, f)
    print(f"[stage2] Saved stats to {os.path.join(save_dir, 'stats.pickle')}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        prefetch_factor=1,
        persistent_workers=cfg.persistent_workers,
        drop_last=False,
    )
    print(
        f"[stage2] Dataloader ready: batches_per_epoch={len(dataloader)}, "
        f"num_workers={cfg.num_workers}"
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    
    print(
        f"[stage2] Preview batch: obs={tuple(batch['obs'].shape)}, "
        f"actions={tuple(batch['actions'].shape)}, "
        f"images={tuple(batch['images'].shape)}"
    )

    # Create vision encoder
    if cfg.vision_feature_dim == 512:
        vision_encoder = get_resnet("resnet18")
    else:
        vision_encoder = ResnetConv(embedding_size=cfg.vision_feature_dim)

    vision_encoder = replace_bn_with_gn(vision_encoder)
    vision_feature_dim = cfg.vision_feature_dim

    # observation and action dimensions
    obs_dim = cfg.obs_dim
    action_dim = cfg.action_dim
    
    # Determine whether to use prototype features
    use_proto = cfg.get("use_proto", False)
    proto_horizon = cfg.get("proto_horizon", obs_horizon)
    proto_dim = cfg.get("proto_dim", 0)

    # create network object
    if use_proto and cfg.get("upsample_proto", False):
        # With prototype upsampling
        global_cond_dim = (vision_feature_dim * obs_horizon + 
                          obs_dim * obs_horizon + 
                          proto_horizon * cfg.upsample_proto_net.out_size)
    elif use_proto:
        # With prototypes but no upsampling
        global_cond_dim = (vision_feature_dim * obs_horizon + 
                          obs_dim * obs_horizon + 
                          proto_horizon * proto_dim)
    else:
        # No prototypes - just vision and obs features
        global_cond_dim = vision_feature_dim * obs_horizon + obs_dim * obs_horizon

    noise_pred_net = hydra.utils.instantiate(
        cfg.noise_pred_net,
        global_cond_dim=global_cond_dim,
    )
    print(f"[stage2] noise_pred_net global_cond_dim={global_cond_dim}")

    # the final arch has 2-4 parts depending on configuration
    nets = nn.ModuleDict({
        "vision_encoder": vision_encoder,
        "noise_pred_net": noise_pred_net,
    })

    # Add prototype networks if enabled
    if use_proto:
        proto_pred_net = hydra.utils.instantiate(
            cfg.proto_pred_net,
            input_dim=vision_feature_dim * obs_horizon + obs_dim * obs_horizon,
        )
        nets["proto_pred_net"] = proto_pred_net
        
        if cfg.get("upsample_proto", False):
            upsample_proto_net = hydra.utils.instantiate(cfg.upsample_proto_net)
            nets["upsample_proto_net"] = upsample_proto_net

    # if hasattr(cfg, "resume_ckpt_path") and cfg.resume_ckpt_path is not None:
    #     if not os.path.exists(cfg.resume_ckpt_path):
    #         print(f"Resume checkpoint not found: {cfg.resume_ckpt_path}. Starting from scratch.")
    #     else:
    #         map_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         nets.load_state_dict(torch.load(cfg.resume_ckpt_path, map_location=map_device))
    #         print(f"Loaded pretrained model from {cfg.resume_ckpt_path}")

    noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    
    # device transfer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = nets.to(device)
    print(f"[stage2] Using device: {device}")
    # ===== Finetune weights-only loading =====
    finetune_ckpt_path = cfg.get("finetune_ckpt_path", None)
    resume_ckpt_path_for_check = cfg.get("resume_ckpt_path", None)

    if (
        finetune_ckpt_path is not None
        and str(finetune_ckpt_path).lower() not in {"", "none", "null"}
    ):
        if (
            resume_ckpt_path_for_check is not None
            and str(resume_ckpt_path_for_check).lower() not in {"", "none", "null"}
        ):
            raise ValueError(
                "Do not set both finetune_ckpt_path and resume_ckpt_path. "
                "Use finetune_ckpt_path for weights-only finetuning, "
                "or resume_ckpt_path for exact training resume."
            )

        finetune_ckpt_path = str(finetune_ckpt_path)
        if not os.path.exists(finetune_ckpt_path):
            raise FileNotFoundError(f"Finetune checkpoint not found: {finetune_ckpt_path}")

        _load_finetune_weights(
            path=finetune_ckpt_path,
            nets=nets,
            device=device,
            strict=bool(cfg.get("finetune_strict", False)),
        )

    xskill_model = _load_xskill_model(cfg, device)
    proto_pipeline = nn.Sequential(
        Tr.CenterCrop((112, 112)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ).to(device)
    print("[stage2] Prototype preprocessing pipeline initialized")

    # Exponential Moving Average
    ema = EMAModel(parameters=nets.parameters(), power=0.75)

    # Standard ADAM optimizer
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * cfg.num_epochs,
    )
        
    # Resume full training state if provided
    start_epoch = 0
    resume_batch_idx = -1
    global_step = 0

    resume_ckpt_path = cfg.get("resume_ckpt_path", None)
    if resume_ckpt_path is not None and str(resume_ckpt_path).lower() not in {"", "none", "null"}:
        resume_ckpt_path = str(resume_ckpt_path)
        if not os.path.exists(resume_ckpt_path):
            print(f"[stage2] Resume checkpoint not found: {resume_ckpt_path}. Starting from scratch.")
        else:
            start_epoch, resume_batch_idx, global_step = _load_training_checkpoint(
                resume_ckpt_path,
                nets=nets,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema=ema,
                device=device,
            )

    # create eval callback
    eval_callback = hydra.utils.instantiate(cfg.eval_callback) if "eval_callback" in cfg else None
    print(f"[stage2] Eval callback enabled: {eval_callback is not None}")

    for epoch_idx in range(start_epoch, cfg.num_epochs):
        epoch_loss = list()
        epoch_action_loss = list()
        epoch_proto_prediction_loss = list()
        print(f"[stage2][epoch {epoch_idx}] starting")

        # batch loop
        for batch_idx, nbatch in enumerate(dataloader):
            if epoch_idx == start_epoch and batch_idx <= resume_batch_idx:
                continue
            # data normalized in dataset
            # (B, obs_horizon, obs_dim)
            nobs = nbatch["obs"].to(device)
            # print(f'raw nobs shape: {nobs.shape}')
            B = nobs.shape[0]
            
            # (B, obs_horizon, 3, H, W) or None
            nimage = nbatch.get("images")
            if nimage is not None:
                nimage = nimage.to(device)
            
            # (B, pred_horizon, action_dim)
            naction = nbatch["actions"].to(device)
            
            nproto = None
            proto_snap = None
            human_proto_seq = None
            if use_proto:
                human_image = nbatch["human_images"].to(device)
                with torch.no_grad():
                    robot_proto_seq = _extract_traj_representation(
                        xskill_model,
                        nimage,
                        proto_pipeline,
                    )
                    human_proto_seq = _extract_traj_representation(
                        xskill_model,
                        human_image,
                        proto_pipeline,
                    )
                nproto = robot_proto_seq[:, -proto_horizon:, :]
                if prototype_snap:
                    proto_snap = _sample_proto_snap_batch(human_proto_seq, snap_frames)
                if batch_idx == 0:
                    print(
                        f"[stage2][epoch {epoch_idx}] proto tensors: "
                        f"robot_proto_seq={tuple(robot_proto_seq.shape)}, "
                        f"human_proto_seq={tuple(human_proto_seq.shape)}, "
                        f"nproto={tuple(nproto.shape)}"
                    )

            # encoder vision features
            if nimage is not None:
                image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))
                image_features = image_features.reshape(
                    *nimage.shape[:2], -1)  # (B, obs_horizon, visual_feature)
            else:
                # No images - use zero features
                image_features = torch.zeros(
                    B, obs_horizon, vision_feature_dim, device=device
                )

            # Concatenate vision and observation features
            # print(f'image_features shape: {image_features.shape}, nobs shape: {nobs.shape}')
            # print(f'proto_snap shape: {proto_snap.shape}')
            # print(f'nproto shape: {nproto.shape}')
            # exit(0)
            obs_feature = torch.cat(
                [image_features, nobs],
                dim=-1
            )  # (B, obs_horizon, low_dim_feature + visual_feature)

            # Predict prototypes if enabled and training proto_pred_net
            predict_proto = None
            proto_prediction_loss = torch.tensor(0.0, device=device)
            
            if use_proto and "proto_pred_net" in nets:
                if proto_snap is None and human_proto_seq is not None:
                    proto_snap = _sample_proto_snap_batch(human_proto_seq, snap_frames)
                if proto_snap is not None:
                    predict_proto = nets["proto_pred_net"](
                        obs_feature.flatten(start_dim=1),
                        proto_snap
                    )
                
                # Compute proto prediction loss if ground truth is available
                if nproto is not None and predict_proto is not None:
                    target_proto = nproto[:, -1, :]
                    proto_prediction_loss = nn.functional.mse_loss(
                        predict_proto, target_proto
                    )

            # Prepare conditioning for noise prediction
            if use_proto and cfg.get("upsample_proto", False) and nproto is not None:
                # Upsample prototypes
                upsample_proto = nets["upsample_proto_net"](
                    nproto.flatten(start_dim=1)
                )
                upsample_proto = upsample_proto.reshape(
                    B, proto_horizon, -1
                )  # (B, proto_horizon, upsample_dim)
                obs_cond = torch.cat(
                    [
                        obs_feature.flatten(start_dim=1),
                        upsample_proto.flatten(start_dim=1),
                    ],
                    dim=1,
                )
            elif use_proto and nproto is not None:
                # Use prototypes without upsampling
                obs_cond = torch.cat(
                    [
                        obs_feature.flatten(start_dim=1),
                        nproto.flatten(start_dim=1)
                    ],
                    dim=1,
                )
            else:
                # No prototypes - condition only on observations
                obs_cond = obs_feature.flatten(start_dim=1)

            # sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device
            ).long()

            # add noise to the clean images according to the noise magnitude
            # at each diffusion iteration (forward diffusion process)
            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps
            )

            # predict the noise residual
            noise_pred = noise_pred_net(
                noisy_actions,
                timesteps,
                global_cond=obs_cond
            )

            # L2 loss
            action_loss = nn.functional.mse_loss(noise_pred, noise)
            
            # Total loss
            if use_proto and nproto is not None:
                loss = action_loss + proto_prediction_loss
            else:
                loss = action_loss
            if batch_idx % 10 == 0:
                print(
                    f"[stage2][epoch {epoch_idx}] batch {batch_idx} losses: "
                    f"action={action_loss.item():.6f}, proto={proto_prediction_loss.item():.6f}, "
                    f"total={loss.item():.6f}", flush=True
                )
                batch_log = {
                    "train/epoch": epoch_idx,
                    "train/batch_idx": batch_idx,
                    "train/loss": loss.item(),
                    "train/action_loss": action_loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                if use_proto and nproto is not None:
                    batch_log["train/proto_prediction_loss"] = proto_prediction_loss.item()
                if use_wandb:
                    wandb.log(batch_log)

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # step lr scheduler every batch
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(nets.parameters())
            
            global_step += 1

            save_every_steps = int(cfg.get("save_every_steps", 2000))
            if save_every_steps > 0 and global_step % save_every_steps == 0:
                last_ckpt_path = os.path.join(save_dir, "last.pt")
                _save_training_checkpoint(
                    path=last_ckpt_path,
                    nets=nets,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    ema=ema,
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx,
                    global_step=global_step,
                    cfg=cfg,
                )

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            epoch_action_loss.append(action_loss.item())
            if use_proto and nproto is not None:
                epoch_proto_prediction_loss.append(proto_prediction_loss.item())

        # Log metrics
        log_dict = {
            "epoch": epoch_idx,
            "epoch_loss": np.mean(epoch_loss),
            "epoch_action_loss": np.mean(epoch_action_loss),
        }
        if use_proto and len(epoch_proto_prediction_loss) > 0:
            log_dict["epoch_proto_prediction_loss"] = np.mean(epoch_proto_prediction_loss)
        
        if use_wandb:
            wandb.log(log_dict)
        
        print(f"Epoch {epoch_idx}: Loss={np.mean(epoch_loss):.4f}, "
              f"Action Loss={np.mean(epoch_action_loss):.4f}")

        # Save checkpoints
        if (epoch_idx + 1) % cfg.ckpt_frequency == 0:
            # ===== Save full resume checkpoint every epoch =====
            epoch_full_ckpt_path = os.path.join(save_dir, f"epoch_{epoch_idx:04d}.pt")
            _save_training_checkpoint(
                path=epoch_full_ckpt_path,
                nets=nets,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema=ema,
                epoch_idx=epoch_idx + 1,
                batch_idx=-1,
                global_step=global_step,
                cfg=cfg,
            )

            # ===== Also update last.pt at epoch end =====
            _save_training_checkpoint(
                path=os.path.join(save_dir, "last.pt"),
                nets=nets,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                ema=ema,
                epoch_idx=epoch_idx + 1,
                batch_idx=-1,
                global_step=global_step,
                cfg=cfg,
            )

            # ===== Save EMA-only checkpoint every epoch for eval/inference =====
            ema_epoch_path = os.path.join(save_dir, f"ema_epoch_{epoch_idx:04d}.pt")
            ema_nets = copy.deepcopy(nets)
            ema.copy_to(ema_nets.parameters())
            _atomic_torch_save(ema_nets.state_dict(), ema_epoch_path)
            print(f"[stage2] Saved EMA epoch checkpoint: {ema_epoch_path}", flush=True)
            
        # Evaluation
        if eval_callback is not None and (epoch_idx + 1) % cfg.eval_cfg.eval_frequency == 0:
            demo_type_list = cfg.get("demo_type_list", ["robot"])
            task_progress_ratio_list = cfg.get("task_progress_ratio_list", [1.0])
            
            for demo_type in demo_type_list:
                cfg.eval_cfg.demo_type = demo_type
                
                for task_progress_ratio in task_progress_ratio_list:
                    if demo_type == "robot":
                        task_progress_ratio = 1
                    
                    print(f"Evaluating {demo_type} ratio: {task_progress_ratio}")
                    
                    # set task progress ratio
                    eval_callback.task_progress_ratio = task_progress_ratio
                    
                    total_rewards = []
                    order_rewards = []
                    
                    n_evaluations = 1 if epoch_idx == 0 else cfg.eval_cfg.n_evaluations
                    
                    for seed in range(n_evaluations):
                        # Create a temporary copy of the model for evaluation
                        eval_nets = copy.deepcopy(nets)
                        ema.copy_to(eval_nets.parameters())
                        total_r, order_r = eval_callback.eval(
                            eval_nets,
                            noise_scheduler,
                            stats,
                            cfg.eval_cfg,
                            save_dir,
                            seed,
                        )
                        total_rewards.append(total_r)
                        order_rewards.append(order_r)
                    
                    if use_wandb:
                        wandb.log({
                            f"eval_score/{demo_type}_{task_progress_ratio}_total_reward":
                                np.mean(total_rewards),
                            f"eval_score/{demo_type}_{task_progress_ratio}_order_reward":
                                np.mean(order_rewards),
                        })
                    
                    if demo_type == "robot":
                        break

    # Final checkpoint save
    final_ema_nets = copy.deepcopy(nets)
    ema.copy_to(final_ema_nets.parameters())
    torch.save(
        final_ema_nets.state_dict(),
        os.path.join(save_dir, "final_ckpt.pt"),
    )
    
    print(f"Training complete. Models saved to {save_dir}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train_xskill_bc()
