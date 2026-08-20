"""
VQ-VAE Pretraining Script for LeRobot Action Data
===================================================
Pretrains the VQ-VAE component of VQ-BeT using action sequences from the
LeRobot sim dataset.

Key optimisations over the original script
-------------------------------------------
1.  **Bulk Arrow extraction** – Instead of calling sub_ds[i] in a Python loop
    (1 M+ round-trips through HF dataset machinery), we pull the entire action
    column in one shot:
        actions_np = np.array(hf_dataset[action_key])   # shape [N, action_dim]
    Then we build sliding windows with np.lib.stride_tricks.sliding_window_view,
    skipping episode boundaries. This replaces the delta_timestamps mechanism
    entirely for this use-case.

2.  **Single pre-allocated tensor** – All valid windows are stacked into one
    contiguous float32 tensor stored in RAM. __getitem__ is a pure index slice –
    zero Python overhead, zero disk I/O, zero HF machinery during training.

3.  **Vectorised normalisation** – min/max bounds are applied to the whole
    tensor in one torch broadcast op rather than per-sample.

4.  **num_workers=0 stays optimal** – everything is in RAM already; forking
    workers just wastes memory and startup time.

Bug fixes
---------
- normalize_action was called with method="min_max" which doesn't exist in
  ActionNormalizer. Fixed to "bounds_q99" (same semantics, correct key).
"""

import os
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tyro

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

from examples.baselines.vqbet.vqbet.vqvae import VqVae
from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_episodes_idx(sub_episodes) -> Optional[List[int]]:
    """Parse episode spec from dataset JSON config.

    Handles all formats used across the codebase:
      None                     → load all episodes
      list                     → use as-is
      str "0:45"               → range(0, 45)
      str "0:" or ":"          → None (load all)
      str "[0,1,2]"            → eval'd list
      dict {"start":0,"end":45}→ range(0, 45)
      dict {"end": None}       → None (load all)
    """
    if sub_episodes is None:
        return None
    if isinstance(sub_episodes, list):
        return sub_episodes
    if isinstance(sub_episodes, str):
        s = sub_episodes.strip()
        if ":" in s:
            parts  = s.split(":", 1)
            start_s, end_s = parts[0].strip(), parts[1].strip()
            start  = int(start_s) if start_s else 0
            if not end_s:
                return None        # "0:" → load all
            return list(range(start, int(end_s)))
        return eval(s)
    if isinstance(sub_episodes, dict):
        start = sub_episodes.get("start", 0)
        end   = sub_episodes.get("end")
        if end is None:
            return None
        return list(range(start, end))
    return None


def _get_norm_bounds(stats: dict, key: str, method: str):
    """Return (min_vec, max_vec) numpy arrays or (None, None)."""
    if key not in stats:
        return None, None
    s = stats[key]
    if method == "bounds":
        return s.get("min"), s.get("max")
    elif method in ("bounds_q99", "min_max"):   # accept both names
        return s.get("q01"), s.get("q99")
    raise ValueError(f"Unknown normalisation method: {method}")


# ---------------------------------------------------------------------------
# ActionOnlyDataset  –  bulk-loaded, RAM-resident
# ---------------------------------------------------------------------------

class ActionOnlyDataset(Dataset):
    """Action-only dataset backed entirely by a single contiguous RAM tensor.

    Loading strategy
    ~~~~~~~~~~~~~~~~
    For each sub-dataset we:
      1. Load LeRobotDataset WITHOUT delta_timestamps (so __getitem__ is never
         called during construction and the HF Arrow table stays lazy).
      2. Pull the action column and episode_index column in one batch read from
         the Arrow table.  This is a single call that returns the whole column
         as a list and is then converted to numpy.  Typical throughput: millions
         of rows in a few seconds.
      3. Build a boolean validity mask that marks frames where the next
         (horizon-1) frames all belong to the same episode.
      4. Use np.lib.stride_tricks.sliding_window_view to create a zero-copy
         view of shape [N_valid, horizon, action_dim].
      5. Copy the valid windows into a single pre-allocated float32 tensor.
      6. Apply vectorised normalisation (one torch broadcast per sub-dataset).

    __getitem__ is then a pure tensor[idx] slice.
    """

    def __init__(
        self,
        sim_root: str,
        sim_dataset_file: str,
        state_type: str = "qpos",
        horizon: int = 16,
        split: str = "train",
        single_arm: bool = False,
        fps: int = 30,
        normalize: bool = True,
        video_backend: str = "torchcodec",
        tolerance_s: float = 0.04,
        norm_method: str = "bounds_q99",
        # Real-robot mode: when set, overrides sim_root / sim_dataset_file
        data_source: str = "sim",
        robot_root: Optional[str] = None,
        robot_dataset_file: Optional[str] = None,
    ):
        self.horizon = horizon

        # Resolve action key (same mapping for sim and robot)
        if state_type == "eepos":
            action_key = "action.eepos_gripper_actions"
        else:
            action_key = "action.qpos_gripper_actions"
        self.action_key = action_key

        # Select dataset source
        if data_source == "robot":
            if not robot_dataset_file:
                raise ValueError("robot_dataset_file is required when data_source='robot'")
            data_root        = robot_root or sim_root
            dataset_cfg_path = robot_dataset_file
        else:
            data_root        = sim_root
            dataset_cfg_path = sim_dataset_file

        with open(dataset_cfg_path, "r") as f:
            dataset_configs = json.load(f)

        # Normalizer (populated per sub-dataset before we apply it)
        normalizer = ActionNormalizer() if normalize else None

        all_chunks: List[torch.Tensor] = []   # one tensor per sub-dataset
        ds_ids_list: List[torch.Tensor] = []  # matching dataset indices

        for ds_idx, ds_cfg in enumerate(tqdm(dataset_configs, desc="Loading datasets")):
            repo_id      = ds_cfg.get("repo_id")
            ds_root      = os.path.join(data_root, ds_cfg.get("root", repo_id))
            sub_episodes = get_episodes_idx(ds_cfg.get(split))

            # -----------------------------------------------------------------
            # Load metadata only; do NOT pass delta_timestamps so the HF table
            # stays unfiltered and we can read columns in bulk.
            # -----------------------------------------------------------------
            sub_ds = LeRobotDataset(
                repo_id=repo_id,
                root=ds_root,
                delta_timestamps=None,      # <-- key: no windowing here
                video_backend=video_backend,
                tolerance_s=tolerance_s,
                episodes=sub_episodes,
            )

            hf = sub_ds.hf_dataset

            # -----------------------------------------------------------------
            # Bulk-read action and episode_index columns from Arrow table.
            # hf[col] returns a list; np.array converts in one shot.
            # -----------------------------------------------------------------
            print(f"  [{repo_id}] Reading {len(hf):,} rows from Arrow table…", flush=True)

            if action_key not in hf.column_names:
                print(f"  WARNING: {action_key} not found, skipping {repo_id}")
                continue

            t0 = time.time()
            actions_np     = np.array(hf[action_key],      dtype=np.float32)   # [N, D]
            episode_idx_np = np.array(hf["episode_index"], dtype=np.int64)     # [N]
            print(f"  [{repo_id}] Arrow read done in {time.time()-t0:.1f}s, "
                  f"shape={actions_np.shape}", flush=True)

            N, D = actions_np.shape

            # single_arm slice (take first half of action dims)
            if single_arm:
                D = D // 2
                actions_np = actions_np[:, :D]

            # -----------------------------------------------------------------
            # Build validity mask:
            # frame i is valid iff frames i..i+horizon-1 are same episode and
            # all exist (i+horizon-1 < N).
            # -----------------------------------------------------------------
            # valid[i] = True  iff  episode_idx[i] == episode_idx[i+h-1]
            # (since episodes are contiguous, checking endpoints is sufficient)
            valid = np.ones(N, dtype=bool)
            valid[N - horizon + 1:] = False                          # guard end
            same_ep = episode_idx_np[:N - horizon + 1] == episode_idx_np[horizon - 1:]
            valid[:N - horizon + 1] &= same_ep

            n_valid = int(valid.sum())
            if n_valid == 0:
                print(f"  [{repo_id}] No valid windows, skipping.")
                continue

            # -----------------------------------------------------------------
            # Sliding-window view (zero-copy stride trick), then copy valid rows
            # -----------------------------------------------------------------
            # Shape of windowed: [N - horizon + 1, horizon, D]
            windowed = np.lib.stride_tricks.sliding_window_view(
                actions_np, window_shape=horizon, axis=0
            ).transpose(0, 2, 1)        # → [N-h+1, horizon, D]

            valid_start = valid[:N - horizon + 1]
            chunk_np = windowed[valid_start].copy()                  # [n_valid, horizon, D]
            chunk = torch.from_numpy(chunk_np)                        # float32

            # -----------------------------------------------------------------
            # Vectorised normalisation
            # -----------------------------------------------------------------
            if normalizer is not None:
                normalizer.add_dataset_stats(
                    dataset_idx=ds_idx,
                    repo_id=repo_id,
                    stats=sub_ds.meta.stats,
                    state_key=action_key,
                    action_key=action_key,
                    single_arm=single_arm,
                )
                a_min_arr, a_max_arr = _get_norm_bounds(
                    sub_ds.meta.stats, action_key, norm_method
                )
                if a_min_arr is not None and a_max_arr is not None:
                    a_min = torch.tensor(a_min_arr, dtype=torch.float32)
                    a_max = torch.tensor(a_max_arr, dtype=torch.float32)
                    if single_arm:
                        a_min = a_min[:D]
                        a_max = a_max[:D]
                    denom = (a_max - a_min).clamp(min=1e-8)
                    chunk = 2.0 * (chunk - a_min) / denom - 1.0
                    chunk = chunk.clamp(-1.0, 1.0)

            all_chunks.append(chunk)
            ds_ids_list.append(
                torch.full((n_valid,), ds_idx, dtype=torch.long)
            )
            print(f"  [{repo_id}] {n_valid:,} valid windows  "
                  f"(action_dim={D})", flush=True)

        if not all_chunks:
            raise RuntimeError("No valid windows found across all sub-datasets!")

        # -----------------------------------------------------------------
        # Concatenate everything into two big contiguous tensors
        # -----------------------------------------------------------------
        self._actions  = torch.cat(all_chunks,  dim=0)   # [Total, horizon, D]
        self._ds_ids   = torch.cat(ds_ids_list, dim=0)   # [Total]
        self.action_dim = self._actions.shape[-1]
        self.normalizer = normalizer

        ram_mb = self._actions.nelement() * 4 / 1e6
        print(
            f"\n[ActionOnlyDataset] Ready: {len(self._actions):,} windows, "
            f"action_dim={self.action_dim}, horizon={horizon}, "
            f"RAM={ram_mb:.0f} MB",
            flush=True,
        )

    def __len__(self) -> int:
        return self._actions.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "robot_actions": self._actions[idx],
            "dataset_idx":   self._ds_ids[idx],
        }


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class PretrainArgs:
    exp_name: Optional[str] = None
    seed: int = 1
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "VQVAEPretrain"
    wandb_entity: Optional[str] = None

    sim_root: str = "demos"
    sim_dataset_file: str = "examples/baselines/lerobot_dataset/config/sim_config.json"
    state_type: str = "qpos"
    single_arm: bool = False
    fps: int = 30
    normalize: bool = True
    norm_method: str = "bounds_q99"

    # ── Real-robot data source ────────────────────────────────────────────────
    # Set data_source="robot" to pretrain VQ-VAE on real robot action data.
    # Useful when finetuning VQ-BeT on robot demonstrations.
    data_source: str = "sim"                      # "sim" | "robot"
    robot_root: str = "demos"
    robot_dataset_file: Optional[str] = None
    """Normalisation method passed to ActionNormalizer.
    Accepted values: bounds | bounds_q99  (min_max is also accepted as alias)."""

    epochs: int = 100
    batch_size: int = 256
    num_dataload_workers: int = 0
    """0 is optimal when all data is pre-loaded in a single RAM tensor."""

    action_dim: int = 16
    act_horizon: int = 10
    n_latent_dims: int = 512
    vqvae_n_embed: int = 32
    vqvae_groups: int = 2
    encoder_loss_multiplier: float = 1.0
    act_scale: float = 1.0

    try_compile: bool = True
    save_path: str = "pretrained_vqvae"
    save_freq: int = 10


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _try_compile(module: nn.Module, label: str) -> nn.Module:
    try:
        compiled = torch.compile(module, mode="default")
        print(f"   torch.compile applied to {label}")
        return compiled
    except Exception as e:
        print(f"   torch.compile unavailable for {label}: {e}")
        return module


def _collate_vqvae(batch):
    return {
        "robot_actions": torch.stack([b["robot_actions"] for b in batch]),
        "dataset_idx":   torch.stack([b["dataset_idx"]   for b in batch]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = tyro.cli(PretrainArgs)
    seed_everything(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    run_name  = args.exp_name or f"vqvae_{args.seed}_{int(time.time())}"
    save_path = Path(args.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    if args.track and _HAS_WANDB:
        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            name=run_name, config=vars(args),
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.data_source == "robot":
        if not args.robot_dataset_file:
            raise ValueError(
                "--robot-dataset-file is required when --data-source robot"
            )
        print(f"[DataSource] Pretraining VQ-VAE on REAL ROBOT data "
              f"(robot_root={args.robot_root})")
    else:
        print(f"[DataSource] Pretraining VQ-VAE on SIM data "
              f"(sim_root={args.sim_root})")

    dataset = ActionOnlyDataset(
        sim_root=args.sim_root,
        sim_dataset_file=args.sim_dataset_file,
        state_type=args.state_type,
        horizon=args.act_horizon,
        split="train",
        single_arm=args.single_arm,
        fps=args.fps,
        normalize=args.normalize,
        norm_method=args.norm_method,
        data_source=args.data_source,
        robot_root=args.robot_root,
        robot_dataset_file=args.robot_dataset_file,
    )

    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_dataload_workers,
        collate_fn=_collate_vqvae,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    if args.num_dataload_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(dataset, **dl_kwargs)
    print(f"   {len(dataloader)} batches/epoch  (workers={args.num_dataload_workers})")

    # ── VQ-VAE ────────────────────────────────────────────────────────────────
    vqvae = VqVae(
        obs_dim=args.action_dim,
        input_dim_h=args.act_horizon,
        input_dim_w=args.action_dim,
        n_latent_dims=args.n_latent_dims,
        vqvae_n_embed=args.vqvae_n_embed,
        vqvae_groups=args.vqvae_groups,
        eval=False,
        device=device,
        load_dir=None,
        encoder_loss_multiplier=args.encoder_loss_multiplier,
        act_scale=args.act_scale,
    )

    if args.try_compile and hasattr(torch, "compile") and device.type == "cuda":
        vqvae.encoder = _try_compile(vqvae.encoder, "encoder")
        vqvae.decoder = _try_compile(vqvae.decoder, "decoder")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_recon  = float("inf")
    global_step = 0

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        enc_sum = vq_sum = recon_sum = 0.0
        n = 0
        all_codes: List[torch.Tensor] = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            action_chunk = batch["robot_actions"].to(device, non_blocking=True)
            enc_loss, vq_loss, vq_code, recon_loss = vqvae.vqvae_update(action_chunk)

            ev = enc_loss.item() if isinstance(enc_loss, torch.Tensor) else enc_loss
            vv = vq_loss.item()  if isinstance(vq_loss,  torch.Tensor) else vq_loss
            rv = recon_loss

            enc_sum += ev; vq_sum += vv; recon_sum += rv; n += 1

            if args.track and _HAS_WANDB:
                wandb.log(
                    {"train/enc": ev, "train/vq": vv,
                     "train/recon": rv, "train/epoch": epoch},
                    step=global_step,
                )
            global_step += 1
            all_codes.append(vq_code.cpu())

        avg_enc, avg_vq, avg_recon = enc_sum/n, vq_sum/n, recon_sum/n
        codes_cat = torch.cat(all_codes, 0)
        n_codes   = len(torch.unique(codes_cat))
        n_combos  = len(torch.unique(codes_cat, dim=0))

        if args.track and _HAS_WANDB:
            wandb.log(
                {"epoch/enc": avg_enc, "epoch/vq": avg_vq,
                 "epoch/recon": avg_recon, "epoch/unique_codes": n_codes,
                 "epoch/unique_combos": n_combos},
                step=global_step,
            )

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"\nEpoch {epoch}: enc={avg_enc:.4f}  vq={avg_vq:.4f}  "
                f"recon={avg_recon:.4f}  codes={n_codes}  combos={n_combos}"
            )

        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            torch.save(vqvae.get_vqvae_state_dict(), save_path / f"vqvae_epoch_{epoch}.pt")

        if avg_recon < best_recon:
            best_recon = avg_recon
            torch.save(vqvae.get_vqvae_state_dict(), save_path / "vqvae_best.pt")
            print(f"  ✨ New best recon={avg_recon:.4f}")

    torch.save(vqvae.get_vqvae_state_dict(), save_path / "vqvae_final.pt")
    with open(save_path / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\n✅ Done! Saved to {save_path}")

    if args.track and _HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()