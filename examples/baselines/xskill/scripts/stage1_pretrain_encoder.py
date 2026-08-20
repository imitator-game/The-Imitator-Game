import sys 
import os
sys.path.append("../../../")
print(sys.path)

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import numpy as np
from examples.baselines.lerobot_dataset.lerobot_xskill_task_dataset import (
    IntegratedDatasetConfig,
    IntegratedTaskPairDataset
)
from xskill.dataset.xskill_imitator_dataset import IndexBatch
from xskill.utility.transform import get_transform_pipeline
from pytorch_lightning import seed_everything


class PairedToXSkillAdapter(torch.utils.data.Dataset):
    """Adapter from IntegratedTaskPairDataset to stage1 expected (robot, human) IndexBatch pair."""

    def __init__(self, paired_dataset: IntegratedTaskPairDataset, transform=None, slide=1):
        self.paired_dataset = paired_dataset
        self.transform = transform   # Store the transform for later use.
        self.slide = slide

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
        robot_clip = sample["robot_video"]
        human_clip = sample["human_video"]
        
        task_name = sample["target_task_id"]
        info = {"task_idx": 0, "task_name": task_name, "vid_idx": int(idx)}

        # robot_batch = IndexBatch(self._to_tchw_float01(robot_clip), idx, info)
        # human_batch = IndexBatch(self._to_tchw_float01(human_clip), idx, info)
        # return robot_batch, human_batch
        robot_clip = self._to_tchw_float01(robot_clip)
        human_clip = self._to_tchw_float01(human_clip)
        def make_views(clip):
            # Sliding-window slicing; each segment calls transform independently.
            clip = torch.from_numpy(clip) 
            sub_clips = [clip[j:j + self.slide + 1] 
                        for j in range(len(clip) - self.slide)]
            
            im_q = torch.stack([self.transform(s) for s in sub_clips])  # (N, slide+1, C, H, W)
            im_k = torch.stack([self.transform(s) for s in sub_clips])  # Call again for independent augmentation.
            return im_q, im_k

        robot_q, robot_k = make_views(robot_clip)
        human_q, human_k = make_views(human_clip)

        robot_batch = IndexBatch((robot_q, robot_k), idx, info)
        human_batch = IndexBatch((human_q, human_k), idx, info)
        return robot_batch, human_batch


@hydra.main(version_base=None,
            config_path="../config",
            config_name="stage1_pretrain_encoder")
def pretrain(cfg: DictConfig):
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"output_dir: {output_dir}")
    pretrain_pipeline = get_transform_pipeline(cfg.augmentations)

    seed_everything(cfg.seed, workers=True)
    print(f"[stage1] Training on {cfg.target_domain} dataset.")
    paired_cfg = IntegratedDatasetConfig(
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
        target_domain=cfg.target_domain,
        split=cfg.split,
        cameras=list(cfg.cameras),
        include_depth=cfg.include_depth,
        image_size=tuple(cfg.image_size),
        num_frames=int(cfg.num_frames),
        sampling_strategy=cfg.sampling_strategy,
        state_type=cfg.state_type,
        single_arm=cfg.single_arm,
        fps=int(cfg.fps),
        include_first_frame=False,
        enable_augmentation=cfg.enable_augmentation,
        pre_decode=cfg.pre_decode,
        pre_decode_cache_dir=cfg.pre_decode_cache_dir,
        pre_decode_num_workers=cfg.pre_decode_num_workers,
        skip_states=True,
    )
    paired_dataset = IntegratedTaskPairDataset(paired_cfg)
    combine_dataset = PairedToXSkillAdapter(paired_dataset, transform=pretrain_pipeline, slide=cfg.Model.slide)

    dataloader = torch.utils.data.DataLoader(
        combine_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=1,
        drop_last=cfg.drop_last)

    steps_per_epoch = len(dataloader)

    model = hydra.utils.instantiate(
        cfg.Model,
        steps_per_epoch=steps_per_epoch,
        pretrain_pipeline=pretrain_pipeline,
    )

    print("dataset len: ", len(combine_dataset))

    # checkpoint_callback = ModelCheckpoint(
    #     every_n_epochs=cfg.callback.every_n_epoch,
    #     # every_n_train_steps=200,
    #     save_top_k=-1,
    #     dirpath=output_dir,
    #     filename="{epoch:02d}",
    # )
    step_checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=2000,
        save_top_k=0,
        save_last=True,
        save_on_train_epoch_end=False,
        dirpath=output_dir,
        filename="step={step}-epoch={epoch:02d}",
    )

    epoch_checkpoint_callback = ModelCheckpoint(
        every_n_epochs=cfg.callback.every_n_epoch,
        every_n_train_steps=None,
        save_top_k=-1,
        save_last=False,
        dirpath=output_dir,
        filename="epoch={epoch:02d}",
    )

    use_wandb = os.environ.get("WANDB_DISABLED", "").lower() not in {"1", "true", "yes"}
    # Set up logger
    if use_wandb:
        wandb.init(project="xskill_pretrain_encoder")
        # wandb_logger = WandbLogger(project="visual_skill_prior")
        wandb.config.update(OmegaConf.to_container(cfg))

    if not torch.cuda.is_available():
        cfg.Trainer.accelerator = "cpu"
        cfg.Trainer.devices = 1
    
    resume_from_checkpoint = OmegaConf.select(
        cfg, "resume_from_checkpoint", default=None
    )

    if resume_from_checkpoint is not None:
        resume_from_checkpoint = str(resume_from_checkpoint)
        if resume_from_checkpoint.lower() in {"", "none", "null", "false", "0"}:
            resume_from_checkpoint = None

    print(f"resume_from_checkpoint: {resume_from_checkpoint}")

    trainer = pl.Trainer(
        # logger=wandb_logger,
        # callbacks=[checkpoint_callback],
        callbacks=[
            step_checkpoint_callback,
            epoch_checkpoint_callback,
        ],
        enable_checkpointing=True,
        default_root_dir=output_dir,
        deterministic=True,
        **cfg.Trainer,
    )

    trainer.fit(model=model, train_dataloaders=dataloader, ckpt_path=resume_from_checkpoint)


if __name__ == "__main__":
    pretrain()
