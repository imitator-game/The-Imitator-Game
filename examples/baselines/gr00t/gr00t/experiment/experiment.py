#!/usr/bin/env python
import json
import logging
import os
from pathlib import Path
import warnings

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from transformers import TrainingArguments, set_seed
import wandb

from gr00t.configs.base_config import Config
# Use custom trainer that profiles data loading & forward times
from gr00t.experiment.trainer import Gr00tTrainer, ProfCallback
from gr00t.experiment.utils import (
    BestMetricCheckpointCallback,
    CheckpointFormatCallback,
    EpochBoundarySaveCallback,
)
from gr00t.model import MODEL_REGISTRY
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, save_initial_actions


def setup_logging(debug: bool = False):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def warn_configs(config: Config):
    # updates to batch size
    assert config.training.global_batch_size % config.training.num_gpus == 0, (
        "global_batch_size must be divisible by num_gpus"
    )

    if config.data.video_backend != "torchcodec":
        warnings.warn(
            "video_backend is not torchcodec. Only torchcodec will be supported in the future."
        )

    if config.training.batch_size is not None:
        warnings.warn(
            "batch_size will be deprecated in the future, please use global_batch_size instead. For now, this will override global_batch_size."
        )

    if config.training.warmup_steps > 0:
        warnings.warn(
            "warmup_steps will be deprecated in the future, please use warmup_ratio instead. For now, this will override warmup_ratio."
        )

    if (
        hasattr(config.model, "backbone_trainable_params_fp32")
        and not config.model.backbone_trainable_params_fp32
    ):
        warnings.warn(
            "backbone_trainable_params_fp32 is not True. This will be deprecated in the future."
        )

    if (
        hasattr(config.model, "use_albumentations_transforms")
        and not config.model.use_albumentations_transforms
    ):
        warnings.warn(
            "use_albumentations_transforms is not True. This will be deprecated in the future."
        )

    if (
        hasattr(config.model, "image_crop_size")
        and hasattr(config.model, "image_target_size")
        and (config.model.image_crop_size is not None or config.model.image_target_size is not None)
    ):
        assert (
            config.model.image_crop_size is not None and config.model.image_target_size is not None
        ), "image_crop_size and image_target_size must be set together"
        warnings.warn(
            "image_crop_size and image_target_size will be deprecated in the future. Please use shortest_image_edge and crop_fraction instead."
        )
        if hasattr(config.model, "shortest_image_edge") and hasattr(config.model, "crop_fraction"):
            assert (
                config.model.shortest_image_edge is None and config.model.crop_fraction is None
            ), (
                "Do not set shortest_image_edge and crop_fraction together with image_crop_size and image_target_size"
            )

    if (
        hasattr(config.model, "shortest_image_edge")
        and hasattr(config.model, "crop_fraction")
        and (config.model.shortest_image_edge is not None or config.model.crop_fraction is not None)
    ):
        assert config.model.use_albumentations_transforms, (
            "use_albumentations_transforms must be True when shortest_image_edge and crop_fraction are set"
        )


def run(config: Config):
    warn_configs(config)

    """Main training function."""
    # If using distributed training, initialize the process group
    if dist.is_initialized():
        global_rank = dist.get_rank()
    elif "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        # only meaningful for torchrun, for ray it is always 0
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        global_rank = dist.get_rank()
    else:
        local_rank = 0
        global_rank = 0

    # Setup
    setup_logging()
    set_seed(config.data.seed)

    # Validate config
    config.validate()

    # Create output directory
    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        experiment_name = output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / "processor"
    config.save(save_cfg_dir / "config.yaml")
    omegaconf_config = OmegaConf.create(config.__dict__)
    omegaconf_config["max_steps"] = config.training.max_steps
    omegaconf_config["save_steps"] = config.training.save_steps
    OmegaConf.save(omegaconf_config, save_cfg_dir / "conf.yaml", resolve=True)
    wandb_config_file = output_dir / "wandb_config.json"
    with open(wandb_config_file, "w") as f:
        json.dump(
            {
                "project": config.training.wandb_project,
                "run_id": experiment_name,
            },
            f,
        )

    logging.info(f"Saved config to {save_cfg_dir}")

    # Initialize wandb if configured, but only on the main process
    if config.training.use_wandb and global_rank == 0:
        # Add git commit hash and version info to config
        config_dict = {
            **config.__dict__,
            "git_commit_hash": os.environ.get("GROOT_COMMIT_HASH", "unknown"),
        }

        wandb.init(
            project=config.training.wandb_project,
            name=experiment_name,
            config=config_dict,
            tags=[config.data.mode],
        )

    # Setup model training pipeline.
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)
    pipeline.setup()
    model = pipeline.return_model()
    train_dataset, eval_dataset = pipeline.return_dataset()
    data_collator = pipeline.return_collator()
    processor = pipeline.return_processor()
    processor.save_pretrained(processor_dir)
    online_evaluator = None

    # deepspeed config
    if config.training.num_gpus > 1 and not config.training.use_ddp:
        deepspeed_config = config.get_deepspeed_config()
    else:
        deepspeed_config = None

    # for now we will let batch_size override global_batch_size, in future we will deprecate batch_size
    if config.training.batch_size is None:
        per_device_train_batch_size = config.training.global_batch_size // config.training.num_gpus
    else:
        per_device_train_batch_size = config.training.batch_size

    effective_max_steps = config.training.max_steps
    effective_save_steps = config.training.save_steps
    epoch_step_boundaries = None
    save_strategy = "steps"
    if config.training.epoch_based_training:
        if not hasattr(train_dataset, "get_num_samples_for_epoch"):
            raise ValueError("epoch_based_training requires a dataset with get_num_samples_for_epoch()")
        if config.training.num_epochs <= 0:
            raise ValueError("num_epochs must be greater than 0 when epoch_based_training is enabled")
        if config.training.save_epochs <= 0:
            raise ValueError("save_epochs must be greater than 0 when epoch_based_training is enabled")
        epoch_step_counts = []
        for epoch_idx in range(config.training.num_epochs):
            num_samples_for_epoch = int(
                train_dataset.get_num_samples_for_epoch(
                    epoch=epoch_idx,
                    num_workers=config.training.dataloader_num_workers,
                )
            )
            if num_samples_for_epoch <= 0:
                raise ValueError(
                    f"epoch_based_training resolved zero samples for epoch {epoch_idx + 1}"
                )
            micro_batches_for_epoch = (
                num_samples_for_epoch + per_device_train_batch_size - 1
            ) // per_device_train_batch_size
            optimizer_steps_for_epoch = max(
                1,
                (micro_batches_for_epoch + config.training.gradient_accumulation_steps - 1)
                // config.training.gradient_accumulation_steps,
            )
            epoch_step_counts.append(optimizer_steps_for_epoch)
        epoch_step_boundaries = []
        cumulative_steps = 0
        for step_count in epoch_step_counts:
            cumulative_steps += int(step_count)
            epoch_step_boundaries.append(int(cumulative_steps))
        effective_max_steps = epoch_step_boundaries[-1]
        effective_save_steps = effective_max_steps + 1
        logging.info(
            "Epoch-based training enabled: "
            f"{config.training.num_epochs} epochs, "
            f"step boundaries={epoch_step_boundaries}"
        )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=effective_max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=effective_save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to="wandb" if config.training.use_wandb else "none",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=config.training.ddp_bucket_cap_mb,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        batch_eval_metrics=True,
        remove_unused_columns=config.training.remove_unused_columns,
        ignore_data_skip=True,
    )

    # Create trainer
    trainer = Gr00tTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        multiprocessing_context=config.data.multiprocessing_context,
        epoch_based_training=config.training.epoch_based_training,
        epoch_step_boundaries=epoch_step_boundaries,
    )

    trainer.add_callback(
        CheckpointFormatCallback(
            run_name=experiment_name,
            exp_cfg_dir=save_cfg_dir,
            processor_dir=processor_dir,
        )
    )

    if config.training.epoch_based_training and epoch_step_boundaries is not None:
        save_step_points = [
            epoch_step_boundaries[idx]
            for idx in range(
                config.training.save_epochs - 1,
                len(epoch_step_boundaries),
                config.training.save_epochs,
            )
        ]
        trainer.add_callback(EpochBoundarySaveCallback(save_step_points))

    if config.training.enable_online_eval:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.experiment.online_eval import (
            Gr00tOnlineEvaluator,
            OnlineEvalCallback,
            OnlineEvalConfig,
        )

        dataset_cfg = config.data.datasets[0]
        online_eval_step_points = None
        if config.training.epoch_based_training and epoch_step_boundaries is not None:
            online_eval_step_points = [
                epoch_step_boundaries[idx]
                for idx in range(
                    config.training.online_eval_epochs - 1,
                    len(epoch_step_boundaries),
                    config.training.online_eval_epochs,
                )
            ]
        online_evaluator = Gr00tOnlineEvaluator(
            OnlineEvalConfig(
                env_id=config.training.online_eval_env_id,
                embodiment_tag=EmbodimentTag(dataset_cfg.embodiment_tag),
                language_source=dataset_cfg.language_source,
                task_mapping_path=dataset_cfg.task_mapping_path,
                human_desc_path=dataset_cfg.human_desc_path,
                sim_desc_path=dataset_cfg.sim_desc_path,
                num_episodes=config.training.online_eval_num_episodes,
                num_envs=config.training.online_eval_num_envs,
                max_episode_steps=config.training.online_eval_max_episode_steps,
                sim_backend=config.training.online_eval_sim_backend,
                control_mode=config.training.online_eval_control_mode,
                obs_mode=config.training.online_eval_obs_mode,
                reward_mode=config.training.online_eval_reward_mode,
                shader=config.training.online_eval_shader,
                capture_video=config.training.online_eval_capture_video,
                output_dir=str(output_dir),
            ),
            processor=processor,
        )
        trainer.add_callback(
            OnlineEvalCallback(
                evaluator=online_evaluator,
                eval_steps=config.training.online_eval_steps,
                eval_step_points=online_eval_step_points,
            )
        )

    if config.training.save_best_eval_metric_name != "":
        trainer.add_callback(
            BestMetricCheckpointCallback(
                metric_name=config.training.save_best_eval_metric_name,
                greater_is_better=config.training.save_best_eval_metric_greater_is_better,
                exp_cfg_dir=save_cfg_dir,
            )
        )

    if hasattr(train_dataset, "get_initial_actions"):
        initial_actions = train_dataset.get_initial_actions()
        if initial_actions:
            initial_actions_path = save_cfg_dir / INITIAL_ACTIONS_FILENAME
            save_initial_actions(initial_actions, initial_actions_path)
            logging.info(f"Saved {len(initial_actions)} initial actions to {initial_actions_path}")

    # Train
    logging.info("🚀 Starting training...")
    resume_from_checkpoint = config.training.resume_from_checkpoint
    if resume_from_checkpoint is None:
        logging.info("Starting a fresh training run (resume_from_checkpoint=None)")
    else:
        logging.info(f"Resuming training from checkpoint {resume_from_checkpoint}")

    if config.training.enable_profiling:
        from functools import partial

        logging.info(f"{global_rank} Starting training with profiling...")

        def on_trace_ready_handler(trainer, profile_dir, prof):
            output_path = (
                profile_dir / f"trace_rank_{global_rank}_iter_{trainer.state.global_step}.json"
            )
            prof.export_chrome_trace(str(output_path))
            logging.info(f"Trace saved to {output_path}")

        profile_dir = output_dir / "profiling"
        profile_dir.mkdir(parents=True, exist_ok=True)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(skip_first=10, wait=1, warmup=1, active=3, repeat=1),
            # profile_memory=True,
            with_stack=True,
            # record_shapes=True,
            on_trace_ready=partial(on_trace_ready_handler, trainer, profile_dir),
        ) as prof:
            trainer.add_callback(ProfCallback(prof=prof))
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    trainer.save_model()
    logging.info(f"Model saved to {output_dir}")

    if config.training.assert_loss_less_than is not None:
        final_loss = trainer.loss
        if final_loss.item() > config.training.assert_loss_less_than:
            raise AssertionError(
                f"Loss too high: {final_loss.item()} vs {config.training.assert_loss_less_than})"
            )

    # # Cleanup
    if hasattr(train_dataset, "close"):
        train_dataset.close()
    if eval_dataset is not None and hasattr(eval_dataset, "close"):
        eval_dataset.close()
    if online_evaluator is not None:
        online_evaluator.close()
    logging.info("Training completed!")
