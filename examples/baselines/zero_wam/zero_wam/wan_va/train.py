# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# IG modifications: portable data/LoRA/checkpoint hooks; native objectives retained.
from pathlib import Path
import gc
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from .. import training as ig_bridge
from .distributed.fsdp import shard_model, apply_ac
from .distributed.util import _configure_model, dist_mean, dist_max
from .modules.utils import load_transformer, resolve_model_component
from .utils import logger, get_mesh_id, sample_timestep_id, data_seq_to_patch, warmup_constant_lambda, FlowMatchScheduler
from .mcp import validate_mcp_settings, shift_latents_for_mcp

def _safe_mean(tensor):
    return tensor.sum() / max(tensor.numel(), 1)

class Trainer:
    @staticmethod
    def _resolve_transformer_path(path):
        transformer_path = Path(resolve_model_component(path, 'transformer'))
        if not (transformer_path / 'config.json').is_file():
            raise FileNotFoundError(
                f"Expected a Transformer config at {transformer_path}/config.json"
            )
        return str(transformer_path)

    def __init__(self, config):
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.enable_mcp = getattr(config, 'enable_mcp', True)
        if int(config.batch_size) < 1 or (config.ig_sampling_mode == "episode" and int(config.batch_size) != 1):
            raise ValueError(
                "Zero-WAM ICL training requires batch_size=1 per rank"
            )
        if self.enable_mcp:
            validate_mcp_settings(
                num_mcp_modules=config.num_mcp_modules,
                mcp_blocks_per_group=config.mcp_blocks_per_group,
                mcp_hidden_collect_layers=config.mcp_hidden_collect_layers,
                mcp_loss_weights=config.mcp_loss_weights,
            )

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = self._resolve_transformer_path(config.resume_from)
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = self._resolve_transformer_path(config.model_path)

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            disable_mcp=not self.enable_mcp,
            icl_model=True,
            enable_mcp=self.enable_mcp,
            num_mcp_modules=config.num_mcp_modules,
            mcp_blocks_per_group=config.mcp_blocks_per_group,
            mcp_hidden_collect_layers=tuple(config.mcp_hidden_collect_layers),
        )
        if self.enable_mcp:
            validate_mcp_settings(
                num_mcp_modules=config.num_mcp_modules,
                mcp_blocks_per_group=config.mcp_blocks_per_group,
                mcp_hidden_collect_layers=config.mcp_hidden_collect_layers,
                mcp_loss_weights=config.mcp_loss_weights,
                num_layers=len(self.transformer.blocks),
            )
            logger.info(
                "MCP architecture: %d modules x %d block, hidden layers=%s",
                config.num_mcp_modules,
                config.mcp_blocks_per_group,
                list(config.mcp_hidden_collect_layers),
            )

        logger.info("Setting up activation checkpointing ...")
        ig_bridge.adapt_transformer(self.transformer, config)
        if config.get("ig_activation_checkpointing", False):
            apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        # LoRA and IG heads were selected before native FSDP sharding.

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = ig_bridge.build_lr_scheduler(self.optimizer, config)

        ig_bridge.setup_data(self)

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)
        self.train_scheduler_mcp = None
        if self.enable_mcp:
            self.train_scheduler_mcp = FlowMatchScheduler(
                shift=self.config.mcp_snr_shift,
                sigma_min=0.0,
                extra_one_step=True,
            )
            self.train_scheduler_mcp.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        ig_bridge.resume(self)
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)

    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)

        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)

        return batch

    @torch.no_grad()
    def _add_noise(
        self,
        latent,
        train_scheduler,
        action_mask=None,
        action_mode=False,
        noisy_cond_prob=0.0,
        noisy_cond_min_timestep_bd=0.5,
        noisy_cond_max_timestep_bd=1.0,
        frame_shift=0,
    ):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=frame_shift,
            action=False,
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                batch_size=F,
                min_timestep_bd=noisy_cond_min_timestep_bd,
                max_timestep_bd=noisy_cond_max_timestep_bd,
                num_train_timesteps=train_scheduler.num_train_timesteps,
            )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Build the single-sample VA training streams and ICL condition."""
        configured_chunk = int(getattr(self.config, 'frame_chunk_size', 0))
        if configured_chunk > 0:
            chunk_size = configured_chunk
        else:
            chunk_size = torch.randint(
                1, int(self.config.max_frame_chunk_size) + 1, (1,)
            ).item()
        latent_dict = self._add_noise(
            latent=batch_dict['latents'],
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            noisy_cond_prob=float(self.config.noisy_img_prob),
            noisy_cond_min_timestep_bd=float(
                self.config.noisy_cond_min_timestep_bd),
            noisy_cond_max_timestep_bd=float(
                self.config.noisy_cond_max_timestep_bd),
        )

        action_dict = self._add_noise(
            latent=batch_dict['actions'],
            train_scheduler=self.train_scheduler_action,
            action_mask=None,
            action_mode=True,
            noisy_cond_prob=0.0)

        action_dict['actions_mask'] = batch_dict['actions_mask']

        use_icl = torch.rand(1).item() >= float(self.config.drop_icl)
        icl_latent_dict = None
        target_text_emb = batch_dict['text_emb']
        target_text_length = target_text_emb.shape[1]
        text_parts = [target_text_emb]
        encoder_seq_parts = [
            torch.zeros(target_text_length, device=self.device, dtype=torch.int)
        ]
        if use_icl:
            icl_latents = batch_dict['icl_latents']
            batch_size, _, frames, height, width = icl_latents.shape
            icl_grid_id = get_mesh_id(
                frames // self.patch_size[0],
                height // self.patch_size[1],
                width // self.patch_size[2],
                t=0,
                h_shift=int(self.config.icl_rope_h),
            ).to(self.device)
            icl_latent_dict = {
                'latent': icl_latents,
                'timesteps': torch.zeros(
                    batch_size, frames, device=self.device, dtype=torch.float32
                ),
                'grid_id': icl_grid_id[None].repeat(batch_size, 1, 1),
            }
            icl_text_emb = batch_dict['icl_text_emb']
            text_parts.append(icl_text_emb)
            encoder_seq_parts.append(
                torch.ones(
                    icl_text_emb.shape[1], device=self.device, dtype=torch.int
                )
            )

        max_window = int(getattr(self.config, 'max_attn_window', 64))
        configured_window = int(getattr(self.config, 'attn_window', 0))
        window_size = (
            configured_window
            if configured_window != 0
            else torch.randint(4, max_window + 1, (1,)).item()
        )
        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'icl_latent_dict': icl_latent_dict,
            'text_emb': torch.cat(text_parts, dim=1),
            'encoder_seq_ids': torch.cat(encoder_seq_parts),
            'chunk_size': chunk_size,
            'max_frame_chunk_size': int(self.config.max_frame_chunk_size),
            'window_size': window_size,
        }
        if self.enable_mcp:
            mcp_latent_dicts = []
            for module_index in range(self.config.num_mcp_modules):
                future_chunk_offset = (
                    1 + module_index * int(self.config.future_chunk_stride)
                )
                frame_shift = future_chunk_offset * chunk_size
                shifted_latents, valid_mask = shift_latents_for_mcp(
                    batch_dict['latents'], frame_shift)
                mcp_latent_dict = self._add_noise(
                    latent=shifted_latents,
                    train_scheduler=self.train_scheduler_mcp,
                    action_mask=None,
                    action_mode=False,
                    noisy_cond_prob=float(self.config.noisy_img_prob),
                    noisy_cond_min_timestep_bd=float(
                        self.config.noisy_cond_min_timestep_bd),
                    noisy_cond_max_timestep_bd=float(
                        self.config.noisy_cond_max_timestep_bd),
                    frame_shift=frame_shift,
                )
                mcp_latent_dict['valid_mask'] = valid_mask
                mcp_latent_dicts.append(mcp_latent_dict)
            input_dict['mcp_latent_dicts'] = mcp_latent_dicts
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            if torch.is_tensor(value):
                input_dict[key] = value.to(self.device)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        if self.enable_mcp:
            latent_pred, action_pred, mcp_pred_list = pred
        else:
            latent_pred, action_pred = pred
            mcp_pred_list = []
        if self.enable_mcp and len(mcp_pred_list) != self.config.num_mcp_modules:
            raise RuntimeError(
                "MCP output count must match num_mcp_modules")
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        batch_size, num_frames = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(
            input_dict['latent_dict']['timesteps'].flatten()
        ).reshape(batch_size, num_frames)
        if not bool(self.config.video_loss_reweight):
            latent_loss_weight = torch.ones_like(latent_loss_weight)

        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict['latent_dict']['targets'].float().detach(),
            reduction='none',
        )
        latent_loss = _safe_mean(
            latent_loss * latent_loss_weight[:, None, :, None, None]
        )

        action_loss_weight = self.train_scheduler_action.training_weight(
            input_dict['action_dict']['timesteps'].flatten()
        ).reshape(batch_size, num_frames)
        if not bool(self.config.action_loss_reweight):
            action_loss_weight = torch.ones_like(action_loss_weight)
        action_loss = F.mse_loss(
            action_pred.float(),
            input_dict['action_dict']['targets'].float().detach(),
            reduction='none',
        )
        action_loss = _safe_mean(
            action_loss
            * action_loss_weight[:, None, :, None, None]
            * input_dict['action_dict']['actions_mask'].float()
        )

        mcp_losses = []
        for mcp_pred, mcp_latent_dict in zip(
                mcp_pred_list, input_dict.get('mcp_latent_dicts', [])):
            mcp_pred = data_seq_to_patch(
                self.patch_size,
                mcp_pred,
                mcp_latent_dict['targets'].shape[-3],
                mcp_latent_dict['targets'].shape[-2],
                mcp_latent_dict['targets'].shape[-1],
                batch_size=mcp_pred.shape[0],
            )
            mcp_batch_size, mcp_num_frames = mcp_latent_dict[
                'timesteps'].shape
            mcp_loss_weight = self.train_scheduler_mcp.training_weight(
                mcp_latent_dict['timesteps'].flatten()).reshape(
                    mcp_batch_size, mcp_num_frames)
            if not bool(self.config.video_loss_reweight):
                mcp_loss_weight = torch.ones_like(mcp_loss_weight)
            mcp_loss = F.mse_loss(
                mcp_pred.float(),
                mcp_latent_dict['targets'].float().detach(),
                reduction='none',
            )
            mcp_loss = mcp_loss * mcp_loss_weight[:, None, :, None, None]
            valid_mask = mcp_latent_dict['valid_mask'].to(
                device=mcp_loss.device, dtype=mcp_loss.dtype)
            valid_count = valid_mask.expand_as(mcp_loss).sum()
            mcp_loss = (mcp_loss * valid_mask).sum() / valid_count.clamp_min(1.)
            mcp_losses.append(mcp_loss / self.gradient_accumulation_steps)

        return (
            latent_loss / self.gradient_accumulation_steps,
            action_loss / self.gradient_accumulation_steps,
            mcp_losses,
        )

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0

        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss, mcp_losses = self.compute_loss(
            input_dict, output)
        mcp_loss = sum(
            weight * depth_loss
            for weight, depth_loss in zip(
                self.config.mcp_loss_weights, mcp_losses)
        ) if mcp_losses else latent_loss.new_zeros(())
        loss = latent_loss + action_loss + mcp_loss

        loss.backward()

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'mcp_losses': [depth_loss.detach() for depth_loss in mcp_losses],
            'mcp_loss': mcp_loss.detach(),
        }

        # Only update weights after accumulating gradients
        if should_sync:
            max_norm = float(self.config.max_norm)
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(), max_norm)
            skip_threshold = (
                float(self.config.skip_step_grad_norm_multiplier) * max_norm
            )
            skip_optimizer_step = (
                not bool(torch.isfinite(total_norm).item())
                or (
                    float(self.config.skip_step_grad_norm_multiplier) > 0
                    and total_norm.item() > skip_threshold
                )
            )
            if not skip_optimizer_step:
                self.optimizer.step()
            elif self.config.rank == 0:
                logger.warning(
                    "Skipping optimizer step: grad_norm=%s threshold=%s",
                    float(total_norm.item()),
                    skip_threshold,
                )
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            losses['total_norm'] = total_norm
            losses['optimizer_step_skipped'] = skip_optimizer_step
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses



    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_mcp_losses = [
            [] for _ in range(self.config.num_mcp_modules)
        ] if self.enable_mcp else []
        accumulated_mcp_total_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()

            losses = self._train_step(batch, step_in_accumulation)

            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            for depth, mcp_loss in enumerate(losses['mcp_losses']):
                accumulated_mcp_losses[depth].append(mcp_loss)
            accumulated_mcp_total_losses.append(losses['mcp_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                mcp_loss_shows = [
                    dist_mean(torch.stack(depth_losses).sum()).detach().cpu().item()
                    for depth_losses in accumulated_mcp_losses
                ]
                mcp_total_loss_show = dist_mean(
                    torch.stack(accumulated_mcp_total_losses).sum()
                ).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_mcp_losses = [
                    [] for _ in range(self.config.num_mcp_modules)
                ] if self.enable_mcp else []
                accumulated_mcp_total_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += 1
                    postfix = {
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}',
                    }
                    if losses['optimizer_step_skipped']:
                        postfix['optimizer_step'] = 'skipped'
                    if self.enable_mcp:
                        postfix['mcp_loss'] = f'{mcp_total_loss_show:.4f}'
                    progress_bar.set_postfix(postfix)
                    if self.config.enable_wandb:
                        log_values = {
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }
                        if self.enable_mcp:
                            log_values['loss_metrics/mcp_weighted_total'] = (
                                mcp_total_loss_show)
                            for depth, mcp_loss_show in enumerate(
                                    mcp_loss_shows):
                                log_values[
                                    f'loss_metrics/mcp_depth_{depth + 1}'] = (
                                        mcp_loss_show)
                        self.wandb.log(log_values, step=self.step)

                self.step += 1
                ig_bridge.log_update(self, dict(video=latent_loss_show, action=action_loss_show,
                    mcp=mcp_total_loss_show, mcp_depths=mcp_loss_shows,
                    lr=lr, grad_norm=float(losses["total_norm"].item()),
                    skipped=losses["optimizer_step_skipped"]))

                if ig_bridge.checkpoint_due(self):
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

                if self.ig_stopping:
                    progress_bar.close()
                    logger.info("Paused after resumable optimizer-boundary checkpoint")
                    return

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")


ig_bridge.install(Trainer)
