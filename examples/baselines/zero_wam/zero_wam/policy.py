# Native tensor/cache adapters use private APIs; runtime/simulator imports are lazy.
# ruff: noqa: PLC0415, SLF001, FBT001, FBT002, FBT003
"""Incremental Zero-WAM inference with the same adapters as native training."""

import hashlib
from pathlib import Path

import torch
from torch import nn

from .config import ZeroWAMConfig
from .lora import adapt_transformer
from .model import pack_actions
from .model import unpack_actions
from .runtime import atomic_torch_save


class ZeroWAMAgent(nn.Module):
    def __init__(self, config: ZeroWAMConfig, *, vae_only: bool = False, icl_guidance_scale: float = 1.0):
        super().__init__()
        from diffusers import AutoencoderKLWan
        from torch._dynamo import config as compiler_config
        from torch.nn.attention.flex_attention import flex_attention

        from .model import eager_block_mask
        from .wan_va.modules import icl_model
        from .wan_va.modules.icl_model import WanICLTransformer3DModel
        from .wan_va.modules.utils import WanVAEStreamingWrapper
        from .wan_va.utils.scheduler import FlowMatchScheduler
        from .wan_va.utils.utils import data_seq_to_patch
        from .wan_va.utils.utils import get_mesh_id

        # Streaming KV lengths grow between queries. Static specializations can
        # exhaust Dynamo's cache and silently fall back to unfused attention.
        compiler_config.recompile_limit = max(compiler_config.recompile_limit, 256)
        icl_model._compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
        icl_model._compiled_create_block_mask = eager_block_mask
        self.args = config
        from .action_space import EpisodeActionSpace, channel_ids
        import json
        self.action_channel_ids = channel_ids(config.action_channel_offset)
        self.action_space = (EpisodeActionSpace(json.loads(Path(config.normalization_path).read_text()))
                             if config.normalization_path else None)
        self.icl_guidance_scale = float(icl_guidance_scale)
        self.use_icl_cfg = self.icl_guidance_scale > 1.0
        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16
        self.pred_horizon = config.pred_horizon
        self.action_dim = config.action_dim
        self.obs_horizon = 1
        self.mesh = get_mesh_id
        self.to_video = data_seq_to_patch
        self.cache_name = "pos"
        self.cache_root = Path(config.cache_dir)
        self.model_path = Path(config.model_path).resolve()
        from .cache import digest_json
        from .cache import encoder_identity

        self.base_identity = digest_json(encoder_identity(self.model_path))
        self.cache_version = f"ig_zero_wam_v2_{self.base_identity[:16]}_{config.human_height}x{config.human_width}"
        self.human_dataset = None
        self.empty_text = self._empty_condition()
        print("Loading frozen Wan VAE", flush=True)
        self.vae = (
            AutoencoderKLWan.from_pretrained(self.model_path / "vae", torch_dtype=self.dtype, local_files_only=True)
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        if getattr(self.vae.config, "scale_factor_temporal", None) != 4:
            raise ValueError("This checkpoint requires the Wan 2.2 VAE (diffusers >=0.35)")
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        if vae_only:
            return
        print("Loading pretrained Zero-WAM transformer", flush=True)
        self.transformer = WanICLTransformer3DModel.from_pretrained(
            self.model_path / "transformer", torch_dtype=self.dtype, local_files_only=True
        )
        self.transformer.disable_mcp_modules()
        # Auxiliary branches are training-only and have no inference path.
        for name in ("mcp_blocks", "mcp_mlp_hidden", "mcp_projections"):
            if hasattr(self.transformer, name):
                delattr(self.transformer, name)
        self.transformer.to(device=self.device, dtype=self.dtype)
        self.adaptation_report = adapt_transformer(self.transformer, config)
        self.transformer.action_embedder.tokens_per_chunk = config.pred_horizon
        self.transformer.ig_freeze_history_action = False
        # Each diagnostic lane uses the released single-sequence streaming path.
        print(
            f"Trainable parameters: {self.adaptation_report['trainable_parameters']:,}; LoRA modules: {len(self.adaptation_report['lora_targets'])}; conditioning=video_only; scope={config.train_scope}; checkpoint=False",
            flush=True,
        )
        self.video_scheduler = FlowMatchScheduler(shift=config.video_snr_shift, sigma_min=0, extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(shift=config.action_snr_shift, sigma_min=0, extra_one_step=True)
        self.video_scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)
        self.clear_cache()

    @torch.no_grad()
    def _empty_condition(self) -> torch.Tensor:
        path = Path(self.args.empty_text_embedding)
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value.shape != (512, 4096):
            raise ValueError(f"Official empty text shape mismatch: {value.shape}")
        self.empty_condition_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"Official TARGET_TEXT_CFG=-1; empty condition={path}; T5 is not loaded", flush=True)
        return value[None].to(device=self.device, dtype=self.dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    @staticmethod
    def _video(video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 4:
            video = video.unsqueeze(1)
        if video.ndim != 5:
            raise ValueError(f"Expected a batched RGB video, got {video.shape}")
        if video.shape[-1] == 3:
            video = video.permute(0, 4, 1, 2, 3)
        elif video.shape[2] == 3:
            video = video.permute(0, 2, 1, 3, 4)
        else:
            raise ValueError(f"Invalid RGB channels: {video.shape}")
        video = video.float()
        if video.max() > 1.01 or video.min() < 0:
            raise ValueError("Expected IG RGB in [0,1], not ImageNet normalization or uint8")
        return video.contiguous()

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor, persistent: bool = True) -> torch.Tensor:
        video = self._video(video)
        if persistent and video.shape[0] != 1:
            raise ValueError("Persistent per-video caching expects one video")
        path = None
        if persistent:
            digest = hashlib.sha256(self.cache_version.encode())
            digest.update(str(tuple(video.shape)).encode())
            digest.update(video.detach().to("cpu", torch.float32).numpy().tobytes())
            key = digest.hexdigest()
            path = self.cache_root / "vae" / key[:2] / f"{key}.pt"
            if path.is_file():
                return torch.load(path, map_location=self.device, weights_only=True).to(self.dtype)
        pixels = self._pixels(video, human=True)
        self.streaming_vae.clear_cache()
        complete_frames = 1 + 4 * ((pixels.shape[2] - 1) // 4)
        chunks = [pixels[:, :, :1], *pixels[:, :, 1:complete_frames].split(4, dim=2)]
        moments = [self.streaming_vae.encode_chunk(chunk) for chunk in chunks if chunk.shape[2]]
        mu = torch.cat(moments, dim=2).chunk(2, dim=1)[0]
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device)[None, :, None, None, None]
        std = torch.tensor(self.vae.config.latents_std, device=self.device)[None, :, None, None, None]
        latent = ((mu.float() - mean) / std).to(self.dtype).detach()
        self.streaming_vae.clear_cache()
        if persistent:
            atomic_torch_save(latent.cpu(), path)
        return latent

    def _pixels(self, video, *, human=False):
        from torch.nn import functional as F
        video = video if video.ndim == 5 and video.shape[1] == 3 and video.shape[-1] != 3 else self._video(video)
        b, c, t, h, w = video.shape
        size = ((self.args.human_height, self.args.human_width) if human else
                (self.args.image_size, self.args.robot_width))
        frames = video.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w)
        frames = F.interpolate(frames, size=size, mode="bilinear", align_corners=False)
        video = frames.reshape(b, t, c, *size).permute(0, 2, 1, 3, 4)
        return (video * 2 - 1).to(device=self.device, dtype=self.dtype)

    def _state(self, obs: dict) -> torch.Tensor:
        state = obs.get("states", obs.get("state"))
        if state is None:
            raise ValueError("Currently observed state is required")
        if state.ndim == 3:
            state = state[:, -1]
        if state.ndim != 2 or state.shape[1] != self.args.state_dim:
            raise ValueError(f"Expected current state [B,18], got {state.shape}")
        state = state.to(device=self.device, dtype=torch.float32)
        if self.action_space:
            if self.episode_origin is None:
                self.episode_origin = state.detach().clone()
            state = self.action_space.encode_state(state, self.episode_origin)
        return state

    def _stream(
        self,
        data: torch.Tensor,
        timestep: float,
        action: bool,
        cache_type: int,
        frame_id: int,
        icl: bool = False,
        frame_shift: int | None = None,
    ) -> dict:
        patch = (1, 1, 1) if action else self.transformer.patch_size
        grid = self.mesh(
            data.shape[2],
            data.shape[-2] // patch[1],
            data.shape[-1] // patch[2],
            int(action),
            f_shift=self.history_frame_offset if frame_shift is None else frame_shift,
            h_shift=self.args.icl_rope_h if icl else 0,
            action=False,
        ).to(self.device)
        count = grid.shape[1]
        key = "action" if action else "latent"
        inputs = {
            f"{key}_res_lst": {
                "noisy_latents": data,
                "timesteps": torch.full([data.shape[2]], float(timestep), device=self.device),
                "cache_type_ids": torch.full([count], cache_type, device=self.device).int(),
            },
            f"{key}_grid_id": grid,
            "current_seq_ids": torch.zeros(count, device=self.device).int(),
            "current_frame_ids": torch.full([count], frame_id, device=self.device).int(),
            "text_emb": self.empty_text.expand(data.shape[0], -1, -1),
            "encoder_seq_ids": torch.zeros(512, device=self.device).int(),
        }
        if not action and not icl and self.use_icl_cfg:
            self._pack_icl_cfg(inputs)
        return inputs

    def _pack_icl_cfg(self, inputs):
        """Native sequence0/1 video CFG; only sequence0 owns human/action KV."""
        stream = inputs["latent_res_lst"]
        tokens = inputs["latent_grid_id"].shape[1]
        stream["noisy_latents"] = stream["noisy_latents"].repeat(1, 1, 2, 1, 1)
        stream["timesteps"] = stream["timesteps"].repeat(2)
        stream["cache_type_ids"] = stream["cache_type_ids"].repeat(2)
        inputs["latent_grid_id"] = inputs["latent_grid_id"].repeat(1, 2)
        inputs["current_seq_ids"] = torch.cat([
            torch.zeros(tokens, device=self.device, dtype=torch.int),
            torch.ones(tokens, device=self.device, dtype=torch.int),
        ])
        inputs["current_frame_ids"] = inputs["current_frame_ids"].repeat(2)
        inputs["text_emb"] = torch.cat([self.empty_text, self.empty_text], dim=1)
        text_tokens = self.empty_text.shape[1]
        inputs["encoder_seq_ids"] = torch.cat([
            torch.zeros(text_tokens, device=self.device, dtype=torch.int),
            torch.ones(text_tokens, device=self.device, dtype=torch.int),
        ])

    def _combine_icl_cfg(self, output):
        if not self.use_icl_cfg:
            return output
        conditioned, unconditioned = output.chunk(2, dim=1)
        return unconditioned + self.icl_guidance_scale * (conditioned - unconditioned)

    @torch.no_grad()
    def _context_forward(self, inputs, **kwargs):
        """Streaming human/video/history writes use the same frozen expert."""
        return self.transformer(inputs, **kwargs)

    @torch.no_grad()
    def prepare_for_eval(self, human_video, robot_obs, human_desc=None, human_vl_ids=None):
        if human_desc is not None:
            raise ValueError("Use input_mode=video_only; no human/task description is allowed")
        del human_vl_ids
        self.eval()
        self.clear_cache()
        self.transformer.action_embedder.current_state = self._state(robot_obs)
        if human_video is None:
            raise ValueError("Human video is required for Zero-WAM")
        latent = (
            self.preview_human_latent.to(device=self.device, dtype=self.dtype)
            if hasattr(self, "preview_human_latent")
            else torch.cat([self.encode_video(v[None]) for v in human_video], dim=0)
        )
        self._context_forward(
            self._stream(latent, 0, False, 2, 0, icl=True), update_cache=1, mode="forward_latent_only"
        )
        if getattr(self, "preview_diagnostics", None) is not None:
            self.preview_diagnostics.begin(robot_obs)
        self.prepared = True

    @torch.no_grad()
    def _encode_stream_observation(self, video: torch.Tensor, *, initial: bool = False) -> torch.Tensor:
        pixels = self._pixels(video)
        if initial:
            self.streaming_vae.clear_cache()
            chunks = [pixels]
        else:
            if pixels.shape[2] % 4:
                raise ValueError("Executed video must contain complete four-frame VAE groups")
            chunks = pixels.split(4, dim=2)
        mu = torch.cat([self.streaming_vae.encode_chunk(x).chunk(2, dim=1)[0] for x in chunks], dim=2)
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device)[None, :, None, None, None]
        std = torch.tensor(self.vae.config.latents_std, device=self.device)[None, :, None, None, None]
        latent = ((mu.float() - mean) / std).to(self.dtype)
        if getattr(self, "preview_diagnostics", None) is not None:
            self.preview_diagnostics.history_latent(latent, initial, self.chunk_idx)
        return latent

    @torch.no_grad()
    def record_executed_action(self, action: torch.Tensor) -> None:
        cached = action.detach().float().clone()
        if self.history_gripper_affine is None:
            # Use the evaluator's exact inverse, including its quantile epsilon.
            low = self.history_action_denormalizer(-torch.ones_like(cached))[..., [7, 15]]
            high = self.history_action_denormalizer(torch.ones_like(cached))[..., [7, 15]]
            self.history_gripper_affine = (low, (high - low) / 2)
        low, scale = self.history_gripper_affine
        grippers = cached[..., [7, 15]]
        raw = (grippers + 1) * scale + low
        limited = raw.clamp(-1, 1)
        changed = raw != limited
        # Do not reapply training's normalized-value clamp: controller bounds
        # can legitimately map outside it. Preserve all unchanged values exactly.
        corrected = (limited - low) / scale - 1
        cached[..., [7, 15]] = torch.where(changed, corrected, grippers)
        self.gripper_history_corrections += changed.sum(dim=0)
        self.gripper_history_max_correction = torch.maximum(
            self.gripper_history_max_correction, (raw - limited).abs().amax(dim=0)
        )
        self.executed_actions.append(cached)
        self.total_executed_steps += 1

    @torch.no_grad()
    def observe_step(self, obs: dict) -> None:
        if self.executed_actions and len(self.executed_actions) % self.args.robot_frame_stride == 0:
            rgb = obs.get("rgb", obs.get("view_1"))
            self.observed_frames.append((rgb[:, -1:] if rgb.ndim == 5 else rgb[:, None]).detach().clone())

    @torch.no_grad()
    def _commit_history(self) -> None:
        expected = self.pred_horizon - self.args.action_per_frame if self.chunk_idx == 0 else self.pred_horizon
        if self.execution_horizon is not None:
            expected = min(expected, self.execution_horizon)
        if (
            len(self.executed_actions) != expected
            or len(self.observed_frames) != expected // self.args.robot_frame_stride
        ):
            raise ValueError(
                f"Incomplete executed history: actions={len(self.executed_actions)}, frames={len(self.observed_frames)}"
            )
        self.transformer.clear_prediction_cache(-1)
        observed = self._encode_stream_observation(torch.cat(self.observed_frames, dim=1))
        actions = pack_actions(torch.stack(self.executed_actions, dim=1), self.args.action_channel_offset).to(self.dtype)
        if self.chunk_idx == 0:
            observed = torch.cat([self.init_latent, observed], dim=2)
            actions = torch.cat([actions.new_zeros(actions.shape[0], self.args.action_per_frame, 30), actions], dim=1)
        committed_frames = actions.shape[1] // self.args.action_per_frame
        if observed.shape[2] != committed_frames:
            raise ValueError("Executed action/video history lengths differ")
        actions = actions.reshape(actions.shape[0], committed_frames, self.args.action_per_frame, 30)
        actions = actions.permute(0, 3, 1, 2).unsqueeze(-1)
        self._context_forward(
            self._stream(observed, 0, False, 0, 2 * self.chunk_idx), update_cache=1, mode="forward_latent_only"
        )
        self.transformer.action_embedder.current_state = self.last_state
        self._context_forward(
            self._stream(actions, 0, True, 0, 2 * self.chunk_idx + 1),
            update_cache=1,
            mode="forward_action_only",
            clean_window_cache=True,
        )
        self.history_frame_offset += committed_frames
        if getattr(self, "evaluation_progress_path", None) is not None:
            from .runtime import atomic_json_save

            atomic_json_save(
                {
                    "committed_queries": self.chunk_idx + 1,
                    "executed_steps": self.total_executed_steps,
                    "history_latent_frames": self.history_frame_offset,
                    "last_commit_actions": expected,
                    "last_commit_latent_frames": committed_frames,
                    "generated_cache_entries": self.transformer.cache_counts().get(1, 0),
                },
                self.evaluation_progress_path,
            )
        self.chunk_idx += 1
        self.executed_actions.clear()
        self.observed_frames.clear()

    @torch.no_grad()
    @torch.autocast("cuda", dtype=torch.bfloat16)
    def get_action(self, obs: dict) -> torch.Tensor:
        if not self.prepared:
            raise RuntimeError("Call prepare_for_eval first")
        from .wan_va.utils.scheduler import FlowMatchScheduler

        if self.last_state is not None:
            self._commit_history()
        current_state = self._state(obs)
        self.transformer.action_embedder.current_state = current_state
        if self.chunk_idx == 0:
            rgb = obs.get("rgb", obs.get("view_1"))
            self.init_latent = self._encode_stream_observation(
                rgb[:, -1:] if rgb.ndim == 5 else rgb[:, None], initial=True
            )
        shape = (current_state.shape[0], 48, self.args.frame_chunk_size, *self.init_latent.shape[-2:])
        latents = torch.randn(shape, device=self.device, dtype=self.dtype)
        scheduler = FlowMatchScheduler(shift=self.args.video_snr_shift, sigma_min=0, extra_one_step=True)
        scheduler.set_timesteps(self.args.video_inference_steps)
        for step, timestep in enumerate([*scheduler.timesteps, 0]):
            last = step == self.args.video_inference_steps
            if self.chunk_idx == 0:
                latents[:, :, :1] = self.init_latent
            inputs = self._stream(latents, timestep, False, 1, 2 * self.chunk_idx)
            if self.chunk_idx == 0:
                inputs["latent_res_lst"]["timesteps"][:: self.args.frame_chunk_size] = 0
            output = self._context_forward(inputs, update_cache=int(last), mode="forward_latent_only")
            if not last:
                velocity = self.to_video(
                    self.transformer.patch_size, self._combine_icl_cfg(output), *latents.shape[-3:], batch_size=current_state.shape[0]
                )
                latents = scheduler.step(velocity, timestep, latents)
        if getattr(self, "preview_diagnostics", None) is not None:
            self.preview_diagnostics.prediction(latents, self.chunk_idx)
        actions = torch.randn(
            current_state.shape[0],
            30,
            self.args.frame_chunk_size,
            self.args.action_per_frame,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        scheduler = FlowMatchScheduler(shift=self.args.action_snr_shift, sigma_min=0, extra_one_step=True)
        scheduler.set_timesteps(self.args.action_inference_steps)
        for timestep in scheduler.timesteps:
            actions[
                :,
                [
                    i
                    for i in range(30)
                    if i
                    not in getattr(
                        self, "action_channel_ids", (14, 15, 16, 17, 18, 19, 20, 28, 21, 22, 23, 24, 25, 26, 27, 29)
                    )
                ],
            ] = 0
            output = self.transformer(
                self._stream(actions, timestep, True, 1, 2 * self.chunk_idx + 1), mode="forward_action_only"
            )
            velocity = output.reshape(
                current_state.shape[0], self.args.frame_chunk_size, self.args.action_per_frame, 30
            )
            velocity = velocity.permute(0, 3, 1, 2).unsqueeze(-1)
            actions = scheduler.step(velocity, timestep, actions)
        actions[
            :,
            [
                i
                for i in range(30)
                if i
                not in getattr(
                    self, "action_channel_ids", (14, 15, 16, 17, 18, 19, 20, 28, 21, 22, 23, 24, 25, 26, 27, 29)
                )
            ],
        ] = 0
        result = unpack_actions(
            actions.permute(0, 2, 3, 4, 1).reshape(current_state.shape[0], self.pred_horizon, 30),
            self.args.action_channel_offset,
        ).float()
        self.last_state = current_state.detach()
        result = result[:, self.args.action_per_frame :] if self.chunk_idx == 0 else result
        if self.execution_horizon is not None:
            result = result[:, : self.execution_horizon]
        print(
            f"[QUERY] index={self.chunk_idx} executed_so_far={self.total_executed_steps} rope_frame_offset={self.history_frame_offset} returned_actions={result.shape[1]}",
            flush=True,
        )
        return result

    def clear_cache(self):
        self.transformer.clear_cache()
        if hasattr(self, "context_runner"):
            self.context_runner.clear()
        self.streaming_vae.clear_cache()
        self.prepared = False
        self.chunk_idx = 0
        self.history_frame_offset = 0
        self.total_executed_steps = 0
        self.execution_horizon = getattr(self, "execution_horizon", None)
        self.init_latent = None
        self.last_state = None
        self.episode_origin = None
        self.executed_actions = []
        self.observed_frames = []
        self.history_gripper_affine = None
        self.gripper_history_corrections = torch.zeros(2, dtype=torch.long, device=self.device)
        self.gripper_history_max_correction = torch.zeros(2, dtype=torch.float32, device=self.device)

    def adapter_state_dict(self) -> dict:
        return {
            name: param.detach().cpu().clone()
            for name, param in self.transformer.named_parameters()
            if param.requires_grad
        }

    def load_adapter(self, state: dict) -> None:
        expected = {name for name, param in self.transformer.named_parameters() if param.requires_grad}
        if set(state) != expected:
            raise ValueError(f"Adapter keys mismatch; missing={expected - set(state)}, extra={set(state) - expected}")
        self.transformer.load_state_dict(state, strict=False)
