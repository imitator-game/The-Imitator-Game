"""
eval_openvla.py - Simplified (no video dependency)
Uses the environment prompt directly as the language input
"""

import argparse
import os
import json
import numpy as np
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, List
from functools import partial

import gymnasium as gym
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from mani_skill.utils import common
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

from examples.baselines.openvla_oft.prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, IGNORE_INDEX
from examples.baselines.openvla_oft.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from examples.baselines.openvla_oft.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from examples.baselines.openvla_oft.prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from examples.baselines.openvla_oft.prismatic.models.action_heads import L1RegressionActionHead, DiffusionActionHead
from examples.baselines.openvla_oft.prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from examples.baselines.openvla_oft.prismatic.models.backbones.llm.prompting import PurePromptBuilder
from examples.baselines.openvla_oft.utils.make_env import (
    SelectCameraObservationWrapper,
    clear_tabletop_task_flags,
    configure_tabletop_task_flags,
    extract_base_env_name,
    extract_level,
    make_eval_envs,
    parse_optional_bool_flag,
)
from examples.baselines.openvla_oft.normalizer import ActionNormalizer as OpenVLAActionNormalizer
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer as LeRobotActionNormalizer
from examples.baselines.lerobot_dataset.evaluate_processor import (
    HumanVideoSimEvaluateProcessor,
    HumanVideoSimEvaluateProcessorConfig,
)

try:
    from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDatasetMetadata
except ImportError:
    LeRobotDatasetMetadata = None


def _summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        print(f"[debug] {name}: type={type(tensor)}")
        return
    try:
        stats_tensor = tensor.detach().flatten()
        if stats_tensor.numel() > 0:
            min_val = float(stats_tensor.min().item())
            max_val = float(stats_tensor.max().item())
            mean_val = float(stats_tensor.mean().item())
        else:
            min_val = max_val = mean_val = float("nan")
    except Exception:
        min_val = max_val = mean_val = float("nan")
    print(
        f"[debug] {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} min={min_val:.6g} max={max_val:.6g} mean={mean_val:.6g}"
    )


def _summarize_array(name: str, array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray):
        print(f"[debug] {name}: type={type(array)}")
        return
    try:
        if array.size > 0:
            min_val = float(array.min())
            max_val = float(array.max())
            mean_val = float(array.mean())
        else:
            min_val = max_val = mean_val = float("nan")
    except Exception:
        min_val = max_val = mean_val = float("nan")
    print(
        f"[debug] {name}: shape={array.shape} dtype={array.dtype} "
        f"min={min_val:.6g} max={max_val:.6g} mean={mean_val:.6g}"
    )


def _sample_tensor_values(tensor: torch.Tensor, num: int = 8) -> list:
    try:
        flat = tensor.detach().flatten()
        if flat.numel() == 0:
            return []
        return flat[:num].cpu().tolist()
    except Exception:
        return []


def _debug_frame_stack(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        return
    if tensor.dim() >= 5:
        num_envs, num_frames = tensor.shape[0], tensor.shape[1]
        print(f"[debug] {name} frame_stack: envs={num_envs} frames={num_frames}")
        _summarize_tensor(f"{name}[0][0]", tensor[0, 0])
        _summarize_tensor(f"{name}[0][-1]", tensor[0, -1])
        sample = _sample_tensor_values(tensor[0, 0], num=6)
        if sample:
            print(f"[debug] {name}[0][0] sample: {sample}")


def _print_debug_obs(obs: Dict[str, Any], info: Dict[str, Any], envs, evaluator) -> None:
    print("[debug] ==== Eval Observation Snapshot ====")
    print(f"[debug] envs.num_envs: {getattr(envs, 'num_envs', 'n/a')}")
    obs_space = getattr(envs, "single_observation_space", None)
    if obs_space is not None:
        print(f"[debug] observation_space: {obs_space}")
    if isinstance(obs, dict):
        print(f"[debug] obs keys: {list(obs.keys())}")
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                _summarize_tensor(f"obs[{key}]", value)
                if key in {"rgb", "depth", "rgbd"}:
                    _debug_frame_stack(f"obs[{key}]", value)
                if key == "state":
                    sample = _sample_tensor_values(value, num=10)
                    if sample:
                        print(f"[debug] obs[state] sample: {sample}")
            elif isinstance(value, np.ndarray):
                _summarize_array(f"obs[{key}]", value)
            elif isinstance(value, dict):
                print(f"[debug] obs[{key}] keys: {list(value.keys())}")
            else:
                print(f"[debug] obs[{key}] type={type(value)}")
    else:
        print(f"[debug] obs type: {type(obs)}")

    print(f"[debug] info keys: {list(info.keys()) if isinstance(info, dict) else type(info)}")
    prompts = info.get("prompt", None)
    if prompts is not None:
        if isinstance(prompts, str):
            prompts = [prompts]
        print(f"[debug] prompt count: {len(prompts)}")
        print(f"[debug] prompt[0]: {prompts[0] if prompts else ''}")

    action_space = getattr(envs, "single_action_space", None)
    if action_space is not None:
        print(f"[debug] action_space: {action_space}")
    print(f"[debug] default_language: {evaluator.args.default_language}")
    print("[debug] ==== End Snapshot ====")


# ============================================================================
# Use HumanVideoSimEvaluateProcessor for all evaluation tasks.
# ============================================================================


def _parse_sim_task_ids(raw: Optional[str], num_envs: int, default_id: str) -> List[str]:
    if not raw:
        return [default_id] * num_envs
    sim_task_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if len(sim_task_ids) != num_envs:
        raise ValueError(
            f"Expected {num_envs} sim task ids, got {len(sim_task_ids)} from '{raw}'"
        )
    return sim_task_ids


def _resolve_eval_env_spec(args, sim_task_ids: List[str]) -> tuple[str, str]:
    primary_id = sim_task_ids[0] if sim_task_ids else args.env_id
    return extract_base_env_name(primary_id), extract_level(primary_id)


# uses HumanVideoSimEvaluateProcessor


def _build_evaluate_processor(args) -> HumanVideoSimEvaluateProcessor:
    """
    Create a unified evaluate_processor (consistent with Pi, required)

    Create HumanVideoSimEvaluateProcessor to handle uniformly:
    - language retrieval
    - normalization/denormalization
    - task mapping
    """
    if not getattr(args, "use_lerobot", False):
        raise ValueError(
            "--use_lerobot is required for OpenVLA evaluation.\n"
            "Old evaluation mode has been removed. Please use:\n"
            "  --use_lerobot \\\n"
            "  --sim_dataset_file <path> \\\n"
            "  --task_mapping_file <path> \\\n"
            "  --human_task_description_file <path>"
        )

    # Check required arguments
    sim_dataset_file = getattr(args, "sim_dataset_file", None)
    if not sim_dataset_file or not os.path.exists(sim_dataset_file):
        raise FileNotFoundError(
            f"sim_dataset_file not found: {sim_dataset_file}\n"
            "This file is required for LeRobot evaluation."
        )

    # Create the config
    config = HumanVideoSimEvaluateProcessorConfig(
        human_root=getattr(args, "human_root", "data"),
        human_split=getattr(args, "human_split", "train"),
        sim_root=getattr(args, "sim_root", "data"),
        human_dataset_file=getattr(args, "human_dataset_file", None),
        sim_dataset_file=sim_dataset_file,
        human_task_description_file=getattr(args, "human_task_description_file", None),
        sim_task_description_file=getattr(args, "sim_task_description_file", None),
        task_mapping_file=getattr(args, "task_mapping_file", None),
        sim_state_type=getattr(args, "state_type", "qpos"),
        sim_single_arm=False,
        normalization_method="bounds_q99",
        vla=True,  # VLA mode: use language instead of video
    )

    try:
        processor = HumanVideoSimEvaluateProcessor(config)
        print("[Info] Created HumanVideoSimEvaluateProcessor (unified interface with Pi)")
        return processor
    except Exception as e:
        raise RuntimeError(f"Failed to create evaluate_processor: {e}")


class OpenVLAEvaluator:
    """OpenVLA Evaluator - uses the unified evaluate_processor interface (consistent with Pi, required)"""

    def __init__(self, args, evaluate_processor: HumanVideoSimEvaluateProcessor):
        if evaluate_processor is None:
            raise ValueError(
                "evaluate_processor is required for OpenVLAEvaluator.\n"
                "Old evaluation mode has been removed. Please use --use_lerobot."
            )

        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug_obs = getattr(args, "debug_obs", False)
        self._debug_printed = False

        # ============ Unified interface: use evaluate_processor (consistent with Pi) ============
        self.evaluate_processor = evaluate_processor

        # unnorm_key used inside the model (if the model supports internal denormalization)
        dataset_name = getattr(args, "dataset_name", None) or getattr(args, "unnorm_key", None)
        if not dataset_name:
            dataset_name = "maniskill_pickcubeycb"
        self.unnorm_key = dataset_name

    def process_inputs(self, image: Image.Image, lang_text: str, processor) -> Dict[str, torch.Tensor]:
        """Process image and language inputs for model inference."""
        # Resize image if needed
        if image.size != tuple(self.args.image_size):
            image = image.resize(self.args.image_size, Image.Resampling.BILINEAR)

        # Process image through processor
        pixel_values = processor.image_processor.apply_transform(image)
        if self.debug_obs and self._debug_printed == "obs":
            _summarize_tensor("input.pixel_values", pixel_values)
            self._debug_printed = "inputs"

        prompt_builder = PurePromptBuilder("openvla")
        prompt_builder.add_turn("human", f"What action should the robot take to {lang_text.lower()}?")
        prompt = prompt_builder.get_prompt()
        text_inputs = processor.tokenizer(prompt, add_special_tokens=True, return_tensors="pt")

        return {
            "pixel_values": pixel_values.unsqueeze(0),
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
        }

    def get_action(
            self,
            vla,
            processor,
            image: Image.Image,
            sim_task_id: str,  # required
            num_envs: int = 1,
            action_head=None,
            noisy_action_projector=None,
            proprio=None,
            proprio_projector=None,
            use_film: bool = False,
            lang_text: Optional[str] = None,
    ) -> np.ndarray:
        """Get action prediction from VLA model.

        Unified interface (consistent with Pi):
            - Use evaluate_processor.get_language() to get the language
            - Use evaluate_processor.denormalize_action() to denormalize

        Args:
            sim_task_id: Simulation task ID (required)
        """
        if not sim_task_id:
            raise ValueError("sim_task_id is required for unified interface")

        if lang_text is None:
            lang_text = self.evaluate_processor.get_language(sim_task_id, 1)[0]

        # Process inputs
        inputs = self.process_inputs(image, lang_text, processor)

        # Move to device; match vision backbone dtype to avoid dtype mismatches.
        pixel_dtype = _infer_model_dtype(vla)
        try:
            vision_backbone = getattr(vla, "vision_backbone", None)
            if vision_backbone is None and hasattr(vla, "model"):
                vision_backbone = vla.model.vision_backbone
            if vision_backbone is not None:
                pixel_dtype = next(vision_backbone.parameters()).dtype
        except StopIteration:
            pass
        pixel_values = inputs["pixel_values"].to(self.device, dtype=pixel_dtype)
        input_ids = inputs["input_ids"].to(self.device, dtype=torch.long)
        attention_mask = inputs["attention_mask"].to(self.device, dtype=torch.long)
        proprio_fusion_mode = getattr(self.args, "proprio_fusion_mode", "input")

        if action_head is not None and hasattr(action_head, "noise_scheduler"):
            num_steps = getattr(self.args, "num_diffusion_steps_inference", None)
            if num_steps is not None:
                action_head.noise_scheduler.set_timesteps(num_steps)

        autocast_dtype = _infer_model_dtype(vla)
        use_autocast = torch.cuda.is_available()

        # Guard: if output-side fusion is requested, all required components must be present.
        # This mode matches the cache-training pipeline where VLA saw no proprio.
        if proprio_fusion_mode == "output":
            missing = []
            if action_head is None:
                missing.append("action_head")
            if proprio is None:
                missing.append("proprio (pass --use_proprio and ensure env returns state)")
            if proprio_projector is None:
                missing.append("proprio_projector (pass --use_proprio so the projector is loaded)")
            if getattr(self.args, "action_mode", None) != "l1_regression":
                missing.append("action_mode must be l1_regression")
            if missing:
                raise RuntimeError(
                    "proprio_fusion_mode='output' requires: " + ", ".join(missing) + ". "
                    "Make sure you pass --use_proprio --proprio_fusion_mode output "
                    "--action_mode l1_regression when evaluating a cache-trained or frozen-VLA hidden-state model."
                )

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
                if (
                    proprio_fusion_mode == "output"
                    and getattr(self.args, "action_mode", None) == "l1_regression"
                    and action_head is not None
                    and proprio is not None
                    and proprio_projector is not None
                ):
                    # Match cache/frozen-VLA hidden-state training: keep proprio out of VLA, then fuse after VLA.
                    _, actions_hidden_states = vla.predict_action(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        action_head=action_head,
                        unnorm_key=self.unnorm_key,
                        noisy_action_projector=None,
                        proprio=None,
                        proprio_projector=None,
                        use_film=use_film,
                    )
                    if not isinstance(actions_hidden_states, torch.Tensor):
                        raise TypeError("Expected actions_hidden_states tensor from vla.predict_action")
                    proprio = proprio.to(self.device, dtype=autocast_dtype)
                    if proprio.dim() == 1:
                        proprio = proprio.unsqueeze(0)
                    proprio_feat = proprio_projector(proprio)
                    actions_hidden_states = actions_hidden_states.to(self.device, dtype=autocast_dtype)
                    fused_hidden_states = actions_hidden_states + proprio_feat.unsqueeze(1)
                    actions = action_head.predict_action(fused_hidden_states)[0]
                else:
                    actions, _ = vla.predict_action(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        action_head=action_head,
                        unnorm_key=self.unnorm_key,
                        noisy_action_projector=noisy_action_projector,
                        proprio=proprio,
                        proprio_projector=proprio_projector,
                        use_film=use_film,
                    )

            # ============ Unified interface: denormalize with evaluate_processor (consistent with Pi) ============
            if not isinstance(actions, torch.Tensor):
                actions = torch.as_tensor(actions)
            if torch.is_floating_point(actions) and actions.dtype == torch.bfloat16:
                actions = actions.to(torch.float32)
            actions = self.evaluate_processor.denormalize_action(actions, sim_task_id)
            if isinstance(actions, torch.Tensor):
                if torch.is_floating_point(actions) and actions.dtype == torch.bfloat16:
                    actions = actions.to(torch.float32)
                actions = actions.detach().cpu().numpy()

        return actions


def evaluate(
        n: int,
        vla,
        processor,
        eval_envs,
        device: str,
        sim_backend: str,
        progress_bar: bool = True,
        action_head=None,
        noisy_action_projector=None,
        proprio_projector=None,
        action_tokenizer=None,
        use_film: bool = False,
        image_size: tuple = (224, 224),
        evaluator: Optional[OpenVLAEvaluator] = None,  # required
        sim_task_ids: Optional[List[str]] = None,  # required
        action_chunk_steps: Optional[int] = None,
):
    """Evaluate VLA policy in ManiSkill environments.

    Args:
        evaluator: OpenVLAEvaluator instance with evaluate_processor (required)
        sim_task_ids: List of simulation task IDs (required)
    """
    if evaluator is None:
        raise ValueError("evaluator is required")
    if sim_task_ids is None:
        raise ValueError("sim_task_ids is required")
    prev_vla_training = vla.training
    prev_action_head_training = action_head.training if action_head is not None else None
    vla.eval()
    if action_head is not None:
        action_head.eval()

    if progress_bar:
        pbar = tqdm(total=n, desc="Evaluating")

    single_action_space = getattr(eval_envs, "single_action_space", None)
    use_dict_action = isinstance(single_action_space, gym.spaces.Dict)

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        num_envs = eval_envs.num_envs
        if len(sim_task_ids) != num_envs:
            raise ValueError(
                f"sim_task_ids length {len(sim_task_ids)} does not match num_envs {num_envs}"
            )

        current_languages = [
            evaluator.evaluate_processor.get_language(sim_task_ids[env_idx], 1)[0]
            for env_idx in range(num_envs)
        ]

        while eps_count < n:

            # Convert observations
            obs = common.to_tensor(obs, device)
            if evaluator is not None and evaluator.debug_obs and not evaluator._debug_printed:
                _print_debug_obs(obs, info, eval_envs, evaluator)
                evaluator._debug_printed = "obs"

            # Get action sequence from VLA for each environment
            action_seq_list = []
            for env_idx in range(num_envs):
                # Extract RGB image
                rgb_image = obs['rgb'][env_idx][0][..., :3]  # (H, W, C)
                rgb_np = rgb_image.cpu().numpy().astype(np.uint8)
                image = Image.fromarray(rgb_np)

                if image.size != image_size:
                    image = image.resize(image_size, Image.Resampling.BILINEAR)

                try:
                    proprio = None
                    if evaluator.args.use_proprio:
                        state_all = obs.get("state")
                        if state_all is None:
                            raise ValueError("use_proprio is enabled but obs has no 'state'")
                        state_env = state_all[env_idx]
                        state_norm, _ = evaluator.evaluate_processor.normalize_state_rgb(
                            state_env,
                            rgb_image,
                            sim_task_ids[env_idx],
                        )
                        if isinstance(state_norm, torch.Tensor):
                            proprio = _prepare_proprio(state_norm, evaluator.args.action_dim)

                    # Unified interface: use evaluate_processor
                    action_seq = evaluator.get_action(
                        vla=vla,
                        processor=processor,
                        image=image,
                        sim_task_id=sim_task_ids[env_idx],
                        num_envs=1,  # single-environment evaluation
                        action_head=action_head,
                        noisy_action_projector=noisy_action_projector,
                        proprio=proprio,
                        proprio_projector=proprio_projector,
                        use_film=use_film,
                        lang_text=current_languages[env_idx],
                    )

                    if len(action_seq.shape) == 1:
                        action_seq = action_seq.reshape(1, -1)

                    action_seq_list.append(action_seq)

                except Exception as e:
                    print(f"Error getting action for env {env_idx}: {e}")
                    action_seq_list.append(np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM)))

            # Stack actions
            try:
                action_seq = np.stack(action_seq_list, axis=0)
            except Exception as e:
                print(f"Error stacking actions: {e}")
                action_seq = np.zeros((eval_envs.num_envs, NUM_ACTIONS_CHUNK, ACTION_DIM))

            # Convert format
            if sim_backend == "physx_cpu":
                action_seq = action_seq
            else:
                action_seq = torch.from_numpy(action_seq).to(device)

            # Execute only the first K steps if requested
            steps_to_execute = action_seq.shape[1]
            if action_chunk_steps is not None:
                steps_to_execute = max(1, min(int(action_chunk_steps), steps_to_execute))
            # Execute action sequence
            for i in range(steps_to_execute):
                if isinstance(action_seq, np.ndarray):
                    action = action_seq[:, i]
                else:
                    action = action_seq[:, i].cpu().numpy()

                if use_dict_action:
                    action = _unflatten_batched_action(single_action_space, action)

                obs, rew, terminated, truncated, info = eval_envs.step(action)

                if truncated.any():
                    break

            # Process episode completion
            if truncated.any():
                assert truncated.all() == truncated.any()

                # Collect metrics
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        if isinstance(v, torch.Tensor):
                            eval_metrics[k].append(v.float().cpu().numpy())
                        else:
                            eval_metrics[k].append(v)
                else:
                    for final_info in info["final_info"]:
                        if final_info is not None and "episode" in final_info:
                            for k, v in final_info["episode"].items():
                                if isinstance(v, torch.Tensor):
                                    eval_metrics[k].append(v.float().cpu().numpy())
                                else:
                                    eval_metrics[k].append(v)

                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)

                # Reset environment
                obs, info = eval_envs.reset()
                current_languages = [
                    evaluator.evaluate_processor.get_language(sim_task_ids[env_idx], 1)[0]
                    for env_idx in range(num_envs)
                ]

    if progress_bar:
        pbar.close()

    # Restore prior modes so training-time eval does not perturb subsequent steps.
    vla.train(prev_vla_training)
    if action_head is not None and prev_action_head_training is not None:
        action_head.train(prev_action_head_training)

    # Convert metrics to numpy arrays
    for k in eval_metrics.keys():
        if len(eval_metrics[k]) > 0:
            eval_metrics[k] = np.stack(eval_metrics[k])

    return eval_metrics


def _infer_model_dtype(model: torch.nn.Module) -> torch.dtype:
    for param in model.parameters():
        if param.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return param.dtype
    return torch.float32


def _is_hf_model_dir(path: Optional[Path]) -> bool:
    if path is None:
        return False
    return (
        (path / "config.json").exists()
        and (
            (path / "model.safetensors").exists()
            or (path / "pytorch_model.bin").exists()
            or any(path.glob("model-*.safetensors"))
            or any(path.glob("pytorch_model-*.bin"))
        )
    )


def _unflatten_batched_action(action_space: gym.Space, action: np.ndarray) -> Dict[str, np.ndarray]:
    action = np.asarray(action)
    if action.ndim == 1:
        action = action[None, :]
    per_env = [gym.spaces.utils.unflatten(action_space, action[i]) for i in range(action.shape[0])]
    return {k: np.stack([d[k] for d in per_env], axis=0) for k in per_env[0].keys()}


def _prepare_proprio(state: torch.Tensor, target_dim: int) -> torch.Tensor:
    if state.dim() >= 2:
        state = state[0]
    if state.shape[-1] < target_dim:
        pad = torch.zeros(target_dim - state.shape[-1], device=state.device, dtype=state.dtype)
        state = torch.cat([state, pad], dim=-1)
    elif state.shape[-1] > target_dim:
        state = state[: target_dim]
    return state


def evaluate_and_save_best(
        iteration: int,
        vla,
        processor,
        eval_envs,
        device: str,
        sim_backend: str,
        demo_path: str,
        action_dim: int,
        output_dir: str,
        norm_stats_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        action_head=None,
        noisy_action_projector=None,
        proprio_projector=None,
        action_tokenizer=None,
        use_film: bool = False,
        action_mode: str = "discrete",
        image_size: tuple = (224, 224),
        num_diffusion_steps_inference: Optional[int] = None,
        attn_implementation: str = "sdpa",
        num_eval_episodes: int = 50,
        eval_freq: int = 5000,
        best_eval_metrics: Optional[Dict] = None,
        save_checkpoint_fn: Optional[Callable] = None,
        writer=None,
        default_language: str = "pick red cube and place on plate.",
        evaluate_processor: Optional[HumanVideoSimEvaluateProcessor] = None,  # unified interface (required)
        sim_task_ids: Optional[List[str]] = None,  # required
        proprio_fusion_mode: str = "input",
) -> Dict:
    """Evaluate VLA and save best checkpoints.

    Args:
        evaluate_processor: HumanVideoSimEvaluateProcessor instance (required)
        sim_task_ids: List of simulation task IDs (required)
    """
    if evaluate_processor is None:
        raise ValueError("evaluate_processor is required")
    if sim_task_ids is None:
        raise ValueError("sim_task_ids is required")

    if best_eval_metrics is None:
        best_eval_metrics = defaultdict(float)

    print(f"\n{'=' * 60}")
    print(f"Evaluating at iteration {iteration}")
    print(f"{'=' * 60}")

    # Create evaluator
    class EvalArgs:
        def __init__(self):
            self.action_mode = action_mode
            self.use_proprio = proprio_projector is not None
            self.image_size = image_size
            self.demo_path = demo_path
            self.action_dim = action_dim
            self.output_dir = output_dir
            self.default_language = default_language
            self.stats_path = norm_stats_path
            self.dataset_name = dataset_name
            self.num_diffusion_steps_inference = num_diffusion_steps_inference
            self.debug_obs = False
            self.attn_implementation = attn_implementation
            self.proprio_fusion_mode = proprio_fusion_mode
    eval_args = EvalArgs()
    evaluator = OpenVLAEvaluator(eval_args, evaluate_processor)
    evaluator.processor = processor

    # Quick evaluation
    quick_eval_metrics = evaluate(
        n=min(10, num_eval_episodes),
        vla=vla,
        processor=processor,
        eval_envs=eval_envs,
        device=device,
        sim_backend=sim_backend,
        progress_bar=True,
        action_head=action_head,
        noisy_action_projector=noisy_action_projector,
        proprio_projector=proprio_projector,
        action_tokenizer=action_tokenizer,
        use_film=use_film,
        image_size=image_size,
        evaluator=evaluator,
        sim_task_ids=sim_task_ids,
    )

    # Full evaluation if promising
    quick_success = np.mean(quick_eval_metrics.get('success_at_end', [0]))
    if quick_success >= 0.3:
        eval_metrics = evaluate(
            n=num_eval_episodes,
            vla=vla,
            processor=processor,
            eval_envs=eval_envs,
            device=device,
            sim_backend=sim_backend,
            progress_bar=True,
            action_head=action_head,
            noisy_action_projector=noisy_action_projector,
            proprio_projector=proprio_projector,
            action_tokenizer=action_tokenizer,
            use_film=use_film,
            image_size=image_size,
            evaluator=evaluator,
            sim_task_ids=sim_task_ids,
        )
    else:
        eval_metrics = quick_eval_metrics

    # Log metrics
    print(f"\nEvaluated {len(eval_metrics.get('success_at_end', []))} episodes")

    metric_summary = {}
    for k in eval_metrics.keys():
        if len(eval_metrics[k]) > 0:
            metric_value = np.mean(eval_metrics[k])
            metric_summary[k] = metric_value

            if writer:
                writer.add_scalar(f"eval/{k}", metric_value, iteration)

            print(f"  {k}: {metric_value:.4f}")

    # Save best checkpoints
    save_on_best_metrics = ["success_once", "success_at_end"]
    for metric_name in save_on_best_metrics:
        if metric_name in metric_summary:
            current_value = metric_summary[metric_name]

            if current_value > best_eval_metrics[metric_name]:
                best_eval_metrics[metric_name] = current_value

                if save_checkpoint_fn:
                    save_checkpoint_fn(f"best_eval_{metric_name}")

                print(f"  ✓ New best {metric_name}: {current_value:.4f} (saved checkpoint)")

    print(f"{'=' * 60}\n")

    return best_eval_metrics


def _register_hf():
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def _resolve_stats_path(args) -> str:
    if args.stats_path:
        return args.stats_path

    stats_file = "openvla_lerobot_stats.json" if args.use_lerobot else \
        "openvla_maniskill_pickcubeycb.json"

    candidates = []
    if args.checkpoint_dir:
        candidates.append(Path(args.checkpoint_dir) / stats_file)
        candidates.append(Path(args.checkpoint_dir).parent / stats_file)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0]) if candidates else stats_file


def _resolve_dataset_name(args) -> str:
    if args.dataset_name:
        return args.dataset_name
    if args.use_lerobot:
        return args.env_id
    return "maniskill_pickcubeycb"


def _load_model_and_processor(args):
    _register_hf()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    quantization_config = None
    use_kbit = bool(args.load_in_8bit)
    if use_kbit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    has_adapter = checkpoint_dir is not None and (checkpoint_dir / "adapter_config.json").exists()

    processor_path = args.model_name_or_path
    if checkpoint_dir is not None and (checkpoint_dir / "preprocessor_config.json").exists():
        processor_path = str(checkpoint_dir)

    processor = AutoProcessor.from_pretrained(
        processor_path,
        local_files_only=True,
        trust_remote_code=False,
    )

    # Force single GPU for evaluation (avoid multi-GPU device conflicts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if has_adapter:
        base_model = AutoModelForVision2Seq.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            attn_implementation=args.attn_implementation,
            device_map={"": device} if use_kbit else None,  # Force single device
            quantization_config=quantization_config,
            local_files_only=True,
        )
        vla = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
    else:
        load_path = args.model_name_or_path
        if _is_hf_model_dir(checkpoint_dir):
            load_path = str(checkpoint_dir)
        elif checkpoint_dir is not None:
            print(
                "[Eval] checkpoint_dir does not contain a full HF/OpenVLA model; "
                "loading base VLA from --model_name_or_path and auxiliary modules from --checkpoint_dir."
            )
        vla = AutoModelForVision2Seq.from_pretrained(
            load_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            attn_implementation=args.attn_implementation,
            device_map={"": device} if use_kbit else None,  # Force single device
            quantization_config=quantization_config,
            local_files_only=True,
        )

    # Move to device if not using quantization
    if not use_kbit:
        vla = vla.to(device)

    vla.eval()
    return vla, processor


def _load_aux_modules(args, vla):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = _infer_model_dtype(vla)

    action_head = None
    noisy_action_projector = None
    proprio_projector = None
    missing = []

    checkpoint_dir = Path(args.checkpoint_dir)

    if args.action_mode == "l1_regression":
        action_head = L1RegressionActionHead(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=args.action_dim,
        )
    elif args.action_mode == "diffusion":
        action_head = DiffusionActionHead(
            input_dim=vla.llm_dim,
            hidden_dim=vla.llm_dim,
            action_dim=args.action_dim,
            num_diffusion_steps_train=args.num_diffusion_steps_train,
        )
        noisy_action_projector = NoisyActionProjector(llm_dim=vla.llm_dim)

    if args.use_proprio:
        proprio_projector = ProprioProjector(
            llm_dim=vla.llm_dim,
            proprio_dim=args.action_dim,
        )

    if action_head is not None:
        action_head_path = checkpoint_dir / "action_head.pt"
        if action_head_path.exists():
            action_head.load_state_dict(torch.load(action_head_path, map_location="cpu"))
        else:
            missing.append("action_head.pt")
        action_head.to(device, dtype=model_dtype).eval()

    if noisy_action_projector is not None:
        proj_path = checkpoint_dir / "noisy_action_projector.pt"
        if proj_path.exists():
            noisy_action_projector.load_state_dict(torch.load(proj_path, map_location="cpu"))
        else:
            missing.append("noisy_action_projector.pt")
        noisy_action_projector.to(device, dtype=model_dtype).eval()

    if proprio_projector is not None:
        proj_path = checkpoint_dir / "proprio_projector.pt"
        if proj_path.exists():
            proprio_projector.load_state_dict(torch.load(proj_path, map_location="cpu"))
        else:
            missing.append("proprio_projector.pt")
        proprio_projector.to(device, dtype=model_dtype).eval()

    if missing:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint_dir} is missing required auxiliary file(s): {', '.join(missing)}"
        )

    return action_head, noisy_action_projector, proprio_projector


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone OpenVLA evaluation")

    parser.add_argument("--checkpoint_dir", required=True, help="Path to saved checkpoint directory (e.g. runs/.../final)")
    parser.add_argument("--model_name_or_path", required=True, help="Base OpenVLA model path")
    parser.add_argument("--stats_path", default=None, help="Path to stats json for action unnormalization")
    parser.add_argument("--dataset_name", default=None, help="Dataset name key inside stats json")
    parser.add_argument("--use_lerobot", action="store_true", default=False)
    parser.add_argument(
        "--demo_path",
        default=None,
        help="Optional .h5 demo path (unused for eval; kept for compatibility).",
    )

    parser.add_argument("--env_id", default="PickCubeYCB-v1")
    parser.add_argument("--obs_mode", default="rgb")
    parser.add_argument("--control_mode", default="pd_joint_pos")
    parser.add_argument("--max_episode_steps", type=int, default=600)
    parser.add_argument("--sim_backend", default="physx_cpu")
    parser.add_argument("--shader", default="rt-fast")
    parser.add_argument("--obs_horizon", type=int, default=1)
    parser.add_argument("--num_eval_envs", type=int, default=1)
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--capture_video", action="store_true", default=False)
    parser.add_argument("--eval_camera", type=str, default=None,
                        help="Optional camera name to use for evaluation observations.")
    parser.add_argument("--lerobot_camera", type=str, default="zed2i")

    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--default_language", default="pick red cube and place on plate.")
    parser.add_argument("--human_root", default="data")
    parser.add_argument("--human_dataset_file",
                        default="examples/baselines/lerobot_dataset/config/human_config.json")
    parser.add_argument("--human_split", default="train")
    parser.add_argument("--human_task_description_file",
                        default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    parser.add_argument("--sim_task_description_file",
                        default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")
    parser.add_argument("--task_mapping_file", default="examples/baselines/lerobot_dataset/task_mapping.json")
    parser.add_argument("--sim_root", default="data")
    parser.add_argument("--sim_dataset_file", default=None)
    parser.add_argument("--eval_sim_task_ids", default=None,
                        help="Comma-separated sim_task_id list matching num_eval_envs.")
    parser.add_argument("--eval_lr_mirror", default="false", choices=["auto", "true", "false"],
                        help="Override tabletop left-right mirror during eval. Default 'false'.")
    parser.add_argument("--eval_lr_mirror_robot_pose", default="false", choices=["auto", "true", "false"],
                        help="Override robot pose swapping under mirror during eval. Default 'false' keeps agents unswapped.")

    parser.add_argument("--action_dim", type=int, default=16)
    parser.add_argument("--action_mode", default="discrete", choices=["discrete", "l1_regression", "diffusion"])
    parser.add_argument("--num_diffusion_steps_train", type=int, default=50)
    parser.add_argument("--num_diffusion_steps_inference", type=int, default=50)
    parser.add_argument("--action_chunk_steps", type=int, default=None,
                        help="Execute only the first K steps of the predicted action chunk (e.g., 4).")
    parser.add_argument("--use_proprio", action="store_true", default=False)
    parser.add_argument("--use_film", action="store_true", default=False)
    parser.add_argument("--proprio_fusion_mode", type=str, default="input", choices=["input", "output"],
                        help="Where to fuse proprio for continuous-action eval. "
                             "'input' keeps legacy VLA-input fusion; "
                             "'output' matches cached-VL-feature training and fuses proprio after VLA.")

    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                        choices=["eager", "sdpa", "flash_attention_2"])

    parser.add_argument("--metrics_out", default=None, help="Optional path to save metrics json")
    parser.add_argument("--debug_obs", action="store_true", default=False,
                        help="Print a one-time snapshot of eval observations for debugging.")

    return parser.parse_args()


def main():
    args = parse_args()

    args.image_size = tuple(args.image_size)
    dataset_name = _resolve_dataset_name(args)
    stats_path = _resolve_stats_path(args)

    vla, processor = _load_model_and_processor(args)
    action_head, noisy_action_projector, proprio_projector = _load_aux_modules(args, vla)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ Unified interface: use evaluate_processor (consistent with Pi, required) ============
    evaluate_processor = _build_evaluate_processor(args)  # raises an exception on failure

    # Parse sim_task_ids
    sim_task_ids = _parse_sim_task_ids(args.eval_sim_task_ids, args.num_eval_envs, args.env_id)
    base_env_name, l_level = _resolve_eval_env_spec(args, sim_task_ids)
    lr_mirror_enabled = parse_optional_bool_flag(args.eval_lr_mirror)
    lr_mirror_robot_pose_enabled = parse_optional_bool_flag(args.eval_lr_mirror_robot_pose)

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="dense",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)

    video_dir = None
    if args.capture_video:
        video_dir = os.path.join(args.checkpoint_dir, "videos")

    wrappers = []
    eval_camera = args.eval_camera
    if eval_camera is None and args.use_lerobot:
        eval_camera = args.lerobot_camera
    if eval_camera:
        wrappers.append(partial(SelectCameraObservationWrapper, camera_names=[eval_camera]))
    wrappers.append(FlattenRGBDObservationWrapper)

    configure_tabletop_task_flags(
        level=l_level,
        lr_mirror_enabled=lr_mirror_enabled,
        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
    )

    eval_envs = make_eval_envs(
        base_env_name,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=video_dir,
        wrappers=wrappers,
        l_level=l_level,
        lr_mirror_enabled=lr_mirror_enabled,
        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
    )

    class EvalArgs:
        def __init__(self):
            self.action_mode = args.action_mode
            self.use_proprio = args.use_proprio
            self.proprio_fusion_mode = args.proprio_fusion_mode
            self.image_size = args.image_size
            self.demo_path = args.demo_path or ""
            self.action_dim = args.action_dim
            self.output_dir = args.checkpoint_dir
            self.default_language = args.default_language
            self.stats_path = stats_path
            self.dataset_name = dataset_name
            self.debug_obs = args.debug_obs

    evaluator = OpenVLAEvaluator(EvalArgs(), evaluate_processor)

    try:
        metrics = evaluate(
            n=args.num_eval_episodes,
            vla=vla,
            processor=processor,
            eval_envs=eval_envs,
            device=device,
            sim_backend=args.sim_backend,
            progress_bar=True,
            action_head=action_head,
            noisy_action_projector=noisy_action_projector,
            proprio_projector=proprio_projector,
            use_film=args.use_film,
            image_size=args.image_size,
            evaluator=evaluator,
            sim_task_ids=sim_task_ids,
            action_chunk_steps=args.action_chunk_steps,
        )

        print("\nEvaluation Summary")
        for k, v in metrics.items():
            if len(v) == 0:
                continue
            try:
                print(f"  {k}: {float(np.mean(v)):.4f}")
            except Exception:
                print(f"  {k}: {v}")

        if args.metrics_out:
            with open(args.metrics_out, "w") as f:
                json.dump({k: np.asarray(v).tolist() for k, v in metrics.items()}, f, indent=2)
    finally:
        eval_envs.close()
        clear_tabletop_task_flags()


if __name__ == "__main__":
    main()
