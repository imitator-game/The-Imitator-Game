"""
RDT Model Evaluation Script - Simplified (Language-only, no video)
Directly uses language prompts from environment without video conditioning
"""

import os
import random
import time
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import tyro
import gymnasium as gym
from gymnasium import spaces
from mani_skill.utils import common
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

# Import RDT components
from examples.baselines.rdt.train_rdt_scratch import RDTAgent, Args as TrainArgs
from examples.baselines.rdt.utils.make_env import make_eval_envs
from examples.baselines.rdt.utils.eval import evaluate
from examples.baselines.rdt.utils.precomputed_lang_loader import HybridLangEmbedding
from examples.baselines.lerobot_dataset.trajectory_metrics import (
    GTTrajectoryProvider,
    TrajectoryMetrics,
)


def _extract_l_level_from_task_id(task_id: Optional[str]) -> Optional[str]:
    if not task_id:
        return None
    match = re.match(r"^(L[0-3])_", str(task_id))
    return match.group(1) if match else None


def _extract_base_env_id(task_id: str) -> str:
    if isinstance(task_id, str) and re.match(r"^L[0-3]_.+", task_id):
        return task_id.split("_", 1)[1]
    return task_id


def _with_l3_env_suffix(env_id: str) -> str:
    if re.search(r"L3-v\d+$", env_id):
        return env_id
    match = re.match(r"^(.+?)(-v\d+)$", env_id)
    if not match:
        raise ValueError(f"Cannot map L3 task id to ManiSkill env id: {env_id}")
    return f"{match.group(1)}L3{match.group(2)}"


def _resolve_eval_env_id(task_id: str) -> tuple[str, Optional[str]]:
    level = _extract_l_level_from_task_id(task_id)
    base_env_id = _extract_base_env_id(task_id)
    if level == "L3":
        # L3 tasks are dedicated ManiSkill envs under dual_tasks_l3. The
        # L0/L1/L2/L3 flags only control object/layout variants, so evaluate
        # these dedicated L3 envs with the default L0 variant.
        return _with_l3_env_suffix(base_env_id), "L0"
    if level in ("L0", "L1", "L2"):
        return base_env_id, level
    if re.search(r"L3-v\d+$", base_env_id):
        return base_env_id, "L0"
    return base_env_id, None


def _parse_optional_bool_flag(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in {"auto", "default", "none"}:
        return None
    if value in {"1", "true", "on", "yes", "enable", "enabled"}:
        return True
    if value in {"0", "false", "off", "no", "disable", "disabled"}:
        return False
    raise ValueError(f"Invalid boolean flag value: {raw}")


def _set_l_level_flags(
    level: Optional[str],
    lr_mirror_enabled: Optional[bool] = None,
    lr_mirror_robot_pose_enabled: Optional[bool] = None,
) -> None:
    level = str(level).upper() if level is not None else None
    if level is not None and level not in ("L0", "L1", "L2", "L3"):
        raise ValueError(f"l_level must be one of L0/L1/L2/L3, got: {level}")
    for env_var in ("MANI_SKILL_L1", "MANI_SKILL_L2", "MANI_SKILL_L3"):
        os.environ.pop(env_var, None)
    if level in ("L1", "L2", "L3"):
        os.environ[f"MANI_SKILL_{level}"] = "1"
    try:
        from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils as lutils
        lutils.set_l1_enabled(level == "L1")
        lutils.set_l2_enabled(level == "L2")
        lutils.set_l3_enabled(level == "L3")
        lutils.set_lr_mirror_enabled(lr_mirror_enabled)
        lutils.set_lr_mirror_mode(None)
        lutils.set_lr_mirror_robot_pose_enabled(lr_mirror_robot_pose_enabled)
    except Exception:
        return


def _action_key_for_state_type(state_type: str) -> str:
    if state_type in ("qpos", "mixpos"):
        return "action.qpos_gripper_actions"
    if state_type == "eepos":
        return "action.eepos_gripper_actions"
    raise ValueError(f"Unknown lerobot_state_type for TSS metrics: {state_type}")


def _build_trajectory_metrics(args: "EvalArgs", normalizer=None):
    if not args.compute_dtw:
        return None, None
    if not args.use_lerobot or not args.lerobot_eval_online:
        print("[TSS] compute_dtw requested, but TSS is only enabled for online LeRobot eval.")
        return None, None

    try:
        action_key = _action_key_for_state_type(args.lerobot_state_type)
        dtw_provider = GTTrajectoryProvider(
            sim_dataset_file=args.lerobot_sim_dataset_file,
            sim_root=args.lerobot_sim_root or args.lerobot_root or "demos",
            action_key=action_key,
            normalizer=normalizer,
        )
        traj_metrics = TrajectoryMetrics(band_ratio=args.dtw_band_ratio)
        print(
            "[TSS] Loaded GT trajectories: "
            f"tasks={len(dtw_provider._episodes)} action_key={action_key} "
            f"band_ratio={args.dtw_band_ratio}"
        )
        return dtw_provider, traj_metrics
    except Exception as exc:
        print(f"[TSS] Failed to initialize TSS metrics; continuing without TSS: {exc}")
        return None, None


@dataclass
class EvalArgs:
    # Checkpoint
    checkpoint_path: str
    """Path to model checkpoint (e.g., runs/experiment/checkpoints/best.pt)"""
    output_dir: Optional[str] = None
    """Directory to store per-task eval outputs such as eval_metrics.json"""

    # Environment
    env_id: str = "PickCubeYCB-v1"
    obs_mode: str = "rgb"
    control_mode: str = "pd_joint_pos"
    reward_mode: str = "dense"
    max_episode_steps: int = 600
    sim_backend: str = "physx_cpu"
    shader: str = "rt-fast"

    # Evaluation
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    seed: int = 0
    compute_dtw: bool = False
    """Compute TSS/nDTW trajectory metrics against sim GT actions during online eval"""
    dtw_band_ratio: float = 0.15
    """Sakoe-Chiba band ratio for TSS/nDTW"""
    gripper_binary_action: bool = True
    gripper_threshold: float = 0.4
    gripper_indices: Tuple[int, ...] = (7, 15)

    # Model config (should match training)
    obs_horizon: int = 2
    pred_horizon: int = 16
    hidden_size: int = 512
    depth: int = 8
    num_heads: int = 8
    img_size: Tuple[int, int] = (224, 224)
    vision_encoder: str = "google/siglip-so400m-patch14-384"
    text_encoder: str = "google/t5-v1_1-xxl"
    t5_version: str = "t5-v1_1-xxl"
    max_lang_len: int = 77
    num_diffusion_iters: int = 100
    num_inference_steps: int = 10

    # Language embedding options
    use_precomputed_lang: bool = False
    """Whether to use precomputed language embeddings"""
    precomputed_lang_dir: Optional[str] = None
    """Directory containing precomputed language embeddings"""

    # Video
    capture_video: bool = True
    video_dir: Optional[str] = None

    # Tabletop eval variants
    eval_lr_mirror: str = "auto"
    """Override tabletop left-right mirror during eval: auto, true, or false"""
    eval_lr_mirror_robot_pose: str = "false"
    """Override robot pose swapping under mirror during eval: auto, true, or false"""

    # Output
    save_results: bool = True
    """Whether to save evaluation results to JSON"""

    # Language handling
    use_dummy_language: bool = False
    """If true, skip text encoder and use zero language embeddings"""

    # LeRobot eval (optional)
    use_lerobot: bool = False
    """If true, evaluate on LeRobot sim dataset"""
    lerobot_repo_id: Optional[str] = None
    """LeRobot dataset repo id (required if use_lerobot is True)"""
    lerobot_root: Optional[str] = None
    """LeRobot dataset root dir"""
    lerobot_cameras: Tuple[str, ...] = ("zed2i",)
    """Camera list for LeRobot inputs (single camera only)"""
    lerobot_image_size: Tuple[int, int] = (224, 224)
    """Resize (H, W) for LeRobot images"""
    lerobot_state_type: str = "qpos"
    """LeRobot state type: eepos, qpos, mixpos"""
    lerobot_include_depth: bool = False
    """Whether to include depth from LeRobot (TODO: wire into RDT)"""
    lerobot_depth_mode: str = "sim"
    """Depth decoding mode for LeRobot"""
    lerobot_video_backend: str = "torchcodec"
    """Video backend for LeRobot"""
    lerobot_tolerance_s: float = 0.05
    """LeRobot video timestamp tolerance (seconds)"""
    lerobot_task_description_file: Optional[str] = None
    """Optional task description mapping for LeRobot language prompts"""
    lerobot_eval_batches: int = 100
    """Number of batches for offline LeRobot eval"""
    lerobot_batch_size: int = 64
    """Batch size for offline LeRobot eval"""
    lerobot_eval_online: bool = False
    """Run online rollout for LeRobot-trained checkpoints"""
    lerobot_debug: bool = False
    """Print online eval debug info for obs/action normalization"""
    lerobot_use_paired_dataset: bool = False
    """Use HumanSimPairedDataset to provide language prompts (VLA mode)"""
    lerobot_human_root: Optional[str] = None
    """Root directory for human LeRobot datasets when using paired dataset"""
    lerobot_sim_root: Optional[str] = None
    """Root directory for sim LeRobot datasets when using paired dataset"""
    lerobot_task_mapping_file: str = "examples/baselines/lerobot_dataset/task_mapping.json"
    """Task mapping file for paired dataset"""
    lerobot_human_dataset_file: str = "examples/baselines/lerobot_dataset/config/human_config.json"
    """Human dataset config file for paired dataset"""
    lerobot_sim_dataset_file: str = "examples/baselines/lerobot_dataset/config/sim_config.json"
    """Sim dataset config file for paired dataset"""
    lerobot_human_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    """Human task description file for paired dataset"""
    lerobot_sim_task_description_file: str = "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"
    """Sim task description file for paired dataset"""
    load_model_config_from_ckpt: bool = True
    """Load model-related args from checkpoint if available"""
    lerobot_use_eval_processor: bool = True
    """Use HumanVideoSimEvaluateProcessor for task-based normalization"""


class FlattenRGBDSelectWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env,
        camera_names,
        rgb=True,
        depth=True,
        state=True,
        sep_depth=True,
        state_type: str | None = None,
        expected_state_dim: int | None = None,
        gripper_threshold: float = 0.4,
        state_normalizer=None,
        normalization_method: str = "bounds_q99",
        dataset_idx: int = 0,
        image_size: tuple[int, int] | None = None,
        debug: bool = False,
        processor=None,
        sim_task_id: Optional[str] = None,
    ):
        self.base_env = env.unwrapped
        super().__init__(env)
        self.camera_names = list(camera_names)
        self.include_rgb = rgb
        self.include_depth = depth
        self.sep_depth = sep_depth
        self.include_state = state
        self.state_type = state_type
        self.expected_state_dim = expected_state_dim
        self.gripper_threshold = gripper_threshold
        self.state_normalizer = state_normalizer
        self.normalization_method = normalization_method
        self.dataset_idx = dataset_idx
        self.image_size = image_size
        self.debug = debug
        self._printed = False
        self.processor = processor
        self.sim_task_id = sim_task_id or self._infer_sim_task_id()

        first_cam = None
        for name in self.camera_names:
            if name in self.base_env._init_raw_obs["sensor_data"]:
                first_cam = self.base_env._init_raw_obs["sensor_data"][name]
                break
        if first_cam is None:
            raise ValueError(f"No matching cameras found in env: {self.camera_names}")
        if "depth" not in first_cam:
            self.include_depth = False
        if "rgb" not in first_cam:
            self.include_rgb = False
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def _infer_sim_task_id(self) -> str:
        spec = getattr(self.base_env, "spec", None)
        if spec is not None and getattr(spec, "id", None):
            return spec.id
        for attr in ("env_id", "task_id", "task_name"):
            value = getattr(self.base_env, attr, None)
            if value:
                return str(value)
        return ""

    def _collect_state_by_key(self, state_dict, key):
        values = []
        if isinstance(state_dict, dict):
            for k, v in state_dict.items():
                if k == key:
                    values.append(v)
                elif isinstance(v, dict):
                    values.extend(self._collect_state_by_key(v, key))
        return values

    def _stack_state(self, values):
        if not values:
            return torch.empty((0,), device=self.base_env.device)
        chunks = []
        batch_dim = None
        for value in values:
            tensor = torch.as_tensor(value, device=self.base_env.device)
            if (
                self.expected_state_dim in (8, 16)
                and tensor.dim() >= 1
                and tensor.shape[-1] == 9
                and self.state_type == "qpos"
            ):
                arm = tensor[..., :7]
                gripper = tensor[..., 7:9].mean(dim=-1, keepdim=True)
                gripper = torch.where(
                    gripper > self.gripper_threshold,
                    torch.ones_like(gripper),
                    -torch.ones_like(gripper),
                )
                tensor = torch.cat([arm, gripper], dim=-1)
            if tensor.dim() == 0:
                tensor = tensor.view(1, 1)
            elif tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.reshape(tensor.shape[0], -1)
            if batch_dim is None:
                batch_dim = tensor.shape[0]
            elif tensor.shape[0] != batch_dim:
                tensor = tensor.reshape(batch_dim, -1)
            chunks.append(tensor)
        if not chunks:
            return torch.empty((batch_dim or 0, 0), device=self.base_env.device)
        return torch.cat(chunks, dim=1)

    def _extract_state(self, observation):
        if not self.state_type:
            return common.flatten_state_dict(
                observation, use_torch=True, device=self.base_env.device
            )
        state_chunks = []
        if self.state_type in ("qpos", "mixpos"):
            state_chunks.extend(self._collect_state_by_key(observation, "qpos"))
        if self.state_type in ("eepos", "mixpos"):
            state_chunks.extend(self._collect_state_by_key(observation, "eepos"))
        if not state_chunks:
            return common.flatten_state_dict(
                observation, use_torch=True, device=self.base_env.device
            )
        return self._stack_state(state_chunks)

    def _resize_hwc(self, img: torch.Tensor) -> torch.Tensor:
        if self.image_size is None:
            return img
        if img.dim() == 5 and img.shape[0] == 1:
            img = img[0]
        h, w = self.image_size
        if img.dim() == 2:
            img = img.unsqueeze(-1)
        if img.dim() == 3:
            img_chw = img.permute(2, 0, 1).unsqueeze(0)
            resized = torch.nn.functional.interpolate(
                img_chw, size=(h, w), mode="bilinear", align_corners=False
            )
            return resized.squeeze(0).permute(1, 2, 0)
        if img.dim() == 4:
            img_chw = img.permute(0, 3, 1, 2)
            resized = torch.nn.functional.interpolate(
                img_chw, size=(h, w), mode="bilinear", align_corners=False
            )
            return resized.permute(0, 2, 3, 1)
        return img

    def _process_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.float()
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        rgb = self._resize_hwc(rgb)
        return rgb

    def _process_depth(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.float()
        if depth.max() > 10.0:
            depth = depth / 1000.0
        depth = self._resize_hwc(depth)
        return depth

    def observation(self, observation):
        sensor_data = observation.pop("sensor_data")
        if "sensor_param" in observation:
            del observation["sensor_param"]
        rgb_images = []
        depth_images = []
        for cam_name in self.camera_names:
            if cam_name not in sensor_data:
                continue
            cam_data = sensor_data[cam_name]
            if self.include_rgb:
                rgb_images.append(self._process_rgb(cam_data["rgb"]))
            if self.include_depth:
                depth_images.append(self._process_depth(cam_data["depth"]))

        if len(rgb_images) > 0:
            rgb_images = torch.concat(rgb_images, axis=-1)
        if len(depth_images) > 0:
            depth_images = torch.concat(depth_images, axis=-1)
        observation = self._extract_state(observation)
        ret = {}
        if (
            self.processor is not None
            and self.include_state
            and self.include_rgb
            and isinstance(rgb_images, torch.Tensor)
        ):
            rgb_concat = rgb_images if isinstance(rgb_images, torch.Tensor) else None
            if rgb_concat is not None:
                rgb_concat = self._resize_hwc(rgb_concat.float())
            state_out, rgb_out = self.processor.normalize_state_rgb(
                observation, rgb_concat, sim_task_id=self.sim_task_id
            )
            ret["state"] = state_out
            if self.include_rgb and not self.include_depth:
                ret["rgb"] = rgb_out
            elif self.include_rgb and self.include_depth:
                if self.sep_depth:
                    ret["rgb"] = rgb_out
                    ret["depth"] = depth_images
                else:
                    ret["rgbd"] = torch.concat([rgb_out, depth_images], axis=-1)
            elif self.include_depth and not self.include_rgb:
                ret["depth"] = depth_images
        else:
            if self.include_state:
                if self.state_normalizer is not None:
                    ret["state"] = self.state_normalizer.normalize_state(
                        observation,
                        dataset_idx=self.dataset_idx,
                        method=self.normalization_method,
                    )
                else:
                    ret["state"] = observation
            if self.include_rgb and not self.include_depth:
                ret["rgb"] = rgb_images
            elif self.include_rgb and self.include_depth:
                if self.sep_depth:
                    ret["rgb"] = rgb_images
                    ret["depth"] = depth_images
                else:
                    ret["rgbd"] = torch.concat([rgb_images, depth_images], axis=-1)
            elif self.include_depth and not self.include_rgb:
                ret["depth"] = depth_images
        if self.debug and not self._printed and "state" in ret:
            state_tensor = torch.as_tensor(ret["state"])
            rgb_tensor = torch.as_tensor(ret.get("rgb")) if "rgb" in ret else None
            depth_tensor = torch.as_tensor(ret.get("depth")) if "depth" in ret else None
            if rgb_tensor is not None:
                rgb_mean = float(rgb_tensor.mean().item())
                rgb_std = float(rgb_tensor.std().item())
                print(f"[lerobot-debug] rgb mean/std: {rgb_mean:.4f} / {rgb_std:.4f}")
            if depth_tensor is not None:
                depth_mean = float(depth_tensor.mean().item())
                depth_std = float(depth_tensor.std().item())
                print(f"[lerobot-debug] depth mean/std: {depth_mean:.4f} / {depth_std:.4f}")
            state_min = float(state_tensor.min().item())
            state_max = float(state_tensor.max().item())
            print(f"[lerobot-debug] state min/max: {state_min:.4f} / {state_max:.4f}")
            self._printed = True
        return ret


class ActionDenormalizeWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env,
        normalizer,
        dataset_idx: int,
        method: str,
        act_dim: int,
        debug: bool = False,
        processor=None,
        sim_task_id: Optional[str] = None,
    ):
        super().__init__(env)
        self.normalizer = normalizer
        self.dataset_idx = dataset_idx
        self.method = method
        self.debug = debug
        self._printed = False
        self.processor = processor
        self.sim_task_id = sim_task_id or self._infer_sim_task_id()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        self.single_action_space = self.action_space

    def _infer_sim_task_id(self) -> str:
        spec = getattr(self.env, "spec", None)
        if spec is not None and getattr(spec, "id", None):
            return spec.id
        for attr in ("env_id", "task_id", "task_name"):
            value = getattr(self.env, attr, None)
            if value:
                return str(value)
        return ""

    def action(self, action):
        action_tensor = torch.as_tensor(action)
        if self.processor is not None:
            denorm = self.processor.denormalize_action(
                action_tensor, sim_task_id=self.sim_task_id
            )
        else:
            denorm = self.normalizer.denormalize_action(
                action_tensor, dataset_idx=self.dataset_idx, method=self.method
            )
        if self.debug and not self._printed:
            norm_min = float(action_tensor.min().item())
            norm_max = float(action_tensor.max().item())
            denorm_min = float(denorm.min().item())
            denorm_max = float(denorm.max().item())
            print(f"[lerobot-debug] action norm min/max: {norm_min:.4f} / {norm_max:.4f}")
            print(f"[lerobot-debug] action denorm min/max: {denorm_min:.4f} / {denorm_max:.4f}")
            self._printed = True
        return denorm.detach().cpu().numpy()


class NormalizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, normalizer, dataset_idx: int, method: str, debug: bool = False):
        super().__init__(env)
        self.normalizer = normalizer
        self.dataset_idx = dataset_idx
        self.method = method
        self._debug = debug
        self._printed = False

    def observation(self, observation):
        if "state" not in observation:
            return observation
        state = observation["state"]
        state_tensor = torch.as_tensor(state)
        norm_state = self.normalizer.normalize_state(
            state_tensor, dataset_idx=self.dataset_idx, method=self.method
        )
        if self._debug and not self._printed:
            raw_min = float(state_tensor.min().item())
            raw_max = float(state_tensor.max().item())
            norm_min = float(norm_state.min().item())
            norm_max = float(norm_state.max().item())
            print(f"[lerobot-debug] state raw min/max: {raw_min:.4f} / {raw_max:.4f}")
            print(f"[lerobot-debug] state norm min/max: {norm_min:.4f} / {norm_max:.4f}")
            self._printed = True
        observation["state"] = norm_state
        return observation


def load_checkpoint(checkpoint_path: str, agent, device):
    """Load checkpoint from file"""
    print(f"Loading checkpoint from {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint["agent"])

    # Print checkpoint info
    iteration = checkpoint.get("iteration", "unknown")
    best_metrics = checkpoint.get("best_metrics", {})

    print(f"✓ Checkpoint loaded (iteration: {iteration})")
    if best_metrics:
        print(f"  Best metrics from training:")
        for k, v in best_metrics.items():
            print(f"    {k}: {v:.2f}%")

    return checkpoint


if __name__ == "__main__":
    args = tyro.cli(EvalArgs)
    sim_task_id_for_mapping = args.env_id
    eval_env_id, requested_level = _resolve_eval_env_id(args.env_id)
    lr_mirror_enabled = _parse_optional_bool_flag(args.eval_lr_mirror)
    lr_mirror_robot_pose_enabled = _parse_optional_bool_flag(args.eval_lr_mirror_robot_pose)
    _set_l_level_flags(
        requested_level,
        lr_mirror_enabled=lr_mirror_enabled,
        lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
    )
    print(
        "[env] "
        f"task_id={args.env_id} eval_env_id={eval_env_id} l_level={requested_level} "
        f"lr_mirror={args.eval_lr_mirror} "
        f"lr_mirror_robot_pose={args.eval_lr_mirror_robot_pose}"
    )

    if args.load_model_config_from_ckpt and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_args = checkpoint.get("args")
        if isinstance(ckpt_args, dict):
            model_keys = {
                "obs_horizon",
                "pred_horizon",
                "hidden_size",
                "depth",
                "num_heads",
                "img_size",
                "vision_encoder",
                "text_encoder",
                "max_lang_len",
                "num_diffusion_iters",
                "num_inference_steps",
            }
            for key in model_keys:
                if key in ckpt_args:
                    setattr(args, key, ckpt_args[key])
            print(f"[ckpt] loaded model config from {args.checkpoint_path}")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lerobot_dataset = None
    action_denorm = None
    include_depth = False
    paired_prompts = None
    eval_processor = None
    dataset_idx_for_eval = 0
    dtw_dataset_idx = 0
    dtw_provider = None
    traj_metrics = None
    if args.use_lerobot:
        include_depth = bool(args.lerobot_include_depth)
        if include_depth:
            # TODO: wire depth into RDT (vision encoder expects RGB-only today).
            include_depth = False

        if args.lerobot_use_paired_dataset:
            from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
                HumanSimPairedDataset,
                PairedDatasetConfig,
            )

            paired_config = PairedDatasetConfig(
                human_root=args.lerobot_human_root or args.lerobot_root or "demos",
                sim_root=args.lerobot_sim_root or args.lerobot_root or "demos",
                task_mapping_file=args.lerobot_task_mapping_file,
                human_dataset_file=args.lerobot_human_dataset_file,
                sim_dataset_file=args.lerobot_sim_dataset_file,
                human_task_description_file=args.lerobot_human_task_description_file,
                sim_task_description_file=args.lerobot_sim_task_description_file,
                split="train",
                cameras=["zed2i"],
                include_depth=include_depth,
                image_size=tuple(args.lerobot_image_size),
                horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                state_type=args.lerobot_state_type,
                fps=30,
                video_backend=args.lerobot_video_backend,
            )
            lerobot_dataset = HumanSimPairedDataset(paired_config)
            lerobot_sim_dataset = lerobot_dataset.sim_dataset
            action_denorm = {
                "normalizer": lerobot_sim_dataset.normalizer,
                "dataset_idx": 0,
                "method": lerobot_sim_dataset.config.normalization_method,
                "act_dim": lerobot_sim_dataset.action_dim,
            }
        else:
            if args.lerobot_repo_id is None:
                raise ValueError("lerobot_repo_id is required when use_lerobot is True")
            from examples.baselines.lerobot_dataset.lerobot_dataloader import (
                LeRobotDataConfig,
                build_lerobot_dataset,
            )

            lerobot_cameras = list(args.lerobot_cameras) if args.lerobot_cameras else ["zed2i"]
            lerobot_cameras = ["zed2i"]  # enforce single camera as requested

            lerobot_config = LeRobotDataConfig(
                source_type="sim",
                root=args.lerobot_root or "./data/lerobot",
                repo_id=args.lerobot_repo_id,
                split="train" if args.lerobot_eval_online else "test",
                image_size=tuple(args.lerobot_image_size),
                state_type=args.lerobot_state_type,
                include_depth=include_depth,
                cameras=lerobot_cameras,
                horizon=args.pred_horizon,
                obs_horizon=args.obs_horizon,
                tolerance_s=args.lerobot_tolerance_s,
                depth_mode=args.lerobot_depth_mode,
                video_backend=args.lerobot_video_backend,
                task_description_file=args.lerobot_task_description_file,
            )
            lerobot_dataset = build_lerobot_dataset(lerobot_config)
            lerobot_sim_dataset = lerobot_dataset
            action_denorm = {
                "normalizer": lerobot_dataset.normalizer,
                "dataset_idx": 0,
                "method": lerobot_dataset.config.normalization_method,
                "act_dim": lerobot_dataset.action_dim,
            }

        if args.lerobot_eval_online and args.lerobot_use_eval_processor:
            from examples.baselines.lerobot_dataset.evaluate_processor import (
                HumanVideoSimEvaluateProcessor,
                HumanVideoSimEvaluateProcessorConfig,
            )

            eval_processor = HumanVideoSimEvaluateProcessor(
                HumanVideoSimEvaluateProcessorConfig(
                    human_root=args.lerobot_human_root or args.lerobot_root or "demos",
                    human_split="train",
                    human_dataset_file=args.lerobot_human_dataset_file,
                    human_task_description_file=args.lerobot_human_task_description_file,
                    human_cameras=["zed2i"],
                    human_include_depth=include_depth,
                    human_image_size=tuple(args.lerobot_image_size),
                    human_video_backend=args.lerobot_video_backend,
                    sim_root=args.lerobot_sim_root or args.lerobot_root or "demos",
                    sim_split="train",
                    sim_dataset_file=args.lerobot_sim_dataset_file,
                    sim_task_description_file=args.lerobot_sim_task_description_file,
                    sim_state_type=args.lerobot_state_type,
                    sim_single_arm=False,
                    normalization_method=lerobot_sim_dataset.config.normalization_method,
                    task_mapping_file=args.lerobot_task_mapping_file,
                )
            )

        def _load_paired_eval_prompts_for_env(sim_task_id: str) -> list[str]:
            if eval_processor is not None:
                return eval_processor.get_language(
                    sim_task_id,
                    num_envs=max(args.num_eval_envs, args.num_eval_episodes),
                )

            mapping_path = args.lerobot_task_mapping_file
            desc_path = (
                args.lerobot_human_task_description_file
                or args.lerobot_task_description_file
            )
            if not mapping_path or not desc_path:
                return []

            try:
                with open(mapping_path, "r") as f:
                    mapping_data = json.load(f)
                with open(desc_path, "r") as f:
                    desc_data = json.load(f)
            except Exception:
                return []

            human_task_id = None
            for mapping in mapping_data.get("task_mappings", []):
                if sim_task_id in mapping.get("sim_task_id", []):
                    human_task_id = mapping.get("human_task_id")
                    break

            if human_task_id is None:
                return []

            prompts = desc_data.get(human_task_id, [])
            if isinstance(prompts, str):
                prompts = [prompts]
            return [p for p in prompts if isinstance(p, str) and p]

        if args.lerobot_use_paired_dataset and args.lerobot_eval_online:
            paired_prompts = _load_paired_eval_prompts_for_env(sim_task_id_for_mapping)

        if args.lerobot_eval_online:
            if eval_processor is not None:
                try:
                    dataset_idx_for_eval = eval_processor._resolve_dataset_idx(sim_task_id_for_mapping)
                except Exception:
                    dataset_idx_for_eval = 0
            dtw_dataset_idx = dataset_idx_for_eval
            dtw_provider, traj_metrics = _build_trajectory_metrics(
                args,
                normalizer=lerobot_sim_dataset.normalizer,
            )

    # Setup environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)

    if args.use_lerobot and not args.lerobot_eval_online:
        num_views = 1
        image_h, image_w = args.lerobot_image_size

        class _DummyVecEnv:
            def __init__(self, obs_horizon, state_dim, act_dim):
                self.single_observation_space = spaces.Dict(
                    {
                        "state": spaces.Box(
                            -float("inf"),
                            float("inf"),
                            shape=(obs_horizon, state_dim),
                            dtype=np.float32,
                        ),
                        "rgb": spaces.Box(
                            0,
                            255,
                            shape=(obs_horizon, image_h, image_w, 3 * num_views),
                            dtype=np.uint8,
                        ),
                    }
                )
                self.single_action_space = spaces.Box(
                    -1.0,
                    1.0,
                    shape=(lerobot_dataset.action_dim,),
                    dtype=np.float32,
                )

            def close(self):
                pass

        envs = _DummyVecEnv(
            args.obs_horizon, lerobot_sim_dataset.state_dim, lerobot_sim_dataset.action_dim
        )
    else:
        from pathlib import Path
        video_dir = args.video_dir if args.video_dir else str(Path(args.checkpoint_path).parent.parent / "videos")
        wrappers = [FlattenRGBDObservationWrapper]
        if args.use_lerobot and args.lerobot_eval_online:
            if eval_processor is None:
                try:
                    for idx, info in lerobot_sim_dataset.normalizer.dataset_info.items():
                        if info.get("repo_id", "").endswith(args.env_id):
                            dataset_idx_for_eval = idx
                            break
                except Exception:
                    pass
            wrappers = [
                lambda env: FlattenRGBDSelectWrapper(
                    env,
                    ["zed2i"],
                    state_type=args.lerobot_state_type,
                    expected_state_dim=lerobot_sim_dataset.state_dim,
                    state_normalizer=lerobot_sim_dataset.normalizer,
                    normalization_method=lerobot_sim_dataset.config.normalization_method,
                    dataset_idx=dataset_idx_for_eval,
                    image_size=tuple(args.lerobot_image_size),
                    depth=include_depth,
                    debug=args.lerobot_debug,
                    processor=eval_processor,
                    sim_task_id=sim_task_id_for_mapping,
                ),
                FlattenActionSpaceWrapper,
            ]
            if action_denorm is not None:
                wrappers.append(
                    lambda env: ActionDenormalizeWrapper(
                        env,
                        action_denorm["normalizer"],
                        dataset_idx=dataset_idx_for_eval,
                        method=action_denorm["method"],
                        act_dim=action_denorm["act_dim"],
                        debug=args.lerobot_debug,
                        processor=eval_processor,
                        sim_task_id=sim_task_id_for_mapping,
                    )
                )

        envs = make_eval_envs(
            eval_env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs,
            video_dir=video_dir if args.capture_video else None,
            wrappers=wrappers,
            l_level=requested_level,
            lr_mirror_enabled=lr_mirror_enabled,
            lr_mirror_robot_pose_enabled=lr_mirror_robot_pose_enabled,
        )
        if action_denorm is not None:
            action_space = gym.spaces.Box(
                -1.0, 1.0, shape=(action_denorm["act_dim"],), dtype=np.float32
            )
            envs.single_action_space = action_space
            envs.action_space = action_space
            if hasattr(envs, "envs") and envs.envs:
                for sub_env in envs.envs:
                    if getattr(sub_env, "single_action_space", None) is None:
                        sub_env.single_action_space = action_space
        if getattr(envs, "single_observation_space", None) is None:
            envs.single_observation_space = envs.observation_space

    # Setup precomputed language embeddings (optional)
    precomputed_lang = None
    if args.use_precomputed_lang and args.precomputed_lang_dir:
        precomputed_lang = HybridLangEmbedding(
            precomputed_dir=args.precomputed_lang_dir,
            text_encoder_path=args.text_encoder,
            device=device,
            max_length=args.max_lang_len,
            t5_version=args.t5_version,
        )
        print(f"Precomputed language embedding stats: {precomputed_lang.get_stats()}")

    print(f"\n{'='*60}")
    print(f"Evaluation Setup")
    print(f"{'='*60}")
    print(f"Environment: {args.env_id}")
    print(f"Resolved ManiSkill env: {eval_env_id}")
    print(f"L-level: {requested_level or 'default'}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Num episodes: {args.num_eval_episodes}")
    print(f"Num envs: {args.num_eval_envs}")
    print(f"Use precomputed lang: {args.use_precomputed_lang}")
    print(f"{'='*60}\n")

    # Create agent
    dummy_train_args = TrainArgs(
        env_id=args.env_id,
        demo_path="",  # Not needed for evaluation
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        vision_encoder=args.vision_encoder,
        text_encoder=args.text_encoder,
        t5_version=args.t5_version,
        max_lang_len=args.max_lang_len,
        num_diffusion_iters=args.num_diffusion_iters,
        num_inference_steps=args.num_inference_steps,
        use_precomputed_lang=args.use_precomputed_lang,
        precomputed_lang_dir=args.precomputed_lang_dir,
        use_dummy_language=args.use_dummy_language,
        use_lerobot=args.use_lerobot,
    )

    print("Creating RDT agent...")
    agent = RDTAgent(dummy_train_args, envs, device, precomputed_lang)

    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint_path, agent, device)

    print(f"\nEvaluation Configuration:")
    print(f"  Language source: Environment prompts")
    print(f"  Precomputed embeddings: {args.use_precomputed_lang}")

    # Run evaluation
    print(f"\n{'='*60}")
    print("Starting Evaluation...")
    print(f"{'='*60}")

    start_time = time.time()
    if args.use_lerobot and not args.lerobot_eval_online:
        from torch.utils.data.dataloader import DataLoader

        def _lerobot_default_language(text):
            if text is None:
                return "pick red cube and place on plate."
            if isinstance(text, str) and text.strip() == "":
                return "pick red cube and place on plate."
            if isinstance(text, str) and text.strip().lower() == "none":
                return "pick red cube and place on plate."
            return text

        def collate_fn_lerobot(batch):
            if args.lerobot_use_paired_dataset:
                states = torch.stack([item["robot_obs"]["states"] for item in batch]).to(device)
                actions = torch.stack([item["robot_actions"] for item in batch]).to(device)
                view_keys = sorted(
                    [k for k in batch[0]["robot_obs"].keys() if k.startswith("view_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                view_key = view_keys[0] if view_keys else None
                rgb = None
                if view_key is not None:
                    view = torch.stack([item["robot_obs"][view_key] for item in batch]).to(device)
                    if view.dim() == 5:
                        rgb = view[:, :, :3]
                    else:
                        rgb = view
                language = [_lerobot_default_language(item.get("language")) for item in batch]
            else:
                states = torch.stack([item["states"] for item in batch]).to(device)
                actions = torch.stack([item["actions"] for item in batch]).to(device)
                view_keys = sorted(
                    [k for k in batch[0].keys() if k.startswith("view_")],
                    key=lambda x: int(x.split("_")[1]),
                )
                view_key = view_keys[0] if view_keys else None
                rgb = None
                if view_key is not None:
                    view = torch.stack([item[view_key] for item in batch]).to(device)
                    rgb = view[:, :, :3]
                language = [_lerobot_default_language(item.get("task_descriptions")) for item in batch]
            return {"states": states, "actions": actions, "rgb": rgb, "language": language}

        lerobot_loader = DataLoader(
            lerobot_dataset,
            batch_size=args.lerobot_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_lerobot,
            multiprocessing_context="forkserver" if args.num_dataload_workers > 0 else None,
        )
        mse_list = []
        for idx, batch in enumerate(lerobot_loader):
            if idx >= args.lerobot_eval_batches:
                break
            obs = {"state": batch["states"]}
            if batch["rgb"] is not None:
                rgb = batch["rgb"]
                if rgb.shape[-1] <= 4:
                    obs["rgb"] = rgb
                else:
                    obs["rgb"] = rgb.permute(0, 1, 3, 4, 2)
            pred_actions = agent.get_action(obs, language_prompt=batch["language"])
            mse = torch.mean((pred_actions - batch["actions"]) ** 2).item()
            mse_list.append(mse)
        eval_time = time.time() - start_time
        mean_mse = float(np.mean(mse_list)) if mse_list else 0.0
        print(f"[lerobot] action_mse: {mean_mse:.6f}")
        results = {"mean_lerobot_action_mse": mean_mse}
        eval_metrics = {}
    else:
        eval_metrics = evaluate(
            n=args.num_eval_episodes,
            agent=agent,
            eval_envs=envs,
            device=device,
            sim_backend=args.sim_backend,
            progress_bar=True,
            precomputed_lang=precomputed_lang,
            language_prompts=paired_prompts,
            gripper_binary_action=args.gripper_binary_action,
            gripper_threshold=args.gripper_threshold,
            gripper_indices=list(args.gripper_indices),
            env_id=sim_task_id_for_mapping,
            dtw_provider=dtw_provider,
            traj_metrics=traj_metrics,
            dtw_dataset_idx=dtw_dataset_idx,
            dtw_normalize_gt=bool(dtw_provider is not None),
        )
        eval_time = time.time() - start_time

        reduced = {k: float(np.mean(v)) for k, v in eval_metrics.items()}
        if args.compute_dtw:
            for key in ("tss_success", "tss_fail", "ndtw_success", "ndtw_fail"):
                reduced.setdefault(key, None)

        # Process results
        results = {}
        for k, v in eval_metrics.items():
            if k in ['success_at_end', 'success_once']:
                results[k] = v.tolist()
            else:
                results[f'mean_{k}'] = float(np.mean(v))
                if k == 'reward':
                    results[f'std_{k}'] = float(np.std(v))

        # Print results
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")

        if 'reward' in eval_metrics:
            print(f"Mean Reward:        {results['mean_reward']:>8.2f} ± {results['std_reward']:.2f}")
        if 'length' in eval_metrics:
            print(f"Mean Episode Length:{results['mean_length']:>8.1f}")
        if 'success_once' in eval_metrics:
            print(f"Success Once:       {np.mean(eval_metrics['success_once'])*100:>8.1f}%")
        if args.compute_dtw:
            print("Trajectory Similarity:")
            tss_success = reduced.get("tss_success")
            tss_fail = reduced.get("tss_fail")
            print(f"  TSS_success:      {tss_success:.4f}" if tss_success is not None else "  TSS_success:      N/A")
            print(f"  TSS_fail:         {tss_fail:.4f}" if tss_fail is not None else "  TSS_fail:         N/A")
    if 'success_at_end' in eval_metrics:
        print(f"Success at End:     {np.mean(eval_metrics['success_at_end'])*100:>8.1f}%")

    print(f"{'='*60}")
    print(f"Evaluation time:    {eval_time:>8.1f}s")
    print(f"Episodes evaluated: {args.num_eval_episodes:>8d}")
    print(f"{'='*60}\n")

    # Save results
    if args.save_results:
        # Determine results directory
        if args.output_dir:
            results_dir = args.output_dir
        elif args.video_dir:
            results_dir = str(Path(args.video_dir).parent)
        elif os.path.isfile(args.checkpoint_path):
            results_dir = os.path.dirname(args.checkpoint_path)
        else:
            results_dir = args.checkpoint_path

        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "eval_metrics.json")

        # Add metadata
        results_with_metadata = {
            "env_id": args.env_id,
            "checkpoint_path": args.checkpoint_path,
            "num_eval_episodes": args.num_eval_episodes,
            "num_eval_envs": args.num_eval_envs,
            "compute_dtw": args.compute_dtw,
            "dtw_band_ratio": args.dtw_band_ratio if args.compute_dtw else None,
            "tss_success_mean": reduced.get("tss_success") if eval_metrics else None,
            "tss_fail_mean": reduced.get("tss_fail") if eval_metrics else None,
            "metrics_mean": reduced if eval_metrics else {},
            "metrics_raw_shape": {
                k: list(v.shape) for k, v in eval_metrics.items() if isinstance(v, np.ndarray)
            },
            "eval_metrics": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in eval_metrics.items()
            },
            "lerobot_results": results if (args.use_lerobot and not args.lerobot_eval_online) else None,
            "config": {
                "use_precomputed_lang": args.use_precomputed_lang,
                "eval_time": eval_time,
            }
        }

        with open(results_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)

        print(f"Results saved to: {results_file}")

    envs.close()
    print("\n✓ Evaluation completed!")
