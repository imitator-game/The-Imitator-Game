#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import examples.baselines.pi.src.openpi.shared.nnx_utils as nnx_utils
from examples.baselines.lerobot_dataset.evaluate_processor import (
    HumanVideoSimEvaluateProcessor,
    HumanVideoSimEvaluateProcessorConfig,
)
from examples.baselines.lerobot_dataset.lerobot_paired_dataset import TaskMapper
from examples.baselines.lerobot_dataset.normalizer import ActionNormalizer


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _torch_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _clean_prompt(text: str) -> str:
    return text.strip().replace("_", " ").replace("\n", " ")


def _format_pi05_prompt(text: str, state: Any) -> str:
    state_np = _torch_to_numpy(state).astype(np.float32).reshape(-1)
    discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(str(int(x)) for x in discretized_state)
    return f"Task: {_clean_prompt(text)}, State: {state_str};\nAction: "


def tokenize_pi_prompts(
    processor,
    prompts: list[str],
    *,
    max_token_len: int,
    pi05: bool,
    states: Any = None,
    return_tensors: str = "np",
):
    if pi05:
        if states is None:
            raise ValueError("Pi05 prompt tokenization requires normalized state.")
        states_np = _torch_to_numpy(states).astype(np.float32)
        if states_np.ndim == 1:
            states_np = states_np[None]
        prompts = [_format_pi05_prompt(prompt, state) for prompt, state in zip(prompts, states_np)]
    else:
        prompts = [_clean_prompt(prompt) for prompt in prompts]
    return processor.tokenizer(
        prompts,
        max_length=max_token_len,
        padding="max_length",
        truncation=True,
        return_tensors=return_tensors,
    )


def canonicalize_pi_images(
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    *,
    pad_missing: bool = True,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    canonical_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    source_keys = list(images.keys())
    if not source_keys:
        raise ValueError("No camera images were provided to build a Pi observation.")

    first_image = images[source_keys[0]]
    batch_size = first_image.shape[0]
    zero_image = np.zeros_like(first_image)
    false_mask = np.zeros(batch_size, dtype=bool)

    mapped_images: dict[str, np.ndarray] = {}
    mapped_masks: dict[str, np.ndarray] = {}
    target_keys = canonical_keys if pad_missing else canonical_keys[: min(len(source_keys), len(canonical_keys))]
    for index, canonical_key in enumerate(target_keys):
        if index < len(source_keys):
            source_key = source_keys[index]
            mapped_images[canonical_key] = images[source_key]
            mapped_masks[canonical_key] = masks[source_key]
        else:
            mapped_images[canonical_key] = zero_image.copy()
            mapped_masks[canonical_key] = false_mask.copy()
    return mapped_images, mapped_masks


def _extract_dataset_idx(item: Dict[str, Any]) -> int:
    value = item.get("dataset_idx", 0)
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def _state_action_keys(state_type: str) -> tuple[str, str]:
    if state_type == "eepos":
        return "observation.eepos_gripper_states", "action.eepos_gripper_actions"
    if state_type == "qpos":
        return "observation.qpos_gripper_states", "action.qpos_gripper_actions"
    if state_type == "mixpos":
        return "observation.eepos_gripper_states", "action.qpos_gripper_actions"
    raise ValueError(f"Unknown state_type: {state_type}")


def build_robot_normalizer(args: argparse.Namespace) -> tuple[ActionNormalizer, dict[str, int]]:
    from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDatasetMetadata

    state_key, action_key = _state_action_keys(args.robot_state_type)
    normalizer = ActionNormalizer()
    repo_id_to_dataset_idx: dict[str, int] = {}
    with open(args.robot_config, "r") as f:
        dataset_configs = json.load(f)

    for idx, ds_cfg in enumerate(dataset_configs):
        repo_id = ds_cfg["repo_id"]
        ds_root = os.path.join(args.robot_root, ds_cfg["root"])
        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=ds_root)
        normalizer.add_dataset_stats(
            dataset_idx=idx,
            repo_id=repo_id,
            stats=meta.stats,
            state_key=state_key,
            action_key=action_key,
            single_arm=args.single_arm,
        )
        repo_id_to_dataset_idx[repo_id] = idx
    return normalizer, repo_id_to_dataset_idx


def _choose_description(desc: Any) -> Optional[str]:
    if isinstance(desc, list):
        if not desc:
            return None
        return str(desc[np.random.randint(0, len(desc))])
    if desc is None:
        return None
    return str(desc)


class TaskDescriptionResolver:
    def __init__(self, args: argparse.Namespace):
        self.mapper = TaskMapper(
            args.task_mapping,
            args.human_task_desc,
            args.robot_task_desc,
            args.sim_task_desc,
        )
        self.evaluate_processor: Optional[HumanVideoSimEvaluateProcessor] = None
        if args.use_evaluate_processor_for_language:
            cfg = HumanVideoSimEvaluateProcessorConfig(
                human_root=args.human_root,
                human_split=args.human_split,
                human_dataset_file=args.human_config,
                human_task_description_file=args.human_task_desc,
                human_cameras=args.cameras,
                human_include_depth=args.include_depth,
                human_num_frames=args.human_num_frames,
                human_image_size=tuple(args.image_size),
                human_sampling_strategy=args.human_sampling_strategy,
                human_video_backend=args.video_backend,
                human_fps=args.fps,
                sim_root=args.sim_root,
                sim_split=args.sim_split,
                sim_dataset_file=args.sim_config,
                sim_task_description_file=args.sim_task_desc,
                sim_state_type=args.robot_state_type,
                sim_single_arm=args.single_arm,
                task_mapping_file=args.task_mapping,
            )
            self.evaluate_processor = HumanVideoSimEvaluateProcessor(cfg)

    def get(self, key: str) -> str:
        if self.evaluate_processor is not None:
            return self.evaluate_processor.get_task_description(key)

        human_task_id = self.mapper.get_human_task_from_robot(key)
        if human_task_id is None:
            human_task_id = self.mapper.get_human_task_from_sim(key)
        if human_task_id is not None:
            language = _choose_description(self.mapper.human_descriptions.get(human_task_id))
            if language:
                return language

        language = _choose_description(self.mapper.robot_descriptions.get(key))
        if language:
            return language
        language = _choose_description(self.mapper.sim_descriptions.get(key))
        if language:
            return language
        return f"Task {human_task_id or key}"


class PiJaxRealRobotAgent:
    def __init__(self, args: argparse.Namespace):
        from transformers import AutoProcessor
        from examples.baselines.pi.eval_pi_lerobot_jax import (
            _load_model as load_pi_jax_model,
        )

        self.args = args
        self.processor = AutoProcessor.from_pretrained(
            args.processor_name_or_path,
            trust_remote_code=True,
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
            local_files_only=args.processor_local_files_only,
        )
        self.model, self.restored_step = load_pi_jax_model(args)
        self.sample_actions = nnx_utils.module_jit(self.model.sample_actions)
        self.rng = jax.random.key(args.seed)
        self.language: str = args.default_prompt

    def prepare_for_eval(self, *, language: str) -> None:
        self.language = language

    def _build_observation(
        self,
        *,
        state: torch.Tensor,
        camera_images: Dict[str, torch.Tensor],
        language: Optional[str] = None,
    ):
        from examples.baselines.pi.src.openpi.models import model as model_lib

        if state.dim() == 1:
            state = state.view(1, 1, -1)
        elif state.dim() == 2:
            state = state.unsqueeze(0)
        state = state.float()
        prompt_state = state[:, -1] if state.ndim > 2 else state
        prompts = [language or self.language]
        tokens = tokenize_pi_prompts(
            self.processor,
            prompts,
            max_token_len=self.args.max_token_len,
            pi05=self.args.pi05,
            states=prompt_state if self.args.pi05 else None,
            return_tensors="pt",
        )

        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.ndarray] = {}
        for cam_name, img in camera_images.items():
            if img.dim() == 3:
                img = img.unsqueeze(0).unsqueeze(0)
            elif img.dim() == 4:
                img = img.unsqueeze(0)
            if img.dim() != 5:
                raise ValueError(f"Expected image tensor for {cam_name} to be 3D/4D/5D, got {tuple(img.shape)}")

            frame = _torch_to_numpy(img[:, -1]).astype(np.float32)
            if frame.shape[1] in (3, 4):
                frame = np.transpose(frame[:, :3], (0, 2, 3, 1))
            elif frame.shape[-1] > 3:
                frame = frame[..., :3]

            if frame.max() <= 1.0:
                frame = frame * 2.0 - 1.0
            else:
                frame = frame / 255.0 * 2.0 - 1.0
            images[cam_name] = frame.astype(np.float32)
            image_masks[cam_name] = np.ones(frame.shape[0], dtype=bool)

        images, image_masks = canonicalize_pi_images(
            images,
            image_masks,
            pad_missing=not self.args.skip_masked_cameras,
        )
        obs_dict = {
            "image": images,
            "image_mask": image_masks,
            "state": _torch_to_numpy(state[:, -1]).astype(np.float32),
            "tokenized_prompt": _torch_to_numpy(tokens["input_ids"]).astype(np.int32),
            "tokenized_prompt_mask": _torch_to_numpy(tokens["attention_mask"]).astype(bool),
        }
        return model_lib.Observation.from_dict(obs_dict)

    def get_action_from_item(self, item: Dict[str, Any], language: Optional[str] = None) -> torch.Tensor:
        camera_images = {}
        for i in range(1, len(self.args.cameras) + 1):
            key = f"view_{i}"
            if key in item:
                camera_images[key] = item[key]
        observation = self._build_observation(
            state=item["states"],
            camera_images=camera_images,
            language=language,
        )
        self.rng, sample_rng = jax.random.split(self.rng)
        actions = self.sample_actions(sample_rng, observation, num_steps=self.args.num_diffusion_steps)
        return torch.from_numpy(np.asarray(jax.device_get(actions), dtype=np.float32))

    def get_action_from_stream_obs(
        self,
        *,
        state: torch.Tensor,
        camera_images: Dict[str, torch.Tensor],
        language: Optional[str] = None,
    ) -> torch.Tensor:
        observation = self._build_observation(
            state=state,
            camera_images=camera_images,
            language=language,
        )
        self.rng, sample_rng = jax.random.split(self.rng)
        actions = self.sample_actions(sample_rng, observation, num_steps=self.args.num_diffusion_steps)
        return torch.from_numpy(np.asarray(jax.device_get(actions), dtype=np.float32))


def get_repo_indices(robot_ds, repo_id: str, max_frames: int) -> list[int]:
    prev = 0
    for ds_idx, curr_ds in enumerate(robot_ds.lerobot_dataset.datasets):
        end = robot_ds.lerobot_dataset.cumulative_sizes[ds_idx]
        if curr_ds.repo_id == repo_id:
            return list(range(prev, min(end, prev + max_frames)))
        prev = end
    return list(range(min(max_frames, len(robot_ds))))


def run_mock(args: argparse.Namespace) -> None:
    from examples.baselines.lerobot_dataset.lerobot_robot_dataset import (
        LeRobotRobotDataConfig,
        LeRobotRobotDataset,
    )

    agent = PiJaxRealRobotAgent(args)
    robot_normalizer, _ = build_robot_normalizer(args)
    desc_resolver = TaskDescriptionResolver(args)
    debug_dir = Path(args.mock_debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    robot_ds = LeRobotRobotDataset(
        LeRobotRobotDataConfig(
            root=args.robot_root,
            split=args.robot_split,
            image_size=tuple(args.image_size),
            state_type=args.robot_state_type,
            include_depth=args.include_depth,
            cameras=args.cameras,
            single_arm=args.single_arm,
            horizon=args.pred_horizon,
            obs_horizon=1,
            dataset_file=args.robot_config,
            task_description_file=args.robot_task_desc,
            normalization_method=args.normalization_method,
            enable_augmentation=False,
        )
    )

    repo_id = args.mock_repo_id
    frame_indices = get_repo_indices(robot_ds, repo_id, args.mock_num_frames)
    if not frame_indices:
        raise RuntimeError(f"No frames available for mock repo_id={repo_id}")

    language_key = args.mock_language_key or repo_id
    language = desc_resolver.get(language_key)
    agent.prepare_for_eval(language=language)

    pred_actions: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    per_step_mse: list[float] = []
    per_dim_sq: list[np.ndarray] = []
    dataset_indices: list[int] = []

    print(f"[PI-JAX-MOCK] checkpoint={args.checkpoint_path}")
    print(f"[PI-JAX-MOCK] restored_step={agent.restored_step}")
    print(f"[PI-JAX-MOCK] repo_id={repo_id} frames={len(frame_indices)}")
    print(f"[PI-JAX-MOCK] language={language[:240]}")

    with torch.no_grad():
        for step, frame_idx in enumerate(frame_indices):
            item = robot_ds[frame_idx]
            dataset_idx = _extract_dataset_idx(item)
            action_seq = agent.get_action_from_item(item, language=language)
            if action_seq.dim() == 2:
                action_seq = action_seq.unsqueeze(0)

            pred_norm = action_seq[:, 0]
            gt_norm = item["actions"][0:1]
            pred_den = robot_normalizer.denormalize_action(
                pred_norm, dataset_idx=dataset_idx, method=args.normalization_method
            )
            gt_den = robot_normalizer.denormalize_action(
                gt_norm, dataset_idx=dataset_idx, method=args.normalization_method
            )

            p = pred_den[0].cpu().numpy()
            g = gt_den[0].cpu().numpy()
            sq_err = (p - g) ** 2
            mse = float(sq_err.mean())

            pred_actions.append(p)
            gt_actions.append(g)
            per_step_mse.append(mse)
            per_dim_sq.append(sq_err)
            dataset_indices.append(dataset_idx)

            if step < 3 or step % args.mock_log_interval == 0:
                print(
                    f"[PI-JAX-MOCK] step={step:04d} dataset_idx={dataset_idx} "
                    f"mse={mse:.6f} pred[:4]={p[:4].round(4).tolist()} gt[:4]={g[:4].round(4).tolist()}"
                )

    _write_mock_report(
        pred_actions=pred_actions,
        gt_actions=gt_actions,
        per_step_mse=per_step_mse,
        per_dim_sq=per_dim_sq,
        dataset_indices=dataset_indices,
        label=repo_id,
        language=language,
        debug_dir=debug_dir,
        checkpoint_path=args.checkpoint_path,
        checkpoint_step=agent.restored_step,
    )


def _write_mock_report(
    *,
    pred_actions: list[np.ndarray],
    gt_actions: list[np.ndarray],
    per_step_mse: list[float],
    per_dim_sq: list[np.ndarray],
    dataset_indices: list[int],
    label: str,
    language: str,
    debug_dir: Path,
    checkpoint_path: str,
    checkpoint_step: int | None,
) -> None:
    preds = np.stack(pred_actions)
    gts = np.stack(gt_actions)
    mses = np.asarray(per_step_mse)
    dim_sq = np.stack(per_dim_sq)
    n, action_dim = preds.shape

    csv_path = debug_dir / "per_step_mse.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "dataset_idx", "mse"] + [f"dim_{i}_sq_err" for i in range(action_dim)])
        for i, (dataset_idx, mse, sq) in enumerate(zip(dataset_indices, per_step_mse, per_dim_sq)):
            writer.writerow([i, dataset_idx, round(mse, 8)] + [round(float(x), 8) for x in sq])

    summary = {
        "label": label,
        "language": language,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": checkpoint_step,
        "n_frames": n,
        "action_dim": action_dim,
        "overall_mse_mean": float(mses.mean()),
        "overall_mse_std": float(mses.std()),
        "overall_mse_max": float(mses.max()),
        "per_dim_mse_mean": dim_sq.mean(axis=0).tolist(),
        "per_dim_mse_std": dim_sq.std(axis=0).tolist(),
        "timestamp": datetime.now().isoformat(),
    }
    json_path = debug_dir / "mock_debug_summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    frames = np.arange(n)
    cols = 2
    rows = (action_dim + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 2.6 * rows), squeeze=False)
    for dim in range(action_dim):
        r, c = divmod(dim, cols)
        ax = axes[r][c]
        ax.plot(frames, gts[:, dim], color="steelblue", linewidth=1.3, label="GT")
        ax.plot(frames, preds[:, dim], color="tomato", linewidth=1.0, linestyle="--", label="Pred")
        ax.set_title(f"dim {dim}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for dim in range(action_dim, rows * cols):
        r, c = divmod(dim, cols)
        axes[r][c].axis("off")
    fig.tight_layout()
    action_plot = debug_dir / "action_comparison.png"
    fig.savefig(action_plot, dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(frames, mses, color="darkorange", linewidth=1.2)
    ax2.axhline(mses.mean(), color="gray", linestyle="--", label=f"mean={mses.mean():.4f}")
    ax2.set_title(f"Pi JAX mock action MSE - {label}")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("MSE")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    mse_plot = debug_dir / "mse_over_time.png"
    fig2.tight_layout()
    fig2.savefig(mse_plot, dpi=120)
    plt.close(fig2)

    print("[PI-JAX-MOCK] report")
    print(f"  frames: {n}")
    print(f"  mse mean/std/max: {summary['overall_mse_mean']:.6f} / {summary['overall_mse_std']:.6f} / {summary['overall_mse_max']:.6f}")
    print(f"  csv: {csv_path}")
    print(f"  json: {json_path}")
    print(f"  action plot: {action_plot}")
    print(f"  mse plot: {mse_plot}")


class PiJaxRealRobotController:
    def __init__(
        self,
        *,
        agent: PiJaxRealRobotAgent,
        robot_normalizer: ActionNormalizer,
        language: str,
        args: argparse.Namespace,
    ):
        import albumentations as A
        import rospy
        from albumentations.pytorch import ToTensorV2
        from cv_bridge import CvBridge

        self.agent = agent
        self.robot_normalizer = robot_normalizer
        self.language = language
        self.args = args
        self.rospy = rospy
        self.bridge = CvBridge()
        self.camera_names = list(args.cameras)
        self.current_rgbs = {cam: None for cam in self.camera_names}
        self.current_depths = {cam: None for cam in self.camera_names}
        self.current_joint_state = None
        self.current_gripper_state = None
        self.static_left_arm = None
        self.static_left_gripper = None
        self.last_grippers = np.zeros(2, dtype=np.float32)
        self.ts = 0
        self.action_buf: Optional[torch.Tensor] = None
        self.tagg_buf: Optional[torch.Tensor] = None
        if args.temporal_agg:
            self.tagg_buf = torch.zeros(
                1,
                args.max_episode_steps,
                args.max_episode_steps + args.pred_horizon,
                args.action_dim,
                dtype=torch.float32,
            )

        self.joint_names = [
            "r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6", "r_joint7",
            "l_joint1", "l_joint2", "l_joint3", "l_joint4", "l_joint5", "l_joint6", "l_joint7",
        ]
        self.gripper_names = ["gripper_1", "gripper_0"]

        h, w = tuple(args.image_size)
        self.geom_tf = A.Compose([A.Resize(height=h, width=w, p=1.0)])
        self.rgb_tf = A.Compose([
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ])
        self.depth_tf = A.Compose([ToTensorV2()])

        rospy.init_node("eval_pi_jax_real_robot")
        self._setup_ros()
        self.agent.prepare_for_eval(language=language)
        self.timer = rospy.Timer(rospy.Duration(1.0 / args.control_rate), self._control_loop)
        rospy.loginfo(f"[PiJaxRealRobot] Ready rate={args.control_rate}Hz language={language[:160]}")

    def _setup_ros(self) -> None:
        import rospy
        from sensor_msgs.msg import Image, JointState

        self.joint_pub = rospy.Publisher("/io_teleop/joint_cmd", JointState, queue_size=1)
        self.gripper_pub = rospy.Publisher("/io_teleop/target_gripper_status", JointState, queue_size=1)
        rospy.Subscriber("/io_teleop/joint_states", JointState, self._joint_cb)
        rospy.Subscriber("/io_teleop/gripper_states", JointState, self._gripper_cb)
        for cam in self.camera_names:
            rospy.Subscriber(f"/{cam}/{cam}/color/image_raw", Image, lambda msg, c=cam: self._rgb_cb(msg, c))
            if self.args.include_depth:
                rospy.Subscriber(
                    f"/{cam}/{cam}/aligned_depth_to_color/image_raw",
                    Image,
                    lambda msg, c=cam: self._depth_cb(msg, c),
                )

    def _joint_cb(self, msg) -> None:
        self.current_joint_state = list(msg.position)
        if self.static_left_arm is None and len(msg.position) >= 14:
            self.static_left_arm = list(msg.position[7:14])

    def _gripper_cb(self, msg) -> None:
        self.current_gripper_state = list(msg.position)
        if self.static_left_gripper is None and len(msg.position) >= 2:
            self.static_left_gripper = float(msg.position[1])

    def _rgb_cb(self, msg, cam: str) -> None:
        import cv2

        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.current_rgbs[cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _depth_cb(self, msg, cam: str) -> None:
        raw_m = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) / 1000.0
        self.current_depths[cam] = raw_m

    def _obs_ready(self) -> bool:
        if self.current_joint_state is None or self.current_gripper_state is None:
            return False
        for cam in self.camera_names:
            if self.current_rgbs[cam] is None:
                return False
            if self.args.include_depth and self.current_depths[cam] is None:
                return False
        return True

    def _build_raw_qpos_state(self) -> torch.Tensor:
        joints = np.asarray(self.current_joint_state, dtype=np.float32)
        grippers = np.asarray(self.current_gripper_state, dtype=np.float32)
        if joints.shape[0] < 14 or grippers.shape[0] < 2:
            raise RuntimeError("Expected 14 joint positions and 2 gripper positions for qpos real-robot Pi eval.")
        raw = np.concatenate([joints[:7], joints[7:14], [grippers[0], grippers[1]]], axis=0)
        state = torch.from_numpy(raw).view(1, 1, -1).float()
        state = self.robot_normalizer.normalize_state(
            state,
            dataset_idx=self.args.real_dataset_idx,
            method=self.args.normalization_method,
        )
        return state

    def _build_camera_images(self) -> Dict[str, torch.Tensor]:
        camera_images: dict[str, torch.Tensor] = {}
        for i, cam in enumerate(self.camera_names, start=1):
            rgb = self.geom_tf(image=self.current_rgbs[cam])["image"]
            rgb_t = self.rgb_tf(image=rgb)["image"]
            if self.args.include_depth:
                depth = self.current_depths[cam].astype(np.float32)[..., None]
                depth = self.geom_tf(image=depth)["image"]
                depth_t = self.depth_tf(image=depth)["image"]
                view = torch.cat([rgb_t, depth_t], dim=0)
            else:
                view = rgb_t
            camera_images[f"view_{i}"] = view.unsqueeze(0).unsqueeze(0)
        return camera_images

    def _threshold_grippers(self, action: np.ndarray) -> np.ndarray:
        if action.shape[0] >= 16:
            indices = [14, 15]
        else:
            indices = [action.shape[0] - 1]
        for slot, idx in enumerate(indices):
            prev = self.last_grippers[min(slot, len(self.last_grippers) - 1)]
            if prev < 0.5:
                new = 1.0 if action[idx] > self.args.gripper_open_threshold else 0.0
            else:
                new = 0.0 if action[idx] < self.args.gripper_close_threshold else 1.0
            action[idx] = new
            self.last_grippers[min(slot, len(self.last_grippers) - 1)] = new
        return action

    def _control_loop(self, _event) -> None:
        if not self._obs_ready():
            self.rospy.logwarn_throttle(5, "[PiJaxRealRobot] waiting for observations")
            return

        state = self._build_raw_qpos_state()
        camera_images = self._build_camera_images()
        if self.args.temporal_agg:
            with torch.no_grad():
                action_seq = self.agent.get_action_from_stream_obs(
                    state=state,
                    camera_images=camera_images,
                    language=self.language,
                )
            if action_seq.dim() == 2:
                action_seq = action_seq.unsqueeze(0)

            if self.tagg_buf is None or self.ts >= self.args.max_episode_steps:
                raw_norm = action_seq[:, 0]
            else:
                self.tagg_buf[:, self.ts, self.ts: self.ts + self.args.pred_horizon] = action_seq.cpu()
                actions_for_curr = self.tagg_buf[:, : self.ts + 1, self.ts]
                weights = torch.exp(-0.01 * torch.arange(self.ts + 1, dtype=torch.float32))
                weights = (weights / weights.sum()).view(1, self.ts + 1, 1)
                raw_norm = (actions_for_curr * weights).sum(dim=1)
        else:
            query_freq = self.args.pred_horizon
            if self.ts % query_freq == 0 or self.action_buf is None:
                with torch.no_grad():
                    action_seq = self.agent.get_action_from_stream_obs(
                        state=state,
                        camera_images=camera_images,
                        language=self.language,
                    )
                if action_seq.dim() == 2:
                    action_seq = action_seq.unsqueeze(0)
                self.action_buf = action_seq
            raw_norm = self.action_buf[:, self.ts % query_freq]

        action = self.robot_normalizer.denormalize_action(
            raw_norm,
            dataset_idx=self.args.real_dataset_idx,
            method=self.args.normalization_method,
        )[0].cpu().numpy()
        action = self._threshold_grippers(action)
        self._publish_qpos_action(action)
        self.ts += 1

    def _publish_qpos_action(self, action: np.ndarray) -> None:
        from sensor_msgs.msg import JointState

        if action.shape[0] >= 16:
            right_joints = action[:7]
            if self.args.static_left_arm and self.static_left_arm is not None:
                left_joints = np.asarray(self.static_left_arm, dtype=np.float32)
                left_gripper = float(self.static_left_gripper or 0.0)
            else:
                left_joints = action[7:14]
                left_gripper = float(action[15])
            right_gripper = float(action[14])
        elif action.shape[0] >= 8:
            right_joints = action[:7]
            left_joints = np.asarray(self.static_left_arm or [0.0] * 7, dtype=np.float32)
            right_gripper = float(action[7])
            left_gripper = float(self.static_left_gripper or 0.0)
        else:
            self.rospy.logwarn_throttle(1, f"[PiJaxRealRobot] invalid action shape {action.shape}")
            return

        joint_msg = JointState()
        joint_msg.header.stamp = self.rospy.Time.now()
        joint_msg.name = self.joint_names
        joint_msg.position = list(right_joints) + list(left_joints)
        self.joint_pub.publish(joint_msg)

        grip_msg = JointState()
        grip_msg.header.stamp = self.rospy.Time.now()
        grip_msg.name = self.gripper_names
        grip_msg.position = [right_gripper, left_gripper]
        self.gripper_pub.publish(grip_msg)


def run_real_robot(args: argparse.Namespace) -> None:
    import rospy

    agent = PiJaxRealRobotAgent(args)
    robot_normalizer, repo_to_idx = build_robot_normalizer(args)
    desc_resolver = TaskDescriptionResolver(args)
    language_key = args.real_language_key or args.real_repo_id or args.real_env_id
    language = args.real_language or desc_resolver.get(language_key)
    if args.real_repo_id and args.real_repo_id in repo_to_idx and args.real_dataset_idx is None:
        args.real_dataset_idx = repo_to_idx[args.real_repo_id]
    if args.real_dataset_idx is None:
        args.real_dataset_idx = 0
    PiJaxRealRobotController(
        agent=agent,
        robot_normalizer=robot_normalizer,
        language=language,
        args=args,
    )
    rospy.loginfo("[PiJaxRealRobot] spinning. Press Ctrl-C to stop.")
    rospy.spin()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pi0.5 JAX real-robot eval and LeRobot robot mock test.")
    parser.add_argument("--checkpoint", "--checkpoint_path", dest="checkpoint_path", required=True)
    parser.add_argument("--checkpoint-step", "--checkpoint_step", dest="checkpoint_step", type=int, default=None)
    parser.add_argument("--processor-name-or-path", "--processor_name_or_path", dest="processor_name_or_path", default="google/paligemma-3b-pt-224")
    parser.add_argument("--processor-local-files-only", "--processor_local_files_only", dest="processor_local_files_only", action="store_true")
    parser.add_argument("--pi05", action="store_true", default=True)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--use-lora", "--use_lora", dest="use_lora", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--robot-root", "--robot_root", dest="robot_root", required=True)
    parser.add_argument("--robot-config", "--robot_config", dest="robot_config", required=True)
    parser.add_argument("--robot-task-desc", "--robot_task_desc", dest="robot_task_desc", default="examples/baselines/lerobot_dataset/task_desc/robot_desc.json")
    parser.add_argument("--robot-state-type", "--robot_state_type", dest="robot_state_type", default="qpos", choices=["qpos", "eepos", "mixpos"])
    parser.add_argument("--robot-split", "--robot_split", dest="robot_split", default="train")

    parser.add_argument("--human-root", "--human_root", dest="human_root", default="demos/demo_data")
    parser.add_argument("--human-config", "--human_config", dest="human_config", default="examples/baselines/lerobot_dataset/config/exp_configs/human_test_config_seen.json")
    parser.add_argument("--human-task-desc", "--human_task_desc", dest="human_task_desc", default="examples/baselines/lerobot_dataset/task_desc/human_desc.json")
    parser.add_argument("--human-split", "--human_split", dest="human_split", default="train")
    parser.add_argument("--human-num-frames", "--human_num_frames", dest="human_num_frames", type=int, default=10)
    parser.add_argument("--human-sampling-strategy", "--human_sampling_strategy", dest="human_sampling_strategy", default="uniform_jitter")

    parser.add_argument("--sim-root", "--sim_root", dest="sim_root", default="demos/imitator_data")
    parser.add_argument("--sim-config", "--sim_config", dest="sim_config", default="examples/baselines/lerobot_dataset/config/exp_configs/sim_test_config_seen.json")
    parser.add_argument("--sim-task-desc", "--sim_task_desc", dest="sim_task_desc", default="examples/baselines/lerobot_dataset/task_desc/sim_desc.json")
    parser.add_argument("--sim-split", "--sim_split", dest="sim_split", default="train")
    parser.add_argument("--task-mapping", "--task_mapping", dest="task_mapping", default="examples/baselines/lerobot_dataset/task_mapping.json")
    parser.add_argument("--use-evaluate-processor-for-language", action="store_true", help="Load HumanVideoSimEvaluateProcessor for language lookup instead of using TaskMapper JSON only.")

    parser.add_argument("--cameras", nargs="+", default=["zed2i"])
    parser.add_argument("--image-size", "--image_size", dest="image_size", nargs=2, type=int, default=[224, 224])
    parser.add_argument("--include-depth", "--include_depth", dest="include_depth", action="store_true", default=False)
    parser.add_argument("--single-arm", "--single_arm", dest="single_arm", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-backend", "--video_backend", dest="video_backend", default="torchcodec")
    parser.add_argument("--normalization-method", "--normalization_method", dest="normalization_method", default="bounds_q99")

    parser.add_argument("--action-dim", "--action_dim", dest="action_dim", type=int, default=16)
    parser.add_argument("--pred-horizon", "--pred_horizon", dest="pred_horizon", type=int, default=50)
    parser.add_argument("--max-token-len", "--max_token_len", dest="max_token_len", type=int, default=200)
    parser.add_argument("--num-diffusion-steps", "--num_diffusion_steps", dest="num_diffusion_steps", type=int, default=10)
    parser.add_argument("--skip-masked-cameras", "--skip_masked_cameras", dest="skip_masked_cameras", action="store_true", default=True)
    parser.add_argument("--use-prefix-kv-cache", "--use_prefix_kv_cache", dest="use_prefix_kv_cache", action="store_true", default=True)

    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-repo-id", "--mock_repo_id", dest="mock_repo_id", default="robot_H1_L0")
    parser.add_argument("--mock-language-key", "--mock_language_key", dest="mock_language_key", default=None)
    parser.add_argument("--mock-num-frames", "--mock_num_frames", dest="mock_num_frames", type=int, default=500)
    parser.add_argument("--mock-debug-dir", "--mock_debug_dir", dest="mock_debug_dir", default="mock_debug/pi_jax")
    parser.add_argument("--mock-log-interval", "--mock_log_interval", dest="mock_log_interval", type=int, default=20)

    parser.add_argument("--real-env-id", "--real_env_id", dest="real_env_id", default="L0_TwoRobotStirSpoon-v1")
    parser.add_argument("--real-repo-id", "--real_repo_id", dest="real_repo_id", default=None)
    parser.add_argument("--real-language-key", "--real_language_key", dest="real_language_key", default=None)
    parser.add_argument("--real-language", "--real_language", dest="real_language", default=None)
    parser.add_argument("--real-dataset-idx", "--real_dataset_idx", dest="real_dataset_idx", type=int, default=None)
    parser.add_argument("--control-rate", "--control_rate", dest="control_rate", type=float, default=10.0)
    parser.add_argument("--max-episode-steps", "--max_episode_steps", dest="max_episode_steps", type=int, default=1000)
    parser.add_argument("--temporal-agg", "--temporal_agg", dest="temporal_agg", action="store_true", default=False)
    parser.add_argument("--static-left-arm", "--static_left_arm", dest="static_left_arm", action="store_true", default=False)
    parser.add_argument("--gripper-open-threshold", "--gripper_open_threshold", dest="gripper_open_threshold", type=float, default=0.9)
    parser.add_argument("--gripper-close-threshold", "--gripper_close_threshold", dest="gripper_close_threshold", type=float, default=0.2)
    parser.add_argument("--default-prompt", "--default_prompt", dest="default_prompt", default="Complete the task.")
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = parse_args()
    if args.mock:
        run_mock(args)
    else:
        if args.robot_state_type != "qpos":
            raise ValueError("Real-robot Pi JAX eval currently supports --robot-state-type qpos only.")
        run_real_robot(args)


if __name__ == "__main__":
    main()
