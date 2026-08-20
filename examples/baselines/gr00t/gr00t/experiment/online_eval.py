from __future__ import annotations

import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from mani_skill.envs.tasks.tabletop.utils import L0_L3_utils
from transformers import TrainerCallback
import wandb

from examples.baselines.openvla_oft.utils.make_env import make_eval_envs
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.policy.gr00t_policy import Gr00tPolicy


def extract_base_env_name(env_id: str) -> str:
    env_name = os.path.basename(env_id)
    if env_name.startswith("L") and "_" in env_name:
        parts = env_name.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            level, base = parts[0], parts[1]
            if level == "L3":
                if "-v" in base:
                    name_part, version_part = base.rsplit("-v", 1)
                    return f"{name_part}L3-v{version_part}"
                return f"{base}L3"
            return base
    return env_id


def extract_level(env_id: str) -> str:
    env_name = os.path.basename(env_id)
    if env_name.startswith("L") and "_" in env_name:
        parts = env_name.split("_", 1)
        if len(parts) == 2 and parts[0] in ["L0", "L1", "L2", "L3"]:
            return parts[0]
    if env_name.rsplit("-v", 1)[0].endswith("L3"):
        return "L3"
    return "L0"


def set_l_level(level: str) -> None:
    L0_L3_utils.set_l1_enabled(False)
    L0_L3_utils.set_l2_enabled(False)
    L0_L3_utils.set_l3_enabled(False)
    os.environ.pop("MANI_SKILL_L1", None)
    os.environ.pop("MANI_SKILL_L2", None)
    os.environ.pop("MANI_SKILL_L3", None)

    if level == "L1":
        L0_L3_utils.set_l1_enabled(True)
        os.environ["MANI_SKILL_L1"] = "1"
    elif level == "L2":
        L0_L3_utils.set_l2_enabled(True)
        os.environ["MANI_SKILL_L2"] = "1"


def configure_lr_mirror(
    lr_mirror_enabled: Optional[bool],
    lr_mirror_robot_pose_enabled: Optional[bool],
) -> None:
    L0_L3_utils.set_lr_mirror_enabled(lr_mirror_enabled)
    L0_L3_utils.set_lr_mirror_robot_pose_enabled(lr_mirror_robot_pose_enabled)


def level_for_env_switches(level: str) -> str:
    return "L0" if level == "L3" else level


def _load_json_if_exists(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    import json

    with open(path_obj, "r") as f:
        return json.load(f)


def _default_lerobot_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "lerobot_dataset"


class TaskPromptProvider:
    def __init__(
        self,
        language_source: str,
        task_mapping_path: str | None = None,
        human_desc_path: str | None = None,
        sim_desc_path: str | None = None,
    ):
        self.language_source = str(language_source).lower()
        self.task_to_human_task_map: dict[str, str] = {}
        self.human_desc_map: dict[str, list[str]] = {}
        self.sim_desc_map: dict[str, list[str]] = {}

        if self.language_source == "task":
            return

        default_dir = _default_lerobot_dataset_dir()
        task_mapping_path = task_mapping_path or str(default_dir / "task_mapping.json")
        human_desc_path = human_desc_path or str(default_dir / "task_desc" / "human_desc.json")
        sim_desc_path = sim_desc_path or str(default_dir / "task_desc" / "sim_desc.json")

        task_mapping_data = _load_json_if_exists(task_mapping_path)
        for mapping in task_mapping_data.get("task_mappings", []):
            human_task_id = mapping.get("human_task_id")
            if not human_task_id:
                continue
            for sim_task_id in mapping.get("sim_task_id", []):
                self.task_to_human_task_map[str(sim_task_id)] = str(human_task_id)
            for robot_task_id in mapping.get("robot_task_id", []):
                self.task_to_human_task_map[str(robot_task_id)] = str(human_task_id)

        human_desc_data = _load_json_if_exists(human_desc_path)
        for task_id, descs in human_desc_data.items():
            if isinstance(descs, list):
                self.human_desc_map[str(task_id)] = [str(desc) for desc in descs if str(desc)]

        sim_desc_data = _load_json_if_exists(sim_desc_path)
        for task_id, descs in sim_desc_data.items():
            if isinstance(descs, list):
                self.sim_desc_map[str(task_id)] = [str(desc) for desc in descs if str(desc)]

    def sample_prompt(self, task_id: str) -> str:
        task_id = str(task_id)
        if self.language_source == "task":
            return task_id

        if self.language_source == "sim_desc":
            sim_descs = self.sim_desc_map.get(task_id, [])
            if sim_descs:
                return random.choice(sim_descs)
            return task_id

        human_task_id = self.task_to_human_task_map.get(task_id)
        if human_task_id:
            human_descs = self.human_desc_map.get(human_task_id, [])
            if human_descs:
                return random.choice(human_descs)
        sim_descs = self.sim_desc_map.get(task_id, [])
        if sim_descs:
            return random.choice(sim_descs)
        return task_id

    def sample_batch(self, task_id: str, batch_size: int) -> list[str]:
        return [self.sample_prompt(task_id) for _ in range(batch_size)]


class Gr00tFlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, modality_configs, camera_names: list[str]):
        self.base_env = env.unwrapped
        super().__init__(env)
        self.modality_configs = modality_configs
        self.camera_names = list(camera_names)
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

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
        chunks = []
        batch_dim = None
        for value in values:
            tensor = torch.as_tensor(value, device=self.base_env.device)
            if tensor.dim() == 0:
                tensor = tensor.view(1, 1)
            elif tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.reshape(tensor.shape[0], -1)
            if batch_dim is None:
                batch_dim = tensor.shape[0]
            chunks.append(tensor)
        if not chunks:
            raise ValueError("Failed to extract qpos state from environment observation")
        return torch.cat(chunks, dim=1)

    def _extract_qpos(self, observation):
        qpos_chunks = self._collect_state_by_key(observation, "qpos")
        qpos = self._stack_state(qpos_chunks)
        if qpos.shape[-1] == 18:
            return {
                "left_arm": qpos[..., :7],
                "left_gripper": qpos[..., 7:9],
                "right_arm": qpos[..., 9:16],
                "right_gripper": qpos[..., 16:18],
            }
        if qpos.shape[-1] == 16:
            return {
                "left_arm": qpos[..., :7],
                "left_gripper": qpos[..., 7:8],
                "right_arm": qpos[..., 8:15],
                "right_gripper": qpos[..., 15:16],
            }
        raise ValueError(f"Unsupported qpos dimension for GR00T online eval: {qpos.shape}")

    def observation(self, observation):
        sensor_data = observation.pop("sensor_data")
        if "sensor_param" in observation:
            del observation["sensor_param"]

        video_obs = {}
        for camera_name in self.camera_names:
            if camera_name in sensor_data:
                video_obs[camera_name] = sensor_data[camera_name]["rgb"]
            elif len(sensor_data) == 1:
                only_key = next(iter(sensor_data.keys()))
                video_obs[camera_name] = sensor_data[only_key]["rgb"]
            else:
                raise KeyError(
                    f"Camera '{camera_name}' not found in environment sensor_data keys {list(sensor_data.keys())}"
                )

        return {
            "video": video_obs,
            "state": self._extract_qpos(observation),
        }


@dataclass
class OnlineEvalConfig:
    env_id: str
    embodiment_tag: EmbodimentTag
    language_source: str
    task_mapping_path: Optional[str]
    human_desc_path: Optional[str]
    sim_desc_path: Optional[str]
    num_episodes: int = 10
    num_envs: int = 1
    max_episode_steps: int = 400
    sim_backend: str = "physx_cpu"
    control_mode: str = "pd_joint_pos"
    obs_mode: str = "rgb"
    reward_mode: str = "sparse"
    shader: str = "rt-fast"
    capture_video: bool = False
    output_dir: str = "./outputs"
    compute_dtw: bool = False
    dtw_band_ratio: float = 0.15
    sim_dataset_file: Optional[str] = None
    sim_root: str = "demos/imitator_data"
    dtw_action_key: str = "action.qpos_gripper_actions"
    lr_mirror_enabled: Optional[bool] = None
    lr_mirror_robot_pose_enabled: Optional[bool] = False


class Gr00tOnlineEvaluator:
    def __init__(self, config: OnlineEvalConfig, processor: BaseProcessor):
        self.config = config
        self.processor = processor
        self.prompt_provider = TaskPromptProvider(
            language_source=config.language_source,
            task_mapping_path=config.task_mapping_path,
            human_desc_path=config.human_desc_path,
            sim_desc_path=config.sim_desc_path,
        )
        self.envs = self._create_eval_envs()
        modality_configs = processor.get_modality_configs()[config.embodiment_tag.value]
        language_keys = modality_configs["language"].modality_keys
        if len(language_keys) != 1:
            raise ValueError("Online eval currently supports exactly one language key")
        self.language_key = language_keys[0]
        self.action_keys = modality_configs["action"].modality_keys
        self._warned_action_dim_fits: set[tuple[str, int, int]] = set()
        self.dtw_provider = None
        self.traj_metrics = None
        self._init_trajectory_metrics()

    def _init_trajectory_metrics(self) -> None:
        if not self.config.compute_dtw:
            return
        if not self.config.sim_dataset_file:
            logging.warning("TSS requested but sim_dataset_file is not set; skipping TSS metrics.")
            return
        try:
            from examples.baselines.lerobot_dataset.trajectory_metrics import (
                GTTrajectoryProvider,
                TrajectoryMetrics,
            )

            self.dtw_provider = GTTrajectoryProvider(
                sim_dataset_file=self.config.sim_dataset_file,
                sim_root=self.config.sim_root,
                action_key=self.config.dtw_action_key,
            )
            self.traj_metrics = TrajectoryMetrics(band_ratio=self.config.dtw_band_ratio)
            logging.info(
                "Loaded GT trajectories for TSS: tasks=%s action_key=%s band_ratio=%s",
                len(self.dtw_provider._episodes),
                self.config.dtw_action_key,
                self.config.dtw_band_ratio,
            )
        except Exception as exc:
            logging.warning("Failed to initialize TSS metrics; continuing without TSS: %s", exc)
            self.dtw_provider = None
            self.traj_metrics = None

    def _create_eval_envs(self):
        modality_configs = self.processor.get_modality_configs()[self.config.embodiment_tag.value]
        camera_names = list(modality_configs["video"].modality_keys)

        env_kwargs = dict(
            control_mode=self.config.control_mode,
            reward_mode=self.config.reward_mode,
            obs_mode=self.config.obs_mode,
            render_mode="rgb_array",
            sensor_configs=dict(shader_pack=self.config.shader),
            human_render_camera_configs=dict(shader_pack=self.config.shader),
            viewer_camera_configs=dict(shader_pack=self.config.shader),
            max_episode_steps=self.config.max_episode_steps,
        )
        other_kwargs = dict(obs_horizon=1)
        video_dir = None
        if self.config.capture_video:
            video_dir = os.path.join(self.config.output_dir, "online_eval_videos")

        requested_level = extract_level(self.config.env_id)
        env_switch_level = level_for_env_switches(requested_level)
        set_l_level(env_switch_level)
        configure_lr_mirror(
            self.config.lr_mirror_enabled,
            self.config.lr_mirror_robot_pose_enabled,
        )
        base_env_name = extract_base_env_name(self.config.env_id)
        logging.info(
            "GR00T eval env mapping: env_id=%s -> gym_env_id=%s requested_level=%s env_switch_level=%s "
            "lr_mirror=%s lr_mirror_robot_pose=%s",
            self.config.env_id,
            base_env_name,
            requested_level,
            env_switch_level,
            self.config.lr_mirror_enabled,
            self.config.lr_mirror_robot_pose_enabled,
        )

        wrappers = [
            partial(
                Gr00tFlattenObservationWrapper,
                modality_configs=modality_configs,
                camera_names=camera_names,
            ),
        ]
        return make_eval_envs(
            base_env_name,
            self.config.num_envs,
            self.config.sim_backend,
            env_kwargs,
            other_kwargs,
            video_dir=video_dir,
            wrappers=wrappers,
            l_level=env_switch_level,
            lr_mirror_enabled=self.config.lr_mirror_enabled,
            lr_mirror_robot_pose_enabled=self.config.lr_mirror_robot_pose_enabled,
        )

    def _to_numpy(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, dict):
            return {k: self._to_numpy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_numpy(v) for v in value]
        return value

    def _build_policy_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        np_obs = self._to_numpy(obs)
        batch_size = next(iter(np_obs["video"].values())).shape[0]
        prompts = self.prompt_provider.sample_batch(self.config.env_id, batch_size)
        np_obs["language"] = {self.language_key: [[prompt] for prompt in prompts]}
        return np_obs

    def _fit_action_dim(self, key: str, action: np.ndarray, target_dim: int) -> np.ndarray:
        current_dim = action.shape[-1]
        if current_dim == target_dim:
            return action
        warn_key = (key, current_dim, target_dim)
        if warn_key not in self._warned_action_dim_fits:
            logging.warning(
                "Adjusting GR00T action '%s' dim from %d to env action dim %d",
                key,
                current_dim,
                target_dim,
            )
            self._warned_action_dim_fits.add(warn_key)
        if current_dim > target_dim:
            return action[..., :target_dim]
        pad_width = [(0, 0)] * action.ndim
        pad_width[-1] = (0, target_dim - current_dim)
        return np.pad(action, pad_width, mode="constant", constant_values=0.0)

    def _concat_action_parts(
        self,
        action_dict: dict[str, np.ndarray],
        keys: list[str],
        *,
        env_key: str,
        target_dim: int | None = None,
    ) -> np.ndarray:
        parts = [action_dict[key] for key in keys if key in action_dict]
        if not parts:
            raise KeyError(f"None of action keys {keys} found in policy output {list(action_dict)}")
        action = np.concatenate(parts, axis=-1).astype(np.float32)
        if target_dim is not None:
            action = self._fit_action_dim(env_key, action, target_dim)
        return action

    def _format_env_action_step(
        self,
        action_dict: dict[str, np.ndarray],
        action_step_idx: int,
    ) -> dict[str, np.ndarray] | np.ndarray:
        step_actions = {
            key: value[:, action_step_idx].astype(np.float32)
            for key, value in action_dict.items()
        }
        action_space = self.envs.action_space

        if isinstance(action_space, gym.spaces.Dict):
            env_keys = set(action_space.spaces.keys())
            if env_keys == set(step_actions.keys()):
                return step_actions

            if {"panda_wristcam-0", "panda_wristcam-1"}.issubset(env_keys):
                return {
                    "panda_wristcam-0": self._concat_action_parts(
                        step_actions,
                        ["left_arm", "left_gripper"],
                        env_key="panda_wristcam-0",
                        target_dim=action_space["panda_wristcam-0"].shape[-1],
                    ),
                    "panda_wristcam-1": self._concat_action_parts(
                        step_actions,
                        ["right_arm", "right_gripper"],
                        env_key="panda_wristcam-1",
                        target_dim=action_space["panda_wristcam-1"].shape[-1],
                    ),
                }

            raise ValueError(
                f"Unsupported eval action space keys {sorted(env_keys)} for GR00T action keys {sorted(step_actions)}"
            )

        return np.concatenate([step_actions[key] for key in self.action_keys], axis=-1)

    def _flatten_action_for_tss(self, env_action: dict[str, np.ndarray] | np.ndarray) -> np.ndarray | None:
        if isinstance(env_action, np.ndarray):
            action = env_action
        elif isinstance(env_action, dict):
            if {"panda_wristcam-0", "panda_wristcam-1"}.issubset(env_action):
                action = np.concatenate(
                    [env_action["panda_wristcam-0"], env_action["panda_wristcam-1"]],
                    axis=-1,
                )
            else:
                parts = [env_action[key] for key in self.action_keys if key in env_action]
                if not parts:
                    return None
                action = np.concatenate(parts, axis=-1)
        else:
            return None

        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None]
        return action

    def _bool_array(self, value: Any, size: int) -> np.ndarray | None:
        if value is None:
            return None
        value = self._to_numpy(value)
        arr = np.asarray(value, dtype=bool)
        if arr.ndim == 0:
            return np.full((size,), bool(arr), dtype=bool)
        if arr.size == 0:
            return None
        if arr.size < size:
            padded = np.zeros((size,), dtype=bool)
            padded[: arr.size] = arr.reshape(-1)
            return padded
        return arr.reshape(-1)[:size]

    def _final_success_array(self, final_info: Any, size: int) -> np.ndarray | None:
        def extract_success(item: Any) -> Any:
            if not isinstance(item, dict):
                return None
            for key in ("success", "success_once", "success_at_end"):
                if key in item:
                    return item[key]
            episode = item.get("episode")
            if isinstance(episode, dict):
                for key in ("success", "success_once", "success_at_end"):
                    if key in episode:
                        return episode[key]
            return None

        if isinstance(final_info, dict):
            return self._bool_array(extract_success(final_info), size)
        if isinstance(final_info, (list, tuple)):
            values = []
            found = False
            for item in final_info[:size]:
                success = extract_success(item)
                if success is None:
                    values.append(False)
                    continue
                arr = self._bool_array(success, 1)
                values.append(bool(arr[0]) if arr is not None and len(arr) else False)
                found = True
            if found:
                values.extend([False] * (size - len(values)))
                return np.asarray(values[:size], dtype=bool)
        return None

    def _sample_gt_trajectory(self) -> np.ndarray | None:
        if self.dtw_provider is None:
            return None
        return self.dtw_provider.sample_gt_trajectory(self.config.env_id, seed=None)

    def _record_tss_metrics(
        self,
        eval_metrics: dict[str, list[Any]],
        action_buffer: Any,
        success_steps: np.ndarray,
        gt_traj: np.ndarray,
        final_info: Any,
        ts: int,
    ) -> np.ndarray | None:
        final_success = self._final_success_array(final_info, self.config.num_envs)
        if final_success is not None:
            success_steps[(success_steps == -1) & final_success] = ts

        pred_trajs = action_buffer.get_and_reset()
        gt_len = len(gt_traj)
        for env_i, pred_traj in enumerate(pred_trajs):
            success_step = int(success_steps[env_i])
            if success_step >= 2:
                try:
                    metric = self.traj_metrics.compute(pred_traj[:success_step], gt_traj)
                    eval_metrics["tss_success"].append(np.array(metric["tss"], dtype=np.float32))
                    eval_metrics["ndtw_success"].append(np.array(metric["ndtw"], dtype=np.float32))
                except Exception as exc:
                    logging.warning("TSS(success) failed for %s env %s: %s", self.config.env_id, env_i, exc)
            else:
                pred_trimmed = pred_traj[:gt_len]
                if len(pred_trimmed) >= 2:
                    try:
                        metric = self.traj_metrics.compute(pred_trimmed, gt_traj)
                        eval_metrics["tss_fail"].append(np.array(metric["tss"], dtype=np.float32))
                        eval_metrics["ndtw_fail"].append(np.array(metric["ndtw"], dtype=np.float32))
                    except Exception as exc:
                        logging.warning("TSS(fail) failed for %s env %s: %s", self.config.env_id, env_i, exc)

        success_steps[:] = -1
        return self._sample_gt_trajectory()

    def _summarize_metrics(self, eval_metrics: dict[str, list[Any]]) -> dict[str, float]:
        summary = {}
        for key, values in eval_metrics.items():
            if not values:
                continue
            flattened = []
            for value in values:
                arr = np.asarray(value)
                if arr.size == 0:
                    continue
                flattened.append(arr.reshape(-1))
            if not flattened:
                continue
            summary[key] = float(np.mean(np.concatenate(flattened, axis=0)))
        return summary

    @torch.inference_mode()
    def evaluate(self, model, global_step: int) -> dict[str, float]:
        was_training = model.training
        processor_was_training = self.processor.training

        model.eval()
        self.processor.eval()

        policy = Gr00tPolicy(
            embodiment_tag=self.config.embodiment_tag,
            model=model,
            processor=self.processor,
            strict=True,
        )

        eval_metrics = defaultdict(list)
        obs, info = self.envs.reset()
        completed_episodes = 0
        ts = 0

        action_buffer = None
        success_steps = None
        gt_traj = None
        compute_dtw = self.dtw_provider is not None and self.traj_metrics is not None
        if compute_dtw:
            from examples.baselines.lerobot_dataset.trajectory_metrics import EpisodeActionBuffer

            gt_traj = self._sample_gt_trajectory()
            if gt_traj is None:
                logging.warning("No GT trajectory for %s; TSS metrics will be skipped.", self.config.env_id)
                compute_dtw = False
            else:
                action_buffer = EpisodeActionBuffer(num_envs=self.config.num_envs)
                success_steps = np.full((self.config.num_envs,), -1, dtype=np.int64)

        while completed_episodes < self.config.num_episodes:
            policy_obs = self._build_policy_observation(obs)
            action_dict, _ = policy.get_action(policy_obs)
            action_horizon = next(iter(action_dict.values())).shape[1]

            for action_step_idx in range(action_horizon):
                step_action = self._format_env_action_step(action_dict, action_step_idx)
                if compute_dtw and action_buffer is not None:
                    flat_action = self._flatten_action_for_tss(step_action)
                    if flat_action is not None:
                        action_buffer.append(flat_action)
                obs, rew, terminated, truncated, info = self.envs.step(step_action)
                ts += 1
                if compute_dtw and success_steps is not None:
                    success = self._bool_array(info.get("success"), self.config.num_envs)
                    if success is not None:
                        success_steps[(success_steps == -1) & success] = ts
                truncated_np = np.asarray(self._to_numpy(truncated))
                if np.any(truncated_np):
                    break

            if np.any(truncated_np):
                final_info = info.get("final_info")
                if isinstance(final_info, dict):
                    episode_info = final_info.get("episode", {})
                    for key, value in episode_info.items():
                        eval_metrics[key].append(self._to_numpy(value))
                else:
                    for episode_final in final_info:
                        if episode_final is None:
                            continue
                        episode_info = episode_final.get("episode", {})
                        for key, value in episode_info.items():
                            eval_metrics[key].append(self._to_numpy(value))

                if compute_dtw and action_buffer is not None and success_steps is not None and gt_traj is not None:
                    gt_traj = self._record_tss_metrics(
                        eval_metrics,
                        action_buffer,
                        success_steps,
                        gt_traj,
                        final_info,
                        ts,
                    )
                    if gt_traj is None:
                        logging.warning("No replacement GT trajectory for %s; disabling TSS.", self.config.env_id)
                        compute_dtw = False

                completed_episodes += self.config.num_envs
                ts = 0
                if completed_episodes < self.config.num_episodes:
                    obs, info = self.envs.reset()

        metrics = self._summarize_metrics(eval_metrics)
        metrics["global_step"] = float(global_step)

        if processor_was_training:
            self.processor.train()
        if was_training:
            model.train()
        return metrics

    def close(self) -> None:
        if self.envs is not None:
            self.envs.close()


class OnlineEvalCallback(TrainerCallback):
    def __init__(
        self,
        evaluator: Gr00tOnlineEvaluator,
        eval_steps: int = 0,
        eval_step_points: Optional[list[int]] = None,
    ):
        self.evaluator = evaluator
        self.eval_steps = max(int(eval_steps), 0)
        self.eval_step_points = set(eval_step_points or [])

    def _should_run(self, global_step: int) -> bool:
        if global_step <= 0:
            return False
        if global_step in self.eval_step_points:
            return True
        return self.eval_steps > 0 and global_step % self.eval_steps == 0

    def _unwrap_model(self, model):
        current = model
        seen = set()
        while True:
            current_id = id(current)
            if current_id in seen:
                return current
            seen.add(current_id)
            if hasattr(current, "module"):
                current = current.module
                continue
            if hasattr(current, "_orig_mod"):
                current = current._orig_mod
                continue
            return current

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        if not self._should_run(state.global_step):
            return control

        logging.info("Starting online eval at step %s", state.global_step)
        eval_model = self._unwrap_model(model)
        metrics = self.evaluator.evaluate(eval_model, state.global_step)
        prefixed_metrics = {f"online_eval/{key}": value for key, value in metrics.items()}

        logging.info("Online eval metrics at step %s: %s", state.global_step, prefixed_metrics)
        if wandb.run is not None:
            wandb.log(prefixed_metrics, step=state.global_step)
        return control
