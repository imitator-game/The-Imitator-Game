# Launch finetuning for N1.6 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path
from typing import Any

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def _default_task_mapping_path() -> Path:
    return Path(__file__).resolve().parents[3] / "lerobot_dataset" / "task_mapping.json"


def _load_training_entries(config_path: str | None) -> list[dict[str, Any]]:
    if config_path is None:
        return []
    with open(config_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in config file: {config_path}")
    return data


def _parse_episode_indices(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        return [int(idx) for idx in raw]
    value = str(raw).strip()
    if not value:
        return None
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"Episode list must decode to a list: {raw}")
        return [int(idx) for idx in parsed]
    if ":" in value:
        start_s, end_s = value.split(":", 1)
        if not end_s:
            return None
        start = int(start_s) if start_s else 0
        end = int(end_s)
        return list(range(start, end))
    return [int(value)]


def _resolve_human_task_id(
    task_id: str,
    task_to_human_map: dict[str, str],
    config_kind: str,
) -> str | None:
    if config_kind == "human":
        return task_id if task_id.startswith("human_") else task_to_human_map.get(task_id)
    if config_kind in {"sim", "robot"}:
        return task_to_human_map.get(task_id)
    raise ValueError(f"Unsupported config kind: {config_kind}")


def _resolve_dataset_roots(
    dataset_parent: str,
    human_config_path: str | None,
    sim_config_path: str | None,
    robot_config_path: str | None,
    task_mapping_path: str | None,
) -> list[dict[str, Any]]:
    active_non_human = [p is not None for p in (sim_config_path, robot_config_path)]
    if human_config_path is None and any(active_non_human):
        raise ValueError("human_config_path is required when sim_config_path or robot_config_path is set")
    if sum(active_non_human) > 1:
        raise ValueError("Only one of sim_config_path or robot_config_path may be provided")
    if human_config_path is None:
        return [dataset_parent]

    mapping_path = Path(task_mapping_path) if task_mapping_path else _default_task_mapping_path()
    with open(mapping_path, "r") as f:
        mapping_data = json.load(f)

    task_to_human_map: dict[str, str] = {}
    for mapping in mapping_data.get("task_mappings", []):
        human_task_id = mapping.get("human_task_id")
        if not human_task_id:
            continue
        task_to_human_map[str(human_task_id)] = str(human_task_id)
        for sim_task_id in mapping.get("sim_task_id", []):
            task_to_human_map[str(sim_task_id)] = str(human_task_id)
        for robot_task_id in mapping.get("robot_task_id", []):
            task_to_human_map[str(robot_task_id)] = str(human_task_id)

    human_entries = _load_training_entries(human_config_path)
    other_kind = "sim" if sim_config_path is not None else "robot"
    other_entries = _load_training_entries(sim_config_path or robot_config_path)

    human_task_ids = {
        resolved
        for entry in human_entries
        if (resolved := _resolve_human_task_id(str(entry.get("repo_id") or entry.get("root")), task_to_human_map, "human"))
    }
    other_task_ids = {
        resolved
        for entry in other_entries
        if (
            resolved := _resolve_human_task_id(
                str(entry.get("repo_id") or entry.get("root")),
                task_to_human_map,
                other_kind,
            )
        )
    }
    intersected_human_task_ids = human_task_ids & other_task_ids
    if not intersected_human_task_ids:
        raise ValueError("No intersected tasks found between the provided config files")

    selected_roots: list[str] = []
    parent_dir = Path(dataset_parent)
    # Human config is only used to define the intersected task set.
    # Training data is sourced exclusively from the sim or robot side.
    for entry in other_entries:
        task_id = str(entry.get("repo_id") or entry.get("root"))
        resolved_human_task = _resolve_human_task_id(task_id, task_to_human_map, other_kind)
        if resolved_human_task not in intersected_human_task_ids:
            continue
        root = str(entry.get("root") or entry.get("repo_id"))
        dataset_root = parent_dir / root
        if not (dataset_root / "meta" / "info.json").exists():
            raise FileNotFoundError(
                f"Resolved dataset path does not look like a LeRobot dataset: {dataset_root}"
            )
        selected_roots.append(
            {
                "path": str(dataset_root),
                "episode_indices": _parse_episode_indices(entry.get("train")),
            }
        )

    deduped_roots = list({json.dumps(root, sort_keys=True): root for root in selected_roots}.values())
    print("Resolved multi-task dataset roots:")
    for root_spec in deduped_roots:
        suffix = ""
        if root_spec.get("episode_indices") is not None:
            suffix = f" episodes={root_spec['episode_indices']}"
        print(f"  - {root_spec['path']}{suffix}")
    return deduped_roots


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = _resolve_dataset_roots(
        dataset_parent=ft_config.dataset_path,
        human_config_path=ft_config.human_config_path,
        sim_config_path=ft_config.sim_config_path,
        robot_config_path=ft_config.robot_config_path,
        task_mapping_path=ft_config.task_mapping_path,
    )

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                        "lerobot_version": ft_config.lerobot_version,
                        "language_source": ft_config.language_source,
                        "task_mapping_path": ft_config.task_mapping_path,
                        "human_desc_path": ft_config.human_desc_path,
                        "sim_desc_path": ft_config.sim_desc_path,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_top_llm_layers = ft_config.tune_top_llm_layers
    config.model.use_backbone_lora = (
        ft_config.backbone_lora_rank if ft_config.use_backbone_lora else 0
    )
    config.model.use_llm_lora = ft_config.llm_lora_rank if ft_config.use_llm_lora else 0
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.use_action_head_diffusion_lora = (
        ft_config.action_head_diffusion_lora_rank
        if ft_config.use_action_head_diffusion_lora
        else 0
    )
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
    if ft_config.crop_fraction is not None:
        config.model.crop_fraction = ft_config.crop_fraction

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    if config.model.use_backbone_lora > 0:
        config.model.tune_visual = False
    if config.model.use_llm_lora > 0:
        config.model.tune_llm = False
    if config.model.use_action_head_diffusion_lora > 0:
        config.model.tune_diffusion_model = False

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.batch_size = ft_config.batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.logging_steps = ft_config.logging_steps
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"
    config.training.epoch_based_training = ft_config.epoch_based_training
    config.training.num_epochs = ft_config.num_epochs
    config.training.save_epochs = ft_config.save_epochs
    config.training.fp16 = ft_config.fp16
    config.training.bf16 = ft_config.bf16
    config.training.gradient_checkpointing = ft_config.gradient_checkpointing
    config.training.enable_online_eval = ft_config.enable_online_eval
    config.training.online_eval_env_id = ft_config.online_eval_env_id
    config.training.online_eval_steps = ft_config.online_eval_steps
    config.training.online_eval_epochs = ft_config.online_eval_epochs
    config.training.online_eval_num_episodes = ft_config.online_eval_num_episodes
    config.training.online_eval_num_envs = ft_config.online_eval_num_envs
    config.training.online_eval_max_episode_steps = ft_config.online_eval_max_episode_steps
    config.training.online_eval_sim_backend = ft_config.online_eval_sim_backend
    config.training.online_eval_control_mode = ft_config.online_eval_control_mode
    config.training.online_eval_obs_mode = ft_config.online_eval_obs_mode
    config.training.online_eval_reward_mode = ft_config.online_eval_reward_mode
    config.training.online_eval_shader = ft_config.online_eval_shader
    config.training.online_eval_capture_video = ft_config.online_eval_capture_video

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run(config)
