import numpy as np
import torch
from tqdm import tqdm

from gr00t.configs.base_config import Config
from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.stats import generate_rel_stats, generate_stats
from gr00t.experiment.dist_utils import barrier


def _normalize_dataset_path_spec(dataset_path_spec):
    if isinstance(dataset_path_spec, dict):
        dataset_path = (
            dataset_path_spec.get("path")
            or dataset_path_spec.get("dataset_path")
            or dataset_path_spec.get("root")
        )
        if dataset_path is None:
            raise ValueError(f"Dataset path spec is missing path: {dataset_path_spec}")
        episode_indices = dataset_path_spec.get("episode_indices")
        if episode_indices is not None:
            episode_indices = [int(idx) for idx in episode_indices]
        return dataset_path, episode_indices
    return dataset_path_spec, None


class DatasetFactory:
    """
    Factory class for building training datasets. Model-agnostic.
    """

    def __init__(self, config: Config):
        self.config = config

    def build(
        self, processor: BaseProcessor
    ) -> tuple[ShardedMixtureDataset, ShardedMixtureDataset | None]:
        """Build the dataset. Returns a tuple of (train_dataset, eval_dataset)."""
        assert self.config.training.eval_strategy == "no", (
            "Sharded dataset does not support evaluation sets"
        )

        all_datasets = []
        all_weights = []
        for dataset_spec in tqdm(
            self.config.data.datasets,
            total=len(self.config.data.datasets),
            desc="Initializing datasets",
        ):
            datasets = []
            for dataset_path_spec in dataset_spec.dataset_paths:
                dataset_path, episode_indices = _normalize_dataset_path_spec(dataset_path_spec)
                embodiment_tag = dataset_spec.embodiment_tag
                assert embodiment_tag is not None, "Embodiment tag is required"
                assert self.config.data.mode == "single_turn", "Only single turn mode is supported"
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        generate_stats(dataset_path)
                        generate_rel_stats(
                            dataset_path,
                            EmbodimentTag(embodiment_tag),
                            lerobot_version=dataset_spec.lerobot_version,
                        )
                else:
                    generate_stats(dataset_path)
                    generate_rel_stats(
                        dataset_path,
                        EmbodimentTag(embodiment_tag),
                        lerobot_version=dataset_spec.lerobot_version,
                    )
                barrier()
                dataset = ShardedSingleStepDataset(
                    dataset_path=dataset_path,
                    embodiment_tag=EmbodimentTag(embodiment_tag),
                    modality_configs=self.config.data.modality_configs[embodiment_tag],
                    lerobot_version=dataset_spec.lerobot_version,
                    language_source=dataset_spec.language_source,
                    task_mapping_path=dataset_spec.task_mapping_path,
                    human_desc_path=dataset_spec.human_desc_path,
                    sim_desc_path=dataset_spec.sim_desc_path,
                    video_backend=self.config.data.video_backend,
                    shard_size=self.config.data.shard_size,
                    episode_sampling_rate=self.config.data.episode_sampling_rate,
                    episode_indices=episode_indices,
                    seed=self.config.data.seed,
                    allow_padding=self.config.data.allow_padding,
                )
                datasets.append(dataset)
            dataset_lengths = np.array([len(dataset) for dataset in datasets])
            dataset_relative_lengths = dataset_lengths / dataset_lengths.sum()
            for dataset, relative_length in zip(datasets, dataset_relative_lengths):
                weight = relative_length * dataset_spec.mix_ratio
                all_datasets.append(dataset)
                all_weights.append(weight)

        return (
            ShardedMixtureDataset(
                datasets=all_datasets,
                weights=all_weights,
                processor=processor,
                seed=self.config.data.seed,
                training=True,
                num_shards_per_epoch=self.config.data.num_shards_per_epoch,
                override_pretraining_statistics=self.config.data.override_pretraining_statistics,
            ),
            None,
        )
