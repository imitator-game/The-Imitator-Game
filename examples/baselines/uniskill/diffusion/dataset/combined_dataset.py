from torch.utils.data import Dataset as TorchDataset

from . import *

class CombinedDataset(TorchDataset):
    def __init__(
        self,
        datasets=["droid", "libero", "bridge", "h2o", "xskill"],
        **kwargs,
    ):
        self.datasets = []
        dataset_classes = {
            "droid": DroidDataset,
            "libero": LIBERODataset,
            "bridge": BridgeDataset,
            "h2o": H2ODataset,
            "xskill": XSkillDataset,
            "sthsthv2": SthSthv2Dataset,
        }

        for dataset_name in datasets:
            if dataset_name in dataset_classes:
                self.datasets.append(dataset_classes[dataset_name](**kwargs))

        self.dataset_lengths = [len(dataset) for dataset in self.datasets]
            
    def __len__(self):
        return sum(self.dataset_lengths)
    
    def __getitem__(self, idx):
        for i, dataset in enumerate(self.datasets):
            if idx < self.dataset_lengths[i]:
                return dataset[idx]
            idx -= self.dataset_lengths[i]
        raise ValueError(f"Index {idx} out of bounds for CombinedDataset")