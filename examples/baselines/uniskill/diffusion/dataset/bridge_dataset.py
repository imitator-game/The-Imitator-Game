from .base_dataset import BaseDataset

class BridgeDataset(BaseDataset):
    def __init__(
        self,
        data_path: str='/workspace/datasets/bridge',
        **kwargs,
    ):
        kwargs['min_predict_future_horizon'] = 5
        kwargs['max_predict_future_horizon'] = 10
        super().__init__(data_path, **kwargs)