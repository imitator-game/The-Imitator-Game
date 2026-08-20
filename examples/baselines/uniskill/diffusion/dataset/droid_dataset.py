import os
import json
from .base_dataset import BaseDataset

class DroidDataset(BaseDataset):
    """
    Droid dataset
    """
    def __init__(
        self,
        data_path: str='/workspace/datasets/droid',
        **kwargs,
    ):
        kwargs['min_predict_future_horizon'] = 20
        kwargs['max_predict_future_horizon'] = 40
        super().__init__(data_path, **kwargs)
        
    def _prepare_data(self, data_path):
        with open(os.path.join(data_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        use_raw_dataset = 'raw' in data_path

        videos = []
        for video in metadata:
            if not use_raw_dataset:
                if metadata[video]["success"] != "success":
                    continue
            videos.append(video)

        videos = sorted(videos)
        
        file_len = len(videos)
        if self.train:
            videos = videos[:int(file_len*0.9)]
        else:
            videos = videos[int(file_len*0.9):]

        image_pair = []

        for video in videos:
            vid_len = metadata[video]['length']
            if vid_len < self.min_predict_future_horizon:
                continue

            if use_raw_dataset:
                lab = metadata[video]['lab']
                ext1_mp4_path = metadata[video]['ext1_mp4_path']
                vid_len -= 1
                video_path = os.path.join(data_path, lab, ext1_mp4_path)
            else:
                video_path = os.path.join(data_path, "exterior_image_1_left", video)
            image_pair.append({
                'path': video_path,
                'length': vid_len,
            })
            
        self.image_pair = image_pair