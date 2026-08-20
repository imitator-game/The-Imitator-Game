import os
import random
import json
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True 

from .base_dataset import BaseDataset

class H2ODataset(BaseDataset):
    def __init__(
        self,
        data_path: str='/workspace/datasets/h2o',
        **kwargs,
    ):
        kwargs['min_predict_future_horizon'] = 30
        kwargs['max_predict_future_horizon'] = 60
        super().__init__(data_path, **kwargs)
        self.data_path = data_path
        
    def _prepare_data(self, data_path):
        with open(os.path.join(data_path, "metadata.json"), "r") as f:
            videos = json.load(f)
        
        file_len = len(videos)
        if self.train:
            videos = videos[:int(file_len*0.9)]
        else:
            videos = videos[int(file_len*0.9):]

        image_pair = []

        for video in videos:
            vid_len = video['length']
            if vid_len < self.min_predict_future_horizon:
                continue

            image_pair.append({
                'path': os.path.join(data_path, video['path']),
                'length': vid_len,
            })
            
        self.image_pair = image_pair
        
    def read_images(self, video_path, prev_idx, next_idx):
        # PIL 이미지로 변환
        curr_image = Image.open(os.path.join(self.data_path, video_path, f"{prev_idx:06d}.png"))
        next_image = Image.open(os.path.join(self.data_path, video_path, f"{next_idx:06d}.png"))
        
        return curr_image, next_image
    