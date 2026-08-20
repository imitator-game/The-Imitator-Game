import os

import numpy as np
from PIL import Image

from .base_dataset import BaseDataset

class LIBERODataset(BaseDataset):
    def __init__(
        self,
        data_path: str='/workspace/datasets/LIBERO',
        **kwargs,
    ):
        kwargs['min_predict_future_horizon'] = 20
        kwargs['max_predict_future_horizon'] = 40
        super().__init__(data_path, **kwargs)
        
    def _prepare_data(self, data_path):
        videos = []
        for suite in os.listdir(data_path):
            if not suite.startswith('libero'):
                continue
            suite_path = os.path.join(data_path, suite)
            for task in os.listdir(suite_path):
                if task.startswith('.'):
                    continue
                task_path = os.path.join(suite_path, task)
                demo_num = len(os.listdir(task_path))
                if self.train:
                    demos = sorted(os.listdir(task_path))[:int(demo_num*0.9)]
                else:
                    demos = sorted(os.listdir(task_path))[int(demo_num*0.9):]
                for demo in demos:
                    videos.append(os.path.join(task_path, demo))

        image_pair = []

        for video in videos:
            vid_len = np.load(video).shape[0]
            if vid_len < self.min_predict_future_horizon:
                continue

            image_pair.append({
                'path': video,
                'length': vid_len,
            })
            
        self.image_pair = image_pair
        
    def read_images(self, video_path, prev_idx, next_idx):
        video = np.load(video_path)
        conditioning_frame = video[prev_idx]
        frame = video[next_idx]

        curr_image = Image.fromarray(conditioning_frame)
        next_image = Image.fromarray(frame)
        
        return curr_image, next_image
    