import os
import json
import random

from PIL import Image
from decord import VideoReader, cpu

from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

class BaseDataset(TorchDataset):
    def __init__(
        self,
        data_path,
        train: bool=True,
        min_predict_future_horizon: int=10,
        max_predict_future_horizon: int=30,
        resolution: int=64,
        idm_resolution: int=224,
        depth_processor=None,
    ):
        self.min_predict_future_horizon = min_predict_future_horizon
        self.max_predict_future_horizon = max_predict_future_horizon
        self.depth_processor = depth_processor
        self.train = train
        

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR
                ),
            ]
        )
        
        self.idm_image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (idm_resolution, idm_resolution), interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
            ]
        )
        
        if train:
            self.fdm_normalize = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                    ), 
                ]
            )
            
        self._prepare_data(data_path)
        
    def _prepare_data(self, data_path):
        with open(os.path.join(data_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
            
        videos = []
        for video in metadata:
            videos.append(video)

        videos = sorted(videos)
        
        file_len = len(videos)
        if self.train:
            videos = videos[:int(file_len*0.9)]
        else:
            videos = videos[int(file_len*0.9):]

        image_pair = []

        for video in videos:
            vid_len = metadata[video]
            if vid_len < self.min_predict_future_horizon:
                continue

            image_pair.append({
                'path': os.path.join(data_path, video),
                'length': vid_len,
            })
            
        self.image_pair = image_pair
            
    def __len__(self):
        return len(self.image_pair)
    
    def __getitem__(self, idx):
        image_pair = self.image_pair[idx]

        video_path = image_pair["path"]
        video_len = image_pair["length"]

        while True:
            predict_future_horizon = random.randint(
                self.min_predict_future_horizon, self.max_predict_future_horizon
            )
            predict_future_horizon = min(predict_future_horizon, video_len - 1)
            prev_idx = random.randint(0, video_len - predict_future_horizon - 1)
            next_idx = prev_idx + predict_future_horizon
            if next_idx < video_len:
                break

        curr_image, next_image = self.read_images(video_path, prev_idx, next_idx)
        
        idm_curr_image = self.idm_image_transforms(curr_image)
        idm_next_image = self.idm_image_transforms(next_image)
        
        curr_image = self.image_transforms(curr_image)
        next_image = self.image_transforms(next_image)
        
        curr_depth_features = self.depth_processor(idm_curr_image, do_rescale=False)["pixel_values"][0]
        next_depth_features = self.depth_processor(idm_next_image, do_rescale=False)["pixel_values"][0]
        
        if self.train:
            curr_image = self.fdm_normalize(curr_image)
            next_image = self.fdm_normalize(next_image)

        return {
            "curr_images": curr_image,
            "next_images": next_image,
            "idm_curr_images": idm_curr_image,
            "idm_next_images": idm_next_image,
            "curr_depth_features": curr_depth_features,
            "next_depth_features": next_depth_features,
        }
        
    def read_images(self, video_path, prev_idx, next_idx):
        # Decord VideoReader로 비디오 읽기
        vr = VideoReader(video_path, ctx=cpu(0))
        
        # prev_idx와 next_idx의 프레임 가져오기
        conditioning_frame = vr[prev_idx].asnumpy()
        frame = vr[next_idx].asnumpy()

        # PIL 이미지로 변환
        curr_image = Image.fromarray(conditioning_frame)
        next_image = Image.fromarray(frame)
        
        return curr_image, next_image
    