import os
import json
import random

from .base_dataset import BaseDataset

VALID_IDS = ['1', '4', '7', '8', '12', '13', '15', '21', '22', '24', '25', '26', '27', '28', '29', '30', '31', '33', '34', '35', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '63', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '144', '145', '146', '147', '148', '149', '150', '153', '156', '157', '158', '159', '160', '161', '164', '171', '172', '173']

class SthSthv2Dataset(BaseDataset):
    def __init__(
        self,
        data_path: str='/workspace/datasets/ststv2',
        **kwargs,
    ):
        self.remove_front_ratio = 0.4
        super().__init__(data_path, **kwargs)
        
    def _prepare_data(self, data_path):
        if self.train:
            metadata_path = os.path.join(data_path, "labels", "metadata_train.json")
        else:
            metadata_path = os.path.join(data_path, "labels", "metadata_val.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        with open(os.path.join(data_path, "labels", "labels.json"), "r") as f:
            labels = json.load(f)

        image_pair = []

        for video, value in metadata.items():
            vid_path = os.path.join(data_path, "sthsthv2", f"{video}.mp4")
            label = value['template'].replace("[", "").replace("]", "")
            label_id = labels[label]
            if 'length' not in value or label_id not in VALID_IDS:
                continue
            vid_len = value['length']
            if (1 - self.remove_front_ratio) * vid_len < self.min_predict_future_horizon:
                continue

            image_pair.append({
                'path': vid_path,
                'length': int(vid_len),
            })
            
        self.image_pair = image_pair
        
    def __getitem__(self, idx):
        image_pair = self.image_pair[idx]

        video_path = image_pair["path"]
        video_len = image_pair["length"]
        
        start_idx = int(self.remove_front_ratio * video_len)

        while True:
            predict_future_horizon = random.randint(
                self.min_predict_future_horizon, self.max_predict_future_horizon
            )
            predict_future_horizon = min(predict_future_horizon, video_len - 1)
            if start_idx > video_len - predict_future_horizon - 1:
                continue
            prev_idx = random.randint(start_idx, video_len - predict_future_horizon - 1)
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