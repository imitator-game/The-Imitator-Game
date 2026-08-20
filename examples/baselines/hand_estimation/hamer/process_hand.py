from pathlib import Path
import shutil

from tqdm import tqdm
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
import h5py

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def predict_hand(detector, model, cpm, renderer, device, imgs, model_cfg, args):
    # Iterate over all images in folder
    results = {}
    for img_id, img in enumerate(imgs):
        img_cv2 = img[:, :, ::-1]

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        
        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)
        
        if len(bboxes) == 0:
            for k in results.keys():
                if isinstance(results[k], dict):
                    for k2 in results[k].keys():
                        results[k][k2] = results[k][k2] + [np.zeros_like(results[k][k2][0])]
                if k != 'is_right':
                    results[k] = results[k] + [np.zeros_like(results[k][0])]
                else:
                    results[k] = results[k] + [-np.ones_like(results[k][0])]    
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        
        # for batch in dataloader:
        batch = next(iter(dataloader))
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        
        for k in out.keys():
            if isinstance(out[k], dict):
                for k2 in out[k].keys():
                    results[k+'.'+k2] = results.get(k+'.'+k2, []) + [out[k][k2].cpu().numpy()]
            else:
                results[k] = results.get(k, []) + [out[k].cpu().numpy()]
        
        results['is_right'] = results.get('is_right', []) + [batch['right'].cpu().numpy()]
        
        if not args.visualize: continue

        multiplier = (2*batch['right']-1)
        pred_cam = out['pred_cam']
        pred_cam[:,1] = multiplier*pred_cam[:,1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        multiplier = (2*batch['right']-1)
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

        # Render the result
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            # Get filename from path img_path
            # img_fn, _ = os.path.splitext(os.path.basename(img_path))
            img_fn = f"test_{img_id}"
            person_id = int(batch['personid'][n])
            white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
            input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            input_patch = input_patch.permute(1,2,0).numpy()

            regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                    out['pred_cam_t'][n].detach().cpu().numpy(),
                                    batch['img'][n],
                                    mesh_base_color=LIGHT_BLUE,
                                    scene_bg_color=(1, 1, 1),
                                    )

            if args.side_view:
                side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        white_img,
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        side_view=True)
                final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
            else:
                final_img = np.concatenate([input_patch, regression_img], axis=1)

            cv2.imwrite(os.path.join(args.vis_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

            # Add all verts and cams to list
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            is_right = batch['right'][n].cpu().numpy()
            verts[:,0] = (2*is_right-1)*verts[:,0]
            cam_t = pred_cam_t_full[n]
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)

            # Save all meshes to disk
            if args.save_mesh:
                camera_translation = cam_t.copy()
                tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                tmesh.export(os.path.join(args.vis_folder, f'{img_fn}_{person_id}.obj'))

        # Render front view
        if args.full_frame and len(all_verts) > 0 and args.visualize:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.vis_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])
    
    for k in results.keys():
        results[k] = np.concatenate(results[k], axis=0)
        assert len(results[k]) == len(imgs), f"{k} has different length {len(results[k])} vs {len(imgs)}"
    return results
    

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--vis_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    
    parser.add_argument('--video_dir', type=str, required=True, help='Path to input video hdf5')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output hdf5, if not set, will overwrite input hdf5')
    parser.add_argument("--visualize", action='store_true', help="If set, visualize the fitted mesh on the image and save to out_folder")
    parser.add_argument("--continue", dest='continue_incomplete', action='store_true', default=False, help="If set, continue processing frames that do not have hand annotations")

    args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.vis_folder, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_file_path = os.listdir(args.video_dir)
    input_file_path = [os.path.join(args.video_dir, f) for f in input_file_path if f.endswith('.h5')]
    

    output_file_path = [os.path.join(args.output_dir, os.path.basename(f)) for f in input_file_path]
    
    for i, file_path in enumerate(input_file_path):
        print(f'Processing {file_path}, saving to {output_file_path[i]}')
        if not os.path.exists(output_file_path[i]) or not args.continue_incomplete:
            shutil.copy(file_path, args.output_dir)
        file = h5py.File(file_path, 'r')
        output_file = h5py.File(output_file_path[i], 'a')
        for frame in tqdm(file.keys(), total=len(file.keys())):
            imgs = [file[frame]['obs'][i] for i in range(file[frame]['obs'].shape[0])]
            if 'hand_annotation' not in output_file[frame].keys():
                output_file[frame].create_group('hand_annotation')
            else:
                if args.continue_incomplete:
                    continue
                    
            hand_annot = predict_hand(detector, model, cpm, renderer, device, imgs, model_cfg, args)
            
            for k in hand_annot.keys():
                if k in output_file[frame]['hand_annotation'].keys():
                    del output_file[frame]['hand_annotation'][k]
                output_file[frame]['hand_annotation'].create_dataset(k, data=hand_annot[k])
            
        output_file.close()
        file.close()
    
    
if __name__ == '__main__':
    main()

