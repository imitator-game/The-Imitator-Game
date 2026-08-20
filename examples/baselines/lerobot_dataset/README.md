# DATASET

<video src="../../../docs/media/real_demos.mp4"></video>

## Data Source

This project uses the human demonstration dataset located in the `demos/demo_data` directory (at the repository root). The dataset contains both the original human demonstration data and the corresponding MANO hand-estimation data. They are not separate datasets; they are different observation features within the same episode/frame data.

The dataset follows the standard LeRobot dataset format, but this project uses a project-maintained LeRobot implementation and several project-specific loaders on top of the native LeRobot APIs. The main files are:

- `data/chunk-*/file-*.parquet`: frame-level data, including timestamps, episode/task indices, and numerical observations.
- `meta/episodes/`: episode-level metadata, such as episode lengths and data chunk/file locations.
- `meta/info.json`: dataset configuration, FPS, splits, path templates, and the dtype/shape of every feature.
- `meta/stats.json`: statistics for numerical features, which can be used for normalization.
- `meta/tasks.parquet`: task IDs and task descriptions.
- `videos/`: video files corresponding to image features. The complete dataset contains the video files; the small example may not include all of them.

## Usage

The dataset is compatible with the native LeRobot dataset format and can be inspected with standard LeRobot metadata utilities. For project-specific loading, preprocessing, training, and evaluation, refer to the implementations under `examples/baselines/lerobot_dataset/`.

![dataset](../../../docs/media/dataset.png)

Set the dataset root to the complete dataset:

```bash
DATA_ROOT=demos/demo_data
```

### Load one frame

The project-maintained LeRobot loader can open a local dataset directly. The
`repo_id` is only the dataset name used by the loader; when `root` already
exists, the data is read locally.

```python
from examples.baselines.lerobot_dataset.lerobot_dataset import LeRobotDataset

DATA_ROOT = "demos/demo_data"
REPO_ID = "human_H41"
ds = LeRobotDataset(repo_id=REPO_ID, root=f"{DATA_ROOT}/{REPO_ID}",
                    download_videos=True, video_backend="pyav")
sample = ds[0]                         # one frame/timestep

rgb = sample["observation.images.cam1"] # tensor, typically [C, H, W]
timestamp = sample["timestamp"]        # seconds from episode start
episode = sample["episode_index"]
```

`sample` is a dictionary. Common keys include `timestamp`, `frame_index`,
`episode_index`, `task`, RGB/depth image keys, and all numerical MANO keys.
The loader uses `episode_index` and `timestamp` from the parquet row to decode
the matching video frame, so RGB/depth and MANO from the same `sample` are
time-aligned. If `delta_timestamps` is configured, the same call can return
stacked observations at the requested relative timestamps.

### Read MANO from a frame

MANO is stored as ordinary numerical features in the same row; it is not a
second dataset and does not require running WiLoR during training.

```python
prefix = "observation.hand.right.cam1"

mano = {
    "global_orient": sample[f"{prefix}.mano_global_orient"], # [3], axis-angle
    "hand_pose":     sample[f"{prefix}.mano_hand_pose"],      # [45], axis-angle
    "betas":         sample[f"{prefix}.mano_betas"],          # [10]
    "cam_t":         sample[f"{prefix}.pred_cam_t_full"],    # [3]
    "joints_3d":     sample[f"{prefix}.pred_keypoints_3d"],  # [63] = 21x3
    "is_right":      sample[f"{prefix}.is_right"],            # 1/0/-1
}

keypoints_3d = mano["joints_3d"].reshape(21, 3)
```

Use the `left`/`right` hand and camera name in the feature prefix to select a
different hand/view. `is_right == -1` marks a missing detection; the
corresponding stored numerical features are zero-filled and should normally
be masked out. `pred_keypoints_3d` is already the 21-joint OpenPose-order
output. If a mesh or a new set of joints is needed, pass `global_orient`,
`hand_pose`, and `betas` to a MANO layer; `cam_t` places the result in the
camera coordinate system. The saved 3D keypoints are raw MANO coordinates and
are not image pixels.

For video-only prompting, the project sampler can instead be used as follows:

```python
from examples.baselines.lerobot_dataset.lerobot_dataloader import (
    LeRobotDataConfig, build_lerobot_dataset,
)

DATA_ROOT = "demos/demo_data"
video_ds = build_lerobot_dataset(LeRobotDataConfig(
    source_type="human", root=DATA_ROOT,
    dataset_file="examples/baselines/lerobot_dataset/config/test_configs/human_test_config.json",
    cameras=["cam1"],
    include_depth=False, num_frames=10, image_size=(224, 224),
))
clip = video_ds[0]       # clip["video"]: sampled RGB clip; no MANO keys here
```

The sampler chooses a task and an episode, samples `num_frames` temporal
indices, decodes the selected camera videos, and applies the configured image
transforms. Its sampling is intentionally episode/clip-oriented; it should not
be used when a model needs the exact MANO row for each frame.

Individual human dataset repositories are usually specified by the project dataset configuration. The `root` field in a configuration is typically a repository subdirectory relative to `DATA_ROOT`. Make sure that the configured `repo_id`, `root`, and train/test episode ranges match the actual dataset structure.

Relevant project implementations include:

- `examples/baselines/lerobot_dataset/lerobot_dataset.py`: project-maintained LeRobot dataset implementation.
- `examples/baselines/lerobot_dataset/lerobot_human_dataset.py`: human video dataset loader.
- `examples/baselines/lerobot_dataset/lerobot_dataloader.py`: unified human/simulation/robot data loader.
- `examples/baselines/lerobot_dataset/config/`: dataset and split configuration files.

## Data Content

The current v2 example contains the following visual inputs:

- RGB: `cam1`, `cam2`, `cam3`, and `zed2i`.
- Depth: `cam1_depth`, `cam2_depth`, `cam3_depth`, and `zed2i_depth`.
- `cam1`, `cam2`, and `cam3` images have resolution `480 x 640`.
- `zed2i` images have resolution `720 x 1280`.
- All video features use 30 FPS.

Each frame also contains standard LeRobot index fields: `timestamp`, `frame_index`, `episode_index`, `index`, and `task_index`.

## MANO

MANO data is estimated from the human demonstration video frames. It is not directly recorded by the original capture devices. The processing pipeline is:

1. Read video frames for each episode and camera.
2. Use a YOLO hand detector to detect hand bounding boxes.
3. Use WiLoR to estimate MANO parameters and 3D hand keypoints for detected hands. The MANO parameters are stored directly, while keypoints can optionally be recomputed through the MANO layer with episode-level mean shape parameters.
4. Store left- and right-hand results as numerical LeRobot features for each timestep and camera.

The feature naming convention is:

```text
observation.hand.<left|right>.<camera>.<feature>
```

The stored MANO-related fields include:

- `mano_global_orient`: 3D axis-angle global orientation converted from WiLoR's rotation-matrix prediction.
- `mano_hand_pose`: 45D axis-angle MANO hand pose for 15 hand joints, with 3 values per joint.
- `mano_betas`: 10D MANO shape coefficients predicted by WiLoR.
- `pred_cam_t_full`: 3D translation in camera coordinates, converted from WiLoR's crop-level camera prediction to the full-image coordinate system.
- `focal_length`: camera focal length vector `(fx, fy)` used for perspective projection.
- `pred_keypoints_3d`: 21 3D hand joints obtained from the MANO layer output after conversion to the OpenPose hand keypoint convention, flattened into 63 values.
- `is_right`: handedness flag; `1` means right hand, `0` means left hand, and `-1` means no detection.

The 21 3D joints follow the OpenPose hand order: wrist, thumb, index, middle, ring, and pinky. Each finger uses the corresponding CMC/MCP/IP/TIP or MCP/PIP/DIP/TIP joints.

When a hand is not detected in a frame, the processing script writes zero-valued default features and sets `is_right` to `-1`. 

The stored `pred_keypoints_3d` are raw MANO joint coordinates.

## Additional Content in v2 Compared with v1

Compared with v1, v2 only adds explicit 3D hand keypoint annotations:

- `pred_keypoints_3d`: 21 3D hand joints derived from the estimated MANO parameters through the MANO layer.

All other observation features, including RGB/depth inputs and MANO parameter fields, remain unchanged.

## MASK

IG-10K provides semantic segmentation masks as optional offline annotations for human, simulation, and robot data. They support dataset inspection, object-centric analysis, and future training objectives, but are not used by the reported policy baselines or benchmark scoring pipeline.

Each segmented RGB feature has a corresponding `_mask` feature:

```text
observation.images.<camera> -> observation.images.<camera>_mask
```

This applies to `cam1`, `cam2`, `cam3`, and `zed2i`; depth streams are not segmented. Mask videos use the same LeRobot chunk/file organization, FPS, and episode timestamps as the source RGB videos. Each decoded frame contains a categorical `uint8` ID map replicated over three RGB channels:

- `0`: background.
- `1..N`: semantic object IDs.
- `meta/mask_labels/<task_id>/<mask_key>/global.json`: ID-to-label mapping.

IDs are local to one task and camera, so consumers must read `global.json` instead of hard-coding them. The videos are encoded losslessly with `libx264rgb`, `rgb24`, and CRF `0`.

The masks are generated offline using Grounded-SAM-2. Human-curated task object inventories provide GroundingDINO text prompts; GroundingDINO detects objects using box threshold `0.35` and text threshold `0.25`; the resulting boxes initialize SAM 2.1 Hiera Large, which propagates masks through each RGB video. The postprocessor converts these predictions into categorical LeRobot videos and adds the corresponding feature definitions, episode-video mappings, and label maps to a copy of the source dataset. Existing observations, actions, states, timestamps, and metadata values are preserved.

These annotations may contain missed objects, false detections, boundary errors, identity swaps, or temporal drift and should not be treated as ground-truth task-success labels.

Loading examples, visualization commands, the full generation configuration, and postprocessing details are documented in [`maskgen/MASK.md`](maskgen/MASK.md).
