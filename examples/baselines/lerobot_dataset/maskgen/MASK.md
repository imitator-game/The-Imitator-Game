# IG-10K Segmentation Masks

This document describes the segmentation-mask annotations shipped with IG-10K: where the current internal builds are stored, how masks are represented and loaded, how to visualize them, and how the Grounded-SAM-2 outputs are converted into LeRobot video features.

The masks are offline auxiliary annotations. They are intended for dataset inspection, object-centric analysis, and future training objectives. They are not policy inputs in the reported baselines and are not used by the benchmark scoring pipeline.

## Current dataset locations

The current internal server locations are:

| Domain | With-mask dataset root |
| --- | --- |
| Human | `imitator_human_v1_withmask` |
| Simulation | `imitator_sim_withmask` |
| Robot | `imitator_robot_withmask` |

These are build-time locations, not paths consumers should hard-code. The released roots may be moved without changing the dataset layout.

## Released mask representation

For an RGB video key such as

```text
observation.images.cam1
```

the corresponding mask feature is

```text
observation.images.cam1_mask
```

The four standard RGB views are:

```text
observation.images.cam1
observation.images.cam2
observation.images.cam3
observation.images.zed2i
```

Depth streams are not segmented.

Mask videos use the same chunk/file organization and episode timestamps as the corresponding RGB videos:

```text
<task_root>/
├── meta/
│   ├── info.json
│   ├── episodes/chunk-*/file-*.parquet
│   └── mask_labels/
│       └── <task_id>/
│           └── observation.images.<camera>_mask/
│               └── global.json
└── videos/
    ├── observation.images.<camera>/
    │   └── chunk-*/file-*.mp4
    └── observation.images.<camera>_mask/
        └── chunk-*/file-*.mp4
```

Each decoded mask frame is an RGB image whose three channels contain the same integer label-ID map:

- `0`: background;
- `1..N`: semantic object IDs;
- the ID-to-label mapping is stored in `global.json`.

IDs are stable within one task and camera, but must not be assumed to have the same meaning across different tasks or cameras. Read `global.json` instead of hard-coding IDs. When two predicted masks overlap, the later instance in the Grounded-SAM-2 record order takes precedence in the overlapping pixels.

Mask videos are encoded with the following lossless-oriented settings:

```text
codec: libx264rgb
pixel format: rgb24
CRF: 0
preset: ultrafast
```

The label ID is replicated over three channels because LeRobot represents image features as videos. The mask is still a categorical ID map, not a natural RGB image and not a palette visualization.

## Loading a mask video directly

The following example decodes one mask video and resolves its labels:

```python
import json
from pathlib import Path

import av
import numpy as np

task_root = Path("/path/to/withmask/robot_H10_L0")
mask_key = "observation.images.zed2i_mask"
mask_video = (
    task_root
    / "videos"
    / mask_key
    / "chunk-000"
    / "file-000.mp4"
)
label_file = (
    task_root
    / "meta"
    / "mask_labels"
    / task_root.name
    / mask_key
    / "global.json"
)

id_to_label = {
    int(key): value
    for key, value in json.loads(label_file.read_text()).items()
}

with av.open(str(mask_video)) as container:
    frame = next(container.decode(video=0))
    mask_rgb = frame.to_ndarray(format="rgb24")

mask_ids = mask_rgb[..., 0].astype(np.uint8)
assert np.array_equal(mask_rgb[..., 0], mask_rgb[..., 1])
assert np.array_equal(mask_rgb[..., 0], mask_rgb[..., 2])

for object_id in np.unique(mask_ids):
    if object_id != 0:
        binary_mask = mask_ids == object_id
        print(object_id, id_to_label[int(object_id)], binary_mask.sum())
```

The episode parquet files contain the corresponding mask-video columns:

```text
videos/<mask_key>/chunk_index
videos/<mask_key>/file_index
videos/<mask_key>/from_timestamp
videos/<mask_key>/to_timestamp
```

They mirror the source RGB camera mappings, so episode/frame alignment follows the normal LeRobot video-loading path.

## Visualization

Visualization files are derived quality-control artifacts and are not the categorical mask representation used for training.

### Overlay one RGB/mask video pair

This is the most useful way to inspect spatial alignment:

```bash
python examples/baselines/lerobot_dataset/maskgen/overlay_withmask_video.py \
  --task-root /path/to/withmask/robot_H10_L0 \
  --video-key observation.images.zed2i \
  --chunk-index 0 \
  --file-index 0 \
  --output-video /tmp/robot_H10_L0_zed2i.overlay.mp4
```

The default overlay alpha is `0.45`, and the legend is read from `global.json`.

### Sample overlay frames across a dataset

This command writes uniformly sampled overlay images to a separate directory:

```bash
python examples/baselines/lerobot_dataset/maskgen/batch_overlay_sample_frames.py \
  --root /path/to/withmask \
  --output-root /tmp/ig10k_mask_overlay_samples \
  --mode batch \
  --video-keys observation.images.zed2i \
  --num-frames 5
```

Tasks without mask features are skipped automatically.

### Generate mask-only color videos

```bash
python examples/baselines/lerobot_dataset/maskgen/generate_vis_from_mask_videos.py \
  --root /path/to/withmask/robot_H10_L0 \
  --mode single
```

This writes `file-*.vis.mp4` next to the categorical mask videos. Run it on a working copy, or remove the derived visualization videos before packaging the release.

## How the masks were generated

The released annotations follow the Grounded-SAM-2 procedure described in the paper:

1. Annotators inspect the episodes for each task and provide the relevant human, robot, and simulation object inventories.
2. The inventories are stored as task-conditioned text prompts in [`object_list_all.json`](object_list_all.json).
3. GroundingDINO detects prompt-matched objects in the first RGB frame of each episode-camera stream.
4. The detected boxes initialize SAM 2.1 masks.
5. SAM 2.1 propagates those masks through the complete episode.
6. Per-frame binary masks and a JSONL index are written as an intermediate annotation.
7. The postprocessor converts those instances into categorical LeRobot mask videos and adds the corresponding metadata to a copy of the source task.

### Generation Configuration

| Setting | Value |
| --- | --- |
| Detector | GroundingDINO SwinT OGC |
| Video segmenter | SAM 2.1 Hiera Large |
| SAM checkpoint | `sam2.1_hiera_large.pt` |
| SAM config | `configs/sam2.1/sam2.1_hiera_l.yaml` |
| GroundingDINO checkpoint | `groundingdino_swint_ogc.pth` |
| Box confidence threshold | `0.35` |
| Text confidence threshold | `0.25` |
| Input modalities | RGB only; depth excluded |
| Detection frame | First frame of each episode-camera stream |
| Temporal masks | SAM 2 video propagation over the complete episode |
| Intermediate mask storage | NumPy packbits in compressed `.npz` |
| Per-frame RGB dumps | Disabled |
| Per-frame visualization grids | Disabled |

The box and text thresholds are the conservative defaults reported in the paper. The CLI supports explicit overrides for controlled reruns, but a rerun should retain `0.35` and `0.25` unless its run manifest documents otherwise.

The robot production helper additionally enables prompt-aware multi-instance handling:

```text
max boxes per label: 3
infer boxes per label from prompt: enabled
```

This allows prompts such as `two tennis balls` or comma-separated, appearance-specific objects to retain the required number of detections while still capping duplicate boxes.

### Generation command

```bash
python examples/baselines/lerobot_dataset/maskgen/mask_gen_whole.py \
  --dataset-root /path/to/lerobot_dataset \
  --output-root /path/to/maskgen_output \
  --object-list examples/baselines/lerobot_dataset/maskgen/object_list_all.json \
  --task-domain robot \
  --object-group auto \
  --gs2-root /path/to/Grounded-SAM-2 \
  --sam2-checkpoint /path/to/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --gdino-config /path/to/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --gdino-checkpoint /path/to/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth \
  --bert-base-path /path/to/bert-base-uncased/snapshot \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --max-boxes-per-label 3 \
  --infer-boxes-per-label-from-prompt \
  --video-backend torchcodec \
  --mask-storage packbits \
  --no-save-frame-images \
  --no-save-vis-grid
```

Use `--task-domain human` or `--task-domain sim` for the other namespaces. Environment setup and the tested CUDA 12.1/13.0 stacks are documented in [`README.md`](README.md).

### Intermediate maskgen layout

```text
<maskgen_root>/<task_id>/episode_000/observation.images.cam1/
├── episode_000.jsonl
├── frame_000000/
│   ├── instance_01_mask.npz
│   └── instance_02_mask.npz
├── frame_000001/
│   └── ...
└── ...
```

Each JSONL record contains:

```json
{
  "task_id": "robot_H10_L0",
  "episode_id": 0,
  "video_key": "observation.images.cam1",
  "frame_idx": 0,
  "label": "object label",
  "mask_path": "frame_000000/instance_01_mask.npz"
}
```

Packbits `.npz` files contain:

```text
encoding = ["packbits_little"]
shape = [height, width]
packed = packed binary pixels
```

## Postprocessing into LeRobot

`postprocess_maskgen_whole_to_lerobot.py` never edits the source dataset. It first copies source tasks to a new dataset root and then:

1. collects normalized labels for each task/camera;
2. assigns sorted task/camera-local IDs `1..N`;
3. renders one categorical ID frame for every source RGB frame;
4. encodes the ID frames as lossless RGB mask videos;
5. adds mask features to the copied `meta/info.json`;
6. adds mask video mapping columns to the copied episode parquet files;
7. writes the task/camera `global.json` label map.

Existing RGB/depth videos, actions, states, timestamps, and other source features are not altered. The copied parquet files are physically rewritten to append mask columns, so their file hashes change even though all existing columns and values are preserved.

## Quality control and limitations

The annotation workflow uses:

- confidence filtering in GroundingDINO;
- sampled RGB/mask overlays for visual inspection;
- audits for missing or empty episode-camera JSONLs;
- manual filtering of problematic results before release.

These masks remain automatically generated annotations and may contain missed objects, false detections, boundary errors, identity swaps, or propagation drift. They should not be treated as ground-truth benchmark success labels. The benchmark metrics and Arena judgments are independent of this annotation layer.