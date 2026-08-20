# The Imitator Game: Simulation Framework

<video src="../docs/media/sim_demos.mp4" controls autoplay loop width="100%"></video>

This directory contains the ManiSkill source snapshot used to build the dual-arm simulation environments for **The Imitator Game**. The benchmark-specific code extends ManiSkill with 50 tabletop tasks, four scene-mismatch levels, dual-Panda motion-planning demonstrations, rule-based evaluation, left-right scene mirroring, and multi-view trajectory replay.

Use the vendored `mani_skill` package in this repository. The environment and simulator dependencies are assumed to be configured already; this document only covers the assets and commands required by the benchmark.

Run all commands below from the repository root.

## Benchmark semantics

The level is a property of a demonstration-execution pair, not an input supplied to a policy. Goal predicates are used for simulator-side evaluation only.

| Level | Demonstration-to-execution change | Highest preserved fidelity | Simulation implementation |
| --- | --- | --- | --- |
| L0 | Same task-relevant objects and layout | Trajectory-level imitation | Base environment |
| L1 | Same objects, rearranged layout | Final object-state imitation | Base environment with spatial offsets |
| L2 | Same task semantics, different object instances | Semantic task completion | Base environment with instance substitution and task-specific mirroring |
| L3 | Different object semantics, reusable affordances | Underlying intent through affordance adaptation | Independent `L3` environment |

L0, L1, and L2 share one registered environment class. L1/L2 are selected before the environment is created. L3 has a separate registered class because its objects, success conditions, and motion-planning solution may differ structurally from the base task.

For example:

| Level | Gymnasium environment ID |
| --- | --- |
| L0 | `TwoRobotStirSpoon-v1` |
| L1 | `TwoRobotStirSpoon-v1` |
| L2 | `TwoRobotStirSpoon-v1` |
| L3 | `TwoRobotStirSpoonL3-v1` |

Names such as `L2_TwoRobotStirSpoon-v1` are dataset and trajectory aliases, not Gymnasium environment IDs. The complete alias mapping is stored in [`examples/baselines/lerobot_dataset/task_mapping.json`](../examples/baselines/lerobot_dataset/task_mapping.json).

## Components

| Component | Location | Responsibility |
| --- | --- | --- |
| Base task environments | [`envs/tasks/tabletop/dual_tasks`](envs/tasks/tabletop/dual_tasks) | L0 scene, L1 rearrangement, L2 instance substitution, task success, and dense reward |
| L3 task environments | [`envs/tasks/tabletop/dual_tasks_l3`](envs/tasks/tabletop/dual_tasks_l3) | Standalone affordance-adapted tasks |
| Level and mirror utilities | [`envs/tasks/tabletop/utils/L0_L3_utils.py`](envs/tasks/tabletop/utils/L0_L3_utils.py) | Runtime level switches, object substitutions, pose transforms, and mirror configuration |
| Shared camera utilities | [`envs/tasks/tabletop/utils/dual_task_camera_utils.py`](envs/tasks/tabletop/utils/dual_task_camera_utils.py) | External four-view replay overrides and deterministic level setup |
| Reward utilities | [`envs/tasks/tabletop/utils/reward_utils.py`](envs/tasks/tabletop/utils/reward_utils.py) | Normalized phase rewards and subgoal progress tracking |
| Base motion-planning solutions | [`examples/motionplanning/dual/solutions`](examples/motionplanning/dual/solutions) | One scripted dual-arm solution per base task |
| L3 motion-planning solutions | [`examples/motionplanning/dual/solutions_l3`](examples/motionplanning/dual/solutions_l3) | One scripted dual-arm solution per L3 task |
| Motion-planning runner | [`examples/motionplanning/dual/two_robot_run.py`](examples/motionplanning/dual/two_robot_run.py) | Runs a task solver and records successful trajectories |
| Dataset scheduler | [`collect_data.py`](../scripts/collect_data.py) | Collects selected tasks and levels across available GPUs, with resume and retry support |
| Multi-view replay | [`scripts/replay_dual_task_multiview.py`](../scripts/replay_dual_task_multiview.py) | Replays recorded states and exports one video per camera |
| End-to-end multi-view batch | [`scripts/run_dual_task_multiview_batch.py`](../scripts/run_dual_task_multiview_batch.py) | Generates trajectories and renders their camera views sequentially |

## Required assets

The benchmark uses four asset namespaces. `MS_ASSET_DIR` can override the default `~/.maniskill` root; the paths below show the default layout.

| Asset set | Required location |
| --- | --- |
| ManiSkill YCB | `~/.maniskill/data/assets/mani_skill2_ycb/` |
| RoboTwin objects | `~/.maniskill/data/robotwin/objects/` |
| PartNet-Mobility | `~/.maniskill/data/partnet_mobility/dataset/` |
| External GLB objects | `~/.maniskill/data/sketchfab/objects/{object_key}/model.glb` |

Download the standard YCB assets with:

```bash
python -m mani_skill.utils.download_asset ycb
```

Download the RoboTwin archives and extract them under the same directory:

```bash
python -m mani_skill.utils.download_robotwin
unzip -o ~/.maniskill/data/robotwin/objects.zip -d ~/.maniskill/data/robotwin
unzip -o ~/.maniskill/data/robotwin/background_texture.zip -d ~/.maniskill/data/robotwin
```

PartNet-Mobility tasks expect the category/model directory hierarchy produced by the repository's PartNet downloader:

```bash
python -m mani_skill.utils.download_partnet
```

### Standalone scale and weight GLBs

Three L3 environments use individual GLB assets that are not included in YCB, RoboTwin, or PartNet-Mobility:

| Object key | Source | Used by |
| --- | --- | --- |
| `balance_scale` | Hunyuan-generated project asset | `TwoRobotPickAppleToScaleL3-v1`, `TwoRobotPlaceFoodScaleL3-v1`, `TwoRobotPutCubeOnScaleL3-v1` |
| `weight_on_scale` | Sketchfab [`Weight`](https://sketchfab.com/3d-models/weight-20226c0b61c449d49c88d8adfa46019e) asset | `TwoRobotPickAppleToScaleL3-v1`, `TwoRobotPlaceFoodScaleL3-v1` |

Download both objects from the [shared GLB asset folder](https://drive.google.com/drive/folders/1W2Zmi3m_0pXwNnj8iwSR_5oBEc1Y-hwm?usp=sharing), then place or rename the downloaded files exactly as follows:

```text
~/.maniskill/data/sketchfab/objects/balance_scale/model.glb
~/.maniskill/data/sketchfab/objects/weight_on_scale/model.glb
```

The loader resolves the stable object keys through [`assets/sketchfab_registry.json`](assets/sketchfab_registry.json). These files must be downloaded manually: the `balance_scale` registry entry has no public direct-download URL, while the `weight_on_scale` entry records its source page rather than a direct GLB endpoint. The complete loader and asset-registration convention is documented in [`assets/sketchfab_README.md`](assets/sketchfab_README.md).

## Create and step an environment

Call `configure_dual_task_level` before `gym.make`. This resets all process-global level and mirror switches, then enables the requested level. Call it again whenever a process changes levels.

```python
import gymnasium as gym
import mani_skill.envs  # Registers all benchmark environments.

from mani_skill.envs.tasks.tabletop.utils.dual_task_camera_utils import (
    configure_dual_task_level,
)

level = "L2"
base_env_id = "TwoRobotStirSpoon-v1"
env_id = base_env_id.replace("-v1", "L3-v1") if level == "L3" else base_env_id

configure_dual_task_level(level)
env = gym.make(
    env_id,
    obs_mode="state",
    control_mode="pd_joint_pos",
    reward_mode="dense",
    sim_backend="physx_cpu",
)

obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

evaluation = env.unwrapped.evaluate()
print(evaluation["success"])
env.close()
```

Each environment contains two `panda_wristcam` agents. The unflattened action space is a dictionary keyed by `panda_wristcam-0` and `panda_wristcam-1`; sampling the environment action space, as above, returns the required two-arm structure.

### Camera configuration

All 50 base environments and all 50 L3 environments expose two independent constructor arguments:

- `hi_res` controls the external scene cameras. It defaults to `True`.
- `wrist_sensor` controls the two robot-mounted wrist cameras. It defaults to `True`.

`hi_res` changes both the external-camera resolution and the number of external cameras:

| `hi_res` | External scene cameras |
| --- | --- |
| `True` (default) | `cam1`, `cam2`, and `cam3` at 640x480, plus `zed2i` at 1280x720 |
| `False` | Only `zed2i` at 224x224 |

When `wrist_sensor=True`, each `panda_wristcam` contributes one 128x128 `hand_camera`. In recorded H5 observations, the two mounted sensors are named `panda_wristcam_0_hand_camera` and `panda_wristcam_1_hand_camera`. Setting `wrist_sensor=False` removes their images without changing the two robot IDs, proprioception, or action space.

The resulting observation-camera combinations are:

| `hi_res` | `wrist_sensor` | Recorded RGB-D views |
| --- | --- | --- |
| `True` | `True` | Four external scene cameras and two wrist cameras (six views; default) |
| `True` | `False` | Four external scene cameras |
| `False` | `True` | One 224x224 `zed2i` and two wrist cameras |
| `False` | `False` | One 224x224 `zed2i` |

Pass the options directly to `gym.make`. For example, the lightweight single-view configuration is:

```python
env = gym.make(
    "TwoRobotStirSpoon-v1",
    obs_mode="rgbd",
    render_mode="sensors",
    control_mode="pd_joint_pos",
    hi_res=False,
    wrist_sensor=False,
)
```

`scripts/collect_data.py` and `two_robot_run` do not override these constructor values, so their current default RGB-D collection uses `hi_res=True` and `wrist_sensor=True`. The four-view replay tools are different: they export only the external `cam1`, `cam2`, `cam3`, and `zed2i` views, not the wrist cameras.

### L2 left-right mirroring

L2 enables the task-specific object substitutions and the left-right scene mirror by default. The mirror is applied after episode initialization, so all task actors and articulations are transformed consistently. Robot root poses are mirrored by default as well.

Keep the scene mirror but preserve the original robot sides with:

```python
configure_dual_task_level("L2", mirror_robot_pose=False)
```

The motion-planning runner exposes the same choice through `--no-mirror-robot-pose`. When a mirrored episode is recorded, `RecordEpisode` stores `lr_mirror_applied` in its metadata and canonicalizes the paired wrist-camera/action ordering.

## Evaluation and reward

Every task implements `evaluate()`. Its `success` field is the final rule-based simulator metric used for demonstration filtering and automated evaluation. Task-specific fields expose intermediate predicates such as reached, grasped, transported, placed, or returned.

Dense rewards are built from ordered phases. `RewardTracker` keeps the peak completion of each phase and produces bounded progress signals; these phase values also support subgoal-level analysis. They are evaluation and training signals and must not be provided as policy conditioning inputs in the Imitator Game protocol.

Use `reward_mode="dense"` to receive the phase reward, `reward_mode="sparse"` for success-based reward, or `reward_mode="none"` when only states/actions are being recorded.

## Generate one motion-planning trajectory

The runner selects the registered solver from `MP_SOLUTIONS`, resets the requested level, executes both robot planners, evaluates the final state, and writes successful episodes through `RecordEpisode`.

L0:

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotStirSpoon-v1 --l0 \
  -n 1 --only-count-success \
  --traj-name L0_TwoRobotStirSpoon-v1 \
  --record-dir demos
```

L1 and L2 use the same environment ID with their corresponding switch:

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotStirSpoon-v1 --l1 \
  -n 1 --only-count-success \
  --traj-name L1_TwoRobotStirSpoon-v1 \
  --record-dir demos

python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotStirSpoon-v1 --l2 \
  -n 1 --only-count-success \
  --traj-name L2_TwoRobotStirSpoon-v1 \
  --record-dir demos
```

L3 uses its standalone environment and an explicit trajectory name. Do not pass the legacy `--l3` switch to a standalone L3 environment; the L3 scene is defined by the `...L3-v1` class itself.

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run \
  -e TwoRobotStirSpoonL3-v1 \
  -n 1 --only-count-success \
  --traj-name L3_TwoRobotStirSpoon-v1 \
  --record-dir demos
```

Add `--save-video` to store the runner's render video. Add `--vis` only when an interactive viewer is required.

The output pair is:

```text
demos/{env_id}/motionplanning/{trajectory_name}.h5
demos/{env_id}/motionplanning/{trajectory_name}.json
```

The HDF5 file contains actions, environment states, and the observations selected by `obs_mode`. The JSON file records environment arguments, reset seeds, control mode, episode results, and mirror metadata. Keep both files together: replay requires the JSON file with the same basename.

## Collect benchmark demonstrations

Use the scheduler for resumable collection. This example collects ten successful trajectories for one task at all four levels on GPU 0:

```bash
python scripts/collect_data.py \
  --demos-dir demos \
  --target-episodes 10 \
  --tasks TwoRobotStirSpoon \
  --levels L0 L1 L2 L3 \
  --gpu-ids 0 \
  --max-procs-per-gpu 1
```

Omit `--tasks` and `--levels` to use all 50 tasks and all four levels. The scheduler:

1. maps L0-L2 to `{task}-v1` and L3 to `{task}L3-v1`;
2. counts successful episodes already present in the companion JSON;
3. launches only the missing trajectories;
4. retries failed or stalled jobs; and
5. writes progress to `demos/generation_stats.json` and logs to the collection log directory.

Inspect the planned task-level jobs without launching simulation:

```bash
python scripts/collect_data.py --demos-dir demos --dry-run
```

## Convert demonstrations to LeRobot and load them in the baselines

The training data path after simulation is:

```text
motion-planning H5
  -> h5_to_lerobot.py
  -> one LeRobot dataset per L{0,1,2,3}_{task}-v1
  -> experiment split config + task_mapping.json
  -> HumanSimPairedDataset
  -> baseline-specific adapter
```

The shared contract is implemented in [`examples/baselines/lerobot_dataset`](../examples/baselines/lerobot_dataset). The baselines share the converted datasets, task mapping, split files, and normalized sample fields. Architectures that need a different tensor layout add a thin adapter around this contract; the simulator data does not need to be converted again for each model.

### Converter input contract

[`h5_to_lerobot.py`](../examples/baselines/lerobot_dataset/h5_to_lerobot.py) reads the following fields from each trajectory group in the H5 file:

| H5 field | Required shape | Meaning |
| --- | --- | --- |
| `actions` | `(N, 16)` | Seven arm joints and one gripper command for each robot |
| `obs/agent/panda_wristcam-0/qpos` | `(N+1, 9)` | Left arm and two-finger gripper state |
| `obs/agent/panda_wristcam-1/qpos` | `(N+1, 9)` | Right arm and two-finger gripper state |
| `obs/sensor_data/<camera>/rgb` | `(N+1, H, W, 3)` | RGB observations for each recorded camera |
| `obs/sensor_data/<camera>/depth` | `(N+1, H, W, 1)` | Depth observations for each recorded camera |

Therefore, trajectories intended for training must be collected with `obs_mode="rgbd"`. This is the default of `two_robot_run` and `scripts/collect_data.py`; do not add `-o none` to those collection commands.

With the current environment defaults, those RGB-D H5 files contain the four external scene cameras and both wrist cameras. Use the `hi_res` and `wrist_sensor` combinations above when creating an environment directly. The batch collection CLIs currently expose neither option and therefore use the environment defaults.

The H5 files produced by `scripts/run_dual_task_multiview_batch.py` are a different intermediate representation: that script intentionally uses `-o none`, stores environment states, and then renders the selected cameras by replay. Those H5 files are suitable for replay and video generation, but they do not contain the observations required by `h5_to_lerobot.py`.

The companion JSON is not read during LeRobot conversion. It should still be retained because simulator replay uses it to reconstruct the environment and episode metadata.

### Run the conversion

First inspect all H5 files that the recursive conversion would process:

```bash
python -m examples.baselines.lerobot_dataset.h5_to_lerobot \
  --input demos \
  --output-dir demos/imitator_data \
  --recursive \
  --no-gpu \
  --dry-run
```

Then convert them:

```bash
python -m examples.baselines.lerobot_dataset.h5_to_lerobot \
  --input demos \
  --output-dir demos/imitator_data \
  --recursive \
  --no-gpu \
  --n-jobs 4 \
  --mem-per-proc 4 \
  --fps 30
```

`--n-jobs` controls parallel H5 conversions and `--mem-per-proc` is the estimated memory budget in GiB for each worker. Increase the worker count only when the machine has enough RAM and storage bandwidth. GPU workers are optional; `--no-gpu` is sufficient for this conversion. Completed datasets are skipped on subsequent runs. Use `--force` only when an existing converted dataset must be rebuilt.

The trajectory name becomes the LeRobot repository ID and output directory. For example:

```text
demos/TwoRobotStirSpoon-v1/motionplanning/L2_TwoRobotStirSpoon-v1.h5
  -> demos/imitator_data/L2_TwoRobotStirSpoon-v1/
```

Keep the uppercase `L0_` through `L3_` names generated by `scripts/collect_data.py`. They are the IDs referenced by the released task mapping and experiment configs.

### Converted sample fields

Each H5 trajectory becomes one LeRobot episode. The converter writes the LeRobot v0.5 layout, including Parquet data, encoded camera videos, per-dataset statistics, episode metadata, and split metadata.

| LeRobot feature | Shape | Construction |
| --- | --- | --- |
| `observation.images.<camera>` | `(H, W, 3)` | RGB frame from each camera present in the H5 file |
| `observation.images.<camera>_depth` | `(H, W, 3)` | Depth converted to the video-compatible RGB representation |
| `observation.qpos_gripper_states` | `(18,)` | Left 9-dimensional qpos followed by right 9-dimensional qpos |
| `action.qpos_gripper_actions` | `(16,)` | Left 8-dimensional action followed by right 8-dimensional action |

There are `N+1` observations but only `N` simulator actions. The converter repeats the final action once so every LeRobot frame has an aligned action. External camera names are preserved. The mounted H5 sensors `panda_wristcam_0_hand_camera` and `panda_wristcam_1_hand_camera` are normalized to the LeRobot camera names `wristcam0` and `wristcam1`, respectively. A baseline still decodes only the cameras listed in its data configuration; for example, `cameras=["zed2i"]` ignores the other converted videos.

### Select datasets and pair tasks

Conversion creates the physical datasets; the JSON configuration files decide which of them enter a particular experiment:

- [`task_mapping.json`](../examples/baselines/lerobot_dataset/task_mapping.json) maps each human demonstration ID to the corresponding L0-L3 simulation repository IDs. For example, `human_H1` is paired with all four `TwoRobotStirSpoon` levels.
- [`config/exp_configs`](../examples/baselines/lerobot_dataset/config/exp_configs) contains the released 15-, 30-, and 45-task training splits and the seen/unseen evaluation splits.
- [`task_desc`](../examples/baselines/lerobot_dataset/task_desc) contains the human, simulation, and robot language descriptions used by language-conditioned modes.

A simulation config entry has the following form:

```json
{
  "repo_id": "L2_TwoRobotStirSpoon-v1",
  "root": "L2_TwoRobotStirSpoon-v1",
  "train": "0:50",
  "test": ""
}
```

`root` is relative to the simulation data root passed to training. The episode range must exist in the converted dataset: if fewer than 50 episodes were collected, update the range instead of copying `0:50` unchanged.

### Unified loader interface

[`build_lerobot_dataset`](../examples/baselines/lerobot_dataset/lerobot_dataloader.py) is the common single-source loader. [`HumanSimPairedDataset`](../examples/baselines/lerobot_dataset/lerobot_paired_dataset.py) uses it for both sides of a human-simulation pair and applies the task mapping:

```python
from examples.baselines.lerobot_dataset.lerobot_paired_dataset import (
    HumanSimPairedDataset,
    PairedDatasetConfig,
)

config = PairedDatasetConfig(
    human_root="demos/demo_data",
    sim_root="demos/imitator_data",
    human_dataset_file=(
        "examples/baselines/lerobot_dataset/config/exp_configs/"
        "human_train_config_15.json"
    ),
    sim_dataset_file=(
        "examples/baselines/lerobot_dataset/config/exp_configs/"
        "sim_train_config_15.json"
    ),
    task_mapping_file="examples/baselines/lerobot_dataset/task_mapping.json",
    human_task_description_file=(
        "examples/baselines/lerobot_dataset/task_desc/human_desc.json"
    ),
    sim_task_description_file=(
        "examples/baselines/lerobot_dataset/task_desc/sim_desc.json"
    ),
    split="train",
    cameras=["zed2i"],
    include_depth=False,
    horizon=16,
    obs_horizon=1,
    state_type="qpos",
    input_mode="video_only",
)

dataset = HumanSimPairedDataset(config)
sample = dataset[0]
```

The common sample contains `robot_obs` (states and `view_1`, `view_2`, ...), `robot_actions`, `human_task_id`, `sim_task_id`, `dataset_idx`, and `sample_id`. Depending on `input_mode`, it also contains a sampled `human_video`, a language description, or both. Skill baselines additionally request `skill_frames`. State and action normalization is computed from each converted simulation dataset's statistics.

The experiment level is used to select and pair a simulation repository through `sim_task_id`; it is not passed to the policy as an explicit privileged level label.

The released baseline implementations consume this contract as follows:

- ACT, Diffusion Policy, and VQ-BeT use `HumanSimPairedDataset` directly.
- OpenVLA-OFT, pi0.5, and RDT adapt the paired sample to their model-specific token, tensor, or JAX layouts.
- XSkill and UniSkill enable the skill-frame fields through wrappers around the paired interface.
- GR00T uses its own episode loader but reads the same LeRobot directories, experiment configs, task mapping, and descriptions.

The common training arguments are therefore the human and simulation data roots, their experiment config files, `task_mapping.json`, the task-description files, camera selection, observation/action horizons, and `input_mode`. Model-specific commands and hyperparameters are documented in the policy READMEs under [`examples/baselines/`](../examples/baselines/); conversion should be run once before following those training instructions.

## Render the four external benchmark views

The external scene-camera set is `cam1`, `cam2`, `cam3`, and `zed2i`. With the default `hi_res=True`, the first three are 640x480 and `zed2i` is 1280x720. These cameras are now defined directly by every base and L3 task class; `patch_dual_task_camera_defaults` remains useful for replaying only a selected external camera or overriding its output resolution. It does not add or export either wrist camera.

To create an environment with exactly the four external views and no wrist views:

```python
import gymnasium as gym
import mani_skill.envs

from mani_skill.envs.tasks.tabletop.utils.dual_task_camera_utils import configure_dual_task_level

configure_dual_task_level("L2")
env = gym.make(
    "TwoRobotStirSpoon-v1",
    obs_mode="rgb",
    render_mode="sensors",
    control_mode="pd_joint_pos",
    hi_res=True,
    wrist_sensor=False,
)
obs, info = env.reset(seed=0)
```

Set `wrist_sensor=True` to include the two additional 128x128 mounted views in the environment observation. The replay command below still writes only the selected external views.

To render an existing trajectory into one MP4 per camera:

```bash
python scripts/replay_dual_task_multiview.py \
  --traj-path demos/TwoRobotStirSpoon-v1/motionplanning/L2_TwoRobotStirSpoon-v1.h5 \
  --level L2 \
  --output-dir demos/multiview \
  --cameras cam1 cam2 cam3 zed2i
```

Replay restores each saved `env_states` frame instead of re-running motion planning. Videos are written as:

```text
{output_dir}/{camera}/{task}/{level}_{env_id}_{camera}_ep{episode_id}.mp4
```

To generate trajectories and all selected camera videos in one sequential job:

```bash
python scripts/run_dual_task_multiview_batch.py \
  --tasks TwoRobotStirSpoon-v1 TwoRobotPlaceBookBookcase-v1 \
  --levels L0 L1 L2 L3 \
  --cameras cam1 cam2 cam3 zed2i \
  --count 1 \
  --out-root demos/dual_task_multiview
```

The batch command stores resumable status in `batch_progress.json` and an append-only event log in `batch_progress.jsonl`. Existing complete trajectories and videos are skipped by default.

## Supported benchmark tasks

The canonical task list is `DEFAULT_TASKS` in [`collect_data.py`](../scripts/collect_data.py). Each base name below supports `{name}-v1` for L0-L2 and `{name}L3-v1` for L3.

<details>
<summary>Show all 50 task base names</summary>

1. `TwoRobotStirSpoon`
2. `TwoRobotPlaceBookBookcase`
3. `TwoRobotPlaceClothBasket`
4. `TwoRobotPickRemoteControl`
5. `TwoRobotPlaceMagazineFolder`
6. `TwoRobotPickWash`
7. `TwoRobotPickAppleBasket`
8. `TwoRobotPickAppleBananaToBaskets`
9. `TwoRobotPickAppleToScale`
10. `TwoRobotPlaceChipsRack`
11. `TwoRobotPlaceCommodityRack`
12. `TwoRobotPlaceFruitBox`
13. `TwoRobotScanMilkBox`
14. `TwoRobotPlacePlateRack`
15. `TwoRobotCutFruit`
16. `TwoRobotPickFruitsToPlate`
17. `TwoRobotPourKetchupFries`
18. `TwoRobotWipePot`
19. `TwoRobotCleanDesk`
20. `TwoRobotPourKettle`
21. `TwoRobotPickFood`
22. `TwoRobotTransFood`
23. `TwoRobotPlaceFoodScale`
24. `TwoRobotPressStapler`
25. `TwoRobotPlaceBurgerTray`
26. `TwoRobotPickTennisBallGolfBall`
27. `TwoRobotScanPillBottle`
28. `TwoRobotPlaceShoeBox`
29. `TwoRobotOpenBox`
30. `TwoRobotFoldBox`
31. `TwoRobotPickPillToRegions`
32. `TwoRobotPlaceFileFolder`
33. `TwoRobotPourLiquidCup`
34. `TwoRobotPlacePillBox`
35. `TwoRobotPlaceBrushRest`
36. `TwoRobotPlaceCupPlate`
37. `TwoRobotPlaceScrewdriver`
38. `TwoRobotPlaceMugRack`
39. `TwoRobotPourLiquidMug`
40. `TwoRobotPutCubeOnScale`
41. `TwoRobotPourCup`
42. `TwoRobotCleanCup`
43. `TwoRobotOpenLiquidCap`
44. `TwoRobotGrindFood`
45. `TwoRobotPourLiquidFilter`
46. `TwoRobotPutBox`
47. `TwoRobotKnifeBowlFork`
48. `TwoRobotLiftLidFromSkillet`
49. `TwoRobotFoldTowel`
50. `TwoRobotPressJuicer`

## Task implementation contract

Each released task is a matched set of four components:

1. one base environment registered as `TwoRobot{Task}-v1`;
2. one standalone L3 environment registered as `TwoRobot{Task}L3-v1`;
3. one base and one L3 motion-planning solution registered in `MP_SOLUTIONS`; and
4. four aliases in the dataset task mapping.

The environment owns scene construction, episode initialization, success predicates, and reward phases. Level utilities own only the controlled L1/L2 changes shared across tasks. Motion-planning solutions access task geometry through the environment and produce the same two-agent `pd_joint_pos` action interface used by recorded demonstrations.

## Adding a new task (community contribution)

New tasks contributed to the community must follow the same contract. Start
from the templates:

- **Environments (L0–L3)**: [`envs/tasks/_template/`](envs/tasks/_template/README.md)
  provides a base `TwoRobotTemplateTask-v1` (L0/L1/L2) class, a standalone
  `TwoRobotTemplateTaskL3-v1` class, and the object-loading helpers for the
  YCB / RoboTwin / PartNet / sketchfab asset namespaces.
- **Robot agents**: [`agents/robots/_template/`](agents/robots/_template/README.md)
  shows how to register a new URDF robot.
- **Motion-planning solutions**: 
  [`examples/motionplanning/dual/`](examples/motionplanning/dual/README.md)
  explains how to write and register a `solve()` for the new task.

Keep the released-task conventions: same `__init__` signature (`hi_res`,
`wrist_sensor`, `robot_init_qpos_noise`), same camera sets, `RewardTracker`
phases, and rule-based `evaluate()` with a `success` field. This keeps the
collection pipeline (`two_robot_run`, `collect_data`, `h5_to_lerobot`),
the level utilities, and the dataset task mapping working for contributed
tasks without changes.
