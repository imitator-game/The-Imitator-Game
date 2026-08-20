# The Imitator Game

---

## [[Website](https://imitator-game.github.io/index.html)]  |  [[Document](https://imitator-game.github.io/docs/index.html)]  |  [[Paper]()]  |  [[Imitator Arena]()]  |  [[IG-10K Dataset](https://huggingface.co/datasets/imitator-game/IG-10K-Dataset)]  |  [[Community]()]

![teaser](docs/media/cover.png)

The Imitator Game is a hierarchical benchmark for robot imitation from human manipulation videos. A human first demonstrates the manipulation skills in front of a camera (the Demonstrator scene); robots then attempt to imitate those skills from video observation alone in the Imitator scene (the Imitator scenes). Evaluation is conducted with blind A/B comparisons in the Imitator Arena or simulation automated metrics to judge which model imitates the human better.

This repo contains the simulation framework (powered by [ManiSkill](https://www.maniskill.ai/) and [SAPIEN](https://sapien.ucsd.edu/)), the IG-10K data collection and preprocessing tooling, and the policy baselines ([ACT](https://arxiv.org/abs/2304.13705), [Diffusion Policy](https://arxiv.org/abs/2303.04137v5), [VQ-BeT](https://arxiv.org/abs/2403.03181), [UniSkill](https://arxiv.org/abs/2505.08787), [XSkill](https://arxiv.org/abs/2307.09955), [GR00T](https://arxiv.org/abs/2503.14734), [RDT](https://arxiv.org/abs/2410.07864), [π0.5](https://arxiv.org/abs/2504.16054), [OpenVLA](https://arxiv.org/abs/2406.09246)) including their training and evaluation recipes.

---

## Community & contributions

The Imitator Game is also an **open-source community** for robot imitation from human
video. Anyone interested in robot imitative ability
(goal-directed imitation, video-conditioned policies, cross-embodiment skill
transfer, intent inference, affordance adaptation, ...) is welcome to
participate and contribute to the community. Contributions will be acknowledged and merged into the project after review.

### **Get involved**

Everyone can contribute to this project by submitting **tasks / models / embodiments** that support the Imitator-Game **data-collection / training / evaluation** framework. Please follow the design templates and contribute via pull request: 

- Contribute new Imitator tasks (L0–L3) designed with the task template under [mani_skill/envs/tasks/_template/](mani_skill/envs/tasks/_template/) (together with the motion-planning solutions);
- contribute new imitation-learning models built on the baseline template under [examples/baselines/_template/](examples/baselines/_template/) (together with evaluation interface);
- contribute new robots with robot template under [mani_skill/agents/robots/_template/](mani_skill/agents/robots/_template/).

### Pairing data service

We operate a **data pairing service**: for
community-contributed tasks, we may collect *human and robot pairing demonstrations*, and merge the paired data into IG-10K-format datasets
under the shared LeRobot format. Community members can request pairing for
their accepted tasks.

### Real-world deployment and evaluation

Contributors can submit their policy for deployment on our real-robot
platform. We will run the submitted policy under the agreed evaluation protocols and record the rollout videos. Approved real-world rollouts can be added to the
[Imitator Arena](https://imitator-game.github.io/arena.html), where the community
can make blind A/B comparisons between anonymous model rollouts against the
same human reference. Supported human, real-robot, and simulation tasks can be found in the [Task Gallery](https://imitator-game.github.io/gallery.html).

For a real-world evaluation request, please contact us in the community by [WeChat](https://imitator-game.github.io/docs/index.html) or [Discord](https://discord.gg/p8QVXWwWyZ). You may provide:

- the model checkpoint or source, inference API, environment dependencies and
  license;
- the task protocol, reset conditions, and task-specific evaluation standard; and
- the commands or configuration needed to reproduce the result.

A hardware run is scheduled only after the package is reproducible;
the repo does not provide an policy upload interface at this point, but will support soon.

### Update

We will keep the simulation framework actively updated with the upstream [ManiSkill](https://www.maniskill.ai/) repo to support the latest simulation features. To keep the ecosystem coherent, **all contributions must follow the shared ManiSkill / Imitator-Game design templates** — the robot agent template, the environment interface, and the motion-planning interface, etc. If you build on a new ManiSkill feature, please
keep the common interfaces intact. We will also actively update both the simulation and real-world framework with more official-support tasks, assets, embodiments, and data.

---

## Installation

Installation only requires a few [uv](https://docs.astral.sh/uv/getting-started/installation/) installs plus a Vulkan setup for rendering:

```bash
uv sync --active
source .venv/bin/activate

# patch a known lerobot bug
curl -sSL https://raw.githubusercontent.com/huggingface/lerobot/0e81a275fcdbf03d74f78aa69eaa28c172a9f256/src/lerobot/datasets/lerobot_dataset.py -o .venv/lib/python3.11/site-packages/lerobot/datasets/lerobot_dataset.py
```

Set up Vulkan for rendering following the [ManiSkill installation instructions](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan).

---

## Downloading assets

We provide the [assets set](https://huggingface.co/datasets/imitator-game/IG-10K-Assets) that we used in The Imitator Game ([ycb](https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip) / [RoboTwin](https://robotwin-platform.github.io/doc/objects/index.html) / [PartNet-Mobility](https://sapien.ucsd.edu/browse) / [sketchfeb](https://sketchfab.com/)), but we encourage the community to add more diverse assets. The assets should be placed in `~/.maniskill/data`.

```bash
# Hugging Face
mkdir -p ~/.maniskill/data
hf download imitator-game/IG-10K-Assets --repo-type dataset --local-dir ~/.maniskill/data
# ModelScope
# modelscope download --dataset Zhouxunzhe/IG-10K-Assets --local_dir ~/.maniskill/data

# Extract
for f in ~/.maniskill/data/*.tar.zst; do tar --use-compress-program=unzstd -xf "$f" -C ~/.maniskill/data; done
rm ~/.maniskill/data/*.tar.zst
```

External GLB assets (Sketchfab, Hunyuan-generated meshes, [Objaverse](https://objaverse.allenai.org/), etc.) use the pipeline documented in [`mani_skill/assets/sketchfab_README.md`](mani_skill/assets/sketchfab_README.md): download the shared assets to `~/.maniskill/data/sketchfab/objects/{object_key}/model.glb`, register them in `mani_skill/assets/sketchfab_registry.json`, and load them in tasks with `create_sketchfab_actor(...)`.

---

## IG-10K Dataset

The dataset is in `LeRobot-0.5.0` format. See [`examples/baselines/lerobot_dataset/README.md`](examples\baselines\lerobot_dataset\README.md) for the dataset layout, observation features (RGB/depth cameras, MANO hand pose, segmentation masks), and the imitative level handling of the paired human/robot data.

```bash
# Hugging Face
hf download imitator-game/IG-10K-Dataset --repo-type dataset --local-dir demos
# ModelScope
# modelscope download --dataset Zhouxunzhe/IG-10K-Dataset --local_dir demos
```

Data is recommended to be placed as the followed paths:

```text
The-Imitator-Game/
└── demos/
    ├── demo_data/          # Human demonstration dataset(s)
    └── imitator_data/      # Robot (simulation) demonstrations in LeRobot format
```

### Collect new data in simulation

See [`scripts/collect_data.py`](scripts/collect_data.py) for collecting simulation motion-planning demonstrations in ManiSkill hdf5-format, and [examples/baselines/lerobot_dataset/h5_to_lerobot.py](examples/baselines/lerobot_dataset/h5_to_lerobot.py ) for converting `.h5` trajectories to `LeRobot-0.5.0` format. An example script for data collection: 

```bash
python -m mani_skill.examples.motionplanning.dual.two_robot_run --shader "rt-fast" -n 50 -e "Task name" --save-video
```

---

## Benchmark

Each baseline has its own README.md with training and evaluation commands under [examples/baselines/](examples/baselines/). An example script for model training and evaluation: 

```bash
# Training
python -m examples.baselines.model.train_model_imitator \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-dataset-file path/to/human_train_config.json \
  --sim-dataset-file path/to/sim_train_config.json \
  --task-mapping-file examples/baselines/lerobot_dataset/task_mapping.json

# Evaluation
python -m examples.baselines.model.eval_model_imitator \
  --eval-config path/to/eval_config.txt \
  --checkpoint path/to/final_model.pt \
  --human-root demos/demo_data \
  --sim-root demos/imitator_data \
  --human-config path/to/human_eval_config.json \
  --sim-config path/to/sim_eval_config.json \
  --task-mapping examples/baselines/lerobot_dataset/task_mapping.json
```

---

## Technical Support

 See the [docs](https://imitator-game.github.io/docs/index.html) for more details. Contact [Xunzhe Zhou](https://zhouxunzhe.github.io/) or join the [Community]() if you need any support.

---

## Citation

```bibtex
@misc{
  the_imitator_game,
  title={The Imitator Game: Evaluating Robot Imitation from Human Demonstration Videos},
  url={https://imitator-game.github.io/},
  year={2026}
}
```

---

## License

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
