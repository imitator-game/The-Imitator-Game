# HaMeR: Hand Mesh Recovery

This directory vendors the HaMeR hand-mesh recovery code (paper: *Reconstructing Hands in 3D with Transformers*, [arXiv:2312.05251](https://arxiv.org/pdf/2312.05251.pdf)) adapted for estimating MANO hand features from the human demonstration videos. It is the supporting hand-tracking tool used by the hand-estimation pipeline (see `examples/baselines/hand_estimation`).

## Installation (uv, recommended)

Make sure the CUDA version in `hamer/pyproject.toml` line 58 matches your local CUDA version (the default `cu124` targets CUDA 12.4; change it to `cu118` for CUDA 11.8).

```bash
cd examples/baselines/hand_estimation/hamer
git submodule update --init --recursive
uv venv
uv sync
uv pip install "chumpy @ git+https://github.com/mattloper/chumpy" "detectron2 @ git+https://github.com/facebookresearch/detectron2" --no-build-isolation
uv pip install -e ./third-party/ViTPose/
```

Then download the trained models:

```bash
bash fetch_demo_data.sh
```

Download the MANO model: register at the [MANO website](https://mano.is.tue.mpg.de) and place the right-hand model `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.

## Running hand estimation

```bash
cd examples/baselines/hand_estimation/hamer
PYGLET_HEADLESS=1 python process_hand.py --video_dir ./example_hdf5/ --output_dir ./video --visualize
```

## Acknowledgements

Parts of the code are taken or adapted from [4DHumans](https://github.com/shubham-goel/4D-Humans), [SLAHMR](https://github.com/vye16/slahmr), [ProHMR](https://github.com/nkolot/ProHMR), [SPIN](https://github.com/nkolot/SPIN), [SMPLify-X](https://github.com/vchoutas/smplify-x), [HMR](https://github.com/akanazawa/hmr), [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), and [Detectron2](https://github.com/facebookresearch/detectron2).

## Citation

```bibtex
@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}
```