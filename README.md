# SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction

This project aims to estimate camera poses of RGB and Thermal images together.

[![Hugging Face](https://img.shields.io/badge/Hugging_Face-SEAR-orange?logo=huggingface)](https://huggingface.co/MalcolmMielle/SEAR) | [![arXiv](https://img.shields.io/badge/arXiv-2603.18774-red?logo=arxiv)](https://arxiv.org/abs/2603.18774)

![](images/sink.gif)
![](images/laptop.gif)
![](images/reflect-robot.gif)
![](images/drone-bathroom.gif)

## Install

Clone this repo and [VGGT](https://github.com/facebookresearch/vggt.git)

```bash
git clone https://github.com/Schindler-EPFL-Lab/SEAR.git
cd SEAR
git clone https://github.com/facebookresearch/vggt.git
```

Install with uv:

```bash
uv sync --all-extras
```

## Train the model

Install VGGT checkpoint [`VGGT-1B`](https://huggingface.co/facebook/VGGT-1B).

To train our model run this script:

```bash
python sear/scripts/train_sear.py --thermal-vggt.vggt-path /path/to/vggt/weights.pth
```

Ablation studies can run by using the other aggregator-types found in `sear/ablation_models/possible_aggregators.py`.

Models can be evaluated after training with `sear/scripts/eval/ablation_vggt.py`.

To run the evaluation see the tutorials for [camera pose and point cloud](docs/evaluation/Main.md), [relative camera pose from two views](docs/evaluation/Pairs.md) and [dependence on thermal ratio](docs/evaluation/ThermalRatio.md).

## Training Data

Our training dataset is a combination of the following dataset:

* [ThermoNeRF](https://github.com/Schindler-EPFL-Lab/thermo-nerf),
* [ThermalNeRF](https://github.com/yvette256/nerfstudio-thermal),
* [ThermalMix](https://mert-o.github.io/ThermalNeRF/),
* [ThermalGaussian](https://thermalgaussian.github.io/)
* [Radar Forest](https://github.com/RNP-lab/viking_hill_radar_lidar_camera_dataset)

We provide a [compilation of all training dataset as well as ours](https://doi.org/10.5281/zenodo.19057885).

See details of the data processing in [Dataset documentation](docs/Dataset.md).

## Cite us

```bib
@misc{skorokhodov2026searsimpleefficientadaptation,
      title={SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction},
      author={Vsevolod Skorokhodov and Chenghao Xu and Shuo Sun and Olga Fink and Malcolm Mielle},
      year={2026},
      eprint={2603.18774},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.18774},
}
```
