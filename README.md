# SEAR: Simple and Efficient Adaptation of Visual Geometric Transformers for RGB+Thermal 3D Reconstruction

This project aims to estimate camera poses of RGB and Thermal images together.

* [Arxiv paper](https://arxiv.org/abs/2603.18774)
* [Checkpoints on HuggingFace](https://huggingface.co/MalcolmMielle/SEAR)

![](images/sink.gif)
![](images/laptop.gif)
![](images/reflect-robot.gif)
![](images/drone-bathroom.gif)

## Install

Install with uv:

```bash
uv sync --all-extras
```

After cloning VGGT, change their pyproject.toml so it looks like:

```toml
# setuptools configuration
[tool.setuptools.packages.find]
where = ["."]
include = ["vggt*", "training*"]
```

Or you can add `sys.path.append(".vvgt/training")` to your scripts.

You can stop doing this when [issue 416](https://github.com/facebookresearch/vggt/issues/416) of VGGT is solved.

## Train the model

To train our model run this script:

```bash
python sear/scripts/train_sear.py --aggregator.type CAMERA_TOKEN --dataset-mode NO_TRAJECTORY_SPECIFIED
```

Ablation studies can run by using the other aggregator-types found in `sear/ablation_models/possible_aggregators.py`.

Models can be evaluated after training with `sear/scripts/eval/ablation_vggt.py`.

To run the evaluation see the tutorials for [camera pose and point cloud](docs/Evaluation/Main.md), [relative camera pose from two views](docs/evaluation/Pairs.md) and [dependence on thermal ratio](docs/Evaluation/ThermalRatio.md)

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
