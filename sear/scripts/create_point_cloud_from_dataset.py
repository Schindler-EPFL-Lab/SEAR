"""
Script to create point clouds from a ThermoScenes dataset scene.

This script generates RGB and thermal point clouds along with camera pyramids
from a ThermoScenes dataset scene. The point clouds are saved in PLY format
and camera extrinsics are visualized as pyramids.

Usage:
    python create_point_cloud_from_dataset.py --scene-path <path_to_scene> \
        [--output-folder <output_directory>] \
        [--depth-eps <depth_threshold>] \
        [--max-num-points <max_points>]

Parameters:
    --scene-path: Path to the ThermoScenes scene directory (required)
    --output-folder: Directory where point clouds will be saved (default: "outputs")
    --depth-eps: Depth threshold for invalid pixels (default: 1e-8)
    --max-num-points: Maximum number of points to generate (default: 100000)

Output files:
    - rgb.ply: RGB point cloud
    - thermal.ply: Thermal point cloud  
    - rgb_cameras.ply: RGB camera extrinsics visualization
    - thermal_cameras.ply: Thermal camera extrinsics visualization

The script processes all images in the dataset, converts depth maps to 3D coordinates,
and generates colored point clouds for both RGB and thermal modalities.
"""

from dataclasses import dataclass
from pathlib import Path

import tyro

from sear.visualization.point_cloud_from_dataset import (
    PointCloudFromDatasetCreator,
)


@dataclass
class PointCloudFromDatasetWithOutputCreator(PointCloudFromDatasetCreator):
    output_folder: Path = Path("outputs")
    """The point clouds are stored in `output_folder`."""


if __name__ == "__main__":
    point_cloud_creator = tyro.cli(PointCloudFromDatasetWithOutputCreator)
    point_cloud_creator.create_point_clouds(
        output_folder=point_cloud_creator.output_folder
    )
