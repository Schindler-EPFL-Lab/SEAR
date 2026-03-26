from dataclasses import dataclass
from pathlib import Path

import tyro

from sear.visualization.point_cloud import PointCloudCreator


@dataclass
class PointCloudWithOutputCreator(PointCloudCreator):
    output_folder: Path = Path("outputs")
    """The point clouds are stored in `output_folder`."""


if __name__ == "__main__":
    point_cloud_creator = tyro.cli(PointCloudWithOutputCreator)
    point_cloud_creator.create_point_clouds(
        output_folder=point_cloud_creator.output_folder
    )
