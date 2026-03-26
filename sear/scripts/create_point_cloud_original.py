from dataclasses import dataclass
from pathlib import Path

import tyro

from sear.visualization.point_cloud_original import PointCloudOriginalCreator


@dataclass
class PointCloudOriginalWithOutputCreator(PointCloudOriginalCreator):
    output_folder: Path = Path("outputs")
    """The point clouds are stored in `output_folder`."""


if __name__ == "__main__":
    point_cloud_creator = tyro.cli(PointCloudOriginalWithOutputCreator)
    point_cloud_creator.create_point_clouds(
        output_folder=point_cloud_creator.output_folder
    )
