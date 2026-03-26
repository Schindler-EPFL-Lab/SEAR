import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from nerfstudio.process_data import colmap_utils
from vggt.utils.geometry import closed_form_inverse_se3

from sear.data_processing.convertion import opengl_to_opencv


@dataclass
class ColmapToThermoScenesParameters:
    """
    A config to convert ThermalGaussian's raw COLMAP files to ThermoScenesVGGT format.
    """

    colmap_reconstruction: Path
    """Path to colmap results, e.g. /path/to/scene/colmap/sparse/0"""
    output_dir: Path
    """Where to store the results"""


def tf_nerfstudio_to_ours(transforms: dict[str, Any]) -> dict[str, Any]:
    """
    Converts `transforms` in nerfstudio format
    https://docs.nerf.studio/quickstart/data_conventions.html

    to ThermoScenesVGGT defined in `src/rebel-pose/README.md`.

    :returns transforms in ThermoScenesVGGT format.
    """

    result_transforms: dict[str, Any] = {}
    result_transforms["frames"] = []

    for i in range(len(transforms["frames"])):
        pose_cam2world_opengl = np.array(transforms["frames"][i]["transform_matrix"])
        pose_cam2world_opencv = (
            opengl_to_opencv(torch.from_numpy(pose_cam2world_opengl).float())
            .numpy()
            .astype(np.float64)
        )
        pose_world2cam = closed_form_inverse_se3(pose_cam2world_opencv[None])[0]

        frame = {}
        frame["fl_x"] = transforms["fl_x"]
        frame["fl_y"] = transforms["fl_y"]
        frame["cx"] = transforms["cx"]
        frame["cy"] = transforms["cy"]
        frame["w"] = transforms["w"]
        frame["h"] = transforms["h"]
        frame["k1"] = 0
        frame["k2"] = 0
        frame["k3"] = 0
        frame["k4"] = 0
        frame["p1"] = 0
        frame["p2"] = 0
        frame["camera_angle_x"] = np.arctan(frame["w"] / (frame["fl_x"] * 2)) * 2
        frame["camera_angle_y"] = np.arctan(frame["w"] / (frame["fl_x"] * 2)) * 2
        frame["fovx"] = frame["camera_angle_x"] * 180 / np.pi
        frame["fovy"] = frame["camera_angle_x"] * 180 / np.pi
        frame["file_path"] = transforms["frames"][i]["file_path"]
        frame["transform_matrix"] = pose_world2cam.tolist()

        frame_rgb = deepcopy(frame)
        frame_rgb["type"] = "rgb"

        result_transforms["frames"].append(
            {
                "rgb": frame_rgb,
            }
        )

    return result_transforms


def colmap_to_thermoscenes(params: ColmapToThermoScenesParameters):
    """
    Converts colmap reconstruction from `params.colmap_reconstruction` into
    ThermoScenesVGGT format and stores the results at `params.output_dir`
    """
    params.output_dir.mkdir(exist_ok=True, parents=True)

    colmap_utils.colmap_to_json(
        recon_dir=params.colmap_reconstruction,
        output_dir=params.output_dir,
        image_id_to_depth_path=None,
        camera_mask_path=None,
        image_rename_map=None,
    )
    with open(params.output_dir / "transforms.json") as f:
        transforms = json.load(f)

    result_transforms = tf_nerfstudio_to_ours(transforms)
    with open(params.output_dir / "transforms.json", "w") as f:
        json.dump(result_transforms, f, indent=4)


if __name__ == "__main__":
    params = tyro.cli(ColmapToThermoScenesParameters)
    colmap_to_thermoscenes(params)
