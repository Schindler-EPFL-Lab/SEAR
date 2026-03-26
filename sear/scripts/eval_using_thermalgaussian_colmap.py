import json
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from sear import logger
from sear.data_processing.frame_info import FrameInfo
from sear.metrics.calculator import MetricsCalculator
from sear.scripts.eval_using_predictions import load_one_scene_outputs
from sear.scripts.thermalgaussian.colmap_to_thermoscenes import (
    ColmapToThermoScenesParameters,
    colmap_to_thermoscenes,
)


@dataclass(kw_only=True)
class EvalParameters(ReverseCli):
    """A config to evaluate outputs of a method using the stored results"""

    method_predictions_folder: Path
    """Directory containing outputs"""
    original_dataset_root: Path
    """Directory the scenes processed with COLMAP"""
    dataset_name: str | None = None
    """
    Dataset used for validation. If not provided then `original_dataset_root.name`
    is used.
    """

    depth_eps: float = 1e-8
    """Depth value smaller this value do not take part in training"""

    output_folder_root: Path = Path("./outputs")
    """Directory to save metrics files"""
    method_name: str | None = None
    """
    The method name used to evaluate. If not provided then
    `method_predictions_folder.parent` is used
    """
    thresholds: list[float] | None = None

    def __post_init__(self) -> None:
        """
        Does necessary post processing for the `EvalParameters` class
        """
        if self.method_name is None:
            self.method_name = self.method_predictions_folder.name
        if self.thresholds is None:
            self.thresholds = [5.0, 15.0, 30.0]
        if self.dataset_name is None:
            self.dataset_name = self.original_dataset_root.name


def shuffle_transforms_like(
    transforms: dict[str, Any],
    transforms_ref: dict[str, Any],
) -> dict[str, Any]:
    """
    Shuffles `transforms["frames"]` to have the same order as in
    `transforms_ref["frames"]`.
    """
    transforms = deepcopy(transforms)
    frames_transfoms_ref = {
        Path(list(frame.values())[0]["file_path"]).stem: i
        for i, frame in enumerate(transforms_ref["frames"])
    }

    # remove files that the method did not register
    transforms["frames"] = [
        frame
        for frame in transforms["frames"]
        if Path(list(frame.values())[0]["file_path"]).stem in frames_transfoms_ref
    ]

    transforms["frames"] = sorted(
        transforms["frames"],
        key=lambda frame: frames_transfoms_ref[
            Path(list(frame.values())[0]["file_path"]).stem
        ],
    )
    return transforms


def load_colmap_poses(
    scene_path: Path,
    transforms_reference: dict[str, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Loads COLMAP data located in `scene_path` using `transforms_reference` to
    shuffle and rename.

    :return: extrinsics in world-to-cam OpenCV format and intrinsics
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        colmap_to_thermoscenes_parameters = ColmapToThermoScenesParameters(
            colmap_reconstruction=scene_path / "colmap/sparse/0",
            output_dir=temp_dir_path,
        )
        colmap_to_thermoscenes(colmap_to_thermoscenes_parameters)

        with open(temp_dir_path / "transforms.json") as f:
            transforms_colmap = json.load(f)

    # check if need to remap
    reference_rgb_files_names = {
        Path(list(frame.values())[0]["file_path"]).stem
        for frame in transforms_reference["frames"]
    }
    colmap_rgb_files_names = {
        Path(frame["rgb"]["file_path"]).stem for frame in transforms_colmap["frames"]
    }

    if not reference_rgb_files_names.issubset(colmap_rgb_files_names):
        remapping: dict[str, str] = {}
        images_eval = sorted(list((scene_path / "rgb" / "test").iterdir()))
        for i, image_eval_path in enumerate(images_eval):
            remapping[image_eval_path.stem] = (
                f"images/frame_eval_{i + 1:05}{image_eval_path.suffix}"
            )

        images_train = sorted(list((scene_path / "rgb" / "train").iterdir()))
        for i, image_train_path in enumerate(images_train):
            remapping[image_train_path.stem] = (
                f"images/frame_train_{i + 1:05}{image_train_path.suffix}"
            )
        for i in range(len(transforms_colmap["frames"])):
            frame_rgb_name = Path(
                transforms_colmap["frames"][i]["rgb"]["file_path"]
            ).stem
            transforms_colmap["frames"][i]["rgb"]["file_path"] = remapping[
                frame_rgb_name
            ]

    transforms_colmap = shuffle_transforms_like(transforms_colmap, transforms_reference)

    extrinsics_world2cam_list: list[npt.NDArray[np.float64]] = []
    intrinsics_list: list[npt.NDArray[np.float64]] = []

    for frame in transforms_colmap["frames"]:
        frame_dict = frame["rgb"]
        extrinsic_world2cam, intrinsic = FrameInfo.dict_to_matrices(frame_dict)
        extrinsics_world2cam_list.append(extrinsic_world2cam.numpy())
        intrinsics_list.append(intrinsic.numpy())

    extrinsics_world2cam = np.stack(extrinsics_world2cam_list, axis=0)
    intrinsics = np.stack(intrinsics_list, axis=0)

    return extrinsics_world2cam, intrinsics


def main(params: EvalParameters) -> None:
    """
    Evaluates predictions from `params.method_predictions_folder` and saves the results
    to `params.output_folder_root`.
    """

    output_folder = params.output_folder_root / params.method_name
    output_folder.mkdir(exist_ok=True, parents=True)
    calculator = MetricsCalculator(
        thresholds=params.thresholds,
        calculate_point_cloud_metrics_datasets=[],
    )

    scenes_paths = sorted(list(params.method_predictions_folder.iterdir()))
    for scene_index, scene_path in enumerate(scenes_paths):
        if (not scene_path.is_dir()) or scene_path.name == "cache":
            continue

        (
            dataset_name,
            scene_name,
            _,
            _,
            pred_extrinsics_world2cam,
            pred_intrinsics,
            duration,
            ratio_reconstructed,
        ) = load_one_scene_outputs(
            scene_path=scene_path,
            transforms_name="transforms.json",
        )

        if dataset_name != params.dataset_name:
            logger.info(
                f"Skipping {scene_path.name},",
                "approx. {scene_index} / {len(scenes_paths)}.",
            )
            continue

        with open(scene_path / "transforms.json") as f:
            transforms_pred = json.load(f)

        (
            colmap_extrinsics_world2cam,
            colmap_intrinsics,
        ) = load_colmap_poses(
            scene_path=params.original_dataset_root / scene_name,
            transforms_reference=transforms_pred,
        )

        calculator.add_data(
            cameras_real_world2cam=colmap_extrinsics_world2cam,
            depths_real=np.empty((0,)),
            intrinsics_real=np.empty((0,)),
            cameras_pred_world2cam=pred_extrinsics_world2cam,
            depths_pred=np.empty((0,)),
            intrinsics_pred=np.empty((0,)),
            ratio_reconstructed=ratio_reconstructed,
            duration=duration,
            scene_name=scene_name,
            dataset_name=dataset_name,
        )
        logger.info(
            f"Evaluated {scene_path.name}, approx. {scene_index} / {len(scenes_paths)}."
        )

    # Save the metrics results
    calculator.per_dataset(output_folder / "per_dataset.json")
    calculator.per_scene(output_folder / "per_scene.json")
    calculator.aggregated(output_folder / "aggregated.json")


if __name__ == "__main__":
    params = tyro.cli(EvalParameters)
    main(params=params)
