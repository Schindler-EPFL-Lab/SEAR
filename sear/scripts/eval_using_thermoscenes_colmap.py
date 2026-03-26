import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import tyro

from sear import logger
from sear.data_processing.frame_info import FrameInfo
from sear.metrics.calculator import MetricsCalculator
from sear.scripts.eval_using_predictions import load_one_scene_outputs
from sear.scripts.eval_using_thermalgaussian_colmap import (
    EvalParameters,
    shuffle_transforms_like,
)
from sear.scripts.thermalgaussian.colmap_to_thermoscenes import (
    tf_nerfstudio_to_ours,
)


def load_colmap_poses(
    scene_path: Path,
    transforms_reference: dict[str, Any],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Loads COLMAP poses located in `scene_path`/transforms.json. It uses the
    `transforms_reference` to shuffle the frames.

    :return: extrinsics in world-to-cam OpenCV format and intrinsics
    """

    with open(scene_path / "transforms.json") as f:
        transforms_colmap = json.load(f)

    transforms_colmap = tf_nerfstudio_to_ours(transforms_colmap)
    transforms_colmap = shuffle_transforms_like(transforms_colmap, transforms_reference)
    # remove frames if gt has less poses
    if len(transforms_reference["frames"]) > len(transforms_colmap["frames"]):
        transforms_reference = shuffle_transforms_like(
            transforms_reference, transforms_colmap
        )

    extrinsics_world2cam_list: list[npt.NDArray[np.float64]] = []
    intrinsics_list: list[npt.NDArray[np.float64]] = []
    extrinsics_world2cam_ref_list: list[npt.NDArray[np.float64]] = []
    intrinsics_ref_list: list[npt.NDArray[np.float64]] = []

    for i in range(len(transforms_colmap["frames"])):
        frame_dict = transforms_colmap["frames"][i]["rgb"]
        extrinsic_world2cam, intrinsic = FrameInfo.dict_to_matrices(frame_dict)
        extrinsics_world2cam_list.append(extrinsic_world2cam.numpy())
        intrinsics_list.append(intrinsic.numpy())

        frame_dict_ref = list(transforms_reference["frames"][i].values())[0]
        extrinsic_world2cam_ref, intrinsic_ref = FrameInfo.dict_to_matrices(
            frame_dict_ref
        )
        extrinsics_world2cam_ref_list.append(extrinsic_world2cam_ref.numpy())
        intrinsics_ref_list.append(intrinsic_ref.numpy())

    extrinsics_world2cam = np.stack(extrinsics_world2cam_list, axis=0)
    intrinsics = np.stack(intrinsics_list, axis=0)

    extrinsics_world2cam_ref = np.stack(extrinsics_world2cam_ref_list, axis=0)
    intrinsics_ref = np.stack(intrinsics_ref_list, axis=0)

    return extrinsics_world2cam, intrinsics, extrinsics_world2cam_ref, intrinsics_ref


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
                f"Skipping {scene_path.name}, approx.",
                "{scene_index} / {len(scenes_paths)}.",
            )
            continue

        with open(scene_path / "transforms.json") as f:
            transforms_pred = json.load(f)

        (
            colmap_extrinsics_world2cam,
            colmap_intrinsics,
            pred_extrinsics_world2cam_part,
            pred_intrinsics_part,
        ) = load_colmap_poses(
            scene_path=params.original_dataset_root / scene_name,
            transforms_reference=transforms_pred,
        )

        with open(params.original_dataset_root / scene_name / "transforms.json") as f:
            transforms_colmap = json.load(f)

        ratio_colmap_rgb_reconstructed = pred_extrinsics_world2cam_part.shape[0] / len(
            transforms_colmap["frames"]
        )

        calculator.add_data(
            cameras_real_world2cam=colmap_extrinsics_world2cam,
            depths_real=np.empty((0,)),
            intrinsics_real=np.empty((0,)),
            cameras_pred_world2cam=pred_extrinsics_world2cam_part,
            depths_pred=np.empty((0,)),
            intrinsics_pred=np.empty((0,)),
            ratio_reconstructed=ratio_colmap_rgb_reconstructed,
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
