import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from sear import logger
from sear.data_processing.frame_info import FrameInfo
from sear.metrics.calculator import MetricsCalculator


@dataclass(kw_only=True)
class EvalParameters(ReverseCli):
    """A config to evaluate outputs of a method using the stored results"""

    method_predictions_folder: Path = Path("folder")
    """Directory containing outputs"""
    depth_eps: float = 1e-8
    """Depth value smaller this value do not take part in training"""
    custom_aggregation_file_path: Path | None = Path(
        "./sear/configs/public_and_self_collected.json"
    )
    """
    Path to the file with custom aggregations, e.g. split to PublicDatasets and
    DifficultScenes.
    """
    output_folder_root: Path = Path("./outputs")
    """Directory to save metrics files"""
    method_name: str | None = None
    """
    The method name used to evaluate. If not provided then
    `method_predictions_folder.parent` is used
    """
    thresholds: list[float] | None = None

    num_bootstrap: int = 0

    def __post_init__(self) -> None:
        """
        Does necessary post processing for the `EvalParameters` class
        """
        if self.method_name is None:
            self.method_name = self.method_predictions_folder.name
        if self.thresholds is None:
            self.thresholds = [5.0, 15.0, 30.0]


def load_one_scene_outputs(
    scene_path: Path,
    transforms_name="transforms_ground_truth.json",
) -> tuple[
    str,
    str,
    list[str],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
    float,
]:
    """
    Loads data from located in `scene_path` using transforms with name
    `transforms_name`. In particular, it loads depths, extrinsics, intrinsics, dataset
    name, scene name, duration and reconstructed ratio.

    :raise: RuntimeError if there are more than one modality in on frame of
        `transforms_name`
    :raise: RuntimeError depth is not of shape (H, W)

    :return: dataset name, scene name, names of read depths, depths of shape (N, H, W),
        extrinsics in world-to-cam OpenCV format, intrinsics, duration and
        ratio_reconstructed.
    """

    # Firstly try to load dataset_name and scene_name from the transforms.json
    with open(scene_path / transforms_name) as f:
        transforms = json.load(f)
    dataset_name = transforms.get("dataset_name", None)
    scene_name = transforms.get("scene_name", None)

    # Otherwise infer the values from the folder name
    if dataset_name is None or scene_name is None:
        dataset_name, scene_name, _ = scene_path.name.split(":")
        transforms["dataset_name"] = dataset_name
        transforms["scene_name"] = scene_name

        with open(scene_path / transforms_name, "w") as f:
            json.dump(transforms, f, indent=4)

    pred_depths_list: list[npt.NDArray[np.float64]] = []
    extrinsics_world2cam_list: list[npt.NDArray[np.float64]] = []
    intrinsics_list: list[npt.NDArray[np.float64]] = []
    depths_files: list[str] = []

    for frame in transforms["frames"]:
        modality_key = list(frame.keys())
        if len(modality_key) != 1 or modality_key[0] not in ["rgb", "thermal"]:
            raise RuntimeError(
                f"The modality should be either rgb or thermal but got {modality_key}"
            )
        modality_key = modality_key[0]
        frame_dict = frame[modality_key]

        depth_file = scene_path / frame_dict["depth_file_path"]
        depth = np.load(depth_file)
        if depth.ndim != 2:
            raise RuntimeError(f"The depth must be of shape (H, W) but got {depth}.")
        extrinsic_world2cam, intrinsic = FrameInfo.dict_to_matrices(frame_dict)

        depths_files.append(depth_file.name)
        pred_depths_list.append(depth)
        extrinsics_world2cam_list.append(extrinsic_world2cam.numpy())
        intrinsics_list.append(intrinsic.numpy())

    depths = np.stack(pred_depths_list, axis=0)
    extrinsics_world2cam = np.stack(extrinsics_world2cam_list, axis=0)
    intrinsics = np.stack(intrinsics_list, axis=0)

    return (
        dataset_name,
        scene_name,
        depths_files,
        depths,
        extrinsics_world2cam,
        intrinsics,
        transforms.get("duration", np.nan),
        transforms.get("ratio_reconstructed", np.nan),
    )


def main(params: EvalParameters) -> None:
    """
    Evaluates predictions from `params.method_predictions_folder` and saves the results
    to `params.output_folder_root`.
    """

    output_folder = params.output_folder_root / params.method_name
    output_folder.mkdir(exist_ok=True, parents=True)
    calculator = MetricsCalculator(
        thresholds=params.thresholds, num_bootstrap=params.num_bootstrap
    )

    scenes_paths = sorted(list(params.method_predictions_folder.iterdir()))
    for scene_index, scene_path in enumerate(scenes_paths):
        if (not scene_path.is_dir()) or scene_path.name == "cache":
            continue

        (
            dataset_name,
            scene_name,
            real_depths_files,
            real_depths,
            real_extrinsics_world2cam,
            real_intrinsics,
            duration,
            ratio_reconstructed,
        ) = load_one_scene_outputs(
            scene_path=scene_path,
            transforms_name="transforms_ground_truth.json",
        )

        (
            dataset_name,
            scene_name,
            pred_depths_files,
            pred_depths,
            pred_extrinsics_world2cam,
            pred_intrinsics,
            duration,
            ratio_reconstructed,
        ) = load_one_scene_outputs(
            scene_path=scene_path,
            transforms_name="transforms.json",
        )

        if real_depths_files != pred_depths_files:
            raise RuntimeError(
                "The `pred_depths_files` and `real_depths_files` must be the same, but "
                + f"got \n{real_depths_files} \nand \n{pred_depths_files}."
            )

        calculator.add_data(
            cameras_real_world2cam=real_extrinsics_world2cam,
            depths_real=real_depths,
            intrinsics_real=real_intrinsics,
            cameras_pred_world2cam=pred_extrinsics_world2cam,
            depths_pred=pred_depths,
            intrinsics_pred=pred_intrinsics,
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
    if params.custom_aggregation_file_path is not None:
        with open(params.custom_aggregation_file_path) as f:
            aggregation_per_scene = json.load(f)
        calculator.custom_aggregation(
            aggregation_per_scene=aggregation_per_scene,
            save_path=output_folder / "custom_aggregation.json",
        )
    calculator.aggregated(output_folder / "aggregated.json")


if __name__ == "__main__":
    params = tyro.cli(EvalParameters)
    main(params=params)
