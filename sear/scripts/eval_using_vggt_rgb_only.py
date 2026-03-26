import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from sear import logger
from sear.data_processing.single_dataset import VGGTSingleDataset
from sear.metrics.calculator import MetricsCalculator
from sear.scripts.eval_using_thermalgaussian_colmap import (
    load_colmap_poses as load_colmap_poses_thermalgaussian,
)
from sear.scripts.eval_using_thermoscenes_colmap import (
    load_colmap_poses as load_colmap_poses_thermonerf,
)


@dataclass(kw_only=True)
class EvalParameters(ReverseCli):
    """A config to evaluate outputs of a method using the stored results"""

    thermoscenes_path: Path
    """Directory containing outputs"""
    original_dataset_root: Path
    """Directory the scenes processed with COLMAP"""
    dataset_name: str
    """
    Dataset used for validation. If not provided then `original_dataset_root.name`
    is used.
    """

    train_test_split_path: Path = Path("./sear/configs/train_test_split.json")

    scenes_per_dataset_path: Path = Path("./sear/configs/scenes_per_dataset.json")

    output_folder_root: Path = Path("./outputs")
    """Directory to save metrics files"""
    thresholds: list[float] | None = None

    def __post_init__(self) -> None:
        """
        Does necessary post processing for the `EvalParameters` class
        """
        if self.thresholds is None:
            self.thresholds = [5.0, 15.0, 30.0]
        if self.dataset_name is None:
            self.dataset_name = self.original_dataset_root.name


def main(params: EvalParameters) -> None:
    """
    Evaluates predictions from `params.method_predictions_folder` and saves the results
    to `params.output_folder_root`.
    """

    method_name = "GT"

    output_folder = params.output_folder_root / method_name
    output_folder.mkdir(exist_ok=True, parents=True)
    calculator = MetricsCalculator(
        thresholds=params.thresholds,
        calculate_point_cloud_metrics_datasets=[],
    )

    with open(params.scenes_per_dataset_path) as f:
        scenes_per_dataset = json.load(f)
    with open(params.train_test_split_path) as f:
        tt_split = json.load(f)

    scenes_of_dataset = scenes_per_dataset[params.dataset_name]
    scenes_of_dataset_eval = set(scenes_of_dataset) & set(tt_split["eval"])
    for scene_index, scene_name in enumerate(scenes_of_dataset_eval):
        scene_path = params.thermoscenes_path / scene_name

        dataset = VGGTSingleDataset(scene_path=scene_path)

        pred_extrinsics_world2cam_list: list[npt.NDArray[np.float64]] = []

        for i in range(len(dataset)):
            dataset_item = dataset[i]
            pred_extrinsics_world2cam_list.append(
                dataset_item.extrinsic_world2cam_rgb.numpy()
            )

        pred_extrinsics_world2cam = np.stack(pred_extrinsics_world2cam_list)
        duration = np.nan
        ratio_reconstructed = 1.0

        with open(scene_path / "transforms.json") as f:
            transforms_pred = json.load(f)

        if params.dataset_name in ["ThermalMix", "ThermalGaussian"]:
            (
                colmap_extrinsics_world2cam,
                colmap_intrinsics,
            ) = load_colmap_poses_thermalgaussian(
                scene_path=params.original_dataset_root / scene_name,
                transforms_reference=transforms_pred,
            )
        if params.dataset_name == "ThermoNeRF":
            (
                colmap_extrinsics_world2cam,
                _,
                pred_extrinsics_world2cam,
                _,
            ) = load_colmap_poses_thermonerf(
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
            dataset_name=params.dataset_name,
        )
        logger.info(
            f"Evaluated {scene_path.name}, ",
            "approx. {scene_index} / {len(scenes_of_dataset_eval)}.",
        )

    # Save the metrics results
    calculator.per_dataset(output_folder / "per_dataset.json")
    calculator.per_scene(output_folder / "per_scene.json")
    calculator.aggregated(output_folder / "aggregated.json")


if __name__ == "__main__":
    params = tyro.cli(EvalParameters)
    main(params=params)
