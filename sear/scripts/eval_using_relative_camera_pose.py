import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from sear import logger
from sear.metrics.calculator import MetricsCalculator


@dataclass(kw_only=True)
class EvalParameters(ReverseCli):
    """A config to evaluate outputs of a method using the stored results"""

    method_predictions_file: Path
    """A json containing relative poses"""
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
            self.method_name = self.method_predictions_file.stem
        if self.thresholds is None:
            self.thresholds = [5.0, 10.0, 20.0]


def main(params: EvalParameters) -> None:
    """
    Evaluates predictions from `params.method_predictions_folder` and saves the results
    to `params.output_folder_root`.
    """

    output_folder = params.output_folder_root / params.method_name
    output_folder.mkdir(exist_ok=True, parents=True)
    calculator = MetricsCalculator(thresholds=params.thresholds)

    with open(params.method_predictions_file) as f:
        data = json.load(f)

    for scene_index, scene_name in enumerate(data.keys()):
        relative_transforms_pred = np.array(data[scene_name]["pred"])
        relative_transforms_real = np.array(data[scene_name]["real"])

        if len(relative_transforms_real) != len(relative_transforms_pred):
            raise RuntimeError("Lengths should be equal")

        calculator.add_data_relative(
            relative_cameras_real_cam2world=relative_transforms_pred,
            relative_cameras_pred_cam2world=relative_transforms_real,
            scene_name=scene_name,
            dataset_name="METU-VisTIR",
        )

        logger.info(f"Evaluated {scene_name}, approx. {scene_index} / {len(data)}.")

    # Save the metrics results
    calculator.per_dataset(output_folder / "per_dataset.json")
    calculator.per_scene(output_folder / "per_scene.json")
    calculator.aggregated(output_folder / "aggregated.json")


if __name__ == "__main__":
    params = tyro.cli(EvalParameters)
    main(params=params)
