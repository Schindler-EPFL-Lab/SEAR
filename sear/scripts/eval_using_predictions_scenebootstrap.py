import tyro

from sear import logger
from sear.metrics.boostrap_calculator import BootstrapMetricsCalculator
from sear.scripts.eval_using_predictions import (
    EvalParameters,
    load_one_scene_outputs,
)


class EvalParametersBootstrap(EvalParameters):
    """
    Evaluation parameters for the evaluation using predictions with bootstrap metrics
    calculator.
    """

    num_bootstrap: int = 1000
    """Number of bootstrap samples to use when calculating the metrics."""


def main(params: EvalParametersBootstrap) -> None:
    """
    Evaluates predictions from `params.method_predictions_folder` and saves the results
    to `params.output_folder_root`.
    """

    output_folder = params.output_folder_root / params.method_name
    output_folder.mkdir(exist_ok=True, parents=True)
    calculator = BootstrapMetricsCalculator(
        thresholds=params.thresholds,
        num_bootstrap=params.num_bootstrap,
        calculate_point_cloud_metrics_datasets=[],
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
    calculator.aggregated_bootstrap(output_folder / "aggregated.json")


if __name__ == "__main__":
    params = tyro.cli(EvalParametersBootstrap)
    main(params=params)
