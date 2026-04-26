import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from sear.metrics.ate import (
    calculate_cameras_ate_given_errors,
)
from sear.metrics.calculator import MetricsResult, PoseErrors
from sear.metrics.point_cloud_accuracy import (
    calculate_point_cloud_metrics_given_errors,
)
from sear.metrics.rpe import calculate_cameras_rpe_given_errors
from sear.metrics.rra_rta_maa import (
    calculate_cameras_rra_rta_maa_given_errors,
)


class BootstrapMetricsCalculator:
    """
    Class for accumulating per-scene camera pose error statistics and computing
    higher-level metrics (RRA, RTA, mAA, RPE, ATE, etc.) per scene, per dataset, or
    aggregated across all scenes.

    It samples scenes with replacement to compute metrics on bootstrapped samples of the
    data, which allows estimating the variability of the metrics. This is better than
    sampling frames with replacement, because the metrics on frames of one scene are not
    independent and therefore the confidence intervals computed by sampling frames with
    replacement are not correct and can be underestimated.
    """

    def __init__(
        self,
        thresholds: list[float],
        calculate_point_cloud_metrics_datasets: list[str] | None = None,
        num_bootstrap: int = 0,
    ) -> None:
        """
        Initializes the class. The `thresholds` are used for computing some metrics
        (e.g. RRA/RTA/mAA). The `calculate_point_cloud_metrics_datasets` is a list of
        dataset names for which the point cloud metrics should be calculated. The
        `num_bootstrap` is the number of bootstrap samples to use when calculating the
        metrics.
        """
        if calculate_point_cloud_metrics_datasets is None:
            calculate_point_cloud_metrics_datasets = ["ORU"]

        self._poses_errors: dict[str, PoseErrors] = {}
        self._thresholds = thresholds
        self._scenes_in_datasets: dict[str, set[str]] = {}
        self._calculate_point_cloud_metrics_datasets = set(
            calculate_point_cloud_metrics_datasets
        )
        self._num_bootstrap = num_bootstrap

    @staticmethod
    def _append_to_dict(
        errors: dict[str, PoseErrors],
        key: str,
        value: PoseErrors,
    ) -> None:
        """
        Appends new poses errors `value` related to scene or other attribute with name
        `key` to the `errors` storing all the pose errors.
        """
        if key not in errors:
            errors[key] = PoseErrors.empty()
        errors[key].append(value)

    def clear(self) -> None:
        """Deleting the internal states of the class."""
        self._poses_errors.clear()
        self._scenes_in_datasets.clear()

    def add_data(
        self,
        cameras_real_world2cam: npt.NDArray[np.float64],
        depths_real: npt.NDArray[np.float64],
        intrinsics_real: npt.NDArray[np.float64],
        cameras_pred_world2cam: npt.NDArray[np.float64],
        depths_pred: npt.NDArray[np.float64],
        intrinsics_pred: npt.NDArray[np.float64],
        ratio_reconstructed: float,
        duration: float,
        scene_name: str,
        dataset_name: str,
    ) -> None:
        """
        Adds camera pose errors between `cameras_real_world2cam` and
        `cameras_pred_world2cam`. The ground truth depths `depths_real` and ground truth
        intrinsics `intrinsics_real`, and predicted depths `depths_pred` and predicted
        intrinsics `intrinsics_pred` are used to calculated point clouds metrics. The
        scene is `scene_name`, and the dataset is `dataset_name`. The duration is the
        time for what the method was running to predict the values.
        """

        if dataset_name in self._calculate_point_cloud_metrics_datasets:
            pose_errors = PoseErrors.from_camera_poses_and_depths(
                cameras_real_world2cam=cameras_real_world2cam,
                depths_real=depths_real,
                intrinsics_real=intrinsics_real,
                cameras_pred_world2cam=cameras_pred_world2cam,
                depths_pred=depths_pred,
                intrinsics_pred=intrinsics_pred,
                ratio_reconstructed=ratio_reconstructed,
                duration=duration,
            )
        else:
            pose_errors = PoseErrors.from_camera_poses(
                cameras_real_world2cam=cameras_real_world2cam,
                cameras_pred_world2cam=cameras_pred_world2cam,
                ratio_reconstructed=ratio_reconstructed,
                duration=duration,
            )

        self._append_to_dict(
            errors=self._poses_errors,
            key=scene_name,
            value=pose_errors,
        )

        if dataset_name not in self._scenes_in_datasets:
            self._scenes_in_datasets[dataset_name] = set()
        self._scenes_in_datasets[dataset_name].add(scene_name)

    @staticmethod
    def _calculate_metrics(
        poses_errors: dict[str, PoseErrors],
        thresholds: list[float],
        save_path: Path | None = None,
    ) -> dict[str, MetricsResult]:
        """
        Iterates over `poses_errors` and calculate camera pose metrics. Optionally
        stores the result in `save_path` if provided. The `thresholds` are used for
        computing some metrics (e.g. RRA/RTA/mAA).
        """

        result: dict[str, MetricsResult] = {}

        for key in poses_errors:
            rra, rta, maa = calculate_cameras_rra_rta_maa_given_errors(
                relative_rotation_errors=poses_errors[key].relative_rotation,
                relative_translation_errors_degree=poses_errors[
                    key
                ].relative_translation_degree,
                thresholds=thresholds,
            )

            rpe_rotation, rpe_translation, rpe_translation_degree = (
                calculate_cameras_rpe_given_errors(
                    relative_rotation_errors=poses_errors[key].relative_rotation,
                    relative_translation_errors_distance=poses_errors[
                        key
                    ].relative_translation_distance,
                    relative_translation_errors_degree=poses_errors[
                        key
                    ].relative_translation_degree,
                )
            )

            ate_rotation, ate_translation, ate_translation_degree = (
                calculate_cameras_ate_given_errors(
                    rotation_errors=poses_errors[key].rotation,
                    translation_errors_distance=poses_errors[key].translation_distance,
                    translation_errors_degree=poses_errors[key].translation_degree,
                )
            )

            accuracy, completeness, chamfer = (
                calculate_point_cloud_metrics_given_errors(
                    point_cloud_errors_from_predicted=poses_errors[
                        key
                    ].point_cloud_distances_from_predicted,
                    point_cloud_errors_from_ground_truth=poses_errors[
                        key
                    ].point_cloud_distances_from_ground_truth,
                )
            )

            num_frames_real = poses_errors[key].num_frames / (
                poses_errors[key].ratio_reconstructed + 1e-5
            )
            fps = num_frames_real / (poses_errors[key].duration + 1e-5)

            result[key] = MetricsResult(
                rra=rra,
                rta=rta,
                maa=maa,
                rpe_rotation=rpe_rotation,
                rpe_translation_distance=rpe_translation,
                rpe_translation_degree=rpe_translation_degree,
                ate_rotation=ate_rotation,
                ate_translation_distance=ate_translation,
                ate_translation_degree=ate_translation_degree,
                point_cloud_accuracy=accuracy,
                point_cloud_completeness=completeness,
                chamfer_distance=chamfer,
                ratio_reconstructed=np.mean(
                    poses_errors[key].ratio_reconstructed
                ).item(),
                duration=np.mean(poses_errors[key].duration).item(),
                fps=np.mean(fps).item(),
            )

        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(
                    {
                        key: metrics_result.as_dict()
                        for key, metrics_result in result.items()
                    },
                    f,
                    indent=4,
                )

        return result

    def aggregated_bootstrap(
        self, save_path: Path | None = None
    ) -> list[MetricsResult]:
        """
        Computes evaluation metrics aggregated over all datasets and scenes. If
        `save_path` is provided, the computed metrics are stored on disk.

        :return: Metrics results computed over all scenes.
        """

        all_scenes_names = list(self._poses_errors.keys())
        aggregated_metrics: list[dict[str, MetricsResult]] = []

        for boot_idx in tqdm(range(self._num_bootstrap)):
            rng = np.random.default_rng(boot_idx)
            boot_scenes_names = rng.choice(
                all_scenes_names, len(all_scenes_names), replace=True
            ).tolist()
            boot_aggregated_errors: dict[str, PoseErrors] = {}

            for scene_name in boot_scenes_names:
                self._append_to_dict(
                    errors=boot_aggregated_errors,
                    key="aggregated",
                    value=self._poses_errors[scene_name],
                )

            boot_aggregated_metrics = self._calculate_metrics(
                poses_errors=boot_aggregated_errors,
                thresholds=self._thresholds,
                save_path=None,
            )
            aggregated_metrics.append(boot_aggregated_metrics)

        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(
                    [agg["aggregated"].as_dict() for agg in aggregated_metrics],
                    f,
                    indent=4,
                )

        return [agg["aggregated"] for agg in aggregated_metrics]
