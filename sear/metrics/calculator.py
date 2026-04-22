import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from vggt.utils.geometry import closed_form_inverse_se3

from sear.metrics.ate import (
    align_pred_to_real,
    calculate_cameras_ate_given_errors,
    rotation_and_translation_errors,
)
from sear.metrics.point_cloud_accuracy import (
    _create_point_cloud,
    calculate_point_cloud_metrics_given_errors,
    point_cloud_distances,
)
from sear.metrics.rpe import calculate_cameras_rpe_given_errors
from sear.metrics.rra_rta_maa import (
    _calculate_rotation_error,
    _calculate_translation_error_degree,
    _check_cameras_shapes,
    calculate_cameras_rra_rta_maa_given_errors,
    relative_rotation_and_translation_errors,
)


@dataclass
class MetricsResult:
    """
    Container class for storing and exporting evaluation metrics related to camera pose
    and trajectory estimation.
    """

    rra: list[float]
    """
    Relative Rotation Accuracy (RRA) values, typically evaluated at multiple rotation
    thresholds.
    """
    rta: list[float]
    """
    Relative Translation Accuracy (RTA) values, typically evaluated at multiple
    translation thresholds.
    """
    maa: list[float]
    """
    Mean Average Accuracy (mAA) values computed over a set of predefined thresholds.
    """

    rpe_rotation: float
    """
    Relative Pose Error (RPE) for rotation, usually measured as a mean or median angular
    error.
    """
    rpe_translation_distance: float
    """Relative Pose Error (RPE) for translation, measured as Euclidean distance."""
    rpe_translation_degree: float
    """Relative Pose Error (RPE) for translation expressed in angular degree space."""

    ate_rotation: float
    """Absolute Trajectory Error (ATE) for rotation."""
    ate_translation_distance: float
    """
    Absolute Trajectory Error (ATE) for translation, measured as Euclidean distance.
    """
    ate_translation_degree: float
    """
    Absolute Trajectory Error (ATE) for translation expressed in angular degree space.
    """

    point_cloud_accuracy: float
    """
    Accuracy metric for point cloud reconstruction which is the mean closest distance
    from points in predicted to the ground truth.
    """

    point_cloud_completeness: float
    """
    Completeness metric for point cloud reconstruction which is the mean closest
    distance from points in ground truth to the predicted.
    """

    chamfer_distance: float
    """Chamfer Distance between the predicted and the ground truth point clouds."""

    ratio_reconstructed: float
    """The ratio of the reconstructed poses or pairs of poses."""

    duration: float
    """The average duration of the method to process scenes"""

    fps: float
    """The average Frame Per Second processing speed"""

    def as_dict(self) -> dict[str, list[float] | float]:
        """
        Converts the stored metrics into a dictionary representation.

        :return: A dictionary mapping metric names to their corresponding values.
        """
        return {
            "RRA": self.rra,
            "RTA": self.rta,
            "mAA": self.maa,
            "RPE_rotation": self.rpe_rotation,
            "RPE_translation_distance": self.rpe_translation_distance,
            "RPE_translation_degree": self.rpe_translation_degree,
            "ATE_rotation": self.ate_rotation,
            "ATE_translation_distance": self.ate_translation_distance,
            "ATE_translation_degree": self.ate_translation_degree,
            "PC_Accuracy": self.point_cloud_accuracy,
            "PC_Completeness": self.point_cloud_completeness,
            "PC_Chamfer": self.chamfer_distance,
            "ratio_reconstructed": self.ratio_reconstructed,
            "duration": self.duration,
            "fps": self.fps,
        }


@dataclass
class PoseErrors:
    """
    Container class for storing pose error values for both relative and absolute pose
    evaluations.
    """

    relative_rotation: npt.NDArray[np.float64]
    """
    Rotation error between the relative motion of two cameras in the ground truth
    trajectory and the relative motion of the corresponding cameras in the predicted
    trajectory.
    """
    relative_translation_distance: npt.NDArray[np.float64]
    """
    Eucledian distance between the relative motion of two cameras in the ground truth
    trajectory and the relative motion of the corresponding cameras in the predicted
    trajectory.
    """
    relative_translation_degree: npt.NDArray[np.float64]
    """
    Distance in degrees between the relative motion of two cameras in the ground truth
    trajectory and the relative motion of the corresponding cameras in the predicted
    trajectory.
    """

    rotation: npt.NDArray[np.float64]
    """Rotation errors between the ground truth and predicted camera poses"""
    translation_distance: npt.NDArray[np.float64]
    """
    Absolute translation errors measured as Euclidean distances. Calculated between the
    ground truth and predicted camera poses.
    """
    translation_degree: npt.NDArray[np.float64]
    """
    Absolute translation errors expressed in degrees. Calculated between the ground
    truth and predicted camera poses.
    """

    point_cloud_distances_from_ground_truth: npt.NDArray[np.float64]
    """
    Distance from the ground truth point cloud to the closest points in predicted point
    cloud.
    """

    point_cloud_distances_from_predicted: npt.NDArray[np.float64]
    """
    Distance from the predicted point cloud to the closest points in predicted point
    cloud.
    """

    ratio_reconstructed: npt.NDArray[np.float64]
    """Reconstruction ratio for poses or pairs of poses."""

    duration: npt.NDArray[np.float64]
    """Durations of the method to process scenes"""

    num_frames: npt.NDArray[np.int64]
    """Number of reconstructed frames"""

    @classmethod
    def empty(cls) -> "PoseErrors":
        """
        Creates an empty :class:`PoseErrors` instance. All fields are initialized as
        empty NumPy.

        :return: An ``PoseErrors`` instance with empty arrays.
        """
        return cls(
            relative_rotation=np.empty((0,), dtype=np.float64),
            relative_translation_distance=np.empty((0,), dtype=np.float64),
            relative_translation_degree=np.empty((0,), dtype=np.float64),
            rotation=np.empty((0,), dtype=np.float64),
            translation_distance=np.empty((0,), dtype=np.float64),
            translation_degree=np.empty((0,), dtype=np.float64),
            point_cloud_distances_from_ground_truth=np.empty((0,), dtype=np.float64),
            point_cloud_distances_from_predicted=np.empty((0,), dtype=np.float64),
            ratio_reconstructed=np.empty((0,), dtype=np.float64),
            duration=np.empty((0,), dtype=np.float64),
            num_frames=np.empty((0,), dtype=np.int64),
        )

    @classmethod
    def from_camera_poses(
        cls,
        cameras_real_world2cam: npt.NDArray[np.float64],
        cameras_pred_world2cam: npt.NDArray[np.float64],
        ratio_reconstructed: float,
        duration: float,
    ) -> "PoseErrors":
        """
        Create `PoseErrors` instance by calculating the camera pose errors between
        `cameras_real_world2cam` and `cameras_pred_world2cam`. The `ratio_reconstructed`
        defines ratio of reconstructed poses or pairs of poses. The `duration` is the
        duration of the method to process the scene.
        """

        _check_cameras_shapes(
            cameras_real=cameras_real_world2cam, cameras_pred=cameras_pred_world2cam
        )

        cameras_real_cam2world = closed_form_inverse_se3(cameras_real_world2cam)
        cameras_pred_cam2world = closed_form_inverse_se3(cameras_pred_world2cam)
        cameras_pred_aligned_cam2world, _, _, _ = align_pred_to_real(
            cameras_real_cam2world=cameras_real_cam2world,
            cameras_pred_cam2world=cameras_pred_cam2world,
        )
        cameras_pred_aligned_world2cam = closed_form_inverse_se3(
            cameras_pred_aligned_cam2world
        )

        (
            current_rotation_errors,
            current_translation_distance,
            current_translation_degree,
        ) = rotation_and_translation_errors(
            cameras_real_cam2world=cameras_real_cam2world,
            cameras_pred_aligned_cam2world=cameras_pred_aligned_cam2world,
        )
        (
            current_relative_rotation,
            current_relative_translation_distance,
            current_relative_translation_degree,
        ) = relative_rotation_and_translation_errors(
            cameras_real_world2cam=cameras_real_world2cam,
            cameras_pred_world2cam=cameras_pred_aligned_world2cam,
        )

        return cls(
            relative_rotation=current_relative_rotation,
            relative_translation_distance=current_relative_translation_distance,
            relative_translation_degree=current_relative_translation_degree,
            rotation=current_rotation_errors,
            translation_distance=current_translation_distance,
            translation_degree=current_translation_degree,
            point_cloud_distances_from_ground_truth=np.empty((0,), dtype=np.float64),
            point_cloud_distances_from_predicted=np.empty((0,), dtype=np.float64),
            ratio_reconstructed=np.array([ratio_reconstructed]),
            duration=np.array([duration]),
            num_frames=np.array([cameras_real_cam2world.shape[0]]),
        )

    @classmethod
    def from_camera_poses_and_depths(
        cls,
        cameras_real_world2cam: npt.NDArray[np.float64],
        depths_real: npt.NDArray[np.float64],
        intrinsics_real: npt.NDArray[np.float64],
        cameras_pred_world2cam: npt.NDArray[np.float64],
        depths_pred: npt.NDArray[np.float64],
        intrinsics_pred: npt.NDArray[np.float64],
        ratio_reconstructed: float,
        duration: float,
    ) -> "PoseErrors":
        """
        Create `PoseErrors` instance by calculating the camera pose errors between
        `cameras_real_world2cam` and `cameras_pred_world2cam`. The `ratio_reconstructed`
        defines ratio of reconstructed poses or pairs of poses. The `duration` is the
        duration of the method to process the scene.
        """

        _check_cameras_shapes(
            cameras_real=cameras_real_world2cam, cameras_pred=cameras_pred_world2cam
        )

        if depths_real.ndim != 3:
            raise ValueError(
                "`depths_real` must have 3 dimensions (N, H1, W1), got "
                + f"{depths_real.shape}"
            )

        if depths_pred.ndim != 3:
            raise ValueError(
                "`depths_pred` must have 3 dimensions (N, H2, W2), got "
                + f"{depths_pred.shape}"
            )

        if intrinsics_real.ndim != 3 or intrinsics_real.shape[-2:] != (3, 3):
            raise ValueError(
                "`intrinsics_real` must have 3 dimensions (N, 3, 3), got "
                + f"{intrinsics_real.shape}"
            )

        if intrinsics_pred.ndim != 3 or intrinsics_pred.shape[-2:] != (3, 3):
            raise ValueError(
                "`intrinsics_pred` must have 3 dimensions (N, 3, 3), got "
                + f"{intrinsics_pred.shape}"
            )

        for element, name in [
            (depths_real, "depths_real"),
            (intrinsics_real, "intrinsics_real"),
            (depths_pred, "depths_pred"),
            (intrinsics_pred, "intrinsics_pred"),
        ]:
            if element.shape[0] != cameras_real_world2cam.shape[0]:
                raise ValueError(
                    "All inputs must share the same first dimension N, expected "
                    + f"N={cameras_real_world2cam.shape[0]} from "
                    + f"`cameras_real_world2cam`, got {element.shape[0]} for `{name}` "
                    + f"with shape {element.shape}."
                )

        pose_errors = cls.from_camera_poses(
            cameras_real_world2cam=cameras_real_world2cam,
            cameras_pred_world2cam=cameras_pred_world2cam,
            ratio_reconstructed=ratio_reconstructed,
            duration=duration,
        )

        cameras_real_cam2world = closed_form_inverse_se3(cameras_real_world2cam)
        cameras_pred_cam2world = closed_form_inverse_se3(cameras_pred_world2cam)
        cameras_pred_aligned_cam2world, _, _, scale = align_pred_to_real(
            cameras_real_cam2world=cameras_real_cam2world,
            cameras_pred_cam2world=cameras_pred_cam2world,
        )
        depths_pred_aligned = depths_pred * scale
        cameras_pred_aligned_world2cam = closed_form_inverse_se3(
            cameras_pred_aligned_cam2world
        )

        all_points_real = _create_point_cloud(
            depths=depths_real,
            cameras_world2cam=cameras_real_world2cam,
            intrinsics=intrinsics_real,
        )
        all_points_pred = _create_point_cloud(
            depths=depths_pred_aligned,
            cameras_world2cam=cameras_pred_aligned_world2cam,
            intrinsics=intrinsics_pred,
        )

        point_cloud_distances_from_ground_truth = point_cloud_distances(
            point_cloud_from=all_points_real, point_cloud_to=all_points_pred
        )

        point_cloud_distances_from_predicted = point_cloud_distances(
            point_cloud_from=all_points_pred, point_cloud_to=all_points_real
        )

        pose_errors.point_cloud_distances_from_ground_truth = (
            point_cloud_distances_from_ground_truth
        )
        pose_errors.point_cloud_distances_from_predicted = (
            point_cloud_distances_from_predicted
        )

        return pose_errors

    @classmethod
    def from_relative_poses(
        cls,
        relative_cameras_real_cam2world: npt.NDArray[np.float64],
        relative_cameras_pred_cam2world: npt.NDArray[np.float64],
    ) -> "PoseErrors":
        """
        Creates `PoseErrors` from relative camera poses in cam2world opencv format,
        which is essential to properly calculate metrics on 2-image settings.
        """

        result = cls.empty()

        relative_rotation_errors = np.zeros(
            (relative_cameras_real_cam2world.shape[0],), dtype=np.float64
        )
        relative_translation_errors_degree = np.zeros(
            (relative_cameras_real_cam2world.shape[0],), dtype=np.float64
        )

        for i in range(relative_cameras_real_cam2world.shape[0]):
            relative_rotation_errors[i] = _calculate_rotation_error(
                rotation_matrix_real=relative_cameras_real_cam2world[i, :3, :3],
                rotation_matrix_pred=relative_cameras_pred_cam2world[i, :3, :3],
            )

            relative_translation_errors_degree[i] = _calculate_translation_error_degree(
                translation_real=relative_cameras_real_cam2world[i, :3, 3],
                translation_pred=relative_cameras_pred_cam2world[i, :3, 3],
            )

        result.relative_rotation = relative_rotation_errors
        result.relative_translation_degree = relative_translation_errors_degree

        return result

    def append(self, other: "PoseErrors") -> None:
        """Appends pose error values from `other`."""
        self.relative_rotation = np.concatenate(
            [self.relative_rotation, other.relative_rotation]
        )
        self.relative_translation_distance = np.concatenate(
            [self.relative_translation_distance, other.relative_translation_distance]
        )
        self.relative_translation_degree = np.concatenate(
            [self.relative_translation_degree, other.relative_translation_degree]
        )
        self.rotation = np.concatenate([self.rotation, other.rotation])
        self.translation_distance = np.concatenate(
            [self.translation_distance, other.translation_distance]
        )
        self.translation_degree = np.concatenate(
            [self.translation_degree, other.translation_degree]
        )
        self.point_cloud_distances_from_ground_truth = np.concatenate(
            [
                self.point_cloud_distances_from_ground_truth,
                other.point_cloud_distances_from_ground_truth,
            ]
        )
        self.point_cloud_distances_from_predicted = np.concatenate(
            [
                self.point_cloud_distances_from_predicted,
                other.point_cloud_distances_from_predicted,
            ]
        )
        self.ratio_reconstructed = np.concatenate(
            [self.ratio_reconstructed, other.ratio_reconstructed]
        )
        self.duration = np.concatenate([self.duration, other.duration])
        self.num_frames = np.concatenate([self.num_frames, other.num_frames])


class MetricsCalculator:
    """
    Class for accumulating per-scene camera pose error statistics and computing
    higher-level metrics (RRA, RTA, mAA, RPE, ATE, etc.) per scene, per dataset,
    or aggregated across all scenes.
    """

    def __init__(
        self,
        thresholds: list[float],
        calculate_point_cloud_metrics_datasets: list[str] | None = None,
    ) -> None:
        """
        Initializes the class. The `thresholds` are used for computing some metrics
        (e.g. RRA/RTA/mAA).
        """
        if calculate_point_cloud_metrics_datasets is None:
            calculate_point_cloud_metrics_datasets = ["ORU"]

        self._poses_errors: dict[str, PoseErrors] = {}
        self._thresholds = thresholds
        self._scenes_in_datasets: dict[str, set[str]] = {}
        self._calculate_point_cloud_metrics_datasets = set(
            calculate_point_cloud_metrics_datasets
        )

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

    def add_data_relative(
        self,
        relative_cameras_real_cam2world: npt.NDArray[np.float64],
        relative_cameras_pred_cam2world: npt.NDArray[np.float64],
        scene_name: str,
        dataset_name: str,
    ) -> None:
        """
        Adds relative camera pose errors between `cameras_real_cam2world` and
        `cameras_pred_cam2world` of a scene `scene_name` of a dataset `dataset_name`.
        """

        self._append_to_dict(
            errors=self._poses_errors,
            key=scene_name,
            value=PoseErrors.from_relative_poses(
                relative_cameras_real_cam2world=relative_cameras_real_cam2world,
                relative_cameras_pred_cam2world=relative_cameras_pred_cam2world,
            ),
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

            ate_rotation = ate_translation = ate_translation_degree = np.nan
            if (
                len(poses_errors[key].rotation) > 0
                and len(poses_errors[key].translation_distance) > 0
                and len(poses_errors[key].translation_degree) > 0
            ):
                ate_rotation, ate_translation, ate_translation_degree = (
                    calculate_cameras_ate_given_errors(
                        rotation_errors=poses_errors[key].rotation,
                        translation_errors_distance=poses_errors[
                            key
                        ].translation_distance,
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

    def per_scene(self, save_path: Path | None = None) -> dict[str, MetricsResult]:
        """
        Computes evaluation metrics independently for each scene. If `save_path` is
        provided, the computed metrics are stored on disk.

        :return: A dictionary where keys are ``"<dataset_name>:<scene_name>"`` and
            metrics results on scene `scene_name` of dataset `dataset_name`.
        """
        return self._calculate_metrics(
            poses_errors={
                f"{dataset_name}:{scene_name}": self._poses_errors[scene_name]
                for dataset_name, scenes_names in self._scenes_in_datasets.items()
                for scene_name in scenes_names
            },
            thresholds=self._thresholds,
            save_path=save_path,
        )

    def per_dataset(self, save_path: Path | None = None) -> dict[str, MetricsResult]:
        """
        Computes evaluation metrics independently for each dataset. If `save_path` is
        provided, the computed metrics are stored on disk.

        :return: A dictionary where keys are dataset names and values are metrics
            results aggregated over all scenes belonging to each dataset.
        """
        per_dataset_errors: dict[str, PoseErrors] = {}
        for dataset_name in self._scenes_in_datasets:
            for scene_name in self._scenes_in_datasets[dataset_name]:
                self._append_to_dict(
                    per_dataset_errors,
                    key=dataset_name,
                    value=self._poses_errors[scene_name],
                )

        return self._calculate_metrics(
            poses_errors=per_dataset_errors,
            thresholds=self._thresholds,
            save_path=save_path,
        )

    def custom_aggregation(
        self, aggregation_per_scene: dict[str, list[str]], save_path: Path | None = None
    ) -> dict[str, MetricsResult]:
        """
        Calculates metrics when scenes are aggregated in custom way defined in
        `aggregation_per_scene`, which has format of
        ```json
        {
            "aggregation_name": ["scene_name_1", ..., "scene_name_n"],
            ...
        }
        ```
        If `save_path` is provided, the computed metrics are stored on disk.

        :return: Metrics results computed over scenes in `aggregation_per_scene`.
        """

        # validate `aggregation_per_scene`
        scene_per_aggregation: dict[str, str] = {}
        for aggregation_name in aggregation_per_scene:
            for scene_name in aggregation_per_scene[aggregation_name]:
                if scene_name not in self._poses_errors:
                    raise RuntimeError(
                        "The scenes in `aggregation_per_scene` must be already added "
                        + f"to the calculator, while {scene_name} is not in "
                        + f"{list(self._poses_errors.keys())}."
                    )
                if scene_name in scene_per_aggregation:
                    raise RuntimeError(
                        "The scenes in `aggregation_per_scene` must be specified once, "
                        + f"but {scene_name} is specified for "
                        + f"{scene_per_aggregation[scene_name]} and for "
                        + f"{aggregation_name}."
                    )
                scene_per_aggregation[scene_name] = aggregation_name

        per_aggregation_errors: dict[str, PoseErrors] = {}
        for aggregation_name in aggregation_per_scene:
            for scene_name in aggregation_per_scene[aggregation_name]:
                self._append_to_dict(
                    per_aggregation_errors,
                    key=aggregation_name,
                    value=self._poses_errors[scene_name],
                )

        return self._calculate_metrics(
            poses_errors=per_aggregation_errors,
            thresholds=self._thresholds,
            save_path=save_path,
        )

    def aggregated(self, save_path: Path | None = None) -> MetricsResult:
        """
        Computes evaluation metrics aggregated over all datasets and scenes. If
        `save_path` is provided, the computed metrics are stored on disk.

        :return: Metrics results computed over all scenes.
        """
        aggregated_errors: dict[str, PoseErrors] = {}
        for scene_name in self._poses_errors:
            self._append_to_dict(
                errors=aggregated_errors,
                key="aggregated",
                value=self._poses_errors[scene_name],
            )

        aggregated_metrics = self._calculate_metrics(
            poses_errors=aggregated_errors,
            thresholds=self._thresholds,
            save_path=save_path,
        )

        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump(aggregated_metrics["aggregated"].as_dict(), f, indent=4)

        return aggregated_metrics["aggregated"]
