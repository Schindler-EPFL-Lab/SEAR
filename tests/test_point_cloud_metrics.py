import unittest

import numpy as np
import numpy.typing as npt
import open3d as o3d
from lightning import seed_everything
from scipy.spatial.transform import Rotation

from sear.metrics.point_cloud_accuracy import (
    calculate_point_cloud_accuracy_and_completeness,
    point_cloud_distances,
)


class TestPointCloudMetrics(unittest.TestCase):
    """
    Tests that functions to evaluate point cloud reconstruction work properly.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        seed_everything(0)

    @staticmethod
    def _random_cameras(num_cameras: int) -> npt.NDArray[np.float64]:
        """Returns `num_cameras` cameras randomly."""
        random_angles = np.random.uniform(size=(num_cameras, 3))
        random_rotation_matrices = np.stack(
            [
                Rotation.from_euler("xyz", random_angle).as_matrix()
                for random_angle in random_angles
            ]
        )

        random_translation = np.random.uniform(size=(num_cameras, 3))
        cameras = np.zeros((num_cameras, 4, 4), dtype=np.float64)
        for i in range(num_cameras):
            cameras[i, :3, :3] = random_rotation_matrices[i]
            cameras[i, :3, 3] = random_translation[i]
            cameras[i, 3, 3] = 1.0
        return cameras

    def test_point_cloud_accuracy_errors_raises(self) -> None:
        """
        Tests that the function `point_cloud_accuracy_errors` raises an error when
        called with incorrect shapes.
        """

        with self.assertRaises(RuntimeError):
            point_cloud_distances(
                point_cloud_from=np.random.rand(5, 3),
                point_cloud_to=np.random.rand(21, 2),
            )

    def test_point_cloud_accuracy_errors(self) -> None:
        """
        Tests that the method `point_cloud_accuracy_errors` works properly
        """

        points_real = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [500.0, 500.0, 501.0],
            ]
        )

        points_pred = np.array(
            [
                [0.0, 0.0, 0.0],
                [500.0, 500.0, 500.0],
            ]
        )

        expected_distances = np.array([1.0, 1.0, 1.0, 0.0, 1.0])
        distances = point_cloud_distances(
            point_cloud_from=points_real, point_cloud_to=points_pred
        )

        self.assertTrue(np.allclose(expected_distances, distances))

    def test_calculate_point_cloud_accuracy_from_errors(self) -> None:
        """Tests that `calculate_point_cloud_accuracy_from_errors` works properly"""

        for _ in range(10):
            num_points_real = np.random.randint(low=5, high=100)
            points_real = np.random.rand(num_points_real, 3)

            num_points_pred = np.random.randint(low=5, high=100)
            points_pred = np.random.rand(num_points_pred, 3)

            errors_1 = point_cloud_distances(
                point_cloud_from=points_real, point_cloud_to=points_pred
            )
            errors_2 = point_cloud_distances(
                point_cloud_from=points_pred, point_cloud_to=points_real
            )

            calculated_chamfer = (errors_1.mean() + errors_2.mean()).item()

            point_cloud_real = o3d.t.geometry.PointCloud()
            point_cloud_real.point.positions = o3d.core.Tensor(
                points_real, dtype=o3d.core.Dtype.Float32
            )
            point_cloud_pred = o3d.t.geometry.PointCloud()
            point_cloud_pred.point.positions = o3d.core.Tensor(
                points_pred, dtype=o3d.core.Dtype.Float32
            )

            chamfer_open3d = (
                point_cloud_real.compute_metrics(
                    point_cloud_pred,
                    (o3d.t.geometry.Metric.ChamferDistance,),
                    o3d.t.geometry.MetricParameters(),
                )
                .numpy()[0]
                .item()
            )

            self.assertAlmostEqual(calculated_chamfer, chamfer_open3d, places=6)

    def test_calculate_point_cloud_accuracy(self) -> None:
        np.random.seed(0)
        for _ in range(10):
            num_frames = np.random.randint(4, 10)
            cameras_real_world2cam = self._random_cameras(num_cameras=num_frames)
            cameras_pred_world2cam = self._random_cameras(num_cameras=num_frames)

            depths_real = np.random.rand(num_frames, 50, 50) * 5 + 0.1
            focal_real = 25 + (-5 + np.random.rand() * 5)
            intrinsics_real = np.array(
                [
                    [focal_real, 0.0, 25.0],
                    [0.0, focal_real, 25.0],
                    [0.0, 0.0, 1.0],
                ]
            )[None].repeat(num_frames, axis=0)

            focal_pred = 25 + (-5 + np.random.rand() * 5)
            intrinsics_pred = np.array(
                [
                    [focal_pred, 0.0, 25.0],
                    [0.0, focal_pred, 25.0],
                    [0.0, 0.0, 1.0],
                ]
            )[None].repeat(num_frames, axis=0)
            depths_pred = np.random.rand(num_frames, 50, 50) * 5 + 0.1

            accuracy, _, _ = calculate_point_cloud_accuracy_and_completeness(
                depths_real=depths_real,
                cameras_real_world2cam=cameras_real_world2cam,
                intrinsics_real=intrinsics_real,
                depths_pred=depths_pred,
                cameras_pred_world2cam=cameras_pred_world2cam,
                intrinsics_pred=intrinsics_pred,
            )

            self.assertTrue(accuracy >= 0.0)
