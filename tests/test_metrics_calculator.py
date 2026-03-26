import unittest

import numpy as np
import numpy.typing as npt
import torch
from lightning import seed_everything
from scipy.spatial.transform import Rotation

from sear.metrics.calculator import MetricsCalculator, PoseErrors


class TestMetricsCalculator(unittest.TestCase):
    """
    Tests that MetricsCalculator works properly.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        seed_everything(0)

    @staticmethod
    def _random_cameras(num_frames: int) -> npt.NDArray[np.float64]:
        """Returns `num_cameras` cameras randomly."""
        random_angles = np.random.uniform(size=(num_frames, 3))
        random_rotation_matrices = np.stack(
            [
                Rotation.from_euler("xyz", random_angle).as_matrix()
                for random_angle in random_angles
            ]
        )

        random_translation = np.random.uniform(size=(num_frames, 3))
        cameras = np.zeros((num_frames, 4, 4), dtype=np.float64)
        for i in range(num_frames):
            cameras[i, :3, :3] = random_rotation_matrices[i]
            cameras[i, :3, 3] = random_translation[i]
            cameras[i, 3, 3] = 1.0
        return cameras

    @staticmethod
    def _random_depths(num_frames: int) -> npt.NDArray[np.float64]:
        """Returns `num_frames` random depths."""
        return np.random.rand(num_frames, 30, 50) * 5

    @staticmethod
    def _random_intrinsics(num_frames: int) -> npt.NDArray[np.float64]:
        """Returns `num_frames` random intrinsics."""
        focals = 25 + (2 * np.random.rand(num_frames) - 1) * 5
        result = np.zeros((num_frames, 3, 3))
        result[:, 2, 2] = 1.0
        result[:, 0, 0] = focals
        result[:, 1, 1] = focals
        result[:, 0, 2] = 25.0
        result[:, 1, 2] = 15.0
        return result

    def test_add_data_single(self) -> None:
        """
        Tests that the method `add_data` works if single scene results are added
        """
        calculator = MetricsCalculator(
            thresholds=[1.0, 5.0, 30.0],
            calculate_point_cloud_metrics_datasets=["dataset_1"],
        )

        calculator.add_data(
            cameras_real_world2cam=self._random_cameras(num_frames=10),
            depths_real=self._random_depths(num_frames=10),
            intrinsics_real=self._random_intrinsics(num_frames=10),
            cameras_pred_world2cam=self._random_cameras(num_frames=10),
            depths_pred=self._random_depths(num_frames=10),
            intrinsics_pred=self._random_intrinsics(num_frames=10),
            ratio_reconstructed=0.5,
            duration=10.0,
            scene_name="scene_1",
            dataset_name="dataset_1",
        )

        self.assertTrue("scene_1" in calculator._poses_errors)
        self.assertEqual(len(calculator._poses_errors["scene_1"].relative_rotation), 45)
        self.assertEqual(len(calculator._poses_errors["scene_1"].rotation), 10)
        self.assertEqual({"dataset_1": {"scene_1"}}, calculator._scenes_in_datasets)

        calculator.add_data(
            cameras_real_world2cam=self._random_cameras(num_frames=15),
            depths_real=self._random_depths(num_frames=15),
            intrinsics_real=self._random_intrinsics(num_frames=15),
            cameras_pred_world2cam=self._random_cameras(num_frames=15),
            depths_pred=self._random_depths(num_frames=15),
            intrinsics_pred=self._random_intrinsics(num_frames=15),
            ratio_reconstructed=0.7,
            duration=11.0,
            scene_name="scene_1",
            dataset_name="dataset_1",
        )

        self.assertTrue("scene_1" in calculator._poses_errors)
        self.assertEqual(
            len(calculator._poses_errors["scene_1"].relative_rotation), 150
        )
        self.assertEqual(len(calculator._poses_errors["scene_1"].rotation), 25)
        self.assertEqual({"dataset_1": {"scene_1"}}, calculator._scenes_in_datasets)

    def test_add_data(self) -> None:
        """
        Tests that the method `add_data` works if multiple scenes results are added
        """
        calculator = MetricsCalculator(
            thresholds=[1.0, 5.0, 30.0],
            calculate_point_cloud_metrics_datasets=["dataset_2"],
        )

        calculator.add_data(
            cameras_real_world2cam=self._random_cameras(num_frames=10),
            depths_real=self._random_depths(num_frames=10),
            intrinsics_real=self._random_intrinsics(num_frames=10),
            cameras_pred_world2cam=self._random_cameras(num_frames=10),
            depths_pred=self._random_depths(num_frames=10),
            intrinsics_pred=self._random_intrinsics(num_frames=10),
            ratio_reconstructed=0.1,
            duration=12.0,
            scene_name="scene_1",
            dataset_name="dataset_1",
        )

        self.assertTrue("scene_1" in calculator._poses_errors)
        self.assertEqual(len(calculator._poses_errors["scene_1"].relative_rotation), 45)
        self.assertEqual(len(calculator._poses_errors["scene_1"].rotation), 10)
        self.assertEqual(
            len(calculator._poses_errors["scene_1"].ratio_reconstructed), 1
        )
        self.assertEqual({"dataset_1": {"scene_1"}}, calculator._scenes_in_datasets)

        calculator.add_data(
            cameras_real_world2cam=self._random_cameras(num_frames=8),
            depths_real=self._random_depths(num_frames=8),
            intrinsics_real=self._random_intrinsics(num_frames=8),
            cameras_pred_world2cam=self._random_cameras(num_frames=8),
            depths_pred=self._random_depths(num_frames=8),
            intrinsics_pred=self._random_intrinsics(num_frames=8),
            ratio_reconstructed=0.2,
            duration=13.0,
            scene_name="scene_2",
            dataset_name="dataset_1",
        )

        self.assertTrue("scene_1" in calculator._poses_errors)
        self.assertEqual(len(calculator._poses_errors["scene_1"].relative_rotation), 45)
        self.assertEqual(len(calculator._poses_errors["scene_1"].rotation), 10)
        self.assertEqual(
            len(calculator._poses_errors["scene_1"].ratio_reconstructed), 1
        )
        self.assertTrue("scene_2" in calculator._poses_errors)
        self.assertEqual(len(calculator._poses_errors["scene_2"].relative_rotation), 28)
        self.assertEqual(len(calculator._poses_errors["scene_2"].rotation), 8)
        self.assertEqual(
            len(calculator._poses_errors["scene_2"].ratio_reconstructed), 1
        )
        self.assertEqual(
            {"dataset_1": {"scene_1", "scene_2"}}, calculator._scenes_in_datasets
        )

        calculator.add_data(
            cameras_real_world2cam=self._random_cameras(num_frames=17),
            depths_real=self._random_depths(num_frames=17),
            intrinsics_real=self._random_intrinsics(num_frames=17),
            cameras_pred_world2cam=self._random_cameras(num_frames=17),
            depths_pred=self._random_depths(num_frames=17),
            intrinsics_pred=self._random_intrinsics(num_frames=17),
            ratio_reconstructed=0.3,
            duration=9.0,
            scene_name="scene_3",
            dataset_name="dataset_2",
        )

        self.assertTrue("scene_1" in calculator._poses_errors)
        self.assertEqual(len(calculator._poses_errors["scene_1"].relative_rotation), 45)
        self.assertEqual(len(calculator._poses_errors["scene_1"].rotation), 10)
        self.assertTrue("scene_2" in calculator._poses_errors)
        self.assertEqual(len(calculator._poses_errors["scene_2"].relative_rotation), 28)
        self.assertEqual(len(calculator._poses_errors["scene_2"].rotation), 8)
        self.assertTrue("scene_3" in calculator._poses_errors)
        self.assertEqual(
            len(calculator._poses_errors["scene_3"].relative_rotation), 136
        )
        self.assertEqual(
            len(calculator._poses_errors["scene_1"].ratio_reconstructed), 1
        )
        self.assertEqual(len(calculator._poses_errors["scene_3"].rotation), 17)
        self.assertEqual(
            {"dataset_1": {"scene_1", "scene_2"}, "dataset_2": {"scene_3"}},
            calculator._scenes_in_datasets,
        )

    def test_calculate_metrics_single(self) -> None:
        """Tests that _calculate_metrics method works properly on one scene."""

        for _ in range(10):
            num_scenes = int(torch.randint(1, 7, size=(1,)).item())
            scenes_ids = [
                int(torch.randint(1, 7, size=(1,)).item()) for _ in range(num_scenes)
            ]
            scenes_names = [f"scene_{scene_id}" for scene_id in scenes_ids]

            pose_errors: dict[str, PoseErrors] = {}
            for scene_name in scenes_names:
                N_cameras = int(torch.randint(2, 100, size=(1,)).item())
                pose_errors[scene_name] = PoseErrors.from_camera_poses_and_depths(
                    cameras_real_world2cam=self._random_cameras(num_frames=N_cameras),
                    depths_real=self._random_depths(num_frames=N_cameras),
                    intrinsics_real=self._random_intrinsics(num_frames=N_cameras),
                    cameras_pred_world2cam=self._random_cameras(num_frames=N_cameras),
                    depths_pred=self._random_depths(num_frames=N_cameras),
                    intrinsics_pred=self._random_intrinsics(num_frames=N_cameras),
                    ratio_reconstructed=0.5,
                    duration=100500,
                )

            results = MetricsCalculator._calculate_metrics(
                thresholds=[1.0, 5.0, 10.0, 15.0, 30.0],
                poses_errors=pose_errors,
            )

            self.assertEqual(set(results.keys()), set(scenes_names))

    def test_clear(self) -> None:
        """Tests that the method `clear` works correct."""
        calculator = MetricsCalculator(
            thresholds=[1.0, 5.0, 30.0],
            calculate_point_cloud_metrics_datasets=["dataset_1"],
        )

        calculator.add_data(
            cameras_real_world2cam=self._random_cameras(num_frames=10),
            depths_real=self._random_depths(num_frames=10),
            intrinsics_real=self._random_intrinsics(num_frames=10),
            cameras_pred_world2cam=self._random_cameras(num_frames=10),
            depths_pred=self._random_depths(num_frames=10),
            intrinsics_pred=self._random_intrinsics(num_frames=10),
            scene_name="scene_1",
            dataset_name="dataset_1",
            ratio_reconstructed=np.random.rand(),
            duration=np.random.rand() * 123,
        )

        calculator.clear()
        self.assertTrue("scene_1" not in calculator._poses_errors)

        calculator.add_data(
            cameras_real_world2cam=self._random_cameras(num_frames=10),
            depths_real=self._random_depths(num_frames=10),
            intrinsics_real=self._random_intrinsics(num_frames=10),
            cameras_pred_world2cam=self._random_cameras(num_frames=10),
            depths_pred=self._random_depths(num_frames=10),
            intrinsics_pred=self._random_intrinsics(num_frames=10),
            ratio_reconstructed=np.random.rand(),
            scene_name="scene_1",
            dataset_name="dataset_1",
            duration=np.random.rand() * 321,
        )

        self.assertTrue("scene_1" in calculator._poses_errors)
        self.assertEqual(len(calculator._poses_errors["scene_1"].relative_rotation), 45)
        self.assertEqual(len(calculator._poses_errors["scene_1"].rotation), 10)

        calculator.clear()

        calculator.add_data(
            cameras_real_world2cam=self._random_cameras(num_frames=10),
            depths_real=self._random_depths(num_frames=10),
            intrinsics_real=self._random_intrinsics(num_frames=10),
            cameras_pred_world2cam=self._random_cameras(num_frames=10),
            depths_pred=self._random_depths(num_frames=10),
            intrinsics_pred=self._random_intrinsics(num_frames=10),
            ratio_reconstructed=np.random.rand(),
            duration=np.random.rand() * 228,
            scene_name="scene_2",
            dataset_name="dataset_1",
        )

        self.assertTrue("scene_1" not in calculator._poses_errors)
        self.assertTrue("scene_2" in calculator._poses_errors)
        self.assertEqual(len(calculator._poses_errors["scene_2"].relative_rotation), 45)
        self.assertEqual(len(calculator._poses_errors["scene_2"].rotation), 10)
