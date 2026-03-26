import unittest
from itertools import product

import numpy as np
import numpy.typing as npt
import torch
from scipy.spatial.transform import Rotation
from vggt.utils.geometry import closed_form_inverse_se3

from sear.metrics.all import calculate_cameras_metrics
from sear.metrics.ate import (
    align_pred_to_real,
    calculate_cameras_ate,
    umeyama_alignment,
)
from sear.metrics.rpe import calculate_cameras_rpe
from sear.metrics.rra_rta_maa import (
    _calculate_auc,
    _calculate_rotation_error,
    _calculate_translation_error_degree,
    _check_cameras_shapes,
    _error_percent,
    _relative_poses,
    calculate_cameras_rra_rta_maa,
)


class TestMetrics(unittest.TestCase):
    """
    Tests that functions to calculate RRA, RTA, mAA, ATE, RPE work properly.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        torch.manual_seed(0)
        cls._rotation_matrix = Rotation.from_euler("xyz", [1.0, 2.0, 3.0]).as_matrix()
        cls._translation = np.array([1.0, 2.0, 3.0])

    def test_calculate_rotation_error_raises(self) -> None:
        """
        Tests that _calculate_rotation_error raises a RuntimeError for invalid input
        shapes or mismatched shapes.
        """
        invalid_inputs = [
            np.random.uniform(size=(1, 3, 3)).astype(np.float64),
            np.random.uniform(size=(2, 3)).astype(np.float64),
        ]
        for invalid_input in invalid_inputs:
            with self.assertRaises(RuntimeError):
                _calculate_rotation_error(invalid_input, self._rotation_matrix)
        with self.assertRaises(RuntimeError):
            _calculate_rotation_error(invalid_inputs[0], invalid_inputs[1])

    def test_calculate_rotation_error_zero(self) -> None:
        """
        Tests that _calculate_rotation_error returns zero when rotation matrices are
        identical.
        """
        matrix1 = Rotation.from_euler("xyz", [1.0, 2.0, 3.0]).as_matrix()
        self.assertAlmostEqual(_calculate_rotation_error(matrix1, matrix1), 0.0)

    def test_calculate_rotation_error(self) -> None:
        """
        Tests that _calculate_rotation_error returns the correct angular difference
        between two rotation matrices.
        """
        matrix1 = Rotation.from_euler("xyz", [90.0, 0.0, 0.0], degrees=True).as_matrix()
        matrix2 = Rotation.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True).as_matrix()
        self.assertAlmostEqual(_calculate_rotation_error(matrix1, matrix2), 90.0)

    def test_calculate_translation_error_raises(self) -> None:
        """
        Tests that _calculate_translation_error raises a RuntimeError for invalid input
        shapes or mismatched shapes.
        """
        invalid_inputs = [
            np.random.uniform(size=(3, 2)).astype(np.float64),
            np.random.uniform(size=(2,)).astype(np.float64),
            np.random.uniform(size=(1, 1, 3)).astype(np.float64),
        ]
        for invalid_input in invalid_inputs:
            with self.assertRaises(RuntimeError):
                _calculate_translation_error_degree(invalid_input, self._translation)
        with self.assertRaises(RuntimeError):
            _calculate_translation_error_degree(invalid_inputs[0], invalid_inputs[1])

    def test_calculate_translation_error_zero(self) -> None:
        """
        Tests that translation error is zero when translation vectors are identical.
        """
        translation = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(
            _calculate_translation_error_degree(translation, translation),
            0.0,
            places=4,
        )

    def test_calculate_translation_error(self) -> None:
        """
        Tests that _calculate_translation_error returns the correct angular error
        between two translation vectors.
        """
        translation1 = np.array([0.0, 0.0, 1.0])
        translation2 = np.array([0.0, 2.0, 0.0])

        self.assertAlmostEqual(
            _calculate_translation_error_degree(translation1, translation2), 90.0
        )

    def test_calculate_auc_raises(self) -> None:
        """
        Tests that calculate_auc raises a RuntimeError when rotation and translation
        error arrays have different lengths.
        """
        rotation_errors = np.zeros((10,), dtype=np.float64)
        translation_errors = np.zeros((5,), dtype=np.float64)
        with self.assertRaises(RuntimeError):
            _calculate_auc(
                rotation_errors=rotation_errors,
                translation_errors=translation_errors,
                thresholds=[10.0, 10.0],
            )

    def test_calculate_auc_one(self) -> None:
        """
        Tests that calculate_auc returns 1.0 for all thresholds when all rotation and
        translation errors are zero.
        """
        rotation_errors = np.zeros((10,), dtype=np.float64)
        translation_errors = np.zeros((10,), dtype=np.float64)

        result = _calculate_auc(
            rotation_errors=rotation_errors,
            translation_errors=translation_errors,
            thresholds=[5.0, 10.0, 15.0, 30.0],
        )

        expected_result = [1.0, 1.0, 1.0, 1.0]

        self.assertTrue(np.allclose(result, expected_result))

    def test_calculate_auc(self) -> None:
        """
        Tests that calculate_auc returns correct normalized AUC values for a small set
        of predefined rotation and translation errors.
        """
        rotation_errors = np.array([5.0, 10.0, 20.0])
        translation_errors = np.array([5.0, 15.0, 10.0])

        result = _calculate_auc(
            rotation_errors=rotation_errors,
            translation_errors=translation_errors,
            thresholds=[5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        )

        expected_result = [0.0, 0.25, 5 / 18, 11 / 24, 0.6, 2 / 3]

        self.assertTrue(np.allclose(result, expected_result))

    def test_error_percent(self) -> None:
        """
        Tests that error_percent returns correct ratios when given a simple square error
        matrix.
        """
        errors = np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float64,
        )

        thresholds = [1.0, 1.5, 2.5, 10.0]
        result = _error_percent(errors, thresholds)

        expected = [0.0, 1 / 3, 2 / 3, 1.0]

        self.assertTrue(np.allclose(result, expected))

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

    def test_relative_rotation_and_translation(self) -> None:
        """
        Tests that relative rotation and translation work by checking it with a naive
        implementation.
        """
        np.random.seed(42)

        N = 5
        cameras = self._random_cameras(num_cameras=N)

        expected_relative_poses = np.empty((N, N, 4, 4), dtype=np.float64)

        for i in range(N):
            for j in range(N):
                expected_relative_poses[i, j] = (
                    cameras[i] @ closed_form_inverse_se3(cameras[j : j + 1])[0]
                )

        relative_poses = _relative_poses(cameras)

        self.assertTrue(np.allclose(relative_poses, expected_relative_poses, atol=1e-5))

    def test_check_cameras_shapes(self) -> None:
        """
        Tests that _check_cameras_shapes raises error when needed
        """

        invalid_inputs_cross = [
            np.random.uniform(size=(15, 3, 5)),
            np.random.uniform(size=(16, 5, 3)),
            np.random.uniform(size=(1, 17, 4, 4)),
            np.random.uniform(size=(4, 4)),
        ]

        invalid_inputs_pairs = [
            [
                np.random.uniform(size=(15, 3, 4)),
                np.random.uniform(size=(16, 4, 4)),
            ]
        ]

        valid_inputs = [
            [
                np.random.uniform(size=(7, 3, 4)),
                np.random.uniform(size=(7, 4, 4)),
            ],
            [
                np.random.uniform(size=(5, 4, 4)),
                np.random.uniform(size=(5, 3, 4)),
            ],
            [
                np.random.uniform(size=(8, 4, 4)),
                np.random.uniform(size=(8, 4, 4)),
            ],
            [
                np.random.uniform(size=(6, 3, 4)),
                np.random.uniform(size=(6, 3, 4)),
            ],
        ]

        invalid_inputs = (
            list(product(invalid_inputs_cross, invalid_inputs_cross))
            + invalid_inputs_pairs
        )
        for invalid_input in invalid_inputs:
            with self.assertRaises(RuntimeError):
                _check_cameras_shapes(
                    cameras_real=invalid_input[0].astype(np.float64),
                    cameras_pred=invalid_input[1].astype(np.float64),
                )

        for valid_input in valid_inputs:
            _check_cameras_shapes(
                cameras_real=valid_input[0].astype(np.float64),
                cameras_pred=valid_input[1].astype(np.float64),
            )

    def test_calculate_cameras_rra_rta(self) -> None:
        """
        Tests that calculate_cameras_rra_rta produces a proper number of outputs
        """
        output = calculate_cameras_rra_rta_maa(
            cameras_real_world2cam=self._random_cameras(num_cameras=5),
            cameras_pred_world2cam=self._random_cameras(num_cameras=5),
            thresholds=[1.0, 5.0, 10.0],
        )

        self.assertEqual(len(output), 3)
        self.assertEqual(len(output[0]), 3)
        self.assertEqual(len(output[1]), 3)
        self.assertEqual(len(output[2]), 3)

    def test_align(self) -> None:
        """
        Tests that `_align` works for small rotations.
        """
        np.random.seed(42)

        for _ in range(10):
            trajectory_pred = np.random.uniform(size=(15, 3)).astype(np.float64)
            rotation_angles = np.random.uniform(size=(3,)).astype(np.float64) * 30 - 60

            translation_real = np.random.uniform(size=(3,)).astype(np.float64) * 5 - 10
            rotation_real = (
                Rotation.from_euler("xyz", rotation_angles, degrees=True)
                .as_matrix()
                .astype(np.float64)
            )

            trajectory_real = trajectory_pred @ rotation_real.T + translation_real

            rotation_pred, translation_pred, _ = umeyama_alignment(
                x=trajectory_pred,
                y=trajectory_real,
                with_scale=False,
            )

            trajectory_pred_to_real = (
                trajectory_pred @ rotation_pred.T + translation_pred
            )

            self.assertTrue(np.allclose(trajectory_pred_to_real, trajectory_real))

    def test_align_pred_to_real(self) -> None:
        """Tests that `align_pred_to_real` works properly"""
        for _ in range(10):
            cameras_real_cam2world = self._random_cameras(num_cameras=7)
            pose = self._random_cameras(num_cameras=1)[0]
            cameras_pred_cam2world = np.matmul(pose, cameras_real_cam2world)

            cameras_pred_aligned_cam2world, _, _, _ = align_pred_to_real(
                cameras_real_cam2world=cameras_real_cam2world,
                cameras_pred_cam2world=cameras_pred_cam2world,
            )

            self.assertTrue(
                np.allclose(cameras_pred_aligned_cam2world, cameras_real_cam2world)
            )

    def test_calculate_ate(self) -> None:
        """Tests that test_calculate_ate produces a proper number of outputs"""
        output = calculate_cameras_ate(
            cameras_real_world2cam=self._random_cameras(num_cameras=7),
            cameras_pred_world2cam=self._random_cameras(num_cameras=7),
        )
        self.assertEqual(len(output), 3)

    def test_calculate_rpe(self) -> None:
        """Tests that test_calculate_rpe produces a proper number of outputs"""
        output = calculate_cameras_rpe(
            cameras_real_world2cam=self._random_cameras(num_cameras=7),
            cameras_pred_world2cam=self._random_cameras(num_cameras=7),
        )
        self.assertEqual(len(output), 3)

    def test_calculate_cameras_metrics(self) -> None:
        """
        Tests that calculate_cameras_metrics produces a proper number of outputs
        """
        output = calculate_cameras_metrics(
            cameras_real_world2cam=self._random_cameras(num_cameras=7),
            cameras_pred_world2cam=self._random_cameras(num_cameras=7),
            thresholds=[1.0, 5.0, 10.0, 15.0],
        )

        self.assertEqual(len(output), 5)
        for i in range(3):
            self.assertEqual(len(output[i]), 4)

        for i in range(3, len(output)):
            self.assertEqual(len(output[i]), 3)
