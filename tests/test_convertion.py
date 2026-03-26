import unittest

import torch

from sear.data_processing.convertion import opengl_to_opencv


class TestConvention(unittest.TestCase):
    """
    Tests that FrameInfo class works properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        pass

    def test_opengl_to_opencv_raises(self) -> None:
        """
        Tests that `opengl_to_opencv` raises when necessary.
        """
        incorrect_shapes = [
            (1, 2, 4),
            (2, 5, 4),
            (1, 1, 3, 4),
            (1, 1, 4, 4),
            (1, 1, 4, 4),
            (1, 5, 3, 4),
        ]

        for incorrect_shape in incorrect_shapes:
            with self.assertRaises(RuntimeError):
                opengl_to_opencv(torch.rand(incorrect_shape))

    def test_opengl_to_opencv(self) -> None:
        """
        Tests that `opengl_to_opencv` works properly.
        """

        pose_cam2world_opengl_4x4 = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        pose_cam2world_opengl_3x4 = pose_cam2world_opengl_4x4[:3, :4]
        expected_pose_cam2world_opencv_4x4 = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0, 2.0],
                [0.0, 0.0, -1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        expected_pose_cam2world_opencv_3x4 = expected_pose_cam2world_opencv_4x4[:3, :4]
        pose_cam2world_opencv_4x4 = opengl_to_opencv(pose_cam2world_opengl_4x4)
        pose_cam2world_opencv_3x4 = opengl_to_opencv(pose_cam2world_opengl_3x4)
        self.assertTrue(
            torch.allclose(
                expected_pose_cam2world_opencv_4x4, pose_cam2world_opencv_4x4
            ),
        )
        self.assertTrue(
            torch.allclose(
                expected_pose_cam2world_opencv_3x4, pose_cam2world_opencv_3x4
            )
        )
