import unittest

import numpy as np

from sear.data_processing.project_points import project_points


class TestProjectPoints(unittest.TestCase):
    """
    Tests that a function to project points works properly.
    """

    def test_project_points(self) -> None:
        """Tests that project points works properly."""
        points = np.array(
            [
                [0.0, 1.0, 0.5],
                [0.0, 1.0, 1.5],
                [1.0, 2.0, 1.5],
                [1.0, -1.0, 0.5],
                [-1.0, -2.0, 0.5],
            ]
        )

        extrinsic_cam2world = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        intrinsic = np.array(
            [
                [2.5, 0, 2.5],
                [0, 2.5, 2.5],
                [0, 0, 1.0],
            ]
        )
        width = 5
        height = 5

        expected_result = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        result = project_points(
            points=points,
            extrinsic_cam2world=extrinsic_cam2world,
            intrinsic=intrinsic,
            width=width,
            height=height,
        )

        self.assertTrue(np.allclose(expected_result, result))
