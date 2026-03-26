import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Remove it when https://github.com/facebookresearch/vggt/issues/416 is fixed
sys.path.append("vggt")

from sear.data_processing.frame_info import FrameInfo


class TestFrameInfo(unittest.TestCase):
    """
    Tests that FrameInfo class works properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """

        cls._extrinsics_world2cam = [
            torch.tensor(
                [
                    [
                        0.9566335678100586,
                        0.0766449123620987,
                        0.2810298800468445,
                        -0.33123430609703064,
                    ],
                    [
                        -0.06353233754634857,
                        0.9964358806610107,
                        -0.05549079179763794,
                        -0.007694344501942396,
                    ],
                    [
                        -0.2842813730239868,
                        0.035229869186878204,
                        0.9580934047698975,
                        -0.0710090845823288,
                    ],
                ]
            ),
            torch.tensor(
                [
                    [
                        0.9998975396156311,
                        -0.014294284395873547,
                        0.0007807575748302042,
                        0.07004871964454651,
                    ],
                    [
                        0.014267958700656891,
                        0.9995326399803162,
                        0.02703489549458027,
                        0.011531968601047993,
                    ],
                    [
                        -0.0011668370570987463,
                        -0.027020985260605812,
                        0.9996342062950134,
                        0.02415219321846962,
                    ],
                ]
            ),
        ]

        cls._intrinsics = [
            torch.tensor(
                [
                    [615.5011596679688, 0.0, 259.0],
                    [0.0, 612.0875244140625, 259.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            torch.tensor(
                [
                    [619.112548828125, 0.0, 259.0],
                    [0.0, 616.6370239257812, 259.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ]

        cls._frame_dicts = [
            {
                "transform_matrix": [
                    [
                        0.9566335678100586,
                        0.0766449123620987,
                        0.2810298800468445,
                        -0.33123430609703064,
                    ],
                    [
                        -0.06353233754634857,
                        0.9964358806610107,
                        -0.05549079179763794,
                        -0.007694344501942396,
                    ],
                    [
                        -0.2842813730239868,
                        0.035229869186878204,
                        0.9580934047698975,
                        -0.0710090845823288,
                    ],
                ],
                "w": 518,
                "h": 518,
                "fl_x": 615.5011596679688,
                "fl_y": 612.0875244140625,
                "k1": 0,
                "k2": 0,
                "k3": 0,
                "k4": 0,
                "p1": 0,
                "p2": 0,
                "cx": 259.0,
                "cy": 259.0,
                "camera_angle_x": 0.7966077035974845,
                "camera_angle_y": 0.8005918738086296,
                "fovx": 45.64225934374431,
                "fovy": 45.87053548170467,
                "file_path": "images/frame_eval_00003.JPG",
                "depth_file_path": "depths/frame_eval_00003.npy",
            },
            {
                "transform_matrix": [
                    [
                        0.9998975396156311,
                        -0.014294284395873547,
                        0.0007807575748302042,
                        0.07004871964454651,
                    ],
                    [
                        0.014267958700656891,
                        0.9995326399803162,
                        0.02703489549458027,
                        0.011531968601047993,
                    ],
                    [
                        -0.0011668370570987463,
                        -0.027020985260605812,
                        0.9996342062950134,
                        0.02415219321846962,
                    ],
                ],
                "w": 518,
                "h": 518,
                "fl_x": 619.112548828125,
                "fl_y": 616.6370239257812,
                "k1": 0,
                "k2": 0,
                "k3": 0,
                "k4": 0,
                "p1": 0,
                "p2": 0,
                "cx": 259.0,
                "cy": 259.0,
                "camera_angle_x": 0.792433396585736,
                "camera_angle_y": 0.7952903079932466,
                "fovx": 45.403089169579246,
                "fovy": 45.566778135672386,
                "file_path": "images/frame_train_00001.JPG",
                "depth_file_path": "depths/frame_train_00001.npy",
            },
        ]

    def test_to_dict(self):
        """
        Tests that `FrameInfo.to_dict` produces the correct frame dict
        """
        expected_keys = [
            "transform_matrix",
            "w",
            "h",
            "fl_x",
            "fl_y",
            "k1",
            "k2",
            "k3",
            "k4",
            "p1",
            "p2",
            "cx",
            "cy",
            "camera_angle_x",
            "camera_angle_y",
            "fovx",
            "fovy",
            "file_path",
            "depth_file_path",
        ]

        for i in range(len(self._extrinsics_world2cam)):
            result = FrameInfo(
                extrinsic_matrix_world2cam=self._extrinsics_world2cam[i],
                intrinsic_matrix=self._intrinsics[i],
                width=self._frame_dicts[i]["w"],
                height=self._frame_dicts[i]["h"],
                image_path=Path(self._frame_dicts[i]["file_path"]),
                depth_path=Path(self._frame_dicts[i]["depth_file_path"]),
            ).to_dict()

            self.assertEqual(set(result.keys()), set(expected_keys))

            for k in expected_keys:
                if k == "transform_matrix":
                    self.assertTrue(
                        torch.allclose(
                            torch.tensor(result["transform_matrix"]),
                            torch.tensor(self._frame_dicts[i]["transform_matrix"]),
                        )
                    )
                if k in {
                    "fl_x",
                    "fl_y",
                    "k1",
                    "k2",
                    "k3",
                    "k4",
                    "p1",
                    "p2",
                    "cx",
                    "cy",
                    "camera_angle_x",
                    "camera_angle_y",
                    "fovx",
                    "fovy",
                }:
                    self.assertTrue(np.allclose(result[k], self._frame_dicts[i][k]))
                else:
                    self.assertEqual(result[k], self._frame_dicts[i][k])

    def test_dict_to_matrices(self):
        """
        Tests that `dict_to_matrices` produces the correct matrices
        """

        for i in range(len(self._frame_dicts)):
            extrinsics_world2cam, intrinsics = FrameInfo.dict_to_matrices(
                frame_dict=self._frame_dicts[i]
            )
            self.assertTrue(
                torch.allclose(self._extrinsics_world2cam[i], extrinsics_world2cam)
            )
            self.assertTrue(torch.allclose(self._intrinsics[i], intrinsics))
