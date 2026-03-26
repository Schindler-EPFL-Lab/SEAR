import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from nerfstudio.data.utils.colmap_parsing_utils import rotmat2qvec

from sear.scripts.eval.read_colmap import (
    colmap_to_json,
    find_best_reconstruction,
    read_cameras,
    read_images,
)


class TestReadColmap(unittest.TestCase):
    """
    Tests that a function to project points works properly.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        local_dir = Path(__file__).parent.resolve()
        cls.dataset_path = local_dir / "data_colmap/"
        cls.best_reconstruction = cls.dataset_path / "reconstruction_2"
        cls.expected_colmap_intrinsics = {
            1: torch.tensor(
                [
                    [3164.7732394346681, 0.0, 259.0],
                    [0.0, 3164.7732394346681, 259.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        }
        cls.frames_names = [
            "frame_00051.png",
            "frame_00048.png",
            "frame_00046.png",
            "frame_00035.png",
            "frame_00038.png",
            "frame_00034.png",
            "frame_00033.png",
            "frame_00052.png",
            "frame_00036.png",
            "frame_00057.png",
            "frame_00008.png",
            "frame_00055.png",
            "frame_00053.png",
            "frame_00037.png",
            "frame_00060.png",
            "frame_00047.png",
            "frame_00059.png",
            "frame_00018.png",
        ]

        cls.expected_colmap_width_height = {1: (518, 518)}

    def test_find_best_reconstruction(self) -> None:
        """Tests that `find_best_reconstruction` works properly"""
        best_reconstruction = find_best_reconstruction(self.dataset_path)
        self.assertEqual(best_reconstruction.name, self.best_reconstruction.name)

    def test_read_cameras(self) -> None:
        """Tests that `read_cameras` works properly"""
        colmap_intrinsics, colmap_width_height = read_cameras(
            self.best_reconstruction / "cameras.txt"
        )
        self.assertTrue(
            self.expected_colmap_intrinsics.keys(), colmap_intrinsics.keys()
        )
        self.assertTrue(
            self.expected_colmap_width_height.keys(), colmap_width_height.keys()
        )
        for camera_id in colmap_intrinsics:
            self.assertTrue(
                torch.allclose(
                    self.expected_colmap_intrinsics[camera_id],
                    colmap_intrinsics[camera_id],
                )
            )
            self.assertEqual(
                self.expected_colmap_width_height[camera_id],
                colmap_width_height[camera_id],
            )

    def test_read_images(self) -> None:
        """Tests that `read_images` works properly"""
        frames_names, extrinsics_world2cam, intrinsics, resolution = read_images(
            images_path=self.best_reconstruction / "images.txt",
            colmap_intrinsics=self.expected_colmap_intrinsics,
            colmap_width_height=self.expected_colmap_width_height,
        )

        self.assertTrue(
            set(frames_names),
            set(self.frames_names),
        )

        self.assertTrue(
            torch.allclose(
                intrinsics,
                self.expected_colmap_intrinsics[1].unsqueeze(0).repeat(18, 1, 1),
            )
        )

        self.assertTrue(
            torch.allclose(
                resolution,
                torch.tensor(self.expected_colmap_width_height[1])
                .unsqueeze(0)
                .repeat(18, 1),
            )
        )

        with open(self.best_reconstruction / "images.txt") as f:
            lines = f.read().splitlines()

        # filter only frames without #
        lines = [line for line in lines if len(line) > 0 and line[0] != "#"]
        lines = lines[::2]

        frames_names: list[str] = []
        for ext_idx, line in enumerate(lines):
            _, qw, qx, qy, qz, tx, ty, tz, _, _ = line.strip().split(" ")
            quat = np.array([float(qw), float(qx), float(qy), float(qz)])
            translation = np.array([float(tx), float(ty), float(tz)])

            self.assertTrue(
                np.allclose(translation, extrinsics_world2cam[ext_idx, :3, 3].numpy())
            )
            self.assertTrue(
                np.allclose(
                    quat, rotmat2qvec(extrinsics_world2cam[ext_idx, :3, :3].numpy())
                )
            )

    def test_colmap_to_json(self) -> None:
        """Tests that `colmap_to_json` works properly"""
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            colmap_to_json(
                recon_dir=self.best_reconstruction,
                output_dir=temp_dir,
                image_rename_map={name: name for name in self.frames_names},
            )

            self.assertTrue((temp_dir / "transforms.json").exists())
            self.assertTrue((temp_dir / "sparse_pc.ply").exists())
