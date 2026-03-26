import json
import sys
import unittest
from pathlib import Path

import torch
from lightning import seed_everything

# Remove it when https://github.com/facebookresearch/vggt/issues/416 is fixed
sys.path.append("vggt")
sys.path.append("vggt/training")

from sear.data_processing.align_part import AlignPart
from sear.data_processing.inference_scene import InferenceScene as InferenceScene
from sear.data_processing.inference_scene_depth_and_poses import (
    InferenceSceneDepthAndPoses as InferenceSceneDepthAndPoses,
)


class TestMultipleDataset(unittest.TestCase):
    """
    Tests that InferenceScenes when each frame in transforms.json contains image of only
    one modality (rgb or thermal), which is different from the ordinary ThermoScenes
    format.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Sets necessary variables for testing"""
        seed_everything(0)
        cls.local_dir = Path(__file__).parent.resolve()

        cls.inference_scene_1 = InferenceScene.from_scene_path(
            scene_path=cls.local_dir / "data_inference_scene/Dimsum"
        )
        cls.inference_scene_2 = InferenceSceneDepthAndPoses.from_scene_path(
            scene_path=cls.local_dir / "data_inference_scene/Dimsum"
        )
        cls.inference_scene_3 = InferenceSceneDepthAndPoses.from_scene_path_aligned(
            scene_path=cls.local_dir / "data_inference_scene/Dimsum",
            align=AlignPart.ALL,
        )

    def test_inference_scene(self) -> None:
        """Tests that InferenceScene works properly"""
        self.assertEqual(self.inference_scene_1.images.ndim, 5)  # (1, S, 3, H, W)
        self.assertEqual(
            self.inference_scene_1.images.shape[0:3], (1, 4, 3)
        )  # (**1, S, 3**, H, W)
        self.assertEqual(self.inference_scene_1.mask_thermal.ndim, 2)  # (1, S)
        self.assertEqual(
            self.inference_scene_1.mask_thermal.tolist(), [[False, True, False, True]]
        )
        self.assertEqual(self.inference_scene_1.scene_name, "Dimsum")

    def test_inference_scene_depth_and_poses(self) -> None:
        """Tests that InferenceSceneDepthAndPoses works properly"""
        self.assertEqual(self.inference_scene_2.images.ndim, 5)  # (1, S, 3, H, W)
        self.assertEqual(
            self.inference_scene_2.images.shape[0:3], (1, 4, 3)
        )  # (**1, S, 3**, H, W)
        self.assertEqual(self.inference_scene_2.mask_thermal.ndim, 2)  # (1, S)
        self.assertEqual(
            self.inference_scene_2.mask_thermal.tolist(), [[False, True, False, True]]
        )
        self.assertEqual(self.inference_scene_2.scene_name, "Dimsum")

        self.assertEqual(self.inference_scene_2.depths.ndim, 4)  # (1, S, H, W)
        self.assertEqual(
            self.inference_scene_2.depths.shape[0:2], (1, 4)
        )  # (**1, S**, H, W)
        self.assertEqual(
            self.inference_scene_2.extrinsics_world2cam.ndim, 4
        )  # (1, S, 4, 4)
        self.assertTrue(
            self.inference_scene_2.extrinsics_world2cam.shape
            in [(1, 4, 3, 4), (1, 4, 4, 4)]
        )  # (1, S, 4, 4)
        self.assertEqual(self.inference_scene_2.intrinsics.ndim, 4)  # (1, S, 3, 3)
        self.assertEqual(
            self.inference_scene_2.intrinsics.shape, (1, 4, 3, 3)
        )  # (1, S, 4, 4)

    def test_inference_scene_depth_and_poses_aligned(self) -> None:
        """Tests that InferenceSceneDepthAndPoses works properly"""
        self.assertEqual(self.inference_scene_3.images.ndim, 5)  # (1, S, 3, H, W)
        self.assertEqual(
            self.inference_scene_3.images.shape[0:3], (1, 4, 3)
        )  # (**1, S, 3**, H, W)
        self.assertEqual(self.inference_scene_3.mask_thermal.ndim, 2)  # (1, S)
        self.assertEqual(
            self.inference_scene_3.mask_thermal.tolist(), [[False, True, False, True]]
        )
        self.assertEqual(self.inference_scene_3.scene_name, "Dimsum")

        self.assertEqual(self.inference_scene_3.depths.ndim, 4)  # (1, S, H, W)
        self.assertEqual(
            self.inference_scene_3.depths.shape[0:2], (1, 4)
        )  # (**1, S**, H, W)
        self.assertEqual(
            self.inference_scene_3.extrinsics_world2cam.ndim, 4
        )  # (1, S, 4, 4)
        self.assertTrue(
            self.inference_scene_3.extrinsics_world2cam.shape
            in [(1, 4, 3, 4), (1, 4, 4, 4)]
        )  # (1, S, 4, 4)
        self.assertEqual(self.inference_scene_3.intrinsics.ndim, 4)  # (1, S, 3, 3)
        self.assertEqual(
            self.inference_scene_3.intrinsics.shape, (1, 4, 3, 3)
        )  # (1, S, 4, 4)

        with open(
            self.local_dir / "data_inference_scene/Dimsum/transforms_ground_truth.json"
        ) as f:
            transforms_real = json.load(f)
        extrinsics_world2cam = [
            list(frame.values())[0]["transform_matrix"]
            for frame in transforms_real["frames"]
        ]
        extrinsics_world2cam = torch.tensor(extrinsics_world2cam, dtype=torch.float32)
        self.assertTrue(
            torch.allclose(
                extrinsics_world2cam,
                self.inference_scene_3.extrinsics_world2cam[0, :, :3, :4],
                atol=1e-5,
            )
        )
