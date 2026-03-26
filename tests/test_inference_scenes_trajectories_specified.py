import sys
import unittest
from pathlib import Path

from lightning import seed_everything

# Remove it when https://github.com/facebookresearch/vggt/issues/416 is fixed
sys.path.append("vggt")
sys.path.append("vggt/training")

from sear.data_processing.inference_two_trajectories import (
    InferenceSceneTwoTrajectories,
)


class TestMultipleDataset(unittest.TestCase):
    """
    Tests that InferenceScenes when the trajectories (rgb and thermal) are specified in
    the `transforms.json` header.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Sets necessary variables for testing"""
        seed_everything(0)
        local_dir = Path(__file__).parent.resolve()

        cls.inference_scene_1 = InferenceSceneTwoTrajectories.from_scene_path(
            scene_path=local_dir / "data_two_trajectories/Dimsum"
        )
        cls.inference_scene_2 = InferenceSceneTwoTrajectories.from_scene_path_rgb_only(
            scene_path=local_dir / "data_two_trajectories/Dimsum"
        )

    def test_inference_scene_two_trajectories(self) -> None:
        """Tests that InferenceSceneTwoTrajectories works properly"""
        self.assertEqual(self.inference_scene_1.images.ndim, 5)  # (1, S, 3, H, W)
        self.assertEqual(
            self.inference_scene_1.images.shape[0:3], (1, 4, 3)
        )  # (**1, S, 3**, H, W)
        self.assertEqual(self.inference_scene_1.mask_thermal.ndim, 2)  # (1, S)
        self.assertEqual(
            self.inference_scene_1.mask_thermal.tolist(), [[False, False, False, True]]
        )
        self.assertEqual(self.inference_scene_1.scene_name, "Dimsum")

    def test_inference_scene_rgb_only(self) -> None:
        """Tests that InferenceSceneRGBOnly works properly"""
        self.assertEqual(self.inference_scene_2.images.ndim, 5)  # (1, S, 3, H, W)
        self.assertEqual(
            self.inference_scene_2.images.shape[0:3], (1, 4, 3)
        )  # (**1, S, 3**, H, W)
        self.assertEqual(self.inference_scene_2.mask_thermal.ndim, 2)  # (1, S)
        self.assertEqual(
            self.inference_scene_2.mask_thermal.tolist(), [[False, False, False, False]]
        )
        self.assertEqual(self.inference_scene_2.scene_name, "Dimsum")
