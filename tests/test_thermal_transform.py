import sys
import unittest
from pathlib import Path

import torch

from sear.augment.thermal import ThermalTransformFactory

# Remove it when https://github.com/facebookresearch/vggt/issues/416 is fixed
sys.path.append("vggt")

from sear.data_processing.single_dataset import VGGTSingleDataset


class TestThermalTransformFactory(unittest.TestCase):
    """
    Tests that ThermalTransformFactory class works properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        torch.manual_seed(0)
        local_dir = Path(__file__).parent.resolve()
        cls.dataset_1 = VGGTSingleDataset(
            scene_path=local_dir / "data/buildingA_winter"
        )
        cls.dataset_2 = VGGTSingleDataset(scene_path=local_dir / "data/Dimsum")
        cls.full_augmentation = ThermalTransformFactory(
            p_gaussian_noise=1.0,
            p_gaussian_blur=1.0,
            p_sharpness=1.0,
            p_random_linear=1.0,
            p_random_power=1.0,
        ).create_transform()

    def test_incorrect_shape(self) -> None:
        """Tests that the class raises an error if called with incorrect shape."""
        for _ in range(10):
            with self.assertRaises(ValueError):
                self.full_augmentation.apply(torch.rand((1, 1, 12, 34)))
            with self.assertRaises(ValueError):
                self.full_augmentation.apply(torch.rand((45, 22)))
            with self.assertRaises(ValueError):
                self.full_augmentation.apply(torch.rand((1, 1, 3, 45, 22)))

    def test_produce_correct_output_one_image(self) -> None:
        """
        Tests that augmentations do not alter object shape if apply augmentations. Also
        checks that output image values lie in the interval of [0.0, 1.0]
        """

        for _ in range(10):
            for dataset in [self.dataset_1, self.dataset_2]:
                for i in range(len(dataset)):
                    thermal_image = dataset[i].thermal
                    processed_image = self.full_augmentation.apply(thermal_image)
                    self.assertEqual(thermal_image.shape, processed_image.shape)
                    self.assertTrue(processed_image.max() <= 1.0)
                    self.assertTrue(processed_image.min() >= 0.0)

    def test_produce_correct_output_multiple_images(self) -> None:
        """
        Tests that augmentations do not alter object shape if apply augmentations on
        multiple images. Also checks that output image values lie in the interval of
        [0.0, 1.0]
        """

        for _ in range(10):
            for dataset in [self.dataset_1, self.dataset_2]:
                thermal_images = torch.stack(
                    [dataset[i].thermal for i in range(len(dataset))]
                )

                processed_images = self.full_augmentation.apply(thermal_images)
                self.assertEqual(thermal_images.shape, processed_images.shape)
                self.assertTrue(processed_images.max() <= 1.0)
                self.assertTrue(processed_images.min() >= 0.0)

    def test_random_linear(self) -> None:
        """Tests that random linear works properly"""
        image = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.5, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        image = image.unsqueeze(0).repeat(3, 1, 1)
        for _ in range(10):
            image_processed, _, _, lower, upper = (
                ThermalTransformFactory._random_linear(
                    x=image,
                    smallest_boundary=0.05,
                    largest_boundary=0.95,
                    minimal_difference=0.5,
                    debug=True,
                )
            )
            self.assertTrue(torch.allclose(image_processed.min(), torch.tensor(lower)))
            self.assertTrue(torch.allclose(image_processed.max(), torch.tensor(upper)))
