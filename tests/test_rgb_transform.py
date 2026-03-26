import sys
import unittest
from pathlib import Path

import torch

from sear.augment.rgb import RGBTransformFactory

# Remove it when https://github.com/facebookresearch/vggt/issues/416 is fixed
sys.path.append("vggt")

from sear.data_processing.single_dataset import VGGTSingleDataset


class TestRGBTransformFactory(unittest.TestCase):
    """
    Tests that RGBTransformFactory class works properly
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
        cls.full_augmentation = RGBTransformFactory(
            p_color_jitter=1.0,
            p_gaussian_noise=1.0,
            p_gaussian_blur=1.0,
            p_random_grayscale=1.0,
            p_sharpness=1.0,
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
        Tests that augmentations do not alter object shape if apply augmentations on one
        image. Also checks that output image values lie in the interval of [0.0, 1.0]
        """

        for _ in range(10):
            for dataset in [self.dataset_1, self.dataset_2]:
                for i in range(len(dataset)):
                    image = dataset[i].image
                    processed_image = self.full_augmentation.apply(image)
                    self.assertEqual(image.shape, processed_image.shape)
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
                images = torch.stack([dataset[i].image for i in range(len(dataset))])

                processed_images = self.full_augmentation.apply(images)
                self.assertEqual(images.shape, processed_images.shape)
                self.assertTrue(processed_images.max() <= 1.0)
                self.assertTrue(processed_images.min() >= 0.0)
