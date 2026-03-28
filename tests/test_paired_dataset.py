import unittest
from pathlib import Path

import torch

from sear.data_processing.paired_dataset import PairedDataset as PairedDataset
from sear.data_processing.paired_item import PairedItem as PairedItem


class TestPairedDataset(unittest.TestCase):
    """
    Tests that paired dataset class works properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up the necessary variables for the tests."""
        local_dir = Path(__file__).parent.resolve()
        cls.dataset_path = local_dir / "data_vistir/"
        cls.dataset1 = PairedDataset(
            dataset_path=cls.dataset_path,
            max_size=1,
        )
        cls.dataset2 = PairedDataset(
            dataset_path=cls.dataset_path,
            max_size=2,
        )

    def test_dataset_length(self) -> None:
        """Tests that the datasets length are correct."""

        self.assertEqual(len(self.dataset1), 4)
        self.assertEqual(len(self.dataset2), 2)

    def test_getitem(self) -> None:
        """Tests that the `getitem` method works properly."""

        item1 = self.dataset1[3]
        item2 = self.dataset2[1]

        self.assertIsInstance(item1, PairedItem)
        self.assertIsInstance(item2, PairedItem)

    def test_paired_items_shapes(self):
        """Tests that the paired items have correct shapes."""

        for dataset in [self.dataset1, self.dataset2]:
            for i in range(len(dataset)):
                item = dataset[i]
                # (N, 2, 3, H, W)
                self.assertEqual(item.images.ndim, 5)
                self.assertEqual(item.images.shape[1], 2)
                self.assertEqual(item.images.shape[2], 3)

                # (N, 2, 3, 3)
                self.assertEqual(item.intrinsics.ndim, 4)
                self.assertEqual(item.intrinsics.shape[1], 2)
                self.assertEqual(item.intrinsics.shape[2:], (3, 3))

                # (N, 2, 4, 4) or (N, 2, 3, 4)
                self.assertEqual(item.extrinsics_world2cam.ndim, 4)
                self.assertEqual(item.extrinsics_world2cam.shape[1], 2)
                self.assertIn(item.extrinsics_world2cam.shape[2:], [(3, 4), (4, 4)])

                # (N, 2)
                for el in item.images_paths:
                    self.assertEqual(len(el), 2)

    def test_combine_rgb_thermal(self) -> None:
        """Tests that the `_combine_rgb_thermal_images` method works properly."""

        for dataset in [self.dataset1, self.dataset2]:
            images_rgb = torch.rand(4, 3, 123, 321)
            images_thermal = torch.rand(4, 3, 200, 150)
            intrinsics = torch.zeros(4, 2, 3, 3)
            intrinsics[:, 0, 0, 0] = 123  # rgb fx
            intrinsics[:, 0, 0, 2] = 60.5  # rgb cx
            intrinsics[:, 0, 1, 1] = 222  # rgb fy
            intrinsics[:, 0, 1, 2] = 121.5  # rgb cy

            intrinsics[:, 1, 0, 0] = 100  # thermal fx
            intrinsics[:, 1, 0, 2] = 100  # thermal cx
            intrinsics[:, 1, 1, 1] = 75  # thermal fy
            intrinsics[:, 1, 1, 2] = 75  # thermal cy

            intrinsics[:, :, 2, 2] = 1

            desired_aspect_ratio = 0.5

            combined_images, combined_intrinsics = dataset._combine_rgb_thermal_images(
                images_rgb=images_rgb,
                images_thermal=images_thermal,
                intrinsics=intrinsics,
                desired_aspect_ratio=desired_aspect_ratio,
            )

            # (N, 2, 3, H, W)
            self.assertEqual(combined_images.ndim, 5)
            self.assertEqual(combined_images.shape[1], 2)
            self.assertEqual(combined_images.shape[2], 3)

            # (N, 2, 3, 3)
            self.assertEqual(combined_intrinsics.ndim, 4)
            self.assertEqual(combined_intrinsics.shape[1], 2)
            self.assertEqual(combined_intrinsics.shape[2:], (3, 3))
