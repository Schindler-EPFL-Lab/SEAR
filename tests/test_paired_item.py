import unittest
from pathlib import Path

import torch

from sear.data_processing.paired_item import PairedItem as PairedItem


class TestPairedItem(unittest.TestCase):
    """
    Tests that paired item class works properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up the necessary variables for the tests."""

        cls.item1 = PairedItem(
            images=torch.rand(4, 2, 3, 123, 321),
            images_paths=[
                [Path("path/to/image1_rgb.png"), Path("path/to/image1_thermal.png")]
            ]
            * 4,
            extrinsics_world2cam=torch.rand(4, 2, 4, 4),
            intrinsics=torch.rand(4, 2, 3, 3),
            mask_thermal=torch.zeros(4, 2),
            scenes_names=["scene1"] * 4,
            datasets_names=["dataset1"] * 4,
        )

        cls.item2 = PairedItem(
            images=torch.rand(16, 2, 3, 518, 518),
            images_paths=[
                [Path("path/to/image2_rgb.png"), Path("path/to/image2_thermal.png")]
            ]
            * 16,
            extrinsics_world2cam=torch.rand(16, 2, 3, 4),
            intrinsics=torch.rand(16, 2, 3, 3),
            mask_thermal=torch.zeros(16, 2),
            scenes_names=["scene2"] * 16,
            datasets_names=["dataset2"] * 16,
        )

    def test_iterate_batched(self) -> None:
        """Tests that the datasets length are correct."""

        items1 = [item for item in self.item1.iterate_batched(batch_size=3)]
        expected_lengths = [3, 1]
        self.assertEqual([item.images.shape[0] for item in items1], expected_lengths)

        items2 = [item for item in self.item2.iterate_batched(batch_size=5)]
        expected_lengths = [5, 5, 5, 1]
        self.assertEqual([item.images.shape[0] for item in items2], expected_lengths)
