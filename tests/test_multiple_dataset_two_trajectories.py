import sys
import unittest
from pathlib import Path

# Remove it when https://github.com/facebookresearch/vggt/issues/416 is fixed
sys.path.append("vggt")
sys.path.append("vggt/training")

from sear.data_processing.multiple_dataset_two_trajectories import (
    MultipleDatasetTwoTrajectories,
)


class TestMultipleDatasetTwoTrajectories(unittest.TestCase):
    """
    Tests that MultipleDatasetTwoTrajectories class works properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        local_dir = Path(__file__).parent.resolve()
        cls.dataset_path = local_dir / "data_two_trajectories/"
        cls.scenes_per_dataset_path = (
            local_dir / ".." / "sear/configs/scenes_per_dataset.json"
        )

        cls.dataset = MultipleDatasetTwoTrajectories(
            root_path=cls.dataset_path,
            scenes_names=["Dimsum"],
            scenes_per_dataset_path=cls.scenes_per_dataset_path,
        )

    def test_length(self) -> None:
        """Tests that len works properly"""
        self.assertEqual(len(self.dataset), 1)

    def test_getitem_raises(self) -> None:
        """Tests that __getitem__ raises error if necessary"""
        with self.assertRaises(RuntimeError):
            self.dataset[1]
        with self.assertRaises(RuntimeError):
            self.dataset[13]

    def test_getitem(self) -> None:
        """Tests that __getitem__ works properly"""
        for _ in range(10):
            item = self.dataset[0]
            self.assertEqual(item.images.shape[:3], (1, 4, 3))
            self.assertEqual(item.depths.shape[:2], (1, 4))
            self.assertTrue(
                item.extrinsics_world2cam.shape in [(1, 4, 3, 4), (1, 4, 4, 4)]
            )
            self.assertEqual(item.intrinsics.shape, (1, 4, 3, 3))
            self.assertEqual(item.point_masks.shape[:2], (1, 4))
            self.assertEqual(item.mask_thermal.shape, (1, 4))
            self.assertEqual(item.mask_thermal.tolist()[0], [False, False, False, True])

    def test_get_chunk_modality_shape_specified(self) -> None:
        """
        Tests that `get_chunk_modality_shape_specified` works properly for
        MultipleDatasetTwoTrajectories
        """
        for _ in range(10):
            item = self.dataset[0]
            item2 = self.dataset.get_chunk_modality_shape_specified(
                index=0, mask_thermal=~item.mask_thermal, sequence_length=4
            )
            self.assertEqual(item2.images.shape[:3], (1, 4, 3))
            self.assertEqual(item2.depths.shape[:2], (1, 4))
            self.assertTrue(
                item2.extrinsics_world2cam.shape in [(1, 4, 3, 4), (1, 4, 4, 4)]
            )
            self.assertEqual(item2.intrinsics.shape, (1, 4, 3, 3))
            self.assertEqual(item2.point_masks.shape[:2], (1, 4))
            self.assertEqual(item2.mask_thermal.shape, (1, 4))
            self.assertEqual(item2.mask_thermal.tolist()[0], [True, True, True, False])
