import sys
import unittest
from pathlib import Path

# Remove it when https://github.com/facebookresearch/vggt/issues/416 is fixed
sys.path.append("vggt")

from sear.data_processing.single_dataset import VGGTSingleDataset


class TestSingleDataset(unittest.TestCase):
    """
    Tests that single dataset class works properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        local_dir = Path(__file__).parent.resolve()
        cls.dataset_path = local_dir / "data/buildingA_winter"
        cls.dataset = VGGTSingleDataset(scene_path=cls.dataset_path)

    def test_length(self):
        """
        Tests that the length of the dataset is correct.
        """
        self.assertEqual(len(self.dataset), 3)

    def test_getitem_keys(self):
        """
        Tests that the tuple returned by getitem has proper shape
        """

        for i in range(len(self.dataset)):
            self.assertEqual(len(self.dataset[i].to_tuple()), 12)

    def test_getitem_values(self):
        """
        Tests that the shapes of the dataset returned tensors are correct.
        """
        expected_shapes = {
            "image": (3, 518, 518),
            "depth_rgb": (518, 518),
            "extrinsic_world2cam_rgb": (3, 4),
            "intrinsic_rgb": (3, 3),
            "thermal": (3, 518, 518),
            "depth_thermal": (518, 518),
            "extrinsic_world2cam_thermal": (3, 4),
            "intrinsic_thermal": (3, 3),
        }

        for i in range(len(self.dataset)):
            dataset_element = self.dataset[i]
            self.assertEqual(dataset_element.image.shape, expected_shapes["image"])
            self.assertEqual(
                dataset_element.depth_rgb.shape, expected_shapes["depth_rgb"]
            )
            self.assertEqual(
                dataset_element.extrinsic_world2cam_rgb.shape,
                expected_shapes["extrinsic_world2cam_rgb"],
            )
            self.assertEqual(
                dataset_element.intrinsic_rgb.shape, expected_shapes["intrinsic_rgb"]
            )
            self.assertEqual(dataset_element.thermal.shape, expected_shapes["thermal"])
            self.assertEqual(
                dataset_element.depth_thermal.shape, expected_shapes["depth_thermal"]
            )
            self.assertEqual(
                dataset_element.extrinsic_world2cam_thermal.shape,
                expected_shapes["extrinsic_world2cam_thermal"],
            )
            self.assertEqual(
                dataset_element.intrinsic_thermal.shape,
                expected_shapes["intrinsic_thermal"],
            )

            self.assertTrue(dataset_element.image.min() >= 0)
            self.assertTrue(dataset_element.image.max() <= 1)
            self.assertTrue(dataset_element.depth_rgb.min() >= 0)
            self.assertTrue(dataset_element.thermal.max() <= 1)
            self.assertTrue(dataset_element.thermal.min() >= 0)
            self.assertTrue(dataset_element.depth_thermal.min() >= 0)
