import sys
import unittest
from pathlib import Path

import torch

# Remove it when https://github.com/facebookresearch/vggt/issues/416 is fixed
sys.path.append("vggt")
sys.path.append("vggt/training")

from sear.augment.geometric import GeometricTransform, GeometricTransformConfig
from sear.augment.rgb import RGBTransformFactory
from sear.augment.thermal import ThermalTransformFactory
from sear.data_processing.multiple_dataset import VGGTMultipleDataset
from sear.data_processing.single_dataset import Item


class TestMultipleDataset(unittest.TestCase):
    """
    Tests that single dataset class works properly
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Sets necessary variables for testing
        """
        local_dir = Path(__file__).parent.resolve()
        cls.dataset_path = local_dir / "data/"
        cls.scenes_per_dataset_path = (
            local_dir / "../sear/configs/scenes_per_dataset.json"
        )
        cls.dataset_1 = VGGTMultipleDataset(
            root_path=cls.dataset_path,
            scenes_names=["buildingA_winter", "Dimsum"],
            min_sequence_length=2,
            elements_number=2,
            drop_last=True,
            shuffle=False,
            rgb_transform=RGBTransformFactory.get_empty(),
            thermal_transform=ThermalTransformFactory.get_empty(),
            geometric_transform=GeometricTransform.empty(),
            scenes_per_dataset_path=cls.scenes_per_dataset_path,
        )
        cls.dataset_2 = VGGTMultipleDataset(
            root_path=cls.dataset_path,
            scenes_names=["buildingA_winter", "Dimsum"],
            min_sequence_length=2,
            elements_number=2,
            drop_last=True,
            shuffle=True,
            rgb_transform=RGBTransformFactory.get_empty(),
            thermal_transform=ThermalTransformFactory.get_empty(),
            geometric_transform=GeometricTransform.empty(),
            scenes_per_dataset_path=cls.scenes_per_dataset_path,
        )

        cls.dataset_3 = VGGTMultipleDataset(
            root_path=cls.dataset_path,
            scenes_names=["buildingA_winter", "Dimsum"],
            min_sequence_length=2,
            elements_number=2,
            drop_last=False,
            shuffle=False,
            rgb_transform=RGBTransformFactory.get_empty(),
            thermal_transform=ThermalTransformFactory.get_empty(),
            geometric_transform=GeometricTransform.empty(),
            scenes_per_dataset_path=cls.scenes_per_dataset_path,
        )

        cls.dataset_4 = VGGTMultipleDataset(
            root_path=cls.dataset_path,
            scenes_names=["buildingA_winter", "Dimsum"],
            min_sequence_length=2,
            elements_number=2,
            drop_last=False,
            shuffle=True,
            rgb_transform=RGBTransformFactory.get_empty(),
            thermal_transform=ThermalTransformFactory.get_empty(),
            geometric_transform=GeometricTransform.empty(),
            scenes_per_dataset_path=cls.scenes_per_dataset_path,
            scale_poses=False,
        )

        cls.dataset_5 = VGGTMultipleDataset(
            root_path=cls.dataset_path,
            scenes_names=["Dimsum"],
            min_sequence_length=2,
            elements_number=3,
            shuffle=True,
            rgb_transform=RGBTransformFactory.get_empty(),
            thermal_transform=ThermalTransformFactory.get_empty(),
            geometric_transform=GeometricTransform.empty(),
            scenes_per_dataset_path=cls.scenes_per_dataset_path,
        )

        cls.dataset_6 = VGGTMultipleDataset(
            root_path=cls.dataset_path,
            scenes_names=["buildingA_winter", "Dimsum"],
            min_sequence_length=2,
            elements_number=2,
            drop_last=True,
            shuffle=True,
            rgb_transform=RGBTransformFactory().create_transform(),
            thermal_transform=ThermalTransformFactory().create_transform(),
            geometric_transform=GeometricTransform(),
            scenes_per_dataset_path=cls.scenes_per_dataset_path,
        )

        cls.dataset_7 = VGGTMultipleDataset(
            root_path=cls.dataset_path,
            scenes_names=["Dimsum"],
            min_sequence_length=2,
            elements_number=4,
            drop_last=True,
            shuffle=True,
            rgb_transform=RGBTransformFactory().create_transform(),
            thermal_transform=ThermalTransformFactory().create_transform(),
            geometric_transform=GeometricTransform(),
            scenes_per_dataset_path=cls.scenes_per_dataset_path,
            scale_poses=False,
        )

        cls.chunk = {
            "images": [
                torch.full((3, 4, 4), fill_value=0.5),
                torch.full((3, 4, 4), fill_value=0.7),
            ],
            "paths": [
                cls.dataset_path / "Dimsum/images/frame_eval_00002.jpg",
                cls.dataset_path / "Dimsum/images/frame_eval_00002.jpg",
            ],
            "thermals": [
                torch.full((3, 4, 4), fill_value=0.8),
                torch.full((3, 4, 4), fill_value=0.1),
            ],
            "extrinsic_matrices": [
                torch.tensor(
                    [
                        [0.0, 0.0, 1.0, -5.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                    ]
                ),
                torch.tensor(
                    [
                        [0.0, 0.0, -1.0, 5.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                    ]
                ),
            ],
            "intrinsic_matrices": [
                torch.tensor(
                    [
                        [1.0, 0.0, 2.0],
                        [0.0, 1.0, 2.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                torch.tensor(
                    [
                        [1.0, 0.0, 2.0],
                        [0.0, 1.0, 2.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
            ],
            "depths": [
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 2.0, 2.0, 1.0],
                        [1.0, 2.0, 2.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [1.0, 2.0, 2.0, 1.0],
                        [1.0, 2.0, 2.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                ),
            ],
        }

        cls.expected_point_masks = torch.ones((2, 4, 4), dtype=torch.bool)
        cls.expected_intrinsics = torch.tensor(
            [
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
            ]
        )[None].repeat(2, 1, 1)
        cls.expected_extrinsics = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            ]
        )

        cls.expected_depths = torch.tensor(
            [
                [
                    [0.43938553, 0.43938553, 0.43938553, 0.43938553],
                    [0.43938553, 0.87877107, 0.87877107, 0.43938553],
                    [0.43938553, 0.87877107, 0.87877107, 0.43938553],
                    [0.43938553, 0.43938553, 0.43938553, 0.43938553],
                ],
                [
                    [0.43938553, 0.43938553, 0.43938553, 0.43938553],
                    [0.43938553, 0.87877107, 0.87877107, 0.43938553],
                    [0.43938553, 0.87877107, 0.87877107, 0.43938553],
                    [0.43938553, 0.43938553, 0.43938553, 0.43938553],
                ],
            ]
        )

        cls.expected_thermal_mask = torch.tensor([False, False, True, True])

    def test_lengths(self) -> None:
        """
        Tests that the length of the dataset is correct.
        """
        self.assertEqual(len(self.dataset_1), 3)
        self.assertEqual(len(self.dataset_2), 3)
        self.assertEqual(len(self.dataset_3), 4)
        self.assertEqual(len(self.dataset_4), 4)
        self.assertEqual(len(self.dataset_5), 1)
        self.assertEqual(len(self.dataset_6), 3)
        self.assertEqual(len(self.dataset_7), 1)

    def test_getitem_access(self) -> None:
        """
        Tests that the tuple returned by a single getitem has the proper number of
        elements.
        """
        self.assertEqual(len(self.dataset_1[2].to_tuple()), 9)
        self.assertEqual(len(self.dataset_2[1].to_tuple()), 9)
        self.assertEqual(len(self.dataset_3[3].to_tuple()), 9)
        self.assertEqual(len(self.dataset_4[1].to_tuple()), 9)
        self.assertEqual(len(self.dataset_5[0].to_tuple()), 9)
        self.assertEqual(len(self.dataset_6[2].to_tuple()), 9)
        self.assertEqual(len(self.dataset_7[0].to_tuple()), 9)

    def test_getitem_keys(self) -> None:
        """
        Tests that the tuple returned by getitem contains the proper number of elements.
        """

        for dataset in [
            self.dataset_1,
            self.dataset_2,
            self.dataset_3,
            self.dataset_4,
            self.dataset_5,
            self.dataset_6,
            self.dataset_7,
        ]:
            for i in range(len(dataset)):
                self.assertEqual(len(dataset[i].to_tuple()), 9)

    def test_getitem_shapes(self) -> None:
        """
        Tests that the shapes of the dataset returned tensors are correct.
        """
        expected_shapes_126 = {
            "images": (1, 2, 3),
            "depths": (1, 2),
            "point_masks": (1, 2),
            "extrinsics_world2cam": (1, 2, 3, 4),
            "intrinsics": (1, 2, 3, 3),
            "thermal_mask": (1, 2),
            "datasets_names": 1,
            "scenes_names": 1,
        }

        expected_shapes_3 = [
            {
                "images": (1, 2, 3),
                "depths": (1, 2),
                "point_masks": (1, 2),
                "extrinsics_world2cam": (1, 2, 3, 4),
                "intrinsics": (1, 2, 3, 3),
                "thermal_mask": (1, 2),
                "datasets_names": 1,
                "scenes_names": 1,
            },
            {
                "images": (1, 1, 3),
                "depths": (1, 1),
                "point_masks": (1, 1),
                "extrinsics_world2cam": (1, 1, 3, 4),
                "intrinsics": (1, 1, 3, 3),
                "thermal_mask": (1, 1),
                "datasets_names": 1,
                "scenes_names": 1,
            },
            {
                "images": (1, 2, 3),
                "depths": (1, 2),
                "point_masks": (1, 2),
                "extrinsics_world2cam": (1, 2, 3, 4),
                "intrinsics": (1, 2, 3, 3),
                "thermal_mask": (1, 2),
                "datasets_names": 1,
                "scenes_names": 1,
            },
            {
                "images": (1, 2, 3),
                "depths": (1, 2),
                "point_masks": (1, 2),
                "extrinsics_world2cam": (1, 2, 3, 4),
                "intrinsics": (1, 2, 3, 3),
                "thermal_mask": (1, 2),
                "datasets_names": 1,
                "scenes_names": 1,
            },
        ]

        expected_shapes_5 = [
            {
                "images": (1, 3, 3),
                "depths": (1, 3),
                "point_masks": (1, 3),
                "extrinsics_world2cam": (1, 3, 3, 4),
                "intrinsics": (1, 3, 3, 3),
                "thermal_mask": (1, 3),
                "datasets_names": 1,
                "scenes_names": 1,
            }
        ]

        expected_shapes_7 = [
            {
                "images": [(1, 4, 3), (2, 2, 3)],
                "depths": [(1, 4), (2, 2)],
                "point_masks": [(1, 4), (2, 2)],
                "extrinsics_world2cam": [(1, 4, 3, 4), (2, 2, 3, 4)],
                "intrinsics": [(1, 4, 3, 3), (2, 2, 3, 3)],
                "thermal_mask": [(1, 4), (2, 2)],
                "datasets_names": [1, 2],
                "scenes_names": [1, 2],
            }
        ]

        for dataset_126 in [self.dataset_1, self.dataset_2, self.dataset_6]:
            for i in range(len(dataset_126)):
                dataset_element = dataset_126[i]
                self.assertEqual(
                    dataset_element.images.shape[:3],
                    expected_shapes_126["images"],
                )
                self.assertEqual(
                    dataset_element.depths.shape[:2],
                    expected_shapes_126["depths"],
                )
                self.assertEqual(
                    dataset_element.point_masks.shape[:2],
                    expected_shapes_126["point_masks"],
                )
                self.assertEqual(
                    dataset_element.extrinsics_world2cam.shape,
                    expected_shapes_126["extrinsics_world2cam"],
                )
                self.assertEqual(
                    dataset_element.intrinsics.shape,
                    expected_shapes_126["intrinsics"],
                )
                self.assertEqual(
                    dataset_element.mask_thermal.shape,
                    expected_shapes_126["thermal_mask"],
                )
                self.assertEqual(
                    len(dataset_element.datasets_names),
                    expected_shapes_126["datasets_names"],
                )
                self.assertEqual(
                    len(dataset_element.scenes_names),
                    expected_shapes_126["scenes_names"],
                )

        for i in range(len(self.dataset_3)):
            dataset_element = self.dataset_3[i]
            self.assertEqual(
                dataset_element.images.shape[:3],
                expected_shapes_3[i]["images"],
            )
            self.assertEqual(
                dataset_element.depths.shape[:2],
                expected_shapes_3[i]["depths"],
            )
            self.assertEqual(
                dataset_element.point_masks.shape[:2],
                expected_shapes_3[i]["point_masks"],
            )
            self.assertEqual(
                dataset_element.extrinsics_world2cam.shape,
                expected_shapes_3[i]["extrinsics_world2cam"],
            )
            self.assertEqual(
                dataset_element.intrinsics.shape,
                expected_shapes_3[i]["intrinsics"],
            )
            self.assertEqual(
                dataset_element.mask_thermal.shape,
                expected_shapes_3[i]["thermal_mask"],
            )
            self.assertEqual(
                len(dataset_element.datasets_names),
                expected_shapes_3[i]["datasets_names"],
            )
            self.assertEqual(
                len(dataset_element.scenes_names),
                expected_shapes_3[i]["scenes_names"],
            )

        for i in range(len(self.dataset_5)):
            dataset_element = self.dataset_5[i]
            self.assertEqual(
                dataset_element.images.shape[:3],
                expected_shapes_5[i]["images"],
            )
            self.assertEqual(
                dataset_element.depths.shape[:2],
                expected_shapes_5[i]["depths"],
            )
            self.assertEqual(
                dataset_element.point_masks.shape[:2],
                expected_shapes_5[i]["point_masks"],
            )
            self.assertEqual(
                dataset_element.extrinsics_world2cam.shape,
                expected_shapes_5[i]["extrinsics_world2cam"],
            )
            self.assertEqual(
                dataset_element.intrinsics.shape,
                expected_shapes_5[i]["intrinsics"],
            )
            self.assertEqual(
                dataset_element.mask_thermal.shape,
                expected_shapes_5[i]["thermal_mask"],
            )
            self.assertEqual(
                len(dataset_element.datasets_names),
                expected_shapes_5[i]["datasets_names"],
            )
            self.assertEqual(
                len(dataset_element.scenes_names),
                expected_shapes_5[i]["scenes_names"],
            )

        for _ in range(10):
            for i in range(len(self.dataset_7)):
                dataset_element = self.dataset_7[i]
                self.assertTrue(
                    dataset_element.images.shape[0] in [1, 2]
                )  # it can be either 1 or 2

                ground_truth_index = 0
                if dataset_element.images.shape[0] == 2:
                    ground_truth_index = 1

                self.assertEqual(
                    dataset_element.images.shape[:3],
                    expected_shapes_7[i]["images"][ground_truth_index],
                )
                self.assertEqual(
                    dataset_element.depths.shape[:2],
                    expected_shapes_7[i]["depths"][ground_truth_index],
                )
                self.assertEqual(
                    dataset_element.point_masks.shape[:2],
                    expected_shapes_7[i]["point_masks"][ground_truth_index],
                )
                self.assertEqual(
                    dataset_element.extrinsics_world2cam.shape,
                    expected_shapes_7[i]["extrinsics_world2cam"][ground_truth_index],
                )
                self.assertEqual(
                    dataset_element.intrinsics.shape,
                    expected_shapes_7[i]["intrinsics"][ground_truth_index],
                )
                self.assertEqual(
                    dataset_element.mask_thermal.shape,
                    expected_shapes_7[i]["thermal_mask"][ground_truth_index],
                )
                self.assertEqual(
                    len(dataset_element.datasets_names),
                    expected_shapes_7[i]["datasets_names"][ground_truth_index],
                )
                self.assertEqual(
                    len(dataset_element.scenes_names),
                    expected_shapes_7[i]["scenes_names"][ground_truth_index],
                )

    def test_scale_poses(self) -> None:
        """
        Test scale poses method on two images with no augmentations.
        """

        expected_result = (
            self.expected_depths,
            self.expected_point_masks,
            self.expected_extrinsics,
        )

        result = self.dataset_1._scale_poses(
            depths=torch.stack(self.chunk["depths"]).unsqueeze(0),
            extrinsic_matrices_world2cam=torch.stack(
                self.chunk["extrinsic_matrices"]
            ).unsqueeze(0),
            intrinsic_matrices=torch.stack(self.chunk["intrinsic_matrices"]).unsqueeze(
                0
            ),
            depth_eps=self.dataset_1._depth_eps,
        )
        self.assertEqual(len(expected_result), len(result))
        for i in range(len(expected_result)):
            if expected_result[i] is not None:
                self.assertTrue(
                    torch.allclose(expected_result[i].unsqueeze(0), result[i]), f"{i}"
                )

    def test_get_chunk_interval(self) -> None:
        """Tests that the method `get_chunk_interval` works properly."""
        torch.manual_seed(0)

        expected_results_126 = [(0, 2), (0, 2), (2, 4)]

        for i, expected_result in enumerate(expected_results_126):
            self.assertEqual(self.dataset_1.get_chunk_interval(i), expected_result)
            self.assertEqual(self.dataset_2.get_chunk_interval(i), expected_result)
            self.assertEqual(self.dataset_6.get_chunk_interval(i), expected_result)

        expected_results_34 = [(0, 2), (2, 3), (0, 2), (2, 4)]

        for i, expected_result in enumerate(expected_results_34):
            self.assertEqual(self.dataset_3.get_chunk_interval(i), expected_result)
            self.assertEqual(self.dataset_4.get_chunk_interval(i), expected_result)

        expected_results_5 = [(0, 3)]
        for i, expected_result in enumerate(expected_results_5):
            self.assertEqual(self.dataset_5.get_chunk_interval(i), expected_result)

        expected_results_7 = [(0, 4)]
        for i, expected_result in enumerate(expected_results_7):
            self.assertEqual(self.dataset_7.get_chunk_interval(i), expected_result)

    def test_get_chunk_modality_shape_specified_raises(self) -> None:
        """
        Tests that the method `get_chunk_modality_shape_specified` raises if
        mask_thermal has invalid shape or if sequence_length is incorrect.
        """
        torch.manual_seed(0)
        with self.assertRaises(RuntimeError):
            self.dataset_1.get_chunk_modality_shape_specified(
                0, torch.rand((3,)) <= 0.5, 2
            )

        with self.assertRaises(RuntimeError):
            self.dataset_2.get_chunk_modality_shape_specified(
                0, torch.rand((1,)) <= 0.5, 2
            )

        with self.assertRaises(RuntimeError):
            self.dataset_3.get_chunk_modality_shape_specified(
                0, torch.rand((2,)) <= 0.5, 3
            )

        with self.assertRaises(RuntimeError):
            self.dataset_7.get_chunk_modality_shape_specified(
                0, torch.rand((4,)) <= 0.5, 1
            )

    def test_get_chunk_modality_shape_specified(self) -> None:
        """
        Tests that the method `get_chunk_modality_shape_specified` works properly.
        """

        torch.manual_seed(0)
        for _ in range(10):
            mask_thermal = torch.rand((2,)) <= 0.5
            chunk = self.dataset_1.get_chunk_modality_shape_specified(
                0, mask_thermal=mask_thermal, sequence_length=2
            )
            self.assertTrue(torch.allclose(chunk.mask_thermal, mask_thermal))

            mask_thermal = torch.rand((2,)) <= 0.5
            chunk = self.dataset_4.get_chunk_modality_shape_specified(
                0, mask_thermal=mask_thermal, sequence_length=2
            )
            self.assertTrue(torch.allclose(chunk.mask_thermal, mask_thermal))

            mask_thermal = torch.rand((4,)) <= 0.5
            chunk = self.dataset_7.get_chunk_modality_shape_specified(
                0, mask_thermal=mask_thermal, sequence_length=2
            )
            self.assertEqual(chunk.mask_thermal.shape, (2, 2))
            self.assertTrue(torch.allclose(chunk.mask_thermal.flatten(), mask_thermal))

    def test_process_chunk(self) -> None:
        """
        Test process chunk method on two images with.
        """

        for _ in range(10):
            chunk_items: list[Item] = []
            for i in range(len(self.chunk["images"])):
                chunk_items.append(
                    Item(
                        image=self.chunk["images"][i],
                        image_path=self.chunk["paths"][i],
                        depth_rgb=self.chunk["depths"][i],
                        depth_rgb_path=self.chunk["paths"][i],
                        extrinsic_world2cam_rgb=self.chunk["extrinsic_matrices"][i],
                        intrinsic_rgb=self.chunk["intrinsic_matrices"][i],
                        thermal=self.chunk["thermals"][i],
                        thermal_path=self.chunk["paths"][i],
                        depth_thermal=self.chunk["depths"][i],
                        depth_thermal_path=self.chunk["paths"][i],
                        extrinsic_world2cam_thermal=self.chunk["extrinsic_matrices"][i],
                        intrinsic_thermal=self.chunk["intrinsic_matrices"][i],
                    )
                )
            mask_thermal = torch.rand((2,)) <= 0.5
            processed_chunk = self.dataset_1._process_chunk(
                chunk_items=chunk_items,
                mask_thermal=mask_thermal,
                sequence_length=2,
            )

            # N - number of sequences, S - sequence length, C - number of channels
            self.assertEqual(processed_chunk.images.ndim, 5)
            self.assertEqual(processed_chunk.images.shape[0], 1)  # N
            self.assertEqual(processed_chunk.images.shape[1], 2)  # S
            self.assertEqual(processed_chunk.images.shape[2], 3)  # C
            self.assertEqual(processed_chunk.depths.shape[0], 1)  # N
            self.assertEqual(processed_chunk.depths.shape[1], 2)  # S
            self.assertEqual(
                processed_chunk.images.shape[3:], processed_chunk.depths.shape[2:]
            )
            self.assertEqual(processed_chunk.point_masks.shape[0], 1)  # N
            self.assertEqual(processed_chunk.point_masks.shape[1], 2)  # S
            self.assertEqual(
                processed_chunk.images.shape[3:], processed_chunk.point_masks.shape[2:]
            )
            self.assertEqual(processed_chunk.extrinsics_world2cam.shape, (1, 2, 3, 4))
            self.assertEqual(processed_chunk.intrinsics.shape, (1, 2, 3, 3))
            self.assertEqual(processed_chunk.mask_thermal.shape, (1, 2))

    def test_process_chunk_with_sequences(self) -> None:
        """
        Test process chunk method on 6 images images with which must be split into
        sequences of 2.
        """

        for _ in range(10):
            chunk_items: list[Item] = []
            for i in range(len(self.chunk["images"] * 3)):
                idx_in_dataset = i % 2
                chunk_items.append(
                    Item(
                        image=self.chunk["images"][idx_in_dataset],
                        image_path=self.chunk["paths"][idx_in_dataset],
                        depth_rgb=self.chunk["depths"][idx_in_dataset],
                        depth_rgb_path=self.chunk["paths"][idx_in_dataset],
                        extrinsic_world2cam_rgb=self.chunk["extrinsic_matrices"][
                            idx_in_dataset
                        ],
                        intrinsic_rgb=self.chunk["intrinsic_matrices"][idx_in_dataset],
                        thermal=self.chunk["thermals"][idx_in_dataset],
                        thermal_path=self.chunk["paths"][idx_in_dataset],
                        depth_thermal=self.chunk["depths"][idx_in_dataset],
                        depth_thermal_path=self.chunk["paths"][idx_in_dataset],
                        extrinsic_world2cam_thermal=self.chunk["extrinsic_matrices"][
                            idx_in_dataset
                        ],
                        intrinsic_thermal=self.chunk["intrinsic_matrices"][
                            idx_in_dataset
                        ],
                    )
                )

            mask_thermal = torch.rand((6,)) <= 0.5
            processed_chunk = self.dataset_1._process_chunk(
                chunk_items=chunk_items,
                mask_thermal=mask_thermal,
                sequence_length=2,
            )

            # N - number of sequences, S - sequence length, C - number of channels
            self.assertEqual(processed_chunk.images.ndim, 5)
            self.assertEqual(processed_chunk.images.shape[0], 3)  # N = 3
            self.assertEqual(processed_chunk.images.shape[1], 2)  # S
            self.assertEqual(processed_chunk.images.shape[2], 3)  # C
            self.assertEqual(processed_chunk.depths.shape[0], 3)  # N
            self.assertEqual(processed_chunk.depths.shape[1], 2)  # S
            self.assertEqual(
                processed_chunk.images.shape[3:], processed_chunk.depths.shape[2:]
            )
            self.assertEqual(processed_chunk.point_masks.shape[0], 3)  # N
            self.assertEqual(processed_chunk.point_masks.shape[1], 2)  # S
            self.assertEqual(
                processed_chunk.images.shape[3:], processed_chunk.point_masks.shape[2:]
            )
            self.assertEqual(processed_chunk.extrinsics_world2cam.shape, (3, 2, 3, 4))
            self.assertEqual(processed_chunk.intrinsics.shape, (3, 2, 3, 3))
            self.assertEqual(processed_chunk.mask_thermal.shape, (3, 2))

    def test_get_max_sequence_length(self) -> None:
        """
        Tests that get_max_sequence_length returns correct maximumal sequence length.
        """

        cases = [
            [["Dimsum"], 4],
            [["buildingA_winter"], 3],
            [["Dimsum", "buildingA_winter"], 4],
        ]

        for i, (scenes_names, expected_output) in enumerate(cases):
            with self.subTest(i=i):
                output = VGGTMultipleDataset._get_max_sequence_length(
                    scenes_root_path=self.dataset_path,
                    scenes_names=scenes_names,
                )
                self.assertEqual(output, expected_output)

    def test_build_train_eval_datasets(self) -> None:
        """
        Tests that build_train_eval_datasets creates datasets with correct total length.
        """

        for i in range(10):
            with self.subTest(i=i):
                torch.manual_seed(i)
                train_dataset, eval_dataset = (
                    VGGTMultipleDataset.build_train_eval_datasets(
                        scenes_root_path=self.dataset_path,
                        val_split_ratio=0.5,
                        elements_number=2,
                        depth_eps=1e-8,
                        seed=i,
                        rgb_transform_factory=RGBTransformFactory(),
                        thermal_transform_factory=ThermalTransformFactory(),
                        geometric_transform_config=GeometricTransformConfig(),
                        scenes_per_dataset_path=self.scenes_per_dataset_path,
                    )
                )

                eval_scenes = eval_dataset.scenes_paths
                eval_scenes_names = {scene.name for scene in eval_scenes}
                if eval_scenes_names == {"Dimsum"}:
                    self.assertEqual(len(train_dataset), 1)
                    self.assertEqual(len(eval_dataset), 1)
                    eval_output = eval_dataset[0]
                    self.assertEqual(len(eval_output.to_tuple()), 9)
                    self.assertEqual(eval_output.images.shape[0:2], (1, 4))

                else:
                    self.assertEqual(len(train_dataset), 2)
                    self.assertEqual(len(eval_dataset), 1)
                    eval_output = eval_dataset[0]
                    self.assertEqual(len(eval_output.to_tuple()), 9)
                    self.assertEqual(eval_output.images.shape[0:2], (1, 3))

    def test_build_train_eval_datasets_no_augmentations(self) -> None:
        """
        Tests that build_train_eval_datasets creates datasets with no shape changes for
        eval dataset.
        """

        for subtest_index in range(10):
            with self.subTest(subtest_index=subtest_index):
                torch.manual_seed(subtest_index)
                _, eval_dataset = VGGTMultipleDataset.build_train_eval_datasets(
                    scenes_root_path=self.dataset_path,
                    val_split_ratio=0.5,
                    elements_number=2,
                    depth_eps=1e-8,
                    seed=subtest_index,
                    rgb_transform_factory=RGBTransformFactory(),
                    thermal_transform_factory=ThermalTransformFactory(),
                    geometric_transform_config=GeometricTransformConfig(),
                    scenes_per_dataset_path=self.scenes_per_dataset_path,
                )
                element_0 = eval_dataset[0].to_tuple()
                for _ in range(3):
                    element = eval_dataset[0].to_tuple()
                    for i in range(len(element_0)):
                        if not torch.is_tensor(element_0[i]):
                            continue
                        self.assertEqual(element_0[i].shape, element[i].shape)

    def test_build_train_eval_datasets_undivided(self) -> None:
        """
        Tests that build_train_eval_datasets_undivided creates datasets with correct
        total length, and produces no augmentations.
        """

        for subtest_index in range(10):
            with self.subTest(subtest_index=subtest_index):
                torch.manual_seed(subtest_index)
                train_dataset, eval_dataset = (
                    VGGTMultipleDataset.build_train_eval_datasets_undivided(
                        scenes_root_path=self.dataset_path,
                        val_split_ratio=0.5,
                        depth_eps=1e-8,
                        seed=subtest_index,
                        scenes_per_dataset_path=self.scenes_per_dataset_path,
                    )
                )

                self.assertEqual(len(train_dataset), 1)
                self.assertEqual(len(eval_dataset), 1)
                self.assertEqual(len(eval_dataset[0].to_tuple()), 9)

    def test_build_train_eval_datasets_undivided_same_shape(self) -> None:
        """
        Tests that build_train_eval_datasets_undivided creates datasets with no
        shape changes.
        """
        for subtest_index in range(10):
            with self.subTest(subtest_index=subtest_index):
                torch.manual_seed(subtest_index)
                train_dataset, eval_dataset = (
                    VGGTMultipleDataset.build_train_eval_datasets_undivided(
                        scenes_root_path=self.dataset_path,
                        val_split_ratio=0.5,
                        depth_eps=1e-8,
                        seed=subtest_index,
                        scenes_per_dataset_path=self.scenes_per_dataset_path,
                    )
                )

                # no augmentations train
                element_0 = train_dataset[0].to_tuple()
                for _ in range(3):
                    element = train_dataset[0].to_tuple()
                    for i in range(len(element_0)):
                        if not torch.is_tensor(element_0[i]):
                            continue
                        self.assertEqual(element_0[i].shape, element[i].shape)

                # no augmentations eval
                element_0 = eval_dataset[0].to_tuple()
                for _ in range(3):
                    element = eval_dataset[0].to_tuple()
                    for i in range(len(element_0)):
                        if not torch.is_tensor(element_0[i]):
                            continue
                        self.assertEqual(element_0[i].shape, element[i].shape)

    def test_thermal_ratio(self) -> None:
        dataset = VGGTMultipleDataset(
            root_path=self.dataset_path,
            scenes_names=["Dimsum"],
            min_sequence_length=4,
            scenes_per_dataset_path=self.scenes_per_dataset_path,
            drop_last=False,
        )

        for i, thermal_ratio in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
            dataset.thermal_ratio = thermal_ratio
            chunk = dataset[0]
            self.assertEqual(chunk.mask_thermal.sum().item(), i)

    def test_collate_chunks(self) -> None:
        """Tests that collate_chunks works properly"""
        dataset_1 = VGGTMultipleDataset(
            root_path=self.dataset_path,
            scenes_names=["Dimsum"],
            min_sequence_length=2,
            elements_number=4,
            scenes_per_dataset_path=self.scenes_per_dataset_path,
            drop_last=True,
            geometric_transform=GeometricTransform(
                GeometricTransformConfig(
                    target_image_width=518,
                    aspect_ratio=(0.8, 0.8),
                    crop_ratio=(1.0, 1.0),
                    p_crop=1.0,
                    p_rotate=0.0,
                )
            ),
        )
        # I need to make sure that the size would be (2, 2)
        dataset_1._possible_sequence_lengths = [2]

        dataset_2 = VGGTMultipleDataset(
            root_path=self.dataset_path,
            scenes_names=["buildingA_winter"],
            min_sequence_length=2,
            elements_number=2,
            scenes_per_dataset_path=self.scenes_per_dataset_path,
            drop_last=True,
            geometric_transform=GeometricTransform(
                GeometricTransformConfig(
                    target_image_width=518,
                    aspect_ratio=(0.8, 0.8),
                    crop_ratio=(1.0, 1.0),
                    p_crop=1.0,
                    p_rotate=0.0,
                )
            ),
        )

        chunk1 = dataset_1[0]
        chunk2 = dataset_2[0]

        chunk_combined = VGGTMultipleDataset.collate_chunks([chunk1, chunk2])
        self.assertEqual(len(chunk_combined.datasets_names), 3)
        self.assertEqual(chunk_combined.datasets_names[0], "ThermalGaussian")
        self.assertEqual(chunk_combined.datasets_names[1], "ThermalGaussian")
        self.assertEqual(chunk_combined.datasets_names[2], "ThermoNeRF")

        self.assertEqual(len(chunk_combined.scenes_names), 3)
        self.assertEqual(chunk_combined.scenes_names[0], "Dimsum")
        self.assertEqual(chunk_combined.scenes_names[1], "Dimsum")
        self.assertEqual(chunk_combined.scenes_names[2], "buildingA_winter")

        self.assertEqual(chunk_combined.images.shape[:3], (3, 2, 3))
        self.assertEqual(chunk_combined.depths.shape[:2], (3, 2))
        self.assertEqual(
            chunk_combined.point_masks.shape[:2],
            (
                3,
                2,
            ),
        )
        self.assertTrue(
            chunk_combined.extrinsics_world2cam.shape in [(3, 2, 3, 4), (3, 2, 4, 4)]
        )
        self.assertEqual(chunk_combined.intrinsics.shape, (3, 2, 3, 3))
