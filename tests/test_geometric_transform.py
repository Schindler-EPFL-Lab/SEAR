import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch
from vggt.utils.geometry import closed_form_inverse_se3

# Remove it when https://github.com/facebookresearch/vggt/issues/416 is fixed
sys.path.append("vggt")
sys.path.append("vggt/training")

from sear.augment.geometric import (
    GeometricTransform,
    GeometricTransformConfig,
)
from sear.data_processing.single_dataset import VGGTSingleDataset


class TestGeometricAugmentation(unittest.TestCase):
    """
    Tests that GeometricAugmentator class works properly
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
        cls.full_augmentation = GeometricTransform(
            GeometricTransformConfig(
                p_crop=1.0,
                p_rotate=1.0,
            )
        )
        cls.zero_augmentation = GeometricTransform.empty()

        cls.extrinsic_matrix_opencv_0_cam2world = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        cls.intrinsic_matrix_0 = np.array(
            [
                [200.0, 0.0, 240.0],
                [0.0, 223.0, 260.0],
                [0.0, 0.0, 1.0],
            ]
        )

        cls.scene = o3d.t.geometry.RaycastingScene()
        sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).translate(
            [0.0, 5.0, 0.0]
        )
        sphere1 = o3d.t.geometry.TriangleMesh.from_legacy(sphere1)
        cls.scene.add_triangles(sphere1)

        sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).translate(
            [3.0, 5.0, 3.0]
        )
        sphere2 = o3d.t.geometry.TriangleMesh.from_legacy(sphere2)
        cls.scene.add_triangles(sphere2)

        radius_max_px = 68
        delta_depth_max = np.linalg.norm([3.0, 5.0, 3.0])
        delta_rgb_max = 1.0
        delta_radius_px = 1

        cls.threshold_depth = (
            2 * np.pi * radius_max_px * delta_radius_px * delta_depth_max
        )
        cls.threshold_rgb = 2 * np.pi * radius_max_px * delta_radius_px * delta_rgb_max

    def test_incorrect_shape(self) -> None:
        """Tests that the class raises an error if called with incorrect shape."""
        correct_images = [torch.rand((1, 3, 22, 33))]
        incorrect_images = [
            torch.rand((1, 1, 12, 34)),
            torch.rand((45, 22)),
            torch.rand((1, 4, 3, 45, 22)),
        ]
        images_to_test = correct_images + incorrect_images

        correct_depths = [torch.rand((2, 22, 33))]
        incorrect_depths = [
            torch.rand((1, 1, 12, 34)),
            torch.rand((45, 22)),
        ]
        depths_to_test = correct_depths + incorrect_depths

        correct_intrinsic_matrices = [torch.rand((3, 3, 3))]
        incorrect_intrinsic_matrices = [torch.rand((1, 2, 3, 3)), torch.rand((2, 4, 7))]
        intrinsic_matrices_to_test = (
            correct_intrinsic_matrices + incorrect_intrinsic_matrices
        )

        correct_extrinsic_matrices = [torch.rand((4, 3, 4))]
        incorrect_extrinsic_matrices = [torch.rand((1, 4, 3, 4)), torch.rand((2, 2, 3))]
        extrinsic_matrices_to_test = (
            correct_extrinsic_matrices + incorrect_extrinsic_matrices
        )

        for images, depths, intrinsic_matrices, extrinsic_matrices in itertools.product(
            images_to_test,
            depths_to_test,
            intrinsic_matrices_to_test,
            extrinsic_matrices_to_test,
        ):
            with self.assertRaises(ValueError):
                self.full_augmentation.apply(
                    images=images,
                    depths=depths,
                    intrinsic_matrices=intrinsic_matrices,
                    extrinsic_matrices_world2cam=extrinsic_matrices,
                )

            with self.assertRaises(ValueError):
                self.zero_augmentation.apply(
                    images=images,
                    depths=depths,
                    intrinsic_matrices=intrinsic_matrices,
                    extrinsic_matrices_world2cam=extrinsic_matrices,
                )

    def test_produce_correct_output_one_element(self) -> None:
        """
        Tests that GeometricAugmentator produces images and depths with correct width,
        that images values lies between [0.0, 1.0], and that depths values are larger
        than 0.0, that the produced extrinsic matrices have correct rotation matrices,
        if run with batch size with shape 1.
        """

        for _ in range(10):
            for dataset in [self.dataset_1, self.dataset_2]:
                for i in range(len(dataset)):
                    dataset_item = dataset[i]

                    image = dataset_item.image
                    depth = dataset_item.depth_rgb
                    extrinsic_matrix_world2cam = dataset_item.extrinsic_world2cam_rgb
                    intrinsic_matrix = dataset_item.intrinsic_rgb

                    (
                        processed_images,
                        processed_depths,
                        processed_extrinsic_matrices_world2cam,
                        processed_intrinsic_matrices,
                    ) = self.full_augmentation.apply(
                        images=image.unsqueeze(0),
                        depths=depth.unsqueeze(0),
                        intrinsic_matrices=intrinsic_matrix.unsqueeze(0),
                        extrinsic_matrices_world2cam=extrinsic_matrix_world2cam.unsqueeze(
                            0
                        ),
                    )

                    self.assertTrue(processed_images.max() <= 1.0)
                    self.assertTrue(processed_images.min() >= 0.0)
                    self.assertTrue(processed_depths.min() >= 0.0)
                    self.assertTrue(
                        processed_images.shape[2]
                        == GeometricTransformConfig.target_image_width
                    )
                    self.assertTrue(
                        processed_depths.shape[1]
                        == GeometricTransformConfig.target_image_width
                    )
                    self.assertTrue(
                        torch.allclose(
                            torch.linalg.det(
                                processed_extrinsic_matrices_world2cam[:, :, :3]
                            ),
                            torch.ones((1,)).to(extrinsic_matrix_world2cam),
                        )
                    )
                    self.assertEqual(
                        processed_extrinsic_matrices_world2cam.shape, (1, 3, 4)
                    )
                    self.assertEqual(processed_intrinsic_matrices.shape, (1, 3, 3))

                    (
                        processed_images,
                        processed_depths,
                        processed_extrinsic_matrices_world2cam,
                        processed_intrinsic_matrices,
                    ) = self.zero_augmentation.apply(
                        images=image.unsqueeze(0),
                        depths=depth.unsqueeze(0),
                        intrinsic_matrices=intrinsic_matrix.unsqueeze(0),
                        extrinsic_matrices_world2cam=extrinsic_matrix_world2cam.unsqueeze(
                            0
                        ),
                    )

                    self.assertTrue(processed_images.max() <= 1.0)
                    self.assertTrue(processed_images.min() >= 0.0)
                    self.assertTrue(
                        processed_images.ndim == 4 and processed_images.shape[0] == 1
                    )
                    self.assertTrue(processed_depths.min() >= 0.0)
                    self.assertTrue(
                        processed_depths.ndim == 3 and processed_depths.shape[0] == 1
                    )
                    self.assertTrue(
                        processed_images.shape[3]
                        == GeometricTransformConfig.target_image_width
                    )
                    self.assertTrue(
                        processed_depths.shape[2]
                        == GeometricTransformConfig.target_image_width
                    )
                    self.assertTrue(
                        torch.allclose(
                            processed_extrinsic_matrices_world2cam,
                            extrinsic_matrix_world2cam.unsqueeze(0),
                        )
                    )
                    self.assertEqual(
                        processed_extrinsic_matrices_world2cam.shape,
                        (1, 3, 4),
                    )
                    self.assertEqual(processed_intrinsic_matrices.shape, (1, 3, 3))

    def test_produce_correct_output_multiple_elements(self) -> None:
        """
        Tests that GeometricAugmentator produces images and depths with correct width,
        that images values lies between [0.0, 1.0], and that depths values are larger
        than 0.0, that the produced extrinsic matrices have correct rotation matrices,
        if run with batch size equal to lenth of the whole dataset.
        """

        for _ in range(10):
            for dataset in [self.dataset_1, self.dataset_2]:
                elements = [dataset[i] for i in range(len(dataset))]
                images = torch.stack([elements[i].image for i in range(len(elements))])
                depths = torch.stack(
                    [elements[i].depth_rgb for i in range(len(elements))]
                )
                extrinsic_matrices_world2cam = torch.stack(
                    [elements[i].extrinsic_world2cam_rgb for i in range(len(elements))]
                )
                intrinsic_matrices = torch.stack(
                    [elements[i].intrinsic_rgb for i in range(len(elements))]
                )

                (
                    processed_images,
                    processed_depths,
                    processed_extrinsic_matrices_world2cam,
                    processed_intrinsic_matrices,
                ) = self.full_augmentation.apply(
                    images=images,
                    depths=depths,
                    intrinsic_matrices=intrinsic_matrices,
                    extrinsic_matrices_world2cam=extrinsic_matrices_world2cam,
                )

                self.assertTrue(processed_images.max() <= 1.0)
                self.assertTrue(processed_images.min() >= 0.0)
                self.assertTrue(
                    processed_images.ndim == 4
                    and processed_images.shape[0] == len(dataset)
                )

                self.assertTrue(processed_depths.min() >= 0.0)
                self.assertTrue(
                    processed_depths.ndim == 3
                    and processed_depths.shape[0] == len(dataset)
                )

                self.assertTrue(
                    processed_images.shape[3]
                    == GeometricTransformConfig.target_image_width
                    or processed_images.shape[2]
                    == GeometricTransformConfig.target_image_width
                )
                self.assertTrue(
                    processed_depths.shape[2]
                    == GeometricTransformConfig.target_image_width
                    or processed_depths.shape[1]
                    == GeometricTransformConfig.target_image_width
                )
                self.assertTrue(
                    torch.allclose(
                        torch.linalg.det(
                            processed_extrinsic_matrices_world2cam[:, :, :3]
                        ),
                        torch.ones((extrinsic_matrices_world2cam.shape[0],)).to(
                            extrinsic_matrices_world2cam
                        ),
                    )
                )
                self.assertEqual(
                    processed_extrinsic_matrices_world2cam.shape[1:], (3, 4)
                )
                self.assertEqual(processed_intrinsic_matrices.shape[1:], (3, 3))

                (
                    processed_images,
                    processed_depths,
                    processed_extrinsic_matrices_world2cam,
                    processed_intrinsic_matrices,
                ) = self.zero_augmentation.apply(
                    images=images,
                    depths=depths,
                    intrinsic_matrices=intrinsic_matrices,
                    extrinsic_matrices_world2cam=extrinsic_matrices_world2cam,
                )

                self.assertTrue(processed_images.max() <= 1.0)
                self.assertTrue(processed_images.min() >= 0.0)
                self.assertTrue(processed_depths.min() >= 0.0)
                self.assertTrue(
                    processed_images.shape[3]
                    == GeometricTransformConfig.target_image_width
                )
                self.assertTrue(
                    processed_depths.shape[2]
                    == GeometricTransformConfig.target_image_width
                )
                self.assertTrue(
                    torch.allclose(
                        processed_extrinsic_matrices_world2cam,
                        extrinsic_matrices_world2cam,
                    )
                )
                self.assertEqual(
                    processed_intrinsic_matrices.shape, intrinsic_matrices.shape
                )

    @staticmethod
    def _render(
        scene: "o3d.t.geometry.RaycastingScene",
        intrinsic_matrix: npt.NDArray[np.float32],
        extrinsic_matrix_cam2world: npt.NDArray[np.float32],
        width: int = 500,
        height: int = 500,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Renders image with of Open3D `scene`, with `width` and `height`. The camera has
        intrinsic parameters specified in `intrinsic_matrix`, and extrinsic parameters
        (position and look-at) in `extrinsic_matrix_cam2world`. The
        `extrinsic_matrix_cam2world` is in camera-to-world opencv convention.
        """

        extrinsic_matrix_world2cam = closed_form_inverse_se3(
            extrinsic_matrix_cam2world[None]
        )[0]
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsic_matrix,
            extrinsic_matrix=extrinsic_matrix_world2cam,
            width_px=width,
            height_px=height,
        )

        ans = scene.cast_rays(rays)
        rgb = (ans["t_hit"].numpy() < 100.0).astype(np.float32)
        rgb = rgb[:, :, None].repeat(3, axis=2)

        depth = ans["t_hit"].numpy()
        depth[depth == np.inf] = 0.0

        return rgb, depth

    def test_simple_scene_crop(self) -> None:
        """
        Checks that intrinsic and extrinsic matrices agree with the image and depth
        after crop.
        """

        image_0, depth_0 = self._render(
            scene=self.scene,
            intrinsic_matrix=self.intrinsic_matrix_0,
            extrinsic_matrix_cam2world=self.extrinsic_matrix_opencv_0_cam2world,
        )

        images_0 = torch.from_numpy(image_0).to(torch.float32).permute(2, 0, 1)[None]
        depths_0 = torch.from_numpy(depth_0).to(torch.float32)[None]
        intrinsics_0 = torch.from_numpy(self.intrinsic_matrix_0)[None]

        for i in range(10):
            torch.manual_seed(i)
            images_1, depths_1, intrinsics_1 = self.full_augmentation._random_crop(
                images=images_0,
                depths=depths_0,
                intrinsic_matrices=intrinsics_0,
                crop_ratio=GeometricTransformConfig.crop_ratio,
                aspect_ratio=GeometricTransformConfig.aspect_ratio,
                target_image_width=GeometricTransformConfig.target_image_width,
                patch_size=GeometricTransformConfig.patch_size,
                safe_bound=GeometricTransformConfig.safe_bound,
            )

            intrinsics_1_np = intrinsics_1[0].numpy()
            images_1_np = images_1[0].permute(1, 2, 0).numpy()
            depths_1_np = depths_1[0].numpy()

            images_1_rendered, depths_1_rendered = self._render(
                scene=self.scene,
                intrinsic_matrix=intrinsics_1_np,
                extrinsic_matrix_cam2world=self.extrinsic_matrix_opencv_0_cam2world,
                width=images_1_np.shape[1],
                height=images_1_np.shape[0],
            )

            self.assertTrue(
                np.linalg.norm(images_1_rendered - images_1_np) < self.threshold_rgb
            )
            self.assertTrue(
                np.linalg.norm(depths_1_rendered - depths_1_np) < self.threshold_depth
            )

    def test_simple_scene_rotation(self) -> None:
        """
        Checks that intrinsic and extrinsic matrices agree with the image and
        depth after rotation.
        """

        image_0, depth_0 = self._render(
            scene=self.scene,
            intrinsic_matrix=self.intrinsic_matrix_0,
            extrinsic_matrix_cam2world=self.extrinsic_matrix_opencv_0_cam2world,
        )

        image_0 = torch.from_numpy(image_0).to(torch.float32).permute(2, 0, 1)
        depth_0 = torch.from_numpy(depth_0).to(torch.float32)
        intrinsic_0 = torch.from_numpy(self.intrinsic_matrix_0)
        extrinsic_0_cam2world = torch.from_numpy(
            self.extrinsic_matrix_opencv_0_cam2world
        )
        extrinsic_0_world2cam = closed_form_inverse_se3(extrinsic_0_cam2world[None])[0]

        for i in range(10):
            torch.manual_seed(i)
            image_1, depth_1, extrinsic_1_world2cam, intrinsic_1 = (
                self.full_augmentation._random_rotation(
                    images=image_0.unsqueeze(0),
                    depths=depth_0.unsqueeze(0),
                    extrinsic_matrices_world2cam=extrinsic_0_world2cam.unsqueeze(0),
                    intrinsic_matrices=intrinsic_0.unsqueeze(0),
                )
            )

            image_1 = image_1[0]
            depth_1 = depth_1[0]
            extrinsic_1_world2cam = extrinsic_1_world2cam[0]
            intrinsic_1 = intrinsic_1[0]

            extrinsics_1_np_world2cam = extrinsic_1_world2cam.numpy()
            extrinsics_1_np_cam2world = closed_form_inverse_se3(
                extrinsics_1_np_world2cam[None]
            )[0]
            image_1_np = image_1.permute(1, 2, 0).numpy()
            depth_1_np = depth_1.numpy()
            intrinsic_1 = intrinsic_1.numpy()

            image_1_rendered, depth_1_rendered = self._render(
                scene=self.scene,
                intrinsic_matrix=intrinsic_1,
                extrinsic_matrix_cam2world=extrinsics_1_np_cam2world,
                width=image_1_np.shape[1],
                height=image_1_np.shape[0],
            )

            self.assertTrue(np.allclose(image_1_rendered, image_1_np))
            self.assertTrue(np.allclose(depth_1_rendered, depth_1_np))

    def test_augmentations_composite(self) -> None:
        """
        Checks that intrinsic and extrinsic matrices agree with the image and depth
        after applying rotation/rotation alpha and crop.
        """

        image_0, depth_0 = self._render(
            scene=self.scene,
            intrinsic_matrix=self.intrinsic_matrix_0,
            extrinsic_matrix_cam2world=self.extrinsic_matrix_opencv_0_cam2world,
        )

        images_0 = torch.from_numpy(image_0).to(torch.float32).permute(2, 0, 1)[None]
        depths_0 = torch.from_numpy(depth_0).to(torch.float32)[None]
        intrinsics_0 = torch.from_numpy(self.intrinsic_matrix_0)[None]
        extrinsics_0_cam2world = torch.from_numpy(
            self.extrinsic_matrix_opencv_0_cam2world
        )[None]
        extrinsics_0_world2cam = closed_form_inverse_se3(extrinsics_0_cam2world)

        for i in range(10):
            torch.manual_seed(i)
            images_1, depths_1, extrinsics_1_world2cam, intrinsics_1 = (
                self.full_augmentation.apply(
                    images=images_0,
                    depths=depths_0,
                    intrinsic_matrices=intrinsics_0,
                    extrinsic_matrices_world2cam=extrinsics_0_world2cam[:, :3, :],
                )
            )

            intrinsics_1_np = intrinsics_1[0].numpy()
            extrinsics_1_world2cam = torch.cat(
                [
                    extrinsics_1_world2cam,
                    torch.zeros_like(extrinsics_1_world2cam[:, 0:1, :]),
                ],
                dim=1,
            )
            extrinsics_1_world2cam[:, 3, 3] = 1.0
            extrinsics_1_np_world2cam = extrinsics_1_world2cam[0].numpy()
            extrinsics_1_np_cam2world = closed_form_inverse_se3(
                extrinsics_1_np_world2cam[None]
            )[0]
            images_1_np = images_1[0].permute(1, 2, 0).numpy()
            depths_1_np = depths_1[0].numpy()

            images_1_rendered, depths_1_rendered = self._render(
                scene=self.scene,
                intrinsic_matrix=intrinsics_1_np,
                extrinsic_matrix_cam2world=extrinsics_1_np_cam2world,
                width=images_1_np.shape[1],
                height=images_1_np.shape[0],
            )

            self.assertTrue(
                np.linalg.norm(images_1_rendered - images_1_np) < self.threshold_rgb
            )
            self.assertTrue(
                np.linalg.norm(depths_1_rendered - depths_1_np) < self.threshold_depth
            )
