"""
A dataset to load paired RGB+Thermal images
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sear import logger
from sear.augment.geometric import GeometricTransform, GeometricTransformConfig
from sear.data_processing.paired_item import PairedItem


@dataclass
class PairedDataset:
    """
    A class to load METU_VisTIR dataset introduced in XoFTR
    https://github.com/OnderT/XoFTR
    """

    def __init__(
        self,
        dataset_path: Path,
        dataset_name: str = "METU_VisTIR",
        max_size: int = 128,
    ) -> None:
        """
        Reads the dataset located at `dataset_path`, the name of the dataset is
        `dataset_name`. The `max_size` defines the maximum size of simultaneously loaded
        and processed images.
        """

        folder_of_interest = "index/scene_info_test"

        scenes_paths = list((dataset_path / folder_of_interest).iterdir())
        scenes_paths = sorted(scenes_paths)
        self._items: list[PairedItem] = []

        for scene_index, scene_path in enumerate(scenes_paths):
            scene_info = dict(np.load(scene_path, allow_pickle=True))

            pairs_ids_raw = scene_info["pair_infos"].astype(np.int32)  # (N, 2)
            intrinsics_raw = scene_info["intrinsics"]  # (M, 2, 3, 3)
            poses_raw = scene_info["poses"]  # (M, 4, 4)
            image_paths_raw = scene_info["image_paths"]  # (M, 2)

            for batch_index in range(0, len(pairs_ids_raw), max_size):
                batched_ids = pairs_ids_raw[batch_index : batch_index + max_size]

                intrinsics0 = intrinsics_raw[batched_ids[:, 0], 0]  # (N, 3, 3)
                intrinsics1 = intrinsics_raw[batched_ids[:, 1], 1]  # (N, 3, 3)
                intrinsics = np.stack([intrinsics0, intrinsics1], axis=1)

                poses0 = poses_raw[batched_ids[:, 0]]  # (N, 4, 4)
                poses1 = poses_raw[batched_ids[:, 1]]  # (N, 4, 4)
                poses = np.stack([poses0, poses1], axis=1)

                image_paths0 = image_paths_raw[batched_ids[:, 0], 0]
                image_paths1 = image_paths_raw[batched_ids[:, 1], 1]
                image_paths = np.stack([image_paths0, image_paths1], axis=1)

                mask_thermal = [
                    ["thermal" in image_path for image_path in image_path_pair]
                    for image_path_pair in image_paths
                ]

                images_rgb = [
                    np.asarray(Image.open(dataset_path / image_path).convert("RGB"))
                    / 255.0
                    for image_path in image_paths[:, 0]
                ]
                images_rgb = np.stack(images_rgb)
                # [N, H, W, 3] -> [N, 3, H, W]
                images_rgb = images_rgb.transpose([0, 3, 1, 2])
                images_thermal = [
                    np.asarray(Image.open(dataset_path / image_path).convert("RGB"))
                    / 255.0
                    for image_path in image_paths[:, 1]
                ]

                images_thermal = np.stack(images_thermal)
                images_thermal = images_thermal.transpose([0, 3, 1, 2])

                self.images_rgb = images_rgb
                self.images_thermal = images_thermal

                processed_images, processed_intrinsics = (
                    self._combine_rgb_thermal_images(
                        images_rgb=torch.from_numpy(images_rgb),
                        images_thermal=torch.from_numpy(images_thermal),
                        intrinsics=torch.from_numpy(intrinsics),
                        desired_aspect_ratio=images_thermal.shape[2]
                        / images_thermal.shape[3],
                    )
                )

                self._items.append(
                    PairedItem(
                        images=processed_images.to(torch.float32),
                        images_paths=[
                            [dataset_path / path for path in pair]
                            for pair in image_paths
                        ],
                        extrinsics_world2cam=torch.from_numpy(poses).to(torch.float32),
                        intrinsics=processed_intrinsics.to(torch.float32),
                        mask_thermal=torch.tensor(mask_thermal),
                        scenes_names=[scene_path.name] * processed_images.shape[0],
                        datasets_names=[dataset_name] * processed_images.shape[0],
                    )
                )
            logger.info(
                f"Loaded {scene_path.name} {scene_index + 1} / {len(scenes_paths)} "
                + "PairedDataset"
            )

    @staticmethod
    def _combine_rgb_thermal_images(
        images_rgb: torch.Tensor,
        images_thermal: torch.Tensor,
        intrinsics: torch.Tensor,
        desired_aspect_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Crops RGB and thermal images to the same size, which is defined by
        `desired_aspect_ratio` and stacks them together.

        :return: processed images of shape (N, 2, 3, H, W) and processed intrinsics of
            shape (N, 2, 3, 3)
        """
        cropping_method = GeometricTransform(
            GeometricTransformConfig(
                target_image_width=518,
                p_crop=1.0,
                crop_ratio=None,
                aspect_ratio=(desired_aspect_ratio, desired_aspect_ratio),
                p_rotate=0.0,
            )
        )
        processed_images_rgb, _, _, processed_intrinsics_rgb = cropping_method.apply(
            images=images_rgb,
            depths=torch.ones(
                images_rgb.shape[0], images_rgb.shape[2], images_rgb.shape[3]
            ),
            intrinsic_matrices=intrinsics[:, 0, :, :],
            extrinsic_matrices_world2cam=torch.eye(4)[None, :, :].repeat(
                images_rgb.shape[0], 1, 1
            )[:, :3, :],
        )
        processed_images_thermal, _, _, processed_intrinsics_thermal = (
            cropping_method.apply(
                images=images_thermal,
                depths=torch.ones(
                    images_thermal.shape[0],
                    images_thermal.shape[2],
                    images_thermal.shape[3],
                ),
                intrinsic_matrices=intrinsics[:, 1, :, :],
                extrinsic_matrices_world2cam=torch.eye(4)[None, :, :].repeat(
                    images_thermal.shape[0], 1, 1
                )[:, :3, :],
            )
        )

        processed_images = torch.cat(
            [
                processed_images_rgb[:, None, ...],
                processed_images_thermal[:, None, ...],
            ],
            dim=1,
        )

        processed_intrinsics = torch.cat(
            [
                processed_intrinsics_rgb[:, None, ...],
                processed_intrinsics_thermal[:, None, ...],
            ],
            dim=1,
        )

        return processed_images, processed_intrinsics

    def __len__(self) -> int:
        """Returns the length of the current dataset"""
        return len(self._items)

    def __getitem__(self, index: int) -> PairedItem:
        """
        Returns a set of pairs at `index`

        :raise: ValueError if index exceeds the size of the dataset.
        """
        if index >= len(self):
            raise ValueError(f"The `index` must be <= {len(self)}")

        return self._items[index]
