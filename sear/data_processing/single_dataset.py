import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image

from sear.data_processing.frame_info import FrameInfo
from sear.data_processing.item import Item


class VGGTSingleDataset(torch.utils.data.Dataset):
    """
    A dataset containing a single VGGT processed scene, which has rgb images, thermal
    images, depth maps, and rgb camera poses.
    """

    def __init__(
        self,
        scene_path: Path,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Instantiates a dataset containing a single VGGT-processed scene. The
        `scene_path` specifies the path where the processed dataset is stored. The
        `dtype` defines the data type of the returned tensors.

        :raises `FileNotFoundError` if:
            - An RGB image specified by `file_path` in `transforms.json` does not exist.
            - A thermal image specified by `thermal_file_path` in `transforms.json` does
              not exist.
            - A depth map specified by `depth_file_path` in `transforms.json` does not
              exist.
        """
        super().__init__()

        self._scene_path = scene_path
        self._dtype = dtype

        self._items: list[Item] = []

        self._rgb_transforms = transforms.ToTensor()

        transforms_path = self._scene_path / "transforms.json"

        # Load the transform.json file
        with open(transforms_path, "r") as f:
            self._transforms_data = json.load(f)

        for frame in self._transforms_data.get("frames", []):
            frame_rgb = frame["rgb"]
            frame_thermal = frame["thermal"]
            extrinsic_matrix_world2cam_rgb, intrinsic_matrix_rgb = (
                FrameInfo.dict_to_matrices(frame_rgb)
            )
            extrinsic_matrix_world2cam_thermal, intrinsic_matrix_thermal = (
                FrameInfo.dict_to_matrices(frame_thermal)
            )

            rgb_filepath = scene_path / Path(frame_rgb["file_path"])
            if not rgb_filepath.exists():
                raise FileNotFoundError(f"RGB image {str(rgb_filepath)} does not exist")
            image = Image.open(rgb_filepath)
            image = self._rgb_transforms(image).to(self._dtype)

            depth_rgb = self._load_depth(
                scene_path / Path(frame_rgb["depth_file_path"]), dtype=self._dtype
            )
            depth_thermal = self._load_depth(
                scene_path / Path(frame_thermal["depth_file_path"]), dtype=self._dtype
            )

            thermal_filepath = scene_path / Path(frame_thermal["file_path"])
            if not thermal_filepath.exists():
                raise FileNotFoundError(
                    f"Thermal map {str(thermal_filepath)} does not exist"
                )
            thermal = Image.open(thermal_filepath).convert("RGB")
            thermal = self._rgb_transforms(thermal).to(self._dtype)

            self._items.append(
                Item(
                    image=image,
                    image_path=rgb_filepath,
                    depth_rgb=depth_rgb,
                    depth_rgb_path=scene_path / Path(frame_rgb["depth_file_path"]),
                    extrinsic_world2cam_rgb=extrinsic_matrix_world2cam_rgb,
                    intrinsic_rgb=intrinsic_matrix_rgb,
                    thermal=thermal,
                    thermal_path=thermal_filepath,
                    depth_thermal=depth_thermal,
                    depth_thermal_path=(
                        scene_path / Path(frame_thermal["depth_file_path"])
                    ),
                    extrinsic_world2cam_thermal=extrinsic_matrix_world2cam_thermal,
                    intrinsic_thermal=intrinsic_matrix_thermal,
                )
            )

    @staticmethod
    def _load_depth(depth_filepath: Path, dtype: torch.dtype) -> torch.Tensor:
        """
        Loads depth from `depth_filepath` and converts it to torch tensor with `dtype`.

        :return: depth with shape (H, W) as a tensor

        :raise: FileNotFoundError if `depth_filepath` does not exist.
        """
        if not depth_filepath.exists():
            raise FileNotFoundError(f"Depth map {str(depth_filepath)} does not exist")
        depth = np.load(depth_filepath)
        depth = torch.from_numpy(depth).to(dtype)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        return depth

    def __len__(self) -> int:
        """Returns the length of the dataset"""
        return len(self._items)

    def __getitem__(self, i: int) -> Item:
        """
        Returns a data element at index `i`

        :returns Let the initial image has shapes (H, W, 3), then the return shapes is a
            tuple of:
                - image: [3, H, W]
                - depth_rgb: [H, W]
                - extrinsic_matrix_world2cam_rgb: [3, 4] - OpenCV world-to-camera
                - intrinsic_matrix_rgb: [3, 3]

                - thermal: [3, H, W]
                - depth_thermal: [H, W]
                - extrinsic_matrix_thermal: [3, 4]
                - intrinsic_matrix_world2cam_thermal: [3, 3] - OpenCV world-to-camera
        """
        if i >= len(self):
            raise RuntimeError(f"key {i} is out of range [0, {len(self)}).")

        return self._items[i]
