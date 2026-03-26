import json
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Chunk:
    """
    Represents a chunk of data containing images (thermal or rgb), their
    corresponding depth maps, camera extrinsic and intrinsic matrices, file paths,
    and thermal mask
    """

    images: torch.Tensor
    """The images tensor of shape (N, S, 3, H, W)"""
    images_paths: list[list[Path]]
    """The file path to the images of shape (N, S)."""
    depths: torch.Tensor
    """The depth map corresponding to the images, shape (N, S, H, W)"""
    depths_paths: list[list[Path]]
    """The file paths to the depths map of shape (N, S)."""
    extrinsics_world2cam: torch.Tensor
    """
    The extrinsic matrix (4x4 or 3x4) for the RGB camera  in OpenCV world-to-camera
    convention of shape (N, S, 4, 4) or (N, S, 3, 4)
    """
    intrinsics: torch.Tensor
    """The intrinsic matrix for the cameras of shape (N, S, 3, 3)."""
    point_masks: torch.Tensor
    """Mask that specifies correct pixels."""
    mask_thermal: torch.Tensor
    """
    A mask of shape (N, S) specifying which images are thermal and which are rgb.
    """
    scenes_per_dataset_path: Path = Path("./sear/configs/scenes_per_dataset.json")
    """Contains information about what scenes belong to what dataset."""

    def __post_init__(self) -> None:
        """
        Validates the shapes and dimensions of after initialization.

        :raise:
            ValueError if
                - `images` does not have 5 dimensions (N, S, 3, H, W).
                - `depths` does not have 4 dimensions (N, S, H, W).
                - `extrinsics_world2cam` does not have 4 dimensions or its last two
                dimensions are not (3, 4) or (4, 4).
                - `intrinsics` does not have 4 dimensions or its last two
                dimensions are not (3, 3).
                - `images_paths`, `mask_thermal`, or `depths_paths` do not have 2
                dimensions (N, S).
                - Any of the elements (`images`, `images_paths`, `depths`,
                `depths_paths`, `extrinsics_world2cam`, `intrinsics`, `point_masks`,
                `mask_thermal`) do not have the same (N, S) shape in their first two
                dimensions.

            RuntimeError if the underlying scene is not presented in the
                `scenes_per_dataset_path`
        """

        if self.images.ndim != 5 or self.images.shape[2] != 3:
            raise ValueError(
                "`images` must have 5 dimensions (N, S, 3, H, W), got "
                + f"{self.images.shape}"
            )
        if self.depths.ndim != 4:
            raise ValueError(
                "`depths` must have 4 dimensions (N, S, H, W), got "
                + f"{self.depths.shape}"
            )
        if self.extrinsics_world2cam.ndim != 4 or self.extrinsics_world2cam.shape[
            -2:
        ] not in [(3, 4), (4, 4)]:
            raise ValueError(
                "`extrinsics_world2cam` must have 4 dimensions (N, S, 4, 4) or "
                + f"(N, S, 3, 4), got {self.extrinsics_world2cam.shape}"
            )
        if self.intrinsics.ndim != 4 or self.intrinsics.shape[-2:] != (3, 3):
            raise ValueError(
                "`intrinsics` must have 4 dimensions (N, S, 3, 3), got "
                + f"{self.intrinsics.shape}"
            )

        def paths_correct_shape(
            paths: list[list[Path]], expected_shape: tuple[int, int]
        ) -> bool:
            if len(paths) != expected_shape[0]:
                return False
            for item in paths:
                if len(item) != expected_shape[1]:
                    return False
            return True

        if not paths_correct_shape(
            self.images_paths, (self.images.shape[0], self.images.shape[1])
        ):
            raise RuntimeError(
                f"The `images_paths` must be of shape {self.images.shape[:2]}"
            )

        if not paths_correct_shape(
            self.depths_paths, (self.depths.shape[0], self.depths.shape[1])
        ):
            raise RuntimeError(
                f"The `depths_paths` must be of shape {self.depths.shape[:2]}"
            )

        if self.mask_thermal.ndim != 2:
            raise ValueError(
                "The `mask_thermal` must be of shape (N, S), got "
                + f"{self.mask_thermal.shape}."
            )

        for element, name in [
            (self.images, "images"),
            (self.depths, "depths"),
            (self.extrinsics_world2cam, "extrinsics_world2cam"),
            (self.intrinsics, "intrinsics"),
            (self.point_masks, "point_masks_torch"),
            (self.mask_thermal, "mask_thermal"),
        ]:
            if element.shape[:2] != self.images.shape[:2]:
                raise ValueError(
                    "All the elements must be of shape (N, S, ...), got "
                    + f"{element.shape} for {name}"
                )

        self._scenes_names: list[str] = []
        for sequence_index in range(len(self.images_paths)):
            all_scenes_names = [
                image_path.parent.parent.name
                for image_path in self.images_paths[sequence_index]
            ]
            unique_scenes = np.unique(all_scenes_names)
            if len(unique_scenes) > 1:
                raise RuntimeError(
                    "Only one scene must be present in one sequence, but got "
                    + f"{unique_scenes} for {self.images_paths}"
                )
            self._scenes_names.append(unique_scenes[0])

        with open(self.scenes_per_dataset_path) as f:
            scenes_per_dataset = json.load(f)

        self._scenes_per_dataset = {
            scene: k
            for k, scene_list in scenes_per_dataset.items()
            for scene in scene_list
        }

        for scene_name in self._scenes_names:
            if scene_name not in self._scenes_per_dataset:
                raise RuntimeError(
                    f"The scene {scene_name} is not specified in "
                    + f"{self.scenes_per_dataset_path}: {scenes_per_dataset}."
                )

        self._datasets_names = [
            self._scenes_per_dataset[scene_name] for scene_name in self._scenes_names
        ]

    def to_device(self, device: torch.device) -> "Chunk":
        """Places all elements on `device`"""
        self.images = self.images.to(device)
        self.depths = self.depths.to(device)
        self.extrinsics_world2cam = self.extrinsics_world2cam.to(device)
        self.intrinsics = self.intrinsics.to(device)
        self.point_masks = self.point_masks.to(device)
        self.mask_thermal = self.mask_thermal.to(device)
        return self

    def to_tuple(
        self,
    ) -> tuple[
        torch.Tensor,
        list[list[Path]],
        torch.Tensor,
        list[list[Path]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Path,
    ]:
        """
        :return: elements of the chunk, which are:
            - images of shape (N, 3, H, W),
            - images paths of length N, which contains absolute paths to images
            - depths of shape (N, H, W),
            - depths paths of length N, which contains absolute paths to depths,
            - camera poses of shape (N, 3, 4) or (N, 4, 4),
            - intrinsics camera parameters of shape (N, 3, 3),
            - point masks of shape (N, H, W) which define what pixels are correct,
            - mask thermal of shape (N,) where True defines that the corresponding image
              is thermal,
            - scenes_per_dataset_path contains information about what scenes belong to
              what datasets.
        """
        return astuple(self)

    @property
    def datasets_names(self) -> list[str]:
        """Returns to what dataset the scene belongs."""
        return self._datasets_names

    @property
    def scenes_names(self) -> list[str]:
        """Returns scene name of the current chunk."""
        return self._scenes_names
