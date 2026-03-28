from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class PairedItem:
    """
    Represents a chunk of data containing images (thermal or rgb), their
    corresponding depth maps, camera extrinsic and intrinsic matrices, file paths,
    and thermal mask
    """

    images: torch.Tensor
    """The images tensor of shape (N, 2, 3, H, W)"""
    images_paths: list[list[Path]]
    """The file path to the images of shape (N, 2)."""
    """The file paths to the depths map of shape (N, 2)."""
    extrinsics_world2cam: torch.Tensor
    """
    The extrinsic matrix (4x4 or 3x4) for the RGB camera in OpenCV world-to-camera
    convention of shape (N, 2, 4, 4) or (N, 2, 3, 4)
    """
    intrinsics: torch.Tensor
    """The intrinsic matrix for the cameras of shape (N, 2, 3, 3)."""
    mask_thermal: torch.Tensor
    """
    A mask of shape (N, 2) specifying which images are thermal and which are rgb.
    """
    scenes_names: list[str]
    """
    Names of the scenes corresponding to each pair of images, of shape (N,).
    """
    datasets_names: list[str]
    """
    Names of the datasets corresponding to each pair of images, of shape (N,).
    """

    def __post_init__(self) -> None:
        """
        Validates the shapes and dimensions of after initialization.

        :raise:
            ValueError if
                - `images` does not have 5 dimensions (N, 2, 3, H, W).
                - `extrinsics_world2cam` does not have 4 dimensions or its last two
                dimensions are not (3, 4) or (4, 4).
                - `intrinsics` does not have 4 dimensions or its last two
                dimensions are not (3, 3).
                - `images_paths`, `mask_thermal`, do not have 2 dimensions (N, 2).
                - Any of the elements (`images`, `images_paths`, `extrinsics_world2cam`,
                `intrinsics`, `point_masks`, `mask_thermal`) do not have the same (N, 2)
                shape in their first two dimensions.
        """

        if self.images.ndim != 5 or self.images.shape[2] != 3:
            raise ValueError(
                "`images` must have 5 dimensions (N, 2, 3, H, W), got "
                + f"{self.images.shape}"
            )
        if self.extrinsics_world2cam.ndim != 4 or self.extrinsics_world2cam.shape[
            -2:
        ] not in [(3, 4), (4, 4)]:
            raise ValueError(
                "`extrinsics_world2cam` must have 4 dimensions (N, 2, 4, 4) or "
                + f"(N, 2, 3, 4), got {self.extrinsics_world2cam.shape}"
            )
        if self.intrinsics.ndim != 4 or self.intrinsics.shape[-2:] != (3, 3):
            raise ValueError(
                "`intrinsics` must have 4 dimensions (N, 2, 3, 3), got "
                + f"{self.intrinsics.shape}"
            )

        def paths_correct_shape(
            paths: list[list[Path]], expected_shape: tuple[int, int]
        ) -> bool:
            """Checks if the paths have the `expected_shape`"""
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

        if self.mask_thermal.ndim != 2:
            raise ValueError(
                "The `mask_thermal` must be of shape (N, 2), got "
                + f"{self.mask_thermal.shape}."
            )

        for element, name in [
            (self.images, "images"),
            (self.images_paths, "images_paths"),
            (self.extrinsics_world2cam, "extrinsics_world2cam"),
            (self.intrinsics, "intrinsics"),
            (self.mask_thermal, "mask_thermal"),
        ]:
            if len(element) != self.images.shape[0] or len(element[0]) != 2:
                raise ValueError(
                    "All the elements must be of shape (N, 2, ...), got "
                    + f"{len(element), len(element[0])} for {name}"
                )

        for element, name in [
            (self.scenes_names, "scenes_names"),
            (self.datasets_names, "datasets_names"),
        ]:
            if len(element) != self.images.shape[0]:
                raise ValueError(
                    f"The {name} elements must be of shape (N,), got {len(element)}"
                )

    def to_device(self, device: torch.device) -> "PairedItem":
        """Places all elements on `device`"""
        self.images = self.images.to(device)
        self.extrinsics_world2cam = self.extrinsics_world2cam.to(device)
        self.intrinsics = self.intrinsics.to(device)
        self.mask_thermal = self.mask_thermal.to(device)
        return self

    def iterate_batched(
        self,
        batch_size: int = 64,
    ) -> Generator["PairedItem", None, None]:
        """
        Iterates over the loaded images and returns pairs of images in batched format
        such that at most `batch_size` pairs are presented in one returned PairedItem.
        """

        for i in range(0, self.images.shape[0], batch_size):
            yield PairedItem(
                images=self.images[i : i + batch_size],
                images_paths=self.images_paths[i : i + batch_size],
                extrinsics_world2cam=self.extrinsics_world2cam[i : i + batch_size],
                intrinsics=self.intrinsics[i : i + batch_size],
                mask_thermal=self.mask_thermal[i : i + batch_size],
                scenes_names=self.scenes_names[i : i + batch_size],
                datasets_names=self.datasets_names[i : i + batch_size],
            )
