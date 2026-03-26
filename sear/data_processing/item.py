from dataclasses import astuple, dataclass
from pathlib import Path

import torch


@dataclass
class Item:
    """
    Item represents a single data sample containing paired RGB and thermal images,
    their corresponding depth maps, camera extrinsic and intrinsic matrices, and
    file paths.
    """

    image: torch.Tensor
    """The RGB image tensor of shape (3, H, W) with values in [0, 1]."""
    image_path: Path
    """The file path to the RGB image."""
    depth_rgb: torch.Tensor
    """The depth map corresponding to the RGB image, shape (H, W) values >= 0."""
    depth_rgb_path: Path
    """The file path to the RGB depth map."""
    extrinsic_world2cam_rgb: torch.Tensor
    """
    The extrinsic matrix (4x4 or 3x4) for the RGB camera  in OpenCV world-to-camera
    convention.
    """
    intrinsic_rgb: torch.Tensor
    """The intrinsic matrix (3x3) for the RGB camera."""

    thermal: torch.Tensor
    """
    The thermal image tensor of shape (3, H, W) with values in [0, 1].
    """
    thermal_path: Path
    """
    The file path to the thermal image.
    """
    depth_thermal: torch.Tensor
    """The depth map corresponding to the thermal image, shape (H, W) values >= 0"""
    depth_thermal_path: Path
    """The file path to the thermal depth map"""
    extrinsic_world2cam_thermal: torch.Tensor
    """
    The extrinsic matrix (4x4 or 3x4) for the thermal camera in OpenCV
    world-to-camera convention
    """
    intrinsic_thermal: torch.Tensor
    """The intrinsic matrix (3x3) for the thermal camera."""

    def __post_init__(self) -> None:
        """
        Validates the shapes, dimensions, value ranges, and file paths of the
        instance attributes after initialization.

        :raise:
            ValueError if
                - The `extrinsic_world2cam_rgb` and `extrinsic_world2cam_thermal`
                    are not of shape (4, 4) or (3, 4).
                - The `intrinsic_rgb` and `intrinsic_thermal` are not of shape (3,
                    3).
                - The `image` and `thermal` tensors are not of shape (3, H, W).
                - The `depth_rgb` and `depth_thermal` tensors are not of shape (H,
                    W).
                - The the spatial resolutions of `image`, `depth_rgb`, `thermal`,
                    and
                `depth_thermal` are not equal.
                - The values in `image` and `thermal` tensors are not the range [0,
                1].
                - The values in `depth_rgb` and `depth_thermal` tensors are
                    negative.

            FileNotFoundError if the file paths for `image_path`, `depth_rgb_path`,
                `thermal_path`, and `depth_thermal_path` do not exist.
        """

        if self.extrinsic_world2cam_rgb.shape not in [(4, 4), (3, 4)]:
            raise ValueError(
                f"extrinsic_world2cam_rgb must be of shape (4, 4) or (3, 4), "
                f"but got {self.extrinsic_world2cam_rgb.shape}"
            )
        if self.extrinsic_world2cam_thermal.shape not in [(4, 4), (3, 4)]:
            raise ValueError(
                f"extrinsic_world2cam_thermal must be of shape (4, 4) or (3, 4), "
                f"but got {self.extrinsic_world2cam_thermal.shape}"
            )

        if self.intrinsic_rgb.shape != (3, 3):
            raise ValueError(
                "intrinsic_rgb must be of shape (3, 3), but got "
                + f"{self.intrinsic_rgb.shape}"
            )
        if self.intrinsic_thermal.shape != (3, 3):
            raise ValueError(
                "intrinsic_thermal must be of shape (3, 3), but got "
                + f"{self.intrinsic_thermal.shape}"
            )

        if self.image.ndim != 3 or self.image.shape[0] != 3:
            raise ValueError("The `image` tensor must be of shape (3, H, W)")
        if self.depth_rgb.ndim != 2:
            raise ValueError("The `depth_rgb` tensor must be of shape (H, W)")
        if self.thermal.ndim != 3 or self.thermal.shape[0] != 3:
            raise ValueError("The `thermal` tensor must be of shape (3, H, W)")
        if self.depth_thermal.ndim != 2:
            raise ValueError("The `depth_thermal` tensor must be of shape (H, W)")

        if not (
            self.image.shape[-2:] == self.depth_rgb.shape[-2:]
            and self.depth_rgb.shape[-2:] == self.thermal.shape[-2:]
            and self.thermal.shape[-2:] == self.depth_thermal.shape
        ):
            raise ValueError(
                "Resolutions of image, depth_rgb, thermal, and depth_thermal must "
                + f"be equal but got {self.image.shape[-2:]}, "
                + f"{self.depth_rgb.shape[-2:]}, {self.thermal.shape[-2:]}, "
                + f"{self.depth_thermal.shape[-2:]} respectively."
            )

        if not torch.all((0 <= self.image) & (self.image <= 1)):
            raise ValueError("The `image` tensor values must be in [0, 1]")
        if not torch.all(0 <= self.depth_rgb):
            raise ValueError("The `depth_rgb` tensor values must be positive")
        if not torch.all((0 <= self.thermal) & (self.thermal <= 1)):
            raise ValueError("The `thermal` tensor values must be in [0, 1]")
        if not torch.all(0 <= self.depth_thermal):
            raise ValueError("The `depth_thermal` tensor values must be positive")

        for path in [
            self.image_path,
            self.depth_rgb_path,
            self.thermal_path,
            self.depth_thermal_path,
        ]:
            if not Path(path).exists():
                raise FileNotFoundError(f"File path does not exist: {path}")

    def to_tuple(
        self,
    ) -> tuple[
        torch.Tensor,
        Path,
        torch.Tensor,
        Path,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Path,
        torch.Tensor,
        Path,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        :return: elements in an item, which are:
            - image of shape (3, H, W),
            - absolute path to the image,
            - rgb depth of shape (H, W),
            - absolute path to the rgb depth,
            - camera pose of rgb image of shape (4, 4) or (3, 4) in world-to-camera
            OpenCV convention,
            - intrinsic rgb camera parameters of shape (3, 3)
            - thermal image of shape (3, H, W),
            - absolute path to the thermal image,
            - thermal depth of shape (H, W),
            - absolute path to the thermal depth,
            - camera pose of thermal image of shape (4, 4) or (3, 4) in world-to-camera
            OpenCV convention,
            - intrinsic thermal camera parameters of shape (3, 3)
        """
        return astuple(self)
