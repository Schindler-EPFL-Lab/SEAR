from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from training.data.dataset_util import (
    crop_image_depth_and_intrinsic_by_pp,
    resize_image_depth_and_intrinsic,
    rotate_90_degrees,
)


@dataclass
class GeometricTransformConfig(ReverseCli):
    """A configuration file to create geometric (crop, rotation) augmentations."""

    target_image_width: int = 518
    """
    The desired output image width (in pixels) after augmentation. The final height is
    computed from this width and a sampled aspect ratio.
    """
    patch_size: int = 14
    """Patch size used by the downstream vision transformer."""
    safe_bound: int = 4
    """
    A padding/safety margin used when resizing/cropping to avoid sampling artifacts near
    image borders.
    """

    p_crop: float = 0.5
    """Probability of applying the random crop/resize augmentation."""
    crop_ratio: tuple[float, float] | None = (0.8, 1.2)
    """
    Range (min, max) used to sample per-axis crop scaling factors relative to the
    original image size. Values are clamped to a maximum of 1.0.
    """
    aspect_ratio: tuple[float, float] | None = (0.33, 1.0)
    """Range (min, max) used to sample the target height/width aspect ratio."""

    p_rotate: float = 0.3
    """Probability of applying a rotation augmentation."""

    def __post_init__(self) -> None:
        if self.target_image_width <= 0:
            raise ValueError("target_image_width must be a positive integer")

        if self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")

        if self.safe_bound < 0:
            raise ValueError("safe_bound must be non-negative")
        if self.safe_bound * 2 >= self.target_image_width:
            raise ValueError("safe_bound is too large relative to target_image_width")

        for name, prob in (
            ("p_crop", self.p_crop),
            ("p_rotate", self.p_rotate),
        ):
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1, got {prob}")

        if self.crop_ratio is not None:
            crop_min, crop_max = self.crop_ratio
            if crop_min <= 0 or crop_max <= 0 or crop_min > crop_max:
                raise ValueError(
                    "crop_ratio must be positive with min <= max, but got "
                    + f"{self.crop_ratio}"
                )

        if (self.aspect_ratio is not None) and (
            self.aspect_ratio[0] <= 0
            or self.aspect_ratio[1] <= 0
            or self.aspect_ratio[0] > self.aspect_ratio[1]
        ):
            raise ValueError(
                "aspect_ratio must be positive with min <= max, got "
                + f"{self.aspect_ratio}"
            )

        if self.target_image_width % self.patch_size != 0:
            raise ValueError(
                "Target image width must be divisible by the patch_size, but"
                + f"{self.target_image_width} % {self.patch_size} != 0"
            )


class GeometricTransform:
    """
    A class to perform geometric (crop/rotation) augmentations on a batch of images.
    """

    def __init__(self, config: GeometricTransformConfig | None = None):
        if config is None:
            config = GeometricTransformConfig()
        self._config = config

    @classmethod
    def empty(cls) -> "GeometricTransform":
        config = GeometricTransformConfig(
            p_crop=0.0,
            aspect_ratio=None,
            crop_ratio=None,
            p_rotate=0.0,
        )
        return cls(config=config)

    @staticmethod
    def _get_target_height(
        target_image_width: int, aspect_ratio: float, patch_size: int
    ) -> int:
        """
        Calculates the target image height based on its `target_image_width` and
        `aspect_ratio`. Also ensures that all the dimentions are divisible by
        `patch_size` to make it processable by vision transformer (DINOv2).
        """

        target_image_height = int(target_image_width * aspect_ratio)
        target_image_height -= target_image_height % patch_size

        return target_image_height

    @staticmethod
    def _random_crop(
        images: torch.Tensor,
        depths: torch.Tensor,
        intrinsic_matrices: torch.Tensor,
        crop_ratio: tuple[float, float] | None,
        aspect_ratio: tuple[float, float] | None,
        target_image_width: int,
        patch_size: int,
        safe_bound: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies random crop to `images`, `depths`, and `intrinsic_matrices`. The image
        is cropped with a crop ratio uniformly sampled from the interval `crop_ratio`,
        then aspect_ratio is randomly sampled from the interval `aspect_ratio`. If
        `aspect_ratio` is None then the original aspect ratio of images is kept. The
        result image has width `target_image_width` and its height is divisible by
        `patch_size`. The `safe_bound` specifies a small margin that can be eliminated
        for the algorithm to work robust.

        :returns processed `images`, `depths`, and `intrinsic_matrices`.

        :raises RuntimeError if:
            - the length of `images`, `depths`, or `intrinsic_matrices` mismatches,
            - the `images` and `depths` have different resolutions (H, W),
            - the `images` does not have shape (B, 3, H, W),
            - the `depths` does not have shape (B, H, W),
            - the `intrinsic_matrices` does not have shape (B, 3, 3).
        """

        if not (
            images.shape[0] == depths.shape[0]
            and depths.shape[0] == intrinsic_matrices.shape[0]
        ):
            raise RuntimeError(
                "`images`, `depths`, and `intrinsic_matrices` must have the same length"
                + f"but get {images.shape[0]} {depths.shape[0]} "
                + f"{intrinsic_matrices.shape[0]} respectively."
            )

        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(
                f"`images` must have shape (B, 3, H, W), but get {images.shape}."
            )

        if depths.ndim != 3:
            raise RuntimeError(
                f"`depths` must have shape (B, H, W), but get {depths.shape}."
            )

        if images.shape[2:] != depths.shape[1:]:
            raise RuntimeError(
                "`images` and `depths` must have the same resolution (H, W), but get "
                + f"{images.shape[2:]} {depths.shape[2:]} respectively."
            )

        if intrinsic_matrices.shape[1:] != (3, 3):
            raise RuntimeError(
                "`intrinsic_matrices` must have shape (B, 3, 3), but get "
                + f"{intrinsic_matrices.shape}."
            )

        crop_ratio_wh = torch.tensor([1.0, 1.0])
        if crop_ratio is not None:
            crop_ratio_wh = (crop_ratio[1] - crop_ratio[0]) * torch.rand(
                (2,)
            ) + crop_ratio[0]
            crop_ratio_wh = torch.minimum(crop_ratio_wh, torch.ones_like(crop_ratio_wh))
        augmented_shape = crop_ratio_wh * torch.tensor(images.shape[2:4])
        augmented_shape = augmented_shape.to(torch.int32)
        augmented_shape = augmented_shape.numpy()

        # crop images, and update depths and intrinsic matrices
        images_np: list[npt.NDArray[np.uint8]] = [
            (images[i].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            for i in range(images.shape[0])
        ]
        depths_np: list[npt.NDArray[np.float32]] = [
            depths[i].numpy() for i in range(depths.shape[0])
        ]
        intrinsic_matrices_np: list[npt.NDArray[np.float32]] = [
            intrinsic_matrices[i].numpy() for i in range(intrinsic_matrices.shape[0])
        ]

        # find target shape with proper aspect ratio
        aspect_ratio_h = images.shape[2] / images.shape[3]
        if aspect_ratio is not None:
            aspect_ratio_h = (aspect_ratio[1] - aspect_ratio[0]) * torch.rand(
                (1,)
            ).item() + aspect_ratio[0]
        target_image_height = GeometricTransform._get_target_height(
            target_image_width=target_image_width,
            aspect_ratio=aspect_ratio_h,
            patch_size=patch_size,
        )
        target_image_shape = np.array([target_image_height, target_image_width])

        for i in range(images.shape[0]):
            image = images_np[i]
            depth = depths_np[i]
            intrinsic_matrix = intrinsic_matrices_np[i]

            # crop image to the augmented shape
            image, depth, intrinsic_matrix, _ = crop_image_depth_and_intrinsic_by_pp(
                image=image,
                depth_map=depth,
                intrinsic=intrinsic_matrix,
                target_shape=augmented_shape,
            )

            original_size = np.array(image.shape[:2])

            # resize in case the original image shape is smaller than the target one
            image, depth, intrinsic_matrix, _ = resize_image_depth_and_intrinsic(
                image=image,
                depth_map=depth,
                intrinsic=intrinsic_matrix,
                target_shape=target_image_shape,
                original_size=original_size,
                safe_bound=safe_bound,
                rescale_aug=crop_ratio is not None and aspect_ratio is not None,
            )

            # crop image to the target shape
            image, depth, intrinsic_matrix, _ = crop_image_depth_and_intrinsic_by_pp(
                image=image,
                depth_map=depth,
                intrinsic=intrinsic_matrix,
                target_shape=target_image_shape,
                strict=True,
            )

            images_np[i] = image
            depths_np[i] = depth
            intrinsic_matrices_np[i] = intrinsic_matrix

        images = (
            torch.from_numpy(np.stack(images_np) / 255).to(images).permute(0, 3, 1, 2)
        )
        depths = torch.from_numpy(np.stack(depths_np)).to(depths)
        intrinsic_matrices = torch.from_numpy(np.stack(intrinsic_matrices_np)).to(
            intrinsic_matrices
        )

        return images, depths, intrinsic_matrices

    def _random_rotation(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        intrinsic_matrices: torch.Tensor,
        extrinsic_matrices_world2cam: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies a rotation of either 90 or -90 degrees to `images`, `depths`, and
        `extrinsic_matrices_world2cam`.

        :returns a tuple of augmented images, depths, and extrinsic matrices.

        :raises RuntimeError if:
            - the length of `images`, `depths`, `intrinsic_matrices` or
              `extrinsic_matrices_world2cam` mismatches,
            - the `images` and `depths` have different resolutions (H, W),
            - the `images` does not have shape (B, 3, H, W),
            - the `depths` does not have shape (B, H, W),
            - the `intrinsic_matrices` does not have shape (B, 3, 3).
            - the `extrinsic_matrices_world2cam` does not have shape (B, 3, 4).
        """
        if not (
            images.shape[0] == depths.shape[0]
            and depths.shape[0] == intrinsic_matrices.shape[0]
            and intrinsic_matrices.shape[0] == extrinsic_matrices_world2cam.shape[0]
        ):
            raise RuntimeError(
                "`images`, `depths`, `intrinsic_matrices`, and "
                "`extrinsic_matrices_world2cam`"
                + f" must have the same length, but get {images.shape[0]} "
                + f"{depths.shape[0]} {intrinsic_matrices.shape[0]} "
                + f"{extrinsic_matrices_world2cam.shape[0]} respectively."
            )

        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(
                f"`images` must have shape (B, 3, H, W), but get {images.shape}."
            )

        if depths.ndim != 3:
            raise RuntimeError(
                f"`depths` must have shape (B, H, W), but get {depths.shape}."
            )

        if images.shape[2:] != depths.shape[1:]:
            raise RuntimeError(
                "`images` and `depths` must have the same resolution (H, W), but get "
                + f"{images.shape[2:]} {depths.shape[2:]} respectively."
            )

        if extrinsic_matrices_world2cam.shape[1:] == (4, 4):
            extrinsic_matrices_world2cam = extrinsic_matrices_world2cam[:, :3, :]

        if extrinsic_matrices_world2cam.shape[1:] != (3, 4):
            raise RuntimeError(
                "`extrinsic_matrices_world2cam` must have shape (B, 3, 4), but get "
                + f"{extrinsic_matrices_world2cam.shape}."
            )

        if intrinsic_matrices.shape[1:] != (3, 3):
            raise RuntimeError(
                "`intrinsic_matrices` must have shape (B, 3, 3), but get "
                + f"{intrinsic_matrices.shape}."
            )

        result_images_np = np.empty(
            (images.shape[0], images.shape[1], images.shape[3], images.shape[2]),
            dtype=np.float32,
        )
        result_depths_np = np.empty(
            (depths.shape[0], depths.shape[2], depths.shape[1]),
            dtype=np.float32,
        )
        result_extrinsic_matrices_world2cam_np = np.empty(
            extrinsic_matrices_world2cam.shape, dtype=np.float32
        )
        result_intrinsic_matrices_np = np.empty(
            intrinsic_matrices.shape, dtype=np.float32
        )

        for i in range(len(images)):
            image_np = images[i].permute(1, 2, 0).numpy()
            depth_np = depths[i].numpy()
            extrinsic_matrix_world2cam_np = extrinsic_matrices_world2cam[i].numpy()
            intrinsic_matrix_np = intrinsic_matrices[i].numpy()

            clockwise = torch.rand((1,)).item() < 0.5
            (
                image_np,
                depth_np,
                extrinsic_matrix_world2cam_np,
                intrinsic_matrix_np,
                _,
            ) = rotate_90_degrees(
                image=image_np,
                depth_map=depth_np,
                extri_opencv=extrinsic_matrix_world2cam_np,
                intri_opencv=intrinsic_matrix_np,
                clockwise=clockwise,
                track=None,
            )

            result_images_np[i] = image_np.transpose([2, 0, 1])
            result_depths_np[i] = depth_np
            result_extrinsic_matrices_world2cam_np[i] = extrinsic_matrix_world2cam_np
            result_intrinsic_matrices_np[i] = intrinsic_matrix_np

        return (
            torch.from_numpy(result_images_np).to(images),
            torch.from_numpy(result_depths_np).to(depths),
            torch.from_numpy(result_extrinsic_matrices_world2cam_np).to(
                extrinsic_matrices_world2cam
            ),
            torch.from_numpy(result_intrinsic_matrices_np).to(intrinsic_matrices),
        )

    def apply(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        intrinsic_matrices: torch.Tensor,
        extrinsic_matrices_world2cam: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies geometric augmentations on a batch of `images`, `depths`,
        `intrinsic_matrices` and `extrinsic_matrices_world2cam`.

        :returns a tuple containing:
            - augmented images
            - augmented depths
            - augmented extrinsic matrices in opencv world-to-camera convention
            - augmented intrinsic matrices

        :raises ValueError if:
            1. The `images` shape is not [B, 3, H, W], the `depths` shape is not [B, H,
               W], the `intrinsic_matrices` shape is not [B, 3, 3], the
               `extrinsic_matrices_world2cam` shape is not [B, 3, 4].
            2. The shapes mismatch between batch size for `images`, `depths`,
               `intrinsic_matrices`, `extrinsic_matrices_world2cam`.
        """
        if images.ndim != 4:
            raise ValueError(
                f"The `images` shape must be [B, 3, H, W], but get {images.shape}"
            )

        if images.shape[1] != 3:
            raise ValueError(
                f"The `images` must have 3 channels, but get {images.shape[1]}"
            )

        if depths.ndim != 3:
            raise ValueError(
                f"The `depths` shape must be [B, H, W], but get {depths.shape}"
            )

        if images.shape[2:] != depths.shape[1:]:
            raise RuntimeError(
                "`images` and `depths` must have the same resolution (H, W), but get "
                + f"{images.shape[2:]} {depths.shape[2:]} respectively."
            )

        if intrinsic_matrices.shape[1:] != (3, 3):
            raise ValueError(
                "The `intrinsic_matrices` shape must be [B, 3, 3], but get "
                + f"{intrinsic_matrices.shape}"
            )

        if extrinsic_matrices_world2cam.shape[1:] != (3, 4):
            raise ValueError(
                "The `extrinsic_matrices_world2cam` shape must be [B, 3, 4], but get "
                + f"{extrinsic_matrices_world2cam.shape}"
            )

        if not (
            images.shape[0] == depths.shape[0]
            and depths.shape[0] == intrinsic_matrices.shape[0]
            and intrinsic_matrices.shape[0] == extrinsic_matrices_world2cam.shape[0]
        ):
            raise ValueError(
                "The lengths of images, depths, intrinsic_matrices, "
                + f"extrinsic_matrices_world2cam must match, but get {images.shape}, "
                + f"{depths.shape}, {intrinsic_matrices.shape}, "
                + f"{extrinsic_matrices_world2cam.shape}."
            )

        crop_ratio = None
        aspect_ratio = None
        if torch.rand((1,)).item() < self._config.p_crop:
            crop_ratio = self._config.crop_ratio
            aspect_ratio = self._config.aspect_ratio

        images, depths, intrinsic_matrices = self._random_crop(
            images=images,
            depths=depths,
            intrinsic_matrices=intrinsic_matrices,
            crop_ratio=crop_ratio,
            aspect_ratio=aspect_ratio,
            target_image_width=self._config.target_image_width,
            patch_size=self._config.patch_size,
            safe_bound=self._config.safe_bound,
        )

        if torch.rand((1,)).item() < self._config.p_rotate:
            images, depths, extrinsic_matrices_world2cam, intrinsic_matrices = (
                self._random_rotation(
                    images=images,
                    depths=depths,
                    extrinsic_matrices_world2cam=extrinsic_matrices_world2cam,
                    intrinsic_matrices=intrinsic_matrices,
                )
            )

        return images, depths, extrinsic_matrices_world2cam, intrinsic_matrices
