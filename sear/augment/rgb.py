from dataclasses import dataclass
from functools import partial

import torch
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from torchvision.transforms import v2 as transforms

from sear.augment.image_transform import ImageTransform


@dataclass
class RGBTransformFactory(ReverseCli):
    """A configuration to create ImageTransform for RGB images."""

    p_apply_together: float = 0.7
    """Probability that augmentations within one batch are the same."""

    p_color_jitter: float = 0.9
    """The probability to apply color jitter."""
    brightness: float = 0.5
    """
    Specifies how much to adjust the brightness. Larger values result in greater
    brightness changes.
    """
    contrast: float = 0.5
    """
    Specifies how much to adjust the contrast. Larger values result in greater contrast
    changes.
    """
    saturation: float = 0.5
    """
    Specifies how much to adjust the saturation. Larger values result in greater
    saturation changes.
    """
    hue: float = 0.1
    """
    Specifies how much to adjust the hue. Larger values result in greater hue changes.
    """

    p_gaussian_noise: float = 0.1
    """The probability to apply gaussian noising."""
    gaussian_noise_sigma: float = 0.05
    """Specifies how much gaussian to add. Larger values result in more noise added."""

    p_gaussian_blur: float = 0.1
    """The probability to apply gaussian blur."""
    gaussian_blur_kernel: int = 5
    """The kernel size of the gaussian blur kernel."""
    gaussian_blur_sigma: tuple[float, float] = (0.1, 2.0)
    """
    The sigma for gaussian blur is uniformly sampled between the provided interval.
    Larger values result in blurrier images.
    """

    p_random_grayscale: float = 0.1
    """The probability to convert an image into grayscale."""

    p_sharpness: float = 0.1
    """The probability adjust sharpness of the image"""
    sharpness_factor: tuple[float, float] = (1.0, 5.0)
    """
    The sharpness factor for sharpening function is uniformly sampled between the
    provided interval. Larger values result in sharper images.
    """

    def __post_init__(self) -> None:
        """Validates the passed fields."""
        prob_fields = {
            "p_gaussian_noise": self.p_gaussian_noise,
            "p_gaussian_blur": self.p_gaussian_blur,
            "p_sharpness": self.p_sharpness,
            "p_color_jitter": self.p_color_jitter,
            "p_random_grayscale": self.p_random_grayscale,
        }
        for name, val in prob_fields.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"`{name}` must be between 0.0 and 1.0, but got {val}")

        interval_fields = {
            "gaussian_blur_sigma": self.gaussian_blur_sigma,
            "sharpness_factor": self.sharpness_factor,
        }
        for name, val in interval_fields.items():
            if val[1] < val[0]:
                raise ValueError(
                    f"`{name}` must has correct order or factors (min, max), "
                    + f"but get {val}"
                )

        if self.gaussian_blur_kernel % 2 != 1:
            raise ValueError(
                f"Gaussian kernel size must be odd, but get {self.gaussian_blur_kernel}"
            )

    @staticmethod
    def get_empty() -> ImageTransform:
        """
        Creates an ImageTransform with no transform.
        """
        return ImageTransform(
            transform_list=None,
            p_apply_together=RGBTransformFactory.p_apply_together,
        )

    def create_transform(self) -> ImageTransform:
        """
        Creates and returns the ImageTransform from inner config parameters.
        """
        transform_list: list[transforms.Transform] = []

        # color jitter
        transform_list.append(
            transforms.RandomApply(
                transforms=[
                    transforms.ColorJitter(
                        brightness=self.brightness,
                        contrast=self.contrast,
                        saturation=self.saturation,
                        hue=self.hue,
                    )
                ],
                p=self.p_color_jitter,
            )
        )

        # gaussian noise
        transform_list.append(
            transforms.RandomApply(
                transforms=[transforms.GaussianNoise(sigma=self.gaussian_noise_sigma)],
                p=self.p_gaussian_noise,
            )
        )

        # gaussian blur
        transform_list.append(
            transforms.RandomApply(
                transforms=[
                    transforms.GaussianBlur(
                        kernel_size=self.gaussian_blur_kernel,
                        sigma=self.gaussian_blur_sigma,
                    )
                ],
                p=self.p_gaussian_blur,
            )
        )

        # random grayscale
        transform_list.append(transforms.RandomGrayscale(p=self.p_random_grayscale))

        # random adjust sharpness

        transform_list.append(
            transforms.RandomApply(
                transforms=[
                    transforms.Lambda(
                        partial(
                            self.random_adjust_sharpness,
                            smallest_sharpness=self.sharpness_factor[0],
                            largest_sharpness=self.sharpness_factor[1],
                        )
                    )
                ],
                p=self.p_sharpness,
            )
        )
        return ImageTransform(
            transform_list=transform_list, p_apply_together=self.p_apply_together
        )

    @staticmethod
    def random_adjust_sharpness(
        x: torch.Tensor, smallest_sharpness: float, largest_sharpness: float
    ) -> torch.Tensor:
        """
        Randomly adjusts sharpness of an image `x` or a batch of images `x`. The
        sharpness factor is uniformly sampled between `smallest_sharpness` and
        `largest_sharpness`.

        :returns processed image or batch of images.
        """

        sharpness_factor = smallest_sharpness + torch.rand((1,)).item() * (
            largest_sharpness - smallest_sharpness
        )

        return transforms.functional.adjust_sharpness(
            x, sharpness_factor=sharpness_factor
        )
