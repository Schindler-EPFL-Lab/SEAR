from dataclasses import dataclass
from functools import partial

import torch
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from torchvision.transforms import v2 as transforms

from sear.augment.image_transform import ImageTransform
from sear.augment.rgb import RGBTransformFactory


@dataclass
class ThermalTransformFactory(ReverseCli):
    """A configuration to create ImageTransform for thermal images."""

    p_apply_together: float = 0.7
    """Probability that augmentations within one batch are the same."""

    p_gaussian_noise: float = 0.1
    """The probability to apply gaussian noising."""
    gaussian_noise_sigma: float = 0.05
    """Specifies how much gaussian to add. Larger values result in more noise added."""

    p_gaussian_blur: float = 0.1
    """The probability to apply gaussian blur."""
    gaussian_blur_kernel: int = 5
    """The kernel size of the gaussian blur kernel."""
    gaussian_blur_sigma: tuple[float, float] = (1.0, 4.0)
    """
    The sigma for gaussian blur is uniformly sampled between the provided interval.
    Larger values result in blurrier thermal images.
    """

    p_sharpness: float = 0.1
    """The probability adjust sharpness of the thermal image"""
    sharpness_factor: tuple[float, float] = (2.0, 10.0)
    """
    The sharpness factor for sharpening function is uniformly sampled between the
    provided interval. Larger values result in sharper thermal images.
    """

    p_random_power: float = 0.5
    """The probability to evaluate the thermal image in degree"""
    random_power: float = 1.5
    """
    The "random power" is sampled uniformly from [1.0, 1.5]. With the probability of
    0.5 the power is turned into in 1/"random power"
    """

    p_random_linear: float = 0.5
    """The probability to change the min and max values of the thermal image"""
    random_linear: tuple[float, float] = (0.05, 0.95)
    """The values from what new min and max are sampled."""
    random_linear_minimal_difference: float = 0.5
    """The minimal acceptable difference between new min and new max."""

    def __post_init__(self) -> None:
        """Validates the passed fields."""
        prob_fields = {
            "p_gaussian_noise": self.p_gaussian_noise,
            "p_gaussian_blur": self.p_gaussian_blur,
            "p_sharpness": self.p_sharpness,
            "p_random_power": self.p_random_power,
            "p_random_linear": self.p_random_linear,
        }
        for name, val in prob_fields.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"`{name}` must be between 0.0 and 1.0, but got {val}")

        interval_fields = {
            "gaussian_blur_sigma": self.gaussian_blur_sigma,
            "sharpness_factor": self.sharpness_factor,
            "random_linear": self.random_linear,
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

        if self.random_power < 1.0:
            raise ValueError(
                f"`random_power` must be greater than 1.0, but get {self.random_power}"
            )

        if not (0.0 <= self.random_linear[0] <= 1):
            raise ValueError("The `random_linear[0]` must be between 0.0 and 1.0")
        if not (0.0 <= self.random_linear[1] <= 1):
            raise ValueError("The `random_linear[1]` must be between 0.0 and 1.0")

        if (
            self.random_linear_minimal_difference
            > self.random_linear[1] - self.random_linear[0]
        ):
            raise ValueError(
                "The values for `random_linear_minimal_difference` and `random_linear` "
                + "are incorrect because it is impossible to make difference between "
                + f"{self.random_linear} larger than "
                + f"{self.random_linear_minimal_difference}"
            )

    @staticmethod
    def get_empty() -> ImageTransform:
        """
        Creates an ImageTransform with no transform.
        """
        return ImageTransform(
            transform_list=None,
            p_apply_together=ThermalTransformFactory.p_apply_together,
        )

    def create_transform(self) -> ImageTransform:
        """
        Creates and returns the ImageTransform from inner config parameters.
        """
        transform_list: list[transforms.Transform] = []

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

        # random adjust sharpness
        transform_list.append(
            transforms.RandomApply(
                transforms=[
                    transforms.Lambda(
                        partial(
                            RGBTransformFactory.random_adjust_sharpness,
                            smallest_sharpness=self.sharpness_factor[0],
                            largest_sharpness=self.sharpness_factor[1],
                        )
                    )
                ],
                p=self.p_sharpness,
            )
        )

        # random power transform
        transform_list.append(
            transforms.RandomApply(
                transforms=[
                    transforms.Lambda(
                        partial(
                            self._random_power_transform,
                            random_power=self.random_power,
                        )
                    )
                ],
                p=self.p_random_power,
            )
        )

        # random linear
        transform_list.append(
            transforms.RandomApply(
                transforms=[
                    transforms.Lambda(
                        partial(
                            self._random_linear,
                            smallest_boundary=self.random_linear[0],
                            largest_boundary=self.random_linear[1],
                            minimal_difference=self.random_linear_minimal_difference,
                        )
                    )
                ],
                p=self.p_random_linear,
            )
        )
        return ImageTransform(
            transform_list=transform_list, p_apply_together=self.p_apply_together
        )

    @staticmethod
    def _random_power_transform(x: torch.Tensor, random_power: float) -> torch.Tensor:
        """
        Apply a randomized power-law (gamma) transform to a non-negative tensor. The
        power is sampled from [1, `random_power`] and with 50% probability inversed
        (1/sampled power).

        :returns x raised to a power.
        """

        gamma = 1.0 + torch.rand((1,)).item() * (random_power - 1.0)
        inverse_power = torch.rand((1,)).item() <= 0.5
        if inverse_power:
            gamma = 1 / gamma
        return torch.pow(x, exponent=gamma)

    @staticmethod
    def _random_linear(
        x: torch.Tensor,
        smallest_boundary: float,
        largest_boundary: float,
        minimal_difference: float,
        debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, float, float, float, float]:
        """
        Linearly rescales tensor a tensor `x` into a randomly chosen interval within
        [`smallest_boundary`, `largest_boundary`]. The function samples two endpoints
        uniformly, enforces a minimal difference between as `minimal_difference`. The
        `debug` specifies whether the return some intermediate parameters.

        :returns a tensor transformed into the chosen interval.
        """

        lower_01, upper_01 = torch.rand((2,)).tolist()
        if lower_01 > upper_01:
            upper_01, lower_01 = lower_01, upper_01

        lower = smallest_boundary + (largest_boundary - smallest_boundary) * lower_01
        upper = smallest_boundary + (largest_boundary - smallest_boundary) * upper_01
        if (upper - lower) <= minimal_difference:
            possible_lower = upper - minimal_difference
            possible_upper = lower + minimal_difference
            if possible_lower >= 0:
                lower = possible_lower
            elif possible_upper <= 1:
                upper = possible_upper
            else:
                lower = smallest_boundary
                upper = largest_boundary

        x_min = x.min()
        x_max = x.max()

        result_image = lower + (x - x_min) / (x_max - x_min + 1e-8) * (upper - lower)
        if debug:
            return result_image, lower_01, upper_01, lower, upper
        return result_image
