import torch
from torchvision.transforms import v2 as transforms


class ImageTransform:
    """
    A class to process images, which is used to augment the images without changing the
    depths and camera parameters.
    """

    def __init__(
        self,
        transform_list: list[transforms.Transform] | None = None,
        p_apply_together: float = 0.7,
    ):
        """
        Initializates the ImageTransform class from a set of transforms `transform_list`
        which later will be applied together on a batch with the probability
        `p_apply_together`.
        """
        self._p_apply_together = p_apply_together
        if transform_list is None or len(transform_list) == 0:
            transform_list = [transforms.Lambda(lambda x: x)]
        self._transform = transforms.Compose(transforms=transform_list)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applied augmentation on an image or a batch of images `x`.

        :returns the processed image or the processed batch of images.
        """

        if x.ndim != 3 and x.ndim != 4:
            raise ValueError(
                "The image must have shape of (C, H, W) or (B, C, H, W) but get "
                + f"{x.shape}"
            )

        if x.ndim == 3 and x.shape[0] != 3:
            raise ValueError(
                f"The image number of channels must be 3, but get {x.shape[0]}."
            )

        if x.ndim == 4 and x.shape[1] != 3:
            raise ValueError(
                f"The image number of channels must be 3, but get {x.shape[1]}."
            )

        if x.ndim == 3:
            return self._transform(x)

        if torch.rand((1,)).item() > self._p_apply_together:
            for i in range(x.shape[0]):
                x[i] = self._transform(x[i])
            return x

        return self._transform(x)
