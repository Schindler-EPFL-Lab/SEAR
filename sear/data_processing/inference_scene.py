import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from vggt.utils.load_fn import load_and_preprocess_images


@dataclass(kw_only=True)
class InferenceScene:
    """
    Represents a chunk of data containing images (thermal or rgb), their corresponding
    file paths, and thermal mask
    """

    images: torch.Tensor
    """The images tensor of shape (1, S, 3, H, W)"""
    images_paths: list[list[Path]]
    """The file path to the images of shape (1, S)."""
    mask_thermal: torch.Tensor
    """A mask of shape (1, S) specifying which images are thermal and which are rgb."""
    scene_name: str
    """The name of the scene the images belong to."""

    def __post_init__(self) -> None:
        """
        Validates the shapes and dimensions of after initialization.

        :raise:
            ValueError if
                - `images` does not have 5 dimensions (S, 3, H, W).
                - `images_paths` or `mask_thermal` do not have 2
                dimensions (S,).
                - Any of the elements (`images`, `images_paths`, `mask_thermal`)
                do not have the same (S,) shape in their first two dimensions.

            RuntimeError if the underlying scene is not presented in the
                `scenes_per_dataset_path`
        """

        if self.images.ndim != 5 or self.images.shape[2] != 3:
            raise ValueError(
                "`images` must have 5 dimensions (1, S, 3, H, W), got "
                + f"{self.images.shape}"
            )
        if self.mask_thermal.ndim != 2:
            raise ValueError(
                "`mask_thermal` must be of shape (1, S) but got "
                + f"{self.mask_thermal.shape}"
            )

        for element, name in [
            (self.images, "images"),
            (self.images_paths, "images_paths"),
            (self.mask_thermal, "mask_thermal"),
        ]:
            if len(element) != len(self.images) or len(element[0]) != len(
                self.images[0]
            ):
                raise ValueError(
                    "All the elements must be of shape (1, S, ...), got "
                    + f"{len(element), len(element[0])} for {name}"
                )

    @staticmethod
    def _load_and_preprocess_images_raw(image_path_list: list[Path]) -> torch.Tensor:
        """
        This function loads images from `image_path_list` and does not crop them in
        comparison to the original VGGT `load_and_preprocess_images` function.

        :return: images from `image_path_list` as a tensor of shape (N, 3, H, W).
        """
        images_list: list[torch.Tensor] = []

        to_tensor = transforms.ToTensor()
        for image_path in image_path_list:
            img = Image.open(image_path).convert("RGB")
            img_tensor = to_tensor(img)
            images_list.append(img_tensor)

        return torch.stack(images_list)

    @classmethod
    def from_scene_path(
        cls,
        scene_path: Path,
        transforms_name: str = "transforms.json",
        crop: bool = True,
    ) -> "InferenceScene":
        """
        Creates an instance of `InferenceScene` for a scene located in `scene_path`
        """
        transforms_path = scene_path / transforms_name
        mask_thermal_list: list[bool] = []
        images_paths_list: list[Path] = []
        with open(transforms_path, "r") as f:
            transforms_data = json.load(f)
            for i, frame in enumerate(transforms_data["frames"]):
                frame_modality = list(frame.keys())
                if len(frame) != 1:
                    raise RuntimeError(
                        "For each frame only one modality must be specified, but frame "
                        + f"{i} in `scene_path` has {frame_modality}"
                    )
                frame_modality = frame_modality[0]
                if frame_modality not in ["rgb", "thermal"]:
                    raise RuntimeError(
                        "A frame modality must be either `rgb` or `thermal` but frame "
                        + f"{i} has {frame_modality}"
                    )

                mask_thermal_list.append(frame_modality == "thermal")
                images_paths_list.append(
                    scene_path / frame[frame_modality]["file_path"]
                )

        if crop:
            images_torch = load_and_preprocess_images(image_path_list=images_paths_list)
        else:
            images_torch = cls._load_and_preprocess_images_raw(
                image_path_list=images_paths_list
            )

        return cls(
            images=images_torch[None],
            images_paths=[images_paths_list],
            mask_thermal=torch.tensor(mask_thermal_list)[None],
            scene_name=scene_path.name,
        )

    def to_device(self, device: torch.device) -> "InferenceScene":
        """Places all elements on `device`"""
        self.images = self.images.to(device)
        self.mask_thermal = self.mask_thermal.to(device)
        return self
