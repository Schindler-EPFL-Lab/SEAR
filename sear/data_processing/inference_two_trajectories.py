import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch
from vggt.utils.load_fn import load_and_preprocess_images

from sear.data_processing.inference_scene import InferenceScene


@dataclass(kw_only=True)
class InferenceSceneTwoTrajectories(InferenceScene):
    """
    Represents a chunk of data containing images (thermal or rgb), their corresponding
    file paths, and thermal mask for two trajectories (rgb and thermal)
    """

    @classmethod
    def from_scene_path(cls, scene_path: Path, crop: bool = True) -> "Self":
        """
        Creates an instance of `InferenceSceneTwoTrajectories` for a scene located
        in `scene_path`
        """
        transforms_path = scene_path / "transforms.json"
        mask_thermal_list: list[bool] = []
        images_paths_list: list[Path] = []
        with open(transforms_path, "r") as f:
            transforms_data = json.load(f)
            rgb_trajectory_indices = transforms_data["rgb_trajectory"]
            thermal_trajectory_indices = transforms_data["thermal_trajectory"]

            for modality_trajectory_indices, modality_key in zip(
                [rgb_trajectory_indices, thermal_trajectory_indices],
                ["rgb", "thermal"],
            ):
                for i in modality_trajectory_indices:
                    frame = transforms_data["frames"][i]
                    mask_thermal_list.append(modality_key == "thermal")
                    images_paths_list.append(
                        scene_path / frame[modality_key]["file_path"]
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

    @classmethod
    def from_scene_path_rgb_only(cls, scene_path: Path, crop: bool = True) -> "Self":
        """
        Creates an instance of `InferenceSceneTwoTrajectories` for a scene located in
        `scene_path`, but it does not pay attention to the rgb and thermal trajectories
        split in "transforms.json" and takes **only** RGB images.
        """
        transforms_path = scene_path / "transforms.json"
        images_paths_list: list[Path] = []
        with open(transforms_path, "r") as f:
            transforms_data = json.load(f)

            for i in range(len(transforms_data["frames"])):
                frame = transforms_data["frames"][i]["rgb"]
                images_paths_list.append(scene_path / frame["file_path"])

        if crop:
            images_torch = load_and_preprocess_images(image_path_list=images_paths_list)
        else:
            images_torch = cls._load_and_preprocess_images_raw(
                image_path_list=images_paths_list
            )

        return cls(
            images=images_torch[None],
            images_paths=[images_paths_list],
            mask_thermal=torch.zeros((images_torch.shape[0]), dtype=torch.bool)[None],
            scene_name=scene_path.name,
        )

    def to_device(self, device: torch.device) -> "Self":
        """Places all elements on `device`"""
        self.images = self.images.to(device)
        self.mask_thermal = self.mask_thermal.to(device)
        return self
