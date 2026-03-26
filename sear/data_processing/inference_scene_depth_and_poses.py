import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
from vggt.utils.geometry import closed_form_inverse_se3

from sear.data_processing.align_part import AlignPart
from sear.data_processing.frame_info import FrameInfo
from sear.data_processing.inference_scene import InferenceScene
from sear.data_processing.single_dataset import VGGTSingleDataset
from sear.metrics.ate import align_pred_to_real


@dataclass(kw_only=True)
class InferenceSceneDepthAndPoses(InferenceScene):
    depths: torch.Tensor
    """The depth map corresponding to the images, shape (1, S, H, W)"""
    depths_paths: list[list[Path]]
    """The file paths to the depths map of shape (1, S)."""
    extrinsics_world2cam: torch.Tensor
    """
    The extrinsic matrix (4x4 or 3x4) for the RGB camera  in OpenCV world-to-camera
    convention of shape (1, S, 4, 4) or (1, S, 3, 4)
    """
    intrinsics: torch.Tensor
    """The intrinsic matrix for the cameras of shape (1, S, 3, 3)."""

    def __post_init__(self) -> None:
        """
        Validates the shapes and dimensions of after initialization.

        :raise:
            ValueError if
                - `depths` does not have 4 dimensions (1, S, H, W).
                - `extrinsics_world2cam` does not have 4 dimensions or its last two
                dimensions are not (3, 4) or (4, 4).
                - `intrinsics` does not have 4 dimensions or its last two
                dimensions are not (3, 3).
                - `depths_paths` doesnt have 2 dimensions (1, S).
                - Any of the elements (`depths`,
                `depths_paths`, `extrinsics_world2cam`, `intrinsics`) do not have the
                same (S,) shape in their first dimension.

            RuntimeError if the underlying scene is not presented in the
                `scenes_per_dataset_path`
        """
        super().__post_init__()

        if self.depths.ndim != 4 or self.depths.shape[0] != 1:
            raise ValueError(
                "`depths` must have 4 dimensions (1, S, H, W), got "
                + f"{self.depths.shape}"
            )
        if (
            self.extrinsics_world2cam.ndim != 4
            or self.extrinsics_world2cam.shape[-2:] not in [(3, 4), (4, 4)]
            or self.extrinsics_world2cam.shape[0] != 1
        ):
            raise ValueError(
                "`extrinsics_world2cam` must have 4 dimensions (1, S, 4, 4) or "
                + f"(1, S, 3, 4), got {self.extrinsics_world2cam.shape}"
            )
        if (
            self.intrinsics.ndim != 4
            or self.intrinsics.shape[-2:] != (3, 3)
            or self.intrinsics.shape[0] != 1
        ):
            raise ValueError(
                "`intrinsics` must have 4 dimensions (1, S, 3, 3), got "
                + f"{self.intrinsics.shape}"
            )

        for element, name in [
            (self.depths, "depths"),
            (self.depths_paths, "depths_paths"),
            (self.extrinsics_world2cam, "extrinsics_world2cam"),
            (self.intrinsics, "intrinsics"),
        ]:
            if len(element) != len(self.images) or len(element[0]) != len(
                self.images[0]
            ):
                raise ValueError(
                    "All the elements must be of shape (1, S, ...), got "
                    + f"{len(element), len(element[0])} for {name}"
                )

    @classmethod
    def from_scene_path(
        cls, scene_path: Path, transforms_name: str = "transforms.json"
    ) -> "Self":
        """
        Creates an instance of `InferenceSceneDepthAndPoses` for a scene located in
        `scene_path`
        """

        inner_class = InferenceScene.from_scene_path(
            scene_path, transforms_name=transforms_name, crop=False
        )

        transforms_path = scene_path / transforms_name
        depths_list: list[torch.Tensor] = []
        depths_paths_list: list[Path] = []
        extrinsics_world2cam_list: list[torch.Tensor] = []
        intrinsics_list: list[torch.Tensor] = []

        with open(transforms_path, "r") as f:
            transforms_data = json.load(f)
            for i, frame in enumerate(transforms_data["frames"]):
                modality_key = list(frame.keys())
                if len(frame) != 1:
                    raise RuntimeError(
                        "For each frame only one modality must be specified, but frame "
                        + f"{i} in `scene_path` has {modality_key}"
                    )
                modality_key = modality_key[0]
                if modality_key not in ["rgb", "thermal"]:
                    raise RuntimeError(
                        "A frame modality must be either `rgb` or `thermal` but frame "
                        + f"{i} has {modality_key}"
                    )

                depth_path = scene_path / frame[modality_key]["depth_file_path"]
                depth = VGGTSingleDataset._load_depth(
                    depth_filepath=depth_path, dtype=torch.float32
                )
                depths_paths_list.append(depth_path)
                depths_list.append(depth)

                extrinsics_world2cam, intrinsic = FrameInfo.dict_to_matrices(
                    frame_dict=frame[modality_key]
                )

                extrinsics_world2cam_list.append(extrinsics_world2cam)
                intrinsics_list.append(intrinsic)

        return cls(
            images=inner_class.images,
            images_paths=inner_class.images_paths,
            mask_thermal=inner_class.mask_thermal,
            scene_name=inner_class.scene_name,
            depths=torch.stack(depths_list)[None],
            depths_paths=[depths_paths_list],
            extrinsics_world2cam=torch.stack(extrinsics_world2cam_list)[None],
            intrinsics=torch.stack(intrinsics_list)[None],
        )

    @classmethod
    def from_scene_path_aligned(
        cls,
        scene_path: Path,
        transforms_name: str = "transforms.json",
        align: AlignPart = AlignPart.RGB,
    ) -> "Self":
        """
        Creates an instance of `InferenceSceneDepthAndPoses` for a scene located in
        `scene_path` and makes extrinsics and depths aligned with the ground truth
        poses.

        :raise: FileNotFoundError if scene_path/transforms_ground_truth.json does not
            exist.
        """

        inner_class = cls.from_scene_path(scene_path, transforms_name=transforms_name)
        transforms_ground_truth_path = scene_path / "transforms_ground_truth.json"
        if not transforms_ground_truth_path.exists():
            raise FileNotFoundError(
                "The scene_path/transforms_ground_truth.json "
                + f"{transforms_ground_truth_path} does not exist."
            )

        extrinsics_real_world2cam_list: list[torch.Tensor] = []

        with open(transforms_ground_truth_path, "r") as f:
            transforms_data = json.load(f)
            for i, frame in enumerate(transforms_data["frames"]):
                modality_key = list(frame.keys())
                if len(frame) != 1:
                    raise RuntimeError(
                        "For each frame only one modality must be specified, but frame "
                        + f"{i} in `scene_path` has {modality_key}"
                    )
                modality_key = modality_key[0]
                if modality_key not in ["rgb", "thermal"]:
                    raise RuntimeError(
                        "A frame modality must be either `rgb` or `thermal` but frame "
                        + f"{i} has {modality_key}"
                    )

                extrinsic_real_world2cam, _ = FrameInfo.dict_to_matrices(
                    frame_dict=frame[modality_key]
                )

                extrinsics_real_world2cam_list.append(extrinsic_real_world2cam)

        extrinsics_real_world2cam = torch.stack(
            extrinsics_real_world2cam_list
        )  # (S, 4, 4)

        extrinsics_real_cam2world = closed_form_inverse_se3(extrinsics_real_world2cam)
        extrinsics_pred_cam2world = closed_form_inverse_se3(
            inner_class.extrinsics_world2cam[0]
        )

        # find the alignment
        if align is AlignPart.RGB:
            align_mask = ~inner_class.mask_thermal[0]
        if align is AlignPart.THERMAL:
            align_mask = inner_class.mask_thermal[0]
        if align is AlignPart.ALL:
            align_mask = torch.ones_like(inner_class.mask_thermal[0])
        _, rotation, translation, scale = align_pred_to_real(
            cameras_real_cam2world=extrinsics_real_cam2world[align_mask].numpy(),
            cameras_pred_cam2world=extrinsics_pred_cam2world[align_mask].numpy(),
        )

        transform = np.zeros((4, 4), dtype=np.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        transform[3, 3] = 1.0

        cameras_pred_aligned_cam2world = np.zeros(
            (extrinsics_pred_cam2world.shape[0], 4, 4), dtype=np.float32
        )
        cameras_pred_aligned_cam2world[:, :3, 3] = extrinsics_pred_cam2world[:, :3, 3]
        cameras_pred_aligned_cam2world[:, :3, :3] = extrinsics_pred_cam2world[:, :3, :3]
        cameras_pred_aligned_cam2world[:, 3, 3] = 1.0
        cameras_pred_aligned_cam2world[:, :3, 3] *= scale
        cameras_pred_aligned_cam2world = np.matmul(
            transform, cameras_pred_aligned_cam2world
        )

        cameras_pred_aligned_world2cam = closed_form_inverse_se3(
            cameras_pred_aligned_cam2world
        )

        depths_pred = inner_class.depths * scale

        return cls(
            images=inner_class.images,
            images_paths=inner_class.images_paths,
            mask_thermal=inner_class.mask_thermal,
            scene_name=inner_class.scene_name,
            depths=depths_pred,
            depths_paths=inner_class.depths_paths,
            extrinsics_world2cam=torch.from_numpy(cameras_pred_aligned_world2cam)[
                None
            ].float(),
            intrinsics=inner_class.intrinsics,
        )

    def to_device(self, device: torch.device) -> "Self":
        """Places all elements on `device`"""
        InferenceScene.to_device(self, device=device)

        self.depths = self.depths.to(device)
        self.extrinsics_world2cam = self.extrinsics_world2cam.to(device)
        self.intrinsics = self.intrinsics.to(device)

        return self
