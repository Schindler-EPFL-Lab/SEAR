import json
from pathlib import Path

import numpy as np
import torch
import torchvision

from sear.data_processing.frame_info import FrameInfo


def store_results_no_ground_truth(
    images_paths: list[Path],
    images: torch.Tensor,
    depths: torch.Tensor,
    thermal_mask: torch.Tensor,
    extrinsics_world2cam: torch.Tensor,
    intrinsics: torch.Tensor,
    job_name: str,
    method_name: str,
    ratio_reconstructed: float,
    duration: float,
    output_folder: Path = Path("./outputs/"),
) -> None:
    """
    Stores camera poses and depths predicted by any method in a structured format. The
    `images_paths` contains paths to images. The `images` is a tensor of shape (N, H, W,
    3). The `depths` is tensor of depth maps with shape (N, H, W). The `thermal_mask` is
    a mask indicating that the modality is thermal. The `extrinsics_world2cam` is
    predicted camera extrinsic matrices of shape (N, 4, 4) or (N, 3, 4). The
    `intrinsics` is predicted camera intrinsics of shape (N, 3, 3). The `method_name` is
    method used. The `output_folder` is directory where to store the results. The
    `job_name` defines the name of the job created the results to save. The `duration`
    defines how long did it take to process the scene. The `ratio_reconstructed` is the
    ratio of reconstructed poses or pairs of poses.

    :raise:
        RuntimeError: If input lists/tensors do not have matching lengths.
    """

    if not (
        len(images_paths) == len(images)
        and len(images) == len(depths)
        and len(depths) == len(thermal_mask)
        and len(thermal_mask) == len(extrinsics_world2cam)
        and len(extrinsics_world2cam) == len(intrinsics)
        and len(intrinsics)
    ):
        raise RuntimeError(
            f"The input lenghts must natch, but got"
            f"{len(images_paths)}, {len(images)}, {len(depths)}, {len(thermal_mask)}, "
            + f"{len(extrinsics_world2cam)}, {len(intrinsics)}, "
            + "`images_path`, `images`, `depths`, "
            + "`thermal_mask`, `extrinsics_world2cam`, `intrinsics`, "
            + "respectively."
        )

    if images.ndim != 4 or images.shape[3] != 3:
        raise RuntimeError(
            f"`images` tensor must have shape (N, H, W, 3), got {images.shape}"
        )
    if depths.ndim != 3:
        raise RuntimeError(
            f"`depths` tensor must have shape (N, H, W), got {depths.shape}"
        )
    if images.shape[1:3] != depths.shape[1:3]:
        raise RuntimeError(
            f"Resolution mismatch: `images` have shape {images.shape[1:3]}, "
            f"but `depths` have shape {depths.shape[1:3]}"
        )

    if extrinsics_world2cam.ndim != 3 or extrinsics_world2cam.shape[1:] not in [
        (3, 4),
        (4, 4),
    ]:
        raise RuntimeError(
            "The `extrinsics_world2cam` must have shape (N, 4, 4) or (N, 3, 4), got "
            + f"{extrinsics_world2cam.shape}"
        )

    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise RuntimeError(
            f"`intrinsics` must have shape (N, 3, 3), got {intrinsics.shape}"
        )

    images_folder = Path("images")
    (output_folder / images_folder).mkdir(exist_ok=True, parents=True)
    depths_folder = Path("depths")
    (output_folder / depths_folder).mkdir(exist_ok=True, parents=True)

    transforms: dict[
        str,
        str | dict[str, list[dict[str, int | float | list[list[float]]]]] | float | int,
    ] = {}

    transforms["type"] = method_name
    transforms["job_name"] = job_name
    transforms["duration"] = duration
    transforms["ratio_reconstructed"] = ratio_reconstructed
    transforms["frames"]: dict[  # type: ignore
        str, list[dict[str, int | float | list[list[float]] | list[float]]]
    ] = []

    for i in range(len(images_paths)):
        image_name = images_paths[i].name
        relative_image_path = images_folder / image_name
        depth_name = images_paths[i].stem + ".npy"
        relative_depth_path = depths_folder / depth_name

        torchvision.utils.save_image(
            images[i].permute(2, 0, 1), output_folder / relative_image_path
        )
        np.save(output_folder / relative_depth_path, depths[i].numpy())

        width = images[i].shape[1]
        height = images[i].shape[0]

        frame = FrameInfo(
            extrinsic_matrix_world2cam=extrinsics_world2cam[i],
            intrinsic_matrix=intrinsics[i],
            width=width,
            height=height,
            image_path=relative_image_path,
            depth_path=relative_depth_path,
        ).to_dict()

        modality = "thermal" if thermal_mask[i] else "rgb"
        transforms["frames"].append({modality: frame})

    with open(output_folder / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)
