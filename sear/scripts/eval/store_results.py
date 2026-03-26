import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision

from sear.data_processing.frame_info import FrameInfo


def store_results(
    images_paths: list[Path],
    images: torch.Tensor,
    depths: torch.Tensor,
    thermal_mask: torch.Tensor,
    extrinsics_world2cam: torch.Tensor,
    intrinsics: torch.Tensor,
    ground_truth_extrinsics_world2cam: torch.Tensor,
    ground_truth_intrinsics: torch.Tensor,
    ground_truth_depth: torch.Tensor,
    job_name: str,
    method_name: str,
    ratio_reconstructed: float,
    duration: float,
    output_folder: Path = Path("./outputs/"),
) -> None:
    """
    Stores camera poses and depths predicted by DUST3R or MAST3R or any other method in
    a structured format. The `file_paths` contains paths to images. The `images` is a
    tensor of shape (N, H, W, 3). The `depths` is tensor of depth maps with shape (N, H,
    W). The `thermal_mask` is a mask indicating that the modality is thermal. The
    `extrinsics_world2cam` is predicted camera extrinsic matrices of shape (N, 4, 4) or
    (N, 3, 4). The `intrinsics` is predicted camera intrinsics of shape (N, 3, 3). The
    `ground_truth_extrinsics_world2cam` is ground truth camera extrinsic matrices of
    shape (N, 4, 4) or (N, 3, 4). The `ground_truth_intrinsics` is ground truth camera
    intrinsic matrices of shape (N, 3, 3). The `ground_truth_depth` is ground truth
    dephs maps of shape (N, H, W). The `method_name` is method used. The `output_folder`
    is directory where to store the results. The `job_name` defines the name of the job
    created the results to save. The `duration` defines how long did it take to process
    the scene. The `ratio_reconstructed` is the ratio of reconstructed poses or pairs of
    poses.

    :raise:
        RuntimeError: If input lists/tensors do not have matching lengths.
    """

    if not (
        len(images_paths) == len(images)
        and len(images) == len(depths)
        and len(depths) == len(thermal_mask)
        and len(thermal_mask) == len(extrinsics_world2cam)
        and len(extrinsics_world2cam) == len(intrinsics)
        and len(intrinsics) == len(ground_truth_extrinsics_world2cam)
        and len(ground_truth_extrinsics_world2cam) == len(ground_truth_depth)
        and len(ground_truth_extrinsics_world2cam) == len(ground_truth_intrinsics)
    ):
        raise RuntimeError(
            f"The input lenghts must natch, but got"
            f"{len(images_paths)}, {len(images)}, {len(depths)}, {len(thermal_mask)}, "
            + f"{len(extrinsics_world2cam)}, {len(intrinsics)}, "
            + f"{len(ground_truth_extrinsics_world2cam)}, "
            + f"{len(ground_truth_intrinsics)}, {len(ground_truth_depth)} for "
            + "`images_path`, `images`, `depths`, "
            + "`thermal_mask`, `extrinsics_world2cam`, `intrinsics`, "
            + "`ground_truth_extrinsics_world2cam`, `ground_truth_intrinsics` and "
            + "`ground_truth_depth` respectively."
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
    if (
        ground_truth_extrinsics_world2cam.ndim != 3
        or ground_truth_extrinsics_world2cam.shape[1:]
        not in [
            (3, 4),
            (4, 4),
        ]
    ):
        raise RuntimeError(
            "`ground_truth_extrinsics_world2cam` must have shape (N, 4, 4) or (N, 3, 4)"
            + f", got {ground_truth_extrinsics_world2cam.shape}"
        )
    if ground_truth_intrinsics.ndim != 3 or ground_truth_intrinsics.shape[1:] != (3, 3):
        raise RuntimeError(
            "`ground_truth_intrinsics` must have shape (N, 3, 3), got "
            + f"{ground_truth_intrinsics.shape}"
        )
    if ground_truth_depth.ndim != 3:
        raise RuntimeError(
            "`ground_truth_depth` tensor must have shape (N, H, W), got "
            + f"{ground_truth_depth.shape}"
        )

    images_folder = Path("images")
    (output_folder / images_folder).mkdir(exist_ok=True, parents=True)
    depths_folder = Path("depths")
    (output_folder / depths_folder).mkdir(exist_ok=True, parents=True)
    ground_truth_depths_folder = Path("real_depths")
    (output_folder / ground_truth_depths_folder).mkdir(exist_ok=True, parents=True)

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
    ] = []  # type: ignore

    transforms_ground_truth = deepcopy(transforms)

    for i in range(len(images_paths)):
        image_name = images_paths[i].name
        relative_image_path = images_folder / image_name
        depth_name = images_paths[i].stem + ".npy"
        relative_depth_path = depths_folder / depth_name
        relative_ground_truth_depth_path = ground_truth_depths_folder / depth_name

        torchvision.utils.save_image(
            images[i].permute(2, 0, 1), output_folder / relative_image_path
        )
        np.save(output_folder / relative_depth_path, depths[i].numpy())
        np.save(
            output_folder / relative_ground_truth_depth_path,
            ground_truth_depth[i].numpy(),
        )

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

        ground_truth_frame = FrameInfo(
            extrinsic_matrix_world2cam=ground_truth_extrinsics_world2cam[i],
            intrinsic_matrix=ground_truth_intrinsics[i],
            width=ground_truth_depth[i].shape[1],
            height=ground_truth_depth[i].shape[0],
            image_path=relative_image_path,
            depth_path=relative_ground_truth_depth_path,
        ).to_dict()
        transforms_ground_truth["frames"].append({modality: ground_truth_frame})

    with open(output_folder / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)

    with open(output_folder / "transforms_ground_truth.json", "w") as f:
        json.dump(transforms_ground_truth, f, indent=4)
