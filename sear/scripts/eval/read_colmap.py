import json
import shutil
from pathlib import Path

import numpy as np
import torch
from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat
from vggt.utils.geometry import closed_form_inverse_se3

from sear.data_processing.frame_info import FrameInfo


def read_images(
    images_path: Path,
    colmap_intrinsics: dict[int, torch.Tensor],
    colmap_width_height: dict[int, tuple[int, int]],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reads camera extrinsics from colmap processed images.txt located at images_path.
    Also defines extrinsics in case of multiple cameras from `colmap_intrinsics` and
    shapes from `width_height_colmap`.

    :return: names of successfully processed images, extrinsics in opencv
        world-to-camera format, intrinsics and the resolution of (width, height).
    """

    with open(images_path, "r") as f:
        lines = f.read().splitlines()
    # filter only frames without #
    lines = [line for line in lines if len(line) > 0 and line[0] != "#"]
    lines = lines[::2]
    extrinsics_world2cam = torch.zeros((len(lines), 4, 4), dtype=torch.float32)
    extrinsics_world2cam[:, 3, 3] = 1.0
    intrinsics = torch.empty((len(lines), 3, 3), dtype=torch.float32)
    resolution = torch.empty((len(lines), 2), dtype=torch.int64)

    frames_names: list[str] = []
    for ext_idx, line in enumerate(lines):
        _, qw, qx, qy, qz, tx, ty, tz, sensor_id, frame_name = line.strip().split(" ")
        qvec = np.array([float(qw), float(qx), float(qy), float(qz)])
        rotation_matrix = qvec2rotmat(qvec)
        extrinsics_world2cam[ext_idx, :3, :3] = torch.from_numpy(rotation_matrix)
        extrinsics_world2cam[ext_idx, :3, 3] = torch.tensor(
            [float(tx), float(ty), float(tz)]
        )
        intrinsics[ext_idx] = colmap_intrinsics[int(sensor_id)]
        frames_names.append(frame_name)
        resolution[ext_idx] = torch.tensor(colmap_width_height[int(sensor_id)])
    return frames_names, extrinsics_world2cam, intrinsics, resolution


def read_cameras(
    camera_path: Path,
) -> tuple[dict[int, torch.Tensor], dict[int, tuple[int, int]]]:
    """
    Reads camera intrinsics and resolution from colmap processed cameras.txt located at
    camera_path.

    :return: camera intrinsics mapping of [sensor_id, intrinsic matrix] and resolution
        mapping of [sensor_id, (width, height)].
    """
    with open(camera_path, "r") as f:
        lines = f.read().splitlines()
    # filter only frames without #
    lines = [line for line in lines if len(line) > 0 and line[0] != "#"]
    colmap_intrinsics: dict[int, torch.Tensor] = {}
    colmap_width_height: dict[int, tuple[int, int]] = {}
    for line in lines:
        camera_id, _, width, height, focal, center_x, center_y, _ = line.strip().split(
            " "
        )
        colmap_intrinsics[int(camera_id)] = torch.Tensor(
            [
                [float(focal), 0.0, float(center_x)],
                [0.0, float(focal), float(center_y)],
                [0.0, 0.0, 1.0],
            ]
        )
        colmap_width_height[int(camera_id)] = int(width), int(height)
    return colmap_intrinsics, colmap_width_height


def find_best_reconstruction(colmap_reconstruction_dir: Path) -> Path | None:
    """
    Find reconstruction containing the most number of reconstructed images from
    `colmap_reconstruction_dir`.

    :return: the directory containing the largest number of reconstructed frames.
    """

    reconstruction_dirs = colmap_reconstruction_dir.iterdir()
    best_reconstruction_dir = None
    best_num_frames = 0
    try:
        for reconstruction_dir in reconstruction_dirs:
            with open(reconstruction_dir / "images.txt", "r") as f:
                lines = f.read().splitlines()
            # filter only frames
            lines = [line for line in lines if len(line) > 0 and line[0] != "#"]
            lines = lines[::2]

            if best_reconstruction_dir is None or best_num_frames < len(lines):
                best_reconstruction_dir = reconstruction_dir
                best_num_frames = len(lines)
    except Exception:
        return None

    return best_reconstruction_dir


def colmap_to_json(
    recon_dir: Path,
    output_dir: Path,
    image_rename_map: dict[str, str] | None,
    ply_filename: str = "sparse_pc.ply",
) -> int:
    """
    Converts colmap reconstructions from `recon_dir` which must contain "cameras.txt"
    and "images.txt" in nerfstudio-like manner. Stores results (transforms.json and ply
    file with name `ply_filename`) in `output_dir`. The `image_rename_map` is a mapping
    between new images names and the old ones.

    :return: number of reconstructed frames.
    """
    # read cameras
    colmap_intrinsics, colmap_width_height = read_cameras(recon_dir / "cameras.txt")

    # read frames
    frames_names, extrinsics_world2cam, intrinsics, resolution = read_images(
        images_path=recon_dir / "images.txt",
        colmap_intrinsics=colmap_intrinsics,
        colmap_width_height=colmap_width_height,
    )
    extrinsic_matrix_cam2world = closed_form_inverse_se3(extrinsics_world2cam)

    # save transforms.json
    transforms: dict[str, dict[str, int | float | str | list[list[float]]]] = {}
    transforms["frames"] = []

    if image_rename_map is None:
        image_rename_map = {name: f"images/{name}" for name in frames_names}

    for ext_idx in range(extrinsic_matrix_cam2world.shape[0]):
        frame_info = FrameInfo(
            extrinsic_matrix_world2cam=extrinsic_matrix_cam2world[ext_idx],
            intrinsic_matrix=intrinsics[ext_idx],
            width=int(resolution[ext_idx, 0].item()),
            height=int(resolution[ext_idx, 1].item()),
            image_path=Path(image_rename_map[frames_names[ext_idx]]),
            depth_path=Path("unknown.png"),
        )
        transforms["frames"].append(frame_info.to_dict())

    with open(output_dir / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)

    # copy rec.ply
    shutil.copy(recon_dir / "rec.ply", output_dir / ply_filename)

    return len(extrinsic_matrix_cam2world)
