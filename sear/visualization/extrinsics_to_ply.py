from pathlib import Path

import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch
from vggt.utils.geometry import closed_form_inverse_se3


def camera_extrinsics_ply(
    extrinsics_world2cam: npt.NDArray[np.float32] | torch.Tensor,
    output_path: Path,
    bbox_size_max: float | None = None,
    color: tuple[float, float, float] = (0.1, 0.4, 0.9),
    height_ratio: float = 0.1,
    side_ratio: float = 0.05,
) -> None:
    """
    Creates camera pyramids for a sequence of extrinsic matrices `extrinsics_cam2world`
    to a .ply file `output_path` using Open3D. The pyramid shape is defined by the
    `height_ratio` and `side_ratio`, which determine its height and base dimensions
    relative to the longest side `bbox_size_max` of the sequence bounding box. If the
    `bbox_size_max` is not specified then it is calculated from the provided
    `extrinsics_world2cam`.
    """

    if torch.is_tensor(extrinsics_world2cam):
        extrinsics_world2cam = extrinsics_world2cam.detach().cpu().numpy()

    extrinsics_cam2world = closed_form_inverse_se3(extrinsics_world2cam)

    if bbox_size_max is None:
        translations = extrinsics_cam2world[:, :3, 3]
        bbox_size_max = (translations.max(axis=0) - translations.min(axis=0)).max()

    side = bbox_size_max * side_ratio
    height = bbox_size_max * height_ratio

    pyramid_camera = np.array(
        [
            [0.0, 0.0, 0.0],
            [-side / 2, side / 2, height],
            [-side / 2, -side / 2, height],
            [side / 2, side / 2, height],
            [side / 2, -side / 2, height],
        ],
        dtype=np.float32,
    )  # (5, 3)

    edges = [(i, j) for i in range(5) for j in range(i + 1, 5)]

    all_points: list[npt.NDArray[np.float32]] = []
    all_lines: list[tuple[int, int]] = []
    all_colors: list[tuple[float, float, float]] = []

    point_offset = 0

    for camera_index in range(extrinsics_cam2world.shape[0]):
        extrinsic = extrinsics_cam2world[camera_index]
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]

        pyramid_world = pyramid_camera @ rotation.T + translation

        all_points.append(pyramid_world)

        for i, j in edges:
            all_lines.append((point_offset + i, point_offset + j))
            all_colors.append(color)

        point_offset += pyramid_world.shape[0]

    points = np.concatenate(all_points, axis=0)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(np.array(all_lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.array(all_colors, dtype=np.float32))
    o3d.io.write_line_set(str(output_path), line_set)
