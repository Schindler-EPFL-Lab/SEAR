import numpy as np
import numpy.typing as npt
import pyvista as pv
import torch
from vggt.utils.geometry import closed_form_inverse_se3


def _plot_cameras(points: npt.NDArray[np.float64]) -> pv.PolyData:
    """
    A helper function that creates a camera representation via set of cuts from `points`
    (corners of the pyramid).

    :return: segment-like representation of a camera.
    """
    segments = []
    for cam_id in range(points.shape[0]):
        shift = cam_id * 5
        for i in range(1, 5):
            segments.append([2, 0 + shift, i + shift])
        for i, j in [(1, 2), (2, 3), (3, 4), (4, 1)]:
            segments.append([2, i + shift, j + shift])

    lines = np.array(segments, dtype=np.int64).ravel()
    poly = pv.PolyData(points.reshape(-1, 3))
    poly.lines = lines
    return poly


def plot_cameras(
    pl,
    extrinsics_world2cam: torch.Tensor,
    intrinsics: torch.Tensor,
    images: torch.Tensor,
    mask_thermal: torch.Tensor,
    z_coeff: float = 0.05,
    bbox_size_max: float | None = None,
):
    """
    Visualizes RGB and thermal cameras in a 3D plot using their extrinsic and intrinsic
    parameters. The `pl` is a pyvista plotting object. The `extrinsics_world2cam` is a
    tensor of shape (N, 4, 4) representing the extrinsics. `intrinsics` is a tensor of
    shape (N, 3, 3) representing the intrinsic matrices of the cameras. The `images` is
    a tensor of shape (N, C, H, W) representing the images. The The `mask_thermal` is a
    boolean tensor which cameras are thermal cameras (True for thermal, False for RGB).
    The `z_coeff` is a scaling coefficient for the depth (z-axis) of the camera
    frustums. The `bbox_size_max` is the maximum size of the bounding box enclosing the
    camera translations. If None, it is computed from the translations.

    :return: a tuple containing:
        - rgb_cameras: The rendered RGB camera frustums as a mesh object.
        - thermal_cameras: The rendered thermal camera frustums as a mesh object.
    """

    extrinsics_cam2world = closed_form_inverse_se3(
        extrinsics_world2cam[0].detach().cpu().numpy()
    )
    intrinsics = intrinsics[0].detach().cpu().numpy()
    translations = extrinsics_cam2world[:, :3, 3]
    bbox_size_max: float = (translations.max(axis=0) - translations.min(axis=0)).max()
    z = z_coeff * bbox_size_max
    h, w = images.shape[-2:]  # (1, S, 3, H, W)

    corners_pix = np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ],
        dtype=np.float64,
    )  # (4, 2)

    # (1, 4) - (N, 1)
    x = (corners_pix[None, :, 0] - intrinsics[:, 0:1, 2]) / intrinsics[
        :, 0:1, 0
    ]  # (N, 4)
    y = (corners_pix[None, :, 1] - intrinsics[:, 1:2, 2]) / intrinsics[
        :, 1:2, 1
    ]  # (N, 4)

    # (N, 4, 4)
    corners_cam = np.stack(
        [x * z, y * z, np.full_like(x, z), np.full_like(x, 1.0)], axis=2
    )
    # (N, 4, 4), (N, 4, 4) -> (N, 4, 4)
    corners_world = np.matmul(corners_cam, extrinsics_cam2world.transpose([0, 2, 1]))
    # (N, 4, 3) -> (N, 4, 3)
    corners_world = corners_world[:, :, :3] / (corners_world[:, :, 3:4] + 1e-6)
    # (N, 4, 3) -> (N, 5, 3)
    corners_world = np.concatenate([translations[:, None, :], corners_world], axis=1)

    rgb_cameras_data = _plot_cameras(
        corners_world[~mask_thermal[0].cpu().numpy()],
    )
    rgb_cameras = pl.add_mesh(
        rgb_cameras_data, line_width=3, color="#F90000", point_size=0
    )

    thermal_cameras_data = _plot_cameras(
        corners_world[mask_thermal[0].cpu().numpy()],
    )
    thermal_cameras = pl.add_mesh(
        thermal_cameras_data, line_width=3, color="#000DFF", point_size=0
    )

    return rgb_cameras, thermal_cameras
