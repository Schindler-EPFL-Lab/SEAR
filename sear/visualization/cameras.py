import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from mpl_toolkits.mplot3d import Axes3D
from vggt.utils.geometry import closed_form_inverse_se3


def _draw_one_sequence_extrinsics(
    extrinsics_cam2world: npt.NDArray[np.float32],
    color: str,
    ax: Axes3D,
    bbox_size_max: float,
    height_ratio: float = 0.1,
    side_ratio: float = 0.05,
    alpha: float = 0.5,
) -> None:
    """
    Draws a camera pyramid for each extrinsic matrix in the sequence `extrinsics` (using
    the OpenCV camera-to-world convention) onto the given matplotlib 3D axes `ax`. The
    pyramid is rendered with the specified `color` and `alpha` transparency. Its shape
    is defined by the pyramid `height_ratio` and the base `side_ratio`, which determine
    its height and base dimensions relative to the longest side of the bounding box.
    """

    side = bbox_size_max * side_ratio
    height = bbox_size_max * height_ratio

    pyramid_camera = np.array(
        [
            [0.0, 0.0, 0.0],
            [-side / 2, side / 2, height],
            [-side / 2, -side / 2, height],
            [side / 2, side / 2, height],
            [side / 2, -side / 2, height],
        ]
    )  # (5, 3)
    for camera_index in range(extrinsics_cam2world.shape[0]):
        extrinsic = extrinsics_cam2world[camera_index]
        rotation = extrinsic[:3, :3]  # (3, 3)
        translation = extrinsic[:3, 3]  # (3,)
        pyramid_world = pyramid_camera @ rotation.T + translation

        for i in range(5):
            for j in range(i, 5):
                ax.plot(
                    [pyramid_world[i, 0], pyramid_world[j, 0]],
                    [pyramid_world[i, 1], pyramid_world[j, 1]],
                    [pyramid_world[i, 2], pyramid_world[j, 2]],
                    color=color,
                    alpha=alpha,
                )


def draw_camera_extrinsics(
    extrinsics_real_world2cam: npt.NDArray[np.float32] | torch.Tensor,
    extrinsics_pred_world2cam: npt.NDArray[np.float32] | torch.Tensor,
    color_real: str = "blue",
    color_pred: str = "red",
    title: str = "",
    alpha: float = 0.2,
    equal_axes: bool = True,
) -> "plt.Figure":
    """
    Draws camera pyramids for sequences of real and predicted extrinsics,
    `extrinsics_real` and `extrinsics_pred` (using the OpenCV camera-to-world
    convention) using matplotlib. Pyramids corresponding to real extrinsics are drawn in
    `color_real`, while those for predicted extrinsics use `color_pred`. All pyramids
    are rendered with the opacity `alpha`. The parameter `equal_axes` controls whether
    the plot enforces equal axis limits.

    :returns matplotlib Figure of the plot.
    """

    if torch.is_tensor(extrinsics_real_world2cam):
        extrinsics_real_world2cam = extrinsics_real_world2cam.numpy()
    if torch.is_tensor(extrinsics_pred_world2cam):
        extrinsics_pred_world2cam = extrinsics_pred_world2cam.numpy()

    extrinsics_real_cam2world = closed_form_inverse_se3(extrinsics_real_world2cam)
    extrinsics_pred_cam2world = closed_form_inverse_se3(extrinsics_pred_world2cam)

    translations = np.concatenate(
        [extrinsics_real_cam2world[:, :3, 3], extrinsics_pred_cam2world[:, :3, 3]],
        axis=0,
    )
    bbox_size_max = (translations.max(axis=0) - translations.min(axis=0)).max()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    _draw_one_sequence_extrinsics(
        extrinsics_cam2world=extrinsics_real_cam2world,
        color=color_real,
        ax=ax,
        alpha=alpha,
        bbox_size_max=bbox_size_max,
    )
    _draw_one_sequence_extrinsics(
        extrinsics_cam2world=extrinsics_pred_cam2world,
        color=color_pred,
        ax=ax,
        alpha=alpha,
        bbox_size_max=bbox_size_max,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)

    if equal_axes:
        limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        max_range = np.diff(limits, axis=1).max()
        mid_x, mid_y, mid_z = limits.mean(axis=1).tolist()
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    return fig


def _draw_one_sequence_trajectory(
    extrinsics_cam2world: npt.NDArray[np.float32],
    color: str,
    ax: Axes3D,
    alpha: float = 0.5,
) -> None:
    """
    Draws the trajectory extrinsic matrices in the sequence `extrinsics` (using
    the OpenCV camera-to-world convention) onto the given matplotlib 3D axes `ax`. The
    sequence is rendered with the specified `color` and `alpha` transparency.
    """

    for camera_index in range(1, extrinsics_cam2world.shape[0]):
        previous_translation = extrinsics_cam2world[camera_index - 1, :3, 3]
        current_translation = extrinsics_cam2world[camera_index, :3, 3]

        ax.plot(
            [previous_translation[0], current_translation[0]],
            [previous_translation[1], current_translation[1]],
            [previous_translation[2], current_translation[2]],
            color=color,
            alpha=alpha,
        )


def draw_camera_trajectories(
    extrinsics_real_world2cam: npt.NDArray[np.float32] | torch.Tensor,
    extrinsics_pred_world2cam: npt.NDArray[np.float32] | torch.Tensor,
    color_real: str = "blue",
    color_pred: str = "red",
    title: str = "",
    alpha: float = 0.5,
    equal_axes: bool = True,
) -> "plt.Figure":
    """
    Draws camera trajectories for sequences of real and predicted extrinsics,
    `extrinsics_real` and `extrinsics_pred` (using the OpenCV camera-to-world
    convention) using matplotlib. Sequences corresponding to real extrinsics are drawn
    in `color_real`, while those for predicted extrinsics use `color_pred`. All
    sequences are rendered with the opacity `alpha`. The parameter `equal_axes` controls
    whether the plot enforces equal axis limits.

    :returns matplotlib Figure of the plot.
    """

    if torch.is_tensor(extrinsics_real_world2cam):
        extrinsics_real_world2cam = extrinsics_real_world2cam.numpy()
    if torch.is_tensor(extrinsics_pred_world2cam):
        extrinsics_pred_world2cam = extrinsics_pred_world2cam.numpy()

    extrinsics_real_cam2world = closed_form_inverse_se3(extrinsics_real_world2cam)
    extrinsics_pred_cam2world = closed_form_inverse_se3(extrinsics_pred_world2cam)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    _draw_one_sequence_trajectory(
        extrinsics_cam2world=extrinsics_real_cam2world,
        color=color_real,
        ax=ax,
        alpha=alpha,
    )
    _draw_one_sequence_trajectory(
        extrinsics_cam2world=extrinsics_pred_cam2world,
        color=color_pred,
        ax=ax,
        alpha=alpha,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)

    if equal_axes:
        limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        max_range = np.diff(limits, axis=1).max()
        mid_x, mid_y, mid_z = limits.mean(axis=1).tolist()
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    return fig
