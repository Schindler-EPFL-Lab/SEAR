import numpy as np
import numpy.typing as npt
from vggt.utils.geometry import closed_form_inverse_se3


def project_points(
    points: npt.NDArray[np.float32],
    extrinsic_cam2world: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    width: int,
    height: int,
) -> npt.NDArray[np.float32]:
    """
    Projects `points` onto an image plane using `extrinsic_cam2world`, and creates a
    depth map using the `intrinsic`, `width`, and `height`.

    :return: a depth map of shape (`height`, `width`).

    :raise: RuntimeError
        - if shape of extrinsic_cam2world is not (4, 4)
        - if shape of intrinsic is not (3, 3)

    """

    if extrinsic_cam2world.shape != (4, 4):
        raise RuntimeError(
            "The `extrinsic_cam2world` shape must be (4, 4) but get "
            + f"{extrinsic_cam2world.shape}"
        )
    if intrinsic.shape != (3, 3):
        raise RuntimeError(
            f"The `intrinsic` shape must be (3, 3) but get {intrinsic.shape}"
        )

    extrinsics_world2cam = closed_form_inverse_se3(extrinsic_cam2world[None])[0]

    # points in camera coordinate system
    points_cam = points @ extrinsics_world2cam[:3, :3].T + extrinsics_world2cam[:3, 3]
    points_cam = points_cam[points_cam[:, 2] > 0]  # z is positive

    # project onto the image plane
    uv = points_cam @ intrinsic.T
    depth_value = uv[:, 2]
    uv = np.round(uv[:, :2] / uv[:, 2:3]).astype(int)

    # create an image
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    dephts = np.full((height, width), fill_value=np.inf, dtype=np.float32)

    np.minimum.at(dephts, (uv[:, 1][valid], uv[:, 0][valid]), depth_value[valid])
    dephts[dephts >= 1e9] = 0.0

    return dephts
