import numpy as np
import numpy.typing as npt
from vggt.utils.geometry import closed_form_inverse_se3

from sear import logger
from sear.metrics.rra_rta_maa import (
    _calculate_rotation_error,
    _calculate_translation_error_degree,
    _calculate_translation_error_distance,
    _check_cameras_shapes,
)


def umeyama_alignment(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    with_scale: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """
    Computes the least squares solution parameters of an Sim(m) matrix that minimizes
    the distance between a set of registered points.  The `x`, `y` is (n, d) matrix of
    points, n = number of data points, d = dimension. The `with_scale` is set to True to
    align also the scale (default: 1.0 scale).

    The implementation is taken from the `evo` package which is based on: Umeyama,
    Shinji: Least-squares estimation of transformation parameters between two point
    patterns. IEEE PAMI, 1991

    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    x = x.T
    y = y.T

    if x.shape != y.shape:
        raise RuntimeError("Data matrices must have the same shape")

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        logger.warning(
            "Degenerate covariance rank, Umeyama alignment is not possible for x, y of "
            + f"shape {x.shape}."
        )
        return np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64), 1.0

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def align_pred_to_real(
    cameras_real_cam2world: npt.NDArray[np.float64],
    cameras_pred_cam2world: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
]:
    """
    Aligns predicted cameras `cameras_real_cam2world` to real cameras
    `cameras_pred_cam2world`.

    :return: The aligned predicted cameras, rotation matrix, translation and scale.

    :raise: RuntimeError: If shapes of `cameras_real_cam2world` and
        `cameras_pred_cam2world` are not equal to (N, 4, 4) or (N, 3, 4), or their
        length is different.
    """

    _check_cameras_shapes(
        cameras_real=cameras_real_cam2world, cameras_pred=cameras_pred_cam2world
    )

    trajectory_real = cameras_real_cam2world[:, :3, 3]
    trajectory_pred = cameras_pred_cam2world[:, :3, 3]

    rotation, translation, scale = umeyama_alignment(
        x=trajectory_pred,
        y=trajectory_real,
        with_scale=True,
    )
    transform = np.zeros((4, 4), dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform[3, 3] = 1.0

    cameras_pred_aligned_cam2world = np.zeros(
        (cameras_pred_cam2world.shape[0], 4, 4), dtype=np.float64
    )
    cameras_pred_aligned_cam2world[:, :3, 3] = cameras_pred_cam2world[:, :3, 3]
    cameras_pred_aligned_cam2world[:, :3, :3] = cameras_pred_cam2world[:, :3, :3]
    cameras_pred_aligned_cam2world[:, 3, 3] = 1.0
    cameras_pred_aligned_cam2world[:, :3, 3] *= scale

    cameras_pred_aligned_cam2world = np.matmul(
        transform, cameras_pred_aligned_cam2world
    )

    return cameras_pred_aligned_cam2world, rotation, translation, scale


def rotation_and_translation_errors(
    cameras_real_cam2world: npt.NDArray[np.float64],
    cameras_pred_aligned_cam2world: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Find rotation (degree) and translation (distance and degree) errors between
    `cameras_real_cam2world` and `cameras_pred_aligned_cam2world`.

    :return: The rotation errors in degrees, translation errors in distance, translation
        errors in degrees

    :raise: RuntimeError: If shapes of `cameras_real_cam2world` and
        `cameras_pred_aligned_cam2world` are not equal to (N, 4, 4) or (N, 3, 4), or
        their length is different.
    """
    _check_cameras_shapes(
        cameras_real=cameras_real_cam2world, cameras_pred=cameras_pred_aligned_cam2world
    )

    rotation_errors = np.array(
        [
            _calculate_rotation_error(
                rotation_matrix_real=cameras_real_cam2world[i, :3, :3],
                rotation_matrix_pred=cameras_pred_aligned_cam2world[i, :3, :3],
            )
            for i in range(cameras_real_cam2world.shape[0])
        ]
    )
    translation_errors_distance = np.array(
        [
            _calculate_translation_error_distance(
                translation_real=cameras_real_cam2world[i, :3, 3],
                translation_pred=cameras_pred_aligned_cam2world[i, :3, 3],
            )
            for i in range(cameras_real_cam2world.shape[0])
        ]
    )
    translation_errors_degree = np.array(
        [
            _calculate_translation_error_degree(
                translation_real=cameras_real_cam2world[i, :3, 3],
                translation_pred=cameras_pred_aligned_cam2world[i, :3, 3],
            )
            for i in range(cameras_real_cam2world.shape[0])
        ]
    )

    return rotation_errors, translation_errors_distance, translation_errors_degree


def calculate_cameras_ate_given_errors(
    rotation_errors: npt.NDArray[np.float64],
    translation_errors_distance: npt.NDArray[np.float64],
    translation_errors_degree: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Calculates Absolute Trajectory Error (ATE) using `rotation_errors`,
    `translation_errors_distance` and `translation_errors_degree`.

    :return: ATE for rotations, ATE for positions, and ATE for positions in degree.
    """

    return (
        np.mean(rotation_errors).item(),
        np.mean(translation_errors_distance).item(),
        np.mean(translation_errors_degree).item(),
    )


def calculate_cameras_ate(
    cameras_real_world2cam: npt.NDArray[np.float64],
    cameras_pred_world2cam: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Calculates Absolute Trajectory Error (ATE) `cameras_real_world2cam` and
    `cameras_pred_world2cam`. The function does not normalize the trajectories to fit
    into [0, 1] and assumes they are already normalized.

    :return: ATE for rotations, ATE for positions, and ATE for positions in degree.

    :raise: RuntimeError: If shapes of `cameras_real_world2cam` and
        `cameras_pred_world2cam` are not equal to (N, 4, 4) or (N, 3, 4), or their
        length is different.
    """
    _check_cameras_shapes(
        cameras_real=cameras_real_world2cam, cameras_pred=cameras_pred_world2cam
    )

    cameras_real_cam2world = closed_form_inverse_se3(cameras_real_world2cam)
    cameras_pred_cam2world = closed_form_inverse_se3(cameras_pred_world2cam)

    cameras_pred_aligned_cam2world, _, _, _ = align_pred_to_real(
        cameras_real_cam2world=cameras_real_cam2world,
        cameras_pred_cam2world=cameras_pred_cam2world,
    )

    rotation_errors, translation_errors, translation_errors_degree = (
        rotation_and_translation_errors(
            cameras_real_cam2world=cameras_real_cam2world,
            cameras_pred_aligned_cam2world=cameras_pred_aligned_cam2world,
        )
    )

    return calculate_cameras_ate_given_errors(
        rotation_errors=rotation_errors,
        translation_errors_distance=translation_errors,
        translation_errors_degree=translation_errors_degree,
    )
