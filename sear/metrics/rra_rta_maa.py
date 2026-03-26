import cv2
import numpy as np
import numpy.typing as npt
from vggt.utils.geometry import closed_form_inverse_se3


def _check_cameras_shapes(
    cameras_real: npt.NDArray[np.float64],
    cameras_pred: npt.NDArray[np.float64],
) -> None:
    """
    Checks that shape of `cameras_real` and `cameras_pred` are correct.

    :raises RuntimeError: If shapes of `cameras_real` and `cameras_pred` are not equal
        to (N, 4, 4) or (N, 3, 4), or their length is different.
    """
    if (
        cameras_real.ndim != 3
        or cameras_real.shape[1] > 4
        or cameras_real.shape[1] < 3
        or cameras_real.shape[2] != 4
    ):
        raise RuntimeError(
            "The `cameras_real` shape must be (N, 3, 4) or (N, 4, 4), but get "
            + f"{cameras_real.shape}"
        )

    if (
        cameras_pred.ndim != 3
        or cameras_pred.shape[1] > 4
        or cameras_pred.shape[1] < 3
        or cameras_pred.shape[2] != 4
    ):
        raise RuntimeError(
            "The `cameras_pred` shape must be (N, 3, 4) or (N, 4, 4), but get "
            + f"{cameras_real.shape}"
        )

    if cameras_real.shape[0] != cameras_pred.shape[0]:
        raise RuntimeError(
            "The `cameras_real` and `cameras_pred` must have equal number of cameras, "
            + f"but get {cameras_real.shape[0]} and {cameras_pred.shape[0]} "
            + "respectively."
        )


def _calculate_rotation_error(
    rotation_matrix_real: npt.NDArray[np.float64],
    rotation_matrix_pred: npt.NDArray[np.float64],
) -> float:
    """
    Computes the rotation error between two rotation matrices `rotation_matrix_real` and
    `rotation_matrix_pred`.

    :return: the rotation error

    :raise: RuntimeError: If shapes of rotation_matrix_real or rotation_matrix_pred are
        not (3, 3)
    """
    if rotation_matrix_real.shape != (3, 3) or rotation_matrix_pred.shape != (3, 3):
        raise RuntimeError(
            "Shapes of `rotation_matrix_real` and `rotation_matrix_pred` must be "
            + f"(3, 3) but get rotation_matrix_real: {rotation_matrix_real.shape}, "
            + f"rotation_matrix_pred: {rotation_matrix_pred.shape}."
        )

    relative_rotation = rotation_matrix_real.T.dot(rotation_matrix_pred)
    rotation_angles = cv2.Rodrigues(relative_rotation)[0]
    rotation_error = np.reshape(rotation_angles, (1, 3))
    rotation_error = np.linalg.norm(rotation_error, axis=1).item()
    return np.degrees(rotation_error)


def _calculate_translation_error_degree(
    translation_real: npt.NDArray[np.float64],
    translation_pred: npt.NDArray[np.float64],
) -> float:
    """
    Computes the rotation error between two vectors `translation_real` and
    `translation_pred`.

    :return: the rotation error.

    :raise: RuntimeError: If shapes of translation_real or translation_pred are not (1,
        3) or (3, 1) or (3,).
    """
    if translation_real.shape not in [
        (1, 3),
        (3, 1),
        (3,),
    ] or translation_pred.shape not in [(1, 3), (3, 1), (3,)]:
        raise RuntimeError(
            "Shapes of `translation_real` and `translation_pred` must be (1, 3) or "
            + f"(3,) or (3, 1) but get translation_real: {translation_real.shape}, "
            + f"translation_pred: {translation_pred.shape}."
        )

    translation_real = translation_real.reshape((3,))
    translation_pred = translation_pred.reshape((3,))

    dot_product = np.dot(translation_real, translation_pred)
    norms = np.linalg.norm(translation_real) * np.linalg.norm(translation_pred)
    cos_angle = dot_product / max(norms, 1e-12)
    angle = np.arccos(cos_angle).item()
    degrees = np.degrees(angle)
    degrees = np.minimum(degrees, 180.0 - degrees)
    return degrees


def _calculate_translation_error_distance(
    translation_real: npt.NDArray[np.float64],
    translation_pred: npt.NDArray[np.float64],
) -> float:
    """
    Computes the translation error between two vectors `translation_real` and
    `translation_pred`.

    :return: the translation error.

    :raise: RuntimeError: If shapes of translation_real or translation_pred are not (1,
        3) or (3, 1) or (3,).
    """
    if translation_real.shape not in [
        (1, 3),
        (3, 1),
        (3,),
    ] or translation_pred.shape not in [(1, 3), (3, 1), (3,)]:
        raise RuntimeError(
            "Shapes of `translation_real` and `translation_pred` must be (1, 3) or "
            + f"(3,) or (3, 1) but get translation_real: {translation_real.shape}, "
            + f"translation_pred: {translation_pred.shape}."
        )

    translation_real = translation_real.reshape((1, 3))
    translation_pred = translation_pred.reshape((1, 3))
    translation_error = np.linalg.norm(
        translation_real - translation_pred, axis=1
    ).item()
    return translation_error


def _calculate_auc(
    rotation_errors: npt.NDArray[np.float64],
    translation_errors: npt.NDArray[np.float64],
    thresholds: list[float],
) -> list[float]:
    """
    Computes normalized area-under-curve (AUC) scores over pose errors. For each sample,
    the maximum of the `rotation_errors` and `translation_errors` is taken. These
    per-sample maximum errors are sorted and used to build a recall curve (fraction of
    samples with error <= a given threshold).

    For every value T in `thresholds`, this function:
        1. Integrates the recall curve from 0 to T.
        2. Normalizes the area by T, so that the result lies in [0, 1] and
            corresponds
        to the best possible area being 1.

    This metric is related to the mean average accuracy strategy described in:
    https://arxiv.org/pdf/2306.15667.

    :return: areas for all thresholds.

    :raise: RuntimeError: If `rotation_errors` and `translation_errors` have different
        shapes.
    """

    if translation_errors.shape != rotation_errors.shape:
        raise RuntimeError(
            "The `translation_errors` and `rotation_errors` must have equal shapes"
            + f", but get {translation_errors.shape}, {rotation_errors.shape}"
        )

    error_matrix = np.concatenate(
        (rotation_errors[:, None], translation_errors[:, None]), axis=1
    )
    max_errors = np.max(error_matrix, axis=1)
    errors = [0] + sorted(list(max_errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return aucs


def _error_percent(
    errors: npt.NDArray[np.float64],
    thresholds: list[float],
) -> list[float]:
    """
    Calculates the ratio of `errors` below `thresholds`. The `errors` is a tensor of
    shape (N,) containing the errors.

    :return: the ratio of pairs of elements below thresholds.
    """

    result = []
    for threshold in thresholds:
        result.append((errors < threshold).sum() / len(errors))
    return result


def _relative_poses(
    cameras: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculates relative rotation and translation between each pair of `cameras`.


    :return: a tuple of (let the `cameras` has length N):
        - relative cameras poses
    """

    N = cameras.shape[0]
    if cameras.shape[1] == 3:
        cameras = np.concatenate(
            [cameras, np.zeros((N, 1, 4)).astype(cameras.dtype)], axis=1
        )
        cameras[:, 3, 3] = 1.0
    cameras_inv = closed_form_inverse_se3(cameras)
    cameras_1 = cameras[:, None, :, :].repeat(N, axis=1)
    cameras_2 = cameras_inv[None, :, :, :].repeat(N, axis=0)
    relative_poses = np.matmul(cameras_1, cameras_2)  # (N, N, 4, 4)

    return relative_poses


def relative_rotation_and_translation_errors(
    cameras_real_world2cam: npt.NDArray[np.float64],
    cameras_pred_world2cam: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate relative rotation (degree) and translation (distance and degree) errors
    between `cameras_real_world2cam` and `cameras_pred_world2cam`.

    :return: computed relative rotation errors in degrees, relative translation errors
        in distance, relative translation errors in degrees.
    """

    real_relative_poses = _relative_poses(cameras_real_world2cam)
    pred_relative_poses = _relative_poses(cameras_pred_world2cam)

    n = cameras_real_world2cam.shape[0]
    relative_rotation_errors = np.zeros((n, n), dtype=np.float64)
    relative_translation_errors_distance = np.zeros((n, n), dtype=np.float64)
    relative_translation_errors_degree = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            relative_rotation_errors[i, j] = _calculate_rotation_error(
                rotation_matrix_real=real_relative_poses[i, j, :3, :3],
                rotation_matrix_pred=pred_relative_poses[i, j, :3, :3],
            )

            relative_translation_errors_distance[i, j] = (
                _calculate_translation_error_distance(
                    translation_real=real_relative_poses[i, j, :3, 3],
                    translation_pred=pred_relative_poses[i, j, :3, 3],
                )
            )

            relative_translation_errors_degree[i, j] = (
                _calculate_translation_error_degree(
                    translation_real=real_relative_poses[i, j, :3, 3],
                    translation_pred=pred_relative_poses[i, j, :3, 3],
                )
            )

    mask_distinct = np.triu(np.ones((n, n), dtype=bool), k=1)
    relative_rotation_distinct = relative_rotation_errors[mask_distinct]
    relative_translation_distinct = relative_translation_errors_distance[mask_distinct]
    relative_translation_degree_distinct = relative_translation_errors_degree[
        mask_distinct
    ]

    return (
        relative_rotation_distinct,
        relative_translation_distinct,
        relative_translation_degree_distinct,
    )


def calculate_cameras_rra_rta_maa_given_errors(
    relative_rotation_errors: npt.NDArray[np.float64],
    relative_translation_errors_degree: npt.NDArray[np.float64],
    thresholds: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """
    Calculates Relative Rotation Accuracy (RRA) and Relative Translation Accuracy (RTA)
    when `relative_rotation_errors` and `relative_translation_errors_degree`
    provided. One is advised to read the original paper to fully understand the metrics:
    https://arxiv.org/pdf/2306.15667. The metrics represents the ratio of pairs of
    cameras where rotation error (for RRA) and translation error (for RTA) are small.

    :return: RRA and RTA for every threshold.
    """
    rra = _error_percent(errors=relative_rotation_errors, thresholds=thresholds)
    rta = _error_percent(
        errors=relative_translation_errors_degree, thresholds=thresholds
    )
    maa = _calculate_auc(
        rotation_errors=relative_rotation_errors,
        translation_errors=relative_translation_errors_degree,
        thresholds=thresholds,
    )

    return rra, rta, maa


def calculate_cameras_rra_rta_maa(
    cameras_real_world2cam: npt.NDArray[np.float64],
    cameras_pred_world2cam: npt.NDArray[np.float64],
    thresholds: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """
    Calculates Relative Rotation Accuracy (RRA) and Relative Translation Accuracy (RTA).
    One is advised to read the original paper to fully understand the metrics:
    https://arxiv.org/pdf/2306.15667. The metrics represents the ratio of pairs of
    cameras where rotation error (for RRA) and translation error (for RTA) are small.

    In the VGGT implementation they compare the metrics on world2cam cameras, therefore
    and would do the same thing:
    https://github.com/facebookresearch/vggt/blob/evaluation/evaluation/test_co3d.py#L142

    :return: RRA and RTA for every threshold as well as raw rotation and translation
        errors.

    :raise: RuntimeError: If shapes of `cameras_real_world2cam` and
        `cameras_pred_world2cam` are not equal to (N, 4, 4) or (N, 3, 4), or their
        length is different.
    """
    _check_cameras_shapes(
        cameras_real=cameras_real_world2cam, cameras_pred=cameras_pred_world2cam
    )

    relative_rotation, _, relative_translation_degree = (
        relative_rotation_and_translation_errors(
            cameras_pred_world2cam=cameras_pred_world2cam,
            cameras_real_world2cam=cameras_real_world2cam,
        )
    )

    rra, rta, maa = calculate_cameras_rra_rta_maa_given_errors(
        relative_rotation_errors=relative_rotation,
        relative_translation_errors_degree=relative_translation_degree,
        thresholds=thresholds,
    )

    return rra, rta, maa
