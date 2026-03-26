import numpy as np
import numpy.typing as npt
from vggt.utils.geometry import closed_form_inverse_se3

from sear.metrics.ate import align_pred_to_real
from sear.metrics.rra_rta_maa import (
    _check_cameras_shapes,
    relative_rotation_and_translation_errors,
)


def calculate_cameras_rpe_given_errors(
    relative_rotation_errors,
    relative_translation_errors_distance,
    relative_translation_errors_degree,
) -> tuple[float, float, float]:
    """
    Calculates Relative Pose Error (RPE) when `relative_rotation_errors`,
    `relative_translation_errors`, `relative_translation_degree_errors` are provided.

    :return: RPE for rotations, RPE for positions in distance, RPE for positions in
        degrees.
    """

    rpe_rotation = np.mean(relative_rotation_errors).item()
    rpe_translation_distance = np.mean(relative_translation_errors_distance).item()
    rpe_translation_degree = np.mean(relative_translation_errors_degree).item()

    return rpe_rotation, rpe_translation_distance, rpe_translation_degree


def calculate_cameras_rpe(
    cameras_real_world2cam: npt.NDArray[np.float64],
    cameras_pred_world2cam: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Calculates Relative Pose Error (RPE) between `cameras_real_world2cam` and
    `cameras_pred_world2cam`. The function does not normalize the trajectories to fit
    into [0, 1] and assumes they are already normalized.

    :return: RPE for positions, RPE for rotations.

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
    cameras_pred_aligned_world2cam = closed_form_inverse_se3(
        cameras_pred_aligned_cam2world
    )

    (
        relative_rotation_errors,
        relative_translation_errors_distance,
        relative_translation_errors_degree,
    ) = relative_rotation_and_translation_errors(
        cameras_real_world2cam=cameras_real_world2cam,
        cameras_pred_world2cam=cameras_pred_aligned_world2cam,
    )

    return calculate_cameras_rpe_given_errors(
        relative_rotation_errors=relative_rotation_errors,
        relative_translation_errors_distance=relative_translation_errors_distance,
        relative_translation_errors_degree=relative_translation_errors_degree,
    )
