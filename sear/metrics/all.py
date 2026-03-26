import numpy as np
import numpy.typing as npt

from sear.metrics.ate import calculate_cameras_ate
from sear.metrics.rpe import calculate_cameras_rpe
from sear.metrics.rra_rta_maa import (
    _check_cameras_shapes,
    calculate_cameras_rra_rta_maa,
)


def calculate_cameras_metrics(
    cameras_real_world2cam: npt.NDArray[np.float64],
    cameras_pred_world2cam: npt.NDArray[np.float64],
    thresholds: list[float],
) -> tuple[
    list[float],
    list[float],
    list[float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """
    Calculates Relative Rotation Accuracy (RRA) Relative Translation Accuracy (RTA), and
    the mean Average Accuracy (mAA), Absolute Trajectory Error (ATE), Relative Pose
    Error (RPE) between real cameras `cameras_real` and predicted cameras
    `cameras_pred`.

    One is advised to read the original paper to fully understand the metrics:
    https://arxiv.org/pdf/2306.15667. The first two metrics represent the ratio of pairs
    of cameras where rotation error (for RRA) and translation error (for RTA) are small.
    The mAA represents a combination of those for rotation and translation. The ATE
    evaluates the mean distance between predicted and real cameras. The RPE evaluates
    the mean distance between the all pairs of predicted and real cameras.

    :return: RRA, RTA, mAA, ATE, RPE for every threshold as well as relative and
        absolute rotation and translation errors.

    :raises RuntimeError: If shapes of `cameras_real` and `cameras_pred` are equal to
        (N, 4, 4) or (N, 3, 4), or their length is different.
    """

    _check_cameras_shapes(
        cameras_real=cameras_real_world2cam, cameras_pred=cameras_pred_world2cam
    )

    rra, rta, maa = calculate_cameras_rra_rta_maa(
        cameras_real_world2cam=cameras_real_world2cam,
        cameras_pred_world2cam=cameras_pred_world2cam,
        thresholds=thresholds,
    )

    ate = calculate_cameras_ate(
        cameras_real_world2cam=cameras_real_world2cam,
        cameras_pred_world2cam=cameras_pred_world2cam,
    )
    rpe = calculate_cameras_rpe(
        cameras_real_world2cam=cameras_real_world2cam,
        cameras_pred_world2cam=cameras_pred_world2cam,
    )

    return rra, rta, maa, ate, rpe
