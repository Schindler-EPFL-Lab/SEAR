import numpy as np
import numpy.typing as npt
import open3d as o3d
from vggt.utils.geometry import closed_form_inverse_se3, depth_to_world_coords_points

from sear.metrics.ate import align_pred_to_real
from sear.metrics.rra_rta_maa import _check_cameras_shapes


def calculate_point_cloud_metrics_given_errors(
    point_cloud_errors_from_predicted: npt.NDArray[np.float64],
    point_cloud_errors_from_ground_truth: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """
    Calculates accuracy of point cloud reconstruction based on provided errors
    `point_cloud_errors_from_predicted` (for accuracy) and
    `point_cloud_errors_from_ground_truth` (for completeness).

    :return: Accuracy, Completeness, and Chamfer distance

    :raise: RuntimeError if point_cloud_errors is not of shape (N,)
    """
    if point_cloud_errors_from_ground_truth.ndim != 1:
        raise RuntimeError(
            "The `point_cloud_errors_from_ground_truth` must be of shape (N,) but got "
            + f"{point_cloud_errors_from_ground_truth.shape}"
        )

    if point_cloud_errors_from_predicted.ndim != 1:
        raise RuntimeError(
            "The `point_cloud_errors_from_predicted` must be of shape (N,) but got "
            + f"{point_cloud_errors_from_predicted.shape}"
        )

    accuracy = np.median(point_cloud_errors_from_predicted).item()
    completeness = np.median(point_cloud_errors_from_ground_truth).item()
    chamfer = (accuracy + completeness) / 2
    return accuracy, completeness, chamfer


def point_cloud_distances(
    point_cloud_from: npt.NDArray[np.float64],
    point_cloud_to: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    For each point in `point_cloud_from` finds distance to the closest point in
    `point_cloud_to`.

    :return: closest distances.

    :raise: RuntimeError if shapes of `points_real` and `points_pred` mismatch or not
        equal to (N, 3).
    """

    if (
        point_cloud_from.ndim != 2
        or point_cloud_to.ndim != 2
        or point_cloud_from.shape[1] != 3
        or point_cloud_to.shape[1] != 3
    ):
        raise RuntimeError(
            "The `point_cloud_from` and `point_cloud_to` must be of shape (N, 3) and "
            + f"(M, 3) but got {point_cloud_to.shape} and {point_cloud_from.shape} "
            + "respectively."
        )

    point_cloud_from_o3d = o3d.geometry.PointCloud()
    point_cloud_from_o3d.points = o3d.utility.Vector3dVector(
        point_cloud_from.astype(np.float64)
    )

    point_cloud_to_o3d = o3d.geometry.PointCloud()
    point_cloud_to_o3d.points = o3d.utility.Vector3dVector(
        point_cloud_to.astype(np.float64)
    )

    return np.asarray(
        point_cloud_from_o3d.compute_point_cloud_distance(point_cloud_to_o3d)
    )


def _create_point_cloud(
    depths: npt.NDArray[np.float64],
    cameras_world2cam: npt.NDArray[np.float64],
    intrinsics: npt.NDArray[np.float64],
    eps: float = 1e-8,
) -> npt.NDArray[np.float64]:
    """
    Creates a single point cloud by unprojecting depth maps using corresponding camera
    extrinsics and intrinsics.

    :return: The concatenated point cloud of shape (M, 3), where M is the total number
        of valid points across all frames.

    :raise: ValueError:
        - If `depths` is not of shape (N, H, W).
        - If `intrinsics` is not of shape (N, 3, 3).
        - If `cameras_world2cam` is not of shape (N, 4, 4) or (N, 3, 4).
        - If the number of frames is not consistent across all inputs.
    """

    if depths.ndim != 3:
        raise ValueError(
            f"`depths` must be of shape (N, H, W), but got {depths.shape}."
        )

    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(
            f"`intrinsics` must be of shape (N, 3, 3), but got {intrinsics.shape}."
        )

    for element, name in [
        (cameras_world2cam, "cameras_world2cam"),
        (depths, "depths"),
        (intrinsics, "intrinsics"),
    ]:
        if element.shape[0] != cameras_world2cam.shape[0]:
            raise ValueError(
                "All the elements must be of shape (N, ...), got "
                + f"{element.shape} for {name}."
            )

    if cameras_world2cam.ndim != 3 or cameras_world2cam.shape[1:] not in [
        (4, 4),
        (3, 4),
    ]:
        raise ValueError(
            f"`cameras_world2cam` must be of shape (N, 4, 4) or (N, 3, 4), "
            f"but got {cameras_world2cam.shape}."
        )

    all_points_list: list[npt.NDArray[np.float64]] = []

    for frame_idx in range(cameras_world2cam.shape[0]):
        points, _, point_mask = depth_to_world_coords_points(
            depth_map=depths[frame_idx],
            extrinsic=cameras_world2cam[frame_idx],
            intrinsic=intrinsics[frame_idx],
            eps=eps,
        )
        all_points_list.append(points[point_mask])

    return np.concatenate(all_points_list)


def calculate_point_cloud_accuracy_and_completeness(
    depths_real: npt.NDArray[np.float64],
    cameras_real_world2cam: npt.NDArray[np.float64],
    intrinsics_real: npt.NDArray[np.float64],
    depths_pred: npt.NDArray[np.float64],
    cameras_pred_world2cam: npt.NDArray[np.float64],
    intrinsics_pred: npt.NDArray[np.float64],
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    """
    Calculates Point Cloud Accuracy between point cloud created from real `depths_real`,
    `cameras_real_world2cam`, `intrinsics_real` and `depths_pred`,
    `cameras_pred_world2cam`, `intrinsics_pred`.

    :return: Accuracy, Completeness, and Chamfer distance

    :raise: ValueError:
        - If `depths_real` and `depths_pred` do not have the same shape or are not of
          shape (N, H, W).
        - If `intrinsics_real` or `intrinsics_pred` are not of shape (N, 3, 3).
        - If the number of views `N` is not consistent across all inputs.
    :raise RuntimeError:
        - If `cameras_real_world2cam` or `cameras_pred_world2cam` are not of shape (N,
          4, 4) or (N, 3, 4).
    """

    _check_cameras_shapes(
        cameras_real=cameras_real_world2cam, cameras_pred=cameras_pred_world2cam
    )

    if depths_real.shape != depths_pred.shape or depths_real.ndim != 3:
        raise ValueError(
            f"`depths_real` and `depths_pred` must have the same shape of (N, H, W), "
            f"but got {depths_real.shape} and {depths_pred.shape}."
        )

    if intrinsics_real.shape != intrinsics_pred.shape:
        raise ValueError(
            f"The `intrinsics_real` and `intrinsics_pred` must be of shape (N, 3, 3), "
            f"but got {intrinsics_real.shape} and {intrinsics_pred.shape} respectively."
        )

    for element, name in [
        (depths_real, "depths_real"),
        (cameras_real_world2cam, "cameras_real_world2cam"),
        (intrinsics_real, "intrinsics_real"),
        (depths_pred, "depths_pred"),
        (cameras_pred_world2cam, "cameras_pred_world2cam"),
        (intrinsics_pred, "intrinsics_pred"),
    ]:
        if element.shape[0] != cameras_real_world2cam.shape[0]:
            raise ValueError(
                "All the elements must be of shape (N, ...), got "
                + f"{element.shape} for {name}"
            )

    cameras_real_cam2world = closed_form_inverse_se3(cameras_real_world2cam)
    cameras_pred_cam2world = closed_form_inverse_se3(cameras_pred_world2cam)

    cameras_pred_aligned_cam2world, _, _, scale = align_pred_to_real(
        cameras_real_cam2world=cameras_real_cam2world,
        cameras_pred_cam2world=cameras_pred_cam2world,
    )
    cameras_pred_aligned_world2cam = closed_form_inverse_se3(
        cameras_pred_aligned_cam2world
    )
    depths_pred_aligned = depths_pred * scale

    all_points_real = _create_point_cloud(
        depths=depths_real,
        cameras_world2cam=cameras_real_world2cam,
        intrinsics=intrinsics_real,
        eps=eps,
    )
    all_points_pred = _create_point_cloud(
        depths=depths_pred_aligned,
        cameras_world2cam=cameras_pred_aligned_world2cam,
        intrinsics=intrinsics_pred,
        eps=eps,
    )

    errors_from_real = point_cloud_distances(
        point_cloud_from=all_points_real, point_cloud_to=all_points_pred
    )

    errors_from_pred = point_cloud_distances(
        point_cloud_from=all_points_pred, point_cloud_to=all_points_real
    )

    return calculate_point_cloud_metrics_given_errors(
        point_cloud_errors_from_predicted=errors_from_pred,
        point_cloud_errors_from_ground_truth=errors_from_real,
    )
