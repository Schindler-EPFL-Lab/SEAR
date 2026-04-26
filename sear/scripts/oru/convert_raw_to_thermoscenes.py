import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import tyro
from vggt.utils.geometry import closed_form_inverse_se3

from sear.augment.geometric import GeometricTransform, GeometricTransformConfig
from sear.data_processing.frame_info import FrameInfo
from sear.data_processing.project_points import project_points
from sear.data_processing.synchronize import synchronize
from sear.data_processing.topics import CamInfo, CamTF, PathToObject


def _build_oru_camera_poses(
    map_odom: list[CamTF],
    odom_base_link: list[CamTF],
    base_link_camera: list[CamTF],
) -> npt.NDArray[np.float32]:
    """
    Composes camera poses for the tree map -> odom -> base_link -> camera to get the
    camera poses in map coordinate space. The `map_odom` specifies the orientation of
    "odom" in the "map" coordinate system. The `odom_base_link` specifies the
    orientation of "base_link" in the "odom" coordinate system. The `base_link_camera`
    specifies the orientation of "camera" in the "base_link" coordinate system.

    :return: a set of poses in camera-to-world OpenCV convention.

    :raise: RuntimeError if lenghts of map_odom, odom_base_link, base_link_camera
        mismatch.
    """

    if not len(map_odom) == len(odom_base_link) == len(base_link_camera):
        raise RuntimeError(
            "The lengths of `map_odom`, `odom_base_link`, `base_link_camera` must match"
            + f", but get {len(map_odom)}, {len(odom_base_link)},"
            + f"{len(base_link_camera)} respectively."
        )
    poses_cam2world_list: list[npt.NDArray[np.float32]] = []
    for i in range(len(map_odom)):
        pose_cam2world = (
            map_odom[i].pose_cam2world
            @ odom_base_link[i].pose_cam2world
            @ base_link_camera[i].pose_cam2world
        )
        poses_cam2world_list.append(pose_cam2world)

    return np.stack(poses_cam2world_list)


def _build_oru_camera_intrinsics(
    camera_info: dict[str, dict[str, int | bool] | int | str | list[float]],
) -> list[CamInfo]:
    """
    Extracts camera intrinsics from `camera_info`, which contain width, height, and K
    (3x3) matrix as a dict.

    :return: intrinsics camera parameters, widths and heights. If the length of
        camera_info is N then:
        - K: (N, 3, 3)
        - width: (N,)
        - height: (N,)
    """

    result: list[CamInfo] = []
    for timestamp_str, data in camera_info.items():
        timestamp = datetime.fromisoformat(timestamp_str)
        result.append(CamInfo.from_dict(timestamp=timestamp, data=data))

    return result


def _create_frame(
    stem: str,
    output_modality_folder_name: str,
    output_depth_modality_folder_name: str,
    modality_image_path: Path,
    extrinsic_cam2world: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    pose_lidar_cam2world: npt.NDArray[np.float32],
    lidar_path: Path,
    width: int,
    height: int,
    desired_width: int,
    desired_aspect_ratio: float,
    output_folder: Path,
) -> FrameInfo:
    """
    Constructs a frame dictionary containing paths to image and depth, and
    camera-related metadata and transforms the image to have `desired_width` and
    `desired_aspect_ratio`. The transformation is helpful when images from different
    modalities (rgb and thermal) must have the same shape. The `stem` is a base filename
    (without suffix) used for saving the modality image and depth map. The
    `output_modality_folder_name` is a subfolder name under `output_folder` where the
    modality image will be stored. The `output_depth_modality_folder_name` is a
    subfolder name under `output_folder` where the depth map will be stored. The
    `modality_image_path` is a path to the input modality image to be copied. The
    `pose_lidar_cam2world` is a lidar pose of shape (4, 4), the `lidar_path` is a path
    to the lidar point cloud associated wit the `modality_images`. The
    `extrinsic_cam2world` is a camera-to-world transformation matrix of shape (4, 4).
    The `intrinsic` is a camera intrinsic matrix of shape (3, 3). The `width` and
    `height` represent the modality_image resolution. The `output_folder` is a root
    directory where modality data will be saved.

    :return: A dictionary containing file paths, camera intrinsics, extrinsics, image
        resolution, distortion parameters, and field-of-view values.

    :raise: RuntimeError if `extrinsic_cam2world` is not of shape (4, 4),
        `pose_lidar_cam2world` is not of shape (4, 4), `intrinsic` is not of shape
        (3,3).
    """

    if extrinsic_cam2world.shape != (4, 4):
        raise RuntimeError(
            "`extrinsic_cam2world` must have shape (4, 4) but got "
            + f"{extrinsic_cam2world.shape}."
        )

    if pose_lidar_cam2world.shape != (4, 4):
        raise RuntimeError(
            "`pose_lidar_cam2world` must have shape (4, 4) but got "
            + f"{pose_lidar_cam2world.shape}."
        )

    if intrinsic.shape != (3, 3):
        raise RuntimeError(
            f"`intrinsic` must have shape (3, 3) but got {intrinsic.shape}."
        )

    geometric_transform = GeometricTransform(
        GeometricTransformConfig(
            target_image_width=desired_width,
            patch_size=1,
            p_crop=1.0,
            crop_ratio=None,
            aspect_ratio=(desired_aspect_ratio, desired_aspect_ratio),
            p_rotate=0.0,
        )
    )

    saved_modality_relpath = Path(output_modality_folder_name) / (
        stem + modality_image_path.suffix
    )

    image = cv2.imread(str(modality_image_path), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transforms.ToTensor()(image)

    point_cloud = o3d.t.io.read_point_cloud(str(lidar_path))
    point_cloud.transform(pose_lidar_cam2world)
    modality_depth = project_points(
        points=point_cloud.point["positions"].numpy().astype(np.float32),
        extrinsic_cam2world=extrinsic_cam2world,
        intrinsic=intrinsic,
        height=height,
        width=width,
    )
    modality_depth_torch = torch.from_numpy(modality_depth)
    saved_depth_modality_relpath = Path(output_depth_modality_folder_name) / (
        stem + ".npy"
    )

    extrinsic_matrix_modality_world2cam = closed_form_inverse_se3(
        extrinsic_cam2world[None]
    )[0]

    (
        image_torch,
        modality_depth_torch,
        extrinsic_matrix_modality_world2cam_torch,
        intrinsic_torch,
    ) = geometric_transform.apply(
        images=image_tensor.unsqueeze(0),
        depths=modality_depth_torch.unsqueeze(0),
        extrinsic_matrices_world2cam=torch.from_numpy(
            extrinsic_matrix_modality_world2cam[:3, :]
        ).unsqueeze(0),
        intrinsic_matrices=torch.from_numpy(intrinsic).unsqueeze(0),
    )

    torchvision.utils.save_image(image_torch[0], output_folder / saved_modality_relpath)
    modality_depth = modality_depth_torch[0].numpy().astype(np.float32)
    np.save(output_folder / saved_depth_modality_relpath, modality_depth)

    return FrameInfo(
        extrinsic_matrix_world2cam=extrinsic_matrix_modality_world2cam_torch[0],
        intrinsic_matrix=intrinsic_torch[0],
        width=image_torch.shape[3],
        height=image_torch.shape[2],
        image_path=saved_modality_relpath,
        depth_path=saved_depth_modality_relpath,
    )


def _build_camera_transform_from_dict(
    frame_id: str,
    child_frame_id: str,
    data: dict[
        str,
        dict[str, dict[str, dict[str, int] | str] | str | dict[str, dict[str, float]]],
    ],
) -> list[CamTF]:
    return [
        CamTF.from_data(timestamp=datetime.fromisoformat(k), data=v)
        for k, v in data.items()
        if v["header"]["frame_id"] == frame_id and v["child_frame_id"] == child_frame_id
    ]


def main(
    root_path: Path,
    tf_static_path: Path | None = None,
    tf_file_name: str = "tf/tf.json",
    image_folder_name: str = "ueye_ids_camera_image_raw_compressed",
    thermal_folder_name: str = "flir_camera_image_raw_compressed",
    flir_info_file_name: str = "flir_camera_camera_info/flir_camera_camera_info.json",
    rgb_info_file_name: str = (
        "ueye_ids_camera_camera_info/ueye_ids_camera_camera_info.json"
    ),
    lidar_deskewed_folder_name: str = "cx_lslidar_point_cloud_deskewed",
    lidar_folder_name: str = "cx_lslidar_point_cloud",
    output_folder: Path = Path("outputs"),
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    desired_width: int = 1024,
    take_every: int = 5,
) -> None:
    """
    Converts the oru (Orebro) dataset located in `root_path` into ThermoScenesVGGT
    format. One must follow the instructions in `sear/scripts/oru/README.md` to extract
    data from raw .mcap ROS2 files. The `tf_static_path` is the location of tf_static
    file (which contains relative positions between the robot and rgb camera, thermal
    camera, and lidar). If not provided then it would be taken from
    `root_path`/tf_static/tf_static.json. The `tf_file_name` contains the file with the
    poses of the robot, and of the measuring device. The `image_folder_name` is the name
    of the folder containing rgb images. The `thermal_folder_name`  is the name of the
    folder containing thermal images. The `flir_info_file_name` and `rgb_info_file_name`
    are relative paths to the files containing internal camera parameters. The
    `lidar_deskewed_folder_name` and the `lidar_folder_name` are the names of folders
    containing lidar point clouds. The `output_folder` specifies the output directory
    where to save the processed dataset. The (`start_ratio`, `end_ratio`) specifies the
    interval from what data is taken. This is particularly helpful if robot is not
    moving at the beginning. The `take_every` specifies the step of taking images. This
    is helpful if robot is slow.

    :raise: FileNotFoundError if tf_static_path does not exist.
    """

    if tf_static_path is None:
        tf_static_path = root_path / "tf_static/tf_static.json"

    if not tf_static_path.exists():
        raise FileNotFoundError(f"tf_static_path {str(tf_static_path)} does not exist")

    with open(tf_static_path) as f:
        tf_static = json.load(f)

    tf_base_link_rgb_camera = _build_camera_transform_from_dict(
        frame_id="base_link", child_frame_id="ids_camera", data=tf_static
    )
    tf_base_link_thermal_camera = _build_camera_transform_from_dict(
        frame_id="base_link", child_frame_id="flir_camera", data=tf_static
    )
    tf_base_link_lidar = _build_camera_transform_from_dict(
        frame_id="base_link", child_frame_id="laser_link", data=tf_static
    )

    with open(root_path / tf_file_name) as f:
        tf = json.load(f)

    # split tf into map <-> odom and odom <-> base_link
    tf_map_odom = _build_camera_transform_from_dict(
        frame_id="map", child_frame_id="odom", data=tf
    )
    tf_odom_base_link = _build_camera_transform_from_dict(
        frame_id="odom", child_frame_id="base_link", data=tf
    )

    with open(root_path / flir_info_file_name) as f:
        flir_camera_info = json.load(f)
    flir_camera_info = _build_oru_camera_intrinsics(flir_camera_info)
    with open(root_path / rgb_info_file_name) as f:
        rgb_camera_info = json.load(f)
    rgb_camera_info = _build_oru_camera_intrinsics(rgb_camera_info)

    images_paths = PathToObject.from_folder(root_path / image_folder_name)
    thermal_paths = PathToObject.from_folder(root_path / thermal_folder_name)
    lidar_paths = PathToObject.from_folder(root_path / lidar_folder_name)
    lidar_deskewed_paths = PathToObject.from_folder(
        root_path / lidar_deskewed_folder_name
    )

    # resample everything to match images paths
    tf_base_link_rgb_camera = synchronize(images_paths, tf_base_link_rgb_camera)
    tf_base_link_thermal_camera = synchronize(images_paths, tf_base_link_thermal_camera)
    tf_base_link_lidar = synchronize(images_paths, tf_base_link_lidar)
    tf_map_odom = synchronize(images_paths, tf_map_odom)
    tf_odom_base_link = synchronize(images_paths, tf_odom_base_link)

    flir_camera_info = synchronize(images_paths, flir_camera_info)
    rgb_camera_info = synchronize(images_paths, rgb_camera_info)

    thermal_paths = synchronize(images_paths, thermal_paths)
    lidar_paths = synchronize(images_paths, lidar_paths)
    lidar_deskewed_paths = synchronize(images_paths, lidar_deskewed_paths)
    images_paths = synchronize(images_paths, images_paths)

    # build rgb and thermal camera poses
    poses_rgb_cam2world = _build_oru_camera_poses(
        map_odom=tf_map_odom,
        odom_base_link=tf_odom_base_link,
        base_link_camera=tf_base_link_rgb_camera,
    )

    poses_thermal_cam2world = _build_oru_camera_poses(
        map_odom=tf_map_odom,
        odom_base_link=tf_odom_base_link,
        base_link_camera=tf_base_link_thermal_camera,
    )

    poses_lidar_cam2world = _build_oru_camera_poses(
        map_odom=tf_map_odom,
        odom_base_link=tf_odom_base_link,
        base_link_camera=tf_base_link_lidar,
    )

    # save the results
    output_folder.mkdir(exist_ok=True)
    output_images_folder_name = "images"
    (output_folder / output_images_folder_name).mkdir(exist_ok=True)
    output_thermal_folder_name = "thermal"
    (output_folder / output_thermal_folder_name).mkdir(exist_ok=True)
    output_depths_rgb_folder_name = "depths_rgb"
    (output_folder / output_depths_rgb_folder_name).mkdir(exist_ok=True)
    output_depths_thermal_folder_name = "depths_thermal"
    (output_folder / output_depths_thermal_folder_name).mkdir(exist_ok=True)

    transforms: dict[
        str,
        str | dict[str, list[dict[str, int | float | list[list[float]]]]] | float | int,
    ] = {}

    transforms["type"] = "oru"
    transforms["frames"]: dict[  # type: ignore
        str, list[dict[str, int | float | list[list[float]]]]
    ] = []  # type: ignore
    transforms["start_ratio"] = start_ratio
    transforms["end_ratio"] = end_ratio

    start_index = int(np.floor(len(images_paths) * start_ratio))
    end_index = int(np.ceil(len(images_paths) * end_ratio)) + 1
    end_index = min(end_index, len(images_paths))

    transforms["start_index"] = start_index
    transforms["end_index"] = end_index
    transforms["take_every"] = take_every

    rgb_aspect_ratio = rgb_camera_info[0].height / rgb_camera_info[0].width

    for i in range(start_index, end_index, take_every):
        common_stem = f"{i:05}"

        frame_rgb = _create_frame(
            stem=common_stem,
            output_modality_folder_name=output_images_folder_name,
            output_depth_modality_folder_name=output_depths_rgb_folder_name,
            modality_image_path=images_paths[i].path,
            extrinsic_cam2world=poses_rgb_cam2world[i],
            intrinsic=rgb_camera_info[i].intrinsic,
            pose_lidar_cam2world=poses_lidar_cam2world[i],
            lidar_path=lidar_paths[i].path,
            width=rgb_camera_info[i].width,
            height=rgb_camera_info[i].height,
            desired_width=desired_width,
            desired_aspect_ratio=rgb_aspect_ratio,
            output_folder=output_folder,
        ).to_dict()
        frame_rgb["type"] = "rgb"

        frame_thermal = _create_frame(
            stem=common_stem,
            output_modality_folder_name=output_thermal_folder_name,
            output_depth_modality_folder_name=output_depths_thermal_folder_name,
            modality_image_path=thermal_paths[i].path,
            extrinsic_cam2world=poses_thermal_cam2world[i],
            intrinsic=flir_camera_info[i].intrinsic,
            pose_lidar_cam2world=poses_lidar_cam2world[i],
            lidar_path=lidar_paths[i].path,
            width=flir_camera_info[i].width,
            height=flir_camera_info[i].height,
            desired_width=desired_width,
            desired_aspect_ratio=rgb_aspect_ratio,
            output_folder=output_folder,
        ).to_dict()
        frame_thermal["type"] = "thermal"

        transforms["frames"].append({"rgb": frame_rgb, "thermal": frame_thermal})

    with open(output_folder / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)


if __name__ == "__main__":
    tyro.cli(main)
