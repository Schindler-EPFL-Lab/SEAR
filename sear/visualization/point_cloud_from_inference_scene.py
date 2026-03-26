from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import open3d as o3d
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from vggt.utils.geometry import depth_to_world_coords_points
from vggt.utils.helper import randomly_limit_trues

from sear.data_processing.align_part import AlignPart
from sear.data_processing.inference_scene_depth_and_poses import (
    InferenceSceneDepthAndPoses,
)
from sear.visualization.extrinsics_to_ply import camera_extrinsics_ply


@dataclass(kw_only=True)
class PointCloudFromInferenceScene(ReverseCli):
    """
    Creates point clouds from a ThermoScenes dataset scene
    """

    scene_path: Path = Path("input")
    """The path to the ThermoScenes scene"""
    depth_eps: float = 1e-8
    """
    The `depth_eps` is a depth value such that pixels with smaller depth are considered
    to be invalid.
    """
    max_num_points: int = 100000
    """The maximum number of generated points."""

    align: AlignPart | None = None
    """Whether to use alignment between the predicted and ground truth images"""

    def create_point_clouds(
        self,
        output_folder: Path,
        transforms_name: str = "transforms.json",
    ) -> None:
        """
        Creates rgb and thermal point clouds and camera pyramids for the scene located
        in `self.scene_path`. The point clouds are stored in `output_folder`/mode with
        mode is either eval or train. The `self.depth_eps` is a depth value such that
        pixels with smaller depth are considered to be invalid. The output .ply files
        are stored at `self.output_path`, with the `self.max_num_points` as the maximum
        number of generated points.
        """

        output_folder.mkdir(exist_ok=True, parents=True)

        if self.align is None:
            dataset = InferenceSceneDepthAndPoses.from_scene_path(
                scene_path=self.scene_path,
                transforms_name=transforms_name,
            )
        else:
            dataset = InferenceSceneDepthAndPoses.from_scene_path_aligned(
                scene_path=self.scene_path,
                align=self.align,
                transforms_name=transforms_name,
            )

        points_rgb_list: list[npt.NDArray[np.float32]] = []
        colors_rgb_list: list[npt.NDArray[np.float32]] = []
        extrinsics_world2cam_rgb_list: list[npt.NDArray[np.float32]] = []
        points_thermal_list: list[npt.NDArray[np.float32]] = []
        colors_thermal_list: list[npt.NDArray[np.float32]] = []
        num_points_one_image = int(np.ceil(self.max_num_points // len(dataset.images)))
        extrinsics_world2cam_thermal_list: list[npt.NDArray[np.float32]] = []

        for i in range(len(dataset.images[0])):
            image = dataset.images[0][i].numpy()
            depth = dataset.depths[0][i].numpy()
            extrinsic_world2cam = dataset.extrinsics_world2cam[0][i].numpy()
            intrinsic = dataset.intrinsics[0][i].numpy()
            thermal_flag = dataset.mask_thermal[0][i].numpy()

            image_np = image.transpose([1, 2, 0])

            world_coords_points, _, point_mask = depth_to_world_coords_points(
                depth_map=depth,
                extrinsic=extrinsic_world2cam,
                intrinsic=intrinsic,
                eps=self.depth_eps,
            )
            point_mask = randomly_limit_trues(
                point_mask, max_trues=num_points_one_image
            )

            points = world_coords_points[point_mask]
            colors = image_np[point_mask]

            if thermal_flag:
                points_thermal_list.append(points)
                colors_thermal_list.append(colors)
                extrinsics_world2cam_thermal_list.append(extrinsic_world2cam)

            else:
                points_rgb_list.append(points)
                colors_rgb_list.append(colors)
                extrinsics_world2cam_rgb_list.append(extrinsic_world2cam)

        # save point clouds
        if len(points_rgb_list) > 0:
            points_rgb = np.concatenate(points_rgb_list, axis=0)
            colors_rgb = np.concatenate(colors_rgb_list, axis=0)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points_rgb)
            point_cloud.colors = o3d.utility.Vector3dVector(colors_rgb)
            o3d.io.write_point_cloud(str(output_folder / "rgb.ply"), point_cloud)

        if len(points_thermal_list) > 0:
            points_thermal = np.concatenate(points_thermal_list, axis=0)
            colors_thermal = np.concatenate(colors_thermal_list, axis=0)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points_thermal)
            point_cloud.colors = o3d.utility.Vector3dVector(colors_thermal)
            o3d.io.write_point_cloud(str(output_folder / "thermal.ply"), point_cloud)

        extrinsics_world2cam_rgb = np.empty((0, 0, 0), dtype=np.float32)
        if len(extrinsics_world2cam_rgb_list) > 0:
            extrinsics_world2cam_rgb = np.stack(extrinsics_world2cam_rgb_list, axis=0)

        extrinsics_world2cam_thermal = np.empty((0, 0, 0), dtype=np.float32)
        if len(extrinsics_world2cam_thermal_list) > 0:
            extrinsics_world2cam_thermal = np.stack(
                extrinsics_world2cam_thermal_list, axis=0
            )

        # save camera extrinsics
        translations_max: list[float] = []
        if len(extrinsics_world2cam_rgb) > 0:
            translations_rgb = extrinsics_world2cam_rgb[:, :3, 3]
            translations_max.append(
                (translations_rgb.max(axis=0) - translations_rgb.min(axis=0)).max()
            )
        if len(extrinsics_world2cam_thermal) > 0:
            translations_thermal = extrinsics_world2cam_thermal[:, :3, 3]
            translations_max.append(
                (
                    translations_thermal.max(axis=0) - translations_thermal.min(axis=0)
                ).max()
            )

        bbox_size_max = max(translations_max) if len(translations_max) > 0 else 1.0

        if len(extrinsics_world2cam_rgb) > 0:
            camera_extrinsics_ply(
                extrinsics_world2cam=extrinsics_world2cam_rgb,
                output_path=output_folder / "rgb_cameras.ply",
                bbox_size_max=bbox_size_max,
            )

        if len(extrinsics_world2cam_thermal) > 0:
            camera_extrinsics_ply(
                extrinsics_world2cam=extrinsics_world2cam_thermal,
                output_path=output_folder / "thermal_cameras.ply",
                color=(0.5, 0.5, 0.5),
                bbox_size_max=bbox_size_max,
            )
