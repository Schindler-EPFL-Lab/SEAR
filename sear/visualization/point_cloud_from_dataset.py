from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import open3d as o3d
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from vggt.utils.geometry import depth_to_world_coords_points
from vggt.utils.helper import randomly_limit_trues

from sear.data_processing.single_dataset import VGGTSingleDataset
from sear.visualization.extrinsics_to_ply import camera_extrinsics_ply


@dataclass(kw_only=True)
class PointCloudFromDatasetCreator(ReverseCli):
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

    def create_point_clouds(
        self,
        output_folder: Path,
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

        dataset = VGGTSingleDataset(scene_path=self.scene_path)

        points_rgb_list: list[npt.NDArray[np.float32]] = []
        colors_rgb_list: list[npt.NDArray[np.float32]] = []
        extrinsics_world2cam_rgb_list: list[npt.NDArray[np.float32]] = []
        points_thermal_list: list[npt.NDArray[np.float32]] = []
        colors_thermal_list: list[npt.NDArray[np.float32]] = []
        num_points_one_image = int(np.ceil(self.max_num_points // len(dataset)))
        extrinsics_world2cam_thermal_list: list[npt.NDArray[np.float32]] = []

        for i in range(len(dataset)):
            dataset_item = dataset[i]

            image = dataset_item.image
            depth_rgb = dataset_item.depth_rgb
            extrinsic_world2cam_rgb = dataset_item.extrinsic_world2cam_rgb
            intrinsic_rgb = dataset_item.intrinsic_rgb
            thermal = dataset_item.thermal
            depth_thermal = dataset_item.depth_thermal
            extrinsic_world2cam_thermal = dataset_item.extrinsic_world2cam_thermal
            intrinsic_thermal = dataset_item.intrinsic_thermal

            image_np = image.permute(1, 2, 0).numpy()

            world_coords_points_rgb, _, point_mask_rgb = depth_to_world_coords_points(
                depth_map=depth_rgb.numpy(),
                extrinsic=extrinsic_world2cam_rgb.numpy(),
                intrinsic=intrinsic_rgb.numpy(),
                eps=self.depth_eps,
            )
            point_mask_rgb = randomly_limit_trues(
                point_mask_rgb, max_trues=num_points_one_image
            )

            points_rgb_list.append(world_coords_points_rgb[point_mask_rgb])
            colors_rgb_list.append(image_np[point_mask_rgb])
            extrinsics_world2cam_rgb_list.append(extrinsic_world2cam_rgb.numpy())

            thermal_np = thermal.permute(1, 2, 0).numpy()
            world_coords_points_thermal, _, point_mask_thermal = (
                depth_to_world_coords_points(
                    depth_map=depth_thermal.numpy(),
                    extrinsic=extrinsic_world2cam_thermal.numpy(),
                    intrinsic=intrinsic_thermal.numpy(),
                    eps=self.depth_eps,
                )
            )

            point_mask_thermal = randomly_limit_trues(
                point_mask_thermal, max_trues=num_points_one_image
            )
            points_thermal_list.append(world_coords_points_thermal[point_mask_thermal])
            colors_thermal_list.append(thermal_np[point_mask_thermal])
            extrinsics_world2cam_thermal_list.append(
                extrinsic_world2cam_thermal.numpy()
            )

        # save point clouds
        points_rgb = np.concatenate(points_rgb_list, axis=0)
        colors_rgb = np.concatenate(colors_rgb_list, axis=0)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_rgb)
        point_cloud.colors = o3d.utility.Vector3dVector(colors_rgb)
        o3d.io.write_point_cloud(str(output_folder / "rgb.ply"), point_cloud)

        points_thermal = np.concatenate(points_thermal_list, axis=0)
        colors_thermal = np.concatenate(colors_thermal_list, axis=0)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_thermal)
        point_cloud.colors = o3d.utility.Vector3dVector(colors_thermal)
        o3d.io.write_point_cloud(str(output_folder / "thermal.ply"), point_cloud)

        extrinsics_world2cam_rgb = np.stack(extrinsics_world2cam_rgb_list, axis=0)
        extrinsics_world2cam_thermal = np.stack(
            extrinsics_world2cam_thermal_list, axis=0
        )

        # save camera extrinsics
        extrinsics_world2cam_all = np.concatenate(
            [extrinsics_world2cam_rgb, extrinsics_world2cam_thermal], axis=0
        )
        translations = extrinsics_world2cam_all[:, :3, 3]
        bbox_size_max = (translations.max(axis=0) - translations.min(axis=0)).max()

        camera_extrinsics_ply(
            extrinsics_world2cam=extrinsics_world2cam_rgb,
            output_path=output_folder / "rgb_cameras.ply",
            bbox_size_max=bbox_size_max,
        )
        camera_extrinsics_ply(
            extrinsics_world2cam=extrinsics_world2cam_thermal,
            output_path=output_folder / "thermal_cameras.ply",
            color=(0.5, 0.5, 0.5),
            bbox_size_max=bbox_size_max,
        )
