import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from training.data.dataset_util import depth_to_world_coords_points
from vggt.utils.geometry import closed_form_inverse_se3

from sear.data_processing.frame_info import FrameInfo
from sear.data_processing.project_points import project_points


@dataclass
class CorrectThermalCamerasAndDepthConfig(ReverseCli):
    scene_path: Path
    """Path to the scene that needs to be updated"""
    calibration_file: Path = Path("./sear/scripts/thermalnerf/calibation.json")
    """Path to the file containing calibration results on thermal and rgb cameras."""

    hardcoded_image_idx_1: int = 2
    """
    It is assumed by authors that the distance between some known frames (3rd and 4th)
    is exactly 1 ft (30.48 cm). The `hardcoded_image_idx_1` and `hardcoded_image_idx_2`
    specify the indices of images where the distance is exactly 1 ft.
    """
    hardcoded_image_idx_2: int = 3
    """
    The second image index such that the distance between the corresponding camera
    poses is is 1 ft
    """

    def run(self) -> None:
        """
        Corrects the thermal camera parameters (extrinsics and intrinsics) and depth
        maps of a ThermalNeRF scene located (after its processed using the script to
        extract rgb poses and depths
        sear/scripts/vggt_rgb_depths_and_cameras.py).
        """

        with open(self.calibration_file) as f:
            calibration_data = json.load(f)
        thermal_intrinsic = np.array(calibration_data["camera_matrix_thermal"])
        thermal_to_rgb = np.array(calibration_data["thermal_rgb_transform"])
        random_thermal_image_path = next((self.scene_path / "thermal").iterdir())
        h, w = cv2.imread(str(random_thermal_image_path)).shape[0:2]

        output_thermal_depth_folder = Path("depths_thermal")
        (self.scene_path / output_thermal_depth_folder).mkdir(exist_ok=True)

        with open(self.scene_path / "transforms.json") as f:
            transforms = json.load(f)

        transforms["type"] = "ThermalNeRF"

        # Find scaling between calibration world (everything is in cm) and the frame
        # with known poses (which is scale-ambiguous).
        all_rgb_images_paths = sorted(
            [frame["rgb"] for frame in transforms["frames"]],
            key=lambda x: x["file_path"],
        )
        frame_rgb_hardcoded_1 = all_rgb_images_paths[self.hardcoded_image_idx_1]
        extrinsic_world2cam_hardcoded_1 = np.array(
            frame_rgb_hardcoded_1["transform_matrix"]
        )
        extrinsic_cam2world_hardcoded_1 = closed_form_inverse_se3(
            extrinsic_world2cam_hardcoded_1[None]
        )[0]
        frame_rgb_hardcoded_2 = all_rgb_images_paths[self.hardcoded_image_idx_2]
        extrinsic_world2cam_hardcoded_2 = np.array(
            frame_rgb_hardcoded_2["transform_matrix"]
        )
        extrinsic_cam2world_hardcoded_2 = closed_form_inverse_se3(
            extrinsic_world2cam_hardcoded_2[None]
        )[0]

        calibration_distance = 12.0 * 2.54  # 1 ft in cm
        poses_distance = np.linalg.norm(
            extrinsic_cam2world_hardcoded_1[:3, 3]
            - extrinsic_cam2world_hardcoded_2[:3, 3]
        )

        # scale transformation to go from the poses space to the calibration space and
        # back
        poses_to_calibration = np.eye(4) * calibration_distance / poses_distance
        poses_to_calibration[3, 3] = 1.0
        calibration_to_poses = np.eye(4) * poses_distance / calibration_distance
        calibration_to_poses[3, 3] = 1.0

        for frame in transforms["frames"]:
            extrinsic_rgb_world2cam_poses, intrinsic_rgb = FrameInfo.dict_to_matrices(
                frame["rgb"]
            )
            extrinsic_rgb_world2cam_poses = extrinsic_rgb_world2cam_poses.numpy()
            intrinsic_rgb = intrinsic_rgb.numpy()

            extrinsic_rgb_world2cam_poses = np.concatenate(
                [extrinsic_rgb_world2cam_poses, np.zeros((1, 4))], axis=0
            )
            extrinsic_rgb_world2cam_poses[3, 3] = 1.0
            extrinsic_rgb_cam2world_poses = closed_form_inverse_se3(
                extrinsic_rgb_world2cam_poses[None]
            )[0]

            # 1. We go from the poses space to the calibration space
            # 2. We do the transformation of the point in the calibration space using
            #    the `thermal_to_rgb` transform
            # 3. We go back from the calibration space to the poses space
            # 4. We perform camera-to-world transformation using the corresponding rgb
            #    camera pose
            extrinsic_thermal_cam2world_poses = (
                extrinsic_rgb_cam2world_poses
                @ calibration_to_poses
                @ thermal_to_rgb
                @ poses_to_calibration
            )
            extrinsic_thermal_world2cam_poses = closed_form_inverse_se3(
                extrinsic_thermal_cam2world_poses[None]
            )[0]

            # building the depth map by projecting point cloud of the corresponding  rgb
            # frame
            depth_rgb = np.load(self.scene_path / frame["rgb"]["depth_file_path"])
            point_cloud, _, point_mask = depth_to_world_coords_points(
                depth_map=depth_rgb[:, :, 0],
                extrinsic=extrinsic_rgb_world2cam_poses,
                intrinsic=intrinsic_rgb,
            )
            depth_thermal = project_points(
                points=point_cloud[point_mask],
                extrinsic_cam2world=extrinsic_thermal_cam2world_poses,
                intrinsic=thermal_intrinsic,
                width=w,
                height=h,
            )
            thermal_depth_file_path = Path(
                output_thermal_depth_folder,
                (Path(frame["thermal"]["file_path"]).stem + ".npy"),
            )
            np.save(self.scene_path / thermal_depth_file_path, depth_thermal)

            frame_thermal_corrected = FrameInfo(
                extrinsic_matrix_world2cam=extrinsic_thermal_world2cam_poses,
                intrinsic_matrix=thermal_intrinsic,
                width=w,
                height=h,
                image_path=Path(frame["thermal"]["file_path"]),
                depth_path=thermal_depth_file_path,
            ).to_dict()
            frame_thermal_corrected["type"] = "thermal"

            frame["thermal"] = frame_thermal_corrected

        with open(self.scene_path / "transforms.json", "w") as f:
            json.dump(transforms, f, indent=4)


if __name__ == "__main__":
    parameters = tyro.cli(CorrectThermalCamerasAndDepthConfig)
    parameters.run()
