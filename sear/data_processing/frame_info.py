from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch


class FrameInfo:
    """
    Stores info about a frame and can convert it to a serializable dict.
    """

    def __init__(
        self,
        extrinsic_matrix_world2cam: torch.Tensor | npt.NDArray[np.float32],
        intrinsic_matrix: torch.Tensor | npt.NDArray[np.float32],
        width: int,
        height: int,
        image_path: Path,
        depth_path: Path,
    ) -> None:
        """
        Initializes the FrameInfo with camera extrinsics in opencv world-to-camera
        format `extrinsic_matrix_world2cam` and intrinsics `intrinsic_matrix`. The
        image resolution is (`height`, `width`). The stored image is in
        `image_path`, and the corresponding depth is located in `depth_path`.

        :raise: RuntimeError
            - if width or height are non positive
            - if extrinsic_matrix_world2cam is not of shape (4, 4) or (3, 4),
            - if intrinsic_matrix is not of shape (3, 3).
        `"""

        if width <= 0 or height <= 0:
            raise RuntimeError(
                "The `width` and `height` must be positive, but got"
                + f"width={width}, height={height}."
            )

        if not (
            extrinsic_matrix_world2cam.shape == (4, 4)
            or extrinsic_matrix_world2cam.shape == (3, 4)
        ):
            raise RuntimeError(
                "The `extrinsic_matrix_world2cam` must have shape (4, 4) or (3, 4)"
                + f", but got {extrinsic_matrix_world2cam.shape}."
            )

        if intrinsic_matrix.shape != (3, 3):
            raise RuntimeError(
                "intrinsic_matrix must have shape (3, 3), but got "
                + f"{intrinsic_matrix.shape}."
            )

        self._width = width
        self._height = height
        self._image_path = image_path
        self._depth_path = depth_path
        if torch.is_tensor(extrinsic_matrix_world2cam):
            extrinsic_matrix_world2cam = extrinsic_matrix_world2cam.numpy()
        self._extrinsic_matrix_world2cam = extrinsic_matrix_world2cam

        if torch.is_tensor(intrinsic_matrix):
            intrinsic_matrix = intrinsic_matrix.numpy()
        self._intrinsic_matrix = intrinsic_matrix

    def to_dict(self) -> dict[str, int | float | str | list[list[float]]]:
        """
        Returns a dictionary representation of the frame frame with the following
        fields:
            - transform_matrix: The 3x4 camera extrinsic matrix (opencv
                world-to-camera transformation).
            - w: Image width in pixels.
            - h: Image height in pixels.
            - fl_x: Focal length in x direction (in pixels).
            - fl_y: Focal length in y direction (in pixels).
            - k1: Radial distortion coefficient k1 (set to 0).
            - k2: Radial distortion coefficient k2 (set to 0).
            - k3: Radial distortion coefficient k3 (set to 0).
            - k4: Radial distortion coefficient k4 (set to 0).
            - p1: Tangential distortion coefficient p1 (set to 0).
            - p2: Tangential distortion coefficient p2 (set to 0).
            - cx: Principal point x-coordinate (in pixels).
            - cy: Principal point y-coordinate (in pixels).
            - camera_angle_x: Horizontal field of view in radians.
            - camera_angle_y: Vertical field of view in radians.
            - fovx: Horizontal field of view in degrees.
            - fovy: Vertical field of view in degrees.
            - file_path: Path to the image file.
            - depth_file_path: Path to the depth map file.
        """

        frame = {}
        frame["transform_matrix"] = self._extrinsic_matrix_world2cam.tolist()
        frame["fl_x"] = self._intrinsic_matrix[0, 0].item()
        frame["fl_y"] = self._intrinsic_matrix[1, 1].item()
        frame["cx"] = self._intrinsic_matrix[0, 2].item()
        frame["cy"] = self._intrinsic_matrix[1, 2].item()
        frame["w"] = self._width
        frame["h"] = self._height
        frame["k1"] = 0
        frame["k2"] = 0
        frame["k3"] = 0
        frame["k4"] = 0
        frame["p1"] = 0
        frame["p2"] = 0
        frame["camera_angle_x"] = np.arctan(frame["w"] / (frame["fl_x"] * 2)) * 2
        frame["camera_angle_y"] = np.arctan(frame["h"] / (frame["fl_y"] * 2)) * 2
        frame["fovx"] = frame["camera_angle_x"] * 180 / np.pi
        frame["fovy"] = frame["camera_angle_y"] * 180 / np.pi

        frame["file_path"] = str(self._image_path)
        frame["depth_file_path"] = str(self._depth_path)

        return frame

    @staticmethod
    def dict_to_matrices(
        frame_dict: dict[str, float | int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a dictionary `frame_dict` of camera parameters into extrinsic and
        intrinsic matrices.

        :return: a tuple containing
            - extrinsic of shape (3, 4)
            - instinsic matrices of shape (3, 3).
        """

        file_path = frame_dict.get("file_path", "unknown")

        extrinsic = torch.tensor(frame_dict["transform_matrix"])
        if extrinsic.shape != (3, 4) and extrinsic.shape != (4, 4):
            raise RuntimeError(
                "The extrinsic must be of shape (3, 4) or (4, 4) but got "
                + f"{extrinsic.shape} for file {file_path}."
            )
        extrinsic = extrinsic[:3, :4]

        intrinsic = torch.tensor(
            [
                [frame_dict["fl_x"], 0.0, frame_dict["cx"]],
                [0.0, frame_dict["fl_y"], frame_dict["cy"]],
                [0.0, 0.0, 1.0],
            ]
        )

        return extrinsic, intrinsic
