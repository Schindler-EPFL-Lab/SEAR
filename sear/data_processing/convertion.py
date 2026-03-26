import torch


def opengl_to_opencv(extrinsics_cam2world_opengl: torch.Tensor) -> torch.Tensor:
    """
    Convert cameras in camera-to-world OpenGL format into camera-to-world OpenCV format.

    The OpenGL one is:
        x - right, y - up, z - backward
    The OpenCV one is:
        x - right, y - down, z - forward


    """
    if extrinsics_cam2world_opengl.ndim not in [
        2,
        3,
    ] or extrinsics_cam2world_opengl.shape[-2:] not in [(3, 4), (4, 4)]:
        raise RuntimeError(
            "The `extrinsics_cam2world_opengl` must be of shape (3, 4), (4, 4), "
            + f"(N, 3, 4), (N, 4, 4) but got {extrinsics_cam2world_opengl.shape}"
        )

    transformation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    extrinsics_cam2world_opencv = extrinsics_cam2world_opengl.clone()
    extrinsics_cam2world_opencv[..., :3, :3] = torch.matmul(
        extrinsics_cam2world_opencv[..., :3, :3], transformation
    )
    return extrinsics_cam2world_opencv
