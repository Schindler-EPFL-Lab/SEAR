import numpy as np
import pyvista as pv


def plot_cameras_from_open3d(cameras: pv.PolyData) -> pv.PolyData:
    """
    A helper function that creates a camera representation via set of cuts from `points`
    (corners of the pyramid) read from Open3D.

    :return: segment-like representation of a camera.
    """
    points = np.asarray(cameras.points)
    points = points.reshape(-1, 5, 3)
    segments = []
    for cam_id in range(points.shape[0]):
        shift = cam_id * 5
        for i in range(1, 5):
            segments.append([2, 0 + shift, i + shift])
        # for i, j in [(1, 2), (2, 3), (3, 4), (4, 1)]:
        for i, j in [(1, 2), (2, 4), (3, 1), (4, 3)]:
            segments.append([2, i + shift, j + shift])

    lines = np.array(segments, dtype=np.int64).ravel()
    poly = pv.PolyData(points.reshape(-1, 3))
    poly.lines = lines
    return poly
