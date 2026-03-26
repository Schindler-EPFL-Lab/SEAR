from datetime import datetime

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from sear.data_processing.topics.base import BaseTopic


class CamTF(BaseTopic):
    def __init__(
        self, timestamp: datetime, pose_cam2world: npt.NDArray[np.float32]
    ) -> None:
        """
        Instantiates CamTF with `pose_cam2world` is opencv world-to-camera camera pose
        and `timestamp` which is the time when the camera pose was measured.

        :raise: RuntimeError if pose_cam2world is not of shape (3, 4) or (4, 4).
        """

        if not (pose_cam2world.shape == (4, 4) or pose_cam2world.shape == (3, 4)):
            raise RuntimeError(
                "The `pose_cam2world` must have shape (4, 4) or (3, 4), but got "
                + f"{pose_cam2world.shape}."
            )

        super().__init__(timestamp=timestamp)
        self._timestamp = timestamp

        self._pose_cam2world = pose_cam2world.copy()

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @property
    def pose_cam2world(self) -> npt.NDArray[np.float32]:
        return self._pose_cam2world

    @classmethod
    def from_rotmat_and_translation(
        cls,
        timestamp: datetime,
        rotmat: npt.NDArray[np.float32],
        translation: npt.NDArray[np.float32],
    ) -> "CamTF":
        """
        Instantiates CamInfo from a `timestamp` and camera orientation (rotation matrix
        `rotmat`) and `translation`.
        """
        pose_cam2world = np.zeros((4, 4), dtype=np.float32)
        pose_cam2world[:3, :3] = rotmat
        pose_cam2world[:3, 3] = translation
        pose_cam2world[3, 3] = 1.0
        return cls(timestamp=timestamp, pose_cam2world=pose_cam2world)

    @classmethod
    def from_data(
        cls,
        timestamp: datetime,
        data: dict[
            str, dict[str, dict[str, int] | str] | str | dict[str, dict[str, float]]
        ],
    ) -> "CamTF":
        """
        Instantiates CamInfo from a `timestamp` and a `data` dict storing camera
        orientation and position.
        """

        transform = data["transform"]

        translation = np.array(
            [
                transform["translation"]["x"],
                transform["translation"]["y"],
                transform["translation"]["z"],
            ]
        ).astype(np.float32)

        quat = np.array(
            [
                transform["rotation"]["x"],
                transform["rotation"]["y"],
                transform["rotation"]["z"],
                transform["rotation"]["w"],
            ]
        ).astype(np.float32)

        rotmat = Rotation.from_quat(quat).as_matrix()
        return cls.from_rotmat_and_translation(
            timestamp=timestamp, rotmat=rotmat, translation=translation
        )
