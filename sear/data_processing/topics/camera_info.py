from datetime import datetime

import numpy as np
import numpy.typing as npt

from sear.data_processing.topics.base import BaseTopic


class CamInfo(BaseTopic):
    """
    Represents camera intrinsic parameters and image dimensions at a specific timestamp.
    """

    def __init__(
        self,
        timestamp: datetime,
        intrinsic: npt.NDArray[np.float32],
        height: int,
        width: int,
    ) -> None:
        """
        Initializes the CamInfo with the `timestamp` when camera intrinsics parameters
        were measured, the `intrinsic` representing 3x3 intrinsic camera parameters, and
        (`height`, `width`) representing image resolution.

        :raise: RuntimeError if intrinsic is not of shape 3x3 or height < 0 or width < 0
        """
        if intrinsic.shape != (3, 3):
            raise RuntimeError(
                f"intrinsic must be of shape 3, 3 but got {intrinsic.shape}"
            )
        if height < 0:
            raise RuntimeError(f"height must be non-negative, but get {height}")
        if width < 0:
            raise RuntimeError(f"width must be non-negative, but get {width}")

        super().__init__(timestamp=timestamp)
        self._intrinsic = intrinsic.copy()
        self._height = height
        self._width = width

    @property
    def intrinsic(self) -> npt.NDArray[np.float32]:
        return self._intrinsic

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @classmethod
    def from_dict(
        cls,
        timestamp: datetime,
        data: dict[str, dict[str, int | bool] | int | str | list[float]],
    ) -> "CamInfo":
        """
        Instantiates CamInfo from a `timestamp` and a dict storing intrinsic parameters.
        """

        intrinsic = np.array(data["k"]).reshape(3, 3).astype(np.float32)
        width = int(data["width"])
        height = int(data["height"])
        return cls(timestamp=timestamp, intrinsic=intrinsic, height=height, width=width)
