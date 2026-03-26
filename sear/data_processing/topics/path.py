from datetime import datetime
from pathlib import Path

from sear.data_processing.topics.base import BaseTopic


class PathToObject(BaseTopic):
    def __init__(self, timestamp: datetime, path: Path) -> None:
        """
        Instantiates PathToObject with `path` and `timestamp` which is the time when the
        file at `path` was measured.
        """
        super().__init__(timestamp=timestamp)
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    @classmethod
    def from_path(cls, path: Path) -> "PathToObject":
        """

        Creates a PathToObject from `path`, which must follow the template:
        <name>_<index>_<timestamp *10^9>.ext, for example
        ueye_ids_camera_image_raw_compressed_0000_1764160636747756319.png, where - name
        = ueye_ids_camera_image_raw_compressed - index = 0000 - timestamp * 10^9 =
        1764160636747756319

        :return: a PathToObject instance.
        """
        file_name = path.stem
        timestamp_str = file_name.split("_")[-1]
        timestamp_int = int(timestamp_str)
        timestamp = datetime.fromtimestamp(timestamp_int / 1e9)
        return cls(timestamp=timestamp, path=path)

    @classmethod
    def from_folder(cls, folder_path: Path) -> list["PathToObject"]:
        """
        Creates a list of PathToObject instances from `folder_path`. Assumes that files
        in the `folder_path` have following names: <name>_<index>_<timestamp *10^9>.ext,
        for example ueye_ids_camera_image_raw_compressed_0000_1764160636747756319.png,
        where - name = ueye_ids_camera_image_raw_compressed - index = 0000 - timestamp *
        10^9 = 1764160636747756319

        :return: a set of PathToObject instances.
        """
        result: list[PathToObject] = []
        for file_path in folder_path.iterdir():
            result.append(PathToObject.from_path(path=file_path))
        return result
