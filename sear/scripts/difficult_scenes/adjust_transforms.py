"""
The script to process DifficultScenes dataset of rgb and thermal with at least two
trajectories. The script updates the transforms.json file by specifying the indices
frames belonging to of rgb or thermal trajectory.

The DifficultScenes dataset contains six scenes: The trajectory indices used to create
the split are:

- conference-room: --rgb-trajectory IMG_6689.png IMG_6722.png
- metallic-container: --rgb-trajectory IMG_5990.png IMG_6040.png
- old-drinking-fountain: --rgb-trajectory IMG_5664.png IMG_5734.png
- parking: --rgb-trajectory IMG_6334.png IMG_6373.png
- statue: --rgb-trajectory IMG_5802.png IMG_5917.png
- telescope: --rgb-trajectory IMG_6517.png IMG_6558.png
"""

import json
import shutil
from pathlib import Path

import tyro


def main(
    scene_path: Path,
    rgb_trajectory: tuple[str, str],
) -> None:
    """
    Creates transforms.json specifying the split into rgb and thermal trajectory for two
    trajectories. The `rgb_trajectory` are inclusive filename range (start, end) used to
    select frames for the RGB trajectory.

    :raises: RuntimeError: If `scene_path` is invalid
    """

    if not scene_path.exists() or not scene_path.is_dir():
        raise RuntimeError(f"The `scene_path` {scene_path} is not a directory")

    with open(scene_path / "transforms.json") as f:
        transforms = json.load(f)

    indices_rgb: list[int] = []
    indices_thermal: list[int] = []
    for index, frame in enumerate(transforms["frames"]):
        rgb_file = Path(frame["rgb"]["file_path"])

        if rgb_trajectory[0] <= rgb_file.name <= rgb_trajectory[1]:
            indices_rgb.append(index)
            frame["rgb"]["is_rgb_trajectory"] = True
            frame["thermal"]["is_rgb_trajectory"] = True
        else:
            indices_thermal.append(index)
            frame["rgb"]["is_rgb_trajectory"] = False
            frame["thermal"]["is_rgb_trajectory"] = False

    transforms["rgb_trajectory"] = indices_rgb
    transforms["thermal_trajectory"] = indices_thermal
    shutil.copy(scene_path / "transforms.json", scene_path / "transforms_original.json")
    with open(scene_path / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)


if __name__ == "__main__":
    tyro.cli(main)
