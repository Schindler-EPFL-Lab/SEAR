"""
The script to process DifficultScenes dataset of rgb and thermal with at least two
trajectories. The script creates transforms.json file by specifying the indices frames
belonging to of rgb or thermal trajectory.

The DifficultScenes for visualization contains three scenes:
The trajectory indices used to create the split are:

- house:
    --rgb-trajectory IMG_6616.png IMG_6686.png
- red-container:
    --rgb-trajectory IMG_6207.png IMG_6261.png
- messy-living-room:
    --rgb-trajectory IMG_6768.png IMG_6819.png
"""

import json
from pathlib import Path

import tyro


def main(
    scene_path: Path,
    rgb_trajectory: tuple[str, str],
) -> None:
    """
    Creates transforms.json specifying the split into rgb and thermal trajectory for two
    trajectories. The `rgb_trajectory` are inclusive filename range (start, end) used to
    select frames for the RGB trajectory. The `thermal_trajectory` are inclusive
    filename range (start, end) used to select frames for the thermal trajectory.

    :raises: RuntimeError: If `scene_path` is invalid or if the discovered RGB/Thermal
        file counts are inconsistent.
    """

    if not scene_path.exists() or not scene_path.is_dir():
        raise RuntimeError(f"The `scene_path` {scene_path} is not a directory")

    rgb_files = list((scene_path / "images").iterdir())
    thermal_files = list((scene_path / "thermal").iterdir())

    rgb_files = sorted(rgb_files)
    thermal_files = sorted(thermal_files)

    if len(rgb_files) != len(thermal_files):
        raise RuntimeError(
            "The lengths of found rgb and thermal files must be consistent "
            + f"but got {len(rgb_files)} and {len(thermal_files)} respectively."
        )

    indices_rgb: list[int] = []
    indices_thermal: list[int] = []

    transforms: dict[
        str,
        str
        | list[int]
        | float
        | list[dict[str, bool | str | float | list[list[float]]]],
    ] = {}
    transforms["type"] = "DifficultScenesVisualization"
    transforms["frames"] = []

    for index, rgb_file, thermal_file in zip(
        range(len(rgb_files)), rgb_files, thermal_files
    ):
        if rgb_trajectory[0] <= rgb_file.name <= rgb_trajectory[1]:
            is_rgb_trajectory = True
            indices_rgb.append(index)
        else:
            is_rgb_trajectory = False
            indices_thermal.append(index)

        transforms["frames"].append(
            {  # type: ignore
                "is_rgb_trajectory": is_rgb_trajectory,
                "rgb": {
                    "type": "rgb",
                    "file_path": str(rgb_file.relative_to(rgb_file.parent.parent)),
                },
                "thermal": {
                    "type": "thermal",
                    "file_path": str(
                        thermal_file.relative_to(thermal_file.parent.parent)
                    ),
                },
            }
        )

    transforms["rgb_trajectory"] = indices_rgb
    transforms["thermal_trajectory"] = indices_thermal

    with open(scene_path / "transforms.json", "w") as f:
        json.dump(transforms, f, indent=4)


if __name__ == "__main__":
    tyro.cli(main)
