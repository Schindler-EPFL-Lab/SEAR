import json
import subprocess
from pathlib import Path

import tyro


def main(
    rosbag_path: Path,
    output_folder: Path,
    original_config_path: Path = Path("./sear/configs/config_unbag.json"),
) -> None:
    """
    Substitutes `output_folder` in the config from `original_config_path` and runs "ros2
    unbag" on a ros bag located in `rosbag_path`.
    """

    output_folder.mkdir(exist_ok=True, parents=True)
    output_config_path = output_folder / "config.json"
    with open(original_config_path) as f:
        config_data = json.load(f)

    for key in config_data:
        if "path" not in config_data[key]:
            continue
        config_data[key]["path"] = str(output_folder)

    with open(output_config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    subprocess.run(
        ["ros2", "unbag", str(rosbag_path), "--config", str(output_config_path)]
    )


if __name__ == "__main__":
    tyro.cli(main)
