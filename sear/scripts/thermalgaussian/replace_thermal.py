import shutil
from pathlib import Path

import cv2
import tyro


def replace_thermal_one_folder(
    original_scene_path: Path,
    output_folder: Path,
    extracted_scene_path: Path | None = None,
) -> None:
    """
    Takes RGB images from a scene located at `original_scene_path` and thermal images
    (extracted from EXIF data) from `extracted_scene_path`, and stores the results in
    `output_folder`. If `extracted_scene_path` is not provided, thermal images are taken
    from `original_scene_path` and converted to grayscale using
    cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)

    :raise: RuntimeError if the number of thermal images in `extracted_scene_path` is
        not equal to the number of rgb images in `original_scene_path`.
    """

    # Extract all rgb images in one folder
    all_rgb_images_paths: list[Path] = []
    splits = ["train", "test"]
    for split in splits:
        paths = list((original_scene_path / "rgb" / split).iterdir())
        paths = sorted(paths)  # make it deterministic
        all_rgb_images_paths.extend(paths)

    # Find path of thermal images to replace
    all_thermal_images_paths_extracted: list[Path] = []
    if extracted_scene_path is not None:
        all_thermal_images_paths_extracted = list(
            (extracted_scene_path / "thermal").iterdir()
        )
        all_thermal_images_paths_extracted = sorted(all_thermal_images_paths_extracted)
    else:
        for split in splits:
            paths = list((original_scene_path / "thermal" / split).iterdir())
            paths = sorted(paths)  # make it deterministic
            all_thermal_images_paths_extracted.extend(paths)

    if len(all_thermal_images_paths_extracted) != len(all_rgb_images_paths):
        raise RuntimeError(
            "The extracted data contain less images than the original data: extracted"
            + f" is {len(all_thermal_images_paths_extracted)} original is "
            + f"{len(all_rgb_images_paths)}."
        )

    output_folder.mkdir(exist_ok=True)
    output_folder_images = output_folder / "images"
    output_folder_images.mkdir(exist_ok=True)
    output_folder_thermal = output_folder / "thermal"
    output_folder_thermal.mkdir(exist_ok=True)

    random_rgb_image = cv2.imread(str(all_rgb_images_paths[0]))

    for image_path in all_rgb_images_paths:
        shutil.copy(image_path, output_folder_images / image_path.name)

    for image_path in all_thermal_images_paths_extracted:
        thermal_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if thermal_image.ndim == 3 and thermal_image.shape[2] == 3:  # fake thermal
            thermal_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)
        thermal_image = cv2.resize(
            thermal_image, (random_rgb_image.shape[1], random_rgb_image.shape[0])
        )
        thermal_image = thermal_image[:, :, None].repeat(3, axis=2)
        cv2.imwrite(str(output_folder_thermal / image_path.name), thermal_image)


def main(
    thermalgaussian_root: Path,
    extracted_thermalgaussian_root: Path,
    output_root: Path,
) -> None:
    """
    For each ThermalGaussian scene in `thermalgaussian_root`, the pipeline uses RGB
    images from `thermalgaussian_root` and thermal images from the corresponding scene
    folder in `extracted_thermalgaussian_root` if the scene was processed successfully.
    If a scene was not processed successfully, thermal images are taken directly from
    `thermalgaussian_root` and converted to grayscale. The resulting scenes are stored
    in output_root. Output scene names match the original dataset scene names, with
    spaces removed.
    """

    all_scenes = [
        "Dark Scenes",
        "Glass Cup",
        "Plant Equipment",
        "Transmission Tower",
        "Building",
        "Ebike",
        "IronIngot",
        "Parterre",
        "RoadBlock",
        "DailyStuff",
        "Dimsum",
        "LandScape",
        "RotaryKiln",
        "Truck",
    ]

    successfully_processed_scenes = [
        "Dark Scenes",
        "Ebike",
        "IronIngot",
        "RoadBlock",
        "DailyStuff",
        "Dimsum",
        "Glass Cup",
        "LandScape",
        "Plant Equipment",
        "RotaryKiln",
        "Truck",
    ]

    for scene in all_scenes:
        extracted_scene_path = None
        if scene in successfully_processed_scenes:
            extracted_scene_path = extracted_thermalgaussian_root / scene

        renamed_scene = scene.replace(" ", "")
        replace_thermal_one_folder(
            original_scene_path=thermalgaussian_root / scene,
            output_folder=output_root / renamed_scene,
            extracted_scene_path=extracted_scene_path,
        )


if __name__ == "__main__":
    tyro.cli(main)
