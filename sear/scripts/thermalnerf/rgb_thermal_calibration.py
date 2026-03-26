import json
import tempfile
from pathlib import Path

import torchvision
import tyro
from vggt.utils.load_fn import load_and_preprocess_images

from thermalnerf.calibration_utils import calibrate_rgb_thermal


def main(
    calibration_folder: Path,
    rgb_folder_name: str = "rgb",
    thermal_folder_name: str = "thermal",
    output_file: Path = Path("./sear/data_processing/thermalnerf/calibation.json"),
) -> None:
    """
    Performs RGB-Thermal camera intrinsics and transform from rgb to thermal using
    images from the `calibration_folder/rgb_folder_name` and
    `calibration_folder/thermal_folder_name` folders, and saves the calibration result
    to the `output_file`.

    The calibration result contains the following fields:
        - camera_matrix_rgb, camera_matrix_thermal: intrinsic rgb and thermal camera
          parameters respectively as 3x3 matrix
        - distortion_coeffs_rgb, distortion_coeffs_thermal: distortion coefficients for
          the rgb and thermal cameras respectively. They are small so we neglect them in
          the data processing.
        - rgb_thermal_transform, thermal_rgb_transform : rgb-to-thermal and
          thermal-to-rgb camera frames transformation in metric space respectively. The
          transformations are done in the metric space with 1.0 being equal to 1 cm.
          When applying those transformations on your poses make sure to ensure the same
          scaling.
    """

    with tempfile.TemporaryDirectory() as tempdict:
        tempdict_root = Path(tempdict)
        rgb_images_paths = sorted(
            list((calibration_folder / rgb_folder_name).iterdir())
        )
        thermal_images_paths = sorted(
            list((calibration_folder / thermal_folder_name).iterdir())
        )
        rgb_images = load_and_preprocess_images(rgb_images_paths)
        thermal_images = load_and_preprocess_images(thermal_images_paths)

        # save processed rgb and thermal images as temporary files
        (tempdict_root / rgb_folder_name).mkdir(exist_ok=True)
        for rgb_image_path, rgb_image in zip(rgb_images_paths, rgb_images):
            torchvision.utils.save_image(
                rgb_image, tempdict_root / rgb_folder_name / rgb_image_path.name
            )

        (tempdict_root / thermal_folder_name).mkdir(exist_ok=True)
        for thermal_image_path, thermal_image in zip(
            thermal_images_paths, thermal_images
        ):
            torchvision.utils.save_image(
                thermal_image,
                tempdict_root / thermal_folder_name / thermal_image_path.name,
            )

        calibration_result = calibrate_rgb_thermal(
            rgb_folders=[tempdict_root / rgb_folder_name],
            thermal_folders=[tempdict_root / thermal_folder_name],
            intrinsic_calibration_mode=4,
            force_radial_distortion_coeff_K3_to_zero=True,
            upsample_thermal=False,
            show_preview=False,
        )

    calibration_result = {k: v.tolist() for k, v in calibration_result.items()}

    with open(output_file, "w") as f:
        json.dump(calibration_result, f, indent=4)


if __name__ == "__main__":
    tyro.cli(main)
