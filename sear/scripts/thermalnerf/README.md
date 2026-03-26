# ThermalNeRF Dataset

This file contains information regarding processing of the [ThermalNeRF](https://yvette256.github.io/thermalnerf/) dataset.

## Dataset downloading

Install Raw data from [ThermalNeRF](https://yvette256.github.io/thermalnerf/).
It should (for 13.12.2025) contain 9 scenes: "calibration-data", "charger", "engine", "generator", "generators", "heater", "pyrex", "sheet", "sink", "trace".
The scene "calibration-data" is used to find the transformation matrix between rgb and thermal images.
All the datasets contain raw images (from flir) as jpg files.
In the EXIF data of those jpe files one can find the real rgb and thermal data.

## Extraction of rgb and thermal images from the EXIF data

1. Install [thermo-nerf](https://github.com/Schindler-EPFL-Lab/thermo-nerf) and activate the uv environment if not installed.
2. For every scene run:

    ```bash
    thermoscenes_preprocess_thermal --msx-images /path/to/raw/thermalnerf/scene/ --output-folder /path/where/to/store/extracted/rgb/and/thermal/images
    ```

## Estimation of poses and depth of rgb images
For each processed scene run

```bash
python3 sear/scripts/vggt_rgb_depths_and_cameras.py \
    --model_path /path/to/original/VGGT/model/checkpoint \
    --scene_dir /path/to/the/scene/folder/ \
    --output_dir /where/to/store/the/result/scene
```
This script estimates the extrinsics and intrinsics parameters of rgb cameras and depth maps of rgb images.
Although the script produces camera parameters and depths for thermal modality, they are incorrect and be fixed in the [next section](#correction-of-poses-and-depth-of-thermal-images).

## Correction of poses and depth of thermal images

1. One needs to get the calibration data, which contains correct thermal camera intrinsics and transformation between an rgb camera frame to the corresponding thermal camera frame.
    To get this [calibration data](./calibation.json) one should run

    ```bash
    python3 sear/data_processing/thermalnerf/rgb_thermal_calibration.py --calibration_folder /path/to/extracted/"calibration-data"/scene
    ```

    The script finds the visible camera intrinsics, thermal camera intrinsics, and the transformations between modalities frames in the **metric space**.
    This space differs from the space obtained from the [previous step](#estimation-of-poses-and-depth-of-rgb-images) in scale.
    The authors assume that the distance between the 3d and 4th frames is exactly 1 ft.
    This information is used to find the convertion between scales of those spaces.

2. For every extracted scene one should run

    ```bash
    python3 sear/data_processing/thermalnerf/correct_thermal_poses_and_depth.py --root_path /path/to/scene/with/rgb/and/thermal/depths/and/poses
    ```

    The script updates extrinsic and intrinsic parameters of thermal cameras, and find the correct depth.
    It overwrites the thermal data in-place.

