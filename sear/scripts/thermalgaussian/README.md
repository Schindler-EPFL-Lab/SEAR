# ThermalGaussian Dataset Processing

## Data Installation

One should download the sequences using a link from the official [github](https://github.com/chen-hangyu/Thermal-Gaussian-main) repo.
One should donwload only `RGBT-Scenes.zip` and `RGB-Scenes-extend.zip` and unzip them. Place the all the scenes from folders `RGBT-Scenes` and `RGBT-Scenes-extend` in one folder.
We would use a subset of 14 scene in our experiments: "Dark Scenes" "Glass Cup" "Plant Equipment" "Transmission Tower" (from `RGBT-Scenes-extend.zip`) and "Building", "Ebike", "IronIngot", "Parterre", "RoadBlock", "DailyStuff", "Dimsum", "LandScape", "RotaryKiln", "Truck" (from `RGBT-Scenes.zip`).

Unfortunately, the provided thermal images are not grayscale but a "fake rgb", which is peformed using a transformation from grayscale into other color palette (e.g. magma).
We do not currently know what color palette they used and how to properly revert the "fake rgb" into real grayscale.
There is already an [issue](https://github.com/chen-hangyu/Thermal-Gaussian-main/issues/13) related to this question.
But we noticed that thermal images extracted from EXIF information from `raw_images` folders are aligned with the provided thermal images.
Therefore we extract those images and replace the provided thermal images with the real thermal ones.

## Extract Thermal Images

1. Install [thermo-nerf](https://github.com/Schindler-EPFL-Lab/thermo-nerf) and activate the uv environment if not installed.
2. For each scene in `RGBT-Scenes` and `RGBT-Scenes-extend` run:

    ```bash
    thermoscenes_preprocess_thermal \
        --msx-images /path/to/raw/thermalgaussian/scene/raw_images \
        --output-folder /path/where/to/store/extracted/rgb/and/thermal/images
    ```

    Please place the extracted scenes in one folder.

    During convertion of some scenes we got errors, so for those we will take the provided thermal images and convert them to grayscale.

## Replace Thermal Images

Run the script:

```bash
python3 sear/scripts/thermalgaussian/replace_thermal.py \
    --thermalgaussian-root /path/to/the/original/thermalgaussian/dataset/ \
    --extracted_thermalgaussian_root /path/to/the/scenes/with/extracted/exif/data/ \
    --output-folder /where/to/store/the/result/
```

For each scene (with name `scene_name`) the script will take RGB images from "/path/to/the/original/thermalgaussian/dataset/`scene_name`" and thermal images from "/path/to/the/scenes/with/extracted/exif/data/`scene_name`" if the extraction was successful.
If it was not successful the script would take thermal images from "/path/to/the/original/thermalgaussian/dataset/`scene_name`" and convert them to grayscale.
The output would be stored at "/where/to/store/the/result/".

## Find Ground Truth Depths and Poses

For each scene with corrected thermal data from the [previous step](#replace-thermal-images) run:

```bash
python3 sear/scripts/vggt_rgb_depths_and_cameras.py \
    --model_path /path/to/original/VGGT/model/checkpoint \
    --scene_dir /path/to/the/scene/with/corrected/thermal/data/ \
    --output_dir /where/to/store/the/result/scene
```

