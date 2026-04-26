# Inspect Features

This tutorial covers how to extract features or RGB and thermal tokens in the Alternating Attention module

Please download packages and checkpoints as described [here](./evaluation/Main.md#vggt)

## Distance Between Features

For VGGT run:

```bash
python3 ./sear/scripts/features_inspection/features_distance_original.py \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --original_vggt_path /path/to/vggt/weights.pth \
    --output_path /where/to/store/output.csv \
    # how many images (rgb and thermal) are processed together
    --max_num_images value
```

    For each scene script split it into chunks of size `max_num_images` and pass to the model separately. 
    Then it computes the distance between RGB and thermal features for each layer and each chunk and saves the distance to the output file.

For SEAR run:

```bash
python3 ./sear/scripts/features_inspection/features_distance_sear.py \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --thermal_vggt.vggt-path /path/to/original/vggt/weights.pth \
    --ckpt_path /path/to/sear/weights.pth \
    --aggregator.type AGGREGATOR_TYPE
    --output_path /where/to/store/output.csv \
    # how many images (rgb and thermal) are processed together
    --max_num_images value
```

## Features Visualization

Extract intermediate features using script, for VGGT run:

```bash
python3 ./sear/scripts/features_inspection/vggt_original_aggregator_features.py \
    --scene_dir /path/to/scene \
    --original_vggt_path /path/to/vggt/weights.pth \
    --output_dir /where/to/store/outputs/ \
    # how many images (rgb and thermal) are processed
    --max_num_images value \
    --thermal-ratio value
```

For SEAR run:
```bash
python3 ./sear/scripts/features_inspection/sear_aggregator_features.py \
    --scene_dir /path/to/a/scene \
    --thermal_vggt.vggt-path /path/to/original/vggt/weights.pth \
    --ckpt_path /path/to/sear/weights.pth \
    --aggregator.type AGGREGATOR_TYPE \
    --output_dir /where/to/store/outputs/
    --max_num_images value \
    --thermal-ratio value
```

    The script forwards only `max_num_images` of the scene (the ratio between RGB and thermal frames is calculated using the `thermal-ratio`). It saves the intermediate aggregator features and thermal mask.

Then run:
```bash
python3 sear/scripts/features_inspection/generate_features_pca_2d.py \
    --aggregator_output_folder /path/to/saved/aggregator/features/and/thermal/mask \
    --output_dir /where/to/store/point/clouds/
```

    The script computes PCA decomposition - it maps outputs of each layer into 2D (independently per layer)

In our work we compare the results when running SEAR on RGB+Thermal and VGGT on RGB-only, for this setting one should run:
```bash
python3 sear/scripts/features_inspection/generate_features_pca_2d_combined.py \
    --aggregator_output_folders /path/to/sear/aggregator/extracted/rgb+thermal /path/to/vggt/aggregator/extracted/rgb-only
```
    This script also computes PCA decomposition, but it first combined tokens for layers of different runs (SEAR and VGGT) and then maps them to 2D.
