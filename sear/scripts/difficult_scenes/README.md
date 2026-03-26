# DifficultScenes dataset

## Intro

This document describes how to process the `DifficultScenes` dataset collected using a FLIR camera, following the provided [instructions](https://github.com/Schindler-EPFL-Lab/thermo-nerf/blob/main/thermo_scenes/docs/Collect_new_dataset.md).

The dataset contains 9 scenes: conference-room, metallic-container, old-drinking-fountain, parking, statue, telescope, red-container, house, and messy-living-room. Each scene is captured along two trajectories. The dataset is divided into two parts:

1. **Part (1):** For each scene, both trajectories are captured under the same lighting conditions, and the trajectories do not intersect.
2. **Part (2):** For each scene, the two trajectories are captured under completely different lighting conditions (e.g., day and night). The trajectories usually intersect. This part of the dataset is used solely for visualization.

For **Part (1)**, ground-truth poses can be obtained and used to evaluate camera pose estimation results, whereas **Part (2)** is intended only for visualization.

## Poses Estimation

As mentioned above, it is possible to estimate camera poses for **Part (1)** of the dataset.  
We assume that the `images/` and `thermal/` folders have already been extracted.

To run the pose estimation, use the following script:

```bash
python3 sear/scripts/vggt_rgb_depths_and_cameras.py \
    --model_path /path/to/original/VGGT/model/checkpoint \
    --scene_dir /path/to/the/scene/folder/ \
    --output_dir /where/to/store/the/result/scene
```

This script estimates camera poses using the RGB images and the original VGGT model.
Note that this script is intended for the following scenes: conference-room, metallic-container, old-drinking-fountain, parking, statue, and telescope. 
It is not recommended for: red-container, house and messy-living-room since the RGB trajectories in these scenes may be too dark for reliable pose estimation.

## Specifying trajectories

### Part (1)

To process **Part (1)**, run:

```
python3 sear/scripts/difficult_scenes/adjust_transforms.py \
    --scene_path /path/to/scene/with/extracted/poses/
    --rgb_trajectory image_name_start.png image_name_end.png
```

RGB images with filenames between `image_name_start.png` and `image_name_end.png` (inclusive) are considered part of the RGB trajectory, while all other frames are assigned to the thermal trajectory.
This script updates the existing transforms.json by adding the indices of frames belonging to the RGB and thermal trajectories.

### Part (2)

To process Part (2), run:
    
```
python3 sear/scripts/difficult_scenes/create_transforms.py \
    --scene_path /path/to/scene/
    --rgb_trajectory image_name_start.png image_name_end.png
```

This script creates a new transforms.json containing the indices of frames belonging to the RGB and thermal trajectories.
