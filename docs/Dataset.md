# Dataset

If you want use your own dataset, it must have the following format:

```
dataset/
├── scene1/
│   ├── images/
│   ├── thermal/
│   ├── depths_rgb/
|   ├── depths_thermal/
│   └── transforms.json
├── scene2/
│   ├── images/
│   ├── thermal/
│   ├── depths/
│   └── transforms.json
├── scene3/
│   ├── images/
│   ├── thermal/
│   ├── depths/
│   └── transforms.json
└── ...
```

## The `transforms.json` format

The `transforms.json` of a scene has the following format:

```json
{
    # we usually specify the type of the dataset
    "type": "oru",
    "frames": [
        {
            # In our training and eval scenes we have the same number of rgb and thermal images.
            # Each frame item of `transforms.json` stores a pair of an rgb and a thermal image.
            # For some scenes the paires follow the natural logic,
            # e.g. (1) rgb and thermal image have the same extrinsic (2) rgb and thermal image were captured at the same time.
            # for scenes without this natural logic the pairing process is done randomly.
            "rgb": {
                "transform_matrix": [ # world-to-camera opencv rgb camera extrinsic
                    [
                        -0.8362690210342407,
                        0.5481734871864319,
                        0.012648627161979675,
                        -33.551666259765625
                    ],
                    [
                        -0.05465104430913925,
                        -0.060375701636075974,
                        -0.9966785311698914,
                        -3.187180757522583
                    ],
                    [
                        -0.545589029788971,
                        -0.8341826796531677,
                        0.08044858276844025,
                        -50.347007751464844
                    ]
                ],
                "fl_x": 782.4822387695312, # horizontal focal distance of rgb camera in pixels
                "fl_y": 782.6173706054688, # vertical focal distance of rgb camera in pixels
                "cx": 512.6800537109375, # horizontal optical center of rgb image in pixels
                "cy": 384.0226135253906, # vertical optical center of rgb image in pixels
                "w": 1024, # width
                "h": 768, # height
                # the next fields are distortion coefficients.
                "k1": 0,
                "k2": 0,
                "k3": 0,
                "k4": 0,
                "p1": 0,
                "p2": 0,
                "camera_angle_x": 1.1588234050902337, # horizontal field-of-view in radians of rgb image
                "camera_angle_y": 0.912297438308035, # vertical field-of-view in radians of rgb image
                "fovx": 66.39569031264931, # horizontal field-of-view in degrees of rgb image
                "fovy": 52.270792875647, # vertical field-of-view in degrees of rgb image
                "file_path": "images/00360.png", relative path of rgb image
                "depth_file_path": "depths_rgb/00360.npy", relative path of depth map corresponding to rgb image
                "type": "rgb" # we usually specify the type of the frame
            },
            "thermal": {
                # world-to-camera opencv thermal camera extrinsic
                "transform_matrix": [
                    [
                        -0.8573916554450989,
                        0.5076297521591187,
                        -0.08480320870876312,
                        -35.57402801513672
                    ],
                    [
                        0.012285392731428146,
                        -0.14454002678394318,
                        -0.9894227385520935,
                        -1.5796369314193726
                    ],
                    [
                        -0.5145178437232971,
                        -0.8493646383285522,
                        0.11769098043441772,
                        -49.03438949584961
                    ]
                ],
                "fl_x": 1288.000732421875, # horizontal focal distance of thermal camera in pixels
                "fl_y": 1295.39892578125, # vertical focal distance of thermal camera in pixels
                "cx": 512.6038818359375, # horizontal optical center of thermal image in pixels
                "cy": 384.2519836425781, # vertical optical center of thermal image in pixels
                "w": 1024, # width
                "h": 768, # height
                # the next fields are distortion coefficients.
                "k1": 0,
                "k2": 0,
                "k3": 0,
                "k4": 0,
                "p1": 0,
                "p2": 0,
                "camera_angle_x": 0.7567251249431306, # horizontal field-of-view in radians of thermal image
                "camera_angle_y": 0.5763636602212376, # vertical field-of-view in radians of thermal image
                "fovx": 43.35715591075129, # horizontal field-of-view in degrees of thermal image
                "fovy": 33.02320519538913, # vertical field-of-view in degrees of thermal image
                "file_path": "thermal/00360.png", relative path of thermal image
                "depth_file_path": "depths_thermal/00360.npy", relative path of depth map corresponding to thermal image
                "type": "thermal" # we usually specify the type of the frame
            }
        },
        ...
        # other frames
    ]
}
```

We want to highlight again that the transform matrices are in **World-to-Camera OpenCV** format.

## The folders format

Next, we consider the folders format.
Typically, there are two folders containing images: `images` (for RGB) and `thermal` (for thermal).
However, the naming of these two folders is optional, since our DataLoaders rely only on the `file_path` fields.
One may even put all the images in one folder but specify the correct `file_path`.

The situation for depth maps is similar: one can either have two distinct folders for RGB and thermal modalities or any other structure, as long as the correct relative paths are specified in `depth_file_path`

The images must be readable using `PIL.Image.open`, while depth files must be readable using `numpy`.

## Processing the Datasets

### Processing ThermoScenes

1. Download data from [Zenodo](https://zenodo.org/records/10835108?)

2. For each scene run the script:

    ```bash
    python3 sear/scripts/vggt_rgb_depths_and_cameras.py \
        --model_path /path/to/the/original/VGGT/model/checkpoint \
        --scene_dir /path/to/the/ThermoNeRF/scene/folder/ \
        --output_dir /where/to/store/the/result/scene/
    ```

    The script uses the original VGGT model to compute depth maps as well as camera extrinsic and intrinsic parameters for all RGB images.
    It assumes that the RGB and thermal images are captured from the same viewpoints and share identical camera intrinsics.
    Based on this assumption, each thermal image is assigned the depth map and camera extrinsic and intrinsic parameters of its corresponding RGB image.

### Processing ThermalNeRF

Follow the [documentation](./sear/scripts/thermalnerf/README.md)

### Processing ThermalMix

1. Download data from [Zenodo](https://zenodo.org/records/11065834)
2. For each scene run the script:

    ```bash
    python3 sear/scripts/vggt_rgb_depths_and_cameras.py \
        --model_path /path/to/the/original/VGGT/model/checkpoint \
        --scene_dir /path/to/the/ThermoNeRF/scene/folder/ \
        --output_dir /where/to/store/the/result/scene/ \
        --rgb_images_name rgb \
        --thermal_images_name warped_thermal
    ```

### Processing ThermalGaussian

Follow the [documentation](./sear/scripts/thermalgaussian/README.md)

### Processing Orebro

If one wishes to process the RF dataset then they should Install ROS2.
The instructions for fedora ROS2 installation are provided in [install_ros2.md](./docs/install_ros2.md)

Follow the [documentation](./sear/scripts/oru/README.md)
