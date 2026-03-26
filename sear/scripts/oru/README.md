# ORU Dataset processing

## Intro

This document describes how to process the dataset provided by the [Orebro University](https://www.oru.se/english/), [Martin Magnusson](https://www.oru.se/english/employee/martin_magnusson) group.
The dataset contains visible and thermal images, their poses (from motion capture system), and depths (from LiDAR).

## Installation

1. Install ROS2 following the instructions from [install_ros2.md](../../../docs/install_ros2.md).
2. Install ROS2 plugin `image_transport_plugins` to read compressed images (rgb and thermal).

    ```bash
    cd ~/ros2_kilted/src
    git clone https://github.com/ros-perception/image_transport_plugins.git
    git clone https://github.com/ros-perception/vision_opencv.git
    rosdep update
    rosdep install --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y
    cd ..
    colcon build --symlink-install
    source install/setup.bash
    ```

3. Install [ros-unbag](https://pypi.org/project/ros2-unbag/1.0.6/) using pypi or uv.


## Data Description

The dataset has topics:

```
/a200_1750/platform/cmd_vel
/a200_1750/platform/odom
/cx/lslidar_point_cloud
/cx/lslidar_point_cloud_deskewed
/emlid_gnss/fix
/emlid_gnss/heading
/emlid_gnss/time_reference
/emlid_gnss/vel
/events/read_split
/flir_camera/camera_info
/flir_camera/image_raw/compressed
/icp_odom
/imu/data
/imu/mag
/imu_odom
/map
/parameter_events
/rosout
/tf
/tf_static
/ueye/ids_camera/camera_info
/ueye/ids_camera/image_raw/compressed
```

We only need the following:

### LiDAR point clouds

- /cx/lslidar_point_cloud
- /cx/lslidar_point_cloud_deskewed

### Camera extrinsic information

- /tf: orientation of the robot (odom_base_link) and measuring system (map_odom)
- /tf_static: orientation of the cameras and LiDAR relative to the robot (base_link)

The relationship between the sensors positions is the following:
```
 map -.
      |
    odom -.
          |
      base_link -.
                 |
              radar_1
              radar_2
              radar_3
                 |
             flir_camera
             ids_camera
                 |
             laser_link
                 |
              imu_link
```

### Camera intrinsic information

- /ueye/ids_camera/camera_info: visible camera intrinsics
- /flir_camera/camera_info: flir camera intrinsics

### Images

- /flir_camera/image_raw/compressed
- /ueye/ids_camera/image_raw/compressed


## Data Extraction

Run:

```bash
python3 sear/scripts/oru/exract.py \
    --rosbag-path /path/to/rosbag.mcap \
    --output-folder /path/to/extracted/data \
    --original-config-path /path/to/original/config.json
```

## Data Processing

Run:

```bash
python3 \
    sear/scripts/oru/convert_raw_to_thermoscenes.py \
    --root-path /path/to/extracted/data \
    --output-folder /path/to/save/processed/data \
    --tf_static_path /path/to/tf_static.json
```

Where /path/to/extracted/data is the path you specified in the [data extraction](#data-extraction).
Regarding the `tf_static_path`, it might be recorded only for the first scene (e.g. 01_Annexet_No_Radars_0.mcap but not for 01_Annexet_No_Radars_1.mcap), therefore one must firstly process 01_Annexet_No_Radars_0.mcap and then use its tf_static data to process 01_Annexet_No_Radars_1.mcap.

## Data Notes

- 4/4 Done
- 01_Annexet_No_Radars_0 - Robot moving & people moving
- 01_Annexet_No_Radars_1 - Robot moving & people moving (but some parts does not contain moving people)
- 01_Annexet_No_Radars_2 - Robot moving & people moving (but not as much as in 01_0)
- 01_Annexet_No_Radars_3 - Robot moving & people moving (quite short)

- 4/4 Done
- 02_Fox_area_no_radars_0 - Robot moving & people moving (bright sun sometimes, very dynamic scene)
- 02_Fox_area_no_radars_1 - Robot moving & people moving (again bright sun sometimes + robot stops sometimes)
- 02_Fox_area_no_radars_2 - Robot moving & people moving (but not as much as in 01_0)
- 02_Fox_area_no_radars_3 - Robot doesn't move & people moving

- 1/2 Done
- 03_Static_with_group_no_radars_0 : Robot doesn't move & people moving -> do not want this

- 2/2 Done
- 04_Forest_pass_no_radars_0 - Robot moving & people moving (very dynamic)
- 04_Forest_pass_no_radars_1 - Robot moving & people moving

- 2/2 Done
- 05_Return_from_the_forest_no_radar_0 - Robot moving & people moving (very dynamic)
- 05_Return_from_the_forest_no_radar_1 - Robot moving & people moving (Quite short)

- 3/3 Done
- 06_Sunset_area_no_radars_0 - Robot moving & people moving (again very dynamic)
- 06_Sunset_area_no_radars_1 - Robot moving & people moving (dynamic)
- 06_Sunset_area_no_radars_2 - Robot moving & people moving
