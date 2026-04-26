# Pairs Validation

## MatchAnything and MINIMA

1. Install [METU_VisTIR](https://drive.google.com/file/d/1Sj_vxj-GXvDQIMSg-ZUJR0vHBLIeDrLg/view?usp=sharing) and unzip
2. Install packages and checkpoints as described [here](./Main.md#matchanything-and-minima)
3. For MatchAnything run:

```bash
python3 sear/scripts/eval/relative_camera_pose/match_relative_camera_pose.py \
    --scenes_root_path /path/to/uncompressed/METU_VisTIR/ \
    --store_results_folder /where/to/store/relative/poses/ \
    --output_dir /where/to/store/cache/and/metrics/
```

4. For MatchAnything run:

```bash
python3 sear/scripts/eval/relative_camera_pose/minima_relative_camera_pose.py \
    --scenes_root_path /path/to/uncompressed/METU_VisTIR/ \
    --store_results_folder /where/to/store/relative/poses/ \
    --checkpoint_path /path/to/minima-roma/weights.pth \
    --output_dir /where/to/store/cache/and/metrics/
```

## VGGT

Install packages and checkpoints as described [here](./Main.md#vggt) and run:

```bash
python3 ./sear/scripts/eval/relative_camera_pose/vggt_relative_camera_pose.py \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --store_results_folder /path/to/store/poses/and/depths \
    --original_vggt_path /path/to/vggt/weights.pth \
    --output_dir /where/to/store/cache/and/metrics/
```

## SEAR

Run:

```bash
python3 ./sear/scripts/eval/relative_camera_pose/sear_relative_camera_pose.py \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --store_results_folder /path/to/store/poses/and/depths \
    --thermal_vggt.vggt-path /path/to/original/vggt/weights.pth \
    --ckpt_path /path/to/sear/weights.pth \
    --aggregator.type AGGREGATOR_TYPE
    --output_dir /where/to/store/cache/and/metrics/
```
