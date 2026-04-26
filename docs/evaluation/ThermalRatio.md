# Thermal Ratio Validation 

1. Install packages and checkpoints as described [here](./Main.md#vggt)
2. For VGGT run:

```bash
python3 ./sear/scripts/eval/thermal_ratio/original_vggt.py \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --store_results_folder /path/to/store/poses/and/depths \
    --original_vggt_path /path/to/vggt/weights.pth \
    --output_dir /where/to/store/cache/and/metrics/
    # integer value from 0 to 100
    --thermal-percent value  \ 
    # how many repeats to do for one scene
    --num_repeat value
```

3. For SEAR run:

```bash
python3 ./sear/scripts/eval/sear.py \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --store_results_folder /path/to/store/poses/and/depths \
    --thermal_vggt.vggt-path /path/to/original/vggt/weights.pth \
    --ckpt_path /path/to/sear/weights.pth \
    --aggregator.type AGGREGATOR_TYPE
    --output_dir /where/to/store/cache/and/metrics/
    # integer value from 0 to 100
    --thermal-percent value  \ 
    # how many repeats to do for one scene
    --num_repeat value
```

Each scripts runs a method `num_repeat` for every eval scene (excluding SEAR-Dataset scenes since thermal values there are fixed by the trajectories split) and saves the results in `store_results_folder`. 

4. To further enhance the result one can calculate statistics using bootstrap by running the script:
```bash
python3 eval_using_predictions_scenebootstrap.py \
    --method_predictions_folder /path/to/store_results_folder/used/during/evaluation/ \
    # how many times to bootstrap
    --num_bootstrap value
```
