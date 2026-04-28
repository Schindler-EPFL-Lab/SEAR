# Validation

## DUST3R and MAST3R

One should install the official [mast3r repo](https://github.com/naver/mast3r) in the root folder of the project and build the CroCo following the official installation [guidelines](https://github.com/naver/mast3r?tab=readme-ov-file#installation):

```bash
git clone --recursive https://github.com/naver/mast3r.git
```

Run the script

```bash
python3 ./sear/scripts/eval/dust3r_eval.py \
    --dust3r_ckpt_path /path/to/dust3r/checkpoint/ \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --store_results_folder /path/to/store/poses/and/depths \
    --output_dir /where/to/store/cache/and/metrics/
```

## COLMAP

One should install and build [nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [Ceres](https://ceres-solver.googlesource.com/ceres-solver.git), [COLMAP](https://github.com/colmap/colmap.git) and [hloc](https://github.com/cvg/Hierarchical-Localization/).
We use a script below for this purpose:

```bash
# Install Ceres
git clone --branch 2.1.0 https://ceres-solver.googlesource.com/ceres-solver.git --single-branch && \
    cd ceres-solver && \
    git checkout $(git describe --tags) && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf ceres-solver

# Install colmap
git clone --branch 3.11.0 https://github.com/colmap/colmap.git --single-branch && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -DCUDA_ENABLED=ON \
             -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    make -j `nproc` && \
    make install && \
    cd ../.. && \
    rm -rf colmap

# Install hloc
git clone --recursive https://github.com/cvg/Hierarchical-Localization/ && \
    cd Hierarchical-Localization/ && \
    git checkout 2e2a5517d88c7c84db0efd77d720c858da19edcd && \
    uv pip install -e . && \
    cd ..
```

After installation one should run:

```bash
python3 ./sear/scripts/eval/colmap.py \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --store_results_folder /path/to/store/poses/and/depths \
    --output_dir /where/to/store/cache/and/metrics/
```

## MatchAnything and MINIMA

1. Install the same packages as for [COLMAP](./Main.md#colmap)
2. Install [deep-image-matching](https://github.com/3DOM-FBK/deep-image-matching.git).
3. Install weights of [`minima-roma`](https://github.com/LSXI7/storage/releases/download/MINIMA/minima_roma.pth).
4. For MatchAnything run:

    ```bash
    python3 ./sear/scripts/eval/match_anything.py \
        --scenes_root_path /path/to/root/with/train/and/eval/scenes \
        --store_results_folder /path/to/store/poses/and/depths \
        --output_dir /where/to/store/cache/and/metrics/
    ```

    Note: ELoFTR checkpoint is be automatically downloaded from HuggingFace [`matchanything_eloftr`](https://huggingface.co/zju-community/matchanything_eloftr).

5. For MINIMA run:

    ```bash
    python3 ./sear/scripts/eval/minima.py \
        --scenes_root_path /path/to/root/with/train/and/eval/scenes \
        --store_results_folder /path/to/store/poses/and/depths \
        --checkpoint_path /path/to/minima-roma/weights.pth \
        --output_dir /where/to/store/cache/and/metrics/
    ```

## VGGT

Install VGGT checkpoint [`VGGT-1B`](https://huggingface.co/facebook/VGGT-1B) and run:

```bash
python3 ./sear/scripts/eval/original_vggt.py \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --store_results_folder /path/to/store/poses/and/depths \
    --original_vggt_path /path/to/vggt/weights.pth \
    --output_dir /where/to/store/cache/and/metrics/
```

## SEAR

Install SEAR checkpoints [`SEAR`](https://huggingface.co/MalcolmMielle/SEAR) and run:

```bash
python3 ./sear/scripts/eval/sear.py \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --store_results_folder /path/to/store/poses/and/depths \
    --thermal_vggt.vggt-path /path/to/original/vggt/weights.pth \
    --ckpt_path /path/to/sear/weights.pth \
    --aggregator.type AGGREGATOR_TYPE
    --output_dir /where/to/store/cache/and/metrics/
```

Where AGGERGATOR_TYPE is selected from [those](../../sear/models/possible_aggregators.py).

## Notes

- It usually takes long time to evaluate on each scene using COLMAP/MatchAnything/MINIMA and for each evaluation script we provide a parameter `--scenes_ids` such that it is possible to evaluate a batch of scenes using their insides:

    ```bash
    python3 ./sear/scripts/eval/<method-name>.py -- \
        ...
        --scenes_ids 0 1 2 3
    ```

    Such that it is possible to run evaluation on several nodes for different scenes and speed up the evaluation.
    After that one should run:

    ```bash
    python3 eval_using_predictions.py \
    --method_predictions_folder /path/to/store_results_folder/used/during/evaluation/ \
    --output_folder_root /where/to/store/the/results
    ```

- For validation on SEAR Datasets run script:

    ```bash
    python3 ./sear/scripts/eval/<method-name>.py -- \
    ...
    --scenes_root_path /path/to/sear/dataset/
    --dataset_mode TWO_TRAJECTORIES
    ```
