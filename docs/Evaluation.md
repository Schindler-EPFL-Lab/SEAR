
# Validation

## DUST3R and MAST3R

One should install the official [mast3r repo](https://github.com/naver/mast3r) in the root folder of the project and build the CroCo following the official installation [guidelines](https://github.com/naver/mast3r?tab=readme-ov-file#installation):

```bash
git clone --recursive https://github.com/naver/mast3r.git
```

Run the script

```bash
python3 ./src/rebel-pose/sear/scripts/dust3r_eval.py \
    --dust3r_ckpt_path /path/to/dust3r/checkpoint/ \
    --scenes_root_path /path/to/root/with/train/and/eval/scenes \
    --output_dir /where/to/store/the/results/
```
