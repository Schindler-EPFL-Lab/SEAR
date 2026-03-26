import json
from pathlib import Path

import tyro
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from sear import logger


def main(
    val_ratio: float = 0.2,
    scenes_per_dataset_path: Path = Path("./sear/configs/scenes_per_dataset.json"),
    output_file: Path = Path("./sear/configs/train_test_split.json"),
    seed: int = 0,
) -> None:
    """
    Performs train/test split of scenes. The `scenes_per_dataset_path` contains
    information about to what scenes belong to what dataset. The
    `scenes_per_dataset_path` with ratio of test scenes `val_ratio`. The result is
    stored in `output_file`, the random seed is `seed`.
    """
    with open(scenes_per_dataset_path) as f:
        scenes_per_dataset = json.load(f)

    eval_scenes_all: list[str] = []
    train_scenes_all: list[str] = []

    for dataset in scenes_per_dataset:
        scenes = sorted(scenes_per_dataset[dataset])
        if dataset == "ORU":
            groups = [scene[:2] for scene in scenes]
            # Need at least two different scenes, otherwise the metrics are not
            # representative due to the small size
            splitter = GroupShuffleSplit(n_splits=1, test_size=2, random_state=seed)
            train_ids, eval_ids = next(splitter.split(scenes, groups=groups))

            train_scenes = [scenes[i] for i in train_ids]
            eval_scenes = [scenes[i] for i in eval_ids]

        else:
            train_scenes, eval_scenes = train_test_split(
                scenes, random_state=seed, test_size=val_ratio
            )

        train_scenes_all.extend(train_scenes)
        eval_scenes_all.extend(eval_scenes)

    logger.info(f"Train scenes {len(train_scenes_all)}.")
    logger.info(f"Eval scenes {len(eval_scenes_all)}.")

    with open(output_file, "w") as f:
        json.dump(
            {
                "train": train_scenes_all,
                "eval": eval_scenes_all,
            },
            fp=f,
            indent=4,
        )


if __name__ == "__main__":
    tyro.cli(main)
