import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import tyro
from matplotlib import pyplot as plt

from sear import logger
from sear.scripts.features_inspection.features_distance_base import (
    FeaturesPartOption,
)
from sear.scripts.features_inspection.generate_features_pca_2d import (
    GeneratePCA2DParams,
    dino_pca_2d,
)


@dataclass(kw_only=True)
class GenerateCombinedPCA2DParams(GeneratePCA2DParams):
    """
    Configuration parameters for generating 2D PCA features from aggregated tokens
    across multiple runs.
    """

    aggregator_output_folders: list[Path]
    """
    List of paths to the folders containing the output from the aggregator for each run,
    including the aggregated tokens and mask thermal information.
    """

    aggregator_output_folder: ClassVar[None] = None
    """
    This parameter is not used in this class, but it is required by the base class.
    """


def main(params: GenerateCombinedPCA2DParams) -> None:
    """
    Generates 2D PCA features from aggregated tokens and save the point clouds.
    """
    aggregated_tokens_list = None
    images_classes_list: list[int] = []
    classes_to_names: dict[int, str] = {}
    for index, aggregator_output_folder in enumerate(params.aggregator_output_folders):
        curr_aggregated_tokens_list = torch.load(
            aggregator_output_folder / "aggregated_tokens_list.pth", map_location="cpu"
        )
        if aggregated_tokens_list is None:
            aggregated_tokens_list = curr_aggregated_tokens_list
        else:
            for i in range(len(aggregated_tokens_list)):
                # (B, **S**, P, C)
                aggregated_tokens_list[i] = torch.cat(
                    [aggregated_tokens_list[i], curr_aggregated_tokens_list[i]], dim=1
                )

        with open(aggregator_output_folder / "mask_thermal.json") as f:
            chunk_info = json.load(f)
        mask_thermal = torch.tensor(chunk_info["mask_thermal"], dtype=torch.int32)
        mask_thermal += index * 2
        images_classes_list.extend(mask_thermal.tolist())
        classes_to_names[2 * index] = f"RGB-{index}"
        classes_to_names[2 * index + 1] = f"Thermal-{index}"
    images_classes = torch.tensor(images_classes_list, dtype=torch.int32)

    output_dir = params.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    layers_ids = params.layers_ids
    if layers_ids is None:
        layers_ids = list(range(len(aggregated_tokens_list)))

    for run_idx, layer_id in enumerate(layers_ids):
        features = aggregated_tokens_list[layer_id]
        features = features[0, :, params.patch_start_idx :, :]

        if params.features_part is FeaturesPartOption.FRAME:
            features = features[..., :1024]
        elif params.features_part is FeaturesPartOption.GLOBAL:
            features = features[..., 1024:]
        elif params.features_part is FeaturesPartOption.FRAMEGLOBAL:
            pass

        dino_pca_rgb_thermal, class_ = dino_pca_2d(
            features,
            images_classes=images_classes,
        )

        np.save(output_dir / f"pca-point-cloud-{layer_id:03}.npy", dino_pca_rgb_thermal)
        np.save(
            output_dir / f"point-class-{layer_id:03}.npy",
            class_,
        )
        with open(output_dir / "classes-to-names.json", "w") as f:
            json.dump(classes_to_names, f, indent=4)

        logger.info(
            f"Processed layer {layer_id},  {run_idx + 1} / "
            + f"{len(aggregated_tokens_list)}"
        )

        for class_id in classes_to_names:
            class_feat = dino_pca_rgb_thermal[class_ == class_id]
            plt.scatter(
                class_feat[:, 0],
                class_feat[:, 1],
                alpha=0.1,
                s=3,
                label=classes_to_names[class_id],
            )

        plt.legend()
        plt.savefig(output_dir / f"pca_{layer_id:02}.png", dpi=100, bbox_inches="tight")
        plt.clf()
        plt.close()


if __name__ == "__main__":
    params = tyro.cli(GenerateCombinedPCA2DParams)
    main(params)
