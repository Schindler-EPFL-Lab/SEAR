import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import tyro
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from sear import logger
from sear.scripts.features_inspection.features_distance_base import (
    FeaturesPartOption,
)


def dino_pca_2d(
    patch_tokens: torch.Tensor,
    images_classes: torch.Tensor,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """
    Performs PCA on the given `patch_tokens` and returns the 2D PCA features along with
    their corresponding classes.
    """
    S, P, _ = patch_tokens.shape

    # (S, P, C) -> (S*P, C)
    patch_tokens_np = patch_tokens.flatten(0, 1).numpy()
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(patch_tokens_np)

    point_class = np.zeros(pca_features.shape[0], dtype=np.int32)
    for i in range(S):
        point_class[i * P : (i + 1) * P] = images_classes[i]

    return pca_features, point_class


@dataclass(kw_only=True)
class GeneratePCA2DParams:
    """
    Configuration parameters for generating 2D PCA features from aggregated tokens.
    """

    output_dir: Path
    """Path to the directory where the PCA plots and point clouds will be saved."""
    aggregator_output_folder: Path
    """
    Path to the folder containing the output from the aggregator, including the
    aggregated tokens and mask thermal information.
    """
    layers_ids: list[int] | None = None
    """
    List of layer IDs to process. If None, all layers will be processed. Each layer
    ID corresponds
    """
    features_part: FeaturesPartOption = FeaturesPartOption.FRAME
    """
    What part of alternation attention module outputs to use for distance calculation. 
    """
    patch_start_idx: int = 5
    """The starting index of the image tokens"""


def main(params: GeneratePCA2DParams) -> None:
    """
    Generates 2D PCA features from aggregated tokens and saves the point clouds.
    """
    aggregated_tokens_list = torch.load(
        params.aggregator_output_folder / "aggregated_tokens_list.pth",
        map_location="cpu",
    )
    with open(params.aggregator_output_folder / "mask_thermal.json") as f:
        chunk_info = json.load(f)
    mask_thermal = torch.tensor(chunk_info["mask_thermal"], dtype=torch.int32)

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

        dino_pca_rgb_thermal, point_class = dino_pca_2d(
            features, images_classes=mask_thermal
        )

        np.save(output_dir / f"pca-point-cloud-{layer_id:03}.npy", dino_pca_rgb_thermal)
        np.save(
            output_dir / f"point-class-{layer_id:03}.npy",
            point_class,
        )

        logger.info(
            f"Processed layer {layer_id},  {run_idx + 1} / "
            + f"{len(aggregated_tokens_list)}"
        )

        plt.title(f"layer_id #{layer_id:02}", fontsize=15)
        rgb_feat = dino_pca_rgb_thermal[point_class == 0]
        thermal_feat = dino_pca_rgb_thermal[point_class == 1]
        plt.scatter(
            thermal_feat[:, 0], thermal_feat[:, 1], alpha=0.1, s=3, label="Thermal"
        )
        plt.scatter(rgb_feat[:, 0], rgb_feat[:, 1], alpha=0.1, s=3, label="RGB")
        plt.legend()
        plt.savefig(output_dir / f"pca_{layer_id:02}.png", dpi=100, bbox_inches="tight")
        plt.clf()
        plt.close()


if __name__ == "__main__":
    params = tyro.cli(GeneratePCA2DParams)
    main(params)
