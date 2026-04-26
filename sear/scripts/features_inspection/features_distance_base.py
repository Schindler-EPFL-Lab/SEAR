import abc
import gc
import json
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from lightning import seed_everything
from torch.distributions import MultivariateNormal, kl_divergence

from sear import logger
from sear.augment.geometric import GeometricTransform
from sear.augment.rgb import RGBTransformFactory
from sear.augment.thermal import ThermalTransformFactory
from sear.data_processing.chunk import Chunk
from sear.data_processing.multiple_dataset import VGGTMultipleDataset


class DistanceOption(Enum):
    """
    Specifies the distance function to use to estimate the distance between thermal and
    RGB features.
    """

    GAUSS = "gauss"
    """Symmetrized KL-divergence"""
    COSINE = "cosine"
    """Cosine distance, described in https://arxiv.org/abs/2512.04012v1"""


class FeaturesPartOption(Enum):
    """
    Specifies the part of the attention module outputs to use for distance calculation.
    """

    FRAME = "frame"
    """Use outputs of frame attention layers for distance calculation"""
    GLOBAL = "global"
    """Use outpus of global attention layers for distance computation"""
    FRAMEGLOBAL = "frameglobal"
    """Use both frame and global attention parts of the outputs"""


@dataclass(kw_only=True)
class GenerateFeatureDistanceParamsBase(ReverseCli):
    """ """

    scene_root_path: Path = Path("input")
    """Path to the directory with images"""
    output_path: Path
    """Path to save output csv file"""
    max_num_images: int = 12
    """The maximum number of images to use"""
    train_test_split_path: Path = Path("./sear/configs/train_test_split.json")
    """
    Path to json file with train test split. The distance is be calculated on the
    eval scenes
    """

    distance_function: DistanceOption = DistanceOption.GAUSS
    """
    The distance function to use to estimate the distance between thermal and
    RGB features.
    """

    features_part: FeaturesPartOption = FeaturesPartOption.FRAMEGLOBAL
    """
    What part of alternation attention module outputs to use for distance calculation. 
    """

    @staticmethod
    def _estimate_normal_parameters(
        x: torch.Tensor, lambd: float = 1e-5
    ) -> MultivariateNormal:
        """
        Estimates Gaussian distribution parameters (mean and covariance) from the input
        features `x` of shape (N, D) and returns a MultivariateNormal distribution. `x`
        is of shape (N, D). The covariance matrix is regularized by adding `lambd` to
        the diagonal to ensure it is positive definite.

        :return: A MultivariateNormal distribution with the estimated mean and
            covariance.
        """
        if x.ndim != 2:
            raise RuntimeError(
                f"The input features must be of shape (N, D), but got {x.shape}"
            )

        mean = x.mean(dim=0)
        cov = torch.cov(x.transpose(-1, -2)) + lambd * torch.eye(x.shape[-1]).to(mean)
        return MultivariateNormal(mean, covariance_matrix=cov)

    @staticmethod
    def _symmetrized_kl_normal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the symmetrized KL-divergence between features `x` and `y`. The
        inputs are of shape (S1, P, C) and (S2, P, C) where S1/S2 is the number of
        images in sequences, P is the number of patches in each image, and C is the
        feature dimension. The function estimates Gaussian distributions for both sets
        of features and calculates the KL-divergence in both directions, returning their
        sum as the symmetrized KL-divergence.

        :return the symmetrized KL-divergence
        """

        if x.ndim != 3 or y.ndim != 3 or x.shape[1:] != y.shape[1:]:
            raise RuntimeError(
                f"The input features must be of shape (S1/S2, P, C), but got {x.shape} "
                + f"and {y.shape}"
            )

        # (S1, P, C) -> (S1*P, C)
        normal_x = GenerateFeatureDistanceParamsBase._estimate_normal_parameters(
            x.flatten(0, 1)
        )
        # (S2, P, C) -> (S2*P, C)
        normal_y = GenerateFeatureDistanceParamsBase._estimate_normal_parameters(
            y.flatten(0, 1)
        )
        distance_x2y = kl_divergence(normal_x, normal_y)
        distance_y2x = kl_divergence(normal_y, normal_x)

        return distance_x2y + distance_y2x

    @staticmethod
    def _cosine_distance(
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 2,
        exclude_corresponding_y: bool = False,
    ) -> torch.Tensor | None:
        """
        Calculates the cosine distance between features `x` and `y` following RobustVGGT
        https://arxiv.org/abs/2512.04012v1 Both inputs are of shape (S1, P, C), (S2, P,
        C), S1/S2 is the number of images in sequences, P is the number of patches in
        each image, and C is the feature dimension. The function performs computations
        in batches of size `batch_size` to handle memory constraints. It excludes the
        corresponding tokens from `y` when `exclude_corresponding_y` is True, which is
        used for calculating RGB-to-RGB distances.

        :return: the average cosine distance between features in `x` and `y`
        """

        if x.ndim != 3 or y.ndim != 3 or x.shape[1:] != y.shape[1:]:
            raise RuntimeError(
                "The input features must be of shape (S, P, C) and have the same P and"
                + f" C dimensions, but got {x.shape} and {y.shape}"
            )

        if exclude_corresponding_y and x.shape != y.shape:
            raise RuntimeError(
                "When `exclude_corresponding_y` is True, the first dimension of `x` and"
                f" `y` must be the same, but got {x.shape} and {y.shape}"
            )

        if exclude_corresponding_y:
            batch_size = 1

        cosine_distances: list[torch.Tensor] = []
        for i_x in range(0, x.shape[0], batch_size):
            for i_y in range(0, y.shape[0], batch_size):
                if exclude_corresponding_y and i_x == i_y:
                    continue

                # (S1, P, C)
                features_x_part = x[i_x : i_x + batch_size]
                features_x_part = F.normalize(features_x_part, dim=-1)
                # (S2, P, C)
                features_y_part = y[i_y : i_y + batch_size]
                features_y_part = F.normalize(features_y_part, dim=-1)
                dist = torch.einsum(
                    "ijk, mnk -> imjn", features_x_part, features_y_part
                )
                cosine_distances.append(dist.mean())

        if len(cosine_distances) == 0:
            return None

        return torch.stack(cosine_distances).mean()

    def _batched_distance_calculation(
        self,
        features: torch.Tensor,
        mask_thermal: torch.Tensor,
    ) -> list[float]:
        """
        features is of shape (B, S, P, C), mask_thermal is of shape (B, S,). Calculates
        the distance between thermal and RGB features. Returns a list of distances for
        each batch.

        :return: list of distances for each batch
        """

        if (
            features.ndim != 4
            or mask_thermal.ndim != 2
            or features.shape[:2] != mask_thermal.shape[:2]
        ):
            raise RuntimeError(
                "The input features must be of shape (B, S, P, C) and mask_thermal "
                + f"must be of shape (B, S), but got {features.shape} and "
                + f"{mask_thermal.shape}"
            )

        distances: list[float] = []
        for b in range(features.shape[0]):
            features_thermal = features[b][
                mask_thermal[b]
            ]  # (S, P, C) -> (num_thermal, P, C)
            features_rgb = features[b][~mask_thermal[b]]  # (S, P, C) -> (num_rgb, P, C)

            if self.distance_function is DistanceOption.GAUSS:
                distance = self._symmetrized_kl_normal(features_thermal, features_rgb)
            elif self.distance_function is DistanceOption.COSINE:
                distance_rgb2rgb = self._cosine_distance(
                    features_rgb,
                    features_rgb,
                    batch_size=1,
                    exclude_corresponding_y=True,
                )
                distance_rgb2thermal = self._cosine_distance(
                    features_rgb, features_thermal
                )
                if distance_rgb2rgb is None or distance_rgb2thermal is None:
                    logger.info(
                        f"Skipping batch of rgb {features_rgb.shape} and thermal "
                        + f"{features_thermal.shape} ..."
                    )
                    continue
                distance = distance_rgb2rgb - distance_rgb2thermal

            distances.append(distance.cpu().item())

        return distances

    def calculate_distances_per_layer(
        self,
        aggregated_tokens_list: list[torch.Tensor],
        patch_start_idx: int,
        mask_thermal: torch.Tensor,
    ) -> dict[int, list[float]]:
        """
        Calculates distance between RGB and thermal features for each layer of
        `aggregated_tokens_list`. The `mask_thermal` specifies which tokens correspond
        to thermal images and which to RGB. The `patch_start_idx` specifies when image
        tokens begin.

        :return: A dictionary where keys are layer ids and values are lists of distances
            for each batch in the input features.
        """

        distances_per_layer: dict[int, list[float]] = {}

        for layer_id in range(len(aggregated_tokens_list)):
            if layer_id not in distances_per_layer:
                distances_per_layer[layer_id] = []
            try:
                features = aggregated_tokens_list[layer_id]
                features = features[:, :, patch_start_idx:, :]

                if self.features_part is FeaturesPartOption.FRAME:
                    features = features[..., :1024]
                elif self.features_part is FeaturesPartOption.GLOBAL:
                    features = features[..., 1024:]
                elif self.features_part is FeaturesPartOption.FRAMEGLOBAL:
                    pass

                distances = self._batched_distance_calculation(
                    features=features, mask_thermal=mask_thermal
                )
                distances_per_layer[layer_id].extend(distances)

            except ValueError as e:
                logger.error(e)
                return {}

        return distances_per_layer


class InferenceAggregatorBase(abc.ABC):
    """
    Base class for running inference through the alternating attention module.
    """

    def __init__(self) -> None:
        """
        Initializes the inference class.
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load_model(self) -> None:
        """Loads the model for inference."""
        raise NotImplementedError("The load_model is not implemented")

    @abstractmethod
    def forward_aggregator(self, chunk: Chunk) -> tuple[list[torch.Tensor], int]:
        """
        Runs the forward pass through the alternating attention module.

        :return: list of outputs for each layer and the patch_start_idx.
        """
        raise NotImplementedError("The forward_aggregator is not implemented")


def distance_between_features(
    aggregator_inferencer: InferenceAggregatorBase,
    params: GenerateFeatureDistanceParamsBase,
) -> None:
    """
    Calculates the distance between thermal and RGB features for each layer of the
    alternating attention module. The distances are saved in a csv file specified by
    `params.output_path`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.output_path.parent.mkdir(exist_ok=True, parents=True)

    # load model
    aggregator_inferencer.load_model()

    seed_everything(0)
    scenes_names = sorted(
        [scene_dir.name for scene_dir in params.scene_root_path.iterdir()]
    )
    scenes_names = set(scenes_names)

    with open(params.train_test_split_path) as f:
        tt_split = json.load(f)
    scenes_eval = set(tt_split["eval"])
    scenes_to_run = sorted(list(scenes_eval & scenes_names))
    logger.info(f"Run on scenes:\n{scenes_to_run}\n")

    dataset = VGGTMultipleDataset(
        root_path=params.scene_root_path,
        scenes_names=scenes_to_run,
        min_sequence_length=params.max_num_images,
        elements_number=params.max_num_images,
        drop_last=False,
        shuffle=True,
        rgb_transform=RGBTransformFactory.get_empty(),
        thermal_transform=ThermalTransformFactory.get_empty(),
        geometric_transform=GeometricTransform.empty(),
    )

    distances_per_layer: list[tuple[str, int, float, float]] = []
    for i in range(len(dataset)):
        rand_value = torch.rand((1,)).item()
        a = max(0.25, 1 / params.max_num_images + 1e-5)
        b = min(0.75, 1.0 - 1 / params.max_num_images - 1e-5)
        thermal_ratio = a + (b - a) * rand_value

        dataset.thermal_ratio = thermal_ratio
        chunk = dataset[i]
        chunk = chunk.to_device(device)
        scene_name = chunk.scenes_names[0]
        logger.info(
            f"Running on {scene_name} with thermal_ratio {dataset.thermal_ratio}"
        )

        aggregated_tokens_list, patch_start_idx = (
            aggregator_inferencer.forward_aggregator(
                chunk=chunk,
            )
        )

        current_distances = params.calculate_distances_per_layer(
            aggregated_tokens_list=aggregated_tokens_list,
            patch_start_idx=patch_start_idx,
            mask_thermal=chunk.mask_thermal,
        )

        for layer_id in current_distances:
            for distance_value in current_distances[layer_id]:
                distances_per_layer.append(
                    (
                        scene_name,
                        layer_id,
                        distance_value,
                        thermal_ratio,
                    )
                )

        # free memory to avoid CUDA OOM
        del aggregated_tokens_list
        gc.collect()
        torch.cuda.empty_cache()

        data = pd.DataFrame(
            distances_per_layer,
            columns=["scene_name", "layer_id", "distance", "thermal_ratio"],
        )
        data.to_csv(params.output_path, index=False)
