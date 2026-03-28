import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from lightning import seed_everything

from sear.augment.geometric import GeometricTransform
from sear.augment.rgb import RGBTransformFactory
from sear.augment.thermal import ThermalTransformFactory
from sear.data_processing.multiple_dataset import VGGTMultipleDataset
from sear.scripts.features_inspection.features_distance_base import (
    InferenceAggregatorBase,
)
from sear.scripts.features_inspection.features_distance_original import (
    InferenceAggregatorOriginalVGGT,
)


@dataclass(kw_only=True)
class VGGTAggregatorFeaturesParametersBase(ReverseCli):
    """
    Configuration parameters for extracting features from the VGGT Aggregator
    """

    scene_dir: Path = Path("input")
    """Directory containing input images"""
    output_dir: Path
    """Directory to save output files"""
    max_num_images: int = 12
    """The maximum number of images to use"""
    thermal_ratio: float = 0.5
    """The ratio of thermal images among the input images"""
    seed: int = 0
    """Random seed for reproducibility"""


@dataclass(kw_only=True)
class OriginalVGGTAggregatorFeaturesParameters(VGGTAggregatorFeaturesParametersBase):
    """
    Configuration parameters for extracting features from the original VGGT Aggregator.
    """

    ckpt_path: Path
    """VGGT Model config"""


def inspect_features(
    params: VGGTAggregatorFeaturesParametersBase,
    aggregator_inferencer: InferenceAggregatorBase,
) -> None:
    """
    Extracts features from the VGGT Aggregator and saves them to the output directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.output_dir.mkdir(exist_ok=True, parents=True)

    aggregator_inferencer.load_model()

    seed_everything(params.seed)
    dataset = VGGTMultipleDataset(
        root_path=params.scene_dir.parent,
        scenes_names=[params.scene_dir.name],
        min_sequence_length=params.max_num_images,
        elements_number=params.max_num_images,
        drop_last=True,
        shuffle=True,
        rgb_transform=RGBTransformFactory.get_empty(),
        thermal_transform=ThermalTransformFactory.get_empty(),
        geometric_transform=GeometricTransform.empty(),
        random_seed=params.seed,
    )

    mask_thermal = torch.zeros((params.max_num_images,), dtype=torch.bool)
    if params.thermal_ratio > 0.0:
        randperm = torch.randperm(params.max_num_images)
        mask_thermal[randperm[: int(params.thermal_ratio * params.max_num_images)]] = (
            True
        )

    chunk = dataset.get_chunk_modality_shape_specified(
        index=0, mask_thermal=mask_thermal, sequence_length=params.max_num_images
    )
    chunk = chunk.to_device(device)

    # run vggt
    with torch.inference_mode():
        with torch.amp.autocast(str(device), dtype=torch.float16):
            aggregated_tokens_list, _ = aggregator_inferencer.forward_aggregator(
                chunk=chunk,
            )

    # save the chunk information
    torch.save(aggregated_tokens_list, params.output_dir / "aggregated_tokens_list.pth")
    with open(params.output_dir / "mask_thermal.json", "w") as f:
        json.dump(
            {
                "mask_thermal": chunk.mask_thermal[0].cpu().tolist(),
                "images_paths": [str(el) for el in chunk.images_paths[0]],
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    params = tyro.cli(OriginalVGGTAggregatorFeaturesParameters)
    vggt_aggregator_inferencer = InferenceAggregatorOriginalVGGT(
        ckpt_path=params.ckpt_path,
    )
    inspect_features(params, aggregator_inferencer=vggt_aggregator_inferencer)
