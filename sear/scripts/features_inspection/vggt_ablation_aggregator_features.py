""" """

from dataclasses import dataclass
from pathlib import Path

import tyro

from sear.ablation_models.aggregator_config import AggregatorConfig
from sear.ablation_models.vggt_wrapper import (
    OptimizationParameters,
    ThermalVGGTConfig,
)
from sear.scripts.features_inspection.features_distance_ablation import (
    InferenceAggregatorVGGTAblation,
)
from sear.scripts.features_inspection.vggt_original_aggregator_features import (
    VGGTAggregatorFeaturesParametersBase,
    inspect_features,
)


@dataclass(kw_only=True)
class VGGTAblationAggregatorFeaturesParameters(VGGTAggregatorFeaturesParametersBase):
    """
    Configuration parameters for extracting features from the Ablation VGGT Aggregator
    """

    thermal_vggt: ThermalVGGTConfig
    """VGGT Model config"""
    optimization: OptimizationParameters
    """VGGT Model optimization config"""
    aggregator: AggregatorConfig
    """
    Specifies parameters of the aggregator, i.e. which layers are updated with LoRA, how
    to process thermal tokens and etc.
    """
    ckpt_path: Path = Path("checkpoint-path")
    """Checkpoint path from which the model should be initilized"""


if __name__ == "__main__":
    params = tyro.cli(VGGTAblationAggregatorFeaturesParameters)
    aggregator_inferencer = InferenceAggregatorVGGTAblation(
        thermal_vggt_config=params.thermal_vggt,
        optimization_config=params.optimization,
        aggregator_config=params.aggregator,
        ckpt_path=params.ckpt_path,
    )
    inspect_features(
        params=params,
        aggregator_inferencer=aggregator_inferencer,
    )
