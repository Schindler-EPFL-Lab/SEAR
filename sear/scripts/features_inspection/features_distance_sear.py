from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from sear.data_processing.chunk import Chunk
from sear.models.aggregator_config import AggregatorConfig
from sear.models.vggt_wrapper import OptimizationParameters, ThermalVGGTConfig
from sear.scripts.features_inspection.features_distance_base import (
    GenerateFeatureDistanceParamsBase,
    InferenceAggregatorBase,
    distance_between_features,
)


@dataclass(kw_only=True)
class GenerateFeatureDistanceParamsAblation(GenerateFeatureDistanceParamsBase):
    """Parameters for generating feature distance for the ablation models."""

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


class InferenceAggregatorVGGTAblation(InferenceAggregatorBase):
    """
    Class for inference through the alternating attention module of VGGT for the
    ablation models.
    """

    def __init__(
        self,
        thermal_vggt_config: ThermalVGGTConfig,
        optimization_config: OptimizationParameters,
        aggregator_config: AggregatorConfig,
        ckpt_path: Path,
    ) -> None:
        """
        Initializes the inference class for the ablation models using the provided
        `config`.
        """
        super().__init__()
        self._thermal_vggt_config = thermal_vggt_config
        self._optimization_config = optimization_config
        self._aggregator_config = aggregator_config
        self._ckpt_path = ckpt_path

    def load_model(self):
        """Loads the Ablation VGGT model for inference."""
        thermal_aggregator = self._aggregator_config.build_aggregator_from_vggt_path(
            vggt_path=self._thermal_vggt_config.vggt_path,
        )

        self._model = ThermalVGGTLightning.load_from_checkpoint(
            checkpoint_path=self._ckpt_path,
            thermal_aggregator=thermal_aggregator,
            config=self._thermal_vggt_config,
            optimization_config=self._optimization_config,
            strict=False,
        )

        self._model = self._model.to(self._device)
        self._model.eval()

    def forward_aggregator(self, chunk: Chunk) -> tuple[list[torch.Tensor], int]:
        """
        Runs the forward pass through the alternating attention module using `chunk` and
        returns the output features along with the patch_start_idx.
        """
        with torch.inference_mode():
            with torch.amp.autocast(str(self._device), dtype=torch.float16):
                return self._model._vggt.aggregator.forward(
                    images=chunk.images,
                    thermal_mask=chunk.mask_thermal,
                )


if __name__ == "__main__":
    params = tyro.cli(GenerateFeatureDistanceParamsAblation)
    aa_inference = InferenceAggregatorVGGTAblation(
        thermal_vggt_config=params.thermal_vggt,
        optimization_config=params.optimization,
        aggregator_config=params.aggregator,
        ckpt_path=params.ckpt_path,
    )
    distance_between_features(
        aggregator_inferencer=aa_inference,
        params=params,
    )
