from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from dataclasses_reverse_cli.reverse_cli import ReverseCli

from sear.models.possible_aggregators import PossibleAggregators
from sear.models.thermal_aggregators.base import ThermalAggregatorBase
from sear.models.thermal_aggregators.custom_patterns import (
    CustomPatterns,
)
from sear.models.thermal_aggregators.lora import (
    ThermalAggregatorLoRA,
)
from sear.models.thermal_aggregators.no_lora import (
    ThermalAggregatorNoLora,
)
from sear.models.thermal_aggregators.thermal_camera_token import (
    ThermalAggregatorThermalCameraToken,
)
from sear.models.thermal_aggregators.thermal_camera_token_no_lora import (  # noqa: E501
    ThermalAggregatorThermalCameraTokenNoLoRA,
)
from sear.models.thermal_aggregators.thermal_modality_token import (
    ThermalAggregatorThermalModalityToken,
)
from sear.models.thermal_aggregators.thermal_projector import (
    ThermalAggregatorThermalProjector,
)


@dataclass(kw_only=True)
class AggregatorConfig(ReverseCli):
    """
    The class describing possible ThermalAggregator to use for ablation studies.
    """

    lora_alpha: int = 128
    """
    LoRA alpha parameter, which determines the scale (alpha / rank) of the additional
    branch. One is advised to look at the Sec 4.1 of the paper
    https://arxiv.org/pdf/2106.09685
    """
    lora_rank: int = 64
    """LoRA rank parameter, the bigger the rank the larget the LoRA adapter is"""
    lora_dropout: float = 0.1
    """LoRA dropout parameter"""

    type: PossibleAggregators = PossibleAggregators.ORIGINAL
    """The type of the aggregator to use"""

    def build_aggregator(
        self,
        vggt_state_dict: Mapping[str, Any],
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
    ) -> ThermalAggregatorBase:
        """
        Builds ThermalAggregator for ablation studies from the defined option and
        provided arguments. The `vggt_state_dict` is the state dict of the original VGGT
        model, the `img_size` specifies the largest image dimention, `patch_size` is an
        image patch size, `embed_dim` is a patch dimention. The `lora_alpha`,
        `lora_rank` and `lora_dropout` specify the LoRA module.

        :raise: RuntimeError if a setting to build the chosen option is not defined.

        :return: An instance of ThermalAggregator class for ablation studies ready to
            use.
        """

        if self.type is PossibleAggregators.NO_LORA:
            return ThermalAggregatorNoLora(
                vggt_state_dict=vggt_state_dict,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
            )

        if self.type is PossibleAggregators.CAMERA_TOKEN_NO_LORA:
            return ThermalAggregatorThermalCameraTokenNoLoRA(
                vggt_state_dict=vggt_state_dict,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
            )

        pattern_mapping_original = {
            PossibleAggregators.ORIGINAL: CustomPatterns.ORIGINAL,
            PossibleAggregators.FRAME_ONLY: CustomPatterns.FRAME_ONLY,
            PossibleAggregators.GLOBAL_ONLY: CustomPatterns.GLOBAL_ONLY,
            PossibleAggregators.FIRST_QUATER: CustomPatterns.FIRST_QUATER,
            PossibleAggregators.FIRST_TWO_QUATERS: CustomPatterns.FIRST_TWO_QUATERS,
            PossibleAggregators.FIRST_THREE_QUATERS: CustomPatterns.FIRST_THREE_QUATERS,
        }

        if self.type in pattern_mapping_original:
            custom_pattern = pattern_mapping_original[self.type]
            return ThermalAggregatorThermalProjector(
                pattern=custom_pattern,
                vggt_state_dict=vggt_state_dict,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )

        pattern_mapping_camera_token = {
            PossibleAggregators.CAMERA_TOKEN: CustomPatterns.ORIGINAL,
            PossibleAggregators.CAMERA_TOKEN_FRAME_ONLY: CustomPatterns.FRAME_ONLY,
            PossibleAggregators.CAMERA_TOKEN_GLOBAL_ONLY: CustomPatterns.GLOBAL_ONLY,
        }
        if self.type in pattern_mapping_camera_token:
            custom_pattern = pattern_mapping_camera_token[self.type]
            return ThermalAggregatorThermalCameraToken(
                pattern=custom_pattern,
                vggt_state_dict=vggt_state_dict,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )

        if self.type is PossibleAggregators.NO_THERMAL_PROJECTOR:
            return ThermalAggregatorLoRA(
                pattern=CustomPatterns.ORIGINAL,
                vggt_state_dict=vggt_state_dict,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )

        if self.type is PossibleAggregators.THERMAL_MODALITY_TOKEN:
            return ThermalAggregatorThermalModalityToken(
                vggt_state_dict=vggt_state_dict,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )

        raise RuntimeError(f"The setting to build {self.type} is not defined.")

    def build_aggregator_from_vggt_path(
        self,
        vggt_path: Path,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
    ) -> ThermalAggregatorBase:
        """
        Builds ThermalAggregator for ablation studies from the defined option and
        provided arguments. The `vggt_path` is the location the original VGGT
        checkpoint, the `img_size` specifies the largest image dimention, `patch_size`
        is an image patch size, `embed_dim` is a patch dimention. The `lora_alpha`,
        `lora_rank` and `lora_dropout` specify the LoRA module.

        :raise: RuntimeError if a setting to build the chosen option is not defined.

        :return: An instance of ThermalAggregator class for ablation studies ready to
            use.
        """

        vggt_state_dict = torch.load(vggt_path)
        return self.build_aggregator(
            vggt_state_dict=vggt_state_dict,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
