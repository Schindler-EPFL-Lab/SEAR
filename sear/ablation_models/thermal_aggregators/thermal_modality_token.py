from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from sear.ablation_models.thermal_aggregators.custom_patterns import (
    CustomPatterns,
)
from sear.ablation_models.thermal_aggregators.lora import ThermalAggregatorLoRA


class ThermalAggregatorThermalModalityToken(ThermalAggregatorLoRA):
    """
    An ablation study for the proposed ThermalVGGT method which applies LoRA and
    adds a new learnable <thermal modality token> to every thermal token.
    """

    KEEP_IN_STATE_DICT: set[str] = {"thermal_modality_token", "lora"}

    def __init__(
        self,
        vggt_state_dict: Mapping[str, Any],
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        lora_alpha: int = 128,
        lora_rank: int = 64,
        lora_dropout: float = 0.1,
    ) -> None:
        """
        Instantiates the wrapper around the vggt aggregator. The `vggt_state_dict` is
        the state dict of the original VGGT model, the `img_size` specifies the largest
        image dimention, `patch_size` is an image patch size, `embed_dim` is a patch
        dimention. The `lora_alpha`, `lora_rank` and `lora_dropout` specify the LoRA
        module.
        """
        super().__init__(
            vggt_state_dict=vggt_state_dict,
            pattern=CustomPatterns.ORIGINAL,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            lora_alpha=lora_alpha,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
        )

        # initialize thermal modality token
        self._thermal_token = nn.Parameter(data=torch.randn(1, 1, 1, embed_dim))
        nn.init.zeros_(self._thermal_token)

    def process_thermal_tokens(
        self,
        B: int,
        S: int,
        tokens: torch.Tensor,
        thermal_mask_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Processes `tokens` (including images patches processed by DINOv2, camera tokens,
        and register tokens) using `thermal_mask_flat` of shape (B*S,). The `B` is a
        batch size, `S` is a sequence length.

        The `tokens` have shape [batch size * sequence length (number of images in one
        batch), number of patches + camera tokens + register tokens, embedding
        dimention].

        For the processing strategy it shifts every thermal token by a learnable
        <thermal modality token> which helps the Aggregator to understand that the
        tokens represent thermal images and might be treated differently.

        :return: processed tokens of the same shape as `tokens`.
        """
        # (1, 1, 1, C) -> (1, 1, C)
        thermal_token = self._thermal_token[0]
        tokens = tokens.clone()  # this is crucial for gradient flow with indexing
        # (N, P, C) += (1, 1, C) but okay because of broadcasting
        tokens[thermal_mask_flat] += thermal_token
        return tokens

    def state_dict(
        self, *args, destination: Mapping[str, Any] | None = None, **kwargs
    ) -> Mapping[str, Any]:
        """
        Returns the state dict of the model which consist of state dicts of the thermal
        token projector and LoRA weights. The `destination` parameter contains the
        current state dict.

        :returns a dict where key represent the module name and value is its weights.
        """
        destination = ThermalAggregatorLoRA.state_dict(self, destination=destination)
        destination["thermal_modality_token"] = self._thermal_token.detach()
        return destination

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        """
        Loads the `state_dict` of the model which must consist of state dicts of the
        thermal projector and LoRA weights.
        """
        ThermalAggregatorLoRA.load_state_dict(
            self, state_dict=state_dict, *args, **kwargs
        )
        self._thermal_token.data.copy_(state_dict["thermal_modality_token"])
