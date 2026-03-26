from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from sear.ablation_models.thermal_aggregators.custom_patterns import (
    CustomPatterns,
)
from sear.ablation_models.thermal_aggregators.lora import ThermalAggregatorLoRA


class ThermalAggregatorThermalProjector(ThermalAggregatorLoRA):
    """
    An model which applies LoRA for a part of AA-attention layers defined using regex,
    and uses a thermal projector to update thermal tokens.
    """

    KEEP_IN_STATE_DICT: set[str] = {"thermal_proj", "lora"}

    def __init__(
        self,
        vggt_state_dict: Mapping[str, Any],
        pattern: CustomPatterns = CustomPatterns.ORIGINAL,
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
        module. The `pattern` defines the what modules to use for LoRA incorporation.
        """
        super().__init__(
            vggt_state_dict=vggt_state_dict,
            pattern=pattern,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            lora_alpha=lora_alpha,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
        )

        self._thermal_proj = nn.Linear(self._embed_dim, self._embed_dim)
        nn.init.eye_(self._thermal_proj.weight)
        nn.init.zeros_(self._thermal_proj.bias)

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

        For the processing strategy it uses <thermal projector>, which updates each
        token by using a learnable linear layer.

        :return: processed tokens of the same shape as `tokens`.
        """
        thermal_tokens = tokens[thermal_mask_flat]
        thermal_tokens = self._thermal_proj(thermal_tokens).to(tokens)
        tokens = tokens.clone()  # this is crucial for gradient flow with indexing
        tokens[thermal_mask_flat] = thermal_tokens
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
        destination["thermal_proj"] = self._thermal_proj.state_dict()
        return destination

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        """
        Loads the `state_dict` of the model which must consist of state dicts of the
        thermal projector and LoRA weights.
        """
        ThermalAggregatorLoRA.load_state_dict(
            self, state_dict=state_dict, *args, **kwargs
        )
        self._thermal_proj.load_state_dict(state_dict["thermal_proj"], *args, **kwargs)
