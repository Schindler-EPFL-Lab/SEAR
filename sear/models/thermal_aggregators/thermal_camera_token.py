from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from vggt.models.aggregator import slice_expand_and_flatten

from sear.models.thermal_aggregators.custom_patterns import (
    CustomPatterns,
)
from sear.models.thermal_aggregators.lora import ThermalAggregatorLoRA


class ThermalAggregatorThermalCameraToken(ThermalAggregatorLoRA):
    """
    An ablation study for the proposed ThermalVGGT method which applies LoRA and
    replaces camera tokens of thermal sequences by a new learnable <thermal camera
    token>
    """

    KEEP_IN_STATE_DICT: set[str] = {"thermal_camera_token", "lora"}

    def __init__(
        self,
        vggt_state_dict: Mapping[str, Any],
        pattern: CustomPatterns = CustomPatterns.ALL_BLOCKS,
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

        # initialize thermal camera token
        self._thermal_camera_token = nn.Parameter(
            self.aggregator.camera_token.detach().clone(),
            requires_grad=True,
        )

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

        For the processing strategy it replaces camera tokens of thermal images by a
        learnable <thermal camera token>.

        :return: processed tokens of the same shape as `tokens`.
        """
        # (1, 2, 1, C) -> (B*S, 1, C)
        thermal_camera_token = slice_expand_and_flatten(
            token_tensor=self._thermal_camera_token, B=B, S=S
        )

        tokens = tokens.clone()  # this is crucial for gradient flow with indexing
        # replace camera tokens, the first token is always the camera token.
        tokens[thermal_mask_flat, 0, :] = thermal_camera_token[thermal_mask_flat, 0, :]
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
        destination["thermal_camera_token"] = self._thermal_camera_token.detach()
        return destination

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        """
        Loads the `state_dict` of the model which must consist of state dicts of the
        thermal projector and LoRA weights.
        """
        ThermalAggregatorLoRA.load_state_dict(
            self, state_dict=state_dict, *args, **kwargs
        )
        self._thermal_camera_token.data.copy_(state_dict["thermal_camera_token"])
