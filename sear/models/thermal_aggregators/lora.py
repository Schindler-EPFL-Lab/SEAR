import re
from collections.abc import Mapping
from typing import Any, OrderedDict

import torch
from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)

from sear.models.thermal_aggregators.base import ThermalAggregatorBase
from sear.models.thermal_aggregators.custom_patterns import (
    CustomPatterns,
)


class ThermalAggregatorLoRA(ThermalAggregatorBase):
    """
    An ablation study for the proposed ThermalVGGT method which applies LoRA for a
    part of layers defined using regex.
    """

    KEEP_IN_STATE_DICT: set[str] = {"lora"}

    def __init__(
        self,
        pattern: CustomPatterns,
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
        module. The `pattern` defines the what modules to use for LoRA incorporation.
        """
        super().__init__(
            vggt_state_dict=vggt_state_dict,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # initialize LoRA
        re_pattern = re.compile(pattern.value)
        target_modules = self._find_injectable_layers(
            pattern=re_pattern, module=self.aggregator
        )

        self._lora_alpha = lora_alpha
        self._lora_rank = lora_rank
        self._lora_dropout = lora_dropout

        lora_config = LoraConfig(
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            r=self._lora_rank,
            bias="none",
            target_modules=target_modules,
        )
        self.aggregator = inject_adapter_in_model(
            peft_config=lora_config, model=self.aggregator
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

        For the processing strategy it does nothing with the input `tokens`.

        :return: processed tokens of the same shape as `tokens`.
        """
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
        if destination is None:
            destination: Mapping[str, Any] = OrderedDict()
        destination["lora"] = get_peft_model_state_dict(self.aggregator)
        return destination

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        """
        Loads the `state_dict` of the model which must consist of state dicts of the
        thermal projector and LoRA weights.
        """
        set_peft_model_state_dict(self.aggregator, state_dict["lora"])
