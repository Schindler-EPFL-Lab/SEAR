import re
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten


class ThermalAggregator(nn.Module):
    """
    A wrapper around the original VGGT aggregator module, which can inject LoRA layers
    and project thermal values before calculating attentions.
    """

    KEEP_IN_STATE_DICT = {"lora", "thermal_proj"}

    def __init__(
        self,
        vggt_state_dict: Mapping[str, Any],
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        lora_alpha: int = 128,
        lora_rank: int = 64,
        lora_dropout: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Instantiates the wrapper around the vggt aggregator. The `vggt_state_dict` is
        the state dict of the original VGGT model, the `img_size` specifies the largest
        image dimention, `patch_size` is an image patch size, `embed_dim` is a patch
        dimention. The `lora_alpha`, `lora_rank` and `lora_dropout` specify the LoRA
        module.
        """
        super().__init__()
        # initialize aggregator
        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, **kwargs
        )
        self.aggregator.load_state_dict(
            state_dict=self.get_state_dict_part(
                state_dict=vggt_state_dict, starts_with="aggregator"
            )
        )

        self._embed_dim = embed_dim

        self._thermal_proj = nn.Linear(self._embed_dim, self._embed_dim)
        nn.init.eye_(self._thermal_proj.weight)
        nn.init.zeros_(self._thermal_proj.bias)

        # initialize LoRA
        self._lora_alpha = lora_alpha
        self._lora_rank = lora_rank
        self._lora_dropout = lora_dropout

        pattern = re.compile(r"(frame_blocks.+)|(global_blocks.+)")

        target_modules = self._find_injectable_layers(
            pattern=pattern, module=self.aggregator
        )

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

    @staticmethod
    def get_state_dict_part(
        state_dict: Mapping[str, Any], starts_with: str
    ) -> Mapping[str, Any]:
        """
        Extracts a subset of a `state_dict` containing only the entries whose keys start
        with `starts_with`.

        In PyTorch, the keys in a state dictionary typically follow a hierarchical
        naming pattern such as `"module_name_1.module_name_2. ... .weight_name"`. Here,
        `module_name_1` represents a top-level module, which may contain submodules like
        `module_name_2`, and so on. The final element (e.g., `weight_name`) usually
        corresponds to a specific parameter or buffer of the last submodule.

        This function filters the `state_dict` to include only keys that begin with the
        given prefix `starts_with`, then removes the first `len(starts_with) + 1`
        components from each matching key. This makes it convenient to load only a
        portion of a model's parameters into a corresponding submodule.

        :returns A dictionary mapping module names to their parameter tensors. The keys
            correspond to the original keys but with the prefix `"{starts_with}."`
            removed.
        """
        starts_with_length = len(starts_with)
        return {
            k[starts_with_length + 1 :]: v
            for k, v in state_dict.items()
            if k.startswith(starts_with)
        }

    @staticmethod
    def _find_injectable_layers(
        pattern: re.Pattern,
        module: nn.Module,
        module_name: str = "",
        root: bool = True,
    ) -> list[str]:
        """
        Finds the names of `module` children whose names correspond to the `pattern` and
        for which LoRA is implemented. The `module_name` defines the name of the current
        module. The `root` defines whether the `module` is the root.

        :returns the list of names of modules.
        """
        injectable_layer_classes = (
            nn.Linear,
            nn.Embedding,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
        )

        if pattern.match(module_name) and isinstance(module, injectable_layer_classes):
            return [module_name]

        result_array: list[str] = []
        for name, layer in module.named_children():
            child_name = name if root else f"{module_name}.{name}"
            current_result = ThermalAggregator._find_injectable_layers(
                pattern=pattern,
                module=layer,
                module_name=child_name,
                root=False,
            )
            result_array.extend(current_result)
        return result_array

    def forward(
        self, images: torch.Tensor, thermal_mask: torch.Tensor
    ) -> tuple[list[torch.Tensor], int]:
        """
        Performs updated forward of vggt aggregator to process `images` but additionally
        handle thermal images positioned at `thermal_mask`. Let B is batch size, S is
        sequence length, 3 is RGB channels, H is height, W is width, then the model
        expects shapes:
            - images: [B, S, 3, H, W]
            - thermal_mask: [B, S] of booleans

        The function is based on the original aggregator forward.

        :returns tokenized and processed images and the start position of the image
            tokens (but not the register and camera tokens) for the first image.

        :raises
            - ValueError if `image` does not have 3 channels,
            - RuntimeError if LoRA has not beed initialized.
        """

        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self.aggregator._resnet_mean) / self.aggregator._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.aggregator.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        # _: B*S
        # P: number of patches
        # C: embedding dimention
        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.aggregator.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.aggregator.register_token, B, S)

        # concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        # ---------------------------- The only difference ----------------------------
        # process thermal tokens in a different way
        thermal_mask_flat = thermal_mask.view(B * S)  # flatten them
        if torch.any(thermal_mask_flat):
            thermal_tokens = tokens[thermal_mask_flat]
            thermal_tokens = self._thermal_proj(thermal_tokens).to(tokens)
            tokens = tokens.clone()  # this is crucial for gradient flow with indexing
            tokens[thermal_mask_flat] = thermal_tokens
        # ------------------------------------------------------------------------------

        pos = None
        if self.aggregator.rope is not None:
            pos = self.aggregator.position_getter(
                B * S,
                H // self.aggregator.patch_size,
                W // self.aggregator.patch_size,
                device=images.device,
            )

        if self.aggregator.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register
            # tokens) so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = (
                torch.zeros(B * S, self.aggregator.patch_start_idx, 2)
                .to(images.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aggregator.aa_block_num):
            for attn_type in self.aggregator.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = (
                        self.aggregator._process_frame_attention(
                            tokens, B, S, P, C, frame_idx, pos=pos
                        )
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = (
                        self.aggregator._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat(
                    [frame_intermediates[i], global_intermediates[i]], dim=-1
                )
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.aggregator.patch_start_idx

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
        destination["thermal_proj"] = self._thermal_proj.state_dict()
        destination["lora"] = get_peft_model_state_dict(self.aggregator)
        return destination

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs) -> None:
        """
        Loads the `state_dict` of the model which must consist of state dicts of the
        thermal projector and LoRA weights.
        """
        self._thermal_proj.load_state_dict(state_dict["thermal_proj"], *args, **kwargs)
        set_peft_model_state_dict(self.aggregator, state_dict["lora"])
