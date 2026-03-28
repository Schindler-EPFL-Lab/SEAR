import abc
import re
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten


class ThermalAggregatorBase(nn.Module, abc.ABC):
    """
    A base class for ablation studies on ThermalAggregator necessary parts - thermal
    projector and LoRA.
    """

    KEEP_IN_STATE_DICT: set[str] = set()

    def __init__(
        self,
        vggt_state_dict: Mapping[str, Any],
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
    ) -> None:
        """
        Instantiates the wrapper around the vggt aggregator. The `vggt_state_dict` is
        the state dict of the original VGGT model, the `img_size` specifies the largest
        image dimention, `patch_size` is an image patch size, `embed_dim` is a patch
        dimention.
        """
        super().__init__()
        # initialize aggregator
        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        _ = self.aggregator.load_state_dict(
            state_dict=self.get_state_dict_part(
                state_dict=vggt_state_dict, starts_with="aggregator"
            )
        )

        self._embed_dim = embed_dim

    @abstractmethod
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

        :return: processed tokens of the same shape as `tokens`. This function updates
            only tokens corresponding to thermal images.
        """
        raise NotImplementedError(
            "The `process_thermal_tokens` is not implemented for ThermalAggregatorBase"
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
        pattern: re.Pattern[str],
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
            current_result = ThermalAggregatorBase._find_injectable_layers(
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
        _, num_of_patches, embedding_dimension = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.aggregator.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.aggregator.register_token, B, S)

        # concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        # ------- The only difference with the original VGGT Aggregator forward  -------
        # process thermal tokens in a different way
        thermal_mask_flat = thermal_mask.view(B * S)  # flatten them
        if torch.any(thermal_mask_flat):
            tokens = self.process_thermal_tokens(
                B=B,
                S=S,
                tokens=tokens,
                thermal_mask_flat=thermal_mask_flat,
            )
        # ------------------------------------------------------------------------------

        position = None
        if self.aggregator.position_getter is not None:
            position = self.aggregator.position_getter(
                B * S,
                H // self.aggregator.patch_size,
                W // self.aggregator.patch_size,
                device=images.device,
            )

        if self.aggregator.patch_start_idx > 0 and position is not None:
            # do not use position embedding for special tokens (camera and register
            # tokens) so set pos to 0 for the special tokens
            position = position + 1
            pos_special = (
                torch.zeros(B * S, self.aggregator.patch_start_idx, 2)
                .to(images.device)
                .to(position.dtype)
            )
            position = torch.cat([pos_special, position], dim=1)

        # update num_of_patches because we added special tokens
        _, num_of_patches, embedding_dimension = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list: list[torch.Tensor] = []

        concat_inter: torch.Tensor | None = None
        frame_intermediates: list[torch.Tensor] | None = None
        global_intermediates: list[torch.Tensor] | None = None

        for _ in range(self.aggregator.aa_block_num):
            for attn_type in self.aggregator.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = (
                        self.aggregator._process_frame_attention(
                            tokens,
                            B,
                            S,
                            num_of_patches,
                            embedding_dimension,
                            frame_idx,
                            pos=position,
                        )
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = (
                        self.aggregator._process_global_attention(
                            tokens,
                            B,
                            S,
                            num_of_patches,
                            embedding_dimension,
                            global_idx,
                            pos=position,
                        )
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            assert frame_intermediates is not None
            assert global_intermediates is not None
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
