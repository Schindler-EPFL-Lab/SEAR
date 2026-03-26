from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vggt.models.vggt import CameraHead, DPTHead

from sear.models.thermal_aggregator import ThermalAggregator


class ThermalVGGT(nn.Module):
    """A wrapper around VGGT to properly handle additional thermal information."""

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
        Instantiates the wrapper around the vggt model. Notably, this implementation has
        only depth and camera heads. The `vggt_state_dict` is the state dict of the
        original VGGT model state dict, `img_size` specifies the largest image
        dimention, `patch_size` is an image patch size, `embed_dim` is a patch
        dimention. The `lora_alpha`, `lora_rank` and `lora_dropout` specify the LoRA
        module.
        """
        super().__init__()

        self.aggregator = ThermalAggregator(
            vggt_state_dict=vggt_state_dict,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            lora_alpha=lora_alpha,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
        )

        # initialize modules
        self.camera_head.load_state_dict(
            state_dict=ThermalAggregator.get_state_dict_part(
                state_dict=vggt_state_dict, starts_with="camera_head"
            )
        )

        self.depth_head.load_state_dict(
            state_dict=ThermalAggregator.get_state_dict_part(
                state_dict=vggt_state_dict, starts_with="depth_head"
            )
        )

        # freeze the camera_head and depth_head parameters
        for p in self.camera_head.parameters():
            p.requires_grad_(False)

        for p in self.depth_head.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_checkpoint_path(
        cls,
        vggt_ckpt_path: Path,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        lora_alpha: int = 128,
        lora_rank: int = 64,
        lora_dropout: float = 0.1,
    ) -> "ThermalVGGT":
        """
        Instantiates the class using `vggt_ckpt_path` instead of `vggt_state_dict`. The
        `vggt_ckpt_path` defines the location of the original VGGT saved model state
        dict, `img_size` specifies the largest image dimention, `patch_size` is an image
        patch size, `embed_dim` is a patch dimention. The `lora_alpha`, `lora_rank` and
        `lora_dropout` specify the LoRA module.
        """
        vggt_state_dict = torch.load(vggt_ckpt_path)
        return cls(
            vggt_state_dict=vggt_state_dict,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            lora_alpha=lora_alpha,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
        )

    def forward(
        self, images: torch.Tensor, thermal_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the wrapped VGGT model. It processed `images` where
        `thermal_mask` specifies the position of thermal images. Let (H, W) is the image
        size, B -batch size, S - sequence length, then `images` has shape [B, S, 3, H,
        W] and `thermal_mask` has shape [B, S].

        :returns A dictionary containing
                - camera pose encoding with shape [B, S, 9],
                - predicted depth maps with shape [B, S, H, W, 1],
                - confidence scores for depth predictions with shape [B, S, H, W],
                - original input images, preserved for visualization.

        :raises RuntimeError if `image` or `thermal_mask` have incorrect or mismatched
            shapes.
        """
        if images.ndim != 5:
            raise RuntimeError(
                f"`images` must be of size [B, S, 3, H, W] but has ndim = {images.ndim}"
            )
        if thermal_mask.ndim != 2:
            raise RuntimeError(
                "`thermal_mask` must be of size [B, S] but has ndim = "
                + f"{thermal_mask.ndim}"
            )
        if images.shape[:2] != thermal_mask.shape[:2]:
            raise RuntimeError(
                "`images` and `thermal_mask` have mismatched shapes: "
                + f"images {images.shape}, thermal {thermal_mask.shape}"
            )

        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images=images, thermal_mask=thermal_mask
        )

        predictions = {}
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[
                -1
            ]  # pose encoding of the last iteration
            predictions["pose_enc_list"] = pose_enc_list

            depth, depth_conf = self.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=patch_start_idx,
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        predictions["images"] = images

        return predictions

    def state_dict(self, *args, **kwargs) -> Mapping[str, Any]:
        """Returns the aggregator state dict"""
        return self.aggregator.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs) -> None:
        """Loads the aggregator state dict"""
        self.aggregator.load_state_dict(*args, **kwargs)
