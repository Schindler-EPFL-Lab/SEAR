from dataclasses import dataclass
from pathlib import Path

import torch

from sear.models.vggt_wrapper import (
    OptimizationParameters,
    ThermalVGGTConfig,
    ThermalVGGTLightning,
)
from sear.visualization.point_cloud_original import PointCloudOriginalCreator


@dataclass(kw_only=True)
class PointCloudCreator(PointCloudOriginalCreator):
    """
    Creates point cloud from a folder with ThermoScenes scenes using the finetuned (on
    the task of predicting poses for a mix of thermal and rgb data) VGGT model. For each
    scene it randomly chooses half images thermal and half images rgb with no shared
    poses, and inferences the finetuned VGGT model.
    """

    thermal_vggt: ThermalVGGTConfig
    """A configuration file of the thermal vggt model."""
    optimization: OptimizationParameters
    """VGGT Model optimization config"""
    ckpt_path: Path = Path("ckpt-path")
    """Loads thermal vggt model from `ckpt_path`"""

    def _load_model(self) -> None:
        """
        Loads the finetuned VGGT model specified in `self.ckpt_path.` and turns
        it into eval mode.
        """

        self.model = ThermalVGGTLightning.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            config=self.thermal_vggt,
            optimization_config=self.optimization,
            strict=False,
        )
        self.model = self.model.cuda()
        self.model.eval()

    @torch.inference_mode()
    def _inference_vggt(
        self, images: torch.Tensor, thermal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the finetuned VGGT model on `images` and uses `thermal_mask`.

        :returns A tuple containing
                - camera pose encoding with shape [B, S, 9],
                - predicted depth maps with shape [B, S, H, W, 1],
                - confidence scores for depth predictions with shape [B, S, H, W],
                - original input images, preserved for visualization.
        """
        return self.model.forward(
            images=images.cuda(), thermal_mask=thermal_mask.cuda()
        )
