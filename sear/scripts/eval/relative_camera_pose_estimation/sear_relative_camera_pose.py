"""
Ablation evaluation script for the VGGT Thermal model.

This script evaluates the performance of a modified VGGT Thermal model with configurable
aggregator and optimization parameters. It computes relative camera pose errors between
image pairs, supporting ablation studies over different aggregator types and model
configurations.

The evaluation follows the same protocol as the original VGGT evaluation but allows
customization of the thermal aggregation mechanism and optimization settings, enabling
comparisons against the baseline.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import override

import torch
import tyro
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# You can stop doing this when
# [issue 416](https://github.com/facebookresearch/vggt/issues/416) of VGGT is solved.
sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "vggt"
        / "training"
    ),
)


from sear.data_processing.chunk import Chunk
from sear.data_processing.paired_item import PairedItem
from sear.models.aggregator_config import AggregatorConfig
from sear.models.vggt_wrapper import (
    OptimizationParameters,
    ThermalVGGTConfig,
    ThermalVGGTLightning,
)
from sear.scripts.eval.relative_camera_pose_estimation.vggt_relative_camera_pose import (  # noqa: E501
    VGGTEvalPairsBase,
    main,
)


@dataclass(kw_only=True)
class VGGTAblationEvalPairs(VGGTEvalPairsBase):
    """
    Evaluates a VGGT Thermal model variant with a configurable thermal aggregator.

    This class extends the base VGGT evaluation to support ablation studies by allowing
    customization of the thermal aggregation mechanism and optimization settings. It
    loads a pre-trained ThermalVGGT model from a checkpoint and evaluates relative
    camera pose reconstruction on image pairs.
    """

    method_name: str = "AblationVGGT"
    """Method name to store the results."""
    aggregator: AggregatorConfig
    """
    Specifies parameters of the aggregator, i.e., which layers are updated with LoRA,
    how to process thermal tokens, etc.
    """
    thermal_vggt: ThermalVGGTConfig
    """VGGT Model configuration."""
    optimization: OptimizationParameters
    """VGGT Model optimization configuration."""
    scenes_root_path: Path = Path("scenes-root-path")
    """Directory containing processed VGGT scenes."""
    ckpt_path: Path = Path("checkpoint-path")
    """Checkpoint path from which the model should be initialized."""

    def __post_init__(self) -> None:
        """
        Sets the method name to include the aggregator type for result storage.

        The name is formatted as: AblationVGGT-{aggregator_type}.
        """
        self.method_name = f"AblationVGGT-{self.aggregator.type.value}"

    @override
    def load_model(self) -> None:
        """
        Loads the AblationVGGT model into memory from the specified checkpoint.

        Constructs the thermal aggregator using the provided configuration, initializes
        the ThermalVGGTLightning model from the checkpoint, and moves it to the target
        device. The model is set to evaluation mode.
        """
        thermal_aggregator = params.aggregator.build_aggregator_from_vggt_path(
            vggt_path=params.thermal_vggt.vggt_path,
        )

        self._vggt_model = ThermalVGGTLightning.load_from_checkpoint(
            checkpoint_path=self.ckpt_path,
            thermal_aggregator=thermal_aggregator,
            config=self.thermal_vggt,
            optimization_config=self.optimization,
            strict=False,
        )
        self._vggt_model = self._vggt_model.to(self._device)
        self._vggt_model.eval()

    @override
    def _inference_vggt(
        self, chunk: Chunk | PairedItem
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts extrinsics and intrinsics camera poses of images from `chunk`, a batch
        of paired images and optional thermal masks.

        :returns: A tuple of (extrinsics_world2cam, intrinsics) tensors in CPU
        numpy-compatible format.
        """
        chunk = chunk.to_device(self._device)
        with torch.inference_mode():
            with torch.amp.autocast(str(self._device), dtype=torch.bfloat16):
                pose_enc_list, _, _, _ = self._vggt_model.forward(
                    images=chunk.images, thermal_mask=chunk.mask_thermal
                )

        extrinsics_pred_world2cam, intrinsics = pose_encoding_to_extri_intri(
            pose_enc_list[-1].to(torch.float32), chunk.images.shape[-2:]
        )

        return extrinsics_pred_world2cam.detach().cpu(), intrinsics.detach().cpu()


if __name__ == "__main__":
    params = tyro.cli(VGGTAblationEvalPairs)
    main(params=params)
