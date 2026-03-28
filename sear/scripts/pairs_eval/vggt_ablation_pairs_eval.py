from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from sear.ablation_models.aggregator_config import AggregatorConfig
from sear.ablation_models.vggt_wrapper import (
    OptimizationParameters,
    ThermalVGGTConfig,
    ThermalVGGTLightning,
)
from sear.data_processing.chunk import Chunk
from sear.data_processing.paired_item import PairedItem
from sear.scripts.pairs_eval.vggt_original_pairs_eval import (
    VGGTEvalPairsBase,
    main,
)


@dataclass(kw_only=True)
class VGGTAblationEvalPairs(VGGTEvalPairsBase):
    """A class to calculate pose errors of the VGGT Thermal model."""

    method_name: str = "AblationVGGT"
    """Method name to store the results"""
    aggregator: AggregatorConfig
    """
    Specifies parameters of the aggregator, i.e. which layers are updated with LoRA, how
    to process thermal tokens and etc.
    """
    thermal_vggt: ThermalVGGTConfig
    """VGGT Model config"""
    optimization: OptimizationParameters
    """VGGT Model optimization config"""
    scenes_root_path: Path = Path("scenes-root-path")
    """Directory containing processed VGGT scenes"""
    ckpt_path: Path = Path("checkpoint-path")
    """Checkpoint path from which the model should be initilized"""

    def __post_init__(self) -> None:
        """Defines the necessary parameters for the evaluation"""
        self.method_name = f"AblationVGGT-{self.aggregator.type.value}"

    def load_model(self) -> None:
        """Loads the AblationVGGT model into memory"""
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

    def _inference_vggt(
        self, chunk: Chunk | PairedItem
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts extrinsics and intrinsics camera poses of images from `chunk`"""
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
