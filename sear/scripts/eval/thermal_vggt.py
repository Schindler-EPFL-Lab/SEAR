import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# You can stop doing this when
# [issue 416](https://github.com/facebookresearch/vggt/issues/416) of VGGT is solved.
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "vggt" / "training")
)


from sear.data_processing.chunk import Chunk
from sear.data_processing.inference_scene import InferenceScene
from sear.models.vggt_wrapper import (
    OptimizationParameters,
    ThermalVGGTConfig,
    ThermalVGGTLightning,
)
from sear.scripts.eval.base import ChunkProcessorBase, EvalParametersBase, main


@dataclass(kw_only=True)
class VGGTThermalEvalParameters(EvalParametersBase):
    """A config to evaluate original VGGT"""

    method_name: str = "VGGT-Thermo"
    """The method name used to mark saved results"""

    thermal_vggt: ThermalVGGTConfig
    """VGGT Model config"""
    optimization: OptimizationParameters
    """VGGT Model optimization config"""
    ckpt_path: Path = Path("checkpoint-path")
    """Checkpoint path from which the model should be initilized"""


class VGGTThermalChunkProcessor(ChunkProcessorBase):
    def __init__(self, config: VGGTThermalEvalParameters) -> None:
        """Instantiates `VGGTThermalChunkProcessor` using parameters from `config`."""
        super().__init__()
        self._config = config

    def load_model(self) -> None:
        """Loads the original VGGT model"""
        self._model = ThermalVGGTLightning.load_from_checkpoint(
            checkpoint_path=self._config.ckpt_path,
            config=self._config.thermal_vggt,
            optimization_config=self._config.optimization,
            strict=False,
        )
        self._model = self._model.to(self._device)
        self._model.eval()

    def process_chunk(
        self, chunk: Chunk | InferenceScene, cache_folder: Path
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """
        Calculates camera parameters (extrinsics and intrinsics) for images of the
        `chunk` using the Thermal VGGT. It stores the indermediate results (if any) in
        the `cache_folder`.

        :return: extrinsics in world-to-camera opencv format of shape (N, 3, 4),
            intrinsics of shape (N, 3, 3), images of shape (N, H, W, 3), depths of shape
            (N, H, W), mask of images processed properly of shape (N,).
        """
        chunk = chunk.to_device(self._device)
        with torch.inference_mode():
            with torch.amp.autocast(str(self._device), dtype=torch.bfloat16):
                pose_enc_list, pred_depth, _, _ = self._model.forward(
                    images=chunk.images, thermal_mask=chunk.mask_thermal
                )

        extrinsics_pred_world2cam, intrinsics = pose_encoding_to_extri_intri(
            pose_enc_list[-1].to(torch.float32), chunk.images.shape[-2:]
        )
        chunk = chunk.to_device(torch.device("cpu"))

        return (
            extrinsics_pred_world2cam[0].detach().cpu(),
            intrinsics[0].detach().cpu(),
            chunk.images[0].detach().cpu().permute(0, 2, 3, 1),
            pred_depth[0, :, :, :, 0].detach().cpu(),
            torch.ones((extrinsics_pred_world2cam.shape[1],), dtype=torch.bool),
        )


if __name__ == "__main__":
    params = tyro.cli(VGGTThermalEvalParameters)
    chunk_processor = VGGTThermalChunkProcessor(config=params)
    main(params, chunk_processor=chunk_processor)
