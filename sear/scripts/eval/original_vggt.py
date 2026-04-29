import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

# You can stop doing this when
# [issue 416](https://github.com/facebookresearch/vggt/issues/416) of VGGT is solved.
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "vggt" / "training")
)

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from sear.data_processing.chunk import Chunk
from sear.data_processing.inference_scene import InferenceScene
from sear.scripts.eval.base import ChunkProcessorBase, EvalParametersBase, main


@dataclass(kw_only=True)
class VGGTOriginalEvalParameters(EvalParametersBase):
    """A config to evaluate original VGGT"""

    method_name: str = "VGGT-Original"
    """The method name used to mark saved results"""

    original_vggt_path: Path = Path("vggt-model")
    """Checkpoint path of the original vggt model"""


class VGGTOriginalChunkProcessor(ChunkProcessorBase):
    def __init__(self, config: VGGTOriginalEvalParameters) -> None:
        """Instantiates `VGGTOriginalChunkProcessor` using parameters from `config`."""
        super().__init__()
        self._config = config

    def load_model(self) -> None:
        """Loads the original VGGT model"""
        self._model = VGGT()
        state_dict = torch.load(self._config.original_vggt_path, map_location="cuda")
        self._model.load_state_dict(state_dict)
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
                model_prediction = self._model.forward(images=chunk.images)
                pose_enc_list = model_prediction["pose_enc_list"]
                pred_depth = model_prediction["depth"]

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
    params = tyro.cli(VGGTOriginalEvalParameters)
    chunk_processor = VGGTOriginalChunkProcessor(config=params)
    main(params, chunk_processor=chunk_processor)
