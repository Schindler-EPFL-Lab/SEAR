import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from vggt.models.vggt import VGGT

from sear.data_processing.chunk import Chunk
from sear.scripts.features_inspection.features_distance_base import (
    GenerateFeatureDistanceParamsBase,
    InferenceAggregatorBase,
    distance_between_features,
)


@dataclass(kw_only=True)
class GenerateFeatureDistanceParamsOriginalVGGT(GenerateFeatureDistanceParamsBase):
    """Parameters for generating feature distance for the original VGGT model."""

    ckpt_path: Path = Path("checkpoint-path")
    """Checkpoint path from which the model should be initilized"""


class InferenceAggregatorOriginalVGGT(InferenceAggregatorBase):
    """Inference class for the original VGGT model."""

    def __init__(self, ckpt_path) -> None:
        """
        Initializes the inference class for the original VGGT model using the provided
        `ckpt_path`.
        """
        super().__init__()
        self._ckpt_path = ckpt_path

    def load_model(self):
        """Loads the original VGGT model for inference."""
        self._model = VGGT()
        state_dict = torch.load(self._ckpt_path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        del self._model.point_head
        self._model.point_head = None
        del self._model.track_head
        self._model.track_head = None
        gc.collect()
        torch.cuda.empty_cache()

        self._model = self._model.to(self._device)
        self._model.eval()

    def forward_aggregator(self, chunk: Chunk) -> tuple[list[torch.Tensor], int]:
        """
        Runs the forward pass through the alternating attention module using `chunk`
        and returns the output features along with the patch_start_idx.
        """
        with torch.inference_mode():
            with torch.amp.autocast(str(self._device), dtype=torch.float16):
                return self._model.aggregator.forward(images=chunk.images)


if __name__ == "__main__":
    params = tyro.cli(GenerateFeatureDistanceParamsOriginalVGGT)
    aa_inference = InferenceAggregatorOriginalVGGT(ckpt_path=params.ckpt_path)
    distance_between_features(
        aggregator_inferencer=aa_inference,
        params=params,
    )
