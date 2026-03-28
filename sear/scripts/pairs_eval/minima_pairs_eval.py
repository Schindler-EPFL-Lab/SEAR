from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from romatch import roma_outdoor
from transformers import EfficientLoFTRImageProcessorFast

from sear.scripts.pairs_eval.base import main
from sear.scripts.pairs_eval.roma_pairs_eval import RomaEvalPairs


@dataclass
class MINIMARomaEvalPairs(RomaEvalPairs):
    """
    A class to evaluate relative camera pose reconstruction between two images using the
    MINIMA model https://arxiv.org/abs/2412.19412
    """

    method_name: str = "minima"
    """Method name to store the results"""

    checkpoint_path: Path = Path("checkpoint-path")
    """Checkpoint path to the MINIMA-RoMA model"""

    def load_model(self) -> None:
        """Loads the RoMA model into memory"""
        self._roma_model = roma_outdoor(device=self._device, use_custom_corr=False)
        state_dict = torch.load(self.checkpoint_path, map_location=self._device)
        self._roma_model.load_state_dict(state_dict=state_dict)

        # need it purely for visuzalization
        self._processor = EfficientLoFTRImageProcessorFast.from_pretrained(
            "zju-community/matchanything_eloftr"
        )


if __name__ == "__main__":
    params = tyro.cli(MINIMARomaEvalPairs)
    main(params=params)
