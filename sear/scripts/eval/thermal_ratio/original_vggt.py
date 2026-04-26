from dataclasses import dataclass
from pathlib import Path

import tyro

from sear.scripts.eval.original_vggt import VGGTOriginalChunkProcessor
from sear.scripts.eval.thermal_ratio.base import (
    EvalRatiosParametersBase,
    main,
)


@dataclass(kw_only=True)
class VGGTOriginalEvalRatiosParameters(EvalRatiosParametersBase):
    method_name: str = "VGGT-Original"
    """The method name used to mark saved results"""
    original_vggt_path: Path = Path("checkpoint-path")
    """Checkpoint path from which the model should be initilized"""

    def __post_init__(self) -> None:
        """Initializes necessary values to run the validation."""
        super().__post_init__()
        self.method_name = f"{self.method_name}-thermal_percent-{self.thermal_percent}"


if __name__ == "__main__":
    params = tyro.cli(VGGTOriginalEvalRatiosParameters)
    chunk_processor = VGGTOriginalChunkProcessor(config=params)
    main(params, chunk_processor=chunk_processor)
