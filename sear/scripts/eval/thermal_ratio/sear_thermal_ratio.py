import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

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

from sear.models.aggregator_config import AggregatorConfig
from sear.models.vggt_wrapper import (
    OptimizationParameters,
    ThermalVGGTConfig,
)
from sear.scripts.eval.sear_eval import VGGTAblationChunkProcessor
from sear.scripts.eval.thermal_ratio.base import EvalRatiosParametersBase, main


@dataclass(kw_only=True)
class ThermoVGGTAblationEvalRatiosParameters(EvalRatiosParametersBase):
    method_name: str = "AblationVGGT"
    """The method name used to mark saved results"""
    aggregator: AggregatorConfig
    """
    Specifies parameters of the aggregator, i.e. which layers are updated with LoRA, how
    to process thermal tokens and etc.
    """
    thermal_vggt: ThermalVGGTConfig
    """VGGT Model config"""
    optimization: OptimizationParameters
    """VGGT Model optimization config"""
    ckpt_path: Path = Path("checkpoint-path")
    """Checkpoint path from which the model should be initilized"""

    def __post_init__(self) -> None:
        """Initializes necessary values to run the validation."""
        super().__post_init__()
        self.method_name = f"AblationVGGT-{self.aggregator.type.value}"
        self.method_name = f"{self.method_name}-thermal_percent-{self.thermal_percent}"


if __name__ == "__main__":
    params = tyro.cli(ThermoVGGTAblationEvalRatiosParameters)
    chunk_processor = VGGTAblationChunkProcessor(config=params)
    main(params, chunk_processor=chunk_processor)
