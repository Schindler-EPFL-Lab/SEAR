import tyro

from sear.scripts.eval.thermal_vggt import (
    VGGTThermalChunkProcessor,
    VGGTThermalEvalParameters,
)
from sear.scripts.inference.base import main

if __name__ == "__main__":
    params = tyro.cli(VGGTThermalEvalParameters)
    chunk_processor = VGGTThermalChunkProcessor(config=params)
    main(params, chunk_processor=chunk_processor)
