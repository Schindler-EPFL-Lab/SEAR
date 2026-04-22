import tyro

from sear.scripts.eval.sear import (
    VGGTAblationChunkProcessor,
    VGGTAblationEvalParameters,
)
from sear.scripts.inference.base import main

if __name__ == "__main__":
    params = tyro.cli(VGGTAblationEvalParameters)
    chunk_processor = VGGTAblationChunkProcessor(config=params)
    main(params, chunk_processor=chunk_processor)
