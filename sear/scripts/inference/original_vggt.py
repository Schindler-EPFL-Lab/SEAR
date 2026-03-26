import tyro

from sear.scripts.eval.original_vggt import (
    VGGTOriginalChunkProcessor,
    VGGTOriginalEvalParameters,
)
from sear.scripts.inference.base import main

if __name__ == "__main__":
    params = tyro.cli(VGGTOriginalEvalParameters)
    chunk_processor = VGGTOriginalChunkProcessor(config=params)
    main(params, chunk_processor=chunk_processor)
