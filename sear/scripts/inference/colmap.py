import tyro

from sear.scripts.eval.colmap import ColmapChunkProcessor, ColmapEvalParameters
from sear.scripts.inference.base import main

if __name__ == "__main__":
    params = tyro.cli(ColmapEvalParameters)
    chunk_processor = ColmapChunkProcessor(params)
    main(params, chunk_processor=chunk_processor)
