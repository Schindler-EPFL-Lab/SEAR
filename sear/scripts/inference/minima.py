import tyro
from mast3r.utils import path_to_dust3r as path_to_dust3r  # noqa: I001

from sear.scripts.eval.minima import MINIMARoMAChunkProcessor
from sear.scripts.eval.minima_config import MINIMARoMAEvalParameters
from sear.scripts.inference.base import main

if __name__ == "__main__":
    params = tyro.cli(MINIMARoMAEvalParameters)
    chunk_processor = MINIMARoMAChunkProcessor(params)
    main(params, chunk_processor=chunk_processor)
