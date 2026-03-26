import tyro
from mast3r.utils import path_to_dust3r as path_to_dust3r  # noqa: I001

from sear.scripts.eval.mast3r_eval import MAST3RChunkProcessor
from sear.scripts.eval.mast3r_eval_config import MAST3REvalParameters
from sear.scripts.inference.base import main

if __name__ == "__main__":
    params = tyro.cli(MAST3REvalParameters)
    chunk_processor = MAST3RChunkProcessor(params)
    main(params, chunk_processor=chunk_processor)
