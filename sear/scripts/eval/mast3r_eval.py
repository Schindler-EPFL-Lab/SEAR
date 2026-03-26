import tempfile
from pathlib import Path

import numpy as np
import torch
import tyro
from dust3r.utils.image import load_images
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.processor import Retriever
from mast3r.utils import path_to_dust3r as path_to_dust3r  # noqa: I001
from vggt.utils.geometry import closed_form_inverse_se3

from sear.data_processing.chunk import Chunk
from sear.data_processing.inference_scene import InferenceScene
from sear.scripts.eval.base import ChunkProcessorBase, main
from sear.scripts.eval.mast3r_eval_config import MAST3REvalParameters
from sear.scripts.eval.scene_graphs.mast3r import Mast3rSceneGraph
from sear.scripts.eval.scene_graphs.representation import (
    scene_graph_representation,
)


class MAST3RChunkProcessor(ChunkProcessorBase):
    """A base class process a Chunk of images using MAST3R"""

    def __init__(self, config: MAST3REvalParameters) -> None:
        """Instantiates `MAST3RChunkProcessor` using parameters from `config`."""
        super().__init__()
        self._config = config

    def load_model(self) -> None:
        """Loads the MAST3R model"""
        if self._config.scenegraph_option is Mast3rSceneGraph.RETRIEVAL:
            if self._config.retrieval_model is None:
                raise RuntimeError(
                    "The `retrieval_model` must be specified if `scenegraph_option` is "
                    + "`Mast3rSceneGraph.RETRIEVAL`"
                )

        self._mast3r_model = AsymmetricMASt3R.from_pretrained(
            self._config.mast3r_ckpt_path
        )
        self._mast3r_model = self._mast3r_model.to(self._device)
        self._mast3r_model.eval()
        self._square_ok = (
            self._mast3r_model.square_ok
            if hasattr(self._mast3r_model, "square_ok")
            else False
        )

    def process_chunk(
        self, chunk: Chunk | InferenceScene, cache_folder: Path
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """
        Runs MAST3R on images from `chunk`. The cache files created during the run are
        stored in `cache_folder`.

        :return: extrinsics in world-to-camera opencv format of shape (N, 3, 4),
            intrinsics of shape (N, 3, 3), images of shape (N, H, W, 3), depths of shape
            (N, H, W), mask of images processed properly of shape (N,).
        """

        images_paths = chunk.images_paths[0]
        images_paths_str = [str(image_path) for image_path in images_paths]

        images = load_images(
            images_paths_str,
            size=self._config.image_size,
            verbose=True,
            patch_size=self._mast3r_model.patch_size,
            square_ok=self._square_ok,
        )

        sim_matrix = None
        if self._config.retrieval_model is not None:
            retriever = Retriever(
                self._config.retrieval_model,
                backbone=self._mast3r_model,
                device=str(self._device),
            )
            with torch.no_grad():
                sim_matrix = retriever(images_paths_str)

            del retriever
            torch.cuda.empty_cache()

        pairs = make_pairs(
            images,
            scene_graph=scene_graph_representation(
                option=self._config.scenegraph_option,
                cyclic=self._config.cyclic,
                window_size=self._config.window_size,
                reference_index=self._config.reference_index,
            ),
            prefilter=None,
            symmetrize=True,
            sim_mat=sim_matrix,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            scene = sparse_global_alignment(
                imgs=images_paths_str,
                pairs_in=pairs,
                cache_path=temporary_directory,
                model=self._mast3r_model,
                lr1=self._config.learning_rate_coarse,
                niter1=self._config.num_iterations_coarse,
                lr2=self._config.learning_rate_fine,
                niter2=self._config.num_iterations_fine,
                device=str(self._device),
                opt_depth=self._config.optimize_depth,
                shared_intrinsics=self._config.shared_intrinsics,
                matching_conf_thr=self._config.matching_confidence_threshold,
            )

            intrinsics = scene.intrinsics.detach().cpu()
            poses_cam2world = scene.get_im_poses().detach().cpu()
            poses_world2cam: torch.Tensor = closed_form_inverse_se3(poses_cam2world)
            images_mast3r = torch.from_numpy(np.stack(scene.imgs))  # [N, H, W, 3]
            depth_maps_list: list[torch.Tensor] = scene.get_dense_pts3d()[1]
            depths_maps = torch.stack(depth_maps_list).detach().cpu()
            depths_maps = depths_maps.reshape(
                -1, images_mast3r.shape[1], images_mast3r.shape[2]
            )

        return (
            poses_world2cam,
            intrinsics,
            images_mast3r,
            depths_maps,
            torch.ones((poses_world2cam.shape[0]), dtype=torch.bool),
        )


if __name__ == "__main__":
    params = tyro.cli(MAST3REvalParameters)
    chunk_processor = MAST3RChunkProcessor(params)
    main(params, chunk_processor=chunk_processor)
