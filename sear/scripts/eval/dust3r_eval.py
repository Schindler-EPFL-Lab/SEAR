from pathlib import Path

import numpy as np
import torch
import tyro
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from mast3r.utils import path_to_dust3r as path_to_dust3r  # noqa: I001
from vggt.utils.geometry import closed_form_inverse_se3

from sear.data_processing.chunk import Chunk
from sear.data_processing.inference_scene import InferenceScene
from sear.scripts.eval.base import ChunkProcessorBase, main
from sear.scripts.eval.dust3r_eval_config import DUST3REvalParameters
from sear.scripts.eval.scene_graphs.representation import (
    scene_graph_representation,
)


class DUST3RChunkProcessor(ChunkProcessorBase):
    """A base class process a Chunk of images using DUST3R"""

    def __init__(self, config: DUST3REvalParameters) -> None:
        """Instantiates `DUST3RChunkProcessor` using parameters from `config`."""
        super().__init__()
        self._config = config

    def load_model(self) -> None:
        """Loads the DUST3R model."""
        self._dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(
            self._config.dust3r_ckpt_path
        )
        self._dust3r_model = self._dust3r_model.to(self._device)
        self._dust3r_model.eval()
        self._square_ok = (
            self._dust3r_model.square_ok
            if hasattr(self._dust3r_model, "square_ok")
            else False
        )

    def process_chunk(
        self, chunk: Chunk | InferenceScene, cache_folder: Path
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """
        Runs DUST3R on images from `chunk`. The cache files created during the run are
        stored in `cache_folder`.

        :return: extrinsics in world-to-camera opencv format of shape (N, 3, 4),
            intrinsics of shape (N, 3, 3), images of shape (N, H, W, 3), depths of shape
            (N, H, W), mask of images processed properly of shape (N,).
        """

        images_paths = chunk.images_paths[0]
        images_paths_str = [str(image_path) for image_path in images_paths]

        images = load_images(
            images_paths_str,
            size=self._config.image_width,
            patch_size=self._dust3r_model.patch_size,
            square_ok=self._square_ok,
        )

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
        )

        output = inference(
            pairs, self._dust3r_model, str(self._device), batch_size=1, verbose=True
        )

        scene = global_aligner(
            output,
            device=str(self._device),
            mode=GlobalAlignerMode.PointCloudOptimizer,
            verbose=True,
            min_conf_thr=self._config.matching_confidence_threshold,
        )

        _ = scene.compute_global_alignment(
            init="mst",
            niter=self._config.align_num_iterations,
            schedule=self._config.align_schedule,
            lr=self._config.align_learning_rate,
        )

        intrinsics = scene.get_intrinsics().detach().cpu()
        poses_cam2world = scene.get_im_poses().detach().cpu()
        poses_world2cam: torch.Tensor = closed_form_inverse_se3(poses_cam2world)
        depths_maps = torch.stack(scene.get_depthmaps()).detach().cpu()
        images_dust3r = torch.from_numpy(np.stack(scene.imgs))

        return (
            poses_world2cam,
            intrinsics,
            images_dust3r,
            depths_maps,
            torch.ones((poses_world2cam.shape[0],), dtype=torch.bool),
        )


if __name__ == "__main__":
    params = tyro.cli(DUST3REvalParameters)
    chunk_processor = DUST3RChunkProcessor(params)
    main(params, chunk_processor=chunk_processor)
