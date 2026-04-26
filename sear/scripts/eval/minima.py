import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch
import tyro
from mast3r.utils import path_to_dust3r as path_to_dust3r  # noqa: I001
from PIL import Image
from vggt.utils.geometry import closed_form_inverse_se3

from sear import logger
from sear.data_processing.chunk import Chunk
from sear.data_processing.frame_info import FrameInfo
from sear.data_processing.inference_scene import InferenceScene
from sear.data_processing.project_points import project_points
from sear.scripts.eval.base import ChunkProcessorBase, main
from sear.scripts.eval.create_custom_pairs import create_custom_pairs
from sear.scripts.eval.files_order import find_files_order
from sear.scripts.eval.minima_config import MINIMARoMAEvalParameters
from sear.scripts.eval.read_colmap import colmap_to_json, find_best_reconstruction


class MINIMARoMAChunkProcessor(ChunkProcessorBase):
    """A base class process a Chunk of images using MINIMA-RoMA"""

    def __init__(self, config: MINIMARoMAEvalParameters) -> None:
        """Instantiates `MINIMARoMAChunkProcessor` using parameters from `config`."""
        super().__init__()
        self._config = config

    def load_model(self) -> None:
        """The MINIMA-RoMA has an underlying model, but it is loaded later."""
        pass

    def process_chunk(
        self, chunk: Chunk | InferenceScene, cache_folder: Path
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """
        Runs MINIMA-RoMA on images from `chunk`. The cache files created during the
        run are stored in `cache_folder`.

        :return: extrinsics in world-to-camera opencv format of shape (N, 3, 4),
            intrinsics of shape (N, 3, 3), images of shape (N, H, W, 3), depths of shape
            (N, H, W), mask of images processed properly of shape (N,).
        """

        images_paths = chunk.images_paths[0]

        cache_images_folder = cache_folder / "images"
        cache_images_folder.mkdir(exist_ok=False, parents=True)
        remapping: dict[str, Path] = {}
        remapped_files: list[str] = []
        for i, image_path in enumerate(images_paths):
            new_frame_name = f"frame_{i:05}.png"
            image_orig = cv2.imread(str(image_path))
            cv2.imwrite(str(cache_images_folder / new_frame_name), image_orig)
            remapping[new_frame_name] = image_path.relative_to(image_path.parent.parent)
            remapped_files.append(new_frame_name)

        cache_folder_colmap = cache_folder / "colmap"
        cache_folder_colmap.mkdir(exist_ok=False, parents=True)
        width, height = Image.open(images_paths[0]).size

        pairs_file = cache_folder / "custom_pairs.txt"
        create_custom_pairs(
            images_names=remapped_files,
            scenegraph_option=self._config.scenegraph_option,
            cyclic=self._config.cyclic,
            window_size=self._config.window_size,
            reference_index=self._config.reference_index,
            save_path=pairs_file,
        )

        cmd = [
            "python3",
            "./match_anything/run.py",
            "--pipeline",
            self._config.pipeline,
            "--config_file",
            str(self._config.config_file),
            "--images",
            str(cache_images_folder),
            "--camera_options",
            str(self._config.camera_options),
            "--outs",
            str(cache_folder_colmap),
            "--checkpoint_path",
            str(self._config.checkpoint_path),
            "--pair_file",
            str(pairs_file),
            "--force",
        ]
        logger.info("Running:" + " ".join(cmd))
        subprocess.run(cmd)

        # find the reconstruction with the most images
        best_reconstruction_dir = find_best_reconstruction(
            cache_folder_colmap / "reconstruction"
        )

        if best_reconstruction_dir is None:
            return None

        if not best_reconstruction_dir.is_absolute():
            best_reconstruction_dir = Path.cwd() / best_reconstruction_dir
        logger.info(f"Taking reconstruction from {best_reconstruction_dir}")

        output_dir = cache_folder / "output"
        output_dir.mkdir(exist_ok=True)

        num_registered = colmap_to_json(
            recon_dir=best_reconstruction_dir,
            output_dir=output_dir,
            image_rename_map=None,
        )
        logger.info(f"Registered {num_registered}")

        with open(output_dir / "transforms.json", "r") as f:
            transforms = json.load(f)

        extrinsics_cam2world_opencv_list: list[torch.Tensor] = []
        intrinsics_list: list[torch.Tensor] = []
        images_paths_found: list[str] = []
        images_found_list: list[npt.NDArray[np.float64]] = []
        for frame in transforms["frames"]:
            extrinsic_cam2world_opencv, intrinsic = FrameInfo.dict_to_matrices(frame)
            extrinsics_cam2world_opencv_list.append(extrinsic_cam2world_opencv)
            intrinsics_list.append(intrinsic)

            not_remapped_name = Path(frame["file_path"]).name
            remapped_frame_path = remapping[not_remapped_name]
            images_paths_found.append(remapped_frame_path.name)

            image = cv2.imread(str(cache_folder / frame["file_path"]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_found_list.append(image / 255.0)

        found_order = find_files_order(
            original_files=[path.name for path in images_paths],
            found_files=images_paths_found,
        )
        found_order = torch.tensor(found_order)

        images_paths_found_set = set(images_paths_found)
        images_found = np.stack(images_found_list)

        extrinsics_cam2world = torch.stack(extrinsics_cam2world_opencv_list)
        extrinsics_cam2world = torch.cat(
            [
                extrinsics_cam2world,
                torch.zeros(extrinsics_cam2world.shape[0], 1, 4).to(
                    extrinsics_cam2world
                ),
            ],
            dim=1,
        )
        extrinsics_cam2world[:, 3, 3] = 1.0
        extrinsics_world2cam: torch.Tensor = closed_form_inverse_se3(
            extrinsics_cam2world
        )
        intrinsics = torch.stack(intrinsics_list)

        point_cloud = o3d.io.read_point_cloud(output_dir / "sparse_pc.ply")
        points_np = np.asarray(point_cloud.points)
        # render depths
        depths_np = np.stack(
            [
                project_points(
                    points=points_np,
                    extrinsic_cam2world=extrinsics_cam2world[i].numpy(),
                    intrinsic=intrinsics[i].numpy(),
                    width=width,
                    height=height,
                )
                for i in range(extrinsics_cam2world.shape[0])
            ]
        )

        # create mask of images take processed successfully using colmap
        mask_found = torch.tensor(
            [(image_path.name in images_paths_found_set) for image_path in images_paths]
        )

        return (
            extrinsics_world2cam[found_order],
            intrinsics[found_order],
            torch.from_numpy(images_found)[found_order],
            torch.from_numpy(depths_np)[found_order],
            mask_found,
        )


if __name__ == "__main__":
    params = tyro.cli(MINIMARoMAEvalParameters)
    chunk_processor = MINIMARoMAChunkProcessor(params)
    main(params, chunk_processor=chunk_processor)
