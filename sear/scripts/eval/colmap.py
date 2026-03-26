import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import open3d as o3d
import torch
import tyro
from nerfstudio.process_data.images_to_nerfstudio_dataset import (
    ImagesToNerfstudioDataset,
)
from PIL import Image

from sear.data_processing.chunk import Chunk
from sear.data_processing.convertion import opengl_to_opencv
from sear.data_processing.frame_info import FrameInfo
from sear.data_processing.inference_scene import InferenceScene
from sear.data_processing.project_points import project_points
from sear.scripts.eval.base import ChunkProcessorBase, EvalParametersBase, main
from sear.scripts.eval.files_order import find_files_order

# from sear.validation.store_results.dust3r import store_results
from vggt.utils.geometry import closed_form_inverse_se3


@dataclass
class ColmapEvalParameters(EvalParametersBase):
    """A config to evaluate COLMAP on camera pose estimation for a trajectory"""

    method_name: str = "COLMAP"
    """The method name used to mark saved results"""

    feature_type: str = "superpoint"
    """Feature extraction method to use"""
    matcher_type: str = "superglue"
    """Matcher method to use"""


class ColmapChunkProcessor(ChunkProcessorBase):
    """A base class process a Chunk of images using COLMAP"""

    def __init__(self, config: ColmapEvalParameters) -> None:
        """Instantiates `ColmapChunkProcessor` using parameters from `config`."""
        super().__init__()
        self._config = config

    def load_model(self) -> None:
        """The COLMAP has no underlying model therefore there is nothing to load."""
        pass

    def process_chunk(
        self, chunk: Chunk | InferenceScene, cache_folder: Path
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """
        Runs COLMAP on images from `chunk`. The cache files created during the colmap
        run are stored in `cache_folder`.

        :return: extrinsics in world-to-camera opencv format of shape (N, 3, 4),
            intrinsics of shape (N, 3, 3), images of shape (N, H, W, 3), depths of shape
            (N, H, W), mask of images processed properly of shape (N,).
        """

        images_paths = chunk.images_paths[0]
        cache_images_folder = cache_folder / "images"
        cache_images_folder.mkdir(exist_ok=False, parents=True)
        remapping: dict[str, Path] = {}
        for i, image_path in enumerate(images_paths):
            new_frame_name = f"frame_{i:05}{image_path.suffix}"
            shutil.copy(
                image_path,
                cache_images_folder / new_frame_name,
            )
            remapping[new_frame_name] = image_path.relative_to(image_path.parent.parent)

        width, height = Image.open(images_paths[0]).size

        cache_colmap_folder = cache_folder / "colmap"
        cache_colmap_folder.mkdir(exist_ok=False)

        processor = ImagesToNerfstudioDataset(
            data=cache_images_folder,
            output_dir=cache_colmap_folder,
            skip_image_processing=True,
            feature_type=self._config.feature_type,
            matcher_type=self._config.matcher_type,
        )
        processor.main()

        with open(cache_colmap_folder / "transforms.json", "r") as f:
            transforms = json.load(f)

        extrinsics_cam2world_opengl_list: list[torch.Tensor] = []
        intrinsics_list: list[torch.Tensor] = []
        images_paths_found: list[str] = []
        images_found_list: list[npt.NDArray[np.float64]] = []
        for frame in transforms["frames"]:
            frame["fl_x"] = transforms["fl_x"]
            frame["fl_y"] = transforms["fl_y"]
            frame["cx"] = transforms["cx"]
            frame["cy"] = transforms["cy"]
            extrinsic_cam2world_opengl, intrinsic = FrameInfo.dict_to_matrices(frame)
            extrinsics_cam2world_opengl_list.append(extrinsic_cam2world_opengl)
            intrinsics_list.append(intrinsic)

            not_remapped_name = Path(frame["file_path"]).name
            remapped_frame_path = remapping[not_remapped_name]
            images_paths_found.append(remapped_frame_path.name)

            image = cv2.imread(str(cache_folder / frame["file_path"]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images_found_list.append(image / 255.0)

        # find a permutation such that indices of found images are in increasing order
        found_order = find_files_order(
            original_files=[path.name for path in images_paths],
            found_files=images_paths_found,
        )
        found_order = torch.tensor(found_order)

        images_paths_found_set = set(images_paths_found)
        images_found = np.stack(images_found_list)

        extrinsics_cam2world_opengl = torch.stack(extrinsics_cam2world_opengl_list)
        extrinsics_cam2world = opengl_to_opencv(extrinsics_cam2world_opengl)
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

        point_cloud = o3d.io.read_point_cloud(cache_colmap_folder / "sparse_pc.ply")
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
    params = tyro.cli(ColmapEvalParameters)
    chunk_processor = ColmapChunkProcessor(config=params)
    main(params, chunk_processor=chunk_processor)
