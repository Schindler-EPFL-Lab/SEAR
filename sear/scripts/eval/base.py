"""
Base module for camera pose estimation evaluation framework.

This module provides the foundational classes and functions for evaluating
different camera pose estimation methods (COLMAP, DUST3R, MAST3R, etc.) on
the ThermoScenes dataset. It defines the abstract interface that all methods
must implement and provides the main evaluation loop.

Key components:
- ChunkProcessorBase: Abstract base class defining the interface for pose estimation
methods
- EvalParametersBase: Configuration for evaluation runs
- main(): Main evaluation function that orchestrates the processing pipeline
"""

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from lightning import seed_everything

from sear import logger
from sear.data_processing.chunk import Chunk
from sear.data_processing.inference_scene import InferenceScene
from sear.data_processing.multiple_dataset import VGGTMultipleDataset
from sear.data_processing.multiple_dataset_two_trajectories import (
    MultipleDatasetTwoTrajectories,
)
from sear.metrics.calculator import MetricsCalculator
from sear.scripts.eval.store_results import store_results


class EvalDatasetMode(Enum):
    """Specifies what dataset should be specified to run the evaluation."""

    NO_TRAJECTORY_SPECIFIED = "NoTrajectorySpecified"
    """
    The original ThermoScenes dataset with no trajectories specified
    """

    TWO_TRAJECTORIES = "TwoTrajectories"
    """
    A dataset in `VGGTMultipleDataset` format but with trajectories indices of rgb and
    thermal specified.
    """


@dataclass(kw_only=True)
class EvalParametersBase(ReverseCli):
    """A config to evaluate a method on camera pose estimation for a trajectory"""

    scenes_root_path: Path = Path("scenes-root-path")
    """Directory containing processed VGGT scenes"""
    depth_eps: float = 1e-8
    """Depth value smaller this value do not take part in training"""
    seed: int = 0
    """Random seed for the dataset"""
    output_folder: Path = Path("./outputs")
    """The folder where to store the intermediate results"""
    store_results_folder: Path = Path("./outputs")
    """Directory to save files to calculate metrics of the method"""
    method_name: str
    """The method name used to mark saved results"""
    scenes_ids: list[int] | None = None
    """
    Indices of eval scenes to run on. If not specified then all the indices are
    used.
    """
    job_name: str = ""
    """
    Used to identify what job produced the data stored in `scenes_root_path`. It will be
    saved as `job_name` in `transforms.json` of each scene. Helpful for debugging.
    """
    dataset_mode: EvalDatasetMode = EvalDatasetMode.NO_TRAJECTORY_SPECIFIED
    """What dataset format to use to run validation"""


class ChunkProcessorBase(ABC):
    """
    Abstract base class for processing chunks of images to estimate camera poses.

    This class defines the interface that concrete implementations must follow to
    integrate with the evaluation framework. Subclasses should implement the specific
    algorithm (e.g., COLMAP, DUST3R, MAST3R) for camera pose estimation.

    The evaluation framework uses this interface to:
    1. Load the method-specific model/algorithm
    2. Process chunks of images from validation scenes
    3. Return standardized outputs for metric calculation
    """

    def __init__(self) -> None:
        """
        Initializes the base processor with device configuration.

        Sets up the computing device (CUDA if available, otherwise CPU) that
        will be used by the concrete implementation for model inference.
        """
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load_model(self) -> None:
        """
        Loads the underlying model or algorithm.

        Concrete implementations should load their specific model weights,
        configure the algorithm, and prepare it for inference. For methods
        that don't require a learned model (e.g., COLMAP), this can be a no-op.

        This method is called once before processing any chunks.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("The method `load_model` is not implemented.")

    @abstractmethod
    def process_chunk(
        self, chunk: Chunk | InferenceScene, cache_folder: Path
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """
        Processes a `chunk` of images to estimate camera parameters.

        This is the main method that concrete implementations must provide.
        It takes a chunk of images (either from validation data or inference scenes)
        and applies the specific algorithm to estimate camera poses, intrinsics,
        and depth maps.


        `chunk` is a Chunk or InferenceScene object containing:
            - images: Tensor of shape (N, S, 3, H, W) or (N, 3, H, W)
            - images_paths: List of image file paths
            - depths: Ground truth depth maps of shape (N, S, H, W) or (N, H, W)
            - extrinsics_world2cam: Ground truth camera poses in OpenCV convention
            - intrinsics: Camera intrinsic matrices
            - mask_thermal: Boolean mask indicating thermal vs RGB images
        `cache_folder` is where intermediate results and cache files should be stored
            during processing. Implementations can use this to store temporary files,
            debug outputs, or intermediate representations.

        :return: a tuple containing:
            - extrinsics_world2cam: Estimated camera poses in OpenCV world-to-camera
            format, shape (N, 3, 4) where N is the number of successfully processed
            images
            - intrinsics: Estimated camera intrinsic matrices, shape (N, 3, 3)
            - images: Processed images in RGB format, shape (N, H, W, 3), normalized to
            [0, 1]
            - depths: Estimated depth maps, shape (N, H, W)
            - mask_found: Boolean mask of shape (N,) indicating which images were
              processed successfully (True) or failed (False)

            Returns None if the entire chunk processing failed.

        Note:
            The returned tensors should be aligned with the original chunk data.
            The mask_found helps the evaluation framework match estimated results
            with ground truth data for metric calculation.

            Implementations should handle errors gracefully and return None or
            appropriate mask_found values when individual images fail to process.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError("The method `process_chunk` is not implemented.")


def main(params: EvalParametersBase, chunk_processor: ChunkProcessorBase) -> None:
    """
    Runs a method on rgb+thermal data from checkpoint on validation scenes from
    `params.scenes_root_path` and saves the result in `params.output_folder` using
    method `chunk_processor` to process a chunk images .
    """

    chunk_processor.load_model()

    store_results_folder = params.store_results_folder
    store_results_folder.mkdir(exist_ok=True, parents=True)

    output_folder = params.output_folder
    if not output_folder.is_absolute():
        output_folder = Path.cwd() / output_folder

    output_folder.mkdir(exist_ok=True)
    cache_folder = output_folder / "cache"
    cache_folder.mkdir(exist_ok=True)

    calculator = MetricsCalculator(thresholds=[5.0, 15.0, 30.0])

    _ = seed_everything(params.seed)
    if params.dataset_mode is EvalDatasetMode.TWO_TRAJECTORIES:
        eval_dataset = MultipleDatasetTwoTrajectories.build_eval_dataset(
            scenes_root_path=params.scenes_root_path,
        )
    if params.dataset_mode is EvalDatasetMode.NO_TRAJECTORY_SPECIFIED:
        _, eval_dataset = (
            VGGTMultipleDataset.build_train_eval_datasets_undivided_fixed_split(
                scenes_root_path=params.scenes_root_path,
                depth_eps=params.depth_eps,
                seed=params.seed,
            )
        )
    random_seeds_for_validation = torch.randint(0, 2**15, size=(len(eval_dataset),))
    logger.info(f"Using random seeds for validation: {random_seeds_for_validation}")

    scenes_ids = params.scenes_ids
    if scenes_ids is None:
        scenes_ids = list(range(len(eval_dataset)))

    for scene_idx in scenes_ids:
        seed = int(random_seeds_for_validation[scene_idx])
        _ = seed_everything(seed)
        chunk_1 = eval_dataset[scene_idx]
        scene_name = chunk_1.scenes_names[0]
        dataset_name = chunk_1.datasets_names[0]
        save_name_1 = f"{dataset_name}:{scene_name}:1"

        process_chunk_1 = None
        time_start_1 = time.time()
        try:
            process_chunk_1 = chunk_processor.process_chunk(
                chunk=chunk_1,
                cache_folder=cache_folder / save_name_1,
            )
        except Exception as e:
            logger.info(f"Could not process 1 {dataset_name} {scene_name}, {repr(e)}")
        time_end_1 = time.time()

        if process_chunk_1 is not None:
            (
                extrinsics_world2cam_1,
                intrinsics_1,
                images_found_1,
                depths_1,
                mask_found_1,
            ) = process_chunk_1

            store_results(
                images_paths=np.array(chunk_1.images_paths[0])[mask_found_1].tolist(),
                images=images_found_1,
                thermal_mask=chunk_1.mask_thermal[0][mask_found_1],
                extrinsics_world2cam=extrinsics_world2cam_1,
                intrinsics=intrinsics_1,
                depths=depths_1,
                ground_truth_extrinsics_world2cam=chunk_1.extrinsics_world2cam[0][
                    mask_found_1
                ],
                ground_truth_intrinsics=chunk_1.intrinsics[0][mask_found_1],
                ground_truth_depth=chunk_1.depths[0][mask_found_1],
                output_folder=store_results_folder / save_name_1,
                job_name=params.job_name,
                ratio_reconstructed=mask_found_1.float().mean().item(),
                duration=time_end_1 - time_start_1,
                method_name=params.method_name,
            )

            calculator.add_data(
                cameras_real_world2cam=chunk_1.extrinsics_world2cam[0][
                    mask_found_1
                ].numpy(),
                depths_real=chunk_1.depths[0][mask_found_1].numpy(),
                intrinsics_real=chunk_1.intrinsics[0][mask_found_1].numpy(),
                cameras_pred_world2cam=extrinsics_world2cam_1.numpy(),
                depths_pred=depths_1.numpy(),
                intrinsics_pred=intrinsics_1.numpy(),
                ratio_reconstructed=mask_found_1.float().mean().item(),
                duration=time_end_1 - time_start_1,
                scene_name=scene_name,
                dataset_name=dataset_name,
            )

        gc.collect()
        torch.cuda.empty_cache()

        # The same chunk as chunk_1 but for each rgb image we take the corresponding
        # thermal image and vice versa for thermal images.
        chunk_2 = eval_dataset.get_chunk_modality_shape_specified(
            index=scene_idx,
            mask_thermal=~chunk_1.mask_thermal,
            sequence_length=chunk_1.images.shape[1],
        )
        save_name_2 = f"{dataset_name}:{scene_name}:2"

        time_start_2 = time.time()
        process_chunk_2 = None
        try:
            process_chunk_2 = chunk_processor.process_chunk(
                chunk=chunk_2,
                cache_folder=cache_folder / save_name_2,
            )
        except Exception as e:
            logger.info(f"Could not process 2 {dataset_name} {scene_name}, {repr(e)}")
        time_end_2 = time.time()

        if process_chunk_2 is not None:
            (
                extrinsics_world2cam_2,
                intrinsics_2,
                images_found_2,
                depths_2,
                mask_found_2,
            ) = process_chunk_2

            store_results(
                images_paths=np.array(chunk_2.images_paths[0])[mask_found_2].tolist(),
                images=images_found_2,
                thermal_mask=chunk_2.mask_thermal[0][mask_found_2],
                extrinsics_world2cam=extrinsics_world2cam_2,
                intrinsics=intrinsics_2,
                depths=depths_2,
                ground_truth_extrinsics_world2cam=chunk_2.extrinsics_world2cam[0][
                    mask_found_2
                ],
                ground_truth_intrinsics=chunk_2.intrinsics[0][mask_found_2],
                ground_truth_depth=chunk_2.depths[0][mask_found_2],
                output_folder=store_results_folder / save_name_2,
                job_name=params.job_name,
                duration=time_end_2 - time_start_2,
                ratio_reconstructed=mask_found_2.float().mean().item(),
                method_name=params.method_name,
            )
            calculator.add_data(
                cameras_real_world2cam=chunk_2.extrinsics_world2cam[0][
                    mask_found_2
                ].numpy(),
                depths_real=chunk_2.depths[0][mask_found_2].numpy(),
                intrinsics_real=chunk_2.intrinsics[0][mask_found_2].numpy(),
                cameras_pred_world2cam=extrinsics_world2cam_2.numpy(),
                depths_pred=depths_2.numpy(),
                intrinsics_pred=intrinsics_2.numpy(),
                ratio_reconstructed=mask_found_2.float().mean().item(),
                duration=time_end_2 - time_start_2,
                scene_name=scene_name,
                dataset_name=dataset_name,
            )

        gc.collect()
        torch.cuda.empty_cache()

    # Save the metrics just for
    calculator.per_dataset(output_folder / "per_dataset.json")
    calculator.per_scene(output_folder / "per_scene.json")
    calculator.aggregated(output_folder / "aggregated.json")
