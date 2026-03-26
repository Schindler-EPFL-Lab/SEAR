import time
from pathlib import Path

import numpy as np

from sear import logger
from sear.data_processing.inference_two_trajectories import (
    InferenceSceneTwoTrajectories,
)
from sear.scripts.eval.base import ChunkProcessorBase, EvalDatasetMode
from sear.scripts.eval.base import EvalParametersBase as InferenceParametersBase
from sear.scripts.eval.store_results_no_ground_truth import (
    store_results_no_ground_truth,
)


def main(params: InferenceParametersBase, chunk_processor: ChunkProcessorBase) -> None:
    """
    Inferences the dataset on scenes from `params.scenes_root_path` and stores the
    results in `params.store_results_folder`.
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

    scenes_paths = sorted(list(params.scenes_root_path.iterdir()))
    scenes_ids = params.scenes_ids
    if scenes_ids is None:
        scenes_ids = list(range(len(scenes_paths)))

    for scene_idx in scenes_ids:
        scene_path = scenes_paths[scene_idx]
        if params.dataset_mode is EvalDatasetMode.NO_TRAJECTORY_SPECIFIED:
            scene = InferenceSceneTwoTrajectories.from_scene_path_rgb_only(
                scene_path=scene_path, crop=True
            )
        if params.dataset_mode is EvalDatasetMode.TWO_TRAJECTORIES:
            scene = InferenceSceneTwoTrajectories.from_scene_path(
                scene_path=scene_path, crop=True
            )
        scene_name = scene.scene_name

        processed = None
        time_start = time.time()
        try:
            processed = chunk_processor.process_chunk(
                chunk=scene,
                cache_folder=cache_folder / scene_name,
            )
        except Exception as e:
            logger.info(f"Could not process {scene_name}, {repr(e)}")
        time_end = time.time()

        if processed is not None:
            (
                extrinsics_world2cam,
                intrinsics,
                images_found,
                depths,
                mask_found,
            ) = processed

            store_results_no_ground_truth(
                images_paths=np.array(scene.images_paths[0])[mask_found].tolist(),
                images=images_found,
                thermal_mask=scene.mask_thermal[0][mask_found],
                extrinsics_world2cam=extrinsics_world2cam,
                intrinsics=intrinsics,
                depths=depths,
                output_folder=store_results_folder / scene_name,
                job_name=params.job_name,
                ratio_reconstructed=mask_found.float().mean().item(),
                duration=time_end - time_start,
                method_name=params.method_name,
            )

        logger.info(f"Inferenced at {scene_name}")
