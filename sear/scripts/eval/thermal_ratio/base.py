import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from lightning import seed_everything

from sear import logger
from sear.data_processing.multiple_dataset import VGGTMultipleDataset
from sear.metrics.calculator import MetricsCalculator
from sear.scripts.eval.base import ChunkProcessorBase, EvalParametersBase
from sear.scripts.eval.store_results import store_results
from sear.scripts.eval.thermal_vggt import VGGTThermalChunkProcessor


@dataclass(kw_only=True)
class EvalRatiosParametersBase(EvalParametersBase):
    thermal_percent: int = 50
    """Defines the thermal-ratio used to save the """

    num_repeat: int = 3
    """
    How much times to repeat the evaluation on one scene with different seeds for
    thermal ratio.
    """

    thermal_ratio: float = 0.5
    """The thermal_ratio which is thermal_percent / 100"""

    def __post_init__(self) -> None:
        """Initializes necessary values to run the validation."""

        if not 0 <= self.thermal_percent <= 100:
            raise ValueError(
                "The `thermal_percent` must be between 0 and 100 but get "
                + f"{self.thermal_percent}"
            )
        self.method_name = f"{self.method_name}-thermal_percent-{self.thermal_percent}"
        self.thermal_ratio = self.thermal_percent / 100.0


def main(params: EvalRatiosParametersBase, chunk_processor: ChunkProcessorBase) -> None:
    """
    Runs evaluation of a params.method_name on rgb+thermal data on scenes from
    `params.scenes_root_path` and saves the result in `params.output_dir`.
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

    seed_everything(params.seed)
    _, eval_dataset = (
        VGGTMultipleDataset.build_train_eval_datasets_undivided_fixed_split(
            scenes_root_path=params.scenes_root_path,
            depth_eps=params.depth_eps,
            seed=params.seed,
        )
    )

    random_seeds_for_validation = torch.randint(
        0, 2**15, size=(len(eval_dataset) * params.num_repeat,)
    ).tolist()
    logger.info(f"Using random seeds for validation: {random_seeds_for_validation}")

    scenes_ids = params.scenes_ids
    if scenes_ids is None:
        scenes_ids = list(range(len(eval_dataset)))
    scenes_ids = np.array(scenes_ids).repeat(repeats=params.num_repeat).tolist()

    for repeat_index, seed, scene_idx in zip(
        range(len(scenes_ids)), random_seeds_for_validation, scenes_ids
    ):
        seed_everything(seed)
        eval_dataset.thermal_ratio = params.thermal_ratio
        logger.info(f"Set thermal_ratio to {eval_dataset.thermal_ratio}")
        chunk = eval_dataset[scene_idx]
        scene_name = chunk.scenes_names[0]
        dataset_name = chunk.datasets_names[0]
        save_name = f"{dataset_name}:{scene_name}:{repeat_index % params.num_repeat}"

        process_chunk = None
        time_start = time.time()
        try:
            process_chunk = chunk_processor.process_chunk(
                chunk=chunk,
                cache_folder=cache_folder / save_name,
            )
        except Exception as e:
            logger.info(f"Could not process {dataset_name} {scene_name}, {repr(e)}")
        time_end = time.time()

        if process_chunk is not None:
            (
                extrinsics_world2cam,
                intrinsics,
                images_found,
                depths,
                mask_found,
            ) = process_chunk

            store_results(
                images_paths=np.array(chunk.images_paths[0])[mask_found].tolist(),
                images=images_found,
                thermal_mask=chunk.mask_thermal[0][mask_found],
                extrinsics_world2cam=extrinsics_world2cam,
                intrinsics=intrinsics,
                depths=depths,
                ground_truth_extrinsics_world2cam=chunk.extrinsics_world2cam[0][
                    mask_found
                ],
                ground_truth_intrinsics=chunk.intrinsics[0][mask_found],
                ground_truth_depth=chunk.depths[0][mask_found],
                output_folder=store_results_folder / save_name,
                job_name=params.job_name,
                ratio_reconstructed=mask_found.float().mean().item(),
                duration=time_end - time_start,
                method_name=params.method_name,
            )

            calculator.add_data(
                cameras_real_world2cam=chunk.extrinsics_world2cam[0][
                    mask_found
                ].numpy(),
                depths_real=chunk.depths[0][mask_found].numpy(),
                intrinsics_real=chunk.intrinsics[0][mask_found].numpy(),
                cameras_pred_world2cam=extrinsics_world2cam.numpy(),
                depths_pred=depths.numpy(),
                intrinsics_pred=intrinsics.numpy(),
                ratio_reconstructed=mask_found.float().mean().item(),
                duration=time_end - time_start,
                scene_name=scene_name,
                dataset_name=dataset_name,
            )

            logger.info(f"Done {scene_name} {repeat_index % params.num_repeat}!")

    # Save the metrics
    calculator.per_dataset(params.output_folder / "per_dataset.json")
    calculator.per_scene(params.output_folder / "per_scene.json")
    calculator.aggregated(params.output_folder / "aggregated.json")


if __name__ == "__main__":
    params = tyro.cli(EvalRatiosParametersBase)
    chunk_processor = VGGTThermalChunkProcessor(config=params)
    main(params=params, chunk_processor=chunk_processor)
