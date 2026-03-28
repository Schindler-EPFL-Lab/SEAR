import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from lightning import seed_everything
from romatch.utils.utils import compute_relative_pose, estimate_pose

from sear import logger
from sear.data_processing.chunk import Chunk
from sear.data_processing.paired_dataset import PairedDataset
from sear.data_processing.paired_item import PairedItem
from sear.metrics.calculator import MetricsCalculator


@dataclass(kw_only=True)
class PairsEvalParametersBase(ReverseCli):
    """A class to evaluate relative camera pose reconstruction between two images."""

    scenes_root_path: Path = Path("scenes-root-path")
    """Directory containing processed VGGT scenes"""
    store_results_folder: Path = Path("./outputs")
    """Directory to save files to calculate metrics of the method"""
    depth_eps: float = 1e-8
    """Depth value smaller this value do not take part in training"""
    val_split_ratio: float = 0.2
    """Ratio of scenes used for validation"""
    seed: int = 0
    """Random seed for the dataset"""

    method_name: str
    """Method name to store the results"""

    batch_size: int = 32
    """Batch size (number of pairs) to avoid Cuda out of Memory errors."""
    output_folder: Path = Path("./outputs")
    """Directory to save metrics files"""

    def __post_init__(self) -> None:
        """Defines the variables necessary for child classes."""
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def load_model(self) -> None:
        """Loads the model"""
        raise NotImplementedError("The `load_model` is an abstractmethod")

    @abstractmethod
    def run_pairs(
        self,
        chunk: Chunk | PairedItem,
        cache_folder: Path,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raise NotImplementedError("The `run_pairs` is an abstractmethod")


@dataclass
class PairsEvalKeypointsParametersBase(PairsEvalParametersBase):
    """
    A class to evaluate relative camera pose reconstruction between two images using the
    keypoint matching methods.
    """

    @abstractmethod
    def _find_keypoints(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        visualize_keypoints_path: Path | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Finds keypoints between images `image1` and `image2` using the inner model.
        Optionally saves the visualization of the found keypoints if the
        `visualize_keypoints_path` is provided.
        """
        raise NotImplementedError("The `_find_keypoints` is an abstractmethod")

    def run_pairs(
        self,
        chunk: PairedItem,
        cache_folder: Path,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Runs the keypoint matching method on images pairs generated from `chunk`. The
        cache files (if any) created during the colmap run are be stored in
        `cache_folder`.

        :return: the real and the predicted relative poses of the image pairs.
        """

        cache_folder.mkdir(exist_ok=True)
        relative_transforms_pred_list: list[npt.NDArray[np.float64]] = []
        relative_transforms_real_list: list[npt.NDArray[np.float64]] = []
        generator = chunk.iterate_batched(batch_size=self.batch_size)

        for chunk_batched in generator:
            # In this case it is easier to process the images pair by pair instead of
            # the batch due to some difficulties in the match-anything batching
            with torch.inference_mode():
                for pair_idx in range(chunk_batched.images.shape[0]):
                    image1_path = chunk_batched.images_paths[pair_idx][0]
                    image2_path = chunk_batched.images_paths[pair_idx][1]

                    kpts0, kpts1 = self._find_keypoints(
                        image1=chunk_batched.images[pair_idx][0],
                        image2=chunk_batched.images[pair_idx][1],
                        visualize_keypoints_path=cache_folder
                        / f"{image1_path.stem}-{image2_path.stem}.png",
                    )

                    K0 = chunk_batched.intrinsics[pair_idx, 0].numpy()
                    K1 = chunk_batched.intrinsics[pair_idx, 1].numpy()
                    norm_threshold = 0.5 / (
                        np.mean(np.absolute(K0[:2, :2]))
                        + np.mean(np.absolute(K1[:2, :2]))
                    )

                    try:
                        R_0_to_1, t_0_to_1, _ = estimate_pose(
                            kpts0=kpts0,
                            kpts1=kpts1,
                            K0=K0,
                            K1=K1,
                            norm_thresh=norm_threshold,
                            conf=0.99999,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Cannot find relative pose for {cache_folder.name}: "
                            + f"{image1_path.stem} <---> {image2_path.stem} because of"
                            + f"\n{repr(e)}"
                        )
                        continue

                    tf_0_to_1 = np.concatenate([R_0_to_1, t_0_to_1], axis=-1)

                    R_0_to_1_real, t_0_to_1_real = compute_relative_pose(
                        R1=chunk_batched.extrinsics_world2cam[pair_idx, 0, :3, :3],
                        t1=chunk_batched.extrinsics_world2cam[pair_idx, 0, :3, 3],
                        R2=chunk_batched.extrinsics_world2cam[pair_idx, 1, :3, :3],
                        t2=chunk_batched.extrinsics_world2cam[pair_idx, 1, :3, 3],
                    )
                    tf_0_to_1_real = torch.cat(
                        [R_0_to_1_real, t_0_to_1_real[:, None]], dim=-1
                    )
                    relative_transforms_pred_list.append(tf_0_to_1)
                    relative_transforms_real_list.append(tf_0_to_1_real.numpy())

        if len(relative_transforms_real_list) == 0:
            return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

        return (
            np.stack(relative_transforms_real_list),
            np.stack(relative_transforms_pred_list),
        )


def main(params: PairsEvalParametersBase) -> None:
    """
    Runs relative camera pose evaluation method on pairs of images generated from the
    evaluation datasets in `params.scenes_root_path` and saves the result in
    `params.output_dir`.
    """

    params.load_model()

    store_results_folder = params.store_results_folder
    store_results_folder.mkdir(exist_ok=True, parents=True)

    output_folder = (
        params.output_folder
        if params.output_folder.is_absolute()
        else Path.cwd() / params.output_folder
    )

    output_folder.mkdir(exist_ok=True)
    cache_folder = output_folder / "cache"
    cache_folder.mkdir(exist_ok=True)

    calculator = MetricsCalculator(thresholds=[5.0, 10.0, 20.0])

    seed_everything(params.seed)
    eval_dataset = PairedDataset(
        dataset_path=params.scenes_root_path,
    )

    logger.info("Running Validation on METU_VisTIR dataset")

    relative_poses_all: dict[str, dict[str, list[list[float]]]] = {}

    for scene_idx in range(len(eval_dataset)):
        chunk = eval_dataset[scene_idx]
        scene_name = chunk.scenes_names[0]
        dataset_name = chunk.datasets_names[0]
        save_name = f"{dataset_name}:{scene_name}"

        (relative_transforms_real, relative_transforms_pred) = params.run_pairs(
            chunk=chunk,
            cache_folder=cache_folder / save_name,
        )
        calculator.add_data_relative(
            relative_cameras_real_cam2world=relative_transforms_real,
            relative_cameras_pred_cam2world=relative_transforms_pred,
            scene_name=scene_name,
            dataset_name=dataset_name,
        )

        if save_name not in relative_poses_all:
            relative_poses_all[save_name] = {
                "real": [],
                "pred": [],
            }
        relative_poses_all[save_name]["real"].extend(relative_transforms_real.tolist())
        relative_poses_all[save_name]["pred"].extend(relative_transforms_pred.tolist())

        logger.info(f"Done {scene_name} approx. {scene_idx + 1}/{len(eval_dataset)}")

    with open(store_results_folder / "relative_poses_all.json", "w") as f:
        json.dump(relative_poses_all, f, indent=4)

    calculator.per_dataset(output_folder / "per_dataset.json")
    calculator.per_scene(output_folder / "per_scene.json")
    calculator.aggregated(output_folder / "aggregated.json")

    with open(store_results_folder / "relative_poses_all.json", "w") as f:
        json.dump(relative_poses_all, f, indent=4)

    calculator.per_dataset(output_folder / "per_dataset.json")
    calculator.per_scene(output_folder / "per_scene.json")
    calculator.aggregated(output_folder / "aggregated.json")
