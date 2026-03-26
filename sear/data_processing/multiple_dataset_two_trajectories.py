import json
from pathlib import Path

import torch

from sear.augment.geometric import GeometricTransform
from sear.augment.rgb import RGBTransformFactory
from sear.augment.thermal import ThermalTransformFactory
from sear.data_processing.chunk import Chunk
from sear.data_processing.multiple_dataset import VGGTMultipleDataset


class MultipleDatasetTwoTrajectories(VGGTMultipleDataset):
    """
    A dataset containing multiple VGGT processed scenes, with two trajectories indices
    (rgb and thermal) speficied. Therefore the  `thermal_ratio` parameter is specified
    and does not play any role.

    The rgb images are taken from the rgb trajectory while the thermal images are taken
    from the thermal trajectory.

    Each element of this dataset is a chunk of elements (images, depths, poses, etc.) of
    the processed dataset, and images in each chunk correspond the same scene.
    """

    def __init__(
        self,
        root_path: Path,
        scenes_names: list[str],
        scenes_per_dataset_path: Path = Path("./sear/configs/scenes_per_dataset.json"),
        depth_eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        scale_poses: bool = True,
    ) -> None:
        """
        Loads the scene located in the `scene_path`. The scenes transforms paths must
        contain information about the trajectory split (rgb and thermal).
        """

        max_sequence_length = VGGTMultipleDataset._get_max_sequence_length(
            scenes_root_path=root_path, scenes_names=scenes_names
        )

        super().__init__(
            root_path=root_path,
            scenes_names=scenes_names,
            min_sequence_length=max_sequence_length,
            elements_number=max_sequence_length + 1,
            shuffle=False,
            drop_last=False,
            rgb_transform=RGBTransformFactory.get_empty(),
            thermal_transform=ThermalTransformFactory.get_empty(),
            geometric_transform=GeometricTransform.empty(),
            scenes_per_dataset_path=scenes_per_dataset_path,
            depth_eps=depth_eps,
            scale_poses=scale_poses,
            dtype=dtype,
            random_seed=0,
        )
        self._thermal_masks_list: list[torch.Tensor] = []

        for scene_idx, scene_name in enumerate(self._scenes_names):
            # read the thermal masks for each single dataset
            with open(root_path / scene_name / "transforms.json") as f:
                transforms = json.load(f)

            rgb_trajectory_indices = transforms["rgb_trajectory"]
            thermal_trajectory_indices = transforms["thermal_trajectory"]

            if (len(rgb_trajectory_indices) + len(thermal_trajectory_indices)) != len(
                self._datasets[scene_idx]
            ):
                raise RuntimeError(
                    "The lengths of `rgb_trajectory_indices` and "
                    + "`thermal_trajectory_indices` must sum up to the length of the "
                    + f"whole dataset, but for {scene_name} of length "
                    + f"{len(self._datasets[scene_idx])} got "
                    + f"{len(rgb_trajectory_indices)} and "
                    + f"{len(thermal_trajectory_indices)}"
                )

            thermal_mask = torch.ones(
                (len(self._datasets[scene_idx]),), dtype=torch.bool
            )

            thermal_mask[rgb_trajectory_indices] = False
            self._thermal_masks_list.append(thermal_mask)

    def __getitem__(self, index: int) -> Chunk:
        """
        Returns a data element at index `index`

        :return: Let the initial image has shapes (H, W, 3), then it returns a Chunk
            containing:
                - images: [N, S, 3, H, W] (rgb or thermal)
                - depths: [N, S, H, W]
                - point_masks: [N, S, H, W]
                - extrinsic_matrices_world2cam: [N, S, 3, 4]
                - intrinsic_matrices: [N, S, 3, 3]
                - thermal_ids: [N, S]
        """

        if index >= len(self):
            raise RuntimeError(f"key {index} is out of range [0, {len(self)}).")

        single_dataset_index = self._get_single_dataset_index(index=index)
        low, high = self.get_chunk_interval(index)

        current_sequence_length = high - low

        mask_thermal = self._thermal_masks_list[single_dataset_index]

        processed_chunk = self.get_chunk_modality_shape_specified(
            index=index,
            mask_thermal=mask_thermal,
            sequence_length=current_sequence_length,
        )

        return processed_chunk

    @classmethod
    def build_eval_dataset(
        cls,
        scenes_root_path: Path,
        scenes_per_dataset_path: Path = Path("./sear/configs/scenes_per_dataset.json"),
    ) -> "MultipleDatasetTwoTrajectories":
        """
        Loads all scenes from `scenes_root_path` as evaluation scenes. The
        `scenes_per_dataset_path` contains information about to what scenes belong to
        what dataset.

        :return: an instance of the model.
        """
        scenes_paths = sorted(list(scenes_root_path.iterdir()))

        return cls(
            root_path=scenes_root_path,
            scenes_names=[path.name for path in scenes_paths],
            scenes_per_dataset_path=scenes_per_dataset_path,
            scale_poses=False,
        )
