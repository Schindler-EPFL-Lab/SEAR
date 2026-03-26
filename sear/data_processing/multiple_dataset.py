import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from training.data.dataset_util import depth_to_world_coords_points
from training.train_utils.normalization import (
    normalize_camera_extrinsics_and_points_batch,
)

from sear import logger
from sear.augment.geometric import GeometricTransform, GeometricTransformConfig
from sear.augment.image_transform import ImageTransform
from sear.augment.rgb import RGBTransformFactory
from sear.augment.thermal import ThermalTransformFactory
from sear.data_processing.chunk import Chunk
from sear.data_processing.single_dataset import Item, VGGTSingleDataset


class VGGTMultipleDataset(torch.utils.data.Dataset):
    """
    A dataset containing multiple VGGT processed scenes.

    Each element of this dataset is a chunk of elements (images, depths, poses, etc.) of
    the processed dataset, and images in each chunk correspond the same scene.
    """

    def __init__(
        self,
        root_path: Path,
        scenes_names: list[str],
        min_sequence_length: int = 2,
        elements_number: int = 24,
        drop_last: bool = True,
        shuffle: bool = True,
        rgb_transform: ImageTransform | None = None,
        thermal_transform: ImageTransform | None = None,
        geometric_transform: GeometricTransform | None = None,
        scenes_per_dataset_path: Path = Path(
            "./sear/configs/scenes_per_dataset.json"
        ),
        depth_eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
        scale_poses: bool = True,
        random_seed: int = 0,
    ) -> None:
        """
        Instantiates a dataset containing multiple VGGT-processed scenes.

        The `root_path` specifies the path where the processed scenes are stored; each
        scene is in an independent directory. The `scenes_names` specifies which scenes
        to take for this dataset. The `elements_number` defines the number of frames in
        each batch. Each batch is split into several sequences with
        `min_sequence_length` defines the minimum length of a sequence. The `drop_last`
        parameter determines whether to drop the last non-full chunk of each scene. The
        `shuffle` parameter specifies whether to shuffle elements within each scene. It
        is guaranteed that during one iteration through the dataset, chunks of the same
        scene contain different images. The `rgb_transform` is a class to perform rgb
        augmentations, if it is None then the default augmentator is created. The
        `thermal_transform` is a class to peform thermal augmentations, if it is None
        then the default thermal augmentation is created. the `geometric_transform` is a
        class to perform geometric augmentations, if it is None then the default thermal
        augmentation is created. The `scenes_per_dataset_path` contains information
        about to what scenes belong to what dataset. The `depth_eps` parameter specifies
        the threshold below which depth pixels are considered *invalid*. The `dtype`
        parameter defines the data type of the returned tensors. The `random_seed`
        specifies the random seed which is set before shuffling. The `scale_poses`
        defines whether to scale poses during chunk formation, which is helpful for
        validation.
        """
        super().__init__()

        self._root_path = root_path
        self._scenes_names = scenes_names
        self._sequence_length = elements_number
        self._drop_last = drop_last
        self._shuffle = shuffle
        self._rgb_transform = (
            RGBTransformFactory().create_transform()
            if rgb_transform is None
            else rgb_transform
        )
        self._thermal_transform = (
            ThermalTransformFactory().create_transform()
            if thermal_transform is None
            else thermal_transform
        )
        self._geometric_transform = (
            GeometricTransform() if geometric_transform is None else geometric_transform
        )

        self._dtype = dtype
        self._depth_eps = depth_eps
        self._random_seed = random_seed
        self._do_scale_poses = scale_poses
        self.thermal_ratio = 0.5
        self._scenes_per_dataset_path = scenes_per_dataset_path

        # load all single-scene datasets
        self._datasets: list[VGGTSingleDataset] = []
        for i, scene_name in enumerate(self._scenes_names):
            self._datasets.append(
                VGGTSingleDataset(
                    scene_path=self._root_path / scene_name,
                    dtype=self._dtype,
                )
            )
            logger.info(f"Loaded {i + 1}/{len(scenes_names)} {scene_name}.")

        # find the number of elements in the dataset
        self._single_datasets_lengths = np.array(
            [len(dataset) for dataset in self._datasets]
        )
        self._number_of_chunks = self._single_datasets_lengths / self._sequence_length
        self._min_sequence_length = min_sequence_length

        self._possible_sequence_lengths = [
            i
            for i in range(min_sequence_length, elements_number + 1)
            if elements_number % i == 0
        ]

        self._number_of_chunks = (
            np.floor(self._number_of_chunks).astype(int)
            if self._drop_last
            else np.ceil(self._number_of_chunks).astype(int)
        )

        self._chunks_cumsum = np.cumsum([0] + self._number_of_chunks.tolist())

        self._shuffled_indices = [
            np.arange(single_datasets_length)
            for single_datasets_length in self._single_datasets_lengths
        ]
        self._number_access = np.zeros_like(self._single_datasets_lengths)
        if self._shuffle:
            np.random.seed(self._random_seed)

            for i in range(len(self._single_datasets_lengths)):
                self._reshuffle(i)

    @staticmethod
    def collate_chunks(batch: Sequence[Chunk]) -> Chunk:
        """
        Combines chunk elements in `batch` into one chunk element. If the shapes are
        incorrect then the combining fails.
        """

        return Chunk(
            images=torch.cat([chunk.images for chunk in batch]),
            images_paths=[
                image_path for chunk in batch for image_path in chunk.images_paths
            ],
            depths=torch.cat([chunk.depths for chunk in batch]),
            depths_paths=[
                depth_path for chunk in batch for depth_path in chunk.depths_paths
            ],
            extrinsics_world2cam=torch.cat(
                [chunk.extrinsics_world2cam for chunk in batch]
            ),
            intrinsics=torch.cat([chunk.intrinsics for chunk in batch]),
            point_masks=torch.cat([chunk.point_masks for chunk in batch]),
            mask_thermal=torch.cat([chunk.mask_thermal for chunk in batch]),
            scenes_per_dataset_path=batch[0].scenes_per_dataset_path,
        )

    @property
    def scenes_paths(self) -> list[Path]:
        """Returns paths to the underlying scenes."""
        return [self._root_path / scene_name for scene_name in self._scenes_names]

    def __len__(self) -> int:
        """Returns the length of the dataset"""
        return self._chunks_cumsum[-1]

    def _reshuffle(self, single_dataset_index: int) -> None:
        """Shuffles the indices of the single dataset at index `single_dataset_index`"""
        if not self._shuffle:
            return

        if single_dataset_index >= len(self._shuffled_indices):
            raise RuntimeError(
                f"key {single_dataset_index} is out of range "
                + "[0, {len(self._shuffled_indices)})."
            )

        self._shuffled_indices[single_dataset_index] = np.random.permutation(
            np.arange(len(self._shuffled_indices[single_dataset_index]))
        )

    def _get_single_dataset_index(self, index: int) -> int:
        """
        Returns the index of the single dataset which corresponds to the chunk with
        `index`
        """
        return np.searchsorted(self._chunks_cumsum, index, side="right").item() - 1

    @staticmethod
    def _scale_poses(
        depths: torch.Tensor,
        extrinsic_matrices_world2cam: torch.Tensor,
        intrinsic_matrices: torch.Tensor,
        depth_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes `depths` and `extrinsic_matrices_world2cam` to a common scale, such
        that the average depth value equals to 1.0 if rendered from camera with
        parameters `extrinsic_matrices_world2cam` and `intrinsic_matrices`. Calculates
        masks of valid depth based on `depth_eps` - depth values below this threshold
        considered to be invalid.

        :returns a tuple of processed depth, point_mask, and camera extrinsics.
        """

        point_masks_list: list[torch.Tensor] = []
        world_coordinate_points_all_list: list[torch.Tensor] = []
        camera_coordinate_points_all_list: list[torch.Tensor] = []

        for i in range(depths.shape[0]):
            for j in range(depths.shape[1]):
                world_coordinate_points, camera_coordinate_points, point_mask = (
                    depth_to_world_coords_points(
                        depth_map=depths[i, j].numpy(),
                        extrinsic=extrinsic_matrices_world2cam[i, j].numpy(),
                        intrinsic=intrinsic_matrices[i, j].numpy(),
                        eps=depth_eps,
                    )
                )

                world_coordinate_points_all_list.append(
                    torch.from_numpy(world_coordinate_points).to(depths)
                )
                camera_coordinate_points_all_list.append(
                    torch.from_numpy(camera_coordinate_points).to(depths)
                )
                point_masks_list.append(torch.from_numpy(point_mask).to(torch.bool))

        world_coordinate_points_all = torch.stack(world_coordinate_points_all_list)
        world_coordinate_points_all = world_coordinate_points_all.reshape(
            *depths.shape[:2], *world_coordinate_points_all.shape[1:]
        )
        camera_coordinate_points_all = torch.stack(camera_coordinate_points_all_list)
        camera_coordinate_points_all = camera_coordinate_points_all.reshape(
            *depths.shape[:2], *camera_coordinate_points_all.shape[1:]
        )
        point_masks = torch.stack(point_masks_list)
        point_masks = point_masks.reshape(*depths.shape[:2], *point_masks.shape[1:])

        extrinsic_matrices_world2cam, _, _, depths = (
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=extrinsic_matrices_world2cam,
                cam_points=camera_coordinate_points_all,
                world_points=world_coordinate_points_all,
                depths=depths,
                scale_by_points=True,
                point_masks=point_masks,
            )
        )

        return depths, point_masks, extrinsic_matrices_world2cam

    @staticmethod
    def _torch_where_lists(
        x: list[torch.Tensor],
        y: list[torch.Tensor],
        condition: list[torch.Tensor] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Creates a tensor from elements of `x` and `y` depending on `condition`. If
        `condition[i]` is True then `x[i]` is taken and `y[i]` is taken otherwise.

        :return: a tensor from elements of either `x` or `y`.

        :raise: RuntimeError if lenghts of `x` or `y` or `condition` mismatch.
        """

        if not len(condition) == len(x) == len(y):
            raise RuntimeError(
                f"Lengths of `x`, `y` and `condition` must match, but get {len(x)}, "
                + f"{len(y)}, {len(condition)} respectively."
            )
        return torch.stack([x[i] if condition[i] else y[i] for i in range(len(x))])

    def _process_chunk(
        self,
        chunk_items: list[Item],
        mask_thermal: torch.Tensor,
        sequence_length: int,
    ) -> Chunk:
        """
        Process a `chunk_items` of per-frame tensors into a single, normalized sample
        for training. The chunk is split into pieces of length `sequence_length` if
        possible, otherwise the sequence length remains the same.

        :returns a chunk of processed and augmented data.
        """

        current_sequence_length = len(chunk_items)
        desired_sequence_length = (
            sequence_length
            if current_sequence_length % sequence_length == 0
            else current_sequence_length
        )

        images_paths = np.where(
            mask_thermal.numpy(),
            np.array([item.thermal_path for item in chunk_items]),
            np.array([item.image_path for item in chunk_items]),
        )
        images_torch = self._torch_where_lists(
            [item.image for item in chunk_items],
            [item.thermal for item in chunk_items],
            ~mask_thermal,
        )

        if torch.any(mask_thermal):
            images_torch[mask_thermal] = self._thermal_transform.apply(
                images_torch[mask_thermal]
            )
        images_torch[~mask_thermal] = self._rgb_transform.apply(
            images_torch[~mask_thermal]
        )

        depths_paths = np.where(
            mask_thermal.numpy(),
            np.array([item.depth_thermal_path for item in chunk_items]),
            np.array([item.depth_rgb_path for item in chunk_items]),
        )
        depths_torch = self._torch_where_lists(
            [item.depth_rgb for item in chunk_items],
            [item.depth_thermal for item in chunk_items],
            ~mask_thermal,
        )

        extrinsic_matrices_world2cam_torch = self._torch_where_lists(
            [item.extrinsic_world2cam_rgb for item in chunk_items],
            [item.extrinsic_world2cam_thermal for item in chunk_items],
            ~mask_thermal,
        )

        intrinsic_matrices_torch = self._torch_where_lists(
            [item.intrinsic_rgb for item in chunk_items],
            [item.intrinsic_thermal for item in chunk_items],
            ~mask_thermal,
        )

        (
            images_torch,
            depths_torch,
            extrinsic_matrices_world2cam_torch,
            intrinsic_matrices_torch,
        ) = self._geometric_transform.apply(
            images=images_torch,
            depths=depths_torch,
            intrinsic_matrices=intrinsic_matrices_torch,
            extrinsic_matrices_world2cam=extrinsic_matrices_world2cam_torch,
        )

        images_paths = images_paths.reshape(-1, desired_sequence_length)
        images_torch = images_torch.reshape(
            -1, desired_sequence_length, *images_torch.shape[1:]
        )
        depths_paths = depths_paths.reshape(-1, desired_sequence_length)
        depths_torch = depths_torch.reshape(
            -1, desired_sequence_length, *depths_torch.shape[1:]
        )
        extrinsic_matrices_world2cam_torch = extrinsic_matrices_world2cam_torch.reshape(
            -1, desired_sequence_length, *extrinsic_matrices_world2cam_torch.shape[1:]
        )
        intrinsic_matrices_torch = intrinsic_matrices_torch.reshape(
            -1, desired_sequence_length, *intrinsic_matrices_torch.shape[1:]
        )
        mask_thermal = mask_thermal.reshape(
            -1, desired_sequence_length, *mask_thermal.shape[1:]
        )

        if self._do_scale_poses:
            depths_torch, point_masks_torch, extrinsic_matrices_world2cam_torch = (
                self._scale_poses(
                    depths=depths_torch,
                    extrinsic_matrices_world2cam=extrinsic_matrices_world2cam_torch,
                    intrinsic_matrices=intrinsic_matrices_torch,
                    depth_eps=self._depth_eps,
                )
            )
        else:
            point_masks_torch = depths_torch <= self._depth_eps

        return Chunk(
            images=images_torch,
            images_paths=images_paths.tolist(),
            depths=depths_torch,
            depths_paths=depths_paths.tolist(),
            extrinsics_world2cam=extrinsic_matrices_world2cam_torch,
            intrinsics=intrinsic_matrices_torch,
            point_masks=point_masks_torch,
            mask_thermal=mask_thermal,
            scenes_per_dataset_path=self._scenes_per_dataset_path,
        )

    def get_chunk_interval(self, index: int) -> tuple[int, int]:
        """
        For each `index` finds the of interval of chunk items taken from the
        corresponding single_dataset.
        """
        if index >= len(self):
            raise RuntimeError(f"key {index} is out of range [0, {len(self)}).")

        single_dataset_index = self._get_single_dataset_index(index=index)

        chunk_index_in_single_dataset = (
            index - self._chunks_cumsum[single_dataset_index]
        )

        low = chunk_index_in_single_dataset * self._sequence_length
        high = (chunk_index_in_single_dataset + 1) * self._sequence_length
        high = min(self._single_datasets_lengths[single_dataset_index], high)
        return low, high

    def get_chunk_modality_shape_specified(
        self, index: int, mask_thermal: torch.Tensor, sequence_length: int
    ) -> Chunk:
        """
        Returns chunk at `index` when `mask_thermal` and `sequence_length` are
        specified.

        :raise: RuntimeError if
            - The `index` is out of range,
            - The `mask_thermal` has incompatible size,
            - The number of elements in a chunk is not divisible by `sequence_length`.
        """
        if index >= len(self):
            raise RuntimeError(f"key {index} is out of range [0, {len(self)}).")

        single_dataset_index = self._get_single_dataset_index(index=index)
        single_dataset = self._datasets[single_dataset_index]

        low, high = self.get_chunk_interval(index)
        current_sequence_length = high - low

        if mask_thermal.numel() != current_sequence_length:
            raise RuntimeError(
                f"The `mask_thermal` has shape {mask_thermal.shape} which is not equal "
                + f"to the number of elements in the chunk at {index} which is "
                + f"{current_sequence_length}."
            )

        if (
            sequence_length not in self._possible_sequence_lengths
            and sequence_length != current_sequence_length
        ):
            raise RuntimeError(
                f"The number of elements {current_sequence_length} in the chunk at "
                + f"{index} must be divisible by the sequence_length {sequence_length}."
            )

        chunk_items: list[Item] = []

        for i in range(low, high):
            element_index = (
                self._shuffled_indices[single_dataset_index][i] if self._shuffle else i
            )
            element = single_dataset[element_index]
            chunk_items.append(element)

        return self._process_chunk(
            chunk_items=chunk_items,
            sequence_length=sequence_length,
            mask_thermal=mask_thermal.flatten(),
        )

    def __getitem__(self, index: int) -> Chunk:
        """
        Returns a data element at index `index`

        :returns Let the initial image has shapes (H, W, 3), then the return is a tuple:
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

        # Generate thermal mask
        num_thermals = math.ceil(current_sequence_length * self.thermal_ratio)
        ids_thermal = torch.randperm(n=current_sequence_length)[:num_thermals]
        mask_thermal = torch.zeros((current_sequence_length,), dtype=torch.bool)
        mask_thermal[ids_thermal] = True

        # Generate sequence length
        random_index = torch.randint(
            low=0, high=len(self._possible_sequence_lengths), size=(1,)
        ).item()
        random_index = int(random_index)
        random_sequence_length = self._possible_sequence_lengths[random_index]

        processed_chunk = self.get_chunk_modality_shape_specified(
            index=index,
            mask_thermal=mask_thermal,
            sequence_length=random_sequence_length,
        )

        if not self._shuffle:
            return processed_chunk

        self._number_access[single_dataset_index] += current_sequence_length
        if (
            self._number_access[single_dataset_index]
            >= self._single_datasets_lengths[single_dataset_index]
        ):
            self._number_access[single_dataset_index] = 0
            self._reshuffle(single_dataset_index)

        return processed_chunk

    @staticmethod
    def _get_max_sequence_length(
        scenes_root_path: Path,
        scenes_names: list[str],
    ) -> int:
        """
        Iterates over scenes `scenes_names` located in `scene_root_path` and determines
        the maximum number of elements in each dataset.
        """

        max_sequence_length = 0
        for scene in scenes_names:
            with open(scenes_root_path / scene / "transforms.json", "r") as f:
                data = json.load(f)
                max_sequence_length = max(max_sequence_length, len(data["frames"]))
        return max_sequence_length

    @classmethod
    def _train_eval_datasets(
        cls,
        scenes_root_path: Path,
        elements_number: int,
        depth_eps: float,
        seed: int,
        rgb_transform_factory: RGBTransformFactory,
        thermal_transform_factory: ThermalTransformFactory,
        geometric_transform_config: GeometricTransformConfig,
        train_scenes_names: list[str],
        eval_scenes_names: list[str],
        scenes_per_dataset_path: Path = Path(
            "./sear/configs/scenes_per_dataset.json"
        ),
    ) -> tuple["VGGTMultipleDataset", "VGGTMultipleDataset"]:
        """
        Creates training and validation datasets from a directory of scene folders. The
        `scenes_root_path` is root directory containing scene subdirectories, `seed` is
        a random seed used for dataset shuffling, `elements_number` is the number of
        frames per sample sequence provided by the datasets, `depth_eps` is a depth
        value such that depths smaller this value do not take part in training. The
        `rgb_transform_factory` is a configuration to create rgb transform, the
        `thermal_transform_factory` is a configuration to create thermal transform, the
        `geometric_transform_config` is a configuration to create geometric transform.
        The `scenes_per_dataset_path` contains information about to what scenes belong
        to what dataset. The `train_scenes_names` and `eval_scenes_names` are scenes
        names in training and validation split.

        :return: a tuple of train and eval datasets
        """

        train_dataset = cls(
            root_path=scenes_root_path,
            scenes_names=train_scenes_names,
            min_sequence_length=2,
            elements_number=elements_number,
            shuffle=True,
            rgb_transform=rgb_transform_factory.create_transform(),
            thermal_transform=thermal_transform_factory.create_transform(),
            geometric_transform=GeometricTransform(geometric_transform_config),
            scenes_per_dataset_path=scenes_per_dataset_path,
            depth_eps=depth_eps,
            dtype=torch.float32,
            random_seed=seed,
        )

        max_sequence_length = cls._get_max_sequence_length(
            scenes_root_path=scenes_root_path, scenes_names=eval_scenes_names
        )

        eval_dataset = cls(
            root_path=scenes_root_path,
            scenes_names=eval_scenes_names,
            min_sequence_length=max_sequence_length,
            elements_number=max_sequence_length + 1,
            shuffle=False,
            drop_last=False,
            rgb_transform=RGBTransformFactory.get_empty(),
            thermal_transform=ThermalTransformFactory.get_empty(),
            geometric_transform=GeometricTransform.empty(),
            scenes_per_dataset_path=scenes_per_dataset_path,
            depth_eps=depth_eps,
            dtype=torch.float32,
            random_seed=seed,
        )

        return train_dataset, eval_dataset

    @classmethod
    def build_train_eval_datasets(
        cls,
        scenes_root_path: Path,
        val_split_ratio: float,
        elements_number: int,
        depth_eps: float,
        seed: int,
        rgb_transform_factory: RGBTransformFactory,
        thermal_transform_factory: ThermalTransformFactory,
        geometric_transform_config: GeometricTransformConfig,
        scenes_per_dataset_path: Path = Path(
            "./sear/configs/scenes_per_dataset.json"
        ),
    ) -> tuple["VGGTMultipleDataset", "VGGTMultipleDataset"]:
        """
        Creates training and validation datasets from a directory of scene folders. The
        `scenes_root_path` is root directory containing scene subdirectories,
        `val_split_ratio` is a fraction of scenes to reserve for validation, `seed` is a
        random seed used for reproducible train/validation split and dataset shuffling,
        `elements_number` is the number of frames per sample sequence provided by the
        datasets, `depth_eps` is a depth value such that depths smaller this value do
        not take part in training. The `rgb_transform_factory` is a configuration to
        create rgb transform, the `thermal_transform_factory` is a configuration to
        create thermal transform, the `geometric_transform_config` is a configuration to
        create geometric transform. The `scenes_per_dataset_path` contains information
        about to what scenes belong to what dataset.

        :return: a tuple of train and eval datasets
        """

        all_scenes_names = scenes_root_path.iterdir()
        all_scenes_names = sorted([el.name for el in all_scenes_names if el.is_dir()])

        train_scenes_names, eval_scenes_names = train_test_split(
            all_scenes_names, test_size=val_split_ratio, random_state=seed
        )
        eval_scenes_names = sorted(eval_scenes_names)

        return cls._train_eval_datasets(
            scenes_root_path=scenes_root_path,
            elements_number=elements_number,
            depth_eps=depth_eps,
            seed=seed,
            rgb_transform_factory=rgb_transform_factory,
            thermal_transform_factory=thermal_transform_factory,
            geometric_transform_config=geometric_transform_config,
            train_scenes_names=train_scenes_names,
            eval_scenes_names=eval_scenes_names,
            scenes_per_dataset_path=scenes_per_dataset_path,
        )

    @classmethod
    def build_train_eval_datasets_fixed_split(
        cls,
        scenes_root_path: Path,
        elements_number: int,
        depth_eps: float,
        seed: int,
        rgb_transform_factory: RGBTransformFactory,
        thermal_transform_factory: ThermalTransformFactory,
        geometric_transform_config: GeometricTransformConfig,
        scenes_per_dataset_path: Path = Path(
            "./sear/configs/scenes_per_dataset.json"
        ),
        train_test_split_path: Path = Path(
            "./sear/configs/train_test_split.json"
        ),
    ) -> tuple["VGGTMultipleDataset", "VGGTMultipleDataset"]:
        """
        Creates training and validation datasets from a directory of scene folders. The
        `scenes_root_path` is root directory containing scene subdirectories, `seed` is
        a random seed used for dataset shuffling, `elements_number` is the number of
        frames per sample sequence provided by the datasets, `depth_eps` is a depth
        value such that depths smaller this value do not take part in training. The
        `rgb_transform_factory` is a configuration to create rgb transform, the
        `thermal_transform_factory` is a configuration to create thermal transform, the
        `geometric_transform_config` is a configuration to create geometric transform.
        The `scenes_per_dataset_path` contains information about to what scenes belong
        to what dataset.The `train_test_split_path` contains names of scenes for
        train/test split.

        :return: a tuple of train and eval datasets
        """

        with open(train_test_split_path) as f:
            train_test_split = json.load(f)

        train_scenes_names = sorted(train_test_split["train"])
        eval_scenes_names = sorted(train_test_split["eval"])

        return cls._train_eval_datasets(
            scenes_root_path=scenes_root_path,
            elements_number=elements_number,
            depth_eps=depth_eps,
            seed=seed,
            rgb_transform_factory=rgb_transform_factory,
            thermal_transform_factory=thermal_transform_factory,
            geometric_transform_config=geometric_transform_config,
            train_scenes_names=train_scenes_names,
            eval_scenes_names=eval_scenes_names,
            scenes_per_dataset_path=scenes_per_dataset_path,
        )

    @classmethod
    def _train_eval_datasets_undivided(
        cls,
        scenes_root_path: Path,
        depth_eps: float,
        seed: int,
        train_scenes_names: list[str],
        eval_scenes_names: list[str],
        scenes_per_dataset_path: Path = Path(
            "./sear/configs/scenes_per_dataset.json"
        ),
    ) -> tuple["VGGTMultipleDataset", "VGGTMultipleDataset"]:
        """
        Creates training and validation datasets with no augmentations with datasets not
        split into smaller pieces. The `scenes_root_path` is root directory containing
        scene subdirectories, `val_split_ratio` is a fraction of scenes to reserve for
        validation, `seed` is a random seed used for reproducible train/validation split
        and dataset shuffling, `depth_eps` is a depth value such that depths smaller
        this value do not take part in training. The `scenes_per_dataset_path` contains
        information about to what scenes belong to what dataset. The
        `train_scenes_names` and `eval_scenes_names` are scenes names in training and
        validation split.

        :return: a tuple of train and eval datasets
        """
        max_sequence_length = cls._get_max_sequence_length(
            scenes_root_path=scenes_root_path,
            scenes_names=train_scenes_names + eval_scenes_names,
        )

        train_dataset = cls(
            root_path=scenes_root_path,
            scenes_names=train_scenes_names,
            min_sequence_length=max_sequence_length,
            elements_number=max_sequence_length + 1,
            shuffle=False,
            drop_last=False,
            rgb_transform=RGBTransformFactory.get_empty(),
            thermal_transform=ThermalTransformFactory.get_empty(),
            geometric_transform=GeometricTransform.empty(),
            scenes_per_dataset_path=scenes_per_dataset_path,
            depth_eps=depth_eps,
            dtype=torch.float32,
            random_seed=seed,
        )

        eval_dataset = cls(
            root_path=scenes_root_path,
            scenes_names=eval_scenes_names,
            min_sequence_length=max_sequence_length,
            elements_number=max_sequence_length + 1,
            shuffle=False,
            drop_last=False,
            rgb_transform=RGBTransformFactory.get_empty(),
            thermal_transform=ThermalTransformFactory.get_empty(),
            geometric_transform=GeometricTransform.empty(),
            scenes_per_dataset_path=scenes_per_dataset_path,
            depth_eps=depth_eps,
            dtype=torch.float32,
            scale_poses=False,
            random_seed=seed,
        )

        return train_dataset, eval_dataset

    @classmethod
    def build_train_eval_datasets_undivided(
        cls,
        scenes_root_path: Path,
        val_split_ratio: float,
        depth_eps: float,
        seed: int,
        scenes_per_dataset_path: Path = Path(
            "./sear/configs/scenes_per_dataset.json"
        ),
    ) -> tuple["VGGTMultipleDataset", "VGGTMultipleDataset"]:
        """
        Creates training and validation datasets with no augmentations with datasets not
        split into smaller pieces. The `scenes_root_path` is root directory containing
        scene subdirectories, `val_split_ratio` is a fraction of scenes to reserve for
        validation, `seed` is a random seed used for reproducible train/validation split
        and dataset shuffling, `depth_eps` is a depth value such that depths smaller
        this value do not take part in training. The `scenes_per_dataset_path` contains
        information about to what scenes belong to what dataset.

        :return: a tuple of train and eval datasets
        """

        all_scenes_names = scenes_root_path.iterdir()
        all_scenes_names = sorted([el.name for el in all_scenes_names if el.is_dir()])

        train_scenes_names, eval_scenes_names = train_test_split(
            all_scenes_names, test_size=val_split_ratio, random_state=seed
        )

        eval_scenes_names = sorted(eval_scenes_names)
        train_scenes_names = sorted(train_scenes_names)
        return cls._train_eval_datasets_undivided(
            scenes_root_path=scenes_root_path,
            depth_eps=depth_eps,
            seed=seed,
            train_scenes_names=train_scenes_names,
            eval_scenes_names=eval_scenes_names,
            scenes_per_dataset_path=scenes_per_dataset_path,
        )

    @classmethod
    def build_train_eval_datasets_undivided_fixed_split(
        cls,
        scenes_root_path: Path,
        depth_eps: float,
        seed: int,
        scenes_per_dataset_path: Path = Path(
            "./sear/configs/scenes_per_dataset.json"
        ),
        train_test_split_path: Path = Path(
            "./sear/configs/train_test_split.json"
        ),
    ) -> tuple["VGGTMultipleDataset", "VGGTMultipleDataset"]:
        """
        Creates training and validation datasets with no augmentations with datasets not
        split into smaller pieces. The `scenes_root_path` is root directory containing
        scene subdirectories, `seed` is a random seed used for dataset shuffling,
        `depth_eps` is a depth value such that depths smaller this value do not take
        part in training. The `scenes_per_dataset_path` contains information about to
        what scenes belong to what dataset. The `train_test_split_path` contains names
        of scenes for train/test split.

        :return: a tuple of train and eval datasets
        """
        with open(train_test_split_path) as f:
            train_test_split = json.load(f)

        train_scenes_names = sorted(train_test_split["train"])
        eval_scenes_names = sorted(train_test_split["eval"])

        return cls._train_eval_datasets_undivided(
            scenes_root_path=scenes_root_path,
            depth_eps=depth_eps,
            seed=seed,
            train_scenes_names=train_scenes_names,
            eval_scenes_names=eval_scenes_names,
            scenes_per_dataset_path=scenes_per_dataset_path,
        )
