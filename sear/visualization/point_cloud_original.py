import gc
from dataclasses import dataclass
from pathlib import Path

import open3d as o3d
import torch
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import randomly_limit_trues
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from sear.data_processing.multiple_dataset import VGGTMultipleDataset
from sear.run_mode import RunMode


@dataclass(kw_only=True)
class PointCloudOriginalCreator(ReverseCli):
    """
    Creates point cloud from a folder with ThermoScenes scenes using the original VGGT
    model. For each scene it randomly chooses half images thermal and half images rgb
    with no shared poses, and inferences the original VGGTmodel.
    """

    original_ckpt_path: Path = Path("original-ckpt-path")
    """Loads original vggt model from this path"""
    scenes_root_path: Path = Path("input")
    """The images are taken from scene"""
    run_mode: RunMode = RunMode.EVAL
    """Create point clouds from train, eval, or all scenes in `scenes_root_path`"""
    depth_eps: float = 1e-8
    """
    The `depth_eps` is a depth value such that pixels with smaller depth are considered
    to be invalid.
    """
    depth_conf_quantile: float = 0.5
    """
    Depth values with small confidence (lower than `depth_conf_quantile`) are removed
    from visualization.
    """
    max_num_points: int = 100000
    """The maximum number of generated points."""
    val_split_ratio: float = 0.2
    """Ratio of scenes used for validation"""
    seed: int = 0
    """
    Half of images are rgb, and another half are thermal. The split is random
    constructed using this random seed.
    """

    def _create_and_save_point_cloud(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        depths_conf: torch.Tensor,
        extrinsics_world2cam: torch.Tensor,
        intrinsics: torch.Tensor,
        output_path: Path,
    ) -> None:
        """
        Creates a colored point cloud. The input `images` have shape [S, 3, H, W] and
        are used to determine the colors of the points in the final point cloud. The
        input `depths` consists of depth maps with shape [S, H, W, 1], and `depths_conf`
        provides the corresponding confidence values with shape [S, H, W]. For each
        image at index *i* in `images`, `depths[i]` represents the depth map for
        `images[i]`. The camera parameters are provided through `extrinsics_world2cam`
        and `intrinsics`. The resulting .ply file is written to `output_path`.
        """

        if images.ndim != 4 or images.shape[1] != 3:
            raise RuntimeError(
                f"The `images` must be [S, 3, H, W], but get {images.shape}"
            )
        if not (
            images.shape[0]
            == depths.shape[0]
            == depths_conf.shape[0]
            == extrinsics_world2cam.shape[0]
            == intrinsics.shape[0]
        ):
            raise RuntimeError(
                "The `images` `depths` `depths_conf` `extrinsics_world2cam`"
                + f"`intrinsics` must have equal lengths but get {images.shape}, "
                + f"{depths.shape}, {depths_conf.shape}, {extrinsics_world2cam.shape} "
                + f"{intrinsics.shape} respectively."
            )

        depth_conf_flat = depths_conf.flatten(start_dim=1)
        depth_conf_values = torch.quantile(
            depth_conf_flat, q=self.depth_conf_quantile, dim=1
        )
        depth_conf_values = depth_conf_values[:, None, None]
        point_masks = depths_conf >= depth_conf_values
        point_masks = randomly_limit_trues(
            point_masks.detach().cpu().numpy(), max_trues=self.max_num_points
        )

        # calculate world coordinate points
        world_coordinate_points = unproject_depth_map_to_point_map(
            depth_map=depths.detach().cpu().numpy(),
            extrinsics_cam=extrinsics_world2cam.detach().cpu().numpy(),
            intrinsics_cam=intrinsics.detach().cpu().numpy(),
        )

        points = world_coordinate_points[point_masks]
        colors = images.permute(0, 2, 3, 1).detach().cpu().numpy()[point_masks]
        if points.shape != colors.shape:
            raise RuntimeError(
                f"points shape {points.shape} != colors shape {colors.shape}."
            )

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(output_path), point_cloud)

    def _load_model(self) -> None:
        """
        Loads the original VGGT model specified in `self.original_ckpt_path` and turns
        it into eval mode.
        """
        self.model = VGGT()
        state_dict = torch.load(self.original_ckpt_path, map_location="cuda")
        self.model.load_state_dict(state_dict)
        self.model = self.model.cuda()
        self.model.eval()

    @torch.inference_mode()
    def _inference_vggt(
        self, images: torch.Tensor, thermal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the original VGGT model on `images` and uses `thermal_mask` to make the
        inheritance appropriate.

        :returns A tuple containing
                - camera pose encoding with shape [B, S, 9],
                - predicted depth maps with shape [B, S, H, W, 1],
                - confidence scores for depth predictions with shape [B, S, H, W],
                - original input images, preserved for visualization.
        """
        predictions = self.model.forward(images=images.cuda())

        pose_enc_list = predictions["pose_enc_list"]
        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]
        images_from_predictions = predictions["images"]

        return (pose_enc_list, depth, depth_conf, images_from_predictions)

    def create_point_clouds(
        self,
        output_folder: Path,
    ) -> None:
        """
        Creates rgb and thermal point clouds for train and eval scenes. Loads the
        original vggt model from `self.original_ckpt_path`. The images are taken from
        `self.scenes_root`. The point clouds are stored in `output_folder`/mode with
        mode is either eval or train. The `self.depth_eps` is a depth value such that
        pixels with smaller depth are considered to be invalid. Depth values with small
        confidence (lower than `self.depth_conf_quantile`) are considered to be invalid.
        The output .ply files are stored at `self.output_path`, with the
        `self.max_num_points` as the maximum number of generated points. All scenes are
        processed if `self.run_mode` is ALL, only eval scenes are processed if
        `self.run_mode` is EVAL, and only train scenes are processed if `self.run_mode`
        is TRAIN. Half of images are rgb, and another half is thermal. The split is
        randomly constructed using random seed `self.seed`.
        """

        self._load_model()

        output_folder.mkdir(exist_ok=True, parents=True)

        train_dataset, eval_dataset = (
            VGGTMultipleDataset.build_train_eval_datasets_undivided(
                scenes_root_path=self.scenes_root_path,
                val_split_ratio=self.val_split_ratio,
                depth_eps=self.depth_eps,
                seed=self.seed,
            )
        )

        datasets = {
            "train": train_dataset,
            "eval": eval_dataset,
        }
        if not set(self.run_mode.value).issubset(set(datasets.keys())):
            raise RuntimeError(
                "The `run_mode.value` must be a subset of {{'train', 'eval'}}, but "
                + f"get {self.run_mode.value}"
            )

        for run_mode in self.run_mode.value:
            dataset = datasets[run_mode]

            scenes_names = [scene_path.name for scene_path in dataset.scenes_paths]
            output_dir_mode = output_folder / run_mode
            output_dir_mode.mkdir(exist_ok=True, parents=True)
            for batch_idx in range(len(dataset)):
                scene_name = scenes_names[batch_idx]

                batch = dataset[batch_idx]

                images = batch.images
                thermal_mask = batch.mask_thermal

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pose_enc_list, depth, depth_conf, _ = self._inference_vggt(
                        images=images.cuda(), thermal_mask=thermal_mask.cuda()
                    )

                    extrinsics_world2cam, intrinsics = pose_encoding_to_extri_intri(
                        pose_enc_list[-1], images.shape[-2:]
                    )

                self._create_and_save_point_cloud(
                    images=images[~thermal_mask].to(torch.float32),
                    depths=depth[~thermal_mask].to(torch.float32),
                    depths_conf=depth_conf[~thermal_mask].to(torch.float32),
                    extrinsics_world2cam=extrinsics_world2cam[~thermal_mask].to(
                        torch.float32
                    ),
                    intrinsics=intrinsics[~thermal_mask].to(torch.float32),
                    output_path=output_dir_mode / f"rgb_{scene_name}.ply",
                )

                self._create_and_save_point_cloud(
                    images=images[thermal_mask].to(torch.float32),
                    depths=depth[thermal_mask].to(torch.float32),
                    depths_conf=depth_conf[thermal_mask].to(torch.float32),
                    extrinsics_world2cam=extrinsics_world2cam[thermal_mask].to(
                        torch.float32
                    ),
                    intrinsics=intrinsics[thermal_mask].to(torch.float32),
                    output_path=output_dir_mode / f"thermal_{scene_name}.ply",
                )

                self._create_and_save_point_cloud(
                    images=images[0].to(torch.float32),
                    depths=depth[0].to(torch.float32),
                    depths_conf=depth_conf[0].to(torch.float32),
                    extrinsics_world2cam=extrinsics_world2cam[0].to(torch.float32),
                    intrinsics=intrinsics[0].to(torch.float32),
                    output_path=output_dir_mode / f"thermal_and_rgb_{scene_name}.ply",
                )

                del (
                    images,
                    depth,
                    depth_conf,
                    pose_enc_list,
                    extrinsics_world2cam,
                    intrinsics,
                )
                gc.collect()
                torch.cuda.empty_cache()
