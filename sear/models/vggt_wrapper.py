from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
import matplotlib.pyplot as plt
import torch
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from training.loss import MultitaskLoss
from transformers import get_scheduler
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from sear.data_processing.multiple_dataset import Chunk, VGGTMultipleDataset
from sear.metrics.calculator import MetricsCalculator
from sear.models.thermal_vggt import ThermalVGGT
from sear.visualization.cameras import (
    draw_camera_extrinsics,
    draw_camera_trajectories,
)


@dataclass
class ThermalVGGTConfig(ReverseCli):
    """Configuration class for VGGT LoRA model"""

    vggt_path: Path = Path("vggt-model")
    """Checkpoint path of the original vggt model"""
    img_size: int = 518
    """Largest dimention of input images"""
    patch_size: int = 14
    """Patch size used in DINO"""
    embed_dim: int = 1024
    """Inner embedding dimention used in computation"""
    lora_alpha: int = 128
    """
    LoRA alpha parameter, which determines the scale (alpha / rank) of the additional
    branch. One is advised to look at the Sec 4.1 of the paper
    https://arxiv.org/pdf/2106.09685
    """
    lora_rank: int = 64
    """LoRA rank parameter, the bigger the rank the larget the LoRA adapter is"""
    lora_dropout: float = 0.1
    """LoRA dropout parameter"""
    loss_camera_weight: float = 5.0
    """coefficient for weighting the camera loss"""
    loss_depth_weight: float = 1.0
    """coefficient for weighting the depth loss"""


@dataclass
class OptimizationParameters(ReverseCli):
    learning_rate: float = 1e-4
    """Learning rate used during the optimization"""
    weight_decay: float = 1e-2
    """Weight decay parameter for AdamW"""
    scheduler_name: str = "constant_with_warmup"
    """
    The scheduler name, please refer to
    https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.SchedulerType
    """
    warmup_ratio: float = 0.1
    """warmup ratio used for learning rate scheduling"""


class ThermalVGGTLightning(L.LightningModule):
    def __init__(
        self, config: ThermalVGGTConfig, optimization_config: OptimizationParameters
    ) -> None:
        """
        Initializes the VGGT wrapper model. The `config` determines the parameters
        required to initialize the inner VGGT model, while the `optimization_config`
        determines necessary parameters for finetuning.
        """
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._config = config
        self._optimization_config = optimization_config

        # load vggt
        self._vggt = ThermalVGGT.from_checkpoint_path(
            vggt_ckpt_path=self._config.vggt_path,
            img_size=self._config.img_size,
            patch_size=self._config.patch_size,
            embed_dim=self._config.embed_dim,
        )

        # initialize loss
        self._loss_fn = MultitaskLoss(
            camera={"weight": self._config.loss_camera_weight, "loss_type": "l1"},
            depth={
                "weight": self._config.loss_depth_weight,
                "gradient_loss_fn": "grad",
                "valid_range": 0.98,
            },
            point=None,
            track=None,
        )

        self._calculator = MetricsCalculator(
            thresholds=[5.0, 15.0, 30.0],
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Removes parameters which are not necessary to save."""
        # The pytorch issue is that it uses `.named_modules` rather than recursive
        # `.state_dict` call. Therefore, even if I write my logic state_dict for inner
        # modules it is not applied when state_dict of the current model is called.

        state: dict[str, Any] = checkpoint["state_dict"]
        for name in state:
            if name not in self._vggt.aggregator.KEEP_IN_STATE_DICT:
                state.pop(name)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Loads state dict for VGGT LoRA and Thermal Projector"""
        state: dict[str, Any] = checkpoint["state_dict"]
        for name in state:
            if name not in self._vggt.aggregator.KEEP_IN_STATE_DICT:
                state.pop(name)
        self._vggt.load_state_dict(state)

    def forward(
        self, images: torch.Tensor, thermal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the wrapped VGGT model. It processed `images` where
        `thermal_mask` specifies the position of thermal images. Let (H, W) is the image
        size, B -batch size, S - sequence length, then `images` has shape [B, S, 3, H,
        W] and `thermal_mask` has shape [B, S].

        :returns A tuple containing
                - camera pose encoding with shape [B, S, 9],
                - predicted depth maps with shape [B, S, H, W, 1],
                - confidence scores for depth predictions with shape [B, S, H, W],
                - original input images, preserved for visualization.
        """
        predictions = self._vggt(images=images, thermal_mask=thermal_mask)
        return (
            predictions["pose_enc_list"],
            predictions["depth"],
            predictions["depth_conf"],
            predictions["images"],
        )

    def on_training_step_start(self) -> None:
        """
        Sets thermal ratio of the dataset by uniformly sampling it from 0.0 to 1.0.
        """
        train_dataset: VGGTMultipleDataset = self.trainer.train_dataloader.dataset
        train_dataset.thermal_ratio = torch.rand((1,)).item()
        self.log("thermal_ratio", train_dataset.thermal_ratio)

    def training_step(
        self,
        batch: Chunk,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Performs the prediction for training `batch` using the model and calculates the
        loss. The `batch` is a tuple of elements: (images, depths, point_masks,
        extrinsics_world2cam, intrinsics, thermal_mask)
        """

        self.on_training_step_start()

        batch.to_device(self.device)
        images = batch.images
        depths = batch.depths
        point_masks = batch.point_masks
        extrinsics_world2cam = batch.extrinsics_world2cam
        intrinsics = batch.intrinsics
        thermal_mask = batch.mask_thermal

        (pose_enc_list, pred_depth, pred_depth_conf, _) = self.forward(
            images=images, thermal_mask=thermal_mask
        )
        loss = self._loss_fn(
            predictions={
                "pose_enc_list": pose_enc_list,
                "depth": pred_depth,
                "depth_conf": pred_depth_conf,
            },
            batch={
                "images": images,
                "depths": depths,
                "point_masks": point_masks,
                "extrinsics": extrinsics_world2cam,
                "intrinsics": intrinsics,
            },
        )

        for key in loss:
            self.log(f"train_{key}", loss[key])
        return loss["objective"]

    def _log_images(
        self,
        title: str,
        extrinsics_real_world2cam: torch.Tensor,
        extrinsics_pred_world2cam: torch.Tensor,
    ) -> None:
        """
        Creates and saves images of camera extrinsics and trajectories for both
        `extrinsics_real_world2cam` and `extrinsics_pred_world2cam`.
        """
        images_dir = Path(self.logger.log_dir, "images")
        images_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        image_extrinsics_name = (
            f"extrinsics-{self.global_step:05}-{title}-{timestamp}.png"
        )
        fig_extrinsics = draw_camera_extrinsics(
            extrinsics_real_world2cam=extrinsics_real_world2cam,
            extrinsics_pred_world2cam=extrinsics_pred_world2cam,
            title=title,
        )
        fig_extrinsics.savefig(images_dir / image_extrinsics_name)
        fig_extrinsics.clf()
        plt.close()

        image_trajectories_name = (
            f"trajectory-{self.global_step:05}-{title}-{timestamp}.png"
        )
        fig_trajectories = draw_camera_trajectories(
            extrinsics_real_world2cam=extrinsics_real_world2cam,
            extrinsics_pred_world2cam=extrinsics_pred_world2cam,
            title=title,
        )
        fig_trajectories.savefig(images_dir / image_trajectories_name)
        fig_trajectories.clf()
        plt.close()

    def validation_step(
        self,
        batch: Chunk,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Performs the prediction for validation `batch` using the model and calculates
        the loss. The `batch` is a tuple of elements: (images, depths, point_masks,
        extrinsics_world2cam, intrinsics, thermal_mask)
        """

        batch.to_device(self.device)
        images = batch.images
        depths = batch.depths
        point_masks = batch.point_masks
        extrinsics_world2cam = batch.extrinsics_world2cam
        intrinsics = batch.intrinsics
        thermal_mask = batch.mask_thermal
        datasets_names = batch.datasets_names
        scenes_names = batch.scenes_names

        (pose_enc_list, pred_depth, pred_depth_conf, _) = self.forward(
            images=images, thermal_mask=thermal_mask
        )
        loss = self._loss_fn(
            predictions={
                "pose_enc_list": pose_enc_list,
                "depth": pred_depth,
                "depth_conf": pred_depth_conf,
            },
            batch={
                "images": images,
                "depths": depths,
                "point_masks": point_masks,
                "extrinsics": extrinsics_world2cam,
                "intrinsics": intrinsics,
            },
        )

        for key in loss:
            self.log(f"eval_{key}", loss[key])

        extrinsics_pred_world2cam, pred_intrinsics = pose_encoding_to_extri_intri(
            pose_enc_list[-1], images.shape[-2:]
        )
        for ext_idx in range(extrinsics_world2cam.shape[0]):
            self._log_images(
                title=f"{datasets_names[ext_idx]}:{scenes_names[ext_idx]}",
                extrinsics_real_world2cam=extrinsics_world2cam[ext_idx].detach().cpu(),
                extrinsics_pred_world2cam=extrinsics_pred_world2cam[ext_idx]
                .detach()
                .cpu(),
            )

            self._calculator.add_data(
                cameras_real_world2cam=extrinsics_world2cam[ext_idx].cpu().numpy(),
                depths_real=depths[ext_idx].cpu().numpy(),
                intrinsics_real=intrinsics[ext_idx].cpu().numpy(),
                cameras_pred_world2cam=extrinsics_pred_world2cam[ext_idx]
                .detach()
                .cpu()
                .numpy(),
                depths_pred=pred_depth[ext_idx, :, :, :, 0].detach().cpu().numpy(),
                intrinsics_pred=pred_intrinsics[ext_idx].detach().cpu().numpy(),
                scene_name=scenes_names[ext_idx],
                ratio_reconstructed=1.0,
                duration=0.0,
                dataset_name=datasets_names[ext_idx],
            )

        return loss["objective"]

    def on_validation_epoch_start(self) -> None:
        """Does necessary preparation steps to perform validation."""
        self._calculator.clear()

    def on_validation_epoch_end(self) -> None:
        """
        Calculates and logs aggregated metrics for pose estimation and point cloud
        estimation.
        """
        aggregated = self._calculator.aggregated()
        for i, threshold_value in enumerate(self._calculator._thresholds):
            self.log(
                f"RRA@{threshold_value}",
                aggregated.rra[i],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"RTA@{threshold_value}",
                aggregated.rta[i],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"mAA@{threshold_value}",
                aggregated.maa[i],
                on_step=False,
                on_epoch=True,
            )

        self.log("RPE_rotation", aggregated.rpe_rotation, on_step=False, on_epoch=True)
        self.log(
            "RPE_translation_distance",
            aggregated.rpe_translation_distance,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "RPE_translation_degree",
            aggregated.rpe_translation_degree,
            on_step=False,
            on_epoch=True,
        )
        self.log("ATE_rotation", aggregated.ate_rotation, on_step=False, on_epoch=True)
        self.log(
            "ATE_translation_distance",
            aggregated.ate_translation_distance,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "ATE_translation_degree",
            aggregated.ate_translation_degree,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "PC_Accuracy", aggregated.point_cloud_accuracy, on_step=False, on_epoch=True
        )

    def configure_optimizers(
        self,
    ) -> dict[
        str,
        torch.optim.Optimizer | dict[str, torch.optim.lr_scheduler.LRScheduler | str],
    ]:
        """
        Configures the optimizer and scheduler used in optimization.
        """

        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(
            self._optimization_config.warmup_ratio * num_training_steps
        )

        optimizer = torch.optim.AdamW(
            params=(p for p in self.parameters() if p.requires_grad),
            lr=self._optimization_config.learning_rate,
            weight_decay=self._optimization_config.weight_decay,
        )

        scheduler = get_scheduler(
            name=self._optimization_config.scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
