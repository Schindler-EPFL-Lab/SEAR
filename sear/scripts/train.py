import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightning
import mlflow
import torch
import tyro
from dataclasses_reverse_cli.reverse_cli import ReverseCli
from lightning import Trainer
from lightning.pytorch.callbacks import DeviceStatsMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from sear.augment.geometric import GeometricTransformConfig
from sear.augment.rgb import RGBTransformFactory
from sear.augment.thermal import ThermalTransformFactory
from sear.data_processing.multiple_dataset import VGGTMultipleDataset
from sear.models.vggt_wrapper import (
    OptimizationParameters,
    ThermalVGGTConfig,
    ThermalVGGTLightning,
)


@dataclass
class VGGTTrainingParameters(ReverseCli):
    thermal_vggt: ThermalVGGTConfig
    """VGGT Model config"""
    optimization: OptimizationParameters
    """VGGT Model optimization config"""
    thermal_transform_factory: ThermalTransformFactory = field(
        default_factory=lambda: ThermalTransformFactory()
    )
    """Configuration for thermal transform factory"""
    rgb_transform_factory: RGBTransformFactory = field(
        default_factory=lambda: RGBTransformFactory()
    )
    """Configuration for rgb transform factory"""
    geometric_transform_config: GeometricTransformConfig = field(
        default_factory=lambda: GeometricTransformConfig()
    )
    """Configuration for geometric transform config"""
    scenes_root_path: Path = Path("input")
    """Directory containing processed VGGT scenes"""
    thermal_vggt_ckpt_folder: Path = Path("input")
    """Directory to store Thermal VGGT Finetuned models"""
    output_dir: Path = Path("./outputs")
    """Directory to save COLMAP files"""
    depth_eps: float = 1e-8
    """Depth value smaller this value are considered to be correct"""
    elements_number: int = 24
    """The number of frames in one dataset element"""
    batch_size: int = 1
    """Batch size used during optimization"""
    accumulation_steps: int = 1
    """Number of gradient accumulation steps used during optimization"""
    seed: int = 0
    """Random seed for the dataset"""
    epochs: int = 100
    """Number of epochs for training"""
    gradient_clip_value: float | None = None
    """gradient clipping value during training. `None` if no clipping"""
    val_split_ratio: float = 0.2
    """Ratio of scenes used for validation"""


def train_thermal_vggt(params: VGGTTrainingParameters) -> None:
    params.thermal_vggt_ckpt_folder.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = params.thermal_vggt_ckpt_folder / "last.ckpt"
    latest_ckpt_path = latest_ckpt_path if latest_ckpt_path.exists() else None

    try:
        mlflow.pytorch.autolog(log_models=False, log_every_n_step=1, log_datasets=False)
        mlflow.log_params(
            params.to_loggable_dict(
                ignore=set(
                    [
                        "scenes_root_path",
                        "thermal_vggt_ckpt_folder",
                        "thermal_vggt.vggt_path",
                    ]
                )
            )
        )
    except mlflow.exceptions.MlflowException as e:
        logging.warning(f"autolog does not work: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lightning.seed_everything(params.seed, workers=True)

    train_dataset, eval_dataset = (
        VGGTMultipleDataset.build_train_eval_datasets_fixed_split(
            scenes_root_path=params.scenes_root_path,
            elements_number=params.elements_number,
            depth_eps=params.depth_eps,
            seed=params.seed,
            rgb_transform_factory=params.rgb_transform_factory,
            thermal_transform_factory=params.thermal_transform_factory,
            geometric_transform_config=params.geometric_transform_config,
        )
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=VGGTMultipleDataset.collate_chunks,
        num_workers=23,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=VGGTMultipleDataset.collate_chunks,
        num_workers=23,
    )

    model = ThermalVGGTLightning(
        config=params.thermal_vggt, optimization_config=params.optimization
    )
    model = model.to(device)
    model.strict_loading = False
    logger = CSVLogger(save_dir=params.output_dir)

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=params.thermal_vggt_ckpt_folder,
        save_last=True,
        monitor="eval_objective",
        save_top_k=3,
        every_n_epochs=1,
        enable_version_counter=False,
    )

    device_stats = DeviceStatsMonitor(cpu_stats=None)

    trainer = Trainer(
        precision="bf16-mixed",
        max_epochs=params.epochs,
        accumulate_grad_batches=params.accumulation_steps,
        enable_progress_bar=False,
        gradient_clip_val=params.gradient_clip_value,
        default_root_dir=params.output_dir,
        callbacks=[model_checkpoint_callback, device_stats],
        log_every_n_steps=1,
        logger=logger,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=eval_loader,
        ckpt_path=latest_ckpt_path,
    )


if __name__ == "__main__":
    params = tyro.cli(VGGTTrainingParameters)
    train_thermal_vggt(params)
