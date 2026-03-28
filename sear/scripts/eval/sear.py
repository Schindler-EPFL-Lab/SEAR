"""
Ablation VGGT Evaluation Script

This script evaluates the original VGGT model with ablation configurations for
thermal-RGB camera pose estimation. It processes chunks of data, predicts camera
parameters (extrinsics and intrinsics), and evaluates the performance.

Usage:
    python ablation_vggt.py --scene-path <path_to_scene> \
        --output-folder <output_directory> \
        --ckpt-path <checkpoint_path> \
        [--aggregator.type <aggregator_type>] \
        [--aggregator.thermal-token-processor <processor_type>] \
        [--thermal-vggt.vggt-path <vggt_model_path>] \
        [--optimization.* <optimization_params>]

Key Components:
    - VGGTAblationEvalParameters: Configuration for evaluation parameters
    - VGGTAblationChunkProcessor: Processes data chunks using the ablation VGGT model
    - ThermalVGGTLightning: The core VGGT model with thermal capabilities

The script supports different aggregator configurations and optimization parameters
to evaluate various ablation scenarios of the VGGT model.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from sear.data_processing.chunk import Chunk
from sear.data_processing.inference_scene import InferenceScene
from sear.models.aggregator_config import AggregatorConfig
from sear.models.vggt_wrapper import (
    OptimizationParameters,
    ThermalVGGTConfig,
    ThermalVGGTLightning,
)
from sear.scripts.eval.base import ChunkProcessorBase, EvalParametersBase, main


@dataclass(kw_only=True)
class VGGTAblationEvalParameters(EvalParametersBase):
    """A config to evaluate original VGGT"""

    method_name: str = "AblationVGGT"  # Should be deleted it is not used...
    """The method name used to mark saved results"""
    aggregator: AggregatorConfig
    """
    Specifies parameters of the aggregator, i.e. which layers are updated with LoRA, how
    to process thermal tokens and etc.
    """
    thermal_vggt: ThermalVGGTConfig
    """VGGT Model config"""
    optimization: OptimizationParameters
    """VGGT Model optimization config"""
    ckpt_path: Path = Path("checkpoint-path")
    """Checkpoint path from which the model should be initialized"""

    def __post_init__(self) -> None:
        self.method_name = f"AblationVGGT-{self.aggregator.type.value}"


class VGGTAblationChunkProcessor(ChunkProcessorBase):
    """
    Chunk processor for evaluating VGGT model with ablation configurations.

    This class implements the ChunkProcessorBase interface to evaluate the original
    VGGT model with various ablation configurations for thermal-RGB camera pose
    estimation. It supports different aggregator types and optimization parameters
    to test various aspects of the VGGT architecture.

    The processor loads a pre-trained VGGT model with thermal capabilities and
    applies it to chunks of images to estimate camera poses, intrinsics, and depth maps.
    """

    def __init__(self, config: VGGTAblationEvalParameters) -> None:
        """
        Initializes the VGGTAblationChunkProcessor with the given configuration.

        `config` is a configuration object that contains model paths, aggregator
        settings, optimization parameters, and other evaluation parameters.

        Sets up the model instance and prepares for inference.
        """
        super().__init__()
        self._config = config
        self._model: ThermalVGGTLightning | None = None

    def load_model(self) -> None:
        """
        Loads the VGGT model with thermal capabilities.

        Constructs the thermal aggregator based on the configuration and loads
        the pre-trained VGGT model from the specified checkpoint. The model is
        configured with the provided thermal aggregator and optimization settings.

        The loaded model is moved to the appropriate device (CUDA/CPU) and set to
        evaluation mode.

        Note:
            This method uses the aggregator configuration to build a custom aggregator
            that determines how thermal and RGB features are processed and combined.
        """
        thermal_aggregator = self._config.aggregator.build_aggregator_from_vggt_path(
            vggt_path=self._config.thermal_vggt.vggt_path,
        )

        self._model = ThermalVGGTLightning.load_from_checkpoint(
            checkpoint_path=self._config.ckpt_path,
            thermal_aggregator=thermal_aggregator,
            config=self._config.thermal_vggt,
            optimization_config=self._config.optimization,
            strict=False,
        )
        self._model = self._model.to(self._device)
        self._model.eval()

    def process_chunk(
        self, chunk: Chunk | InferenceScene, cache_folder: Path
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """
        Processes a `chunk` of images using the VGGT ablation model.

        Applies the loaded VGGT model to estimate camera poses, intrinsics, and
        depth maps for the images in the chunk. The processing uses mixed precision
        inference for efficiency.

        `chunk` contains images, camera parameters, and metadata. It is expected to have
         both RGB and thermal images as specified by the thermal mask.
        On the other hand, `cache_folder` is the path for storing intermediate results.

        :return: A tuple containing:
        - extrinsics_pred_world2cam: Estimated camera poses in OpenCV world-to-camera
            format, shape (N, 3, 4)
        - intrinsics: Estimated camera intrinsic matrices, shape (N, 3, 3)
        - images: Processed images in RGB format, shape (N, H, W, 3), normalized
        - depths: Estimated depth maps, shape (N, H, W)
        - mask_found: Boolean mask indicating all images were processed successfully
            (always True for this implementation)

        Note:
        The implementation assumes the model can process all images successfully
        and returns a mask with all True values. The method uses bfloat16 precision
        for inference to balance accuracy and performance.

        The returned tensors are moved to CPU and detached from the computation graph
        to save memory.
        """
        chunk = chunk.to_device(self._device)
        with torch.inference_mode():
            with torch.amp.autocast(str(self._device), dtype=torch.bfloat16):
                pose_enc_list, pred_depth, _, _ = self._model.forward(
                    images=chunk.images, thermal_mask=chunk.mask_thermal
                )

        extrinsics_pred_world2cam, intrinsics = pose_encoding_to_extri_intri(
            pose_enc_list[-1].to(torch.float32), chunk.images.shape[-2:]
        )
        assert intrinsics is not None
        chunk = chunk.to_device(torch.device("cpu"))

        return (
            extrinsics_pred_world2cam[0].detach().cpu(),
            intrinsics[0].detach().cpu(),
            chunk.images[0].detach().cpu().permute(0, 2, 3, 1),
            pred_depth[0, :, :, :, 0].detach().cpu(),
            torch.ones((extrinsics_pred_world2cam.shape[1],), dtype=torch.bool),
        )


if __name__ == "__main__":
    params = tyro.cli(VGGTAblationEvalParameters)
    chunk_processor = VGGTAblationChunkProcessor(params)
    main(params, chunk_processor=chunk_processor)
