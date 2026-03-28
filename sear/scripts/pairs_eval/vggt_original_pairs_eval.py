from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import tyro
from vggt.models.vggt import VGGT
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from sear.data_processing.chunk import Chunk
from sear.data_processing.paired_item import PairedItem
from sear.scripts.pairs_eval.base import PairsEvalParametersBase, main


@dataclass
class VGGTEvalPairsBase(PairsEvalParametersBase):
    """
    A class to evaluate relative camera pose reconstruction between two images using the
    VGGT-like (can be extended to feed-forward) models.
    """

    method_name: str = "VGGT-Original"
    """Method name to store the results"""

    original_vggt_path: Path = Path("vggt-model")
    """Checkpoint path of the original vggt model"""
    depth_eps: float = 1e-8
    """Depth value smaller this value do not take part in training"""

    @abstractmethod
    def _inference_vggt(
        self,
        chunk: Chunk | PairedItem,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts extrinsics and intrinsics camera poses of images from `chunk`.
        """
        raise NotImplementedError("The `inference_vggt` is an abstractmethod")

    @staticmethod
    def to_4x4(camera_poses: torch.Tensor) -> torch.Tensor:
        """
        Converts cameras of shape (..., 3, 4) to (..., 4, 4)
        """
        if camera_poses.shape[-2:] == (3, 4):
            new_row = torch.zeros_like(camera_poses[..., 0:1, :])
            new_row[..., -1] = 1.0
            camera_poses = torch.cat([camera_poses, new_row], dim=-2)
        return camera_poses

    def run_pairs(
        self,
        chunk: PairedItem,
        cache_folder: Path,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Runs VGGT model on pairs generated from `chunk` and saves intermetiate results
        in the `cache_folder`.

        :return: relative ground truth camera poses, and relative predicted camera
            poses.
        """

        cache_folder.mkdir(exist_ok=True)
        relative_transforms_pred_list: list[npt.NDArray[np.float64]] = []
        relative_transforms_real_list: list[npt.NDArray[np.float64]] = []
        generator = chunk.iterate_batched(batch_size=self.batch_size)

        for chunk_batched in generator:
            # [B, 2, 3, 3]
            extrinsics_pred_world2cam, _ = self._inference_vggt(chunk=chunk_batched)
            extrinsics_pred_world2cam = self.to_4x4(extrinsics_pred_world2cam)
            relative_extrinsics_pred = torch.matmul(
                extrinsics_pred_world2cam[:, 1, :, :],
                closed_form_inverse_se3(extrinsics_pred_world2cam[:, 0, :, :]),
            )
            relative_transforms_pred_list.append(
                relative_extrinsics_pred.detach().cpu().numpy()
            )

            extrinsics_real_world2cam = chunk_batched.extrinsics_world2cam
            extrinsics_real_world2cam = self.to_4x4(extrinsics_real_world2cam)
            relative_extrinsics_real = torch.matmul(
                extrinsics_real_world2cam[:, 1, :, :],
                closed_form_inverse_se3(extrinsics_real_world2cam[:, 0, :, :]),
            )
            relative_transforms_real_list.append(
                relative_extrinsics_real.detach().cpu().numpy()
            )

        return (
            np.concatenate(relative_transforms_real_list),
            np.concatenate(relative_transforms_pred_list),
        )


@dataclass
class VGGTOriginalEvalPairs(VGGTEvalPairsBase):
    """
    A class to evaluate relative camera pose reconstruction between two images using the
    original VGGT model https://arxiv.org/abs/2503.11651
    """

    def load_model(self) -> None:
        """Loads the original VGGT model into memory"""
        self._vggt_model = VGGT()
        state_dict = torch.load(self.original_vggt_path, map_location="cuda")
        self._vggt_model.load_state_dict(state_dict)
        self._vggt_model = self._vggt_model.to(self._device)
        self._vggt_model.eval()

    def _inference_vggt(
        self, chunk: Chunk | PairedItem
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicts extrinsics and intrinsics camera poses of images from `chunk`"""
        chunk = chunk.to_device(self._device)
        with torch.inference_mode():
            with torch.amp.autocast(str(self._device), dtype=torch.bfloat16):
                model_prediction = self._vggt_model.forward(images=chunk.images)
                pose_enc_list = model_prediction["pose_enc_list"]

        extrinsics_pred_world2cam, intrinsics = pose_encoding_to_extri_intri(
            pose_enc_list[-1].to(torch.float32), chunk.images.shape[-2:]
        )
        return extrinsics_pred_world2cam.detach().cpu(), intrinsics.detach().cpu()


if __name__ == "__main__":
    params = tyro.cli(VGGTOriginalEvalPairs)
    main(params=params)
