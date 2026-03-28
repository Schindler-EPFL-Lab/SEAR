import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torchvision
import tyro
from romatch import roma_outdoor
from transformers import EfficientLoFTRImageProcessorFast
from transformers.image_utils import load_image

from sear import logger
from sear.scripts.pairs_eval.base import PairsEvalKeypointsParametersBase, main


@dataclass
class RomaEvalPairs(PairsEvalKeypointsParametersBase):
    """
    A class to evaluate relative camera pose reconstruction between two images using the
    RoMA model https://arxiv.org/abs/2305.15404
    """

    method_name: str = "roma"
    """Method name to store the results"""

    visualize_max_num: int = 100
    """The maximum number of matches to visualize"""
    num_samples: int = 10000
    """Number of samples sampled matches."""

    def load_model(self) -> None:
        """Loads the RoMA model into memory"""
        self._roma_model = roma_outdoor(device=self._device, use_custom_corr=False)
        # need it purely for visuzalization
        self._processor = EfficientLoFTRImageProcessorFast.from_pretrained(
            "zju-community/matchanything_eloftr"
        )

    def _find_keypoints(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        visualize_keypoints_path: Path | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Find keypoints between images located at `image1_path` and `image2_path` using
        RoMA model. Optionally saves the visualization of the found keypoints if the
        `visualize_keypoints_path` is provided.

        :return: found matches coordinates between the first and the second image.
        """

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir_path = Path(temp_dir_str)
            image1_path = temp_dir_path / "image_1.png"
            torchvision.utils.save_image(image1, image1_path)
            image2_path = temp_dir_path / "image_2.png"
            torchvision.utils.save_image(image2, image2_path)

            warp, certainty = self._roma_model.match(
                image1_path, image2_path, device=self._device
            )

            image1_loaded = load_image(str(image1_path))
            image2_loaded = load_image(str(image2_path))
            images_loaded = [image1_loaded, image2_loaded]

        H_A, W_A = image1.shape[-2:]
        H_B, W_B = image2.shape[-2:]

        matches, certainty = self._roma_model.sample(
            matches=warp, certainty=certainty, num=self.num_samples
        )
        kpts1, kpts2 = self._roma_model.to_pixel_coordinates(
            coords=matches, H_A=H_A, W_A=W_A, H_B=H_B, W_B=W_B
        )

        if visualize_keypoints_path is not None:
            visualize_indices = torch.randperm(certainty.shape[0])[
                : self.visualize_max_num
            ]

            outputs_2 = [
                {
                    "keypoints0": kpts1[visualize_indices],
                    "keypoints1": kpts2[visualize_indices],
                    "matching_scores": certainty[visualize_indices],
                }
            ]

            try:
                self._processor.visualize_keypoint_matching(images_loaded, outputs_2)[
                    0
                ].save(visualize_keypoints_path)
            except Exception as e:
                logger.info(f"Could not visualize, {repr(e)}")

        return kpts1.detach().cpu().numpy(), kpts2.detach().cpu().numpy()


if __name__ == "__main__":
    params = tyro.cli(RomaEvalPairs)
    main(params=params)
