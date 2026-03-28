import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torchvision
import tyro
from transformers import (
    EfficientLoFTRForKeypointMatching,
    EfficientLoFTRImageProcessorFast,
)
from transformers.image_utils import load_image

from sear.scripts.eval.relative_camera_pose_estimation.base import (
    PairsEvalKeypointsParametersBase,
    main,
)


@dataclass
class MatchAnythingEvalPairs(PairsEvalKeypointsParametersBase):
    """
    A class to evaluate relative camera pose reconstruction between two images using the
    MatchAnything ELoFTR model (the only MatchAnything model based on RoMA is not
    available for 15.01.2026) https://arxiv.org/abs/2501.07556
    """

    method_name: str = "match-anything"
    """Method name to store the results"""

    processor_size: tuple[int, int] = (640, 480)
    """
    The input image must be preprocessed to a specific resolution of `processor_size`
    before passing it to the model.
    """
    matching_threshold: float = 0.1
    """
    If the matching score is above this value then the match is considered to be
    correct.
    """

    def load_model(self) -> None:
        """Loads the MatchAnything ELoFTR model into memory"""
        self._processor = EfficientLoFTRImageProcessorFast.from_pretrained(
            "zju-community/matchanything_eloftr"
        )
        self._model = EfficientLoFTRForKeypointMatching.from_pretrained(
            "zju-community/matchanything_eloftr"
        )
        self._model = self._model.to(self._device)
        self._model.eval()

    def _find_keypoints(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        visualize_keypoints_path: Path | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Finds keypoints between images `image1` and `image2` using the MatchAnything
        ELoFTR model. Optionally saves the visualization of the found keypoints if the
        `visualize_keypoints_path` is provided.

        :return: found matches coordinates between the first and the second image.
        """

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir_path = Path(temp_dir_str)
            image1_path = temp_dir_path / "image_1.png"
            torchvision.utils.save_image(image1, image1_path)
            image2_path = temp_dir_path / "image_2.png"
            torchvision.utils.save_image(image2, image2_path)

            image1_loaded = load_image(str(image1_path))
            image2_loaded = load_image(str(image2_path))
            images = [image1_loaded, image2_loaded]

        inputs = self._processor.preprocess(
            images,
            return_tensors="pt",
            size=(self.processor_size[1], self.processor_size[0]),
            do_grayscale=False,
        )
        inputs = inputs.to(self._device)

        with torch.inference_mode():
            model_outputs = self._model(**inputs)

        image_shape = image1.shape[-2:]
        image_sizes = [[image_shape] * 2]
        outputs_2 = self._processor.post_process_keypoint_matching(
            model_outputs, image_sizes, threshold=self.matching_threshold
        )

        if visualize_keypoints_path is not None:
            self._processor.visualize_keypoint_matching(images, outputs_2)[0].save(
                visualize_keypoints_path
            )

        return (
            outputs_2[0]["keypoints0"].detach().cpu().numpy(),
            outputs_2[0]["keypoints1"].detach().cpu().numpy(),
        )


if __name__ == "__main__":
    params = tyro.cli(MatchAnythingEvalPairs)
    main(params=params)
