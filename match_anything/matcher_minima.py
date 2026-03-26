from pathlib import Path

import numpy as np
import torch
from deep_image_matching.matchers.roma import RomaMatcher
from PIL import Image


class MINIMARoMAMatcher(RomaMatcher):
    """
    MINIMARomaMatcher class for feature matching using MINIMA-RoMa model.
    """

    def __init__(self, checkpoint_path: Path, config={}) -> None:
        """
        Initializes the `MINIMARoMAMatcher` using the parameters from config.
        """
        super().__init__(config)

        state_dict = torch.load(checkpoint_path, map_location=self._device)
        self.matcher.load_state_dict(state_dict=state_dict)

    @torch.no_grad()
    def _match_pairs(
        self,
        feature_path: Path,
        img0_path: Path,
        img1_path: Path,
    ):
        """
        Perform matching by tile. The `feature_path` is a path to the feature file. The
        `img0` is a path to the first image. The `img1` is a path to the second image.
        The `method` is a Tile selection method. The `select_unique` is a Flag to select
        unique features.

        :return: array containing the indices of matched keypoints.

        NOTE: The function is taken from
        deep_image_matching/matchers/roma.py:_match_by_tile with little changes.
        """

        img0_name = img0_path.name
        img1_name = img1_path.name

        # Run inference
        W_A, H_A = Image.open(img0_path).size
        W_B, H_B = Image.open(img1_path).size

        warp, certainty = self.matcher.match(
            str(img0_path), str(img1_path), device=self._device, batched=False
        )
        matches, certainty = self.matcher.sample(
            warp, certainty, num=self.config["matcher"]["num_sampled_points"]
        )
        kptsA, kptsB = self.matcher.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        kptsA, kptsB = kptsA.cpu().numpy(), kptsB.cpu().numpy()

        # Create a 1-to-1 matching array
        matches0 = np.arange(kptsA.shape[0])
        matches = np.hstack((matches0.reshape((-1, 1)), matches0.reshape((-1, 1))))
        self._update_features_h5(
            feature_path,
            img0_name,
            img1_name,
            kptsA,
            kptsB,
            matches,
        )

        return matches
