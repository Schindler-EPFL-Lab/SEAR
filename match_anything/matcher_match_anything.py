import logging
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from deep_image_matching.constants import TileSelection, Timer
from deep_image_matching.io.h5 import get_features
from deep_image_matching.matchers.matcher_base import (
    DetectorFreeMatcherBase,
    tile_selection,
)
from deep_image_matching.utils.geometric_verification import geometric_verification
from deep_image_matching.utils.tiling import Tiler
from deep_image_matching.visualization import viz_matches_cv2
from tqdm import tqdm
from transformers import (
    EfficientLoFTRForKeypointMatching,
    EfficientLoFTRImageProcessorFast,
)
from transformers.image_utils import load_image

from match_anything import logger


class MatchAnythingMatcher(DetectorFreeMatcherBase):
    """MatchAnythingMatcher class for feature matching using MatchAnything"""

    _default_conf = {}

    grayscale = False
    """Whether to convert images to grayscale"""
    as_float = True
    """Convert images to float32 when reading"""
    max_tile_pairs = 250
    """
    Maximum number of tile pairs to match, raise an error if more than this number to
    avoid slow and likely inaccurate matching
    """
    min_matches_per_tile = 3
    """The minimal number of matcher between tiles to consider the match correct"""
    keep_tiles = True
    """Whether to keep tiles after use"""

    def __init__(self, config={}) -> None:
        """Initializes the MatchAnythingMatcher class"""
        super().__init__(config)
        self._processor = EfficientLoFTRImageProcessorFast.from_pretrained(
            "zju-community/matchanything_eloftr"
        )
        self.matcher = EfficientLoFTRForKeypointMatching.from_pretrained(
            "zju-community/matchanything_eloftr"
        )
        self.matcher = self.matcher.to(self._device)
        self._preprocess_shape = self.config["matcher"]["preprocess_shape"]
        self._match_threshold = self.config["matcher"]["match_threshold"]

    @torch.inference_mode()
    def inference_match_anything(
        self,
        img0_path: str,
        img1_path: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inferences the MatchAnything model on two images located at `img0_path`,
        `img1_path` with calculations performed on `device`.

        :return: Detected matches coordinates of the first image, detected matches
            coordinates of the second image, matches confidences.
        """
        images = [load_image(img0_path), load_image(img1_path)]
        inputs = self._processor.preprocess(
            images,
            return_tensors="pt",
            size=(self._preprocess_shape[1], self._preprocess_shape[0]),
            do_grayscale=self.grayscale,
        )
        inputs = inputs.to(device)
        outputs = self.matcher(**inputs)

        image_sizes = [[(image.height, image.width) for image in images]]
        outputs = self._processor.post_process_keypoint_matching(
            outputs, image_sizes, threshold=self._match_threshold
        )
        logger.info(f"Found #raw matches: {len(outputs[0]['matching_scores'])}")

        return (
            outputs[0]["keypoints0"],
            outputs[0]["keypoints1"],
            outputs[0]["matching_scores"],
        )

    def match(
        self,
        feature_path: Path,
        matches_path: Path,
        img0: Path,
        img1: Path,
        try_full_image: bool = False,
    ):
        """
        Matches features between two images. The `feature_path` is path to the feature
        file, the `matches_path` is path to save the matches. The `img0` is path to the
        first image. The `img1` is path to the second image. The `try_full_image` is
        flag to attempt matching on full images.

        :raise: RuntimeError: If there are too many features to match on full images.
        :raise: FileNotFoundError If `feature_path` does not exist

        :return: array containing the indices of matched keypoints.

        NOTE: The function is taken from
        deep_image_matching/matchers/roma.py:match with litte changes.
        """
        timer_match = Timer(log_level=logging.DEBUG)

        # Check that feature_path exists
        if not Path(feature_path).exists():
            raise FileNotFoundError(f"Feature file {feature_path} does not exist.")
        else:
            self._feature_path = Path(feature_path)

        # Get features from h5 file
        img0 = Path(img0)
        img1 = Path(img1)
        img0_name = img0.name
        img1_name = img1.name

        # Perform matching
        if self._tiling == TileSelection.NONE:
            matches = self._match_pairs(self._feature_path, img0, img1)
            timer_match.update("[match] Match full images")
        else:
            matches = self._match_by_tile(
                feature_path,
                img0,
                img1,
                method=self._tiling,
                select_unique=True,
            )
            timer_match.update("[match] Match by tile")

        # do geometric verification
        features0 = get_features(feature_path, img0_name)
        features1 = get_features(feature_path, img1_name)

        # rescale threshold according the image original image size
        img_shape = cv2.imread(str(img0)).shape
        tile_size = max(self.config["general"]["tile_size"])
        scale_fct = np.floor(max(img_shape) / tile_size / 2)
        gv_threshold = self.config["general"]["gv_threshold"] * scale_fct

        # apply geometric verification
        _, inlMask = geometric_verification(
            kpts0=features0["keypoints"][matches[:, 0]],
            kpts1=features1["keypoints"][matches[:, 1]],
            method=self.config["general"]["geom_verification"],
            threshold=gv_threshold,
            confidence=self.config["general"]["gv_confidence"],
        )
        matches = matches[inlMask]
        timer_match.update("Geom. verification")

        # Save to h5 file
        n_matches = len(matches)
        with h5py.File(str(matches_path), "a", libver="latest") as fd:
            group = fd.require_group(img0_name)
            if n_matches >= self.min_inliers_per_pair:
                group.create_dataset(img1_name, data=matches)
            else:
                logger.debug(
                    f"Too few matches found.Skipping image pair {img0.name}-{img1.name}"
                )
                return None
        timer_match.update("save to h5")

        timer_match.print(f"{__class__.__name__} match")

    @torch.no_grad()
    def _match_pairs(
        self,
        feature_path: Path,
        img0_path: Path,
        img1_path: Path,
    ):
        """
        Perform matching between feature pairs. The `feature_path` is the path to the
        feature file. The `img0_path` is the path to the first image. The `img1_path` is
        the path to the second image.

        :return: array containing the indices of matched keypoints.

        NOTE: The function is taken from
        deep_image_matching/matchers/roma.py:_match_pairs with little changes.
        """

        img0_name = img0_path.name
        img1_name = img1_path.name

        kptsA, kptsB, _ = self.inference_match_anything(
            img0_path=str(img0_path),
            img1_path=str(img1_path),
            device=self._device,
        )
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

    def _match_by_tile(
        self,
        feature_path: Path,
        img0: Path,
        img1: Path,
        method: TileSelection = TileSelection.PRESELECTION,
        select_unique: bool = True,
    ) -> np.ndarray:
        """
        Perform matching by tile. The `feature_path` is a path to the feature file. The
        `img0` is a path to the first image. The `img1` is a path to the second image.
        The `method` is a Tile selection method. The `select_unique` is a Flag to select
        unique features.

        :return: array containing the indices of matched keypoints.

        NOTE: The function is taken from
        deep_image_matching/matchers/roma.py:_match_by_tile with little changes.
        """

        def write_tiles_disk(output_dir: Path, tiles: dict) -> None:
            output_dir = Path(output_dir)
            if output_dir.exists():
                return None
            output_dir.mkdir(parents=True)
            for i, tile in tiles.items():
                name = str(output_dir / f"tile_{i}.png")
                cv2.imwrite(name, tile)

        timer = Timer(log_level=logging.DEBUG, cumulate_by_key=True)

        tile_size = self.config["general"]["tile_size"]
        overlap = self.config["general"]["tile_overlap"]
        img0_name = img0.name
        img1_name = img1.name

        # Select tile pairs to match
        tile_pairs = tile_selection(
            img0,
            img1,
            method=method,
            quality=self._quality,
            tile_size=tile_size,
            tile_overlap=overlap,
            preselction_extractor=self._preselction_extractor,
            preselction_matcher=self._preselction_matcher,
            pipeline=self.config["general"]["preselection_pipeline"],
            tile_preselection_size=self.tile_preselection_size,
            min_matches_per_tile=self.min_matches_per_tile,
            device=self._device,
            debug_dir=self.config["general"]["output_dir"] / "debug"
            if self.config["general"]["verbose"]
            else None,
        )

        if len(tile_pairs) > self.max_tile_pairs:
            raise RuntimeError(
                f"Too many tile pairs ({len(tile_pairs)}) to match, the matching "
                + "process will be too slow and it may be inaccurate. Try to reduce the"
                + " image resolution using a lower 'Quality' parameter."
            )
        else:
            logger.info(f"Matching {len(tile_pairs)} tile pairs")
        timer.update("tile selection")

        # Read images and resize them if needed
        image0 = cv2.imread(str(img0))
        image1 = cv2.imread(str(img1))
        image0 = self._resize_image(self._quality, image0)
        image1 = self._resize_image(self._quality, image1)

        # If tiling is used, extract tiles with proper size for MatchAnything matching
        # and save them to disk
        tiler = Tiler(tiling_mode="size")
        tiles0, t_origins0, _ = tiler.compute_tiles_by_size(
            input=image0, window_size=tile_size, overlap=overlap
        )
        tiles1, t_origins1, _ = tiler.compute_tiles_by_size(
            input=image1, window_size=tile_size, overlap=overlap
        )
        tiles_dir = Path(self.config["general"]["output_dir"]) / "tiles"
        write_tiles_disk(tiles_dir / img0.name, tiles0)
        write_tiles_disk(tiles_dir / img1.name, tiles1)
        logger.debug(f"Tiles saved to {tiles_dir}")

        # Match each tile pair
        mkpts0_full = np.array([], dtype=np.float32).reshape(0, 2)
        mkpts1_full = np.array([], dtype=np.float32).reshape(0, 2)
        conf_full = np.array([], dtype=np.float32)

        for tidx0, tidx1 in tqdm(tile_pairs, leave=True, desc="Matching tiles"):
            logger.debug(f"  - Matching tile pair ({tidx0}, {tidx1})")

            tile_path0 = tiles_dir / img0.name / f"tile_{tidx0}.png"
            tile_path1 = tiles_dir / img1.name / f"tile_{tidx1}.png"

            # Run inference
            kptsA, kptsB, certainty = self.inference_match_anything(
                img0_path=str(tile_path0),
                img1_path=str(tile_path1),
                device=self._device,
            )
            kptsA, kptsB = kptsA.cpu().numpy(), kptsB.cpu().numpy()

            # Get match confidence
            conf = certainty.cpu().numpy()

            logger.debug(f"     Found {len(kptsA)} matches")

            # Viz for debugging
            if self.config["general"]["verbose"]:
                tile_match_dir = (
                    Path(self.config["general"]["output_dir"])
                    / "debug"
                    / "matches_by_tile"
                )
                tile_match_dir.mkdir(parents=True, exist_ok=True)
                t0 = cv2.imread(str(tile_path0))
                t1 = cv2.imread(str(tile_path1))
                viz_matches_cv2(
                    image0=t0,
                    image1=t1,
                    pts0=kptsA,
                    pts1=kptsB,
                    save_path=tile_match_dir
                    / f"{img0.stem}-{img1.stem}_t{tidx0}-{tidx1}.jpg",
                    line_thickness=1,
                    autoresize=False,
                    jpg_quality=60,
                )

            # Restore original image coordinates (not cropped)
            kptsA = kptsA + np.array(t_origins0[tidx0]).astype("float32")
            kptsB = kptsB + np.array(t_origins1[tidx1]).astype("float32")

            # Check if any keypoints are outside the original image (non-padded) or too
            # close to the border
            def kps_in_image(kp, img_size, border_thr=2):
                return (
                    (kp[:, 0] >= border_thr)
                    & (kp[:, 0] < img_size[1] - border_thr)
                    & (kp[:, 1] >= border_thr)
                    & (kp[:, 1] < img_size[0] - border_thr)
                )

            border_thr = 50
            maskA = kps_in_image(kptsA, image0.shape[:2], border_thr)
            maskB = kps_in_image(kptsB, image1.shape[:2], border_thr)
            msk = maskA & maskB
            kptsA = kptsA[msk]
            kptsB = kptsB[msk]

            # Append to full arrays
            mkpts0_full = np.vstack((mkpts0_full, kptsA))
            mkpts1_full = np.vstack((mkpts1_full, kptsB))
            conf_full = np.concatenate((conf_full, conf))

        logger.info("Tiles completed")
        logger.info(f"Total matches before geometric verification: {len(mkpts0_full)}")

        # Rescale keypoints to original image size
        mkpts0_full = self._resize_keypoints(self._quality, mkpts0_full)
        mkpts1_full = self._resize_keypoints(self._quality, mkpts1_full)

        # Select uniue features on image 0, on rounded coordinates
        if select_unique is True:
            decimals = 1
            _, unique_idx = np.unique(
                np.round(mkpts0_full, decimals), axis=0, return_index=True
            )
            mkpts0_full = mkpts0_full[unique_idx]
            mkpts1_full = mkpts1_full[unique_idx]

        # Viz for debugging
        if self.config["general"]["verbose"]:
            tile_match_dir = (
                Path(self.config["general"]["output_dir"]) / "debug" / "matches_by_tile"
            )
            tile_match_dir.mkdir(parents=True, exist_ok=True)
            image0 = cv2.imread(str(img0))
            image1 = cv2.imread(str(img1))
            viz_matches_cv2(
                image0,
                image1,
                mkpts0_full,
                mkpts1_full,
                save_path=tile_match_dir / f"{img0.stem}-{img1.stem}.jpg",
                line_thickness=-1,
                autoresize=True,
                jpg_quality=60,
            )

        # Create a 1-to-1 matching array
        matches0 = np.arange(mkpts0_full.shape[0])
        matches = np.hstack((matches0.reshape((-1, 1)), matches0.reshape((-1, 1))))
        matches = self._update_features_h5(
            feature_path,
            img0_name,
            img1_name,
            mkpts0_full,
            mkpts1_full,
            matches,
        )

        # Remove tiles from disk
        if not self.keep_tiles:
            shutil.rmtree(tiles_dir)

        return matches
