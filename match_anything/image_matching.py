import logging
import sys
from pathlib import Path

import torch
from deep_image_matching import extractors, matchers
from deep_image_matching.config import Config
from deep_image_matching.constants import Timer
from deep_image_matching.extractors import extractor_loader
from deep_image_matching.image_matching import ImageMatcher
from deep_image_matching.matchers import matcher_loader
from deep_image_matching.utils import ImageList
from deep_image_matching.utils.image import IMAGE_EXT

sys.path.append("./src/rebel-pose/")
from match_anything.matcher_match_anything import MatchAnythingMatcher
from match_anything.matcher_minima import MINIMARoMAMatcher

logger = logging.getLogger("dim")
timer = Timer(logger=logger)


class CustomImageMatcher(ImageMatcher):
    def __init__(self, config: Config, checkpoint_path: Path) -> None:
        """Initializes the ImageMatcher class using parameters from `config`."""
        # One must not run super().__init__(...) in this case, because
        # the __init__ of ImageMatcher would try to import MatchAnythingMatcher, but the
        # way how the class should be imported differs from the original class logic

        self.config = config
        self.image_dir = Path(config.general["image_dir"])
        self.output_dir = Path(config.general["output_dir"])
        self.strategy = config.general["matching_strategy"]
        self.extraction = config.extractor["name"]
        self.matching = config.matcher["name"]
        self.pair_file = config.general["pair_file"]
        self.rotated_images = []

        self.image_list = ImageList(self.image_dir)
        images = self.image_list.img_names
        if len(images) == 0:
            raise ValueError(f"Image folder empty. Supported formats: {IMAGE_EXT}")
        elif len(images) == 1:
            raise ValueError("Image folder must contain at least two images")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractor
        try:
            Extractor = extractor_loader(extractors, self.extraction)
        except AttributeError:
            raise ValueError(
                f"Invalid local feature extractor. {self.extraction} is not supported."
            )
        self._extractor = Extractor(self.config)

        # Initialize matcher
        try:
            if self.matching == "match_anything":
                Matcher = MatchAnythingMatcher
            elif self.matching == "minima_roma":
                Matcher = MINIMARoMAMatcher
            else:
                Matcher = matcher_loader(matchers, self.matching)

        except AttributeError:
            raise ValueError(f"Invalid matcher. {self.matching} is not supported.")

        if self.matching == "lightglue":
            self._matcher = Matcher(local_features=self.extraction, config=self.config)
        elif self.matching == "minima_roma":
            self._matcher = Matcher(checkpoint_path=checkpoint_path, config=config)
        else:
            self._matcher = Matcher(self.config)

        # Log the configuration
        logger.info("Running image matching with the following configuration:")
        logger.info(f"  Image folder: {self.image_dir}")
        logger.info(f"  Output folder: {self.output_dir}")
        logger.info(f"  Number of images: {len(self.image_list)}")
        logger.info(f"  Matching strategy: {self.strategy}")
        logger.info(f"  Image quality: {self.config.general['quality'].name}")
        logger.info(f"  Tile selection: {self.config.general['tile_selection'].name}")
        logger.info(f"  Feature extraction method: {self.extraction}")
        logger.info(f"  Matching method: {self.matching}")
        logger.info(
            f"  Geometric verification: {self.config.general['geom_verification'].name}"
        )
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
