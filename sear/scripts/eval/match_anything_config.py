from dataclasses import dataclass
from pathlib import Path

from sear.scripts.eval.base import EvalParametersBase
from sear.scripts.eval.scene_graphs.dust3r import Dust3rSceneGraph


@dataclass
class MatchAnythingEvalParameters(EvalParametersBase):
    """A config to evaluate MatchAnything on camera pose estimation for a trajectory"""

    method_name: str = "MatchAnything"
    """The method name used to mark saved results"""

    config_file: Path = Path("./sear/configs/match_anything_config.yaml")
    """Configuration file used to run match anything"""
    pipeline: str = "match_anything"
    """Pipeline to run"""
    camera_options: Path = Path("./sear/configs/cameras.yaml")
    """Configuration file used to specify predicted cameras"""

    method_name: str = "match-anything"
    """The name of the method"""

    scenegraph_option: Dust3rSceneGraph = Dust3rSceneGraph.LOG_WINDOW
    """
    How to compute matches between images. If the number of images is too large then one
    should use other strategies but not the "complete" one.
    """
    window_size: int = 5
    """
    For each image it represents the amount of neighbors used to find matches. The
    bigger the window_size the more neighbors are taken
    """
    reference_index: int = 0
    """Index of the reference image used to find matches."""
    cyclic: bool = False
    """Whether the scene graph is a closed loop"""
