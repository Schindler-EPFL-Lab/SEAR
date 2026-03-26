from dataclasses import dataclass
from pathlib import Path

from sear.scripts.eval.base import EvalParametersBase
from sear.scripts.eval.scene_graphs.dust3r import Dust3rSceneGraph


@dataclass
class MINIMARoMAEvalParameters(EvalParametersBase):
    """A config to evaluate MINIMA-RoMA on camera pose estimation for a trajectory"""

    config_file: Path = Path("./sear/configs/minima_roma.yaml")
    """Configuration file used to run MINIMA-RoMA"""
    pipeline: str = "minima_roma"
    """Pipeline to run"""
    camera_options: Path = Path("./sear/configs/cameras.yaml")
    """Configuration file used to specify predicted cameras"""
    method_name: str = "minima-roma"
    """The name of the method"""
    checkpoint_path: Path = Path("checkpoint-path")
    """Checkpoint path to the MINIMA-RoMA model"""

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
