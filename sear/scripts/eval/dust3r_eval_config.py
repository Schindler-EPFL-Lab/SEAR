from dataclasses import dataclass
from pathlib import Path

from sear.scripts.eval.base import EvalParametersBase
from sear.scripts.eval.scene_graphs.dust3r import Dust3rSceneGraph


@dataclass(kw_only=True)
class DUST3REvalParameters(EvalParametersBase):
    """A config to evaluate DUST3R on 3D camera pose estimation"""

    method_name: str = "DUST3R"
    """The method name used to mark saved results"""

    dust3r_ckpt_path: Path = Path("input")
    """DUST3R Model checkpoint"""
    align_schedule: str = "cosine"
    """Scheduler used for global aligning"""
    align_learning_rate: float = 0.01
    """Learning rate used for global aligning"""
    align_num_iterations: int = 300
    """Number of iterations for global aligning"""
    image_width: int = 512
    """Max width of the loaded images"""
    verbose: bool = True
    """Whether to print additional info during images loading"""

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
    matching_confidence_threshold: float = 1.0
    """
    The threshold such that if confidence is above the values then the match is
    considered to be correct.
    """
