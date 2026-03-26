from dataclasses import dataclass
from pathlib import Path

from sear.scripts.eval.base import EvalParametersBase
from sear.scripts.eval.scene_graphs.dust3r import Dust3rSceneGraph
from sear.scripts.eval.scene_graphs.mast3r import Mast3rSceneGraph


@dataclass(kw_only=True)
class MAST3REvalParameters(EvalParametersBase):
    """A config to evaluate MAST3R"""

    method_name: str = "MAST3R"
    """The method name used to mark saved results"""

    mast3r_ckpt_path: Path = Path("input")
    """MAST3R Model checkpoint"""
    scenes_root_path: Path = Path("scenes-root-path")
    """Directory containing processed VGGT scenes"""
    depth_eps: float = 1e-8
    """Depth value smaller this value do not take part in training"""
    val_split_ratio: float = 0.2
    """Ratio of scenes used for validation"""
    seed: int = 0
    """Random seed for the dataset"""
    output_dir: Path = Path("./outputs")
    """Directory to save metrics files"""

    align_schedule: str = "cosine"
    """Scheduler used for global aligning"""
    align_learning_rate: float = 0.01
    """Learning rate used for global aligning"""
    align_num_iterations: int = 300
    """Number of iterations for global aligning"""
    image_size: int = 512
    """Max width of the loaded images"""
    verbose: bool = True
    """Whether to print additional info during images loading"""
    retrieval_model: Path | None = None
    """Path to the retrieval model to load"""
    scenegraph_option: Dust3rSceneGraph | Mast3rSceneGraph = Dust3rSceneGraph.LOG_WINDOW
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

    learning_rate_coarse: float = 0.07
    """Learning rate utilized during coarse optimization (3D optim)."""
    num_iterations_coarse: int = 300
    """Number of iterations of coarse optimization (3D optim)."""

    learning_rate_fine: float = 0.01
    """
    Learning rate utilized during the refinement stage.
    """
    num_iterations_fine: int = 300
    """Number of iterations of the refinement stage."""

    optimize_depth: bool = True
    """Whether to optimize depth maps during matching."""

    shared_intrinsics: bool = True
    """Whether the intrinsics are the same for all images"""

    matching_confidence_threshold: float = 0.0
    """
    The threshold such that if confidence is above the values then the match is
    considered to be correct.
    """
