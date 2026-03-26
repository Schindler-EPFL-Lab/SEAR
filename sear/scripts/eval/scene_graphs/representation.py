from sear.scripts.eval.scene_graphs.dust3r import Dust3rSceneGraph
from sear.scripts.eval.scene_graphs.mast3r import Mast3rSceneGraph


def scene_graph_representation(
    option: Dust3rSceneGraph | Mast3rSceneGraph,
    cyclic: bool,
    window_size: int = 3,
    reference_index: int = 1,
) -> str:
    """
    Converts a graph scene representation to a format expected by the `make_pair`
    function implemented in DUST3R or MAST3R. The `option` specifies the type of the
    graph while the `cyclic`, `window_size`, `reference_index` determine the parameters
    of the scene graph. The `window_size` represents the amount of neighbors used to
    find matches. The bigger the window_size the more neighbors are taken. The
    `reference_index` is the index of the reference image used to find matches. The
    `cyclic` defines whether the scene graph is a closed loop.

    :raise: RuntimeError `option` is not from `Dust3rSceneGraph` or `Mast3rSceneGraph`.
    """
    cyclic_str = "noncyclic" if not cyclic else "cyclic"
    if option is Dust3rSceneGraph.COMPLETE:
        return option.value
    elif option is Dust3rSceneGraph.WINDOW or option is Dust3rSceneGraph.LOG_WINDOW:
        return f"{option.value}_{cyclic_str}-{window_size}"
    elif option is Dust3rSceneGraph.ONEREF:
        return f"{option.value}_{reference_index}"
    elif option is Mast3rSceneGraph.RETRIEVAL:
        return option.value
    else:
        raise RuntimeError(
            "The `option` must be from `Dust3rSceneGraph` or `Mast3rSceneGraph` but got"
            + f" {option}."
        )
