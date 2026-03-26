from pathlib import Path

from dust3r.image_pairs import make_pairs

from sear.scripts.eval.scene_graphs.dust3r import Dust3rSceneGraph
from sear.scripts.eval.scene_graphs.representation import (
    scene_graph_representation,
)


def create_custom_pairs(
    images_names: list[str],
    scenegraph_option: Dust3rSceneGraph,
    window_size: int,
    reference_index: int,
    cyclic: bool,
    save_path: Path | None = None,
) -> list[tuple[str, str]]:
    """
    Create pairs for `images_names` based on graph parameters `scenegraph_option` (how
    to create the graph) `window_size` (how many neighbors to look at),
    `reference_index` (if option is ONEREF then it is the index of the reference to look
    at), `cyclic` (if the graph is a closed loop). Optionally stores the result to
    `save_path`.

    :return: created names of pairs
    """
    pairs = make_pairs(
        images_names,
        scene_graph=scene_graph_representation(
            option=scenegraph_option,
            cyclic=cyclic,
            window_size=window_size,
            reference_index=reference_index,
        ),
        prefilter=None,
        symmetrize=True,
    )

    if save_path is not None:
        with open(save_path, "w") as f:
            for pair in pairs:
                f.write(f"{pair[0]} {pair[1]}\n")

    return pairs
