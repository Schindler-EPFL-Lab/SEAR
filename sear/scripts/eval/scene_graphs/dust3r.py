from enum import Enum


class Dust3rSceneGraph(Enum):
    """Contains possible scene graph variants for dust3r"""

    COMPLETE = "complete"
    """Find matches between all images"""

    WINDOW = "swin"
    """For each image find matches with its neighbors"""

    LOG_WINDOW = "logwin"
    """
    For each image find matches with other images which are far away by 1, 2, 4, 8,
    ... indices, i.e. exponentially away.
    """

    ONEREF = "oneref"
    """Find matches between one reference image and all other images"""
