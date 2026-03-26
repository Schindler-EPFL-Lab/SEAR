from enum import Enum


class AlignPart(Enum):
    """
    What part of images use for alignment - RGB/Thermal or all.
    """

    RGB = "rgb"
    """Use only RGB poses for alignment"""
    THERMAL = "thermal"
    """Use only thermal poses for alignment"""
    ALL = "all"
    """Use all poses for alignment"""
