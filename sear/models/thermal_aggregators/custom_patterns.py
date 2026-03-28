from enum import Enum


class CustomPatterns(Enum):
    """
    Enum containing different settings for incorporating LoRA into different modules of
    the Alternating Attention Module.
    """

    FRAME_ONLY = r"^frame_blocks.+"
    """Add LoRA to frame attention blocks only"""
    GLOBAL_ONLY = r"^global_blocks.+"
    """Add LoRA to global attention blocks only"""
    ORIGINAL = r"(^frame_blocks.+)|(^global_blocks.+)"
    """
    Add LoRA to frame and global attention blocks, which is the original model setting
    """
    FIRST_QUATER = r"^(?:frame|global)_blocks\.[0-5](\..*|$)"
    """Add LoRA to first 1/4 global and frame attention blocks only"""
    FIRST_TWO_QUATERS = r"^(?:frame|global)_blocks\.([0-9]|1[0-1])(\..*|$)"
    """Add LoRA to first 2/4 global and frame attention blocks only"""
    FIRST_THREE_QUATERS = r"^(?:frame|global)_blocks\.([0-9]|1[0-7])(\..*|$)"
    """Add LoRA to first 3/4 global and frame attention blocks only"""
