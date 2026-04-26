from enum import Enum


class PossibleAggregators(Enum):
    """
    The class describing possible ThermalAggregator to use for ablation studies.
    """

    NO_LORA = "no-lora"
    """Use Thermal Projector but no LoRA"""
    NO_THERMAL_PROJECTOR = "no-thermal-projector"
    """Use LoRA but no Thermal Projector"""
    CAMERA_TOKEN = "camera-token"
    """
    Use LoRA and learnable <thermal camera token> instead of the Thermal projector. For
    each thermal sequence it replaces a <camera token> with a <thermal camera token>
    """
    THERMAL_MODALITY_TOKEN = "thermal-modality-token"
    """
    Use LoRA and learnable <thermal modality token> instead of the Thermal projector.
    For each thermal token the aggregator sums it with the <thermal modality token>.
    """
    THERMAL_PROJECTOR = "thermal-projector"
    """
    Use Thermal Projector and add LoRA to frame and global attention blocks, which is 
    the original model setting
    """
    FRAME_ONLY = "frame-only"
    """Use Thermal Projector and add LoRA to frame attention blocks only"""
    GLOBAL_ONLY = "global-only"
    """Use Thermal Projector and add LoRA to global attention blocks only"""
    FIRST_QUATER = "first-quater"
    """
    Use Thermal Projector and add LoRA to first 1/4 global and frame attention blocks
    only
    """
    FIRST_TWO_QUATERS = "first-two-quaters"
    """
    Use Thermal Projector and add LoRA to first 2/4 global and frame attention blocks
    only
    """
    FIRST_THREE_QUATERS = "first-three-quaters"
    """
    Use Thermal Projector and add LoRA to first 3/4 global and frame attention blocks
    only
    """
    CAMERA_TOKEN_FRAME_ONLY = "frame-only-camera-token"
    """Use thermal camera token and adds LoRA to frame attention blocks only"""
    CAMERA_TOKEN_GLOBAL_ONLY = "global-only-camera-token"
    """Use thermal camera token and adds LoRA to global attention blocks only"""
    CAMERA_TOKEN_NO_LORA = "no-lora-camera-token"
    """Use camera token but no LoRA"""
