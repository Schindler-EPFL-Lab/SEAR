from enum import Enum


class RunMode(Enum):
    """Specifies what part of data should be used."""

    TRAIN = ["train"]
    """Use only training scenes"""
    EVAL = ["eval"]
    """Use only eval scenes"""
    ALL = ["train", "eval"]
    """Use training and eval scenes"""
