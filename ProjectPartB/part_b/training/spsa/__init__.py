"""SPSA gradient estimation with PyTorch models and optimisers."""

from .config import SPSAConfig
from .trainer import evaluate, train

__all__ = ["SPSAConfig", "evaluate", "train"]
