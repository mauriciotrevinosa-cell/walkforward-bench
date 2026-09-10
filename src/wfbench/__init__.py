"""Walk-forward evaluation that reports its own selection bias.

Reduced showcase extract from Atlas. See README.md for what was left out and why.
"""

from .bench import (
    DEFAULT_TEST_BARS,
    DEFAULT_TRAIN_BARS,
    MIN_FOLDS,
    MIN_POSITION_CHANGES,
    ConfigResult,
    Fold,
    SweepReport,
    evaluate_config,
    sweep,
)
from .strategies import REGISTRY, Strategy, get_strategy

__all__ = [
    "sweep", "evaluate_config",
    "SweepReport", "ConfigResult", "Fold",
    "Strategy", "get_strategy", "REGISTRY",
    "DEFAULT_TRAIN_BARS", "DEFAULT_TEST_BARS", "MIN_FOLDS", "MIN_POSITION_CHANGES",
]

__version__ = "0.1.0"
