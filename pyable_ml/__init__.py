"""
pyable-ml: Shared ML utilities for pyfe and treno

Lightweight ML utilities with optional heavy dependencies.
"""

__version__ = "0.1.0"

# Core exports (always available)
from .io import save_model, load_model

__all__ = [
    "save_model",
    "load_model",
]

# Optional exports (require extras)
try:
    from .tuning import OptunaOptimizer, suggest_params
    __all__.extend(["OptunaOptimizer", "suggest_params"])
except ImportError:
    pass

try:
    from .plotting import (
        plot_optuna_study,
        plot_training_curves,
        plot_feature_heatmap,
    )
    __all__.extend([
        "plot_optuna_study",
        "plot_training_curves",
        "plot_feature_heatmap",
    ])
except ImportError:
    pass

try:
    from .training import ExperimentTracker
    __all__.extend(["ExperimentTracker"])
except ImportError:
    pass

try:
    from .evaluation import ModelEvaluator
    __all__.extend(["ModelEvaluator"])
except ImportError:
    pass
