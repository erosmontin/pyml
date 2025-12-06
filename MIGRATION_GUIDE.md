# pyml Migration Guide

This document explains how to migrate `pyfe` and `treno` to use the shared `pyml` utilities package.

## What Changed

### Before (Duplicated Code)
- **pyfe**: Had its own `pyfe.learn.optimizer`, `pyfe.learn.visualization`, `pyfe.learn.tracker`, `pyfe.learn.evaluator`
- **treno**: Had its own `save_model`, `load_model`, checkpoint utilities in `treno.models`

### After (Shared Utilities)
- **pyml**: New shared package containing:
    - `pyml.tuning`: Optuna hyperparameter optimization
    - `pyml.io`: Model save/load for sklearn and PyTorch
    - `pyml.logging`: Structured logging with MLFlow support
    - `pyml.plotting`: Training curves, Optuna plots, feature heatmaps
    - `pyml.training`: Experiment tracking with SQLite
    - `pyml.evaluation`: Cross-validation evaluators

## Migration Steps

### 1. Install pyml

```bash
# For pyfe (needs plotting and optuna)
pip install -e ../pyml[optuna,plotting]

# For treno (needs torch and plotting)
pip install -e ../pyml[torch,plotting]
```

### 2. Update pyproject.toml Dependencies

**pyfe/pyproject.toml:**
```toml
dependencies = [
    # ... existing deps ...
    "pyml[optuna,plotting]",
]
```

**treno/pyproject.toml:**
```toml
dependencies = [
    # ... existing deps ...
    "pyml[torch,plotting]",
]
```

### 3. Update Import Statements

#### In pyfe

**Old imports:**
```python
from pyfe.learn.optimizer import OptunaOptimizer, suggest_params
from pyfe.learn.visualization import plot_optuna_study, plot_grid_search_results
from pyfe.learn.tracker import ExperimentTracker
from pyfe.learn.evaluator import ModelEvaluator
```

**New imports:**
```python
from pyml.tuning import OptunaOptimizer, suggest_params
from pyml.plotting import plot_optuna_study, plot_grid_search_results, plot_training_curves
from pyml.training import ExperimentTracker
from pyml.evaluation import ModelEvaluator
```

#### In treno

**Old imports:**
```python
from treno.models import save_model, load_model, save_checkpoint, load_checkpoint
```

**New imports:**
```python
from pyml.io import save_model, load_model, save_checkpoint, load_checkpoint
```

### 4. Add Backward Compatibility Shims (Temporary)

To maintain backward compatibility for one release, add re-exports in the old locations:

**pyfe/pyfe/learn/optimizer.py:**
```python
"""
DEPRECATED: Use pyml.tuning instead.
This module is kept for backward compatibility and will be removed in v3.0.
"""
import warnings
warnings.warn(
    "pyfe.learn.optimizer is deprecated. Use pyml.tuning instead.",
    DeprecationWarning,
    stacklevel=2
)

from pyml.tuning import OptunaOptimizer, suggest_params

__all__ = ["OptunaOptimizer", "suggest_params"]
```

**treno/treno/models.py** (keep existing functions but add deprecation):
```python
from pyml.io import save_model as _save_model, load_model as _load_model

def save_model(model, path):
    """DEPRECATED: Use pyml.io.save_model instead."""
    import warnings
    warnings.warn(
        "treno.models.save_model is deprecated. Use pyml.io.save_model instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _save_model(model, path)

def load_model(model, path, device='cpu'):
    """DEPRECATED: Use pyml.io.load_model instead."""
    import warnings
    warnings.warn(
        "treno.models.load_model is deprecated. Use pyml.io.load_model instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _load_model(path, model_class=model, device=device)
```

### 5. Update __init__.py Exports

**pyfe/pyfe/__init__.py:**
```python
# Add new exports
from pyml.tuning import OptunaOptimizer
from pyml.plotting import plot_optuna_study
from pyml.training import ExperimentTracker

__all__ = [
    # ... existing exports ...
    "OptunaOptimizer",
    "plot_optuna_study",
    "ExperimentTracker",
]
```

**treno/treno/__init__.py:**
```python
# Replace old exports
from pyml.io import save_model, load_model, save_checkpoint, load_checkpoint

__all__ = [
    # ... existing exports ...
    "save_model",
    "load_model",
    "save_checkpoint",
    "load_checkpoint",
]
```

### 6. Update Examples and Tests

Search for old imports in examples and tests:
```bash
# Find old imports in pyfe
cd /home/erosm/packages/pyfe
grep -r "from pyfe.learn.optimizer" .
grep -r "from pyfe.learn.visualization" .

# Find old imports in treno
cd /home/erosm/packages/treno
grep -r "from treno.models import save_model" .
```

Replace with new imports from `pyml`.

### 7. Run Tests

```bash
# Test pyfe
cd /home/erosm/packages/pyfe
python -m pytest tests/

# Test treno
cd /home/erosm/packages/treno
python -m pytest tests/
```

### 8. Update Documentation

- Update README files to mention `pyable-ml` dependency
- Add installation instructions with extras
- Update code examples to use new imports

## API Changes

### Model I/O

**Old (treno):**
```python
save_model(model, "model.pth")
load_model(model_instance, "model.pth", device="cuda")
```

**New (pyable-ml):**
```python
from pyml.io import save_model, load_model

# Save
save_model(model, "model.pth")

# Load (for PyTorch, provide model instance)
model = load_model("model.pth", model_class=model_instance, device="cuda")

# For sklearn, no model_class needed
model = load_model("model.pkl")
```

### Optuna Optimization

No API changes - same interface in `pyml.tuning` as in `pyfe.learn.optimizer`.

### Plotting

**New feature added:**
```python
from pyml.plotting import plot_training_curves

history = {
    'loss': [0.5, 0.4, 0.3],
    'val_loss': [0.6, 0.5, 0.4],
    'accuracy': [0.7, 0.8, 0.85]
}

fig, axes = plot_training_curves(history)
fig.savefig('training.png')
```

## Deprecation Timeline

- **v2.1** (current): Add `pyable-ml` dependency, add deprecation warnings, keep old modules
- **v2.2** (3 months): Remove old modules, require imports from `pyable-ml`
- **v3.0**: Clean up any remaining shims

## Benefits

1. **No duplication**: Shared code means bug fixes and improvements benefit both packages
2. **Smaller packages**: `pyfe` and `treno` are now smaller and more focused
3. **Optional dependencies**: Users only install what they need (optuna, plotting, torch)
4. **Easier testing**: Shared utilities have their own test suite
5. **Consistent API**: Same logging, I/O, and tuning interface across packages

## Troubleshooting

### Import errors
If you see `ModuleNotFoundError: No module named 'pyml'`:
```bash
pip install -e /path/to/pyml
```

### Missing optuna
If you see errors about optuna:
```bash
pip install pyable-ml[optuna]
```

### Missing matplotlib/seaborn
If you see plotting errors:
```bash
pip install pyable-ml[plotting]
```

### PyTorch model loading
For PyTorch models, you must provide a model instance:
```python
model_instance = MyModel()
loaded = load_model("model.pth", model_class=model_instance, device="cpu")
```

## Questions?

Open an issue in the pyable-ml repository or contact the maintainer.
