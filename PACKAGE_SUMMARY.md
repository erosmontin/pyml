# pyable-ml Package Summary

## What We Created

A shared ML utilities package (`pyable-ml`) that eliminates code duplication between `pyfe` (radiomics/ML) and `treno` (deep learning).

## Package Structure

```
pyable-ml/
├── pyproject.toml          # Package config with optional dependencies
├── README.md               # Usage documentation
├── LICENSE                 # MIT license
├── MIGRATION_GUIDE.md      # Migration instructions for pyfe/treno
├── pyable_ml/
│   ├── __init__.py         # Main exports
│   ├── io.py               # Model save/load (sklearn + PyTorch)
│   ├── tuning.py           # Optuna hyperparameter optimization
│   ├── logging.py          # Structured logging + MLFlow support
│   ├── plotting.py         # Training curves, Optuna plots, heatmaps
│   ├── training.py         # ExperimentTracker (SQLite DB)
│   └── evaluation.py       # ModelEvaluator (CV + SMOTE)
└── tests/
    ├── test_io.py          # Model I/O tests
    ├── test_logging.py     # Logging tests
    └── test_smoke.py       # Import smoke tests
```

## Key Features

### 1. Model I/O (`pyable_ml.io`)
- Unified save/load for sklearn and PyTorch models
- Automatic backend detection
- Metadata sidecar files (.meta.json)
- Checkpoint utilities for training

```python
from pyable_ml.io import save_model, load_model

# Sklearn
save_model(sklearn_model, "model.pkl")
model = load_model("model.pkl")

# PyTorch
save_model(torch_model, "model.pth")
model = load_model("model.pth", model_class=MyModel(), device="cuda")
```

### 2. Hyperparameter Tuning (`pyable_ml.tuning`)
- Optuna wrapper with smart parameter spaces for 20+ estimators
- TPE sampler and median pruner by default
- Reproducible seeds

```python
from pyable_ml.tuning import OptunaOptimizer

optimizer = OptunaOptimizer(
    estimator_class=RandomForestClassifier,
    n_trials=50
)

result = optimizer.optimize(objective_func)
best_model = optimizer.get_best_estimator()
```

### 3. Logging (`pyable_ml.logging`)
- Structured logging with file + console handlers
- Metrics logger with automatic averaging
- Optional MLFlow integration
- Warning suppression utilities

```python
from pyable_ml.logging import get_logger, log_metrics, MetricsLogger

logger = get_logger(__name__, log_file="train.log")
log_metrics({"loss": 0.5, "acc": 0.85}, step=10)

metrics = MetricsLogger()
metrics.update({"loss": 0.5})
avg = metrics.get_average()
```

### 4. Plotting (`pyable_ml.plotting`)
- Grid search results visualization
- Optuna optimization history and parameter importance
- Training curves (loss + metrics over epochs)
- Feature heatmaps (correlation or sample-feature matrix)

```python
from pyable_ml.plotting import plot_training_curves, plot_optuna_study

fig, axes = plot_training_curves(history)
fig.savefig("training.png")

fig, ax = plot_optuna_study(study)
```

### 5. Experiment Tracking (`pyable_ml.training`)
- SQLite database for storing experiment results
- Tracks selectors, estimators, hyperparameters, metrics
- Query and compare results across experiments

```python
from pyable_ml.training import ExperimentTracker

tracker = ExperimentTracker(db_path="experiments.db")
tracker.log_result(
    selector="mutual_info",
    estimator="random_forest",
    metrics={"f1": 0.85}
)

results = tracker.get_results()
```

### 6. Model Evaluation (`pyable_ml.evaluation`)
- Cross-validation with optional SMOTE oversampling
- Per-fold metrics tracking
- Auto-detection of classification vs regression
- Multiple scoring metrics

```python
from pyable_ml.evaluation import ModelEvaluator

evaluator = ModelEvaluator(cv=5, use_smote=True)
results = evaluator.evaluate(model, X, y)
print(results["mean_scores"])
```

## Optional Dependencies

Install only what you need:

```bash
# Core (numpy, pandas, scikit-learn)
pip install pyable-ml

# With Optuna
pip install pyable-ml[optuna]

# With plotting
pip install pyable-ml[plotting]

# With PyTorch support
pip install pyable-ml[torch]

# With MLFlow
pip install pyable-ml[mlflow]

# Everything
pip install pyable-ml[all]
```

## Migration Status

### pyfe
- ✅ Added `pyable-ml[optuna,plotting]` to dependencies
- ✅ Created backward-compatibility shims in `pyfe/learn/`
- ✅ Old imports still work but show deprecation warnings
- ⏳ Next: Update examples and tests to use new imports

### treno
- ✅ Added `pyable-ml[torch,plotting]` to dependencies
- ✅ Added deprecation warnings to `save_model`/`load_model`
- ✅ Old imports still work
- ⏳ Next: Update examples and tests to use new imports

## Benefits

1. **No duplication**: ~1500 lines of code now shared between packages
2. **Smaller packages**: pyfe and treno are more focused
3. **Optional deps**: Users only install what they need (no forced PyTorch for radiomics users)
4. **Consistent API**: Same logging, I/O, and tuning across both packages
5. **Easier testing**: Shared utilities have their own test suite
6. **Independent evolution**: Can improve shared code without releasing both packages

## Code Removal Plan

### Phase 1 (v2.1 - Current)
- ✅ Create pyable-ml package
- ✅ Add as dependency to pyfe and treno
- ✅ Add backward-compatibility shims with deprecation warnings
- ✅ Keep old code functional

### Phase 2 (v2.2 - In 3 months)
- Remove old implementations from pyfe.learn
- Require direct imports from pyable_ml
- Update all examples and documentation

### Phase 3 (v3.0/v4.0 - Major version)
- Remove all shims
- Clean deprecation warnings

## Testing

All core functionality tested:

```bash
# Test pyable-ml
cd /home/erosm/packages/pyable-ml
python tests/test_smoke.py   # ✅ All imports work
python tests/test_io.py      # ✅ Save/load works
python tests/test_logging.py # ✅ Logging works

# Test backward compatibility in pyfe
cd /home/erosm/packages/pyfe
python -c "from pyfe.learn.optimizer import OptunaOptimizer"  # Shows deprecation warning

# Test backward compatibility in treno
cd /home/erosm/packages/treno
python -c "from treno.models import save_model"  # Shows deprecation warning
```

## Next Steps

1. ✅ Package created and tested
2. ✅ Dependencies updated in pyfe and treno
3. ✅ Backward compatibility maintained
4. ⏳ Update examples in pyfe to use pyable_ml imports
5. ⏳ Update examples in treno to use pyable_ml imports
6. ⏳ Add CI workflow to test pyable-ml separately
7. ⏳ Publish pyable-ml to PyPI (optional) or keep as git dependency

## Installation for Development

```bash
# Install pyable-ml in editable mode
cd /home/erosm/packages/pyable-ml
pip install -e .[all]

# Install pyfe (will use local pyable-ml)
cd /home/erosm/packages/pyfe
pip install -e .

# Install treno (will use local pyable-ml)
cd /home/erosm/packages/treno
pip install -e .
```

## Questions?

See `MIGRATION_GUIDE.md` for detailed migration instructions or open an issue.
