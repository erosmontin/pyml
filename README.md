# pyml

Shared ML utilities for `pyfe` (radiomics/feature extraction) and `treno` (deep learning).

## Features

- **Hyperparameter tuning** (`pyml.tuning`): Optuna wrappers with smart parameter spaces
- **Model I/O** (`pyml.io`): Unified save/load for sklearn and PyTorch models
- **Logging** (`pyml.logging`): Structured logging with optional MLFlow integration
- **Plotting** (`pyml.plotting`): Training curves, Optuna visualizations, feature importance
- **Training utilities** (`pyml.training`): Callbacks, metrics, experiment tracking
- **Evaluation** (`pyml.evaluation`): Cross-validation evaluators with SMOTE support

## Installation

```bash
# Basic install (no heavy dependencies)
pip install pyml

# With Optuna for hyperparameter tuning
pip install pyml[optuna]

# With plotting capabilities
pip install pyml[plotting]

# With PyTorch model I/O
pip install pyml[torch]

# Everything
pip install pyml[all]
```

## Quick Start

### Hyperparameter Tuning

```python
from pyml.tuning import OptunaOptimizer
from sklearn.ensemble import RandomForestClassifier

optimizer = OptunaOptimizer(
    estimator_class=RandomForestClassifier,
    n_trials=50,
    direction="maximize"
)

def objective(estimator):
    return cross_val_score(estimator, X, y, cv=5).mean()

result = optimizer.optimize(objective)
best_model = optimizer.get_best_estimator()
```

### Model I/O

```python
from pyml.io import save_model, load_model

# Works with sklearn and PyTorch
save_model(model, "model.pkl")
loaded_model = load_model("model.pkl")
```

### Experiment Tracking

```python
from pyml.training import ExperimentTracker

tracker = ExperimentTracker(db_path="experiments.db", experiment_name="my_exp")
tracker.log_result(
    selector="mutual_info",
    estimator="random_forest",
    feature_count=50,
    metrics={"f1": 0.85, "accuracy": 0.88}
)

# Query results
results = tracker.get_results()
```

### Plotting

```python
from pyml.plotting import plot_optuna_study, plot_training_curves

# Plot Optuna optimization
fig, ax = plot_optuna_study(study)
fig.savefig("optuna.png")

# Plot training history
fig, ax = plot_training_curves(history)
fig.savefig("training.png")
```

## Design Philosophy

- **Lightweight core**: Core package has minimal dependencies (numpy, pandas, scikit-learn)
- **Optional heavy deps**: Use extras to install Optuna, matplotlib, PyTorch only when needed
- **Backend agnostic**: Model I/O auto-detects sklearn vs PyTorch and handles appropriately
- **Return figures**: Plotting functions return (fig, ax) for customization rather than showing inline

## License

MIT License - see LICENSE file
