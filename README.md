<!-- Project README for pyml (v3) -->

# pyml

Lightweight utilities for machine learning workflows used by `pyfe` and `treno`.

This repository provides small, well-scoped helpers for hyperparameter tuning, model
I/O, structured logging (with optional MLflow), plotting utilities, experiment tracking,
and evaluation helpers. The core package keeps runtime dependencies minimal; pull extras
when you need heavier tools.

**Package name:** `pyml` — version `3`

## Key Features

- Hyperparameter tuning: `pyml.tuning` — Optuna helpers and an `OptunaOptimizer` wrapper
- Model I/O: `pyml.io` — save/load supporting scikit-learn and PyTorch backends
- Logging: `pyml.logging` — structured metrics logger with optional MLflow integration
- Plotting: `pyml.plotting` — functions that return `(fig, ax)` for training and Optuna
- Training utilities: `pyml.training` — `ExperimentTracker` for lightweight experiment storage
- Evaluation: `pyml.evaluation` — CV evaluators with optional SMOTE handling

## Installation

Install from PyPI (if published) or directly from this repository. Available extras:

```bash
# From GitHub tag `v3` (recommended to match this release):
pip install "git+https://github.com/erosmontin/pyml.git@v3"

# Basic install (minimal deps)
pip install pyml

# With extras
pip install pyml[optuna]
pip install pyml[plotting]
pip install pyml[torch]
pip install pyml[mlflow]
pip install pyml[all]
```

Note: installing `pyml[all]` pulls several heavy dependencies (PyTorch, MLflow, Optuna).

## Quick examples

Save and load a scikit-learn model:

```python
from pyml.io import save_model, load_model

# sklearn estimator
save_model(model, "model.pkl")
loaded = load_model("model.pkl")
```

Hyperparameter tuning (sketch):

```python
from pyml.tuning import OptunaOptimizer
from sklearn.ensemble import RandomForestClassifier

opt = OptunaOptimizer(estimator_class=RandomForestClassifier, n_trials=30)

# objective should accept a parameterized estimator or return a score
def objective(trial):
    params = opt.suggest_params(trial)
    clf = opt.create_estimator(params)
    return cross_val_score(clf, X, y, cv=3).mean()

opt.optimize(objective)
best = opt.create_estimator(opt.best_params or {})
```

Plotting returns `matplotlib` objects for flexible use:

```python
from pyml.plotting import plot_training_curves
fig, ax = plot_training_curves(history)
fig.savefig("training.png")
```

Experiment tracking (simple SQLite-backed tracker):

```python
from pyml.training import ExperimentTracker
tracker = ExperimentTracker(db_path="experiments.db", experiment_name="run_001")
tracker.log_result(selector="mi", estimator="rf", feature_count=50, metrics={"f1": 0.82})
```

## Migration note

This repository formerly used the module name `pyable_ml`. That code has been migrated
to `pyml` and legacy imports have been removed. If you maintained code importing
`pyable_ml`, update to `pyml` and update any references in CI or docs.

## Running tests (developer)

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[all]
pytest -q
```

Or run tests against the GitHub `v3` tag in a temp env (example):

```bash
python3 -m venv /tmp/pyml-test-env
source /tmp/pyml-test-env/bin/activate
pip install "git+https://github.com/erosmontin/pyml.git@v3"
# copy tests and run
```

## Contributing

Please open issues or PRs on the GitHub repository `https://github.com/erosmontin/pyml`.

## License

MIT — see the `LICENSE` file.

