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

## Cite Us

If you use `pyml` in your research or publications, please consider citing one of the following related works from the authors and collaborators who developed the underlying methods and tools:

- Montin, E., Kijowski, R., Youm, T., & Lattanzi, R. (2024). Radiomics features outperform standard radiological measurements in detecting femoroacetabular impingement on three‐dimensional magnetic resonance imaging. Journal of Orthopaedic Research. Wiley. https://doi.org/10.1002/jor.25952

- Montin, E., Kijowski, R., Youm, T., & Lattanzi, R. (2023). A radiomics approach to the diagnosis of femoroacetabular impingement. Frontiers in Radiology, 3. Frontiers Media SA. https://doi.org/10.3389/fradi.2023.1151258

- Cavatorta, C., Meroni, S., Montin, E., Oprandi, M. C., Pecori, E., Lecchi, M., Diletto, B., Alessandro, O., Peruzzo, D., Biassoni, V., Schiavello, E., Bologna, M., Massimino, M., Poggi, G., Mainardi, L., Arrigoni, F., Spreafico, F., Verderio, P., Pignoli, E., & Gandola, L. (2021). Retrospective study of late radiation-induced damages after focal radiotherapy for childhood brain tumors. PLOS ONE, 16(2), e0247748. https://doi.org/10.1371/journal.pone.0247748

- Montin, E., Belfatto, A., Bologna, M., Meroni, S., Cavatorta, C., Pecori, E., Diletto, B., Massimino, M., Oprandi, M. C., Poggi, G., Arrigoni, F., Peruzzo, D., Pignoli, E., Gandola, L., Cerveri, P., & Mainardi, L. (2020). A multi-metric registration strategy for the alignment of longitudinal brain images in pediatric oncology. Medical & Biological Engineering & Computing, 58(4), 843–855. https://doi.org/10.1007/s11517-019-02109-4


