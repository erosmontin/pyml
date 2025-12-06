## LLM-Ready Agent Instructions for `pyml` (v3)

This document gives a compact, agent-friendly description of the `pyml` package and how an LLM-based assistant should interact with it.

Goal
- Help an LLM quickly answer developer questions and generate code snippets or commands that correctly use the `pyml` public API.

High-level package map (public modules and key symbols)
- `pyml.io`:
  - `save_model(model, path, metadata=None)`
  - `load_model(path)`
  - `_detect_backend(model)` (internal helper)
- `pyml.tuning`:
  - `OptunaOptimizer` — high-level wrapper (create_estimator, suggest_params, optimize)
  - `suggest_*_params` helpers (estimator-specific)
- `pyml.logging`:
  - `get_logger(name)`
  - `log_metrics(metrics, step=None, prefix=None)`
  - `MetricsLogger` — in-memory metrics aggregation
- `pyml.plotting`:
  - `plot_optuna_study(study)`
  - `plot_training_curves(history)`
  - `plot_feature_heatmap(X, feature_names)`
- `pyml.training`:
  - `ExperimentTracker(db_path, experiment_name)`
  - `tracker.log_result(...)`, `tracker.get_results()`
- `pyml.evaluation`:
  - `ModelEvaluator` — `evaluate(estimator, X, y, ...)`
  - `evaluate_pipeline(...)` convenience function

Principles for an LLM agent integrating with `pyml`
- Prefer public APIs listed above. Only suggest internal helpers (leading underscore) if necessary and flagged as internal.
- When producing runnable code, include explicit imports, example inputs (X, y), and minimal mocking for objects that tests/examples expect (e.g., sklearn estimators).
- If the user asks to run tests or install, provide explicit shell commands and mention optional extras for heavy dependencies.

Example agent prompts and outputs
- Prompt: "Show me how to save/load a scikit-learn model with metadata." 
  - Agent response: short code snippet using `save_model` and `load_model` showing metadata file naming conventions.

- Prompt: "Run a quick Optuna tuning for RandomForest and return the best params." 
  - Agent response: show code using `OptunaOptimizer`, `suggest_params`, and `optimize` with a simple `objective` that uses `cross_val_score` and returns the best params; include a note to install `pyml[optuna]`.

Error handling and dependency guidance
- If code touches optional functionality (Optuna, MLflow, PyTorch), the agent must warn the user to install the appropriate extras (e.g., `pip install pyml[optuna]`).
- For saving/loading PyTorch models the agent should remind the user about `torch` extras and device handling.

Testing & verification
- Prefer local unit tests copied from the repo `tests/` when validating code. Commands to run tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[all]
pytest -q
```

Prompt templates (concise)
- "You are a Python assistant. Provide a minimal working example using `pyml.<module>` to accomplish: <task>. Include necessary imports and setup." 
- "Given `X, y` arrays, produce code that uses `pyml.evaluation.ModelEvaluator` to compute CV metrics and print a summary." 

Notes for maintainers
- Keep this file updated when public API changes. The LLM agent uses this as the canonical quick reference for code generation and examples.
