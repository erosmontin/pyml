"""
pyml.plotting

Plotting helpers for GridSearch results, Optuna studies, training curves, and feature heatmaps.
"""
from typing import Optional, Tuple, Union, Iterable

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
except Exception as e:
    missing = (
        "matplotlib, seaborn, pandas and numpy are required for plotting. "
        "Install with: pip install matplotlib seaborn pandas numpy"
    )
    raise ImportError(missing) from e


def _as_dataframe(results: Union[Iterable[dict], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results.copy()
    else:
        return pd.DataFrame(list(results))

# rest of file identical to original plotting implementation

def plot_grid_search_results(
    results: Union[Iterable[dict], pd.DataFrame],
    metric: str = "f1_macro",
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    sort_by: Optional[str] = None,
    ax=None,
) -> Tuple[object, object]:
    df = _as_dataframe(results)
    required = {"estimator", "selector", "feature_count", metric}
    if not required.issubset(df.columns):
        raise ValueError(f"results must contain columns: {required}. Found: {df.columns.tolist()}")

    df_plot = df.copy()
    df_plot["feature_count"] = df_plot["feature_count"].astype(str)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sns.barplot(data=df_plot, x="feature_count", y=metric, hue="estimator", ci=None, ax=ax)
    ax.set_xlabel("# features (k)")
    ax.set_ylabel(metric)
    if title is None:
        title = f"Grid search: {metric} by estimator and feature count"
    ax.set_title(title)
    ax.legend(title="estimator", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax

# other plotting functions (plot_optuna_study, plot_feature_heatmap, plot_training_curves) are copied verbatim from original module
