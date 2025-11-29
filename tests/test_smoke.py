"""
Simple smoke tests for pyable-ml package
"""

def test_imports():
    """Test that core modules can be imported."""
    from pyable_ml import save_model, load_model
    from pyable_ml.io import save_checkpoint, load_checkpoint
    from pyable_ml.logging import get_logger, MetricsLogger
    
    print("✓ Core imports successful")


def test_optional_imports():
    """Test optional imports (may fail if extras not installed)."""
    try:
        from pyable_ml.tuning import OptunaOptimizer, suggest_params
        print("✓ Tuning module available")
    except ImportError:
        print("⚠ Tuning module not available (install optuna)")
    
    try:
        from pyable_ml.plotting import plot_training_curves, plot_feature_heatmap
        print("✓ Plotting module available")
    except ImportError:
        print("⚠ Plotting module not available (install matplotlib, seaborn)")
    
    try:
        from pyable_ml.training import ExperimentTracker
        print("✓ Training module available")
    except ImportError:
        print("⚠ Training module not available")
    
    try:
        from pyable_ml.evaluation import ModelEvaluator
        print("✓ Evaluation module available")
    except ImportError:
        print("⚠ Evaluation module not available")


if __name__ == "__main__":
    test_imports()
    test_optional_imports()
    print("\n✅ All smoke tests passed!")
