"""
Unit tests for pyable_ml.io module
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from pyable_ml.io import save_model, load_model


def test_save_load_sklearn_model():
    """Test save/load with sklearn model."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create and train a simple model
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pkl"
        save_model(model, path)
        
        # Load model
        loaded_model = load_model(path)
        
        # Verify predictions match
        pred1 = model.predict(X[:10])
        pred2 = loaded_model.predict(X[:10])
        np.testing.assert_array_equal(pred1, pred2)


def test_save_load_with_metadata():
    """Test save/load with metadata."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    metadata = {"version": "1.0", "author": "test"}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pkl"
        save_model(model, path, metadata=metadata)
        
        # Check metadata file was created
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        assert metadata_path.exists()
        
        # Load and verify
        loaded_model = load_model(path)
        pred1 = model.predict(X[:10])
        pred2 = loaded_model.predict(X[:10])
        np.testing.assert_array_equal(pred1, pred2)


def test_backend_autodetection():
    """Test automatic backend detection."""
    from sklearn.tree import DecisionTreeClassifier
    from pyable_ml.io import _detect_backend
    
    model = DecisionTreeClassifier()
    backend = _detect_backend(model)
    assert backend == "sklearn"


if __name__ == "__main__":
    test_save_load_sklearn_model()
    test_save_load_with_metadata()
    test_backend_autodetection()
    print("✅ All I/O tests passed!")
