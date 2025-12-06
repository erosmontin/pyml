import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from pyml.io import save_model, load_model


def test_save_load_iris_model():
    data = load_iris()
    X, y = data.data, data.target

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "iris_model.pkl"
        save_model(model, path)

        loaded = load_model(path)

        # Compare predictions for first 20 samples
        pred1 = model.predict(X[:20])
        pred2 = loaded.predict(X[:20])
        np.testing.assert_array_equal(pred1, pred2)
