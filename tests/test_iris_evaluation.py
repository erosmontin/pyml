from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from pyml.evaluation import ModelEvaluator


def test_model_evaluator_on_iris():
    data = load_iris()
    X, y = data.data, data.target

    evaluator = ModelEvaluator(cv=3, random_state=42, use_smote=False)

    estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    results = evaluator.evaluate(estimator, X, y)

    assert 'test_scores' in results
    # Default scoring for classification includes 'accuracy'
    assert 'accuracy' in results['test_scores']
    mean_acc = results['test_scores']['accuracy']['mean']
    assert 0.0 <= mean_acc <= 1.0
