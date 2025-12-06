"""
Unit tests for pyable_ml.logging module
"""

import pytest
from pyml.logging import get_logger, log_metrics, MetricsLogger


def test_get_logger():
    """Test logger creation."""
    logger = get_logger("test_logger")
    assert logger is not None
    assert logger.name == "test_logger"


def test_metrics_logger():
    """Test MetricsLogger class."""
    logger = MetricsLogger()
    
    # Update metrics
    logger.update({"loss": 0.5, "accuracy": 0.8})
    logger.update({"loss": 0.3, "accuracy": 0.9})
    
    # Get average
    avg = logger.get_average()
    assert avg["loss"] == pytest.approx(0.4)
    assert avg["accuracy"] == pytest.approx(0.85)
    
    # Reset and check history
    logger.reset()
    assert len(logger.history) == 1
    assert logger.current == {}


def test_log_metrics():
    """Test log_metrics function."""
    # Should not raise
    log_metrics({"loss": 0.5, "acc": 0.85}, step=1)
    log_metrics({"f1": 0.9}, prefix="val")


if __name__ == "__main__":
    test_get_logger()
    test_metrics_logger()
    test_log_metrics()
    print("✅ All logging tests passed!")
