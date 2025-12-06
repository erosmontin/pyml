"""
Structured logging utilities with optional MLFlow integration.

Provides consistent logging interface across pyfe and treno.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path to log to
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_metrics(
    metrics: Dict[str, Any],
    step: Optional[int] = None,
    prefix: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log metrics to console and optionally to MLFlow.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Format metrics for logging
    metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                           for k, v in metrics.items())
    
    if step is not None:
        logger.info(f"Step {step}: {metric_str}")
    else:
        logger.info(metric_str)
    
    # Try MLFlow if available
    try:
        import mlflow
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}" if prefix else name
            mlflow.log_metric(full_name, value, step=step)
    except ImportError:
        pass
    except Exception:
        pass


def log_params(
    params: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log parameters to console and optionally to MLFlow.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Parameters: {params}")
    
    # Try MLFlow if available
    try:
        import mlflow
        mlflow.log_params(params)
    except ImportError:
        pass
    except Exception:
        pass


class MetricsLogger:
    """
    Metrics logger with automatic averaging and history tracking.
    """
    
    def __init__(self):
        self.history = []
        self.current = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update with new metrics."""
        for key, value in metrics.items():
            if key not in self.current:
                self.current[key] = 0.0
                self.counts[key] = 0
            
            self.current[key] += value
            self.counts[key] += 1
    
    def get_average(self) -> Dict[str, float]:
        """Get average of accumulated metrics."""
        return {
            key: self.current[key] / self.counts[key]
            for key in self.current.keys()
        }
    
    def reset(self) -> None:
        """Reset accumulated metrics."""
        # Save to history
        if self.current:
            self.history.append(self.get_average())
        
        self.current = {}
        self.counts = {}
    
    def get_history(self) -> list:
        """Get history of all averaged metrics."""
        return self.history


def suppress_warnings():
    """Suppress common ML library warnings."""
    import warnings
    warnings.filterwarnings("ignore")
    
    # Suppress sklearn warnings
    try:
        from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=DataConversionWarning)
    except ImportError:
        pass
    
    # Suppress optuna logging
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        pass
