"""
Model I/O utilities for sklearn and PyTorch models.

Provides unified save/load interface with automatic backend detection.
"""

import pickle
from pathlib import Path
from typing import Any, Optional, Union
import json
import warnings


def save_model(
    model: Any,
    path: Union[str, Path],
    backend: str = "auto",
    metadata: Optional[dict] = None
) -> None:
    """
    Save a model with automatic backend detection.
    
    Args:
        model: Model to save (sklearn estimator or PyTorch model)
        path: Path to save model to
        backend: 'auto', 'sklearn', 'torch', or 'pickle'
        metadata: Optional metadata dict to save alongside model
        
    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> save_model(model, "model.pkl")
        
        >>> import torch
        >>> model = torch.nn.Linear(10, 2)
        >>> save_model(model, "model.pth")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect backend
    if backend == "auto":
        backend = _detect_backend(model)
    
    # Save model
    if backend == "torch":
        _save_torch_model(model, path, metadata)
    elif backend in ["sklearn", "pickle"]:
        _save_pickle_model(model, path, metadata)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    print(f"✓ Model saved to {path}")


def load_model(
    path: Union[str, Path],
    backend: str = "auto",
    device: Optional[str] = None,
    model_class: Optional[Any] = None
) -> Any:
    """
    Load a model with automatic backend detection.
    
    Args:
        path: Path to model file
        backend: 'auto', 'sklearn', 'torch', or 'pickle'
        device: Device for PyTorch models ('cpu', 'cuda')
        model_class: For PyTorch state_dict loading, provide model instance
        
    Returns:
        Loaded model
        
    Examples:
        >>> model = load_model("model.pkl")
        
        >>> # For PyTorch state dict
        >>> import torch
        >>> model_instance = torch.nn.Linear(10, 2)
        >>> model = load_model("model.pth", model_class=model_instance, device="cpu")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Load metadata if exists
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    # Auto-detect backend from metadata or file extension
    if backend == "auto":
        if metadata and "backend" in metadata:
            backend = metadata["backend"]
        else:
            backend = _detect_backend_from_path(path)
    
    # Load model
    if backend == "torch":
        model = _load_torch_model(path, device, model_class)
    elif backend in ["sklearn", "pickle"]:
        model = _load_pickle_model(path)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    print(f"✓ Model loaded from {path}")
    return model


# ============================================================================
# Backend detection
# ============================================================================

def _detect_backend(model: Any) -> str:
    """Detect model backend from model type."""
    model_type = type(model).__module__
    
    # Check if PyTorch
    if "torch" in model_type:
        return "torch"
    
    # Check if sklearn or compatible
    if hasattr(model, "fit") and hasattr(model, "predict"):
        return "sklearn"
    
    # Fallback to pickle
    return "pickle"


def _detect_backend_from_path(path: Path) -> str:
    """Detect backend from file extension."""
    suffix = path.suffix.lower()
    
    if suffix in [".pth", ".pt"]:
        return "torch"
    elif suffix in [".pkl", ".pickle", ".joblib"]:
        return "sklearn"
    else:
        # Try to infer from file content
        try:
            with open(path, "rb") as f:
                # Check for torch magic number
                magic = f.read(8)
                if magic[:2] == b"PK":  # ZIP file (torch uses ZIP)
                    return "torch"
        except Exception:
            pass
        
        return "sklearn"


# ============================================================================
# PyTorch save/load
# ============================================================================

def _save_torch_model(model: Any, path: Path, metadata: Optional[dict]) -> None:
    """Save PyTorch model."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required to save torch models. "
            "Install with: pip install pyable-ml[torch]"
        )
    
    # Save state dict
    torch.save(model.state_dict(), path)
    
    # Save metadata
    if metadata is None:
        metadata = {}
    
    metadata["backend"] = "torch"
    metadata["model_class"] = type(model).__name__
    
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _load_torch_model(
    path: Path,
    device: Optional[str],
    model_class: Optional[Any]
) -> Any:
    """Load PyTorch model."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required to load torch models. "
            "Install with: pip install pyable-ml[torch]"
        )
    
    if device is None:
        device = "cpu"
    
    if model_class is None:
        raise ValueError(
            "For PyTorch models, you must provide model_class instance. "
            "Example: load_model(path, model_class=MyModel())"
        )
    
    # Load state dict
    state_dict = torch.load(path, map_location=device)
    model_class.load_state_dict(state_dict)
    model_class.eval()
    
    return model_class


# ============================================================================
# Pickle save/load
# ============================================================================

def _save_pickle_model(model: Any, path: Path, metadata: Optional[dict]) -> None:
    """Save model using pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    
    # Save metadata
    if metadata is not None:
        metadata["backend"] = "pickle"
        metadata_path = path.with_suffix(path.suffix + ".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def _load_pickle_model(path: Path) -> Any:
    """Load model using pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# Checkpoint utilities (for PyTorch training)
# ============================================================================

def save_checkpoint(
    model: Any,
    optimizer: Any,
    epoch: int,
    loss: float,
    path: Union[str, Path],
    **kwargs
) -> None:
    """
    Save complete training checkpoint (PyTorch).
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss
        path: Save path
        **kwargs: Additional state to save
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for checkpoints. "
            "Install with: pip install pyable-ml[torch]"
        )
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs
    }
    
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved to {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    device: str = "cpu"
) -> dict:
    """
    Load training checkpoint (PyTorch).
    
    Args:
        path: Checkpoint path
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to map tensors to
        
    Returns:
        Checkpoint dict with epoch, loss, and any custom fields
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for checkpoints. "
            "Install with: pip install pyable-ml[torch]"
        )
    
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"✓ Checkpoint loaded from {path}")
    return checkpoint
