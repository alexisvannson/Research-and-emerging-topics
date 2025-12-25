"""Utility functions for summarization models."""

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def get_device(prefer_cuda: bool = True) -> str:
    """Get available device.

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_time(seconds: float) -> str:
    """Format time in seconds to readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length in words

    Returns:
        Truncated text
    """
    words = text.split()
    if len(words) <= max_length:
        return text
    return " ".join(words[:max_length]) + "..."


def compute_compression_ratio(source: str, summary: str) -> float:
    """Compute compression ratio between source and summary.

    Args:
        source: Source document
        summary: Summary text

    Returns:
        Compression ratio (source_len / summary_len)
    """
    source_len = len(source.split())
    summary_len = len(summary.split())

    if summary_len == 0:
        return float("inf")

    return source_len / summary_len


def load_model_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    path: str,
    **kwargs,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        path: Save path
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        **kwargs,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_model_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        model: Model to load state into
        path: Checkpoint path
        optimizer: Optional optimizer to load state into
        device: Device to load to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        """Initialize meter."""
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Update meter with new value.

        Args:
            val: Value to add
            n: Number of items
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def batch_iterator(data: List[Any], batch_size: int, shuffle: bool = False):
    """Create batches from data.

    Args:
        data: Input data list
        batch_size: Size of each batch
        shuffle: Whether to shuffle data

    Yields:
        Batches of data
    """
    if shuffle:
        indices = np.random.permutation(len(data))
        data = [data[i] for i in indices]

    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def print_model_summary(model: torch.nn.Module, input_size: Optional[tuple] = None):
    """Print model summary.

    Args:
        model: Model to summarize
        input_size: Optional input size for shape inference
    """

    print("Model Summary")

    print("\nArchitecture:")
    print(model)

    print("\nParameters:")
    params = count_parameters(model)
    for key, value in params.items():
        print(f"  {key}: {value:,}")


def main():
    """Test utility functions."""
    print("Testing utility functions...")

    # Test seed setting
    set_seed(42)
    print(f"Random number: {random.random()}")

    # Test device detection
    device = get_device()
    print(f"Device: {device}")

    # Test time formatting
    print(f"Formatted time: {format_time(3725.5)}")

    # Test compression ratio
    source = "This is a long document with many words in it."
    summary = "Long document with words."
    ratio = compute_compression_ratio(source, summary)
    print(f"Compression ratio: {ratio:.2f}")

    # Test average meter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg:.2f}")


if __name__ == "__main__":
    main()
