"""Utility functions for training.

Implements:
- Function timing decorators
- Loss aggregation
- Directory creation
"""

import time
from pathlib import Path
from typing import List, Dict, Callable, Any
import os
from loguru import logger
from torch import Tensor
import torch

PROJECT_NAME = "NeuralSVG"
ENABLE_NAMEIT = os.getenv("ENABLE_NAMEIT") in ("true", "1", "yes", "y")


def nameit(func: Callable) -> Callable:
    """Time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """

    def named(*args, **kwargs):
        if ENABLE_NAMEIT:
            logger.info(f"Running {func.__qualname__} within 'nameit'...")
            t0 = time.time()
            result = func(*args, **kwargs)
            t1 = time.time()
            logger.info(
                f"{func.__qualname__} Done! {func.__qualname__} took {t1 - t0:0.3f} seconds to run."
            )
            return result
        else:
            return func(*args, **kwargs)

    return named


def nameit_torch(func: Callable) -> Callable:
    """Time CUDA function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """

    def named(*args, **kwargs):
        if ENABLE_NAMEIT:
            logger.info(f"Running {func.__qualname__} within 'nameit_torch'...")

            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)

            t0.record()
            result = func(*args, **kwargs)
            t1.record()

            torch.cuda.synchronize()
            elapsed_time_seconds = t0.elapsed_time(t1) / 1000
            logger.info(
                f"{func.__qualname__} Done! {func.__qualname__} took {elapsed_time_seconds:0.3f} seconds to run."
            )
            return result
        else:
            return func(*args, **kwargs)

    return named


def aggregated_loss_dict(agg_loss_dict: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate loss dictionaries.

    Args:
        agg_loss_dict: List of loss dicts

    Returns:
        Aggregated loss dict
    """
    mean_values = {}
    for output in agg_loss_dict:
        for key, value in output.items():
            mean_values[key] = mean_values.setdefault(key, []) + [value]
    for key, vals in mean_values.items():
        if len(vals) > 0:
            mean_values[key] = sum(vals) / len(vals)
        else:
            logger.info(f"{key} has no value")
            mean_values[key] = 0
    return mean_values


def create_dir(dir_path: Path) -> Path:
    """Create directory if not exists.

    Args:
        dir_path: Directory path

    Returns:
        Created directory path
    """
    dir_path.mkdir(exist_ok=True, parents=True)
    return dir_path
