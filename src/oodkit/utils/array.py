"""
Array conversion helpers for detector boundaries.
"""

import numpy as np

from oodkit.types import ArrayLike


def to_numpy(x: ArrayLike) -> np.ndarray:
    """
    Convert supported inputs to NumPy arrays.

    Accepts NumPy arrays directly and torch tensors via ``detach().cpu().numpy()``.
    """
    if x is None:
        raise ValueError("Expected array-like input, got None")
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
