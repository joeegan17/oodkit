"""
Array conversion helpers for detector boundaries.
"""

import numpy as np

from oodkit.types import ArrayLike


def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert array-likes to ``np.ndarray`` (NumPy arrays or torch tensors).

    Args:
        x: NumPy array, torch tensor (via ``detach().cpu().numpy()``), or object
            convertible with ``np.asarray``.

    Returns:
        A NumPy array view or copy.

    Raises:
        ValueError: If ``x`` is ``None``.
    """
    if x is None:
        raise ValueError("Expected array-like input, got None")
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
