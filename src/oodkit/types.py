"""
Shared type aliases and typing utilities.

Centralized to avoid circular imports when modules depend on each other.
"""

from typing import Any, TypeAlias

# MVP typing policy:
# - Public APIs accept broad array-like inputs (NumPy arrays, torch tensors, etc.).
# - Detector internals are NumPy-first and may convert inputs once at boundaries.
ArrayLike: TypeAlias = Any
Tensor: TypeAlias = Any
