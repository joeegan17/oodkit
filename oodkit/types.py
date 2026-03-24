"""
Shared type aliases and typing utilities.

Centralized to avoid circular imports when modules depend on each other.
"""

from typing import Union, TypeAlias

# Placeholder aliases — adjust to actual backend (numpy, torch, etc.) as needed
ArrayLike: TypeAlias = "Union[object]"  # numpy array, torch tensor, or array-like
Tensor: TypeAlias = "Union[object]"  # framework-specific tensor type
