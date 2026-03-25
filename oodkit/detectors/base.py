"""
Abstract base class for OOD detectors.

All detectors inherit from BaseDetector and implement fit, score, and predict.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from oodkit.types import ArrayLike

if TYPE_CHECKING:
    from oodkit.data.features import Features


class BaseDetector(ABC):
    """
    Abstract base class for out-of-distribution detectors.

    Provides sklearn-style interface:
    - fit: learn from in-distribution data
    - score: return per-sample OOD scores (higher = more OOD)
    - predict: binary ID/OOD labels from scores

    Notes
    -----
    Input type policy for MVP:
    - Public detector methods may accept broad array-like inputs through `Features`
      (including NumPy arrays and torch tensors).
    - Implementations are free to normalize these inputs to NumPy once at method
      boundaries and return NumPy outputs.
    """

    @abstractmethod
    def fit(self, features: "Features", **kwargs) -> "BaseDetector":
        """
        Fit the detector on in-distribution features.

        Parameters
        ----------
        features : Features
            Container with logits, embeddings, or both (depends on detector).
        **kwargs
            Detector-specific options.

        Returns
        -------
        self : BaseDetector
        """
        ...

    @abstractmethod
    def score(self, features: "Features", **kwargs) -> ArrayLike:
        """
        Compute per-sample OOD scores.

        Higher scores indicate more out-of-distribution.

        Parameters
        ----------
        features : Features
            Container with logits, embeddings, or both.
        **kwargs
            Detector-specific options.

        Returns
        -------
        scores : ArrayLike
            Shape (n_samples,).
        """
        ...

    def predict(self, features: "Features", threshold: Optional[float] = None, **kwargs) -> ArrayLike:
        """
        Predict binary ID (0) / OOD (1) labels from scores.

        Parameters
        ----------
        features : Features
            Container with logits, embeddings, or both.
        threshold : float, optional
            Score threshold; samples above are OOD. Detector-specific default if None.
        **kwargs
            Passed to score().

        Returns
        -------
        labels : ArrayLike
            Shape (n_samples,), 0 = ID, 1 = OOD.
        """
        # Subclasses override to provide detector-specific default threshold
        raise NotImplementedError("predict to be implemented by subclass")
