"""
Abstract base class for OOD detectors.

All detectors inherit from ``BaseDetector`` and implement ``fit``, ``score``, and
``predict``.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from oodkit.types import ArrayLike

if TYPE_CHECKING:
    from oodkit.data.features import Features


class BaseDetector(ABC):
    """Abstract base for out-of-distribution detectors.

    Sklearn-style API: ``fit`` learns from in-distribution data, ``score`` returns
    per-sample OOD scores (higher = more OOD), ``predict`` maps scores to binary
    ID/OOD labels.

    Note:
        Public methods may accept array-likes (including torch tensors) via
        ``Features``; implementations typically convert to NumPy once at method
        boundaries and return NumPy outputs.
    """

    @abstractmethod
    def fit(self, features: "Features", **kwargs) -> "BaseDetector":
        """Fit the detector on in-distribution features.

        Args:
            features: Container with logits, embeddings, or both (detector-specific).
            **kwargs: Detector-specific options.

        Returns:
            ``self`` for chaining.
        """
        ...

    @abstractmethod
    def score(self, features: "Features", **kwargs) -> ArrayLike:
        """Compute per-sample OOD scores (higher = more OOD).

        Args:
            features: Container with logits, embeddings, or both.
            **kwargs: Detector-specific options.

        Returns:
            Per-sample scores, shape ``(n_samples,)``.
        """
        ...

    def predict(self, features: "Features", threshold: Optional[float] = None, **kwargs) -> ArrayLike:
        """Predict binary ID (0) / OOD (1) from scores.

        Args:
            features: Container passed through to ``score()``.
            threshold: Score cutoff; samples above are OOD. Subclasses may define
                a default when ``None``.
            **kwargs: Forwarded to ``score()``.

        Returns:
            Integer labels, shape ``(n_samples,)`` with values in ``{0, 1}``.

        Raises:
            NotImplementedError: If the subclass does not implement ``predict``.
        """
        raise NotImplementedError("predict to be implemented by subclass")
