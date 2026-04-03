"""
Maximum softmax probability (MSP) OOD detector.

Paper: https://arxiv.org/pdf/1610.02136
"""

from typing import TYPE_CHECKING

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features


class MSP(BaseDetector):
    """Negative maximum softmax probability at temperature ``T``.

    Score is ``-max_c softmax(logits/T)_c`` so that **higher score ⇒ more OOD**,
    matching ``BaseDetector``. Requires ``features.logits`` with shape
    ``(n_samples, n_classes)``. ``fit`` is a no-op.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize with softmax temperature.

        Args:
            temperature: Positive temperature for scaled softmax.

        Raises:
            ValueError: If ``temperature <= 0``.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def fit(
        self,
        features_train: "Features",
        **kwargs: object,
    ) -> "MSP":
        """No-op; MSP has no trainable state.

        Args:
            features_train: Ignored.
            **kwargs: Ignored.

        Returns:
            ``self``.
        """
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """Per-sample MSP scores (more OOD when less confident).

        Args:
            features_test: Must provide ``logits`` with shape ``(n_samples, n_classes)``.
            **kwargs: Unused.

        Returns:
            Scores in roughly ``[-1, 0]``, shape ``(n_samples,)``.

        Raises:
            ValueError: If logits are missing or not 2D.
        """
        if features_test.logits is None:
            raise ValueError("MSP.score requires Features.logits")
        logits = to_numpy(features_test.logits)
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N, C], got {logits.shape}")

        scaled = logits / self.temperature
        m = np.max(scaled, axis=1, keepdims=True)
        exps = np.exp(scaled - m)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        max_prob = np.max(probs, axis=1)
        return -max_prob

    def predict(
        self,
        features: "Features",
        threshold: float = -0.5,
        **kwargs: object,
    ) -> ArrayLike:
        """Label OOD (1) when ``score > threshold``.

        Args:
            features: Must provide ``logits``.
            threshold: Default ``-0.5`` is illustrative; tune on validation.
            **kwargs: Forwarded to ``score()``.

        Returns:
            Integer labels, shape ``(n_samples,)``.
        """
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)
