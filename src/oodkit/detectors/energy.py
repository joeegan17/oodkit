"""
Energy-based OOD detector (log-sum-exp over logits).
"""

from typing import TYPE_CHECKING

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features


class Energy(BaseDetector):
    """Energy score from logits: ``E(x) = -T * log(sum_c exp(logits_c / T))``.

    Higher scores indicate more OOD; calibrate ``predict`` thresholds on validation
    data. Requires ``features.logits`` with shape ``(n_samples, n_classes)``.

    ``fit`` is a no-op.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize with softmax temperature.

        Args:
            temperature: Positive ``T`` in ``logits / T`` before log-sum-exp.

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
    ) -> "Energy":
        """No-op; Energy has no trainable state.

        Args:
            features_train: Ignored.
            **kwargs: Ignored.

        Returns:
            ``self``.
        """
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """Return per-sample energy scores.

        Args:
            features_test: Must provide ``logits`` with shape ``(n_samples, n_classes)``.
            **kwargs: Unused.

        Returns:
            Energies, shape ``(n_samples,)``, ``float64``.

        Raises:
            ValueError: If logits are missing or not 2D.
        """
        if features_test.logits is None:
            raise ValueError("Energy.score requires Features.logits")
        logits = to_numpy(features_test.logits)
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N, C], got {logits.shape}")

        T = self.temperature
        scaled = logits / T
        m = np.max(scaled, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(scaled - m), axis=1)) + m.ravel()
        return (-T * log_sum_exp).astype(np.float64)

    def predict(
        self,
        features: "Features",
        threshold: float,
        **kwargs: object,
    ) -> ArrayLike:
        """Label samples OOD (1) when ``score > threshold``.

        Args:
            features: Must provide ``logits``.
            threshold: Validation-tuned cutoff (higher energy ⇒ OOD).
            **kwargs: Forwarded to ``score()``.

        Returns:
            Integer labels, shape ``(n_samples,)``.
        """
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)
