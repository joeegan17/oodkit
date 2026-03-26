"""
Energy-based OOD detector (log-sum-exp energy over logits).

Uses logits only. No training step.
"""

from typing import TYPE_CHECKING

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features


class Energy(BaseDetector):
    """
    Energy score: `E(x) = -T * log(sum_c exp(logits_c / T))`.

    **Higher energy ⇒ more OOD** (typical usage; calibrate thresholds on validation).

    Expected inputs
    ---------------
    - `features.logits`, shape `(n_samples, n_classes)`
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def fit(
        self,
        features_train: "Features",
        **kwargs: object,
    ) -> "Energy":
        """
        No-op for Energy (there is nothing to fit).

        Parameters
        ----------
        features_train : Features
            Ignored.
        **kwargs : object
            Ignored.

        Returns
        -------
        self : Energy
        """
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """
        Compute per-sample energy scores.

        Parameters
        ----------
        features_test : Features
            Must provide `features_test.logits`.
        **kwargs : object
            Reserved.

        Returns
        -------
        scores : ArrayLike
            Shape `(n_samples,)`.
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
        """
        Binary ID (0) / OOD (1) via `score > threshold`.

        Energy scale depends on model and dataset; **threshold must be chosen on
        validation data** (unlike MSP, there is no universal default).

        Parameters
        ----------
        features : Features
            Must provide `features.logits`.
        threshold : float
            Calibrated cutoff (higher energy ⇒ OOD).
        **kwargs : object
            Passed to `score()`.

        Returns
        -------
        labels : ArrayLike
            Shape `(n_samples,)`, values in `{0, 1}`.
        """
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)
