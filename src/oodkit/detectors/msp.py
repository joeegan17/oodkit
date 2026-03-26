"""
Maximum softmax probability (MSP) OOD detector.

Uses logits only. No training step; scores are purely functional of logits.
"""

from typing import TYPE_CHECKING

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features


class MSP(BaseDetector):
    """
    Maximum softmax probability detector.

    Uses temperature-scaled softmax; score is **negative** max probability so that
    **higher score ⇒ more OOD** (consistent with `BaseDetector`).

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
    ) -> "MSP":
        """
        No-op for MSP (there is nothing to fit).

        Parameters
        ----------
        features_train : Features
            Ignored.
        **kwargs : object
            Ignored.

        Returns
        -------
        self : MSP
        """
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        """
        Per-sample score = `-max_c softmax(logits / T)_c`.

        Higher score indicates more OOD (lower predictive confidence).

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
        """
        Binary ID (0) / OOD (1) via `score > threshold`.

        Parameters
        ----------
        features : Features
            Must provide `features.logits`.
        threshold : float, default=-0.5
            Scores lie in about `[-1, 0]`; **calibrate on validation** for your model.
        **kwargs : object
            Passed to `score()`.

        Returns
        -------
        labels : ArrayLike
            Shape `(n_samples,)`, values in `{0, 1}`.
        """
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)
