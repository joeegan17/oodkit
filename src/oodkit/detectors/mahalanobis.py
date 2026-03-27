"""
Class-conditional Mahalanobis OOD detector.

Uses embeddings only. Fit computes per-class means and a shared covariance
inverse; score is the minimum squared Mahalanobis distance to any class mean.

Note
----
These distance-based methods often perform better when embeddings come from
contrastive objectives (or similarly metric-aware training), because in-class
clusters tend to be tighter and between-class separation improves.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from oodkit.detectors.base import BaseDetector
from oodkit.types import ArrayLike
from oodkit.utils.array import to_numpy

if TYPE_CHECKING:
    from oodkit.data.features import Features


class Mahalanobis(BaseDetector):
    """
    Class-conditional Mahalanobis detector with shared covariance.

    Expected inputs
    ---------------
    - `features.embeddings`, shape `(n_samples, n_features)`

    Training labels
    ---------------
    - `y`, shape `(n_samples,)`, optional.
    - If `y` is `None`, fit falls back to a single Gaussian and prints a warning.
      This mode is supported for convenience, but class-conditional fitting is
      the intended use.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def fit(
        self,
        features_train: "Features",
        y: Optional[ArrayLike] = None,
        **kwargs: object,
    ) -> "Mahalanobis":
        if features_train.embeddings is None:
            raise ValueError("Mahalanobis.fit requires Features.embeddings")
        embeddings = to_numpy(features_train.embeddings)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, F], got {embeddings.shape}")

        n_samples, n_features = embeddings.shape
        if n_samples < 2:
            raise ValueError("Mahalanobis.fit requires at least 2 samples")

        if y is None:
            print(
                "Mahalanobis.fit: y is None, fitting a single Gaussian. "
                "Class-conditional labels are recommended for best results."
            )
            y_np = np.zeros(n_samples, dtype=np.int64)
        else:
            y_np = np.asarray(to_numpy(y)).reshape(-1)
            if y_np.shape[0] != n_samples:
                raise ValueError(
                    f"y must have shape [N], got {y_np.shape}, expected ({n_samples},)"
                )

        classes, inverse = np.unique(y_np, return_inverse=True)
        n_classes = classes.shape[0]

        class_means = np.empty((n_classes, n_features), dtype=np.float64)
        for class_index in range(n_classes):
            mask = inverse == class_index
            class_means[class_index] = embeddings[mask].mean(axis=0)

        centered = embeddings - class_means[inverse]
        denom = max(n_samples - n_classes, 1)
        covariance = (centered.T @ centered) / float(denom)
        covariance += self.eps * np.eye(n_features, dtype=np.float64)
        inverse_covariance = np.linalg.pinv(covariance)

        self.classes_ = classes
        self.class_means_ = class_means
        self.inverse_covariance_ = inverse_covariance
        self.n_features_in_ = n_features
        return self

    def score(self, features_test: "Features", **kwargs: object) -> ArrayLike:
        self._check_is_fitted()

        if features_test.embeddings is None:
            raise ValueError("Mahalanobis.score requires Features.embeddings")
        embeddings = to_numpy(features_test.embeddings)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must have shape [N, F], got {embeddings.shape}")
        if embeddings.shape[1] != self.n_features_in_:
            raise ValueError(
                "embeddings feature dimension mismatch: "
                f"expected {self.n_features_in_}, got {embeddings.shape[1]}"
            )

        deltas = embeddings[:, None, :] - self.class_means_[None, :, :]
        projected = np.einsum("ncf,fg->ncg", deltas, self.inverse_covariance_)
        squared_distances = np.einsum("ncg,ncg->nc", projected, deltas)
        return np.min(squared_distances, axis=1)

    def predict(
        self,
        features: "Features",
        threshold: float,
        **kwargs: object,
    ) -> ArrayLike:
        scores = self.score(features, **kwargs)
        return (scores > threshold).astype(int)

    def _check_is_fitted(self) -> None:
        missing = [
            name
            for name in ("classes_", "class_means_", "inverse_covariance_", "n_features_in_")
            if not hasattr(self, name)
        ]
        if missing:
            raise RuntimeError(f"Mahalanobis instance is not fitted yet. Missing: {missing}")
